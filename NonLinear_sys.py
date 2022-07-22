from numpy.linalg import pinv
import warnings
import time
from Plot import plot_results
from Utils import reduce_girard
import numpy as np
import sys
import pickle
from tqdm import tqdm
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from Zonotope import Zonotope
from MatZonotope import MatZonotope
from Interval import Interval
import numpy.matlib as matlib
np.random.seed(5)
def cstrdiscr(dt, x, u, rho = 1000, Cp = 0.239, deltaH = -5e4, E_R = 8750, k0 = 7.2e10, UA = 5e4, q = 100, Tf = 350, V = 100, C_Af = 1, C_A0 = 0.5, T_0 = 350, T_c0 = 300):
    """
        discrete-time version of the stirred-tank reactor system
    """
    U = np.matmul(np.array([-3, -6.9]), x)
    x_temp = np.array([x[0, 0] + C_A0, x[1, 0]+T_0]).reshape(2, 1)
    U = U + np.array([T_c0])
    f1 = ((1-(q*dt)/(2*V) - k0*dt*np.exp(-E_R /
          x_temp[1, 0]))*x_temp[0, 0] + q/V * C_Af * dt)/(1 + (q*dt)/(2*V)) + u[0]*dt
    f2 = (x_temp[1, 0]*(1-0.5*dt - (dt*UA)/(2*V*rho*Cp)) + dt*(Tf*q/V + (UA*U)/(V*rho*Cp)) - x_temp[0, 0] *
          (deltaH*k0*dt)/(rho*Cp) * np.exp(-E_R/x_temp[1, 0])) / (1+0.5*dt*q/V+(dt*UA)/(2*V*rho*Cp)) + u[1, 0]*dt
    f1 = f1 - C_A0
    f2 = f2 - T_0
    return np.array([f1, f2]).reshape(2, 1)

def load_model(n=5):
    """
    loads Model based reachability of discrete-time version of the stirred-tank reactor system
    """
    with open('NonLinear_Model.obj', 'rb') as f:
        return pickle.load(f)[:n]
    

class NonLinear_sys():
    """
    Discrete time systems
    x(k+1) = f(x(k), u(k)) + w(k)

    dt: time step
    U: Input Zonotope
    R0: Initial reachable set
    wfac: scaling factor for the disturbance
    dim_x: dimension of the state space
    initpoints: number of initial points that will be propagated to get a simulation of the system
    steps: number of time steps that will be propagated
    func: function that defines the system dynamics
    """
    def __init__(self, dt:int, U:Zonotope, R0:Zonotope, wfac:float, dim_x:int, initpoints:int, steps:int, func, zonoOrder:int = 3, tensorOrder:int = 2, errorOrder:int = 5):
        if isinstance(R0, Zonotope) == False:
            raise TypeError(f"X0 must be of type Zonotope")
        elif isinstance(U, Zonotope) == False:
            raise TypeError(f"U must be of type Zonotope")
        self.dim_x = dim_x
        self.dt = dt
        self.U = U
        self.R0 = R0
        self.initpoints = initpoints
        self.steps = steps
        self.totalsamples = steps * initpoints
        self.wfac = wfac
        self.buildMW()
        self.zonotopeOrder = zonoOrder
        self.tensorOrder = tensorOrder
        self.errorOrder = errorOrder
        self.func = func
        self.u = [U.rand_point() for _ in range(self.totalsamples)]
        """ self.ZepsFlag = False """


    def buildMW(self):
        """ 
        Builds the noise MatZonotope Mw from the noise zonotope W
        """
        self.W = Zonotope(np.array(np.zeros((self.dim_x, 1))), self.wfac * np.ones((self.dim_x, 1)))
        self.GW = []
        for i in range(self.W.generators().shape[1]):
            vec = np.reshape(self.W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(
                np.hstack((vec, np.zeros((self.dim_x, self.totalsamples - 1)))))
            for j in range(1, self.totalsamples, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            self.GW.append(np.array(dummy))
        self.GW = np.array(self.GW[0])
        self.Wmatzono = MatZonotope(
            np.zeros((self.dim_x, self.totalsamples)), self.GW)

    def Simulate_sys(self):
        """
        Simulates the system with the given parameters
        """
        x = []
        idx = 0
        print(f"Propagating {self.initpoints} initpoints {self.steps} time...")
        for _ in tqdm(range(0, self.initpoints*self.dim_x, self.dim_x)):
            rand = self.R0.rand_point()
            tempx = [rand]
            curr_point = rand.tolist()
            for i in range(self.steps):
                tempx.append(
                    self.func(self.dt, tempx[i], self.u[idx]) + self.W.rand_point())
                idx += 1
                jdx = 0
                for ele in curr_point:
                    ele.append(tempx[i+1][jdx][0])
                    jdx += 1
            x.append(curr_point)
        print("Simulation done!\n" + 60*"=" + "\n")
        return x

    def combine_trajs(self, x):
        final_x = x[0]
        for initpoint in range(1, len(x)):
            for dim in range(self.dim_x):
                for step in range(self.steps):
                    final_x[dim].append(x[initpoint][dim][step])
        X_0t = np.array([ele[:-1] for ele in final_x])
        #print(X_0t.shape)
        X_1t = np.array([ele[1:] for ele in final_x])
        #print(X_1t.shape)
        U_full = np.array(self.u).reshape((-1, self.totalsamples))
        #print(U_full.shape)
        self.X_0t = X_0t
        self.X_1t = X_1t
        self.U_full = U_full
        return X_0t, X_1t, U_full

    def compute_Lipschits_const(self, steps, initpoints):

        """Computes the Lipschitz constant of the system"""
        X_0t, X_1t, = self.X_0t, self.X_1t
        normtype = 2
        L = []
        gamma = []
        for dim in range(self.dim_x):
            L.append([0])
            gamma.append(0)
            for i in range(self.totalsamples):
                z1 = np.hstack([np.array(X_0t[:, i]).flatten(), np.array(self.u).flatten(order='F')[i]])
                f1 = np.array([X_1t[:, i]])
                for j in range(self.totalsamples):
                    if i != j:
                        z2 = np.hstack([np.array(X_0t)[:, j].flatten(), np.array(self.u).flatten(order='F')[j]])
                        f2 = np.array([X_1t[:, j]])
                        newnorm = np.linalg.norm(np.subtract(f1, f2), normtype)  / np.linalg.norm(np.subtract(z1, z2), normtype)
                        newgamma = np.linalg.norm(np.subtract(z1, z2), normtype)
                        if newnorm > L[dim]:
                            L[dim] = newnorm 
                            #eps = L * np.linalg.norm(np.subtract(z1, z2))
                        if newgamma > gamma[dim]:
                            gamma[dim] = newgamma
        eps = [L[i] * gamma[i]/2 for i in range(self.dim_x)]
        self.Zeps = Zonotope(np.array(np.zeros(
            (self.dim_x, 1))),  np.diag(eps))

    def linReach_DT(self, data):
        xStar = data.center()
        uStar = self.U.center()
        xStarMat = matlib.repmat(xStar, 1, self.X_0t.shape[1])
        uStarMat = matlib.repmat(uStar, 1, self.U_full.shape[1])
        oneMat = matlib.repmat(
            np.array([1]), 1, self.U_full.shape[1])
        IAB = np.dot(self.X_1t, pinv(np.vstack(
            [oneMat, self.X_0t + (-1 * xStarMat), self.U_full + -1 * uStarMat])))
        V = -1 * (self.Wmatzono + np.dot(IAB, np.vstack([oneMat, self.X_0t+(-1*xStarMat), self.U_full +
            -1 * uStarMat]))) + self.X_1t
        VInt = V.interval_matrix()
        leftLimit = VInt.Inf
        rightLimit = VInt.Sup
        V_one = Zonotope(Interval(leftLimit.min(
            axis=1).T, rightLimit.max(axis=1).T))
        x = data+(-1*xStar)
        result =  (x.cart_prod(self.U + (-1 * uStar)).cart_prod(
            [1]) * IAB) + V_one  + self.W
        return result

    def Data_Driven_Reachability(self, totalsteps, ZepsFlag=True, plot=False, save=''):
        """
        Computes the data driven reachable set of the system
        
        totalsteps: number of steps to be computed
        ZepsFlag: if true compute the Lipschitz constant of the system and add eps zonotope to each steps of the result
        plot: if True Plots the results
        save: if not empty create a directory and saves plots of the results to that directory
        """
        x = self.Simulate_sys()
        X_0t, X_1t, u = self.combine_trajs(x)
        R_data = [self.R0]
        print("Computing reachability...")
        t1 = time.time()
        for i in tqdm(range(totalsteps)):
            data = self.linReach_DT(R_data[i])
            R_data.append(reduce_girard(data, self.zonotopeOrder ))
        if ZepsFlag:
            self.compute_Lipschits_const(X_0t, X_1t)
            for ele in R_data[1:]:
                ele += self.Zeps
        t2 = time.time() - t1
        print("Reachability took {} seconds.\n\n".format(t2))
        if plot or save != '':
            plot_results([R_data], plot, save, ["Data driven Reachability"], x0=self.R0, fillFirstResult=False)
        return R_data
 

if __name__ == "__main__":
    dim_x = 2
    U = Zonotope(np.array(np.array([0.01, 0.01]).reshape((2, 1))),np.diag([0.1, .2]))
    R0 = Zonotope(np.array([-1.9, -20]).reshape((dim_x, 1)),np.diag([0.005, .3]))
    dt = 0.015
    initpoints = 1
    steps = 25
    wfac = 10**-4
    nl_sys = NonLinear_sys(dt, U, R0, wfac, dim_x, initpoints, steps, cstrdiscr)
    data = nl_sys.Data_Driven_Reachability(5, True, plot=False)
    model = load_model()
    plot_results([model, data], True, "", ["Model based Reachability", "Data driven Reachability"], x0=R0)
