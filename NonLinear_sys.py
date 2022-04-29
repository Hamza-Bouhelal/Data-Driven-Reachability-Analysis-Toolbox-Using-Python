from numpy.linalg import pinv
import warnings
import time
from Plot import plot_results
from reducegens import reduce_girard
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from Zonotope import Zonotope
from MatZonotope import MatZonotope
from utils.Options import Options
from utils.Params import Params
from reachability_nonlinear import params2options, checkOptionsReach
import numpy.matlib as matlib
from Interval import Interval
from scipy.io import loadmat

def cstrdiscr(dt, x, u):
    """
        discrete-time version of the stirred-tank reactor system
    """
    rho = 1000
    Cp = 0.239
    deltaH = -5e4
    E_R = 8750
    k0 = 7.2e10
    UA = 5e4
    q = 100
    Tf = 350
    V = 100
    C_Af = 1
    C_A0 = 0.5
    T_0 = 350
    T_c0 = 300
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

def load_model():
    return [reduce_girard(Zonotope(loadmat(f'model_for_stirred_tank\\{i}.mat')["var"]), 3) for i in range(0, 5)]

def linReach_DT(data, options):
        options.params["Uorig"] = options.params["U"] + \
            options.params["uTrans"]
        xStar = data.center()
        uStar = options.params["Uorig"].center()
        xStarMat = matlib.repmat(xStar, 1, options.params["X_0T"].shape[1])
        uStarMat = matlib.repmat(uStar, 1, options.params["U_full"].shape[1])
        oneMat = matlib.repmat(
            np.array([1]), 1, options.params["U_full"].shape[1])
        IAB = np.dot(options.params["X_1T"], pinv(np.vstack(
            [oneMat, options.params["X_0T"] + (-1 * xStarMat), options.params["U_full"] + -1 * uStarMat])))
        V = -1 * (options.params["Wmatzono"] + np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"] +
                                                                      -1 * uStarMat]))) + options.params["X_1T"]
        VInt = V.interval_matrix()
        leftLimit = VInt.Inf
        rightLimit = VInt.Sup
        V_one = Zonotope(Interval(leftLimit.min(
            axis=1).T, rightLimit.max(axis=1).T))
        x = data+(-1*xStar)
        result = (x.cart_prod(options.params["Uorig"] + (-1 * uStar)).cart_prod(
            [1]) * IAB) + V_one + options.params["W"] + options.params["Zeps"]
        return result



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
    def __init__(self, dt:int, U:Zonotope, R0:Zonotope, wfac:float, dim_x:int, initpoints:int, steps:int, func, zonoOrder:int = 100, tensorOrder:int = 2, errorOrder:int = 5):
        if isinstance(R0, Zonotope) == False:
            raise TypeError(f"X0 must be of type Zonotope")
        elif isinstance(U, Zonotope) == False:
            raise TypeError(f"U must be of type Zonotope")
        self.dim_x = dim_x
        self.dt = dt
        self.U = U
        self.R0 = R0
        self.params = Params(tFinal=self.dt * 5, dt=self.dt)
        self.options = Options()
        self.options.params["dim_x"] = self.dim_x
        self.initpoints = initpoints
        self.steps = steps
        self.totalsamples = steps * initpoints
        self.wfac = wfac
        self.buildMW()
        self.options.params["zonotopeOrder"] = zonoOrder
        self.options.params["tensorOrder"] = tensorOrder
        self.options.params["errorOrder"] = errorOrder
        self.u = [U.rand_point() for _ in range(self.totalsamples)]
        self.params.params["U"] = U
        self.params.params["R0"] = R0
        self.func = func

    def buildMW(self):
        """ Builds the noise Zonotope Mw """
        self.W = Zonotope(np.array(np.zeros(
            (self.options.params["dim_x"], 1))), self.wfac * np.ones((self.options.params["dim_x"], 1)))
        self.params.params["W"] = self.W
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
        self.params.params["Wmatzono"] = MatZonotope(
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
        return X_0t, X_1t, U_full

    def compute_Lipschits_const(self, X_0t, X_1t, U_full):
        """Computes the Lipschitz constant of the system"""
        self.options.params["X_0T"] = X_0t
        self.options.params["X_1T"] = X_1t
        self.options.params["U_full"] = U_full
        """ L = 0
        for i in range(self.totalsamples):
            z1 = np.hstack([np.array(X_0t[:, i]).flatten(), np.array(self.u).flatten(order='F')[i]])
            f1 = np.array([X_1t[:, i]])
            for j in range(self.totalsamples):
                if i != j:
                    z2 = np.hstack([np.array(X_0t)[:, j].flatten(), np.array(self.u).flatten(order='F')[j]])
                    f2 = np.array([X_1t[:, j]])
                    newnorm = np.linalg.norm(f1 - f2)  / np.linalg.norm(z1 - z2)
                    if newnorm > L:
                        L = newnorm 
                        eps = L * np.linalg.norm(z1 - z2) """
        eps = 0.0035
        self.options.params["Zeps"] = Zonotope(np.array(np.zeros(
            (self.dim_x, 1))), eps * np.diag(np.ones((self.options.params["dim_x"], 1)).T[0]))
        self.options.params["ZepsFlag"] = True

    

    def run_reachability(self, totalsteps, plot=False, save=''):
        """
        Compute the forward reachable set of the system
        
        totalsteps: number of steps to be computed
        plot: if True Plots the results
        save: if not empty create a directory and saves plots of the results to that directory
        """
        x = self.Simulate_sys()
        X_0t, X_1t, ufull = self.combine_trajs(x)
        self.compute_Lipschits_const(X_0t, X_1t, ufull)
        options = params2options(self.params, self.options)
        options = checkOptionsReach(options, 0)
        R_data = [self.params.params["R0"]]
        print("Computing reachability...")
        t1 = time.time()
        for i in tqdm(range(totalsteps)):
            if('uTransVec' in options.params):
                options.params['uTrans'] = options.params["uTransVec"][:, i]
            data = linReach_DT(R_data[i], options)
            R_data.append(reduce_girard(data, 3))
        t2 = time.time() - t1
        print("Reachability took {} seconds.\n\n".format(t2))
        if plot or save != '':
            plot_results([R_data], plot, save, ["Reachability"])
        return R_data


if __name__ == "__main__":
    dim_x = 2
    U = Zonotope(np.array(np.array([0.01, 0.01]).reshape((2, 1))),
                 np.diag([0.1, .2]))
    R0 = Zonotope(np.array([-1.9, -20]).reshape((dim_x, 1)),
                  np.diag([0.005, .3]))
    dt = 0.015
    initpoints = 1
    steps = 20
    wfac = 1e-4
    nl_sys = NonLinear_sys(dt, U, R0, wfac, dim_x, initpoints, steps, cstrdiscr)
    data = nl_sys.run_reachability(5, False)
    model = load_model()
    plot_results([model, data], True, "", ["Model", "Reachability"])