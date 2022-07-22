from Utils import *
import numpy as np
from MatZonotope import MatZonotope
from Plot import *
from tqdm import tqdm
from numpy.linalg import pinv
from Interval import Interval
from Zonotope import Zonotope
import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import time
import pickle

np.random.seed(1)

def poly_func(dt, x, u):
    return np.array([.320*(x[0, 0]**2) + u[0, 0] + 0.7*x[0, 0], .320*u[1, 0]*x[0, 0] + .4*(x[1, 0]**2) + .09*x[0, 0]]).reshape(2, 1)

def load_model(n):
    with open('Poly_sys_model.obj', 'rb') as f:
        return pickle.load(f)[0:n]

class Poly_sys:
    """
    Polynomial system
    x(k+1) = fp(x(k), u(k)) + w(k)

    dt: time step
    U: input zonotope
    R0: initial set
    wfac: noise factor
    dim_x: dimension of the state
    initpoints: number of initial points for the simulation
    steps: number of steps for the simulation
    func: function that defines the system
    zonoOrder: zonotope reduction order
    """
    def __init__(self, dt, U, R0, wfac, dim_x, initpoints, steps, func, zonoOrder=3, tensorOrder=2, errorOrder=5):
        self.dt = dt
        self.tFinal = dt * steps
        self.U = U
        self.R0 = R0
        self.wfac = wfac
        self.dim_x = dim_x
        self.initpoints = initpoints
        self.steps = steps
        self.totalsamples = initpoints*steps
        self.zonotopeOrder = zonoOrder
        self.tensorOrder = tensorOrder
        self.errorOrder = errorOrder
        self.u = [U.rand_point() for _ in range(self.totalsamples)]
        self.func = func
        self.buildMW()

    def buildMW(self):
        """ 
        Builds the noise MatZonotope Mw from the noise zonotope W
        """
        self.W = Zonotope(np.array(np.zeros(
            (self.dim_x, 1))), self.wfac * np.ones((self.dim_x, 1)))
        GW = []
        for i in range(self.W.generators().shape[1]):
            vec = np.reshape(self.W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(
                np.hstack((vec, np.zeros((self.dim_x, self.totalsamples - 1)))))
            for j in range(1, self.totalsamples, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            GW.append(np.array(dummy))
        GW = np.array(GW[0])
        self.Wmatzono = MatZonotope(
            np.zeros((self.dim_x, self.totalsamples)), GW)

    def jacobian(self, x, u):
        """ 
        Computes the Jacobian of the system at state x with input u
        """
        eps = 1e-10
        J = np.zeros([len(x), len(x)])
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i][0] += eps
            x2[i][0] -= eps
            f1 = self.func(self.dt, x1, u)
            f2 = self.func(self.dt, x2, u)
            J[:, i] = ((f1 - f2) / (2 * eps)).reshape(len(x), )
        return J

    def linReach(self, x):
        """
        Computes the next step of the system (Model based reachability)
        NOTE: NOT functional yet
        x: current state
        """
        # linearize nonlinear system
        px = x.center()
        pu = self.U.center()
        f = self.func(self.dt, px, pu)
        A_lin, B_lin = self.jacobian(px, pu)
        Udelta = B_lin * (self.U + (-self.U.center()))
        #print(Udelta)
        print(f)
        print(Udelta)
        U1 =  f + Udelta
        Rdelta = x - px
        return A_lin*Rdelta + U1

    def Model_Based_Reachability(self, totalsteps, plot=False, save=''):
        """
        Model based reachability
        Computes the reachable set of the system using the model based approach

        totalsteps: number of steps to be computed
        plot: if True, plots the reachable set
        save: if not empty, saves the reachable set to the file
        """
        data = [self.R0]
        for i in range(totalsteps):
            data[i] = reduce_girard(data[i], self.zonotopeOrder)
            newState = self.linReach(data[i])
            data.append(newState + self.W)
        data[-1] = reduce_girard(data[-1], self.zonotopeOrder)
        if plot:
            plot_results(data, plot, save)
        return data
    
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
        # print(X_0t.shape)
        X_1t = np.array([ele[1:] for ele in final_x])
        # print(X_1t.shape)
        U_full = np.array(self.u).reshape((-1, self.totalsamples))
        # print(U_full.shape)
        return X_0t, X_1t, U_full

    def compute_monomials_2d(self, X_0t, X_1t, U_full):
        """
        Compute the monomials of the system using the simulated data
        """
        X_2 = X_0t * X_0t
        X1X2 = X_0t[0, :] * X_0t[1, :]
        U_2 = U_full * U_full
        U1U2 = U_full[0, :] * U_full[1, :]
        XU = X_0t * U_full
        X1U2X2U1 = X_0t * U_full[[1, 0], :]
        data = np.vstack([np.ones((1, self.totalsamples)),
                         X_0t, X_2, X1X2, U_full, U_2, U1U2, XU, X1U2X2U1])
        rank = np.linalg.matrix_rank(data)
        AB = (-1*self.Wmatzono + X_1t) * pinv(data)
        return AB

    def Data_Driven_Reachability(self, totalsteps, plot=False, save=''):
        """
        Computes the data driven reachable set of the system

        totalsteps: number of steps to be computed
        plot: if True, plots the reachable set
        save: if not empty, saves the reachable set to the file
        """
        if self.dim_x != 2:
            raise ValueError("Data driven reachability for polynomial systems is only implemented for 2D systems")
        x = self.Simulate_sys()
        X_0t, X_1t, U_full = self.combine_trajs(x)
        AB = self.compute_monomials_2d(X_0t, X_1t, U_full)
        data = [self.R0]
        print(f"Computing Reachability...")
        t1 = time.time()
        for i in tqdm(range(totalsteps)):
            data[i] = reduce_girard(data[i], self.zonotopeOrder)
            X_z1 = data[i].to_interval()
            U_int = self.U.to_interval()
            ints = [Interval(np.array([1])), X_z1, interval_mul_2(X_z1, X_z1), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(X_z1, [1])), U_int, interval_mul_2(U_int, U_int), Interval_multiplication(Interval_selector(
                U_int, [0]), Interval_selector(U_int, [1])), interval_mul_2(X_z1, U_int), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(U_int, [1])), Interval_multiplication(Interval_selector(X_z1, [1]), Interval_selector(U_int, [0]))]
            cardint = Zonotope(intervals_To_interval(ints))
            #cardInt = Zonotope(np.vstack([Interval(np.array([1])), X_z1, Interval_multiplication(X_z1, X_z1), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(X_z1, [1])), U_int, Interval_multiplication(U_int, U_int), Interval_multiplication(Interval_selector(
            #    U_int, [0]), Interval_selector(U_int, [1])), Interval_multiplication(X_z1, U_int), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(U_int, [1])), Interval_multiplication(Interval_selector(X_z1, [1]), Interval_selector(U_int, [0]))]))
            data.append(AB * cardint + self.W)
        data[-1] = reduce_girard(data[-1], self.zonotopeOrder)
        t2 = time.time() - t1
        print("Reachability took {} seconds.\n\n".format(t2))
        if plot:
            plot_results(data, plot, save, ["Data Driven Reachability"], x0=self.R0)
        return data


if __name__ == "__main__":
    N = 3
    dt = 0.015
    U = Zonotope(np.array(np.array([0.2, 0.3]).reshape((2, 1))),
                 np.diag([0.01, .02]))
    R0 = Zonotope(np.array([1, 2]).reshape((2, 1)),
                  np.diag([0.05, .3]))
    dim_x = 2
    initpoints = 1
    steps = 7
    wfac = 0.000001
    poly_sys = Poly_sys(dt, U, R0, wfac, dim_x,
                        initpoints, steps, poly_func)
    data = poly_sys.Data_Driven_Reachability(N, False)
    model = load_model(N)
    plot_results([model, data], True, "", ["Model based Reachability", "Data driven Reachability"], x0=R0)
  