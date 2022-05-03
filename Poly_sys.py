import sys
from Utils import reduce_girard, Interval_multiplication, Interval_selector
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import numpy as np
from MatZonotope import MatZonotope
from Zonotope import Zonotope 
from Plot import plot_results
from tqdm import tqdm
from numpy.linalg import pinv
from Interval import Interval

def poly_func(dt, x, u):
    return np.array([.320*(x[0, 0]**2) + u[0, 0] + 0.7*x[0, 0], .320*u[1, 0]*x[0, 0] + .4*(x[1, 0]**2) + .09*x[0, 0]])
    
class Poly_sys:
    def __init__(self, N, dt, U, R0, wfac, dim_x, initpoints, steps, func, zonoOrder=200, tensorOrder=2, errorOrder=5):
        self.N = N
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
        self.Wmatzono = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), GW)

    def Simulate_sys(self):
        """
        Simulates the system with the given parameters
        """
        x = []
        self.u = [U.rand_point() for _ in range(self.totalsamples)]
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
    
    def compute_monomials(self, X_0t, X_1t, U_full):
        X_2 = X_0t * X_0t
        X1X2  = X_0t[0, :] * X_0t[1, :]
        U_2 = U_full * U_full
        U1U2 = U_full[0, :] * U_full[1, :]
        XU = X_0t * U_full
        X1U2X2U1  = X_0t * U_full[[1, 0], :]
        data = np.vstack([np.ones((1, self.totalsamples)), X_0t, X_2, X1X2, U_full, U_2, U1U2, XU, X1U2X2U1])
        #rank = np.linalg.matrix_rank(data)
        AB = (-1*self.Wmatzono + X_1t ) * pinv(data)
        print(AB.generators.shape)
        return AB

    def run_reachability(self, totalsteps, plot=False, save=''):
        self.buildMW()
        x = self.Simulate_sys()
        X_0t, X_1t, U_full = self.combine_trajs(x)
        AB = self.compute_monomials(X_0t, X_1t, U_full)
        data = [self.R0]
        for i in range(totalsteps):
            data[i] = reduce_girard(data[i], self.zonotopeOrder)
            X_z1 = Interval(data[i])
            U_int = Interval(self.U)
            cardInt = Zonotope(np.vstack([Interval(np.array([1])), X_z1, Interval_multiplication(X_z1, X_z1), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(X_z1, [1])), U_int, Interval_multiplication(U_int, U_int), Interval_multiplication(Interval_selector(U_int, [0]), Interval_selector(U_int, [1])), Interval_multiplication(X_z1,U_int), Interval_multiplication(Interval_selector(X_z1, [0]), Interval_selector(U_int, [1])), Interval_multiplication(Interval_selector(X_z1, [1]), Interval_selector(U_int, [0]))]))
            data.append(AB * cardInt + self.W)
        data[-1] = reduce_girard(data[-1], self.zonotopeOrder)
        if plot:
            plot_results(data, plot, save)


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
    wfac = 0.00007
    poly_sys = Poly_sys(N, dt, U, R0, wfac, dim_x, initpoints, steps, poly_func)
    poly_sys.run_reachability(1, True)