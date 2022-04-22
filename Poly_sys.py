import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import numpy as np
from MatZonotope import MatZonotope
from Zonotope import Zonotope 
from Plot import plot_result

def poly_func(dt, x, u):
    return np.array([.320*(x[0, 0]**2) + u[0, 0] + 0.7*x[0, 0], .320*u[1, 0]*x[0, 0] + .4*(x[1, 0]**2) + .09*x[0, 0]])
    
class Poly_sys:
    def __init__(self, N, dt, U, R0, wfac, dim_x, initpoints, steps):
        self.N = N
        self.dt = dt
        self.U = U
        self.R0 = R0
        self.wfac = wfac
        self.dim_x = dim_x
        self.initpoints = initpoints
        self.steps = steps
        self.totalsamples = initpoints*steps

        self.W = Zonotope(np.array(np.zeros(
            (self.options.params["dim_x"], 1))), self.wfac * np.ones((self.options.params["dim_x"], 1)))
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
        self.zonotopeOrder = 100
        self.tensorOrder = 2
        self.errorOrder = 5
        self.u = [U.rand_point() for _ in range(self.totalsamples)]

    def get_state_trajs(self, fun):
        x = []
        idx = 0
        for _ in range(0, self.initpoints*self.dim_x, self.dim_x):
            rand = self.R0.rand_point()
            tempx = [rand]
            curr_point = rand.tolist()
            for i in range(self.steps):
                tempx.append(fun(self.dt, tempx[i], self.u[idx])  + self.W.rand_point())
                idx += 1
                jdx = 0
                for ele in curr_point:
                    ele.append(tempx[i+1][jdx][0])
            x.append(curr_point)
        return x
    
    def combine_trajs(self, x):
        X_0t = [ele[:-1] for ele in x]
        X_1t = [ele[1:] for ele in x]
        X_0t = np.array(X_0t )
        X_0t = X_0t.tolist()
        ele = []
        for sample in X_0t:
            ele += sample
        X_0t = np.array(ele).reshape((self.dim_x, -1))
        print(X_0t.shape)
        X_1t = np.array(X_1t)
        X_1t = X_1t.tolist()
        ele = []
        for sample in X_1t:
            ele += sample
        X_1t = np.array(ele).reshape((self.dim_x, -1))
        print(X_1t.shape)
        U_full = np.array(self.u).reshape((-1, self.totalsamples))
        print(U_full.shape)
        self.X_0T = X_0t
        self.X_1T = X_1t
        self.U_full = U_full
        return X_0t, X_1t