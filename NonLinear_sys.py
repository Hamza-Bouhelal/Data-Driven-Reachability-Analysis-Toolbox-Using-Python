import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import numpy as np
from Zonotope import Zonotope
from MatZonotope import MatZonotope
from utils.Options import Options
from utils.Params import Params
from reachability_nonlinear import reach_DT
from Plot import plot_results
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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
    U =np.matmul(np.array([-3, -6.9]), x)
    x_temp =np.array([x[0, 0] + C_A0, x[1, 0]+T_0]).reshape(2, 1)
    U = U + np.array([T_c0])
    f1 = ((1-(q*dt)/(2*V) - k0*dt*np.exp(-E_R/x_temp[1, 0]))*x_temp[0, 0] + q/V * C_Af * dt)/(1 + (q*dt)/(2*V)) + u[0]*dt
    f2 = (x_temp[1, 0]*(1-0.5*dt- (dt*UA)/(2*V*rho*Cp)) + dt*(Tf*q/V + (UA*U)/(V*rho*Cp))- x_temp[0, 0]*(deltaH*k0*dt)/(rho*Cp) * np.exp(-E_R/x_temp[1, 0])) / (1+0.5*dt*q/V+(dt*UA)/(2*V*rho*Cp)) + u[1, 0]*dt
    f1 = f1 - C_A0
    f2 = f2 - T_0
    return np.array([f1, f2]).reshape(2, 1)


class NonLinear_sys():
    def __init__(self, dt, U, R0, wfac, dim_x, initpoints, steps):
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

        self.options.params["zonotopeOrder"] = 100
        self.options.params["tensorOrder"] = 2
        self.options.params["errorOrder"] = 5
        self.u = [U.rand_point() for _ in range(self.totalsamples)]
        self.params.params["U"] = U
        self.params.params["R0"] = R0

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
        final_x = x[0]
        for initpoint in range(1, len(x)):
            for dim in range(self.dim_x):
                for step in range(self.steps):
                    final_x[dim].append(x[initpoint][dim][step])
        s = np.array(x)
        print(s.shape)
        X_0t = np.array([ele[:-1] for ele in final_x])
        print(X_0t.shape)
        X_1t = np.array([ele[1:] for ele in final_x])
        print(X_1t.shape)
        U_full = np.array(self.u).reshape((-1, self.totalsamples))
        print(U_full.shape)
        self.options.params["X_0T"] = X_0t
        self.options.params["X_1T"] = X_1t
        self.options.params["U_full"] = U_full
        return X_0t, X_1t

    def compute_Lipschits_const(self, X_0t, X_1t):
        L = 0
        eps = 0
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
                        eps = L* np.linalg.norm(z1-z2)
        print(f"l: {L}")
        self.options.params["Zeps"] = Zonotope(np.array(np.zeros(
            (self.dim_x, 1))), eps * np.diag(np.ones((self.options.params["dim_x"], 1)).T[0]))
        self.options.params["ZepsFlag"] = True
    
    def run_reachability(self, fun, plot=False, save=''):
        x = self.get_state_trajs(fun)
        X_0t, X_1t = self.combine_trajs(x)
        self.compute_Lipschits_const(X_0t, X_1t)
        print("Computing reachability...")
        t1 = time.time()
        data =  reach_DT(self.params, self.options)
        t2 = time.time() - t1
        print("Reachability took {} seconds.\n\n".format(t2))
        if plot or save !='':
            plot_results(data, plot, save)
        return data
    
if __name__ == "__main__":
    dim_x = 2
    U = Zonotope(np.array(np.array([0.01, 0.01]).reshape((2, 1))),
        np.diag([0.1, .2]))
    R0 = Zonotope(np.array([-1.9, -20]).reshape((dim_x, 1)),
        np.diag([0.005, .1]))
    dt = 0.015
    initpoints = 1
    steps = 120
    wfac = 1e-4
    nl_sys = NonLinear_sys(dt, U, R0, wfac, dim_x, initpoints, steps)
    data = nl_sys.run_reachability(cstrdiscr, True)

