import matplotlib.pyplot as plt
import numpy as np
from control import c2d, ss
from numpy.linalg import pinv
from tqdm import tqdm
from Plot import plot_results
from Utils import reduce_girard
import time
#from numba import jit
import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
from Zonotope import Zonotope
from MatZonotope import MatZonotope
np.random.seed(5)


class linear_sys():
    """
    Linear time system
    x(k+1) = Ax(k) + Bu(k) + w(k)
    A, B, C, D represent the system dynamics

    X0: Initial reachable set zonotope
    U: Input zonotope
    W: Noise zonotope
    dim_x: State dimension
    initpoints:  number of initial points that will be propagated to get a simulation of the system
    steps: Number of steps
    samplingtime: Sampling time

    """

    def __init__(self, A: np.ndarray, B_ss: np.ndarray, C: np.ndarray, D, X0: Zonotope, U: Zonotope, W: Zonotope, dim_x: int, initpoints: int, steps: int, samplingtime, zonoOrder=3):
        if A.shape[0] != dim_x:
            raise ValueError(
                f"Shape error: A must be of shape ({dim_x}, {dim_x})")
        elif B_ss.shape[0] != dim_x:
            raise ValueError(
                f"Shape error: B must be of shape ({dim_x}, dim_u)")
        elif isinstance(X0, Zonotope) == False:
            raise TypeError(f"X0 must be of type Zonotope")
        elif isinstance(U, Zonotope) == False:
            raise TypeError(f"U must be of type Zonotope")
        elif isinstance(W, Zonotope) == False:
            raise TypeError(f"W must be of type Zonotope")
        self.X0 = X0
        self.U = U
        self.W = W
        self.sys_c = ss(A, B_ss, C, D)
        self.sys_d = c2d(self.sys_c, samplingtime)
        self.totalsamples = initpoints*steps
        self.initpoints = initpoints
        self.dim_x = dim_x
        self.steps = steps
        self.samplingtime = samplingtime
        self.zonoOrder = zonoOrder

    def Model_Based_Reachability(self, totalsteps: int, plot=False, save=''):
        """
        Compute the system data to compare to the reachability analysis results
        
        totalsteps: number of steps to build the model
        plot: if True Plots the model
        save: if not empty create a directory and saves plots of the model to that directory
        """
        model = [self.X0]
        print(f"Building model...")
        for i in tqdm(range(totalsteps)):
            model[i] = reduce_girard(model[i], self.zonoOrder)
            model.append(model[i] * self.sys_d.A +
                         self.U * self.sys_d.B + self.W)
        print("Model built!\n" + 60*"=" + "\n")
        if plot or save != '':
            plot_results(model, plot, save)
        return model

    def buildMw(self):
        """ 
        Builds the noise MatZonotope Mw from the noise zonotope W
        """
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
        GW = np.array(GW)
        Wmatzono = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), GW)
        Wmatzono.generators = Wmatzono.generators[0]
        return Wmatzono

    def simulate_sys(self, plot=False):
        """
        Simulate the system with the given parameters

        plot: if True Plots the simulated trajectories
        """
        x = []
        utraj = []
        print(f"Propagating {self.initpoints} initpoints {self.steps} time...")
        for _ in tqdm(range(self.initpoints)):
            #rand = np.array([[0.904], [0.9052], [0.9057]])
            rand = self.X0.rand_point()
            tempx = [rand]
            curr_point = rand.tolist()
            temp = []
            for i in range(self.steps):
                temp.append(self.U.rand_point())
                tempx.append(np.add(np.add(np.matmul(self.sys_d.A, tempx[i]), np.matmul(
                    self.sys_d.B, temp[i])), self.W.rand_point()))
                idx = 0
                for ele in curr_point:
                    ele.append(tempx[i+1][idx][0])
                    idx += 1
            utraj.append(temp)
            x.append(curr_point)
        print("Simulation done!\n" + 60*"=" + "\n")
        if plot:
            for j in range(len(x)):
                for ele in x[j]:
                    plt.plot(ele)
                plt.show()
        return x, utraj

    def concat_data_trajs(self, x, utraj):
        final_x = x[0]
        for dim in range(self.dim_x):
            for initpoint in range(1, len(x)):
                for step in range(self.steps):
                    final_x[dim].append(x[initpoint][dim][step])
        X_0t = np.array([ele[:-1] for ele in final_x])
        # print(X_0t.shape)
        X_1t = np.array([ele[1:] for ele in final_x])
        # print(X_1t.shape)
        U_full = np.array(utraj).reshape((-1, self.totalsamples))
        # print(U_full.shape)
        return X_0t, X_1t, U_full
    
    def get_AB(self, X_0t, X_1t, U_full):
        """Computes AB that are consistent with the simulated data"""
        Wmatzono = self.buildMw()
        X1W_cen = np.subtract(X_1t,  Wmatzono.center)
        X1W = MatZonotope(X1W_cen, Wmatzono.generators)
        AB = X1W * pinv(np.concatenate((X_0t, U_full), axis=0))
        intAB11 = AB.interval_matrix()
        intAB1 = intAB11.int
        return AB

    def Data_Driven_Reachability(self, totalsteps: int, plot=False, save=''):
        """
        Compute the forward reachable set of the system
        
        totalsteps: number of steps to be computed
        plot: if True Plots the results
        save: if not empty create a directory and saves plots of the results to that directory
        """
        x, utraj = self.simulate_sys()
        X_0t, X_1t, U_full = self.concat_data_trajs(x, utraj)
        AB = self.get_AB(X_0t, X_1t, U_full)
        X_data = [self.X0]
        print(f"Computing Reachability...")
        t1 = time.time()
        for i in tqdm(range(totalsteps)):
            #print(f"\nxdata[{i}] before: {X_data[i].generators().shape}")
            X_data[i] = reduce_girard(X_data[i], self.zonoOrder)
            #print(f"xdata[{i}] after: {X_data[i].generators().shape}")
            X_data.append(AB * X_data[i].cart_prod(self.U) + self.W)
        X_data[-1] = reduce_girard(X_data[-1], self.zonoOrder)
        t2 = time.time() - t1
        print("Reachability took {} seconds.\n\n".format(t2))
        if plot or save != '':
            plot_results(X_data, plot, save, x0=self.X0)
        return X_data

    #@jit(nopython=True)
    def Run_reachability(self, totalsteps, plot=False, save=''):
        """
        Runs both the reachability and model based approaches
        
        totalsteps: number of steps to be computed
        plot: if True Plots the results
        save: if not empty create a directory and saves plots of the results to that directory
        """
        model = self.Model_Based_Reachability(totalsteps)
        X_data = self.Data_Driven_Reachability(totalsteps)
        if plot or save != '':
            plot_results([model[1:], X_data], plot, save,
                         titles=['Model based Reachability', 'Data driven Reachability'], x0=self.X0)
        return model, X_data


if  __name__ == "__main__":
    steps = 50
    initpoints = 1
    dim_x = 5
    X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.1 *np.diag(np.ones((dim_x, 1)).T[0]))
    U = Zonotope(10, 0.25)
    W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.0005 * np.ones((dim_x, 1)))
    A = np.array([[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0],[0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]])
    B_ss = np.array([1, 1, 1, 1, 1])
    C = np.array([1, 0, 0, 0, 0])
    D = 0
    L_sys = linear_sys(A, B_ss, C, D, X0, U, W, dim_x, initpoints, steps, 0.05)
    model, X_data = L_sys.Run_reachability(5, plot=True)
    for path in sys.path:
        if path == "D:\\Desktop\\thesis\\brsl\\scripts\\reachability":
            sys.path.remove(path)