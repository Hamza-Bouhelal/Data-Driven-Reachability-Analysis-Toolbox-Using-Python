import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import matplotlib.pyplot as plt
import numpy as np
from control import c2d, ss
from MatZonotope import MatZonotope
from Zonotope import Zonotope 
from numpy.linalg import pinv
from tqdm import tqdm
from Plot import plot_results
from reducegens import reduce_girard
np.random.seed(5)

class linear_sys():
    def __init__(self, A:np.ndarray, B_ss:np.ndarray, C:np.ndarray, D, X0:Zonotope, U:Zonotope, W:Zonotope, dim_x:int, initpoints:int, steps:int, samplingtime):
        if A.shape[0] != dim_x:
            raise ValueError(f"Shape error: A must be of shape ({dim_x}, {dim_x})")
        elif B_ss.shape[0] != dim_x:
            raise ValueError(f"Shape error: B must be of shape ({dim_x}, dim_u)")
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
        
    def build_model(self, totalsteps:int, plot=False, save=''):
        model = [self.X0]
        print(f"Building model...")
        for i in tqdm(range(totalsteps)):
            model[i] = reduce_girard(model[i], 10)
            model.append(model[i] * self.sys_d.A + self.U * self.sys_d.B + self.W)
        print("Model built!\n" + 60*"=" + "\n")
        if plot or save != '':
            plot_results(model, plot, save)
        return model


    def buildMw(self):
        GW = []
        for i in range(self.W.generators().shape[1]):
            vec = np.reshape(self.W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(np.hstack((vec, np.zeros((self.dim_x, self.totalsamples - 1)))))
            for j in range(1, self.totalsamples, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            GW.append(np.array(dummy))
        GW = np.array(GW)
        Wmatzono = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), GW)
        Wmatzono.generators = Wmatzono.generators[0]
        return  Wmatzono

    def simulate_sys(self, plot=False):
        x = []
        utraj = []
        print(f"Propagating {self.initpoints} initpoints {self.steps} time...")
        for _ in tqdm(range(self.initpoints)):
            #rand = np.array([[0.904], [0.9052], [0.9057], [0.9492], [1.072]])
            rand = self.X0.rand_point()
            tempx = [rand]
            curr_point = rand.tolist()
            temp = []
            for i in range(self.steps):
                temp.append(self.U.rand_point())
                tempx.append(np.add(np.add(np.matmul(self.sys_d.A, tempx[i]), np.matmul(self.sys_d.B,temp[i])), self.W.rand_point()))
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
        U_full = np.array(utraj).reshape((-1, self.totalsamples))
        print(U_full.shape)
        Wmatzono = self.buildMw()
        X1W_cen = X_1t - Wmatzono.center
        X1W = MatZonotope(X1W_cen, Wmatzono.generators)
        AB = X1W * pinv(np.concatenate((X_0t, U_full), axis=0))
        intAB11 = AB.interval_matrix()
        intAB1 = intAB11.int
        return AB

    def reach(self, totalsteps:int, plot=False, save=''):
        x, utraj = self.simulate_sys()
        AB = self.concat_data_trajs(x, utraj)
        X_data = [self.X0]
        print(f"Computing Reachability...")  
        for i in tqdm(range(totalsteps)):
            #print(f"\nxdata[{i}] before: {X_data[i].generators().shape}")
            X_data[i] = reduce_girard(X_data[i], 10)
            #print(f"xdata[{i}] after: {X_data[i].generators().shape}")
            X_data.append(AB * X_data[i].cart_prod(self.U) + self.W)
        if plot or save !='':
            plot_results(X_data, plot, save)
        return X_data
    
    def Run_reachability_and_model_based(self, totalsteps, plot=False, save=''):
        model = self.build_model(totalsteps)
        X_data = self.reach(totalsteps)
        if plot or save !='':
            plot_results([model, X_data], plot, save, titles=['Model', 'Reachability'])
        return model, X_data

        
    
if __name__ == "__main__":
    steps = 120
    initpoints = 1
    dim_x = 5
    X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.1 * np.diag(np.ones((dim_x,1 )).T[0]))
    U = Zonotope(10, 0.25)
    W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.005 * np.ones((dim_x, 1)))
    A = np.array([[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]])
    B_ss = np.ones([5, 1])
    C = np.array([1,0,0,0,0])
    D = 0
    L_sys = linear_sys(A, B_ss, C, D, X0, U, W, dim_x, initpoints, steps, 0.05)
    #data = L_sys.Run_reachability_and_model_based(5, True)
    x, utraj = L_sys.simulate_sys()
    AB = L_sys.Run_reachability_and_model_based(25, True)
