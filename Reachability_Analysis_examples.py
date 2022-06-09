from Linear_sys import linear_sys
import NonLinear_sys as nl_sys
import Poly_sys as pl_sys
from Plot import plot_results
import numpy as np
import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
from Zonotope import Zonotope


np.random.seed(3)

def Linear_sys_example():
    steps = 100
    initpoints = 1
    dim_x = 3
    X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.15 *np.diag(np.ones((dim_x, 1)).T[0]))
    U = Zonotope(10, 0.25)
    W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.005 * np.ones((dim_x, 1)))
    A = np.array([[1, 0, 0], [1, 1, 0],[0, 1, 1]])
    B_ss = np.array([1, -1, 0])
    C = np.array([1, 0, 0])
    D = 0
    L_sys = linear_sys(A, B_ss, C, D, X0, U, W, dim_x, initpoints, steps, 0.05)
    model, X_data = L_sys.Run_reachability(6, plot=True)

def NonLinear_sys_example():
    dim_x = 2
    U = Zonotope(np.array(np.array([0.01, 0.01]).reshape((2, 1))),np.diag([0.1, .2]))
    R0 = Zonotope(np.array([-1.9, -20]).reshape((dim_x, 1)),np.diag([0.005, .3]))
    dt = 0.015
    initpoints = 1
    steps = 20
    wfac = 1e-4
    nl_sys1 = nl_sys.NonLinear_sys(dt, U, R0, wfac, dim_x, initpoints, steps, nl_sys.cstrdiscr)
    data = nl_sys1.Data_Driven_Reachability(5, plot=False)
    model = nl_sys.load_model()
    plot_results([model, data], True, "", ["Model based Reachability", "Data driven Reachability"], x0=R0)

def Poly_sys_example():
    N = 3
    dt = 0.015
    U = Zonotope(np.array(np.array([0.2, 0.3]).reshape((2, 1))),
                 np.diag([0.01, .02]))
    R0 = Zonotope(np.array([1, 2]).reshape((2, 1)),
                  np.diag([0.05, .3]))
    dim_x = 2
    initpoints = 1
    steps = 7
    wfac = 0.000007
    poly_sys = pl_sys.Poly_sys(dt, U, R0, wfac, dim_x,
                        initpoints, steps, pl_sys.poly_func)
    data = poly_sys.Data_Driven_Reachability(N, False)
    model = pl_sys.load_model(N)
    plot_results([model, data], True, "", ["Model based Reachability", "Data driven Reachability"], x0=R0)

Linear_sys_example()
NonLinear_sys_example()
Poly_sys_example()
    
