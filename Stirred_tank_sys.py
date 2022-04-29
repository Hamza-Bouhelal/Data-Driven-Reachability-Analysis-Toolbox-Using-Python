import numpy as np
from scipy.io import loadmat
import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
from Zonotope import Zonotope


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

def read_matlab(filename, var):
    mat = loadmat(filename)
    return mat[var]

def load_model():
    return [Zonotope(read_matlab(f'model_for_stirred_tank\\{i}.mat', 'var')) for i in range(0, 2)]
