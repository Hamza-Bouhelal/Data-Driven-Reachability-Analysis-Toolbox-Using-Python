import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import numpy as np
from Zonotope import Zonotope
from Interval import Interval
import numpy.matlib as matlib

def picked_generators(zono, order):
        Z = Zonotope()
        Z = Z.copy(zono)
        c = Z.center()
        G = Z.generators()
        Gunred = np.array([])
        Gred = np.array([])
        if(np.sum(G.shape) != 0):
            G = zono.nonzero_filter(G)
            d, nr_of_gens = G.shape
            if(nr_of_gens > d * order):
                h = np.apply_along_axis(lambda row: np.linalg.norm(row, ord=1), 0, G) - np.apply_along_axis(lambda row: np.linalg.norm(row, ord=np.inf), 0, G)
                n_unreduced = np.floor(d * (order - 1))
                n_reduced = int(nr_of_gens - n_unreduced)
                idx = np.argpartition(h, n_reduced - 1)
                Gred = G[:, idx[: n_reduced]]
                Gunred = G[:, idx[n_reduced:]]
            else:
                Gunred = G
        return c, Gunred, Gred

def reduce_girard(zono, order):
        Zred = Zonotope()
        Zred = Zred.copy(zono)
        center, Gunred, Gred = picked_generators(Zred, order)
        #print('shapes ', center.shape, Gunred.shape, Gred.shape)
        if(Gred.size == 0):
            Zred.Z = np.hstack((center, Gunred))
        else:
            d = np.sum(np.abs(Gred), axis=1)
            Gbox = np.diag(d)
            center = center.reshape((center.shape[0], -1))
            Gunred = Gunred.reshape((center.shape[0], -1))
            Gbox = Gbox.reshape((center.shape[0], -1))
            Zred.Z = np.hstack((center, Gunred, Gbox))
        return Zred

def Interval_multiplication(op1, op2):
    if isinstance(op1, Interval) and isinstance(op2, Interval):
        I1 = op1.inf
        S1 = op1.sup
        try:
            m, n = I1.shape
        except:
            print('I1.shape', I1.shape)
            m, n = I1.shape[0]
        try:
            m1, n1 = op2.inf.shape
        except:
            m1, n1 = op2.inf.shape[0]
        A = Interval()
        Binf = []
        Bsup = []
        for i in range(m):
            A.inf = matlib.repmat(I1[i, :],n1, 0).conj().T
            A.sup = matlib.repmat(S1[i, :],n1, 0).conj().T
            B = Interval_multiplication(A, op2)
            Binf.append(B.inf.sum(axis=0))
            Bsup.append(B.sup.sum(axis=0))
        print("done")
        return Interval(np.array(Binf), np.array(Bsup))
    else:
        raise Exception("Interval multiplication is only defined for Intervals!")

def Interval_selector(obj, S):
    if isinstance(obj, Interval):
        newObj = Interval(np.array(obj.inf), np.array(obj.sup))
        if len(S)==1:
            newObj.inf=obj.inf[S[0]]
            newObj.sup=obj.sup[S[0]]
        elif len(S)==2:
                row=S[0]
                column=S[1]
                newObj.inf=obj.inf[row,column]
                newObj.sup=obj.sup[row,column]
        return newObj
    else:
        raise Exception("Interval selector is only defined for Intervals!")