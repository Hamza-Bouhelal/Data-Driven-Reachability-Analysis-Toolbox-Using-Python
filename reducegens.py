import sys
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
import numpy as np
from Zonotope import Zonotope
from MatZonotope import MatZonotope
from numpy import linalg as LA

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