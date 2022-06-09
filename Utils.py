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

def interval_square(op1):
    if isinstance(op1, Interval):
        infs = get_elements_of_nested_list(op1.inf.tolist())
        sups = get_elements_of_nested_list(op1.sup.tolist())
        return Interval(np.array([ele**2 for ele in infs]), np.array([ele**2 for ele in sups]))
    else:
        raise Exception("Interval square is only defined for Intervals!")

def interval_mul_2(op1, op2):
    infs1 = get_elements_of_nested_list(op1.inf.tolist())
    sups1 = get_elements_of_nested_list(op1.sup.tolist())
    infs2 = get_elements_of_nested_list(op2.inf.tolist())
    sups2 = get_elements_of_nested_list(op2.sup.tolist())
    final_inf = []
    final_sup = []
    for i in range(len(infs1)):
        final_inf.append(infs1[i]*infs2[i])
        final_sup.append(sups1[i]*sups2[i])
    return Interval(np.array(final_inf), np.array(final_sup))

def Interval_multiplication(op1, op2):
    if isinstance(op1, Interval) and isinstance(op2, Interval):
        if op1.inf.shape[0] == 1 and op1.sup.shape[0] == 1 and op2.inf.shape[0] == 1 and op2.sup.shape[0] == 1:
            i =  Interval()
            i.inf = min([op1.inf[0]*op2.inf[0], op1.inf[0]*op2.sup[0], op1.sup[0]*op2.inf[0], op1.sup[0]*op2.sup[0]])
            i.sup = max([op1.inf[0]*op2.inf[0], op1.inf[0]*op2.sup[0], op1.sup[0]*op2.inf[0], op1.sup[0]*op2.sup[0]])
            return i
        else:
            I1 = op1.inf
            S1 = op1.sup
            m = I1.shape[0]
            n1 = op2.inf.shape[1]
            A = Interval()
            Binf = []
            Bsup = []
            for i in range(m):
                A.inf = matlib.repmat(I1[i, :],n1, 0).conj().T
                A.sup = matlib.repmat(S1[i, :],n1, 0).conj().T
                B = Interval_multiplication(A, op2)
                Binf.append(B.inf.sum(axis=0))
                Bsup.append(B.sup.sum(axis=0))
            return Interval(np.array(Binf), np.array(Bsup))
    elif isinstance(op1, Interval) and isinstance(op2, float):
        i =  Interval()
        i.inf = op1.inf*op2
        i.sup = op1.sup*op2
        return i
    elif isinstance(op1, float) and isinstance(op2, Interval):
        i =  Interval()
        i.inf = op1*op2.inf
        i.sup = op1*op2.sup
        return i
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
    
def zonoToInterval(zono):
    c = zono.center()
    delta = np.sum(np.abs(zono.Z)) - np.abs(c)
    leftLimit = np.subtract(c, delta)
    rightLimit = np.add(c, delta)
    return Interval(leftLimit,rightLimit)

def intervals_To_interval(intervals):
    infs = []
    sups = []
    for currint in intervals:
        infs += get_elements_of_nested_list(currint.inf.tolist())
        sups += get_elements_of_nested_list(currint.sup.tolist())
    return Interval(np.array(infs).reshape(-1, 1), np.array(sups).reshape(-1, 1))

def get_elements_of_nested_list(element):
    res = []
    if isinstance(element, list):
        for each_element in element:
            res += get_elements_of_nested_list(each_element)
    else:
        res += [element]
    return res