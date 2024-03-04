import numpy as np
from scipy.optimize import linprog

def convert(m, n):
    A = np.arange(m*n)
    A = A.reshape((m,n))
    B = []
    
    for row in A:
        temp = np.zeros(m*n)
        for ele in row:
            temp[ele] = 1
        B.append(temp)
    for col in A.T:
        temp = np.zeros(m*n)
        for ele in col:
            temp[ele] = 1
        B.append(temp)
    return np.array(B)

def cost_matrix(X, Y):
    cost = np.array([])
    for dotX in X:
        for dotY in Y:
            dist = np.linalg.norm(dotX - dotY)
            cost = np.append(cost, dist**2)
            
    return cost.reshape(X.shape[0], Y.shape[0])

def emd(cost_matrix, source, target):
    source = source/sum(source)
    target = target/sum(target)
    n_source, n_target = len(source), len(target)
    
    l = cost_matrix.ravel()
    aeq = convert(n_source, n_target)
    beq = np.append(source, target)
    
    bound_x = [(0, None) for _ in range(n_source*n_target)]
    res = linprog(l, A_eq = aeq, b_eq = beq, bounds = bound_x, method = 'highs')
    #res = linprog(l, A_eq = aeq, b_eq = beq)#, method = 'highs')
    return res
