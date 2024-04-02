import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from mpmath import mp
mp.dps = 500
def accept(A, y, b, jmax, sign_bin):
    """Return bool c which y is correct with Ay < b
        Return matrix smatrix which is sign matrix A"""

    s = np.array([1]*3)
    for i in range(3):
        # 0 is negative(-)
        # 1 is positive(+)
        if sign_bin >> i & 1 == 0:
            s[i] = -1 

    smatrix = A.copy()
    smatrix[:, jmax] *= s
    
    # print(sign_, 'Ay < b', (smatrix @ y).T[0] <= b)
    
    if ((smatrix @ y).T[0] <= b).sum() == y.size:
        return 1, smatrix
    return 0, None



def run():
    mu = np.array([0, 0, 0]).reshape((3, 1))
    n = mu.size
    mean = 0
    sigma = 1
    epsilon = np.random.normal(mean, sigma, n).reshape((n,1))

    X = mu + epsilon

    #choose maxium element from X
    jmax = np.argmax(np.abs(X))

    eta = np.zeros((n, 1))
    eta[jmax][0] = 1.0

    etaTx_obs = (eta.T @ X)[0][0]

    print(f"x1: {X[0]}, x2: {X[1]}, x3: {X[2]} ")
    A = []
    b = []
    # |x_i| - |x_imax| <= 0
    for i in range(n):
        eta_i = np.zeros((n, 1))
        eta_i[i][0] = 1.0

        eta_imax = eta

        A.append((eta_i - eta_imax).T[0])
        b.append(0)

    A = np.array(A)
    # print(A)
    b = np.array(b)
    sign = X/np.abs(X)
    A = (A*sign).copy()
    print(A)
    #choose polyhedra (Ay < b)
    polyhedras =[]
    # if X[jmax] > 0:
    #     signjmax = 1
    # else:
    #     signjmax = -1
    for sign in range(2**3):
        correct, smatrix = accept(A,X,b,jmax, sign)
        if correct:
            polyhedras.append(smatrix)
    #     # print(smatrix)

    print(len(polyhedras), '(polyhedra)')
    # for polyhedra in polyhedras:
    #     print(polyhedra)

    Sigma = sigma*np.identity(n)

    # Compute vector c in Equation 5.3 of Lee et al. 2016
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
    c = np.dot(Sigma, eta) / etaT_Sigma_eta

    # Compute vector z in Equation 5.2 of Lee et al. 2016
    z = np.dot(np.identity(n) - np.dot(c, eta.T), X)

    # Az = np.dot(A, z)
    # Ac = np.dot(A, c)    

    Az = np.dot(polyhedras[1], z)
    Ac = np.dot(polyhedras[1], c)


    Vminus = np.NINF
    Vplus = np.Inf
    
    for j in range(len(b)):
        left = np.around(Ac[j][0], 5)
        right = np.around(b[j] - Az[j][0], 5)

        if left == 0:
            if right < 0:
                print('Error')
        else:
            temp = right / left

            if left > 0:
                Vplus = min(temp, Vplus)
            else:
                Vminus = max(temp, Vminus)

    # compute cdf of truncated gaussian distribution
    numerator = mp.ncdf(etaTx_obs / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)
    return selective_p_value

if __name__ == "__main__":
    max_iteration = 10000
    list_p_value = []

    alpha = 0.05
    count = 0

    for iter in range(max_iteration):
        if iter % 100 == 0:
            print(iter)

        selective_p_value = run()
        list_p_value.append(selective_p_value)

        if selective_p_value <= alpha:
            count = count + 1

    print()
    print('False positive rate:', count / max_iteration)
    plt.hist(list_p_value)
    plt.show()
    # run()