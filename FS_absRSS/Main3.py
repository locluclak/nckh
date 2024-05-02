import numpy as np
from gendata import generate
import ForwardSelection as FS
from sklearn.linear_model import LinearRegression
from mpmath import mp
import scipy 
import matplotlib.pyplot as plt
mp.dps = 500


def run():
    #create data
    true_beta = np.array([0,0,0])
    n_sample = 50
    n_fea = len(true_beta)
    X, Y = generate(n_sample, n_fea, true_beta)

    #create A matrix & b vector
    A=[]
    b=[]

    SELECTION_F, r= FS.fixedSelection(Y, X,1) # r is rss of model, isn't important here
    # SELECTION_F = FS.Selection(Y, X)

    #step k
    k = len(SELECTION_F)
    
    for step in range(1,k+1):
        M_hat, r = FS.fixedSelection(Y, X, step)

        #Take selected data
        Xfs = X[:, M_hat[:step]].copy()

        #Residual vector of selected feature ee = KY
        I = np.identity(n_sample)
        K = I - Xfs @ np.linalg.pinv(Xfs.T @ Xfs) @ Xfs.T
 
        s = np.sign(K@Y).flatten()

        rss_row=0
        
        for i in range(n_sample):
            e = np.zeros((n_sample, 1))
            e[i][0] = 1
            rss_row += s[i] * e.T[0] @ K

        #condition (2) RSS > 0
        A.append(-1 * rss_row)
        b.append(0)


        #test RSS of chose features
        rss0 = (rss_row.T @ Y).item()

        for i in range(n_fea):
            if (i not in M_hat[:step]):
                
                #compute absRSS of B
                Xtemp = X[:, M_hat[:step - 1] + [i]].copy()

                I_ = np.identity(n_sample)
                K_ = I_ - Xfs @ np.linalg.pinv(Xfs.T @ Xfs) @ Xfs.T
        
                s_ = np.sign(K_@Y).flatten()

                rss_rown = 0
                for i_ in range(n_sample):
                    ee = np.zeros((n_sample, 1))
                    ee[i_][0] = 1
                    rss_rown += s_[i_] * ee.T[0] @ K
                #condition (2) RSS > 0
                b.append(0)
                A.append(-1*rss_rown)

                #condition (1) RSS(jk) - RSS(j) <= 0
                b.append(0)
                A.append(rss_row-rss_rown)



    A = np.array(A)
    b = np.array(b)

    #print(A.shape)
    #vector eta
    e1 = np.zeros((n_fea, 1))
    e1[np.random.choice(SELECTION_F)][0] = 1
    eta =  e1.T @ np.linalg.inv(X.T @ X) @ X.T
    eta = eta.reshape((-1,1))

    Sigma = np.identity(n_sample)
    obs = (eta.T @ Y).item()
 
    # Compute vector c in Equation 5.3 of Lee et al. 2016
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)
    etaT_Sigma_eta = etaT_Sigma_eta.item()
    c = np.dot(Sigma, eta) / etaT_Sigma_eta

    # Compute vector z in Equation 5.2 of Lee et al. 2016
    z = (np.identity(n_sample) - (c @ eta.T)) @ Y

    # Following Lemma 5.1 of Lee et al. 2016 to compute V^{-} and V^{+}
    Az = A @ z
    Ac = A @ c
  
    Vminus = np.NINF
    Vplus = np.Inf

    for j in range(len(b)):
        left = np.around(Ac[j][0], 5)
        right = np.around(b[j] - Az[j][0], 5)
        # print(left, right)
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
    numerator = mp.ncdf(obs / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)
    return selective_p_value

if __name__ == "__main__":
    # print(run())
    max_iteration = 1100
    list_p_value = []

    alpha = 0.05
    count = 0

    for iter in range(max_iteration):
        if iter % (max_iteration/100) == 0:
            print(iter/max_iteration * 100, "%" )

        selective_p_value = run()
        list_p_value.append(selective_p_value)

        if selective_p_value <= alpha:
            count = count + 1

    print()
    print('False positive rate:', count / max_iteration)
    print(scipy.stats.kstest(list_p_value, 'uniform'))
    plt.hist(list_p_value)
    plt.show()
