
import numpy as np
from sklearn.linear_model import LinearRegression
def fw_selec(Y, X):
    Cp = np.inf
    n = X.shape[0]

    fullmodel = LinearRegression(fit_intercept=False)
    fullmodel.fit(X, Y)
    sigma2 = RSS(Y,X, fullmodel.coef_ , fullmodel.intercept_) / X.shape[1]

    for i in range(1, X.shape[1] + 1):
        sset, rss = fsfix(Y, X, i)
        d = i
        cp = (rss + 2*d*sigma2) / n
        if cp < Cp:
            bset = sset
            Cp = cp
    return bset


def fsfix(Y, X, k):
    selection = []
    rest = list(range(X.shape[1]))

    #i = 1
    for i in range(1, k+1):
        rss = np.inf
        sele = selection.copy()
        selection.append(None)
        for feature in rest:
            #remove unnessesary feature
            # print(sele + [feature])
            X_temp = X[:, sele + [feature]]# = X.filter(sele + [feature])
            
            #create linear model
            F = LinearRegression(fit_intercept = False)
            F.fit(X_temp, Y)
            #calculate rss of model
            rss_temp = RSS(Y, X_temp, F.coef_, F.intercept_)
            #rss_temp = calculate_rss(X_temp, Y)
            
            # choose feature having minimum rss and append to selection
            if rss > rss_temp:
                rss = rss_temp
                selection.pop()
                selection.append(feature)

    return selection, rss
def RSS(Y, X, coef, intercept = 0):
    RSS = 0;
    for i_sample in range(np.shape(X)[0]):
        
        Y_hat = np.dot(X[i_sample], coef) + intercept
        RSS += (Y[i_sample] - Y_hat)**2
    return RSS        