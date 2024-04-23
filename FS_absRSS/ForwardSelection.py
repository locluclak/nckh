import numpy as np
from sklearn.linear_model import LinearRegression

def Selection(Y, X):
    #Y = Y.flatten()
    Cp = np.inf
    n = X.shape[0]

    fullmodel = np.linalg.inv(X.T @ X) @ X.T @ Y

    sigma2 = RSS(Y,X, fullmodel) / X.shape[1]

    for i in range(1, X.shape[1] + 1):
        sset, rss = fixedSelection(Y, X, i)
        d = i
        cp = (rss + 2*d*sigma2) / n
        if cp < Cp:
            bset = sset
            Cp = cp
    return bset


def fixedSelection(Y, X, k):
    selection = []
    rest = list(range(X.shape[1]))

    #i = 1
    for i in range(1, k+1):
        rss = np.inf
        sele = selection.copy()
        selection.append(None)
        for feature in rest:
            #select nessesary data
            X_temp = X[:, sele + [feature]].copy()
            #create linear model
            model = np.linalg.pinv(X_temp.T @ X_temp) @ X_temp.T @ Y
            #calculate rss of model
            rss_temp = RSS(Y, X_temp, model)
            
            # choose feature having minimum rss and append to selection
            if rss > rss_temp:
                rss = rss_temp
                selection.pop()
                selection.append(feature)

    return selection, rss
def RSS(Y, X, coef, intercept = 0):
    RSS = 0
    for i_sample in range(np.shape(X)[0]):
        
        Y_hat = (X[i_sample] @ coef).item() + intercept
        RSS += np.abs(Y[i_sample][0] - Y_hat)
    return RSS        