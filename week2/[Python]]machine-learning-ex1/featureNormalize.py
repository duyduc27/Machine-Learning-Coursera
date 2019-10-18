import numpy as np
def featureNormalize(X):
    """Normalizes the features in X 
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms 
    
    arg:
        X: numpy array, Input

    return:
        X_norm: normalized X
        mu: mean of X
        sigma: standard deviantion of X

    """
    mu = np.mean(X, 0) # (2,)
    
    sigma = np.std(X, 0) # (2,)
    
    X_norm = (X - mu)/sigma # 47 x2
    return X_norm, mu, sigma