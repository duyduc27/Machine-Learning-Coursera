import numpy as np
def polyFeatures(X, p):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
    [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

    Args
    X: numpy array (data matrix X)
    p: integer (degree of polynomial features)

    Return
    X_poly: numpy array (polynomial matrix)
    """
    # You need to return the following variables correctly.
    X_poly = np.zeros((len(X), p))


    #   ====================== YOUR CODE HERE ======================
    #   Instructions: Given a vector X, return a matrix X_poly where the p-th 
    #                 column of X contains the values of X to the p-th power.

    X = X.reshape(-1,1)
    X_poly[:,:p] = X**(np.array([range(1,p+1)]))
    
    return X_poly