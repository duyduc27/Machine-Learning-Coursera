from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
import numpy as np
def learningCurve(X, y, Xval, yval, lbd):
    """
    LEARNINGCURVE Generates the train and cross validation set errors needed 
    to plot a learning curve
    [error_train, error_val] = ...
    LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    cross validation set errors for a learning curve. In particular, 
    it returns two vectors of the same length - error_train and 
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.

    Args
    X: numpy array (X train)
    y: numpy array (y train)
    Xval: numpy array (X validation)
    yval: numpy array (y validation)
    lbd: scalar (lambda)

    Return
    error_train: numpy array (train errors)
    error_val: numpy array (val errors)        

    """
    # Number of training examples
    m = len(X)
    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    for i in range(m):
        Xtrain = X[:i+1, :]
        ytrain = y[:i+1]
        theta = trainLinearReg(Xtrain, ytrain, lbd)

        error_train[i] = linearRegCostFunction(Xtrain, ytrain, theta, 0)[0] 
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)[0]
    
    return error_train, error_val