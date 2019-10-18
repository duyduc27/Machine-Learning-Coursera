import numpy as np
def computeCost(X, y, theta):
    """This function computes cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y

    Args:
        X: numpy array, Input numpy array to train linear regression model
        y: numpy array, Output true label to train linear regression model
        theta: numpy array, weight matrix

    Returns:
        J: float, cost value of linear regression model
    """
    m = len(y) # number of training samples
    h = X.dot(theta)
    error = h - y
    J = (1.0/(2*m))*(error.T.dot(error))
    return J