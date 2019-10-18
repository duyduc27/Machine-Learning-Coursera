def computeCostMulti(X, y, theta):
    """
    This function computes cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y

    Args:
        X: numpy array, Input numpy array to train linear regression model
        y: numpy array, Output true label to train linear regression model
        theta: numpy array, weight matrix

    Returns:
        J: float, cost value of linear regression model
    """
    y = y.reshape(-1,1)
    m = len(y) # number of samples
    h = X.dot(theta)
    J = (1/(2*m)) * (h-y).T.dot(h-y)
    return J