import numpy as np
def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression 
    NORMALEQN(X,y) computes the closed-form solution to linear 
    regression using the normal equations.
        Args:

        X: numpy array, Input
        y: numpy array, Output

        Return:
        theta: numpy array 

    """
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta