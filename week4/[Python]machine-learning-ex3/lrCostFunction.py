import numpy as np
from sigmoid import sigmoid
def lrCostFunction(theta, X, y, lbd):
    """ Compute cost and gradient for logistic regression with 
    regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    
    Args
    theta = numpy array (weight matrix)
    X = numpy array (Input)
    y = numnpy array (Output -label)
    lbd = lambda (scalar)

    Return
    J : cost value
    grad: gradient descent (new theta)

    """
    m = len(y)
    z = X.dot(theta)
    h_z = sigmoid(z)
    reg = lbd/(2*m) * (theta[1:].T.dot(theta[1:]))
    J = (-1/m)*(y.T.dot(np.log(h_z)) + (1-y).T.dot(np.log(1-h_z))) + reg
    
    theta[0]= 0
    grad= (1/m)*X.T.dot((h_z-y)) + (lbd/m)*theta

    return J, grad

