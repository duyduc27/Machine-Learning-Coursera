from sigmoid import sigmoid
import numpy as np
def costFunctionReg(theta, X, y, lbd):
    """
    Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.

    Args

    theta: numpy array (weight matrix)
    X : numpy array (Input)
    y: numpy array (Output)
    lbd: lambda - (scalar)

    Return

    J: cost value
    grad: gradient descent
    """
    grad = np.zeros(theta.shape) # initial theta
    m = len(y) # number of samples
    z = X.dot(theta) 
    h_z = sigmoid(z) 
    reg = lbd/(2*m) * (theta[1:].T.dot(theta[1:]))
    J = (-1/m) *(y.T.dot(np.log(h_z)) + (1-y).T.dot(np.log(1- h_z))) + reg
    theta[0] = 0
    grad = (1/m)*X.T.dot((h_z - y)) + (lbd/m)*theta

    return J, grad