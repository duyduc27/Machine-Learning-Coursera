from sigmoid import sigmoid
import numpy as np
def costFunction(theta, X, y):
    """Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.

    Args

    theta: numpy array (weight matrix)
    X : numpy array (Input)
    y : numpy array (Output)

    Return

    J : cost value
    grad: gradient descent (or the last theta)
    """
    m = len(y)
    z = X.dot(theta) #100x1
    h_z = sigmoid(z) # 100x1
    J = (-1/m) *(y.T.dot(np.log(h_z)) + (1-y).T.dot(np.log(1- h_z))) # cost
    grad = (1/m)* X.T.dot((h_z - y)) # gradient
    return J, grad