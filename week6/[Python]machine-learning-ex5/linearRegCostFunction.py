import numpy as np
def linearRegCostFunction(X, y, theta, lbd):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    regression with multiple variables
    [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    cost of using theta as the parameter for linear regression to fit the 
    data points in X and y. Returns the cost in J and the gradient in grad


    Args
    X : numpy array (training set)
    y : numpy array (output of training set)
    theta : numpy array (weight matrix)
    lbd : scalar (lambda of regularization)

    Return
    J: value of cost function
    grad: gradient descent
    """

    m = len(y)

    J = 0
    

    # force to be 2D vector
    # theta = np.reshape(theta, (-1,y.shape[1]))    
    theta = theta.reshape(-1,1)

    grad = np.zeros(theta.shape)
    #  ====================== YOUR CODE HERE ======================
    #  Instructions: Compute the cost and gradient of regularized linear 
    #                regression for a particular choice of theta.
    # 
    #                You should set J to the cost and grad to the gradient.
    # 

    h_x = X.dot(theta) # 12x1
    reg = lbd/(2*m)* theta[1:].T.dot(theta[1:])
    J = 1/(2*m)* (h_x-y).T.dot((h_x- y)) + reg
    
    # grad = (1/m)* np.dot(X.T, h_x- y) + (lbd/m)*theta
    # grad_no_regularization =  (1/m)* np.dot(X.T, h_x-y)
    # grad[0] = grad_no_regularization[0]
    grad[0] = (1/m)* X[:,0].T.dot(h_x-y)
    grad[1:] = (1/m)* X[:, 1:].T.dot(h_x-y) + (lbd/m)*theta[1:]

    return J, grad.flatten()