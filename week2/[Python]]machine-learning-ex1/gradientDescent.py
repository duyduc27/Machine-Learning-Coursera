import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha

    args:
        X: numpy array, Input
        y: numpy array, Output
        theta: numpy array, weight matrix
        alpha: learning rate
        num_iters: number of epochs
    
    return:
        theta: the last theta after updates
        J_history: values of cost function
    """
    J_history = []
    m = len(y) # number of samples
    for i in range(num_iters):
        error = X.dot(theta) - y
        gradient = X.T.dot(error)
        theta += - alpha/m *gradient

        J_history.append( computeCost(X, y, theta) )
    return theta, J_history
