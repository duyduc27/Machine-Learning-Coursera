import numpy as np
from computeCostMulti import computeCostMulti
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """ Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    y = y.reshape(-1,1)
    for i in range(num_iters):
        h = X.dot(theta)
        error = h - y
        
        theta += -(alpha/m) *(X.T.dot(error))

        J_history[i] = computeCostMulti(X,y, theta)
    
    return theta, J_history