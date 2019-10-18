import numpy as np
def sigmoid(z):
    """Compute sigmoid function
    g = SIGMOID(z) computes the sigmoid of z.
   
    arg
    z: can be a matrix, vector or scalar

    return: the sigmoid of z
    """
    return 1.0/(1+ np.exp(-z))
