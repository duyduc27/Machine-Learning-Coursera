from sigmoid import sigmoid
def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function
    evaluated at z
    g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.

    Arg
    z: scalar, matrix (array)

    return: sigmoid gradient of z (numpy array)
    """
    return sigmoid(z)*(1- sigmoid(z))