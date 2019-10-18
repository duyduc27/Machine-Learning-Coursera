import numpy as np
def mapFeature(x1, x2):
    """
    Feature mapping function to polynomial features
 
    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.
 
    Returns a new feature array with more features, comprising of 
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
 
    Inputs X1, X2 must be the same size
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out