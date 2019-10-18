import numpy as np
import lrCostFunction as lrcf
from scipy.optimize import minimize
def oneVsAll(X, y, num_labels, lbd):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i
"""
    m, n = X.shape
    y = y.ravel()

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    print(all_theta.shape)

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))
    print(X.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda. 

    for c in range(num_labels):

        # initial theta for c/class
        initial_theta = np.zeros((n + 1, 1))
        print(initial_theta.shape)

        print("Training {:d} out of {:d} categories...".format(c+1, num_labels))

        myargs = (X, (y%10==c).astype(int), lbd)
        theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)

        all_theta[c,:] = theta["x"]

    return all_theta