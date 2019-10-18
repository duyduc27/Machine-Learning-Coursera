from sigmoid import sigmoid
import numpy as np
def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic 
    regression parameters theta
    p = PREDICT(theta, X) computes the predictions for X using a 
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    Args

    theta: numpy array (weight matrix)
    X: numpy array (Input)
    
    Return: label array
    """
    label_array = np.array([]) # initial array
    result_array = sigmoid(X.dot(theta))
    for i in result_array:
        if i >= 0.5:
            label_array= np.append(label_array, 1)
        else:
            label_array= np.append(label_array, 0)
    return label_array
