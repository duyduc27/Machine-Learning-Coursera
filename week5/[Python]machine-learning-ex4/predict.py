import numpy as np 
from sigmoid import sigmoid
def predict(Theta1, Theta2, X):
    """
    PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    # Useful values
    m = X.shape[0] # 5000
    num_labels = Theta2.shape[0] # 10

    # You need to return the following variables correctly 
    p = np.zeros((X.shape[0], 1))

    a1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) #add bias 1 -> 5000x401
    a2 = sigmoid(a1.dot(Theta1.T)) # 5000 x25
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1) #add bias 1 -> 5000x26

    a3 = sigmoid(a2.dot(Theta2.T)) # 5000x10

    p = np.argmax(a3, axis=1)

    return p+1
