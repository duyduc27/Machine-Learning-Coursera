from sigmoid import sigmoid
import numpy as np
def predict(Theta1, Theta2, X):
    """
    PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    
    Args
    Theta1: numpy array (trained weight)
    Theta2: numpy array (trained weight)
    X: numpy array

    Return:
    p: label of hand digits's number
    
    """

    # Useful values
    # X shape: 5000x 400
    # Theta1 shape: 25 x401
    # Theta2 shape: 10 x26


    m = X.shape[0] # 5000
    num_labels = Theta2.shape[0] # 10

    # You need to return the following variables correctly 
    p = np.zeros((X.shape[0], 1)) # 5000 x1

#  ====================== YOUR CODE HERE ======================
#  Instructions: Complete the following code to make predictions using
#                your learned neural network. You should set p to a 
#                vector containing labels between 1 to num_labels.
# 
#  Hint: The max function might come in useful. In particular, the max
#        function can also return the index of the max element, for more
#        information see 'help max'. If your examples are in rows, then, you
#        can use max(A, [], 2) to obtain the max for each row.    

    a1 = np.concatenate((np.ones((m, 1)), X), axis=1) # 5000x 401
    z2 = a1.dot(Theta1.T) # -> 5000 x 25
    a2 = sigmoid(z2)

    # a2 = [ones(size(a2,1), 1) a2]
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1) #5000x26

    z3 = a2.dot(Theta2.T) # -> 5000x 10
    a3= sigmoid(z3)
    # [prob, p] = max(a3, [], 2)
    p = np.argmax(a3, axis=1)
    return p + 1 # add 1 since python count from 0
                 # and data matlab's labels count from 1
