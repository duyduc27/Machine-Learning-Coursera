""" Machine Learning Online Class - Exercise 1: Linear Regression

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions
  in this exericse:

     warmUpExercise.m
     plotData.m
     gradientDescent.m
     computeCost.m """

"""==================== Part 1: Basic Function ===================="""
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
import warmUpExercise
warmUpExercise.warmUpExercise()

"""======================= Part 2: Plotting ======================="""
print('\nPlotting Data ...\n')
import pandas as pd
data = pd.read_csv("ex1data1.txt", sep=',', header=None)
x = data[0]
y = data[1]
m = len(y)

import plotData
plotData.plotData(x, y)

"""=================== Part 3: Cost and Gradient descent ==================="""

import numpy as np
y = np.array(y)
y = y.reshape((y.shape[0], 1))
X = np.c_[x, np.ones((x.shape[0]))] # Add a column of ones to x

theta = np.zeros((2, 1)) # initialize fitting parameters

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

import computeCost
J = computeCost.computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost.computeCost(X, y, np.array([[2] , [-1]]))
print('\nWith theta = [2 ; -1]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 54.24\n')


print('\nRunning Gradient Descent ...\n')
# run gradient descent
import gradientDescent
theta = gradientDescent.gradientDescent(X, y, theta, alpha, iterations) 

# print theta to screen
print('Theta found by gradient descent:\n')
print('\n', theta)
print('Expected theta values (approx)\n')
print('  1.1664\n -3.6303\n\n')