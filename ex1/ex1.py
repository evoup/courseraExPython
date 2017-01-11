# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot
import numpy as np

from computeCost import computeCost
from warmUpExercise import warmUpExercise

# import scipy.io
import os


## Machine Learning Online Class - Exercise 1: Linear Regression

##  Instructions
##  ------------
##
##  This file contains code that helps you get started on the
##  linear exercise. You will need to complete the following functions
##  in this exericse:
##
##     warmUpExercise.m
##     plotData.m
##     gradientDescent.m
##     computeCost.m
##     gradientDescentMulti.m
##     computeCostMulti.m
##     featureNormalize.m
##     normalEqn.m
##
##  For this exercise, you will not need to change any code in this file,
##  or any other files other than those mentioned above.
##
## x refers to the population size in 10,000s
## y refers to the profit in $10,000s
##
##
## % Initialization
## ==================== Part 1: Basic Function ====================
## Complete warmUpExercise.m
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

warmUpExercise()

# raw_input('Program paused. Press enter to continue.\n')


## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
arr = np.loadtxt(os.getcwd() + '/ex1data1.txt', delimiter=',', usecols=(0, 1), unpack=True)

# feature x vector
X = arr[0]

# result y vector
y = arr[1]

# sample number totoal is :97
m = len(y)

print('number of samples:%s' % m)

# X = [6.1101, 5.5277,...]
# y = [17.592, 9.1302,...]

matplotlib.pyplot.scatter(X, y)

matplotlib.pyplot.show()



#raw_input('Program paused. Press enter to continue.\n');
##%% =================== Part 3: Gradient descent ===================
# convert to column vec
X.shape = (97, 1)


##X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

#insert one col before it
#np.insert param(original_matrix, offset, insert_value, insert_column_num)
X = np.insert(X, 0, 1, axis=1)


print X

##theta = zeros(2, 1); % initialize fitting parameters
#theta = np.array([0,0])
#theta.shape =(2, 1)
theta = np.array([[0],[0]]) # theta is column vec

print "theta:"
print theta

#% Some gradient descent settings
iterations = 1500
alpha = 0.01

#% compute and display initial cost
#computeCost(X, y, theta)
J = computeCost(X, y, theta)

print "done"







