# -*- coding: utf-8 -*-
# %% Machine Learning Online Class
# %  Exercise 1: Linear regression with multiple variables
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear regression exercise.
# %
# %  You will need to complete the following functions in this
# %  exericse:
# %
# %     warmUpExercise.m
# %     plotData.m
# %     gradientDescent.m
# %     computeCost.m
# %     gradientDescentMulti.m
# %     computeCostMulti.m
# %     featureNormalize.m
# %     normalEqn.m
# %
# %  For this part of the exercise, you will need to change some
# %  parts of the code below for various experiments (e.g., changing
# %  learning rates).
# %
#
# %% Initialization
#
# %% ================ Part 1: Feature Normalization ================
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.compat import scipy

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti

print('Loading data ...\n')

# %% Load Data
# data = load('ex1data2.txt');
# X = data(:, 1:2);
# y = data(:, 3);
# m = length(y);
data = np.loadtxt(os.getcwd() + '/ex1data2.txt', delimiter=',', usecols=(0, 1, 2), unpack=True, dtype=float)
data = data.transpose()
X = data[:, [0, 1]]  # get first and second col
y = data[:, 2]  # third col
m = len(y)
#% Print out some data points
print('First 10 examples from the dataset: \n')
#print(' x = [%.2f %.2f], y = %.2f \n' %([X[1:10, :], y[1:10, :]]))

#X[0:47] [0:10]
print(' X = %s \n' % X[0:47] [0:10])
#y[0:m][0:10]
print(' y = %s \n' % y[0:m][0:10])

#% Scale features and set them to zero mean
print('Normalizing Features ...\n')
#[X mu sigma] = featureNormalize(X);

X_norm, mu, sigma = featureNormalize(X)
# % Add intercept term to X
# X = [ones(m, 1) X];
X = np.insert(X_norm, 0, 1, axis=1)

# %% ================ Part 2: Gradient Descent ================
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: We have provided you with the following starter
# %               code that runs gradient descent with a particular
# %               learning rate (alpha).
# %
# %               Your task is to first make sure that your functions -
# %               computeCost and gradientDescent already work with
# %               this starter code and support multiple variables.
# %
# %               After that, try running gradient descent with
# %               different values of alpha and see which one gives
# %               you the best result.
# %
# %               Finally, you should complete the code at the end
# %               to predict the price of a 1650 sq-ft, 3 br house.
# %
# % Hint: By using the 'hold on' command, you can plot multiple
# %       graphs on the same figure.
# %
# % Hint: At prediction, make sure you do the same feature normalization.
# %

print('Running gradient descent ...\n')

# % Choose some alpha value
alpha = 0.01
num_iters = 400

# % Init Theta and Run Gradient Descent
# theta = zeros(3, 1);
# [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.plot(np.linspace(0, 400, 400), J_history)
plt.show()


print "done"



