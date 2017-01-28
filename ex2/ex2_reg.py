# -*- coding: utf-8 -*-
# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the second part
# %  of the exercise which covers regularization with logistic regression.
# %
# %  You will need to complete the following functions in this exericse:
# %
# %     sigmoid.m
# %     costFunction.m
# %     predict.m
# %     costFunctionReg.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#
# %% Load Data
# %  The first two columns contains the X values and the third column
# %  contains the label (y).
#
# data = load('ex2data2.txt');
# X = data(:, [1, 2]); y = data(:, 3);
#
# plotData(X, y);
#
# % Put some labels
# hold on;
#
# % Labels and Legend
# xlabel('Microchip Test 1')
# ylabel('Microchip Test 2')
#
# % Specified in plot order
# legend('y = 1', 'y = 0')
# hold off;
#
#
# %% =========== Part 1: Regularized Logistic Regression ============
# %  In this part, you are given a dataset with data points that are not
# %  linearly separable. However, you would still like to use logistic
# %  regression to classify the data points.
# %
# %  To do so, you introduce more features to use -- in particular, you add
# %  polynomial features to our data matrix (similar to polynomial
# %  regression).
# %
#
# % Add Polynomial Features
#
# % Note that mapFeature also adds a column of ones for us, so the intercept
# % term is handled
# X = mapFeature(X(:,1), X(:,2));
#
# % Initialize fitting parameters
# initial_theta = zeros(size(X, 2), 1);
#
# % Set regularization parameter lambda to 1
# lambda = 1;
#
# % Compute and display initial cost and gradient for regularized logistic
# % regression
# [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
#
# fprintf('Cost at initial theta (zeros): %f\n', cost);
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ============= Part 2: Regularization and Accuracies =============
# %  Optional Exercise:
# %  In this part, you will get to try different values of lambda and
# %  see how regularization affects the decision coundart
# %
# %  Try the following values of lambda (0, 1, 10, 100).
# %
# %  How does the decision boundary change when you vary lambda? How does
# %  the training set accuracy vary?
# %
#
# % Initialize fitting parameters
# initial_theta = zeros(size(X, 2), 1);
#
# % Set regularization parameter lambda to 1 (you should vary this)
# lambda = 1;
#
# % Set Options
# options = optimset('GradObj', 'on', 'MaxIter', 400);
#
# % Optimize
# [theta, J, exit_flag] = ...
# 	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
#
# % Plot Boundary
# plotDecisionBoundary(theta, X, y);
# hold on;
# title(sprintf('lambda = %g', lambda))
#
# % Labels and Legend
# xlabel('Microchip Test 1')
# ylabel('Microchip Test 2')
#
# legend('y = 1', 'y = 0', 'Decision boundary')
# hold off;
#
# % Compute accuracy on our training set
# p = predict(theta, X);
#
# fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
#
#
import numpy as np
import os
from scipy.optimize import fmin_bfgs

from costFunctionReg import costFunctionReg, costFunctionReg2
from plotDecisionBoundary import plotDecisionBoundary
from mapFeature import mapFeature
from plotData import plotData
from predict import predict
import matplotlib.pyplot as plt

arr = np.loadtxt(os.getcwd() + '/ex2data2.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)
X = arr.T[:, [0, 1]]  # get first and second col
y = arr.T[:, 2]  # third col
plotData(X, y, True, plt, 'Microchip Test 1', 'Microchip Test 2', ['y = 1', 'y = 0'])
X = mapFeature(X[:, 0], X[:, 1])
_, cols = X.shape
initial_theta = np.zeros((cols, 1))
lambda_param = 1

cost, grad = costFunctionReg(X, y, initial_theta, lambda_param)
print('Cost at initial theta (zeros): %f\n', cost)
print('Gradient at initial theta (zeros): \n')
for i in range(len(grad)):
    print ("%f " % (grad[i])),
print "\n"

# def decorated_cost(theta):
#     return costFunctionReg2(theta, X, y, lambda_param)
# fmin_bfgs param 1 can alse specify with decorated_cost like defined above
result_theta = fmin_bfgs(lambda t: costFunctionReg2(t, X, y, lambda_param), initial_theta, maxiter=400)

plotDecisionBoundary(result_theta, X, y)
p = predict(result_theta, X)
print('Train Accuracy: %f\n', np.mean(p == y) * 100)
print "done"
