# -*- coding: utf-8 -*-
# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the logistic
# %  regression exercise. You will need to complete the following functions
# %  in this exericse:
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
# %  The first two columns contains the exam scores and the third column
# %  contains the label.
#
# data = load('ex2data1.txt');
# X = data(:, [1, 2]); y = data(:, 3);
import numpy as np
import os
from scipy.optimize import fmin_bfgs, fmin

from costFunction import costFunction, costFunction2
from plotDecisionBoundary import plotDecisionBoundary
from plotData import plotData

arr = np.loadtxt(os.getcwd() + '/ex2data1.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)
data = arr.transpose()
X = data[:, [0, 1]]  # get first and second col
y = data[:, 2]  # third col

# %% ==================== Part 1: Plotting ====================
# %  We start the exercise by first plotting the data to understand the
# %  the problem we are working with.
#
print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'])
#
# plotData(X, y);
#
plotData(X, y, True)

# % Put some labels
# hold on;
# % Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
#
# % Specified in plot order
# legend('Admitted', 'Not admitted')
# hold off;
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ============ Part 2: Compute Cost and Gradient ============
# %  In this part of the exercise, you will implement the cost and gradient
# %  for logistic regression. You neeed to complete the code in
# %  costFunction.m
#
# %  Setup the data matrix appropriately, and add ones for the intercept term
# [m, n] = size(X);
m, n = X.shape

#
# % Add intercept term to x and X_test
# X = [ones(m, 1) X];
X = np.insert(X, 0, 1, axis=1)
#
# % Initialize fitting parameters
# initial_theta = zeros(n + 1, 1);
initial_theta = np.zeros((n + 1, 1))
#
# % Compute and display initial cost and gradient
# [cost, grad] = costFunction(initial_theta, X, y);
cost, grad = costFunction(initial_theta, X, y)
#
print('Cost at initial theta (zeros): %0.2f\n', cost)
print('Gradient at initial theta (zeros): \n')
# fprintf(' %f \n', grad);
print ("%.2f %.2f %.2f" % (grad[0][0], grad[0][1], grad[0][2]))
#

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ============= Part 3: Optimizing using fminunc  =============
# %  In this exercise, you will use a built-in function (fminunc) to find the
# %  optimal parameters theta.
#
# %  Set options for fminunc
# options = optimset('GradObj', 'on', 'MaxIter', 400);
#
# %  Run fminunc to obtain the optimal theta
# %  This function will return theta and the cost
# [theta, cost] = ...
# 	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
# f_, fprime = func_wrapper(costFunction2(initial_theta, X, y))
#
# print fmin_bfgs(f_, initial_theta, maxiter=400)

options = {'full_output': True, 'maxiter': 400}
theta, cost, _, _, _ = fmin(lambda t: costFunction2(X, y, t), initial_theta, **options)
print("Cost at theta found by fmin: %.6f" % cost)
print("theta: %.6f %.6f %.6f" % (theta[0], theta[1], theta[2]))
#
# % Print theta to screen
# fprintf('Cost at theta found by fminunc: %f\n', cost);
# fprintf('theta: \n');
# fprintf(' %f \n', theta);
#
# % Plot Boundary
# plotDecisionBoundary(theta, X, y);
#
# % Put some labels
# hold on;
# % Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
#
# % Specified in plot order
# legend('Admitted', 'Not admitted')
# hold off;

plotDecisionBoundary(theta, X, y)

#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of
# %  our model.
# %
# %  Your task is to complete the code in predict.m
#
# %  Predict probability for a student with score 45 on exam 1
# %  and score 85 on exam 2
#
# prob = sigmoid([1 45 85] * theta);
# fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
#          'probability of %f\n\n'], prob);
#
# % Compute accuracy on our training set
# p = predict(theta, X);
#
# fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#

print "done"
