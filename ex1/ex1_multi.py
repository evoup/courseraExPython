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
from pandas.compat import scipy

print('Loading data ...\n')

# %% Load Data
# data = load('ex1data2.txt');
# X = data(:, 1:2);
# y = data(:, 3);
# m = length(y);
data = np.loadtxt(os.getcwd() + '/ex1data2.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)
data = data.transpose()
X = data[:, [0, 1]]  # get first and second col
y = data[:, 2]  # third col

print "done"



