# -*- coding: utf-8 -*-
from warmUpExercise import warmUpExercise
import numpy as np
import scipy.io
import os
# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
#
#% Initialization
# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

warmUpExercise()

#raw_input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
arr = np.loadtxt(os.getcwd() + '/ex1data1.txt', delimiter=',', usecols=(0, 1), unpack=True)

#feature x vector
X = arr[0]

#result y vector
y = arr[1]

#sample number
m = len(y)

print('number of samples:%s' % m)



print('test')






