# -*- coding: utf-8 -*-
# function g = sigmoid(z)
# %SIGMOID Compute sigmoid functoon
# %   J = SIGMOID(z) computes the sigmoid of z.
#
# % You need to return the following variables correctly
# g = zeros(size(z));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
# %               vector or scalar).
#
# f=@(h) 1 ./ (1 + exp(-h)); % 1 / (1 + exp(-h));
#
#
# g = f(z);
#
#
#
# % =============================================================
#
# end
import numpy as np


def sigmoid(z):
# f = @(h) 1 ./ (1 + exp(-h)); % 1 / (1 + exp(-h));
# --------------------------------------------------------------
# For a simpler example:
#
# F=@(x) x(x>5);
# F is the function handle, and x(x>5) is the code that it executes if you enter
#
# y=F(1:10)
# you will see that it returns the numbers greater than 5.
# Python
# 's lambda functions are somewhat similar:
#
# In[1]: fn = lambda x: x ** 2 + 3 * x - 4
#
# In[2]: fn(3)
# Out[2]: 14
# --------------------------------------------------------------
    f = lambda h: 1/(1 + np.exp(-h))
    return f(z)

