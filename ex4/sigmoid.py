# -*- coding: utf-8 -*-
# function g = sigmoid(z)
# %SIGMOID Compute sigmoid functoon
# %   J = SIGMOID(z) computes the sigmoid of z.
#
# g = 1.0 ./ (1.0 + exp(-z));
# end
import numpy as np


def sigmoid(z):
    f = lambda h: 1 / (1 + np.exp(-h))
    return f(z)
