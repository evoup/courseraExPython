# -*- coding: utf-8 -*-

# function [theta] = normalEqn(X, y)
# %NORMALEQN Computes the closed-form solution to linear regression
# %   NORMALEQN(X,y) computes the closed-form solution to linear
# %   regression using the normal equations.
#
# theta = zeros(size(X, 2), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the code to compute the closed form solution
# %               to linear regression and put the result in theta.
# %
#
# % ---------------------- Sample Solution ----------------------
#
#
# theta = pinv(X' * X) * X' * y
#
# % -------------------------------------------------------------
#
#
# % ============================================================
#
# end



import numpy as np


def normalEqn(X, y):
    row, col = X.shape
    theta = np.zeros((col, 1))
    XT = X.transpose()
    theta = np.linalg.pinv(np.dot(XT, X))
    theta = np.dot(theta, X.transpose())
    theta = np.dot(theta, y)
    return theta