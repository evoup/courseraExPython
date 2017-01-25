# -*- coding: utf-8 -*-
# function p = predict(theta, X)
# %PREDICT Predict whether the label is 0 or 1 using learned logistic
# %regression parameters theta
# %   p = PREDICT(theta, X) computes the predictions for X using a
# %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
#
# m = size(X, 1); % Number of training examples
#
# % You need to return the following variables correctly
# p = zeros(m, 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the following code to make predictions using
# %               your learned logistic regression parameters.
# %               You should set p to a vector of 0's and 1's
# %
#
#
# f=@(n) round(n);
#
# s = sigmoid(X * theta);
#
# p = f(s);
#
#
#
#
#
# % =========================================================================
#
#
# end
import numpy as np

from sigmoid import sigmoid


def predict(theta, X):
    m, _ = X.shape
    p = np.zeros((m, 1))
    f = lambda n: np.round(n)
    s = sigmoid(np.dot(X, theta))
    p = f(s)
    p.shape = (m, 1)  # must convert to matrix, otherwise can not use == for compare 2 matrixes
    return p
