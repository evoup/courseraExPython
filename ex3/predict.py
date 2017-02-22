# -*- coding: utf-8 -*-
# function p = predict(Theta1, Theta2, X)
# %PREDICT Predict the label of an input given a trained neural network
# %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
# %   trained weights of a neural network (Theta1, Theta2)
#
# % Useful values
# m = size(X, 1);
# num_labels = size(Theta2, 1);
#
# % You need to return the following variables correctly
# p = zeros(size(X, 1), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the following code to make predictions using
# %               your learned neural network. You should set p to a
# %               vector containing labels between 1 to num_labels.
# %
# % Hint: The max function might come in useful. In particular, the max
# %       function can also return the index of the max element, for more
# %       information see 'help max'. If your examples are in rows, then, you
# %       can use max(A, [], 2) to obtain the max for each row.
# %
#
#
# X = [ones(m, 1) X];
#
# for j=1:m,
#
# 	%first layer propagation
# 	z2 = sigmoid(X(j,:) * Theta1');
#
# 	%bias to hidden layer
# 	z2 = [1 z2];
#
# 	%hidden layer propagation and getting max (candidate)
# 	[trash,p(j)] = max(sigmoid(z2 * Theta2'));
# end;
#
#
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


def predict(Theta1, Theta2, X):
    m, _ = X.shape
    num_labels = len(Theta2)
    p = np.zeros((m, 1))
    X = np.insert(X, 0, 1, axis=1)
    for j in range(1, m):
        z2 = sigmoid(np.dot(X[j - 1, :], Theta1.T))
        z2.shape = (len(z2), 1)
        z2 = np.insert(z2, 0, 1, axis=0)
        pred = sigmoid(np.dot(z2.T, Theta2.T))
        res = pred.argmax(axis=1) + 1  # because py max index start from 0 so, 0 will be 1,9 will be 10
        p[j - 1] = res
    return p
    # for j=1:m,
    #
    # 	%first layer propagation
    # 	z2 = sigmoid(X(j,:) * Theta1');
    #
    # 	%bias to hidden layer
    # 	z2 = [1 z2];
    #
    # 	%hidden layer propagation and getting max (candidate)
    # 	[trash,p(j)] = max(sigmoid(z2 * Theta2'));
    # end
