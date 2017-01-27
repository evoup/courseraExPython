# -*- coding: utf-8 -*-
# function out = mapFeature(X1, X2)
# % MAPFEATURE Feature mapping function to polynomial features
# %
# %   MAPFEATURE(X1, X2) maps the two input features
# %   to quadratic features used in the regularization exercise.
# %
# %   Returns a new feature array with more features, comprising of
# %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
# %
# %   Inputs X1, X2 must be the same size
# %
#
# degree = 6;
# out = ones(size(X1(:,1)));
# for i = 1:degree
#     for j = 0:i
#         out(:, end+1) = (X1.^(i-j)).*(X2.^j);
#     end
# end
#
# end
import numpy as np


def mapFeature(X1, X2):
    degree = 6
    rows = X1.shape
    out = np.ones((rows))
    rows_of_out = out.shape[0]
    out.shape = (rows_of_out, 1)
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            insert_col = (np.power(X1, (i - j))) * (np.power(X2, j))
            # if is not matrix, how to insert col at last? if is, ask this question again
            cols = out.shape[1]
            out = np.insert(out, cols, insert_col, axis=1)
    return out
