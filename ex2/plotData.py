# -*- coding: utf-8 -*-
# function plotData(X, y)
# %PLOTDATA Plots the data points X and y into a new figure
# %   PLOTDATA(x,y) plots the data points with + for the positive examples
# %   and o for the negative examples. X is assumed to be a Mx2 matrix.
#
# % Create New Figure
# figure; hold on;
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Plot the positive and negative examples on a
# %               2D plot, using the option 'k+' for the positive
# %               examples and 'ko' for the negative examples.
# %
#
# % Find Indices of Positive and Negative Examples
# pos = find(y==1); neg = find(y == 0);
# % Plot Examples
# plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
# plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
#
#
#
# % =========================================================================
#
#
#
# hold off;
#
# end

# emulate octave/matlab find
# useage:
# a = [1, 2, 3, 1, 2, 3, 1, 2, 3]
#
# inds = indices(a, lambda x: x > 2)
#
# >>> inds
# [2, 5, 8]

import matplotlib.pyplot as plt


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def plotData(X, y, will_show, x_lable=None, y_lable=None, legend=None):
    pos = indices(y, lambda z: z == 1)
    neg = indices(y, lambda z: z == 0)
    plt.scatter(X[pos][:, 0], X[pos][:, 1], marker='+', c="black", linewidths=1)
    plt.scatter(X[neg][:, 0], X[neg][:, 1], marker='o', c="yellow", linewidths=1)
    if x_lable is not None:
        plt.xlabel(x_lable)
    else:
        plt.xlabel('Exam 1 score')
    if y_lable is not None:
        plt.ylabel(y_lable)
    else:
        plt.ylabel('Exam 2 score')
    if legend is not None:
        plt.legend(legend)
    else:
        plt.legend(['Admitted', 'Not admitted'])
    if will_show:
        plt.show()
