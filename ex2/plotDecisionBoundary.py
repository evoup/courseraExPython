# -*- coding: utf-8 -*-
# function plotDecisionBoundary(theta, X, y)
# %PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
# %the decision boundary defined by theta
# %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
# %   positive examples and o for the negative examples. X is assumed to be
# %   a either
# %   1) Mx3 matrix, where the first column is an all-ones column for the
# %      intercept.
# %   2) MxN, N>3 matrix, where the first column is all-ones
#
# % Plot Data
# plotData(X(:,2:3), y);
# hold on
#
# if size(X, 2) <= 3
#     % Only need 2 points to define a line, so choose two endpoints
#     plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
#
#     % Calculate the decision boundary line
#     plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
#
#     % Plot, and adjust axes for better viewing
#     plot(plot_x, plot_y)
#
#     % Legend, specific for the exercise
#     legend('Admitted', 'Not admitted', 'Decision Boundary')
#     axis([30, 100, 30, 100])
# else
#     % Here is the grid range
#     u = linspace(-1, 1.5, 50);
#     v = linspace(-1, 1.5, 50);
#
#     z = zeros(length(u), length(v));
#     % Evaluate z = theta*x over the grid
#     for i = 1:length(u)
#         for j = 1:length(v)
#             z(i,j) = mapFeature(u(i), v(j))*theta;
#         end
#     end
#     z = z'; % important to transpose z before calling contour
#
#     % Plot z = 0
#     % Notice you need to specify the range [0, 0]
#     contour(u, v, z, [0, 0], 'LineWidth', 2)
# end
# hold off
#
# end
import numpy as np

import matplotlib
from matplotlib import cm

from mapFeature import mapFeature
from plotData import plotData
import matplotlib.pyplot as plt


def plotDecisionBoundary(theta, X, y):

    _, col = X.shape
    if col <= 3:
        plotData(X[:, (1, 2)], y, False)
        np.max(X[:, 1])
        plt.hold(True)
        plot_x = [np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]
        plot_y = np.dot((-1 / theta[2]), (np.dot(theta[1],  plot_x) + theta[0]))
        plt.plot(plot_x, plot_y, color='blue')
        plt.show()
    else:
        # TODO else another branch
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(1, len(u)):
            for j in range(1, len(v)):
                theta.shape = (28, 1)
                l = mapFeature(u[i], v[j])
                r = theta
                z[i, j] = l.dot(r)
        z = z.T
        fig = plt.figure()
        ax = fig.gca()
        ax.contour(u, v, z, [0, 0], cmap=cm.coolwarm)
        # ax.plot(X[:, (1, 2)], y)
        plotData(X[:, (1, 2)], y, False, ax, 'Microchip Test 1', 'Microchip Test 2',
                 ['y = 1', 'y = 0', 'Decision boundary'])
        plt.show()
