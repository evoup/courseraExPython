# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from warmUpExercise import warmUpExercise
from computeCost import computeCost
from gradientDescent import gradientDescent

# import scipy.io
import os

## Machine Learning Online Class - Exercise 1: Linear Regression

##  Instructions
##  ------------
##
##  This file contains code that helps you get started on the
##  linear exercise. You will need to complete the following functions
##  in this exericse:
##
##     warmUpExercise.m
##     plotData.m
##     gradientDescent.m
##     computeCost.m
##     gradientDescentMulti.m
##     computeCostMulti.m
##     featureNormalize.m
##     normalEqn.m
##
##  For this exercise, you will not need to change any code in this file,
##  or any other files other than those mentioned above.
##
## x refers to the population size in 10,000s
## y refers to the profit in $10,000s
##
##
## % Initialization
## ==================== Part 1: Basic Function ====================
## Complete warmUpExercise.m
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

warmUpExercise()

# raw_input('Program paused. Press enter to continue.\n')


## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
arr = np.loadtxt(os.getcwd() + '/ex1data1.txt', delimiter=',', usecols=(0, 1), unpack=True)

# feature x vector
X = arr[0]

# result y vector
y = arr[1]

# sample number totoal is :97
m = len(y)

print('number of samples:%s' % m)

# X = [6.1101, 5.5277,...]
# y = [17.592, 9.1302,...]

matplotlib.pyplot.scatter(X, y)

#matplotlib.pyplot.show()

# raw_input('Program paused. Press enter to continue.\n');
##%% =================== Part 3: Gradient descent ===================
# convert to column vec
X.shape = (97, 1)

##X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

# insert one col before it
# np.insert param(original_matrix, offset, insert_value, insert_column_num)
X = np.insert(X, 0, 1, axis=1)

print X

##theta = zeros(2, 1); % initialize fitting parameters
# theta = np.array([0,0])
# theta.shape =(2, 1)
# or
# theta = np.array([[0], [0]])  # theta is column vec
theta = np.zeros((2, 1))

print "theta:"
print theta

# % Some gradient descent settings
iterations = 1500
alpha = 0.01

# % compute and display initial cost
# computeCost(X, y, theta)
J = computeCost(X, y, theta)

# % run gradient descent
# theta = gradientDescent(X, y, theta, alpha, iterations);
newTheta = gradientDescent(X, y, theta, alpha, iterations)
#% print theta to screen
#fprintf('Theta found by gradient descent: ');
#fprintf('%f %f \n', theta(1), theta(2));
print('Theta found by gradient descent: ')
#print ("%.2f" % newTheta[0][0])
print ("%.2f %.2f" % (newTheta[0][0], newTheta[1][0]))


#% Plot the linear fit
#hold on; % keep previous plot visible
#plot(X(:,2), X*theta, '-')
#legend('Training data', 'Linear regression')
#hold off % don't overlay any more plots on this figure


#matplotlib.pyplot.hold(True)
X = arr[0]

X.shape = (97, 1)
X1 = np.insert(X, 0, 1, axis=1)

matplotlib.pyplot.plot(X, np.dot(X1, newTheta), color='blue')

#matplotlib.pyplot.ylim(-10, 30)
#matplotlib.pyplot.ylim(4, 30)
#matplotlib.pyplot.hold(False)

#% Predict values for population sizes of 35,000 and 70,000
#predict1 = [1, 3.5] *theta;
#fprintf('For population = 35,000, we predict a profit of %f\n',...
#    predict1*10000);
#predict2 = [1, 7] * theta;
#fprintf('For population = 70,000, we predict a profit of %f\n',...
#    predict2*10000);

predict1 = np.dot([1, 3.5], newTheta)
profit = float(predict1)*10000
print('For population = 35,000, we predict a profit of %0.6f\n' % profit)

predict1 = np.dot([1, 7], newTheta)
profit = float(predict1)*10000
print('For population = 70,000, we predict a profit of %0.6f\n' % profit)


#%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

#% Grid over which we will calculate J
#theta0_vals = linspace(-10, 10, 100);
#theta1_vals = linspace(-1, 4, 100);
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
#% Fill out J_vals

#for i = 1:length(theta0_vals)
#    for j = 1:length(theta1_vals)
#	  t = [theta0_vals(i); theta1_vals(j)];
#	  J_vals(i,j) = computeCost(X, y, t);
#    end
#end

for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        t.shape = (2, 1)
        J_vals[i][j] = computeCost(X1, y, t)


#% Because of the way meshgrids work in the surf command, we need to
#% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)

#% Surface plot
#figure;
#surf(theta0_vals, theta1_vals, J_vals)
#xlabel('\theta_0'); ylabel('\theta_1');



fig = plt.figure()
ax = fig.gca(projection='3d')
X = theta0_vals
Y = theta1_vals
X, Y = np.meshgrid(X, Y)

Z = J_vals
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J_val')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf, shrink=0.5, aspect=5)



np.logspace(-2, 3, 20)



matplotlib.pyplot.show()




print "done"
