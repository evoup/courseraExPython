# function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
# %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
# %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
# %   taking num_iters gradient steps with learning rate alpha
#
# % Initialize some useful values
# m = length(y); % number of training examples
# J_history = zeros(num_iters, 1);
#
# for iter = 1:num_iters
#
#     % ====================== YOUR CODE HERE ======================
#     % Instructions: Perform a single gradient step on the parameter vector
#     %               theta.
#     %
#     % Hint: While debugging, it can be useful to print out the values
#     %       of the cost function (computeCostMulti) and gradient here.
#     %
#
#
#     predictions =  X * theta;
#
#     updates = X' * (predictions - y);
#
#     theta = theta - alpha * (1/m) * updates;
#
#
#
#
#     % ============================================================
#
#     % Save the cost J in every iteration
#     J_history(iter) = computeCostMulti(X, y, theta);
#
# end
#
# end
import numpy as np

from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for num_iter in range(0, num_iters):
        # % Instructions: Perform a single gradient step on the parameter vector theta.
        # % Hint: While debugging, it can be useful to print out the values
        # of the cost function (computeCost) and gradient here.
        predictions = np.dot(X, theta)

        # ignore derivation, update is a partial derivative
        #updates = X' * (predictions - y);
        y.shape = (m, 1)
        updates = np.dot(np.transpose(X), (predictions-y))

        theta = theta - alpha * (float(1) / m) * updates

        J_history[num_iter] = computeCostMulti(X, y, theta)

    return theta, J_history