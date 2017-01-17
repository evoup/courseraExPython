# -*- coding: utf-8 -*-
import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    # %GRADIENTDESCENT Performs gradient descent to learn theta
    # %   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    # %   taking num_iters gradient steps with learning rate alpha

    # % Initialize some useful values
    # m = length(y); % number of training examples
    m = len(y)  # % number of training examples
    J_history = np.zeros((num_iters, 1))

    for num_iter in range(0, num_iters):
        # % Instructions: Perform a single gradient step on the parameter vector theta.
        # % Hint: While debugging, it can be useful to print out the values
        # of the cost function (computeCost) and gradient here.
        predictions = np.dot(X, theta)

        # ignore derivation, update is a partial derivative
        #updates = X' * (predictions - y);
        updates = np.dot(np.transpose(X), (predictions-y))

        theta = theta - alpha * (float(1) / m) * updates

    return theta
