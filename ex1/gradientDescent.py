# -*- coding: utf-8 -*-

def gradientDescent(X, y, theta, alpha, iterations):
    # %GRADIENTDESCENT Performs gradient descent to learn theta
    # %   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    # %   taking num_iters gradient steps with learning rate alpha

    # % Initialize some useful values
    # m = length(y); % number of training examples
    m = len(y) #% number of training examples

    print ""

