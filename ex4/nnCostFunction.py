# -*- coding: utf-8 -*-
# function [J grad] = nnCostFunction(nn_params, ...
#                                    input_layer_size, ...
#                                    hidden_layer_size, ...
#                                    num_labels, ...
#                                    X, y, lambda)
# %NNCOSTFUNCTION Implements the neural network cost function for a two layer
# %neural network which performs classification
# %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
# %   X, y, lambda) computes the cost and gradient of the neural network. The
# %   parameters for the neural network are "unrolled" into the vector
# %   nn_params and need to be converted back into the weight matrices.
# %
# %   The returned parameter grad should be a "unrolled" vector of the
# %   partial derivatives of the neural network.
# %
#
# % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# % for our 2 layer neural network
# Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                  hidden_layer_size, (input_layer_size + 1));
#
# Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                  num_labels, (hidden_layer_size + 1));
#
# % Setup some useful variables
# m = size(X, 1);
#
# % You need to return the following variables correctly
# J = 0;
# Theta1_grad = zeros(size(Theta1));
# Theta2_grad = zeros(size(Theta2));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: You should complete the code by working through the
# %               following parts.
# %
# % Part 1: Feedforward the neural network and return the cost in the
# %         variable J. After implementing Part 1, you can verify that your
# %         cost function computation is correct by verifying the cost
# %         computed in ex4.m
# %
# % Part 2: Implement the backpropagation algorithm to compute the gradients
# %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
# %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# %         Theta2_grad, respectively. After implementing Part 2, you can check
# %         that your implementation is correct by running checkNNGradients
# %
# %         Note: The vector y passed into the function is a vector of labels
# %               containing values from 1..K. You need to map this vector into a
# %               binary vector of 1's and 0's to be used with the neural network
# %               cost function.
# %
# %         Hint: We recommend implementing backpropagation using a for-loop
# %               over the training examples if you are implementing it for the
# %               first time.
# %
# % Part 3: Implement regularization with the cost function and gradients.
# %
# %         Hint: You can implement this around the code for
# %               backpropagation. That is, you can compute the gradients for
# %               the regularization separately and then add them to Theta1_grad
# %               and Theta2_grad from Part 2.
# %
#
#
# X = [ones(m,1) X];
#
#
# % foward propagation
# % a1 = X;
# a2 = sigmoid(Theta1 * X');
# a2 = [ones(m,1) a2'];
#
# h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3
#
# % y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
# yk = zeros(num_labels, m);
# for i=1:m,
#   yk(y(i),i)=1;
# end
#
# % follow the form
# J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));
#
#
#
# % Note that you should not be regularizing the terms that correspond to the bias.
# % For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
# t1 = Theta1(:,2:size(Theta1,2));
# t2 = Theta2(:,2:size(Theta2,2));
#
# % regularization formula
# Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);
#
# % cost function + reg
# J = J + Reg;
#
#
# % -------------------------------------------------------------
#
# % Backprop
#
# for t=1:m,
#
# 	% dummie pass-by-pass
# 	% forward propag
#
# 	a1 = X(t,:); % X already have bias
# 	z2 = Theta1 * a1';
#
# 	a2 = sigmoid(z2);
# 	a2 = [1 ; a2]; % add bias
#
# 	z3 = Theta2 * a2;
#
# 	a3 = sigmoid(z3); % final activation layer a3 == h(theta)
#
#
# 	% back propag (god bless me)
#
# 	z2=[1; z2]; % bias
#
# 	delta_3 = a3 - yk(:,t); % y(k) trick - getting columns of t element
# 	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);
#
# 	% skipping sigma2(0)
# 	delta_2 = delta_2(2:end);
#
# 	Theta2_grad = Theta2_grad + delta_3 * a2';
# 	Theta1_grad = Theta1_grad + delta_2 * a1; % I don't know why a1 doesn't need to be transpost (brute force try)
#
# end;
#
# % Theta1_grad = Theta1_grad ./ m;
# % Theta2_grad = Theta2_grad ./ m;
#
#
# % Regularization (here you go)
#
#
# 	Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
#
# 	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
#
#
# 	Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
#
# 	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));
#
#
#
# % -------------------------------------------------------------
#
# % =========================================================================
#
# % Unroll gradients
# grad = [Theta1_grad(:) ; Theta2_grad(:)];
#
#
# end
import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_param):
    #global grad
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    m, _ = X.shape
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    X = np.insert(X, 0, 1, axis=1)
    # foward propagation
    a2 = sigmoid(np.dot(Theta1, X.T))
    a2 = np.insert(a2.T, 0, 1, axis=1)
    h_theta = sigmoid(np.dot(Theta2, a2.T))  # h_theta equals z3
    # y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
    yk = np.zeros((num_labels, m))
    for i in range(1, m + 1):
            yk[y[i - 1] - 1, i - 1] = 1
    # follow the form
    J = (float(1)/m) * np.sum(np.sum(np.multiply((-yk), np.log(h_theta)) - np.multiply((1 - yk), np.log(1 - h_theta))))
    # Note that you should not be regularizing the terms that correspond to the bias.
    # For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
    _, col = Theta1.shape
    t1 = Theta1[:, 1:col]
    t2 = Theta2[:, 1:col]
    # regularization formula
    Reg = lambda_param * (np.sum(np.sum(np.power(t1, 2))) + np.sum(np.sum(np.power(t2, 2)))) / (float(2) * m)
    # cost function + reg
    J += Reg
    # -------------------------------------------------------------
    # Backprop
    for t in range(1, m + 1):
        a1 = X[t - 1, :].reshape(1, col)  # X already have bias, shape is (1, 401)
        z2 = np.dot(Theta1, a1.T)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis=0)  # add bias
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)  # final activation layer a3 == h(theta)
        #  back propag (god bless me, wtf?)
        z2 = np.insert(z2, 0, 1, axis=0)  # bias
        delta_3 = a3 - yk[:, t - 1].reshape(num_labels, 1)  # y(k) trick - getting columns of t element
        delta_2 = np.multiply(np.dot(Theta2.T, delta_3), sigmoidGradient(z2))
        # skipping sigma2(0)
        delta_2 = delta_2[1:]
        Theta2_grad = Theta2_grad + np.dot(delta_3, a2.T)
        Theta1_grad = Theta1_grad + np.dot(delta_2, a1)  # I don't know why a1 doesn't need to be transpost (brute force try)

        # Regularization (here you go)
        Theta1_grad[:, 0] = np.divide(Theta1_grad[:, 0], m)
        Theta1_grad[:, 1:] = np.divide(Theta1_grad[:, 1:], m) + (np.dot(lambda_param / m, Theta1[:, 1:]))
        Theta2_grad[:, 0] = np.divide(Theta2_grad[:, 0], m)
        Theta2_grad[:, 1:] = np.divide(Theta2_grad[:, 1:], m) + (np.dot(lambda_param / m, Theta2[:, 1:]))
        # Unroll gradients
        grad = np.array(Theta1_grad.ravel().tolist() + Theta2_grad.ravel().tolist())
    return J, grad