# -*- coding: utf-8 -*-
# function [J, grad] = costFunctionReg(theta, X, y, lambda)
# %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
# %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
# %   theta as the parameter for regularized logistic regression and the
# %   gradient of the cost w.r.t. to the parameters.
#
# % Initialize some useful values
# m = length(y); % number of training examples
#
# % You need to return the following variables correctly
# J = 0;
# grad = zeros(size(theta));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta.
# %               You should set J to the cost.
# %               Compute the partial derivatives and set grad to the partial
# %               derivatives of the cost w.r.t. each parameter in theta
#
# predictions =  sigmoid(X*theta);
#
# left = -y' * log(predictions);
#
# right = (1 - y') * log(1 - predictions);
#
# thetaZero = theta;
#
# thetaZero(1) = 0;
#
# lambaCostPart = (lambda / (2 * m)) * sum(thetaZero .^ 2);
#
# lambdaGradPart = lambda / m * thetaZero;
#
# J = (1 / m) * (left - right) + lambaCostPart;
#
# grad = ((1/m) * (X' * (predictions - y))) + lambdaGradPart;
#
#
#
#
# % =============================================================
#
# end
import numpy as np

from sigmoid import sigmoid


def costFunctionReg(X, y, theta, lambda_param):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    predictions = sigmoid(np.dot(X, theta))
    left = -np.dot(y.T, np.log(predictions))
    right = np.dot((1 - y.T), np.log(1 - predictions))
    thetaZero = theta.copy()
    thetaZero[0] = 0  # should not regularize the parameter θ 0
    lambdaCostPart = (float(lambda_param) / (2 * m)) * np.sum(np.power(thetaZero, 2))
    lambdaGradPart = float(lambda_param) / m * thetaZero
    J = (float(1) / m) * (left - right) + lambdaCostPart
    y.shape = (len(y), 1)  # convert to matrix
    predictions.shape = (len(predictions), 1)
    lambdaGradPart.shape = (len(lambdaGradPart), 1)
    grad = (float(1) / m) * np.dot(X.T, predictions - y) + lambdaGradPart
    return J, grad


#  samed as costFunctionReg, just param order and retrun value differences
def costFunctionReg2(theta, X, y, lambda_param):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    predictions = sigmoid(np.dot(X, theta))
    left = -np.dot(y.T, np.log(predictions))
    right = np.dot((1 - y.T), np.log(1 - predictions))
    thetaZero = theta.copy()
    thetaZero[0] = 0  # should not regularize the parameter θ 0
    lambdaCostPart = (float(lambda_param) / (2 * m)) * np.sum(np.power(thetaZero, 2))
    lambdaGradPart = float(lambda_param) / m * thetaZero
    J = (float(1) / m) * (left - right) + lambdaCostPart
    y.shape = (len(y), 1)  # convert to matrix
    predictions.shape = (len(predictions), 1)
    lambdaGradPart.shape = (len(lambdaGradPart), 1)
    grad = (float(1) / m) * np.dot(X.T, predictions - y) + lambdaGradPart
    return J

    # def gradientReg(theta, X, y, lambda_param):
    #     m = len(y)
    #     predictions = sigmoid(np.dot(X, theta))
    #     thetaZero = theta.copy()
    #     thetaZero[0] = 0  # should not regularize the parameter θ 0
    #     lambdaCostPart = (float(lambda_param) / (2 * m)) * np.sum(np.power(thetaZero, 2))
    #     lambdaGradPart = float(lambda_param) / m * thetaZero
    #     y.shape = (len(y), 1)  # convert to matrix
    #     predictions.shape = (len(predictions), 1)
    #     lambdaGradPart.shape = (len(lambdaGradPart), 1)
    #     grad = (float(1) / m) * np.dot(X.T, predictions - y) + lambdaGradPart
    #     return grad
