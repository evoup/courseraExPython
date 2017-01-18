#function [X_norm, mu, sigma] = featureNormalize(X)
#%FEATURENORMALIZE Normalizes the features in X 
#%   FEATURENORMALIZE(X) returns a normalized version of X where
#%   the mean value of each feature is 0 and the standard deviation
#%   is 1. This is often a good preprocessing step to do when
#%   working with learning algorithms.

#% You need to set these values correctly
#X_norm = X;
#mu = zeros(1, size(X, 2));
#sigma = zeros(1, size(X, 2));

#% ====================== YOUR CODE HERE ======================
#% Instructions: First, for each feature dimension, compute the mean
#%               of the feature and subtract it from the dataset,
#%               storing the mean value in mu. Next, compute the 
#%               standard deviation of each feature and divide
#%               each feature by it's standard deviation, storing
#%               the standard deviation in sigma. 
#%
#%               Note that X is a matrix where each column is a 
#%               feature and each row is an example. You need 
#%               to perform the normalization separately for 
#%               each feature. 
#%
#% Hint: You might find the 'mean' and 'std' functions useful.
#%       

#mu = mean(X)
#sigma = std(X)

#mu_matrix = repmat(mu, size(X, 1), 1);
#sigma_matrix = repmat(sigma, size(X, 1), 1);
#X_norm = (X - mu_matrix) ./ sigma_matrix;




#% ============================================================

#end
import numpy as np


def featureNormalize(X):
    X_norm = X
    row, col = X.shape
    mu = np.zeros((1, col))
    sigma = np.zeros((1, col))
    mu = np.mean(X, 0)
    sigma = np.std(X, 0, ddof=1)

    mu_matrix = np.tile(mu, (row, 1))
    sigma_matrix = np.tile(sigma, (row, 1))

    X_norm = np.divide((X - mu_matrix), sigma_matrix)
    return X_norm, mu , sigma
    
