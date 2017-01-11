# -*- coding: utf-8 -*-
import numpy as np

def computeCost(X, y, theta):
    predictions = np.dot(X, theta)
    ## cacu sqrt errors
    ##sqerrors = (predictions - y). ^ 2;
    # force convert to col vec
    m = len(y) # 97
    predictions.shape = (m, 1)
    y.shape = (m, 1)
    sqerrors = pow((predictions - y), 2)
    #J = 1 / (2 * m) * sum(sqerrors);
    sqerrors.shape = (m, 1)


    J = float(1) / (2 * m) * np.sum(sqerrors)
    return J


