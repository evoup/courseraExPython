# -*- coding: utf-8 -*-
# function [h, display_array] = displayData(X, example_width)
# %DISPLAYDATA Display 2D data in a nice grid
# %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
# %   stored in X in a nice grid. It returns the figure handle h and the
# %   displayed array if requested.
#
# % Set example_width automatically if not passed in
# if ~exist('example_width', 'var') || isempty(example_width)
# 	example_width = round(sqrt(size(X, 2)));
# end
#
# % Gray Image
# colormap(gray);
#
# % Compute rows, cols
# [m n] = size(X);
# example_height = (n / example_width);
#
# % Compute number of items to display
# display_rows = floor(sqrt(m));
# display_cols = ceil(m / display_rows);
#
# % Between images padding
# pad = 1;
#
# % Setup blank display
# display_array = - ones(pad + display_rows * (example_height + pad), ...
#                        pad + display_cols * (example_width + pad));
#
# % Copy each example into a patch on the display array
# curr_ex = 1;
# for j = 1:display_rows
# 	for i = 1:display_cols
# 		if curr_ex > m,
# 			break;
# 		end
# 		% Copy the patch
#
# 		% Get the max value of the patch
# 		max_val = max(abs(X(curr_ex, :)));
# 		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
# 		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
# 						reshape(X(curr_ex, :), example_height, example_width) / max_val;
# 		curr_ex = curr_ex + 1;
# 	end
# 	if curr_ex > m,
# 		break;
# 	end
# end
#
# % Display Image
# h = imagesc(display_array, [-1 1]);
#
# % Do not show axis
# axis image off
#
# drawnow;
#
# end

import numpy as np
# from matplotlib import use
# use('TkAgg')
import matplotlib.pyplot as plt

from show import show


def displayData(X):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""

    # Compute rows, cols
    if len(X.shape) > 1:  # if is multiply cell graphic
        m, n = X.shape
        example_width = round(np.sqrt(n))
        example_height = (n / example_width)
    else:  # if a only a single graphic

        X.shape = (1, len(X))  # (1, 400)
        m = 1
        # n = X.shape
        example_width = 20
        example_height = 20
    # example_width = round(np.sqrt(n))
    # example_height = (n / example_width)

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            rows = [pad + j * (example_height + pad) + x for x in np.arange(example_height + 1)]
            cols = [pad + i * (example_width + pad) + x for x in np.arange(example_width + 1)]
            reshapeMat = X[curr_ex, :].reshape(example_height, example_width)
            display_array[min(rows):max(rows), min(cols):max(cols)] = reshapeMat / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

            # Display Image
    display_array = display_array.astype('float32')
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
    # Do not show axis
    plt.axis('off')
    plt.show()
