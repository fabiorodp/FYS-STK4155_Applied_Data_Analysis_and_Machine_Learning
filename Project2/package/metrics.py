# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np


def r2(y_real, y_predict):
    """The r2 score function."""
    return 1.0 - np.sum(
        (y_real - y_predict) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)


def mse(y_true, y_hat):
    """The MSE function."""
    n = np.size(y_hat)
    return np.sum((y_hat - y_true) ** 2.0) / n
