# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
cost_functions.py
~~~~~~~~~~

A module to implement the cost function algorithms.
"""

import numpy as np


def r2(y_real, y_predict):
    """The r2 score function."""
    return 1 - np.sum(
        (y_real - y_predict) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)


def mse(y_true, y_hat):
    """The MSE function."""
    n = np.size(y_hat)
    return np.sum((y_hat - y_true) ** 2.) / n


def mse_prime(y_true, y_hat):
    """Derivative of the MSE function."""
    n = np.size(y_hat)
    return 2. / n * (y_hat - y_true)


def accuracy_score(y_real, y_pred):
    return np.sum(y_real == y_pred) / len(y_real)


def crossentropy(x, y):
    return - (y * np.log(x) + (1 - y) * np.log(1 - x)).mean()


def crossentropy_prime(x, y):
    return -y / x + (1 - y) / (1 - x)


def bias(y_real, y_pred):
    return np.mean((y_real - np.mean(y_pred))**2)
