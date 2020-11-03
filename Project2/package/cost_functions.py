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


def mse(y_true, y_hat):
    """The MSE function."""
    return 0.5 * ((y_hat - y_true) ** 2).mean()


def mse_prime(y_true, y_hat):
    """Derivative of the MSE function."""
    return 2.0/len(y_true) * (y_hat - y_true)


def accuracy_score(y_true, y_hat):
    return np.sum(y_true == y_hat) / len(y_hat)


def accuracy_score_prime(y_true, y_hat):
    return (y_hat - y_true) / len(y_true)


def crossentropy(y_hat, y_true):
    return - (y_true * np.log(y_hat) + (1.0 - y_true)
              * np.log(1.0 - y_hat)).mean()


def crossentropy_prime(y_hat, y_true):
    return -y_true / y_hat + (1.0 - y_true) / (1.0 - y_hat)
