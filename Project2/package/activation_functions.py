# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
activation_functions.py
~~~~~~~~~~

A module to implement the activation functions.
"""

import numpy as np


def identity(y_hat):
    r"""Identity function."""
    return y_hat


def identity_prime(y_hat):
    r"""Derivative of the identity function."""
    return np.ones(y_hat.shape)


def sigmoid(y_hat):
    r"""The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-y_hat))


def sigmoid_prime(y_hat):
    r"""Derivative of the sigmoid function."""
    return sigmoid(y_hat) * (1.0 - sigmoid(y_hat))


def tanh(y_hat):
    r"""The tanh function."""
    return np.tanh(y_hat)


def tanh_prime(y_hat):
    r"""Derivative of the tanh function."""
    return 1.0 - tanh(y_hat) ** 2


def relu(y_hat):
    r"""The ReLu function."""
    return (y_hat >= 0) * y_hat


def relu_prime(y_hat):
    r"""Derivative of the ReLu function."""
    return 1.0 * (y_hat >= 0)


def softmax(y_hat):
    r"""The softmax function."""
    exp_term = np.exp(y_hat)
    return exp_term / exp_term.sum()  # (axis=1, keepdims=True)


def softmax_prime(y_hat):
    r"""The softmax function."""
    return softmax(y_hat) * (1.0 - softmax(y_hat))
