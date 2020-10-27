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


def identity(z):
    r"""Identity function."""
    return z


def identity_prime(z):
    r"""Derivative of the identity function."""
    return np.ones(z.shape)


def sigmoid(z):
    r"""The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    r"""Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    r"""The tanh function."""
    return np.tanh(z)


def tanh_prime(z):
    r"""Derivative of the tanh function."""
    return 1 - tanh(z) ** 2


def relu(z):
    r"""The ReLu function."""
    return (z >= 0) * z


def relu_prime(z):
    r"""Derivative of the ReLu function."""
    return 1. * (z >= 0)


def softmax(z):
    r"""The softmax function."""
    exp_term = np.exp(z)
    return exp_term / exp_term.sum(axis=1, keepdims=True)
