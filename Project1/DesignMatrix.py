# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np


def design_matrix(x, y, degree):
    """
    Creates a design matrix (X) from polynomial equations of a
    specific degree.

    :param x: explanatory variable x of shape (n, n).
    :param y: explanatory variable y of shape (n, n).
    :param degree: The degree of the polynomial equations.

    :return: X : array : The design matrix (X) of shape (n*n, p),
    where p are the number of the columns or quantity of factor or
    complexity of the data.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))

    for i in range(1, degree + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X
