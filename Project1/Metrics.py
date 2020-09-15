# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np


def r2(y_real, y_predict):
    """Calculates the r-square accuracy metric score.

    :param y_real: series: The real values.
    :param y_predict: The values obtained from a prediction.

    :return: float: The r-square value.
    """
    return 1 - np.sum((y_real - y_predict) ** 2) \
        / np.sum((y_real - np.mean(y_real)) ** 2)


def mse(y_real, y_predict):
    """Calculates the mean square error accuracy metric
    score.

    :param y_real: The real values.
    :param y_predict: The values obtained from a prediction.

    :return: float: The mean square error value.
    """
    n = np.size(y_predict)
    return np.sum((y_real-y_predict)**2)/n
