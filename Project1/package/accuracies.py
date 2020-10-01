# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample


def r2(y_real, y_predict):
    """
    Calculates the r-square accuracy metric score.

    :param y_real: series: The real values.
    :param y_predict: The values obtained from a prediction.

    :return: float: The r-square value.
    """
    return 1 - np.sum((y_real - y_predict) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)


def mse(y_real, y_predict):
    """
    Calculates the mean square error accuracy metric score.

    :param y_real: The real values.
    :param y_predict: The values obtained from a prediction.

    :return: float: The mean square error value.
    """
    n = np.size(y_predict)
    return np.sum((y_real - y_predict) ** 2) / n


def bias_variance_decomp(model, X, z, n_boostraps=1,
                         random_seed=None, plot=False):


    y_pred = np.empty((z_test.shape[0], n_boostraps))

    for i in range(n_boostraps):
        x_, y_ = resample(X_train, z_train)
        y_pred[:, i] = model.fit(x_, y_).predict(X_test).ravel()

    polydegree[degree] = degree
    error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

    if plot is not False:
        plt.plot(complexities, mse, label='Error/MSE')
        plt.plot(complexities, avg_bias, "--", label='bias')
        plt.plot(complexities, avg_var, label='Variance')
        plt.ylabel("Metrics: [Error/MSE, bias^2 and variance]")
        plt.xlabel("Polynomial degrees")
        plt.title("Bias-variance tradeoff")
        plt.legend()
        plt.show()

