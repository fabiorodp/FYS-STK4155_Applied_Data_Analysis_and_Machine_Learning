# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Bootstrap:

    def __init__(self, CreateData, ML_Model, seed=10):
        self.CreateData = CreateData
        self.ML_Model = ML_Model
        self.seed = seed

    def plot(self, n_boostraps=100, maxdegree=14, verbose=False):
        error = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)

        for degree in range(maxdegree):

            # creating data according to nr_samples and plydegree
            cd = self.CreateData(
                model_name="FrankeFunction",
                seed=10,
                nr_samples=100,
                degree=degree
            )
            cd.fit()
            X, z = cd.get()

            # splitting train and test data
            X_train, X_test, z_train, z_test =\
                train_test_split(X, z, test_size=0.2, random_state=self.seed)

            model = self.ML_Model()
            y_pred = np.empty((z_test.shape[0], n_boostraps))

            for i in range(n_boostraps):
                x_, y_ = resample(X_train, z_train)
                model.fit(x_, y_)
                y_pred[:, i] = model.predict(X_test).ravel()

            polydegree[degree] = degree
            error[degree] = np.mean(np.mean((z_test[:, np.newaxis] - y_pred) ** 2, axis=1, keepdims=True))
            bias[degree] = np.mean((z_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
            variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

            if verbose is True:
                print('Polynomial degree:', degree)
                print('Error:', error[degree])
                print('Bias^2:', bias[degree])
                print('Var:', variance[degree])
                print('{} >= {} + {} = {}'\
                      .format(error[degree],
                              bias[degree],
                              variance[degree],
                              bias[degree] +
                              variance[degree]))

        plt.plot(polydegree, error, label='Error')
        plt.plot(polydegree, bias, label='bias')
        plt.plot(polydegree, variance, label='Variance')
        plt.ylabel("Metrics: [error, bias^2 and variance]")
        plt.xlabel("Polynomial degrees")
        plt.legend()
        plt.show()
