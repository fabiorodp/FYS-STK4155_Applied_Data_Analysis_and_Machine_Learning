# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mlxtend.evaluate import bias_variance_decomp


class Bootstrap:

    def __init__(self, CreateData, ML_Model, n_boostraps=100,
                 seed=10, function_of="nr_samples"):

        self.CreateData = CreateData
        self.ML_Model = ML_Model
        self.seed = seed

        self.n_boostraps = n_boostraps
        self.function_of = function_of
        self.error = None
        self.bias = None
        self.variance = None
        self.complexities = None
        self.mse_train = None
        self.mse_test = None

    def fit(self, complexities=None, verbose=False):
        if isinstance(complexities, (list, np.ndarray)) is False:
            raise TypeError("Complexities must be in list or np.narray.")

        self.error = np.zeros(len(complexities))
        self.bias = np.zeros(len(complexities))
        self.variance = np.zeros(len(complexities))
        self.complexities = np.zeros(len(complexities))
        self.mse_train = np.zeros(len(complexities))
        self.mse_test = np.zeros(len(complexities))

        for cplx_idx, complexity in enumerate(complexities):

            # calling the regression model
            model = self.ML_Model

            # generating data
            X, z = None, None
            cd = self.CreateData

            if self.function_of == "nr_samples":
                cd.set_nr_samples(complexity)
                cd.fit()
                X, z = cd.get()

            elif self.function_of == "poly_degrees":
                cd.set_poly_degree(complexity)
                cd.fit()
                X, z = cd.get()

            elif self.function_of == "lambda":
                cd.fit()
                X, z = cd.get()
                model.set_lambda(complexity)

            # splitting train and test data
            X_train, X_test, z_train, z_test = \
                train_test_split(X, z, test_size=0.2, random_state=self.seed)

            # scaling data ??
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaler.fit(X_train)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

            # empty array to store the predictions
            y_pred_train = np.empty((z_train.shape[0], self.n_boostraps))
            y_pred = np.empty((z_test.shape[0], self.n_boostraps))

            # bootstrapping the data, fitting and predicting
            for i in range(self.n_boostraps):
                x_, y_ = resample(X_train, z_train)
                model.fit(x_, y_)
                y_pred_train[:, i] = model.predict(X_train).ravel()
                y_pred[:, i] = model.predict(X_test).ravel()

            # calculating error, bias, variance, mse_train and mse_test
            self.calculate(cplx_idx, complexity, z_train, z_test,
                           y_pred, y_pred_train)

            # printing the results
            if verbose is True:
                self._verbose(cplx_idx, complexity)

    def calculate(self, cplx_idx, complexity, z_train, z_test, y_pred, y_pred_train):
        self.complexities[cplx_idx] = complexity

        self.error[cplx_idx] = \
            np.mean(np.mean((z_test[:, np.newaxis] - y_pred) ** 2, axis=1, keepdims=True))

        self.bias[cplx_idx] = \
            np.mean((z_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)

        self.variance[cplx_idx] = \
            np.mean(np.var(y_pred, axis=1, keepdims=True))

        self.mse_train[cplx_idx] = mean_squared_error(
            z_train, y_pred_train[:, cplx_idx])

        self.mse_test[cplx_idx] = mean_squared_error(
            z_test, y_pred[:, cplx_idx])

    def plot(self):
        plt.plot(self.complexities, self.error, label='Error')
        plt.plot(self.complexities, self.bias, label='bias')
        plt.plot(self.complexities, self.variance, label='Variance')
        plt.plot(self.complexities, self.mse_train, label='MSE train')
        plt.plot(self.complexities, self.mse_test, label='MSE test')
        plt.ylabel("Metrics: [error, bias^2 and variance]")
        if self.function_of == "poly_degrees":
            plt.xlabel("Polynomial degrees")
        elif self.function_of == "nr_samples":
            plt.xlabel("Number of samples")
        elif self.function_of == "lambda":
            plt.xlabel("Lambda value")
        plt.title("Bootstrapping and bias-variance tradeoff")
        plt.legend()
        plt.show()

    def _verbose(self, cplx_idx, complexity):
        if self.function_of == "poly_degrees":
            print('Polynomial degree:', complexity)

        elif self.function_of == "nr_samples":
            print('Number of samples:', complexity)

        elif self.function_of == "lambda":
            print('Lambda value:', complexity)

        print('Error:', self.error[cplx_idx])
        print('Bias^2:', self.bias[cplx_idx])
        print('Var:', self.variance[cplx_idx])
        print('{} >= {} + {} = {}' \
              .format(self.error[cplx_idx],
                      self.bias[cplx_idx],
                      self.variance[cplx_idx],
                      self.bias[cplx_idx] +
                      self.variance[cplx_idx]))


class BootstrapMLextend:
    def __init__(self, CreateData, ML_Model, function_of="nr_samples",
                 num_rounds=100, loss="mse", random_seed=10):

        self.num_rounds = num_rounds
        self.loss = loss
        self.random_seed = random_seed
        self.CreateData = CreateData
        self.ML_Model = ML_Model
        self.function_of = function_of

        self.complexities = None
        self.error = None
        self.bias = None
        self.variance = None

    def fit(self, complexities=None, verbose=False):
        self.complexities = np.zeros(len(complexities))
        self.error = np.zeros(len(complexities))
        self.bias = np.zeros(len(complexities))
        self.variance = np.zeros(len(complexities))

        for cplx_idx, complexity in enumerate(complexities):
            self.complexities[cplx_idx] = complexity

            # calling the regression model
            model = self.ML_Model

            # generating data
            cd = self.CreateData
            X, z = None, None

            if self.function_of == "nr_samples":
                cd.set_nr_samples(complexity)
                cd.fit()
                X, z = cd.get()

            elif self.function_of == "poly_degrees":
                cd.set_poly_degree(complexity)
                cd.fit()
                X, z = cd.get()

            elif self.function_of == "lambda":
                cd.fit()
                X, z = cd.get()
                model.set_lambda(complexity)

            # splitting train and test data
            X_train, X_test, z_train, z_test = \
                train_test_split(X, z, test_size=0.2, random_state=self.random_seed)

            # scaling data ??
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaler.fit(X_train)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

            avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
                model, X_train, z_train, X_test, z_test,
                num_rounds=self.num_rounds, loss=self.loss,
                random_seed=self.random_seed)

            self.error[cplx_idx] = avg_expected_loss
            self.bias[cplx_idx] = avg_bias
            self.variance[cplx_idx] = avg_var

    def plot(self):
        plt.plot(self.complexities, self.error, label='error')
        plt.plot(self.complexities, self.bias, label='bias')
        plt.plot(self.complexities, self.variance, label='variance')
        plt.ylabel("Metrics")

        if self.function_of == "poly_degrees":
            plt.xlabel("Complexity: Polynomial degrees")

        elif self.function_of == "nr_samples":
            plt.xlabel("Complexity: Number of samples")

        elif self.function_of == "lambda":
            plt.xlabel("Complexity: Lambda value")

        plt.title("Bootstrapping and bias-variance tradeoff")
        plt.legend()
        plt.show()
