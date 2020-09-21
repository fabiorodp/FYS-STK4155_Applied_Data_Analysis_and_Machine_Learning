# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


class AccuracyStudies:
    def __init__(self, CreateData, ML_Model,
                 seed=10, function_of="nr_samples", metric="MSE"):

        self.CreateData = CreateData
        self.ML_Model = ML_Model
        self.seed = seed
        self.metric = metric

        self.function_of = function_of
        self.train_metric = None
        self.test_metric = None
        self.complexities = None

    def fit(self, complexities=None, verbose=False):

        if isinstance(complexities, (list, np.ndarray)) is False:
            raise TypeError("Complexities must be in list or np.narray.")

        self.train_metric = np.zeros(len(complexities))
        self.test_metric = np.zeros(len(complexities))
        self.complexities = np.zeros(len(complexities))

        for cplx_idx, complexity in enumerate(complexities):
            self.complexities[cplx_idx] = complexity

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

            model.fit(X_train, z_train)
            z_pred_train = model.predict(X_train).ravel()
            z_pred_test = model.predict(X_test).ravel()

            if self.metric == "MSE":
                self.train_metric[cplx_idx] = mean_squared_error(z_train, z_pred_train)
                self.test_metric[cplx_idx] = mean_squared_error(z_test, z_pred_test)

            elif self.metric == "R2-score":
                self.train_metric[cplx_idx] = r2_score(z_train, z_pred_train)
                self.test_metric[cplx_idx] = r2_score(z_test, z_pred_test)

            # printing the results
            if verbose is True:
                self._verbose(complexity, self.train_metric[cplx_idx],
                              self.test_metric[cplx_idx])

    def plot(self):
        if self.metric == "MSE":
            plt.plot(self.complexities, self.train_metric, label='MSE Train')
            plt.plot(self.complexities, self.test_metric, label='MSE Test')
            plt.ylabel("MSE")

        elif self.metric == "R2-score":
            plt.plot(self.complexities, self.train_metric, label='R2-score Train')
            plt.plot(self.complexities, self.test_metric, label='R2-score Test')
            plt.ylabel("R2-scores")

        if self.function_of == "poly_degrees":
            plt.xlabel("Complexity: Polynomial degrees")

        elif self.function_of == "nr_samples":
            plt.xlabel("Complexity: Number of samples")

        elif self.function_of == "lambda":
            plt.xlabel("Complexity: Lambda value")

        plt.legend()
        plt.show()

    def _verbose(self, complexity, train_metric, test_metric):
        print('Complexity: {}: '.format(self.function_of), complexity)
        print('Train {}: '.format(self.metric), train_metric)
        print('Test MSE: ', test_metric)

    def print_best_metric(self):
        if self.metric == "R2-score":
            max_train = np.argmax(self.train_metric)
            max_test = np.argmax(self.test_metric)
            print("Best Train R2-score as function of {} ({}): "
                  .format(self.function_of, self.complexities[max_train]),
                  self.train_metric[max_train])
            print("Best Test R2-score as function of {} ({}): "
                  .format(self.function_of, self.complexities[max_test]),
                  self.train_metric[max_test])

        elif self.metric == "MSE":
            min_train = np.argmin(self.train_metric)
            min_test = np.argmin(self.test_metric)
            print("Best Train MSE as function of {} ({}): "
                  .format(self.function_of, self.complexities[min_train]),
                  self.train_metric[min_train])
            print("Best Test MSE as function of {} ({}): "
                  .format(self.function_of, self.complexities[min_test]),
                  self.train_metric[min_test])
