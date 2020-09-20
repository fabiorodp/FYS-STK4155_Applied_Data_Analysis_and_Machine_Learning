# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class ComplexityStudy:
    results = {
        "nr_samples": [],
        "poly_degrees": [],
        "MSE_train_values": [],
        "R2_train_values": [],
        "MSE_test_values": [],
        "R2_test_values": []
    }

    def __init__(self, pipeline):
        self.pipe = pipeline
        self.model_name, self.nr_samples, self.degree, self.seed = \
            pipeline.get_CreateData_parameters()

    def fit(self, nr_samples, poly_degrees):

        # reset results dictionary
        ComplexityStudy.results = \
            {k: [] for k in ComplexityStudy.results.keys()}

        self.degree = poly_degrees
        self.nr_samples = nr_samples

        for nr_sample in self.nr_samples:
            for poly_degree in self.degree:
                ComplexityStudy.results["nr_samples"].append(nr_sample)
                ComplexityStudy.results["poly_degrees"].append(poly_degree)

                self.pipe.set_CreateData_parameters(
                    self.model_name, nr_sample, poly_degree, self.seed)

                z_predict_train, z_predict_test = self.pipe.fit_predict()
                z_train, z_test = self.pipe.get_z_values()

                ComplexityStudy.results["R2_train_values"] \
                    .append(self.r2(z_train, z_predict_train))

                ComplexityStudy.results["MSE_train_values"] \
                    .append(self.mse(z_train, z_predict_train))

                ComplexityStudy.results["R2_test_values"] \
                    .append(self.r2(z_test, z_predict_test))

                ComplexityStudy.results["MSE_test_values"] \
                    .append(self.mse(z_test, z_predict_test))

    def r2(self, y_real, y_predict):
        """Calculates the r-square accuracy metric score.
        :param y_real: series: The real values.
        :param y_predict: The values obtained from a prediction.
        :return: float: The r-square value.
        """
        return 1 - np.sum((y_real - y_predict) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)

    def mse(self, y_real, y_predict):
        """Calculates the mean square error accuracy metric
        score.
        :param y_real: The real values.
        :param y_predict: The values obtained from a prediction.
        :return: float: The mean square error value.
        """
        n = np.size(y_predict)
        return np.sum((y_real - y_predict) ** 2) / n

    def get_results(self, form="df"):
        if form == "df":
            return pd.DataFrame(ComplexityStudy.results)
        else:
            return ComplexityStudy.results

    def plot_study(self, function_of):
        if function_of == "poly_degrees":
            for nr_sample in self.nr_samples:
                lst_i = []
                lst_mse_train = []
                lst_mse_test = []

                for idx, i in enumerate(ComplexityStudy.results["nr_samples"]):
                    if i == nr_sample:
                        lst_i.append(i)
                        lst_mse_train.append(
                            ComplexityStudy.results["MSE_train_values"][idx])

                        lst_mse_test.append(
                            ComplexityStudy.results["MSE_test_values"][idx])

                self._plot_complexity(title="For {}".format(nr_sample),
                                      complexity=self.degree,
                                      train_mse_values=lst_mse_train,
                                      test_mse_values=lst_mse_test)

        elif function_of == "nr_samples":
            for poly_degree in self.degree:
                lst_i = []
                lst_mse_train = []
                lst_mse_test = []

                for idx, i in enumerate(ComplexityStudy.results["poly_degrees"]):
                    if i == poly_degree:
                        lst_i.append(i)
                        lst_mse_train.append(ComplexityStudy.results["MSE_train_values"][idx])
                        lst_mse_test.append(ComplexityStudy.results["MSE_test_values"][idx])

                self._plot_complexity(title="For {}".format(poly_degree),
                                      complexity=self.nr_samples,
                                      train_mse_values=lst_mse_train,
                                      test_mse_values=lst_mse_test)
        else:
            pass

    @staticmethod
    def _plot_complexity(title, complexity, train_mse_values, test_mse_values):
        plt.plot(
            complexity,
            train_mse_values,
            label='Train set')

        plt.plot(
            complexity,
            test_mse_values,
            label='Test set')

        plt.legend()
        plt.title(title)
        plt.xlabel('Model complexity')
        plt.ylabel('Prediction Error [MSE]')
        plt.show()


class Bootstrap:

    def __init__(self, CreateData, PreProcessing, ML_Model, seed=10):
        self.CreateData = CreateData
        self.PreProcessing = PreProcessing
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
