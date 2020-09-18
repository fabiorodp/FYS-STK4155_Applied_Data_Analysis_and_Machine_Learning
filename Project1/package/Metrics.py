# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BiasVarianceStudy:

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
            pipeline.get_data_model_parameters()

    def fit(self, nr_samples, poly_degrees):

        # reset results dictionary
        BiasVarianceStudy.results = \
            {k: [] for k in BiasVarianceStudy.results.keys()}

        self.degree = poly_degrees
        self.nr_samples = nr_samples

        for nr_sample in self.nr_samples:
            for poly_degree in self.degree:
                BiasVarianceStudy.results["nr_samples"].append(nr_sample)
                BiasVarianceStudy.results["poly_degrees"].append(poly_degree)

                self.pipe.set_data_model_new_parameters(
                    self.model_name, nr_sample, poly_degree, self.seed)

                z_predict_train, z_predict_test = self.pipe.fit_predict()
                z_train, z_test = self.pipe.get_z_values()

                BiasVarianceStudy.results["R2_train_values"] \
                    .append(self.r2(z_train, z_predict_train))

                BiasVarianceStudy.results["MSE_train_values"] \
                    .append(self.mse(z_train, z_predict_train))

                BiasVarianceStudy.results["R2_test_values"] \
                    .append(self.r2(z_test, z_predict_test))

                BiasVarianceStudy.results["MSE_test_values"] \
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
            return pd.DataFrame(BiasVarianceStudy.results)
        else:
            return BiasVarianceStudy.results

    def plot_study(self, function_of):
        if function_of == "poly_degrees":
            for nr_sample in self.nr_samples:
                lst_i = []
                lst_mse_train = []
                lst_mse_test = []

                for idx, i in enumerate(BiasVarianceStudy.results["nr_samples"]):
                    if i == nr_sample:
                        lst_i.append(i)
                        lst_mse_train.append(
                            BiasVarianceStudy.results["MSE_train_values"][idx])

                        lst_mse_test.append(
                            BiasVarianceStudy.results["MSE_test_values"][idx])

                self._plot_complexity_mse(title="For {}".format(nr_sample),
                                          complexity=self.degree,
                                          train_mse_values=lst_mse_train,
                                          test_mse_values=lst_mse_test)

        elif function_of == "nr_samples":
            for poly_degree in self.degree:
                lst_i = []
                lst_mse_train = []
                lst_mse_test = []

                for idx, i in enumerate(BiasVarianceStudy.results["poly_degrees"]):
                    if i == poly_degree:
                        lst_i.append(i)
                        lst_mse_train.append(BiasVarianceStudy.results["MSE_train_values"][idx])
                        lst_mse_test.append(BiasVarianceStudy.results["MSE_test_values"][idx])

                self._plot_complexity_mse(title="For {}".format(poly_degree),
                                          complexity=self.nr_samples,
                                          train_mse_values=lst_mse_train,
                                          test_mse_values=lst_mse_test)
        else:
            pass

    @staticmethod
    def _plot_complexity_mse(title, complexity, train_mse_values, test_mse_values):
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
