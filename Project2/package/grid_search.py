# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
grid_search.py
~~~~~~~~~~

A module to implement Grid Searches.
"""

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GridSearch:
    """Class to find the best parameters for ETAs and LAMBDAs."""

    @staticmethod
    def _plot_heatmaps(params, param1, param2, r2_train, mse_train, r2_test,
                       mse_test):

        datas = [mse_train, mse_test, r2_train, r2_test]

        titles = ["MSE values for Training",
                  "MSE values for Testing",
                  "R2-score values for Training",
                  "R2-score values for Testing"]

        for data, title in zip(datas, titles):
            sns.heatmap(data=data, xticklabels=param1,
                        yticklabels=param2, annot=True)

            if params == 'ETAxLAMBDA':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Regularization $\lambda$ = {param2}")

            elif params == 'ETAxEPOCHS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Epochs = {param2}")

            plt.title(title)
            plt.show()

    @staticmethod
    def _verbose(params, p1, p2, r2_train_value, mse_train_value,
                 r2_test_value, mse_test_value):

        if params == 'ETAxLAMBDA':
            print(f'Eta and Lambda: {p1}, {p2}.')

        elif params == 'ETAxEPOCHS':
            print(f'Eta and Epochs: {p1}, {p2}.')

        print(f'Training R2 and MSE: '
              f'{r2_train_value}, {mse_train_value}.')

        print(f'Testing R2 and MSE: '
              f'{r2_test_value}, {mse_test_value}.')

    def __init__(self, model, params='ETAxLAMBDA', random_state=None):
        """
        Constructor for the class.

        :param model: class: Regression model.
        :param params: string: Parameter for the Grid-Search study.
        :param random_state: int: Seed for the experiment.
        """
        self.model = model
        self.params = params
        self.random_state = random_state

    def run(self, X_train, X_test, y_train, y_test, lambdas=None,
            etas=None, epochs=None, plot_results=False, verbose=False):
        """
        Run the experiment to search the best parameters.
        """
        param1, param2 = None, None

        if self.params == 'ETAxLAMBDA':
            param1, param2 = etas, lambdas

        elif self.params == 'ETAxEPOCHS':
            param1, param2 = etas, epochs

        mse_train = np.zeros(shape=(len(param1), len(param2)))
        mse_test = np.zeros(shape=(len(param1), len(param2)))
        r2_train = np.zeros(shape=(len(param1), len(param2)))
        r2_test = np.zeros(shape=(len(param1), len(param2)))

        for r_idx, p2 in enumerate(param2):
            for c_idx, p1 in enumerate(param1):

                # set parameters
                if self.params == 'ETAxLAMBDA':
                    self.model.set_lambda(new_lambda=p2)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_epochs(new_epochs=epochs)

                elif self.params == 'ETAxEPOCHS':
                    self.model.set_epochs(new_epochs=p2)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_lambda(new_lambda=lambdas)

                # train model
                self.model.fit(X_train, y_train)

                # assess model
                y_hat = self.model.predict(X_train)
                r2_train_value = r2_score(y_train, y_hat)
                r2_train[r_idx, c_idx] = r2_train_value
                mse_train_value = mean_squared_error(y_train, y_hat)
                mse_train[r_idx, c_idx] = mse_train_value

                y_tilde = self.model.predict(X_test)
                r2_test_value = r2_score(y_test, y_tilde)
                r2_test[r_idx, c_idx] = r2_test_value
                mse_test_value = mean_squared_error(y_test, y_tilde)
                mse_test[r_idx, c_idx] = mse_test_value

                if verbose is True:
                    self._verbose(params=self.params, p1=p1, p2=p2,
                                  r2_train_value=r2_train_value,
                                  mse_train_value=mse_train_value,
                                  r2_test_value=r2_test_value,
                                  mse_test_value=mse_test_value)

        if plot_results is True:
            self._plot_heatmaps(params=self.params, param1=param1,
                                param2=param2, r2_train=r2_train,
                                mse_train=mse_train, r2_test=r2_test,
                                mse_test=mse_test)

        return r2_train, mse_train, r2_test, mse_test
