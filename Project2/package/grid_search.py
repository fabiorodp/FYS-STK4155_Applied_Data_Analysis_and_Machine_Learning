# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
grid_search.py
~~~~~~~~~~

A module to implement Grid Searches.
"""

from Project2.package.cost_functions import r2, mse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time


class GridSearch:
    """Class to find the best parameters for ETAs as function of
     Batch-sizes, epochs or lambdas."""

    @staticmethod
    def _plot_heatmaps(params, param1, param2, r2_train, mse_train, r2_test,
                       mse_test, elapsed):

        datas = [mse_train, mse_test, r2_train, r2_test, elapsed]

        titles = ["MSE values for Training",
                  "MSE values for Testing",
                  "R2-score values for Training",
                  "R2-score values for Testing",
                  "Time elapsed (in sec) for training the model"]

        for data, title in zip(datas, titles):
            sns.heatmap(data=data, xticklabels=param1,
                        yticklabels=param2, annot=True,
                        annot_kws={"size": 8.5}, fmt=".2f")

            if params == 'ETASxLAMBDAS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Regularization $\lambda$ = {param2}")

            elif params == 'ETASxEPOCHS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Epochs = {param2}")

            elif params == 'ETASxBATCHES':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Batch sizes = {param2}")

            elif params == 'ETASxDECAYS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Decays = {param2}")

            elif params == 'ETASxGAMMAS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Gamma $\gamma$ = {param2}")

            plt.title(title)
            plt.show()

    @staticmethod
    def _verbose(params, p1, p2, r2_train_value, mse_train_value,
                 r2_test_value, mse_test_value, elapsed):

        if params == 'ETASxLAMBDAS':
            print(f'Eta and Lambda: {p1}, {p2}.')

        elif params == 'ETASxEPOCHS':
            print(f'Eta and Epochs: {p1}, {p2}.')

        elif params == 'ETASxBATCHES':
            print(f'Eta and Batch size: {p1}, {p2}.')

        elif params == 'ETASxDECAYS':
            print(f'Eta and decay: {p1}, {p2}.')

        elif params == 'ETASxGAMMAS':
            print(f'Eta and gamma: {p1}, {p2}.')

        print(f'Training R2 and MSE: '
              f'{r2_train_value}, {mse_train_value}.')

        print(f'Testing R2 and MSE: '
              f'{r2_test_value}, {mse_test_value}.')

        print(f'Time elapsed for training {elapsed} s.')

    def __init__(self, model, params='ETASxEPOCHS', random_state=None):
        """
        Constructor for the class.

        :param model: class: Regression model.
        :param params: string: Parameter for the Grid-Search study.
        :param random_state: int: Seed for the experiment.
        """

        if random_state is not None:
            np.random.seed(random_state)

        self.model = model
        self.params = params
        self.random_state = random_state

    def run(self, X_train, X_test, y_train, y_test, epochs, etas,
            batch_sizes, lambdas, decays, gammas, plot_results=False,
            verbose=False):
        """
        Run the experiment to search the best parameters.
        """
        param1, param2 = None, None

        if self.params == 'ETASxLAMBDAS':
            param1, param2 = etas, lambdas

        elif self.params == 'ETASxEPOCHS':
            param1, param2 = etas, epochs

        elif self.params == 'ETASxBATCHES':
            param1, param2 = etas, batch_sizes

        elif self.params == 'ETASxDECAYS':
            param1, param2 = etas, decays

        elif self.params == 'ETASxGAMMAS':
            param1, param2 = etas, gammas

        mse_train = np.zeros(shape=(len(param2), len(param1)))
        mse_test = np.zeros(shape=(len(param2), len(param1)))
        r2_train = np.zeros(shape=(len(param2), len(param1)))
        r2_test = np.zeros(shape=(len(param2), len(param1)))
        elapsed = np.zeros(shape=(len(param2), len(param1)))

        for c_idx, p1 in enumerate(param1):
            for r_idx, p2 in enumerate(param2):

                # set parameters
                if self.params == 'ETASxLAMBDAS':
                    self.model.set_lambda(new_lambda=p2)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_decay(new_decay=decays)
                    self.model.set_epochs(new_epochs=epochs)
                    self.model.set_batch_size(new_batch_size=batch_sizes)

                elif self.params == 'ETASxEPOCHS':
                    self.model.set_epochs(new_epochs=p2)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_decay(new_decay=decays)
                    self.model.set_lambda(new_lambda=lambdas)
                    self.model.set_batch_size(new_batch_size=batch_sizes)

                elif self.params == 'ETASxBATCHES':
                    self.model.set_epochs(new_epochs=epochs)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_decay(new_decay=decays)
                    self.model.set_lambda(new_lambda=lambdas)
                    self.model.set_batch_size(new_batch_size=p2)

                elif self.params == 'ETASxDECAYS':
                    self.model.set_epochs(new_epochs=epochs)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_decay(new_decay=p2)
                    self.model.set_lambda(new_lambda=lambdas)
                    self.model.set_batch_size(new_batch_size=batch_sizes)

                elif self.params == 'ETASxGAMMAS':
                    self.model.set_epochs(new_epochs=epochs)
                    self.model.set_eta(new_eta=p1)
                    self.model.set_decay(new_decay=decays)
                    self.model.set_lambda(new_lambda=lambdas)
                    self.model.set_batch_size(new_batch_size=batch_sizes)
                    self.model.set_gamma(new_gamma=p2)

                # train model
                time0 = time()
                self.model.fit(X_train, y_train)
                time1 = time()
                elapsed_value = time1 - time0
                elapsed[r_idx, c_idx] = elapsed_value

                # assess model
                y_hat = self.model.predict(X_train)
                r2_train_value = r2(y_train, y_hat)
                r2_train[r_idx, c_idx] = r2_train_value
                mse_train_value = mse(y_train, y_hat)
                mse_train[r_idx, c_idx] = mse_train_value

                y_tilde = self.model.predict(X_test)
                r2_test_value = r2(y_test, y_tilde)
                r2_test[r_idx, c_idx] = r2_test_value
                mse_test_value = mse(y_test, y_tilde)
                mse_test[r_idx, c_idx] = mse_test_value

                if verbose is True:
                    self._verbose(params=self.params, p1=p1, p2=p2,
                                  r2_train_value=r2_train_value,
                                  mse_train_value=mse_train_value,
                                  r2_test_value=r2_test_value,
                                  mse_test_value=mse_test_value,
                                  elapsed=elapsed_value)

        if plot_results is True:
            self._plot_heatmaps(params=self.params, param1=param1,
                                param2=param2, r2_train=r2_train,
                                mse_train=mse_train, r2_test=r2_test,
                                mse_test=mse_test, elapsed=elapsed)

        return r2_train, mse_train, r2_test, mse_test, elapsed
