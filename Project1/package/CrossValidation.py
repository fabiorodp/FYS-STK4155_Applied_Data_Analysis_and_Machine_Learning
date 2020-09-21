# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class CrossValidation:
    def __init__(self, CreateData, ML_Model, function_of="nr_samples",
                 random_seed=10):

        self.random_seed = random_seed
        self.CreateData = CreateData
        self.ML_Model = ML_Model
        self.function_of = function_of

        self.complexities = None
        self.mse = None

    def fit(self, complexities, k=5, verboose=False):
        self.complexities = np.zeros(len(complexities))
        self.mse = np.zeros(len(complexities))
        kfold = KFold(n_splits=k)

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

            estimated_mse_folds = \
                cross_val_score(
                    model, X, z[:, np.newaxis],
                    scoring='neg_mean_squared_error', cv=kfold)

            # cross_val_score return an array containing the estimated negative mse for every fold.
            # we have to the the mean of every array in order to get an estimate of the mse of the model
            self.mse[cplx_idx] = np.mean(-estimated_mse_folds)

    def plot(self):
        plt.plot(self.complexities, self.mse, label='mse')

        plt.ylabel("MSE")

        if self.function_of == "poly_degrees":
            plt.xlabel("Complexity: Polynomial degrees")

        elif self.function_of == "nr_samples":
            plt.xlabel("Complexity: Number of samples")

        elif self.function_of == "lambda":
            plt.xlabel("Complexity: Lambda value")

        plt.title("Cross Validation")
        plt.legend()
        plt.show()
