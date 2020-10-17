# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
gradient_descent.py
~~~~~~~~~~

A module to implement the many variants of gradient descent learning such as:
Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD).
"""

import numpy as np


class BGD:
    """Batch Gradient Descent."""

    def __init__(self, eta=0.1, epochs=1000, lambda_=0, regularization=None,
                 random_state=None):
        """
        Constructor for the class.

        :param eta: float: Learning rate.
        :param epochs: int: Number of epochs/interactions.
        :param lambda_: float: Regularization value.
        :param regularization: string: It can be "l2" for ridge or
                                                 "l1" lasso.
        :param random_state: int: Seed for the experiment.
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.eta = eta
        self.epochs = epochs
        self.lambda_ = lambda_
        self.regularization = regularization
        self.m, self.p, self.coef_ = None, None, None

    def fit(self, X, z):
        """
        Fitting parameters to regression model.

        :param X: ndarray with the design matrix.
        :param z: ndarray with the labeled response variables.
        """
        self.m = X.shape[0]                      # sample space
        self.p = X.shape[1]                      # feature space
        self.coef_ = np.random.randn(self.p, 1)  # initializing coeff_

        for _ in range(self.epochs):
            gradients = 2.0 / self.m * X.T @ (X @ self.coef_ - z)

            if self.regularization == "l2":      # Ridge regularization
                gradients += self.lambda_ * self.coef_

            elif self.regularization == "l1":    # Lasso regularization
                gradients += self.lambda_ * self.subgradient_vector()

            self.coef_ -= self.eta * gradients

    def subgradient_vector(self):
        """Lasso's regularization sub-gradient vector."""
        g = np.zeros(self.coef_.shape)
        for idx, theta in enumerate(self.coef_):

            if theta < 0:
                g[idx, 0] = -1.0

            elif theta == 0:
                g[idx, 0] = 0.0

            elif theta > 0:
                g[idx, 0] = 1.0

        return g

    def set_eta(self, new_eta):
        """
        Changing eta Learning rate.

        :param new_eta: float: New eta value.
        """
        self.eta = new_eta

    def set_epochs(self, new_epochs):
        """
        Changing the number of interactions (epochs).

        :param new_epochs: int: New N_interactions value.
        """
        self.epochs = new_epochs

    def set_lambda(self, new_lambda):
        """
        Changing the regularization.

        :param new_lambda: float: New lambda regularization.
        """
        self.lambda_ = new_lambda

    def predict(self, X):
        """
        Predicts the response variable from the design
        matrix times the optimal beta.

        :param X: np.array: Given explanatory variables.

        :return: np.array: Predictions z given X.
        """
        return X @ self.coef_


class SGD(BGD):
    """Stochastic Gradient Descent."""

    def __init__(self, eta=0.1, epochs=1000, lambda_=0, regularization=None,
                 random_state=None):
        """
        Constructor for the class.

        :param lambda: float: Regularization value of the model.
        :param random_state: int: Seed for the experiment.
        """
        super().__init__(eta, epochs, lambda_, regularization, random_state)

    def fit(self, X, z):
        """
        Fitting parameters to regression model.

        :param X: ndarray with the design matrix.
        :param z: ndarray with the response variables.
        """
        self.m = X.shape[0]                      # sample space
        self.p = X.shape[1]                      # number of features
        self.coef_ = np.random.randn(self.p, 1)  # initialising coeff_

        for _ in range(self.epochs):
            for _ in range(self.m):
                random_index = np.random.randint(self.m)
                xi = X[random_index:random_index + 1]
                zi = z[random_index:random_index + 1]
                gradients = 2 * xi.T @ (xi @ self.coef_ - zi)

                if self.regularization == "l2":  # Ridge regularization
                    gradients += self.lambda_ * self.coef_

                elif self.regularization == "l1":  # Lasso regularization
                    gradients += self.lambda_ * self.subgradient_vector()

                self.coef_ -= self.eta * gradients
