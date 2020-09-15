# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np


class LinearRegressionTechniques:
    """Class with many different linear regression techniques
    for machine learning studies."""

    def __init__(self, technique_name="OLS"):
        self.technique = technique_name
        self.X, self.y, self.beta_hat, self.y_predict = \
            None, None, None, None

    def fit(self, X, y):
        self.X, self.y = X, y

        if self.technique == "OLS":
            self.OLS()
        elif self.technique == "ridge":
            self.ridge()
        elif self.technique == "lasso":
            self.lasso()

    def OLS(self):
        # linear inversion technique first
        try:
            self.beta_hat = np.linalg.inv(self.X.T @ self.X) \
                            @ self.X.T @ self.y

        # if it does not work, use SVD technique
        except:
            self.beta_hat = np.linalg.pinv(self.X.T @ self.X) \
                            @ self.X.T @ self.y

    def coef_(self):
        return self.beta_hat

    def coef_var(self):
        # linear inversion technique first
        try:
            return np.linalg.inv(self.X.T @ self.X)

        # if it does not work, use SVD technique
        except:
            return np.linalg.pinv(self.X.T @ self.X)

    def confidence_interval(self):
        pass

    def predict(self, X=None):
        """Predicts the response variable from the design
        matrix times the optimal beta."""
        if X is None:
            self.y_predict = self.X @ self.beta_hat
            return self.y_predict
        else:
            self.y_predict = X @ self.beta_hat
            return self.y_predict

    def ridge(self):
        pass

    def lasso(self):
        pass
