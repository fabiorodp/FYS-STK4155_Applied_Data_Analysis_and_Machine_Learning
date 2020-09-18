# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from scipy import stats


class LinearRegressionTechniques:
    """Class with many different linear regression techniques
    for machine learning studies."""

    def __init__(self, technique_name="OLS"):
        self.technique = technique_name
        self.coef_OLS, self.coef_var_OLS = None, None
        self.z_var = None

    def fit(self, X, z, ld=0):
        if self.technique == "OLS":
            self.OLS(X, z)

        elif self.technique == "ridge":
            self.ridge(X, z, ld)

        elif self.technique == "lasso":
            self.lasso(X, z, ld)

    def OLS(self, X, z):
        """ Calculates the optimal coefficients and its
        variance."""
        # linear inversion technique first
        try:
            self.coef_OLS = np.linalg.inv(X.T @ X) @ X.T @ z
            self.coef_var_OLS = np.linalg.inv(X.T @ X)
            self.z_var = np.var(z)

        # if it does not work, use SVD technique
        except:
            self.coef_OLS = np.linalg.pinv(X.T @ X) @ X.T @ z
            self.coef_var_OLS = np.linalg.pinv(X.T @ X)
            self.z_var = np.var(z)

    def beta_CI_OLS(self, percentile=0.95):
        """ Calculates the confidence interval of the coefficients
        beta of OLS.

        :param percentile: float: Significance level

        :return: array: Beta Confidence intervals.
        """
        cov = self.z_var * self.coef_var_OLS
        std_err_beta_hat = np.sqrt(np.diag(cov))
        z_score = stats.norm(0, 1).ppf(percentile)
        self.coef_CI_OLS = z_score * std_err_beta_hat
        return self.coef_CI_OLS

    def predict(self, X):
        """Predicts the response variable from the design
        matrix times the optimal beta.

        :param X: np.array: Given explanatory variables.

        :return: np.array: Predictions z given X.
        """
        if self.technique == "OLS":
            z_predict = X @ self.coef_OLS
            return z_predict

        elif self.technique == "ridge":
            pass

        elif self.technique == "lasso":
            pass

    def ridge(self, X, z, ld):
        pass

    def lasso(self, X, z, ld):
        pass

    def set_new_parameters(self, technique_name="OLS"):
        self.technique = technique_name
