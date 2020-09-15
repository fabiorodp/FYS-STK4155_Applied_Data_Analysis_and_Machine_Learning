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
        self.X, self.y = None, None
        self.coef_OLS, self.coef_var_OLS, self.coef_CI_OLS = \
            None, None, None
        self.y_predict = None

    def fit(self, X, y):
        self.X, self.y = X, y

        if self.technique == "OLS":
            self.OLS(), self.beta_CI_OLS(), self.predict()

        elif self.technique == "ridge":
            self.ridge()

        elif self.technique == "lasso":
            self.lasso()

    def OLS(self):
        """ Calculates the optimal coefficients and its
        variance."""
        # linear inversion technique first
        try:
            self.coef_OLS = np.linalg.inv(self.X.T @ self.X) \
                            @ self.X.T @ self.y
            self.coef_var_OLS = np.linalg.inv(self.X.T @ self.X)

        # if it does not work, use SVD technique
        except:
            self.coef_OLS = np.linalg.pinv(self.X.T @ self.X) \
                            @ self.X.T @ self.y
            self.coef_var_OLS = np.linalg.pinv(self.X.T @ self.X)

    def beta_CI_OLS(self, percentile=0.95):
        """ Calculates the confidence interval of the coefficients
        beta of OLS.

        :param percentile: float: Significance level

        :return: float: Confidence interval.
        """
        cov = np.var(self.y) * np.linalg.pinv(self.X.T @ self.X)
        std_err_beta_hat = np.sqrt(np.diag(cov))
        z_score = stats.norm(0, 1).ppf(percentile)
        self.coef_CI_OLS = z_score * std_err_beta_hat
        return self.coef_CI_OLS

    def predict(self, X=None):
        """Predicts the response variable from the design
        matrix times the optimal beta.

        :param X: np.array: Given explanatory variables.

        :return: np.array: Predictions y given X.
        """
        if self.technique == "OLS":
            if X is None:
                self.y_predict = self.X @ self.coef_OLS
                return self.y_predict
            else:
                self.y_predict = X @ self.coef_OLS
                return self.y_predict

        elif self.technique == "ridge":
            pass

        elif self.technique == "lasso":
            pass

    def ridge(self):
        pass

    def lasso(self):
        pass
