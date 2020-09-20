# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np


class CreateData:

    def __init__(self, model_name="FrankeFunction", seed=10,
                 nr_samples=None, degree=None):

        self.model = model_name
        self.nr_samples = nr_samples
        self.degree = degree
        self.seed = seed
        self.X, self.z = None, None

    def fit(self):
        if (self.nr_samples is None) or (self.degree is None):
            raise ValueError("Error: It can not fit if there are any "
                             "'None' parameter. Please set new parameters "
                             "before fit.")

        if self.model == "FrankeFunction":
            # seeding the random variables
            np.random.seed(self.seed)

            # generating the explanatory variables x and y
            x = np.random.random(self.nr_samples)  # shape (n,)
            y = np.random.random(self.nr_samples)  # shape (n,)

            # generating repeated values matrix x, y
            # to obtain all possible combinations
            x, y = np.meshgrid(x, y)  # shape (n,n)
            self._design_matrix(x, y)  # shape (n*n, p)

            # generating the response variables (z),
            # from the explanatory variables,
            # using FrankeFunction
            self._franke_function(x, y)  # shape (n*n,)

            # generation a random normally distributed noise
            epsilon = np.random.normal(0, 1, self.z.shape[0])

            # adding the noise to FrankeFunction
            self.z += epsilon

        else:
            return NotImplemented

    def get(self):
        if self.X is None:
            raise ValueError("Not fitted yet")

        else:
            return self.X, self.z

    def _design_matrix(self, x, y):
        """
        Creates a design matrix (X) from polynomial equations of a
        specific degree.
        :param x: explanatory variable x of shape (n, n).
        :param y: explanatory variable y of shape (n, n).
        :param degree: The degree of the polynomial equations.
        :return: X : array : The design matrix (X) of shape (n*n, p),
        where p are the number of the columns or quantity of factor or
        complexity of the data.
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((self.degree + 1) * (self.degree + 2) / 2)
        X = np.ones((N, l))

        for i in range(1, self.degree + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y ** k)

        self.X = X

    def _franke_function(self, x, y):
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        self.z = np.ravel(term1 + term2 + term3 + term4)

    def set_new_parameters(self, model_name, nr_samples, degree, seed):
        self.model = model_name
        self.nr_samples = nr_samples
        self.degree = degree
        self.seed = seed
