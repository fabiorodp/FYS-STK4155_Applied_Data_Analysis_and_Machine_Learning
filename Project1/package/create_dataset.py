# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from imageio import imread
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class CreateData:

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, nr_samples, degree=None, terrain_file=None,
            plot=False):

        if terrain_file is None:

            if degree is None:
                raise ValueError("Error: degree not provided.")

            # seeding the random variables
            np.random.seed(self.random_state)

            # generating the explanatory variables x and y
            x = np.random.random(nr_samples)
            y = np.random.random(nr_samples)

            # generating repeated values matrix x, y
            # to obtain all possible combinations
            x, y = np.meshgrid(x, y)                # shape (n,n)
            X = self._design_matrix(x, y, degree)   # shape (n*n, p)

            # generating the response variables (z),
            # from the explanatory variable sets x, y,
            # using FrankeFunction
            z = self._franke_function(x, y).ravel()  # shape (n*n,)

            if plot is True:
                self._plot(x, y, z)

            # generation a random normally distributed noise
            epsilon = np.random.normal(0, 1, z.shape) * 0.15

            # adding the noise to FrankeFunction
            z += epsilon

            if plot is True:
                self._plot(x.flatten(), y.flatten(), z)

            return X, z[:, np.newaxis]

        elif terrain_file is not None:

            terrain = None

            if isinstance(terrain_file, np.ndarray):
                terrain = terrain_file

            # checking if the image exists
            elif os.path.isfile('{}'.format(terrain_file)) is not True:
                raise ValueError('Error: Image does not exit.')

            else:
                # converting image in ndarray
                terrain = imread(terrain_file)

            # setting a region of the terrain dataset
            terrain = terrain[:nr_samples, :nr_samples]

            # creating mesh of image pixels
            x = np.linspace(0, 1, np.shape(terrain)[0])
            y = np.linspace(0, 1, np.shape(terrain)[1])
            x, y = np.meshgrid(x, y)

            # generating design matrix X
            X = self._design_matrix(x, y, degree)

            # generating the response set z
            z = np.asarray(terrain.flatten())

            if plot is True:
                self._plot(x, y, z)

            return X, z[:, np.newaxis]

    @staticmethod
    def _design_matrix(x, y, degree):
        """
        Creates a design matrix (X) from polynomial equations of a
        specific degree.

        :param x: explanatory variable x of shape (n, n).
        :param y: explanatory variable y of shape (n, n).
        :param degree: The degree of the polynomial equations.

        :return X : ndarray: The design matrix (X) of shape (N=n*n, p),
        where p are the number of the columns or quantity of factor or
        complexity of the data.
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree + 1) * (degree + 2) / 2)
        X = np.ones((N, l))

        for i in range(1, degree + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y ** k)

        return X

    @staticmethod
    def _franke_function(x, y):
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    @staticmethod
    def _plot(x, y, z):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(x, y, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
