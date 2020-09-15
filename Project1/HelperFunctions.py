# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
from DesignMatrix import design_matrix
from FrankeFunction import franke_function


def create_data(nr_samples, poly_degree, seed):
    # seeding the random variables
    np.random.seed(seed)

    # generating the explanatory variables x and y
    x = np.random.random(nr_samples)             # shape (n,)
    y = np.random.random(nr_samples)             # shape (n,)

    # generating repeated values matrix x, y
    # to obtain all possible combinations
    x, y = np.meshgrid(x, y)                     # shape (n,n)
    X = design_matrix(x, y, poly_degree)         # shape (n*n, p)

    # generating the response variables (z),
    # from the explanatory variables,
    # using FrankeFunction
    z = np.ravel(franke_function(x, y))          # shape (n*n,)

    # generation a random normally distributed noise
    epsilon = np.random.normal(0, 1, nr_samples*nr_samples)

    # adding the noise to FrankeFunction
    z += epsilon

    return X, z