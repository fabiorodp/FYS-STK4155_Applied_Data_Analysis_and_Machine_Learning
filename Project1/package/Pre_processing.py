# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class PreProcessing:
    def __init__(self, test_size=None, seed=None, split=True,
                 scale=True):

        self.test_size = test_size
        self.seed = seed
        self.split = split
        self.scale = scale

        self.X_train = None
        self.X_test = None
        self.z_train = None
        self.z_test = None

    def fit(self, X, z):
        if (self.test_size is None) or (self.seed is None):
            raise ValueError("Error: It can not fit if there are any "
                             "'None' parameter. Please set new parameters "
                             "before fit.")

        elif (self.split is True) and (self.scale is True):
            self.split_data(X, z)
            self.scale_data()

        elif (self.split is True) and (self.scale is False):
            self.split_data(X, z)

        else:
            return NotImplemented

    def get(self):
        if self.X_train is None:
            raise ValueError("Not fitted yet")

        else:
            return self.X_train, self.X_test, self.z_train, self.z_test

    def split_data(self, X, z):
        # splitting the data in test and training data
        self.X_train, self.X_test, self.z_train, self.z_test = \
            train_test_split(X, z, test_size=self.test_size,
                             random_state=self.seed)

    def scale_data(self):
        # scaling X_train and X_test
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_train = scaler.transform(self.X_train)

    def set_new_parameters(self, test_size, seed, split=True, scale=True):
        self.test_size = test_size
        self.seed = seed
        self.split = split
        self.scale = scale
