# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
import_data.py
~~~~~~~~~~

A module to import data-sets.
"""

import numpy as np
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def terrain_data(file='data/SRTM_data_Norway_1.tif', slice_size=20,
                 test_size=0.2, shuffle=True, stratify=None, scale_X=True,
                 scale_z=False, random_state=None):
    """Module to import the GeoTIF terrain data."""
    terrain = imread(file)
    terrain = terrain[:slice_size, :slice_size]
    x1 = np.linspace(0, 1, np.shape(terrain)[0])
    x2 = np.linspace(0, 1, np.shape(terrain)[1])
    x1, x2 = np.meshgrid(x1, x2)
    x1 = np.asarray(x1.flatten())
    x2 = np.asarray(x2.flatten())
    X = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
    z = np.asarray(terrain.flatten())[:, np.newaxis]

    # splitting data
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=test_size, shuffle=shuffle, stratify=stratify,
        random_state=random_state)

    # scaling X data
    if scale_X is True:
        scaler = StandardScaler(with_mean=False,
                                with_std=True)
        scaler.fit(X_train)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

    # scaling z data
    if scale_z is True:
        scale = StandardScaler()
        scale.fit(z_train)
        z_train = scale.transform(z_train)
        z_test = scale.transform(z_test)

    return X_train, X_test, z_train, z_test
