# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


def terrain_data(file='data/SRTM_data_Norway_1.tif', slice_size=15,
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


def MNIST(test_size=0.2, shuffle=True, stratify=None, scale_X=True,
          verbose=False, plot=False, random_state=None):

    # seeding random numbers or choices
    np.random.seed(random_state)

    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    if verbose is True:
        print("inputs = (n_inputs, pixel_width, pixel_height) = "
              + str(inputs.shape))
        print("labels = (n_inputs) = " + str(labels.shape))

    # flattening the image
    # the value -1 means dimension is inferred from the remaining
    # dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)

    if verbose is True:
        print("X = (n_inputs, n_features) = " + str(inputs.shape))

    if plot is True:
        # choosing some random images to display
        indices = np.arange(n_inputs)
        random_indices = np.random.choice(indices, size=5)

        for i, image in enumerate(digits.images[random_indices]):
            plt.subplot(1, 5, i + 1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title("Label: %d" % digits.target[random_indices[i]])
        plt.show()

    X_train, X_test, Z_train, Z_test = \
        train_test_split(inputs, labels, test_size=test_size,
                         shuffle=shuffle, stratify=stratify,
                         random_state=random_state)

    # scaling X data
    if scale_X is True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

    return X_train, X_test, Z_train[:, np.newaxis], Z_test[:, np.newaxis]


def breast_cancer(test_size=0.2, random_state=None):
    """Load breast cancer data set from Scikit-Learn.
    """
    data = load_breast_cancer()

    # Loading, splitting and scaling data
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_test = scale_data(X_train, X_test, scaler='minmax')

    # One hot encoding targets
    y_train = y_train.reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto')
    y_train_encoded = encoder.fit_transform(y_train).toarray()
    y_test_encoded = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

    return X_train, X_test, y_train_encoded, y_test_encoded


def scale_data(train_data, test_data, scaler='standard'):
    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        print('Scaler must be "standard" or "minmax"!')
        return None

    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    return train_data, test_data

if __name__ == '__main__':
    pass
