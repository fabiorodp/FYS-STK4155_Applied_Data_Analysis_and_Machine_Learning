# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
gradient_descent.py
~~~~~~~~~~

A module to implement the many variants of gradient descent learning such as:
Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD),
Mini-batch Stochastic Gradient Descent (MiniSGD) and Logistic Regression (LR).
"""

import numpy as np
import matplotlib.pyplot as plt
from Project2.package.activation_functions import sigmoid
from Project2.package.cost_functions import crossentropy, crossentropy_prime
from Project2.package.cost_functions import accuracy_score_prime
from sklearn.metrics import accuracy_score


class BGDM:
    """Batch Gradient Descent with momentum."""

    def __init__(self, epochs=1000, eta0=0.1, decay=0.0, lambda_=0.0,
                 gamma=0.0, regularization=None, random_state=None):
        """
        Constructor for the class.

        :param epochs: int: Number of epochs/interactions.
        :param eta0: float: Learning rate.
        :param decay: float: Amount of reduction of the learning rate.
                             Equals to 0 means constant eta value for
                             all epochs.
        :param lambda_: float: Regularization value.
        :param gamma: float: Momentum value between 0 and 1.
        :param regularization: string: It can be "l2" for ridge or
                                                 "l1" lasso.
        :param random_state: int: Seed for the experiment.
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.epochs = epochs
        self.eta0 = eta0
        self.etas = []
        self.decay = decay
        self.lambda_ = lambda_
        self.regularization = regularization
        self.velocity = 0.0
        self.gamma = gamma
        self.coef_ = None

    def fit(self, X, z, plot_etas=False):
        """
        Fitting parameters to regression model.

        :param X: ndarray with the design matrix.
        :param z: ndarray with the labeled response variables.
        :param plot_etas: bool: if True, plot epoch X eta.
        """
        sample_space, feature_space = X.shape[0], X.shape[1]
        self.coef_ = np.random.randn(feature_space, 1)

        for epoch in range(self.epochs):
            gradients = 2.0 / sample_space * X.T @ (X @ self.coef_ - z)

            # regularization
            if self.regularization == "l2":  # Ridge regularization
                gradients += self.lambda_ * self.coef_

            elif self.regularization == "l1":  # Lasso regularization
                gradients += self.lambda_ * self.subgradient_vector()

            # calculating momentum, learning rate and updating weights
            if 0.0 <= self.gamma <= 1.0:
                self.velocity = self.gamma * self.velocity + \
                                self.eta(epoch) * gradients
                self.coef_ -= self.velocity

            else:
                raise ValueError(
                    "Gamma's value must be: 0 <= gamma <= 1.")

        if plot_etas is True:
            self.plot_etas()

    def subgradient_vector(self):
        """Lasso's regularization sub-gradient vector."""
        g = np.zeros(self.coef_.shape)
        for idx, theta in enumerate(self.coef_):

            if theta < 0.0:
                g[idx, 0.0] = -1.0

            elif theta == 0.0:
                g[idx, 0.0] = 0.0

            elif theta > 0.0:
                g[idx, 0.0] = 1.0

        return g

    def set_eta(self, new_eta):
        """
        Changing eta Learning rate.

        :param new_eta: float: New eta value.
        """
        self.eta0 = new_eta

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

    def set_decay(self, new_decay):
        """Set new decay for eta."""
        self.decay = new_decay

    def eta(self, epoch):
        """Update learning rates (etas values) for every epoch."""
        eta = self.eta0 * (1.0 / (1.0 + self.decay * epoch))
        self.etas.append(eta)
        return eta

    def set_gamma(self, new_gamma):
        """Set new parameter gamma."""
        self.gamma = new_gamma

    def plot_etas(self):
        plt.plot(x=range(len(self.etas)), y=self.etas, label='Learning rates')
        plt.title('Epochs x Learning rates with decay')
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate with decay ($\eta$)')
        plt.tight_layout()
        plt.show()


class MiniSGDM(BGDM):
    """Mini-batch Stochastic Gradient Descent with momentum."""

    def __init__(self, batch_size=15, epochs=1000, eta0=0.1, decay=0.0,
                 lambda_=0.0, gamma=0.0, regularization=None,
                 random_state=None):
        """
        Constructor for the class.

        :param lambda: float: Regularization value of the model.
        :param random_state: int: Seed for the experiment.
        """
        super().__init__(epochs, eta0, decay, lambda_, gamma,
                         regularization, random_state)
        self.batch_size = batch_size

    def fit(self, X, z, plot_etas=False):
        """
        Fitting parameters to regression model.

        :param X: ndarray with the design matrix.
        :param z: ndarray with the response variables.
        :param plot_etas: bool: if True, plot epoch X eta.
        """
        sample_space, feature_space = X.shape[0], X.shape[1]
        self.coef_ = np.random.randn(feature_space, 1)
        batch_space = int(sample_space / self.batch_size)

        for epoch in range(self.epochs):
            for _ in range(batch_space):
                batch_idxs = np.random.choice(sample_space,
                                              int(self.batch_size),
                                              replace=False)

                xi, zi = X[batch_idxs], z[batch_idxs]
                gradients = 2.0 * xi.T @ (xi @ self.coef_ - zi)

                # regularization
                if self.regularization == "l2":  # Ridge regularization
                    gradients += self.lambda_ * self.coef_

                elif self.regularization == "l1":  # Lasso regularization
                    gradients += self.lambda_ * self.subgradient_vector()

                # calculating momentum, learning rate and updating weights
                if 0.0 <= self.gamma <= 1.0:
                    if self.decay == 0:
                        self.velocity = self.gamma * self.velocity + \
                                        self.eta0 * gradients

                        self.coef_ -= self.velocity

                    elif 0.0 < self.decay <= 1.0:
                        self.velocity = self.gamma * self.velocity + \
                                        self.eta(epoch) / batch_space * \
                                        gradients

                        self.coef_ -= self.velocity

                    else:
                        raise ValueError(
                            "Decay's value must be: 0 <= decay <= 1.")

                else:
                    raise ValueError(
                        "Gamma's value must be: 0 <= gamma <= 1.")

        if plot_etas is True:
            self.plot_etas()

    def set_batch_size(self, new_batch_size):
        """
        Changing the size of the mini batches.

        :param new_batch_size: float: New mini-batch size.
        """
        self.batch_size = new_batch_size


class LR(MiniSGDM):
    """Logistic Regression with Sigmoid"""

    def __init__(self, batch_size=15, epochs=100, eta0=0.1, decay=0.0,
                 lambda_=0.0, gamma=0.0, regularization=None,
                 random_state=None):
        """
        Constructor for the class.

        :param lambda: float: Regularization value of the model.
        :param random_state: int: Seed for the experiment.
        """
        super().__init__(batch_size, epochs, eta0, decay, lambda_, gamma,
                         regularization, random_state)
        self.score = None
        self.yi = None
        self.z_pred = None

    def fit(self, X, z, plot_etas=False):
        """
        Fitting parameters to regression model.

        :param X: ndarray with the design matrix.
        :param z: ndarray with the response variables.
        :param plot_etas: bool: if True, plot epoch X eta.
        """
        n = len(z)
        n_batches = int(n / self.batch_size)

        # Initializing beta randomly
        self.coef_ = np.random.randn(X.shape[1], 1)

        for epoch in range(self.epochs):
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            j = 0

            # mini-bach with gradient descent
            for batch in range(n_batches):
                rand_idxs = \
                    idxs[j * self.batch_size:(j + 1) * self.batch_size]
                X_i, self.yi = X[rand_idxs, :], z[rand_idxs]

                # computing gradients
                # z_hat = sigmoid(X_i @ self.coef_)
                # gradients = accuracy_score_prime(y_hat=z_hat, y_true=self.yi)

                gradients = X_i.T @ (sigmoid(X_i @ self.coef_) - self.yi)

                # regularization
                if self.regularization == "l2":  # Ridge regularization
                    gradients += self.lambda_ * self.coef_

                elif self.regularization == "l1":  # Lasso regularization
                    gradients += self.lambda_ * self.subgradient_vector()

                self.coef_ -= self.eta0 * gradients
                j += 1

    def predict(self, X, probability=False):
        self.z_pred = sigmoid(X @ self.coef_)

        if probability is False:
            self.z_pred = (self.z_pred > 0.5).astype(np.int)

        return self.z_pred
