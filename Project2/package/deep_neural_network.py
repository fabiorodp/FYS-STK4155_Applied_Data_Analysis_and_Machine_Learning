# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
deep_neural_network.py
~~~~~~~~~~

A module to implement a deep neural network so-called Multi-layer
perceptron MLP.
"""

import numpy as np
import matplotlib.pyplot as plt
from Project2.package.activation_functions import identity, identity_prime
from Project2.package.activation_functions import sigmoid, sigmoid_prime
from Project2.package.activation_functions import tanh, tanh_prime
from Project2.package.activation_functions import relu, relu_prime
from Project2.package.activation_functions import softmax
from Project2.package.cost_functions import mse, mse_prime
from Project2.package.cost_functions import accuracy_score, crossentropy


class MLP:
    """Multi-layer Perceptron."""

    @staticmethod
    def set_learning_rate(learning_rate):
        """
        Setting a learning rates algorithms.

        :param learning_rate: string: The name of the learning rate.
        """
        if learning_rate == 'constant':
            return learning_rate

        else:
            raise ValueError("Error: Learning rate not implemented.")

    @staticmethod
    def set_act_function(act_function):
        """
        Activation function algorithms.

        :param act_function: string: The name of the activation function.
        """
        if act_function == 'identity':
            return identity, identity_prime

        elif act_function == 'sigmoid':
            return sigmoid, sigmoid_prime

        elif act_function == 'tanh':
            return tanh, tanh_prime

        elif act_function == 'relu':
            return relu, relu_prime

        else:
            raise ValueError("Error: Activation function not implemented.")

    @staticmethod
    def set_cost_function(cost_function):
        """
        Cost function algorithms.

        :param cost_function: string: The name of the cost function.
        """
        if cost_function == 'softmax':
            return softmax

        elif cost_function == 'accuracy_score':
            return accuracy_score

        elif cost_function == 'mse':
            return mse, mse_prime

        elif cost_function == 'crossentropy':
            return crossentropy

        else:
            raise ValueError("Error: Output activation function not "
                             "implemented.")

    def __init__(self, lmbd=0.0, bias=0.1, hidden_layers=[50, 10, 5, 5],
                 batch_size=15, eta=0.01, epochs=1000,
                 act_function='sigmoid', out_act_function='identity',
                 cost_function='mse', random_state=None):
        """
        Constructor of the class.

        Parameters:
        ~~~~~~~~~~
        :param lmbd: float: "L2" regularization.
        :param bias: float: Bias to be added to the weights.
        :param hidden_layers: list: It contains the number of neurons in the
                                    respective hidden layers of the network.
                                    For example, if [5, 4], then it would
                                    be a 2-hidden-layer network, with the
                                    first hidden layer containing 5 and
                                    second 4 neurons.
        :param batch_size: int: Number of mini-batches.
        :param eta: float: Learning rate.
        :param epochs: int: Number of interactions.
        :param act_function: str: Activation function name for the hidden
                                  layers.
        :param out_act_function: str: Activation function name for output.
        :param cost_function: str: Cost function name.
        :param random_state: int: The seed for random numbers.

        Notes:
        ~~~~~~~~~~
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.

        The first layer is assumed to be an input layer, and has not any
        biases for those neurons.
        """

        self.lmbd = lmbd
        self.bias = bias
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.costs = np.empty(epochs)
        self.act_function, self.act_function_prime = \
            self.set_act_function(act_function)
        self.out_act_function, self.out_act_function_prime = \
            self.set_act_function(out_act_function)
        self.cost_function, self.cost_function_prime = \
            self.set_cost_function(cost_function)
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(random_state)

        self.act, self.net_input = [None], [None]
        self.weights, self.biases = [None], [None]

    def _initialize_parameters(self, feature_space, labels_space):
        """Function to initialize parameters for MLP."""
        for idx, n_neurons in enumerate(self.hidden_layers):
            if idx == 0:
                self.weights.append(np.random.randn(feature_space, n_neurons))
                self.biases.append(np.zeros((1, n_neurons)) + self.bias)

            elif 0 < idx <= len(self.hidden_layers):
                self.weights.append(np.random.randn(
                    self.hidden_layers[idx - 1], n_neurons))
                self.biases.append(np.zeros((1, n_neurons)) + self.bias)

            self.act.append(None)
            self.net_input.append(None)

        for idx in range(1, labels_space+1):
            self.weights.append(np.random.randn(self.hidden_layers[-1], idx))
            self.biases.append(np.zeros((1, labels_space)) + self.bias)

            self.act.append(None)
            self.net_input.append(None)

    def _feed_forward(self, X_mini_batch):
        """Function to apply feed-forward."""
        self.act[0] = X_mini_batch

        for step in range(len(self.act)):
            if 0 <= step < len(self.act)-2:
                self.net_input[step+1] = \
                    self.act[step] @ self.weights[step+1] \
                    + self.biases[step+1]

                self.act[step+1] = self.act_function(self.net_input[step+1])

            elif step == len(self.act)-2:
                self.net_input[step+1] = \
                    self.act[step] @ self.weights[step+1] \
                    + self.biases[step+1]

                self.act[step+1] = \
                    self.out_act_function(self.net_input[step+1])

    def _back_propagation(self, epoch, z_mini_batch):
        deltas = [None for _ in range(len(self.weights))]
        for step in range(1, len(self.weights)):

            if step == 1:
                # calculating the total cost/loss/error
                self.costs[epoch] = self.cost_function(
                    y_true=z_mini_batch, y_hat=self.act[-step])

                # calculating the partial derivatives
                # for cost, act.func and net_input
                cost_prime = self.cost_function_prime(
                    y_true=z_mini_batch, y_hat=self.act[-step])
                act_func_prime = self.out_act_function_prime(
                    self.net_input[-step])
                net_input_prime_w = self.act[-(step+1)].T
                net_input_prime_b = cost_prime * act_func_prime

                # calculating the gradients
                # for weight and bias
                w_gradient = net_input_prime_w @ (cost_prime * act_func_prime)
                b_gradient = np.sum(net_input_prime_b, axis=0)

                # calculating the regularization "l2"
                if self.lmbd > 0.0:
                    w_gradient += self.lmbd * self.weights[-step]

                # updating weight and bias
                self.weights[-step] -= self.eta * w_gradient
                self.biases[-step] -= self.eta * b_gradient

                # saving delta for next step
                deltas[-step] = cost_prime * act_func_prime

            else:
                # calculating the partial derivatives
                cost_prime = deltas[-(step-1)] @ self.weights[-(step-1)].T
                act_func_prime = self.act_function_prime(
                    self.net_input[-step])
                net_input_prime_w = self.act[-(step+1)].T
                net_input_prime_b = cost_prime * act_func_prime

                # calculating the gradient
                w_gradient = net_input_prime_w @ (cost_prime * act_func_prime)
                b_gradient = np.sum(net_input_prime_b, axis=0)

                # calculating the regularization "l2"
                if self.lmbd > 0.0:
                    w_gradient += self.lmbd * self.weights[-step]

                # updating weights and biases
                self.weights[-step] -= self.eta * w_gradient
                self.biases[-step] -= self.eta * b_gradient

                # saving delta for next step
                deltas[-step] = cost_prime * act_func_prime

    def fit(self, X, z, plot_costs=False):
        sample_space, feature_space = X.shape[0], X.shape[1]
        labels_space = z.shape[1]
        batch_space = sample_space // self.batch_size

        self._initialize_parameters(feature_space, labels_space)

        for epoch in range(self.epochs):
            for _ in range(batch_space):
                batch_idxs = np.random.choice(sample_space,
                                              self.batch_size,
                                              replace=False)

                # ############################## mini-batches training
                X_mini_batch, z_mini_batch = X[batch_idxs], z[batch_idxs]

                # ############################## feed-forward
                self._feed_forward(X_mini_batch)

                # ############################## back-propagation
                self._back_propagation(epoch, z_mini_batch)

            print(f'Epoch {epoch + 1}/{self.epochs}  |   '
                  f'Total Error: {self.costs[epoch]}', end='\r')

        if plot_costs is True:
            plt.plot(np.arange(self.epochs), self.costs)
            plt.tight_layout()
            plt.show()

    def _feed_forward_out(self, X):
        # a = X
        # for l in range(1, self.n_layers):
        #     z = a @ self.weights[l] + self.biases[l]
        #     a = self.act_func(z)
        #
        # # Overwriting output with chosen output function
        # a = self.output_func(z)
        # return a
        pass

    def predict_class(self, X):
        output = self._feed_forward_out(X)
        return np.argmax(output, axis=1)

    def predict(self, X):
        return self._feed_forward_out(X)
