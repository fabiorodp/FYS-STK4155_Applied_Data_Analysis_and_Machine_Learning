# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from Project2.package.activation_functions import identity, identity_prime
from Project2.package.activation_functions import sigmoid, sigmoid_prime
from Project2.package.activation_functions import tanh, tanh_prime
from Project2.package.activation_functions import relu, relu_prime
from Project2.package.activation_functions import softmax, softmax_prime
from Project2.package.cost_functions import mse, mse_prime
from Project2.package.cost_functions import crossentropy, crossentropy_prime
from Project2.package.cost_functions import accuracy_score
from Project2.package.cost_functions import accuracy_score_prime


class MLP:
    """A module to implement a deep neural network so-called multi-layer
    perceptron (MLP)."""

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

        elif act_function == 'softmax':
            return softmax, softmax_prime

        else:
            raise ValueError("Error: Activation function not implemented.")

    @staticmethod
    def set_cost_function(cost_function):
        """
        Cost function algorithms.

        :param cost_function: string: The name of the cost function.
        """
        if cost_function == 'accuracy_score':
            return accuracy_score, accuracy_score_prime

        elif cost_function == 'mse':
            return mse, mse_prime

        elif cost_function == 'crossentropy':
            return crossentropy, crossentropy_prime

        else:
            raise ValueError("Error: Output activation function not "
                             "implemented.")

    def __init__(self, hidden_layers=[50], epochs=1000, batch_size=100,
                 eta0=0.01, learning_rate='constant', decay=0.0, lmbd=0.0,
                 bias0=0.01, init_weights='normal', act_function='sigmoid',
                 output_act_function='identity', cost_function='mse',
                 random_state=None, verbose=False):
        """
        Constructor of the class.

        Parameters:
        ~~~~~~~~~~
        :param hidden_layers: list: It contains the number of neurons in the
                                    respective hidden layers of the network.
                                    For example, if [5, 4], then it would
                                    be a 2-hidden-layer network, with the
                                    first hidden layer containing 5 and
                                    second 4 neurons.
        :param epochs: int: Number of interactions.
        :param batch_size: int: Number of mini-batches.
        :param eta0: float: Learning rate.
        :param learning_rate: str: Type of learning rate that can be
                                   'constant' or 'decay'.
        :param decay: float: Eta's decay that is between 0 and 1.
        :param lmbd: float: "L2" regularization.
        :param bias0: float: Bias to be added to the weights.
        :param init_weights: str: The ways of initialization of the weights
                                  and biases. It can be 'constant' or
                                  'xavier'.
        :param act_function: str: Activation function name for the hidden
                                  layers. It can be 'identity', 'sigmoid',
                                  'tanh' or 'relu'.
        :param output_act_function: str: Activation function name for the
                                         output layers. It can be 'identity',
                                         'sigmoid', 'tanh' or 'relu'.
        :param cost_function: str: Cost function name. It can be 'mse',
                                   'accuracy_score', 'crossentropy' or
                                   'softmax'.
        :param random_state: int: The seed for random numbers.
        :param verbose: bool: True to print training costs for every epoch.
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layers
        self.epochs = epochs
        self.costs = np.zeros(self.epochs)
        self.batch_size = batch_size
        self.eta0 = eta0 / batch_size
        self.decay = decay
        self.learning_rate = learning_rate
        self.lmbd = lmbd
        self.bias0 = bias0
        self.init_weights = init_weights
        self.verbose = verbose
        self.a_funct = act_function

        # setting activation function for hidden layers
        self.act_function, self.act_function_prime = \
            self.set_act_function(act_function)

        # setting activation function for output layers
        self.output_act_function, self.output_act_func_prime = \
            self.set_act_function(output_act_function)

        # setting cost function
        self.cost_function, self.cost_function_prime = \
            self.set_cost_function(cost_function)

        # parameters for each layer
        self.weights, self.biases, self.a, self.net_input, self.delta = \
            [None], [None], [None], [None], [None]

    def _eta(self, epoch):
        """Update learning rates (etas values) for every epoch."""
        eta = self.eta0 * (1.0 / (1.0 + self.decay * epoch))
        return eta

    def _init_parameters(self, X, y):
        """Initialization of parameters."""
        self.n_inputs, self.n_features = X.shape
        self.n_categories = y.shape[1]
        self.n_batches = self.n_inputs // self.batch_size
        self.layers = ([self.n_features] + self.hidden_layer_sizes +
                       [self.n_categories])
        self.n_layers = len(self.layers)
        self.n_iterations = self.n_inputs // self.batch_size

        for l in range(1, self.n_layers):
            # Initializing weights using std normal distribution
            if self.init_weights == 'normal':
                self.weights.append(
                    np.random.randn(self.layers[l - 1], self.layers[l]))

            # Initializing weights using xavier method
            elif self.init_weights == 'xavier':

                if self.a_funct == 'sigmoid':
                    self.weights.append(
                        np.random.normal(
                            loc=0.0, scale=np.sqrt(2. / (self.layers[l - 1]
                                                         + self.layers[l])),
                            size=(self.layers[l - 1], self.layers[l])))

                elif self.a_funct == 'tanh':
                    self.weights.append(
                        np.random.normal(
                            loc=0.0, scale=4*np.sqrt(2. / (self.layers[l - 1]
                                                         + self.layers[l])),
                            size=(self.layers[l - 1], self.layers[l])))

                elif self.a_funct == 'relu':
                    self.weights.append(
                        np.random.normal(
                            loc=0.0, scale=np.sqrt(2)*np.sqrt(2. / (
                                    self.layers[l - 1] + self.layers[l])),
                            size=(self.layers[l - 1], self.layers[l])))

            self.biases.append(np.zeros(self.layers[l]) + self.bias0)
            self.net_input.append(None)
            self.a.append(None)
            self.delta.append(None)

    def fit(self, X, y, plot_costs=False):
        """
        Class method to feed the model with the training data.

        :param X: ndarray: Explanatory data-set variables.
        :param y: ndarray: Target or response labeled variables.
        :param plot_costs: bool: True to line-plot the training costs.
        """
        # initializing parameters
        self._init_parameters(X, y)

        # each interaction/epoch
        for epoch in range(self.epochs):
            j = 0
            idxs = np.arange(self.n_inputs)
            np.random.shuffle(idxs)

            # computing eta decay for every epoch
            if self.learning_rate == 'decay':
                self.eta0 = self._eta(epoch) / self.n_batches

            # Stochastically Mini-batches
            for batch in range(self.n_batches):
                rand_idxs = \
                    idxs[j * self.batch_size:(j + 1) * self.batch_size]
                Xi, yi = X[rand_idxs, :], y[rand_idxs]
                self._feed_forward(Xi)
                self._backpropagation(yi)
                j += 1

            # saving accuracy scores
            self.costs[epoch] = self.cost

            # printing accuracy of each epoch
            if self.verbose is True:
                print(f'Epoch {epoch + 1} of {self.epochs}   |   '
                      f'Accuracy loss: {self.cost}')

        # plotting accuracy scores for each epoch
        if plot_costs is True:
            plt.plot(np.arange(self.epochs), self.costs)
            plt.title("Epochs x Accuracy score")
            plt.xlabel("Epoch numbers")
            plt.ylabel("Accuracy scores")
            plt.tight_layout()
            plt.show()

    def _feed_forward(self, Xi):
        """Class method to perform feed-forward for training."""
        self.a[0] = Xi
        for l in range(1, self.n_layers):
            self.net_input[l] = \
                self.a[l - 1] @ self.weights[l] + self.biases[l]

            self.a[l] = self.act_function(y_hat=self.net_input[l])

        # Overwriting last output with the chosen output function
        self.a[-1] = self.output_act_function(self.net_input[-1])

    def _backpropagation(self, yi):
        """Class method to perform back-propagation technique."""
        self.cost = self.cost_function(y_hat=self.a[-1],
                                       y_true=yi)

        self.delta[-1] = self.cost_function_prime(y_hat=self.a[-1],
                                                  y_true=yi)

        # self.delta[-1] = self.a[-1] - yi

        # computing gradients for weight and biases
        dw = self.a[-2].T @ self.delta[-1]
        db = np.sum(self.delta[-1], axis=0)

        # 'l2' ridge regularization
        if self.lmbd > 0.0:
            dw += self.lmbd * self.weights[-1]

        # updating weights and biases
        self.weights[-1] -= self.eta0 * dw
        self.biases[-1] -= self.eta0 * db

        for l in range(self.n_layers - 2, 0, -1):
            self.delta[l] = (self.delta[l + 1] @ self.weights[l + 1].T *
                             self.act_function_prime(y_hat=self.net_input[l]))

            # weights' gradient for hidden layers
            dw = self.a[l - 1].T @ self.delta[l]

            # 'l2' ridge regularization
            if self.lmbd > 0.0:
                dw += self.lmbd * self.weights[l]

            # update weights and biases
            self.weights[l] -= self.eta0 * dw
            self.biases[l] -= self.eta0 * np.sum(self.delta[l], axis=0)

    def _feed_forward_out(self, X):
        """Class method to perform feed-forward for predictions."""
        a, net_input = X, None
        for l in range(1, self.n_layers):
            net_input = a @ self.weights[l] + self.biases[l]
            a = self.act_function(y_hat=net_input)

        a = self.output_act_function(y_hat=net_input)
        return a

    def predict_class(self, X):
        """Class method to predict classification targets."""
        output = self._feed_forward_out(X)
        return np.argmax(output, axis=1)

    def predict(self, X):
        """Class method to predict regression targets."""
        return self._feed_forward_out(X)
