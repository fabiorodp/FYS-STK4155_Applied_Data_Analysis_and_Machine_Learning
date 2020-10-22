import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

# ################################## generating data from GeoTIF image
terrain = imread('data/SRTM_data_Norway_1.tif')
terrain = terrain[:20, :20]
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x, y = np.meshgrid(x, y)
x = np.asarray(x.flatten())
y = np.asarray(y.flatten())
X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
z = np.asarray(terrain.flatten())[:, np.newaxis]

del x, y, terrain

# ################################################## splitting data
X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

scale = StandardScaler()
scale.fit(z_train)
z_train = scale.transform(z_train)
z_test = scale.transform(z_test)

del X, z, scale


# ################################################## functions
def sigmoid(z):
    r"""The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    r"""Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def mse(y_true, y_hat):
    n = np.size(y_hat)
    return np.sum((y_hat - y_true) ** 2.) / n


def mse_prime(y_true, y_hat):
    n = np.size(y_hat)
    return 2. / n * (y_hat - y_true)

# ##################################################  Deep neural network
# ##################################################  init parameters
lmbd = 0
bias = 1
hidden_layers = [100]
batch_size = 15
eta = 0.01
epochs = 100
costs = np.empty(epochs)
act_function = sigmoid
act_function_prime = sigmoid_prime
out_act_function = lambda x: x
out_act_function_prime = lambda x: np.ones(x.shape)
cost_function = mse
cost_function_prime = mse_prime
random_state = 10
np.random.seed(random_state)

# ######################################## fit's initialization
sample_space, feature_space = X_train.shape[0], X_train.shape[1]
labels_space = z_train.shape[1]
batch_space = sample_space // batch_size

# ######################################## activations and weights
act, net_input = [None], [None]
weights, biases = [None], [None]

# ######################################## Gaussian weights for hidden layers
for idx, n_neurons in enumerate(hidden_layers):  # 50, 10, 5 , 5
    if idx == 0:
        weights.append(np.random.randn(feature_space, n_neurons))
        biases.append(np.zeros((1, n_neurons)) + bias)
        # biases.append(np.zeros((n_neurons)) + bias)

    elif 0 < idx <= len(hidden_layers):  # idx: 1, 2, 3  |  n_: 10, 5, 5
        weights.append(np.random.randn(hidden_layers[idx - 1], n_neurons))
        biases.append(np.zeros((1, n_neurons)) + bias)
        # biases.append(np.zeros((n_neurons)) + bias)

    act.append(None)
    net_input.append(None)

for idx in range(1, labels_space + 1):
    weights.append(np.random.randn(hidden_layers[-1], idx))
    biases.append(np.zeros((1, labels_space)) + bias)
    # biases.append(np.zeros((labels_space)) + bias)

    act.append(None)
    net_input.append(None)

# ######################################## training data
for epoch in range(epochs):
    # epoch = 0  # delete it
    for j in range(batch_space):
        batch_idxs = np.random.choice(sample_space, batch_size, replace=False)

        # ######################################## mini-batches training
        X_mini_batch, z_mini_batch = X_train[batch_idxs], z_train[batch_idxs]

        # ######################################## Feed-forward:
        act[0] = X_mini_batch

        for step in range(len(act)):
            if 0 <= step < len(act)-2:
                net_input[step+1] = \
                    act[step] @ weights[step+1] + biases[step+1]

                act[step+1] = act_function(net_input[step+1])

            elif step == len(act)-2:
                net_input[step+1] = \
                    act[step] @ weights[step+1] + biases[step+1]

                act[step+1] = out_act_function(net_input[step+1])

        # ######################################## Back-propagation:
        deltas = [None for _ in range(len(weights))]
        for step in range(1, len(weights)):

            if step == 1:
                # calculating the total cost/loss/error
                costs[epoch] = cost_function(y_true=z_mini_batch,
                                             y_hat=act[-step])

                # calculating the partial derivatives
                # for cost, act.func and net_input
                cost_prime = cost_function_prime(y_true=z_mini_batch,
                                                 y_hat=act[-step])
                act_func_prime = out_act_function_prime(net_input[-step])
                net_input_prime_w = act[-(step+1)].T
                net_input_prime_b = cost_prime * act_func_prime

                # calculating the gradients
                # for weight and bias
                w_gradient = net_input_prime_w @ (cost_prime * act_func_prime)
                b_gradient = np.sum(net_input_prime_b, axis=0)

                # calculating the regularization "l2"
                if lmbd > 0:
                    w_gradient += lmbd * weights[-step]

                # updating weight and bias
                weights[-step] -= eta * w_gradient
                biases[-step] -= eta * b_gradient

                # saving delta for next step
                deltas[-step] = cost_prime * act_func_prime

            else:
                # calculating the partial derivatives
                cost_prime = deltas[-(step-1)] @ weights[-(step-1)].T
                act_func_prime = act_function_prime(net_input[-step])
                net_input_prime_w = act[-(step+1)].T
                net_input_prime_b = cost_prime * act_func_prime

                # calculating the gradient
                w_gradient = net_input_prime_w @ (cost_prime * act_func_prime)
                b_gradient = np.sum(net_input_prime_b, axis=0)

                # calculating the regularization "l2"
                if lmbd > 0:
                    w_gradient += lmbd * weights[-step]

                # updating weights and biases
                weights[-step] -= eta * w_gradient
                biases[-step] -= eta * b_gradient

                # saving delta for next step
                deltas[-step] = cost_prime * act_func_prime

    print(f'Epoch {epoch+1}/{epochs}  |   '
          f'Total Error: {costs[epoch]}', end='\r')

import matplotlib.pyplot as plt
plt.plot(np.arange(epochs), costs)
plt.show()
