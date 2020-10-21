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
bias = 0.1
hidden_layers = [50, 10, 5, 5]
batch_size = 1
eta = 0.01
epochs = 1000
error = [0 for _ in range(epochs)]
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
a, net_input = [None], [None]
weights, weight_biases = [None], [None]

# ######################################## Gaussian weights for hidden layers
for idx, n_neurons in enumerate(hidden_layers):  # 50, 10, 5 , 5
    if idx == 0:
        weights.append(np.random.randn(feature_space, n_neurons))
        weight_biases.append(np.zeros((1, n_neurons)) + bias)

    elif 0 < idx <= len(hidden_layers):  # idx: 1, 2, 3  |  n_: 10, 5, 5
        weights.append(np.random.randn(hidden_layers[idx - 1], n_neurons))
        weight_biases.append(np.zeros((1, n_neurons)) + bias)

    a.append(None)
    net_input.append(None)

for idx in range(1, labels_space + 1):
    weights.append(np.random.randn(hidden_layers[-1], idx))
    weight_biases.append(np.zeros((1, labels_space)) + bias)

    a.append(None)
    net_input.append(None)

# ######################################## training data
for epoch in range(epochs):
    # epoch = 1  # delete it
    for j in range(batch_space):
        batch_idxs = np.random.choice(sample_space, batch_size, replace=False)

        # ######################################## mini-batches training
        Xi, zi = X_train[batch_idxs], z_train[batch_idxs]

        # ######################################## Feed-forward:
        a[0] = Xi  # input activation is the mini-batch of X_train

        for idx in range(len(a)):  # idx: 0, 1, 2, 3, 4, 5
            if 0 <= idx < len(a)-2:  # idx: 0, 1, 2, 3, 4
                net_input[idx+1] = \
                    a[idx] @ weights[idx+1] + weight_biases[idx+1]

                a[idx + 1] = act_function(net_input[idx+1])

            elif idx == len(a)-2:  # output | idx: 5
                net_input[idx+1] = \
                    a[idx] @ weights[idx+1] + weight_biases[idx+1]

                a[idx + 1] = out_act_function(net_input[idx+1])

        # ######################################## Back-propagation:
        deltas = [None for _ in range(len(weights))]
        for idx in range(1, len(weights)):  # 1, 2, 3, 4, 5
            if idx == 1:  # -1
                # idx = 1  # delete it
                error[epoch] = cost_function(y_true=zi, y_hat=a[-1])
                error_prime = cost_function_prime(y_true=zi, y_hat=a[-idx])
                z_prime = out_act_function_prime(net_input[-idx])
                w_gradient = a[-(idx+1)].T @ error_prime @ z_prime  # delta-1
                deltas[-idx] = w_gradient
                wb_gradient = error_prime @ z_prime

                # regularization "l2"
                if lmbd > 0:
                    w_gradient = w_gradient + lmbd * weights[-idx]

                # updating weights and biases
                weights[-idx] -= eta * w_gradient
                weight_biases[-idx] -= eta * wb_gradient

            else:  # -2, -3, -4, -5
                # idx = 2  # delete it
                erro_prime = deltas[-(idx-1)]
                z_prime = act_function_prime(net_input[-idx])
                w_gradient = a[-(idx+1)].T @ erro_prime @ z_prime
                deltas[-idx] = w_gradient
                wb_gradient = a[-idx] @ deltas[-idx]

                # regularization "l2"
                if lmbd > 0:
                    w_gradient = w_gradient + lmbd * weights[-idx]

                # updating weights and biases
                weights[-idx] -= eta * w_gradient
                weight_biases[-idx] -= eta * wb_gradient

    print(f'Epoch {epoch+1}/{epochs}  |   '
          f'Total Error: {error[epoch]}', end='\r')
