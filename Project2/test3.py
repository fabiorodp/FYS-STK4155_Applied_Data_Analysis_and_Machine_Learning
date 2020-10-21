import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

# ################################## generating data from GeoTIF image
terrain = imread('Project2/data/SRTM_data_Norway_1.tif')
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
costs = [0 for _ in range(epochs)]
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
a, zs = [None], [None]
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
    zs.append(None)

for idx in range(1, labels_space + 1):
    weights.append(np.random.randn(hidden_layers[-1], idx))
    weight_biases.append(np.zeros((1, labels_space)) + bias)

    a.append(None)
    zs.append(None)

# ######################################## training data
for i in range(epochs):
    cost = None
    for j in range(batch_space):
        batch_idxs = np.random.choice(sample_space, batch_size, replace=False)

        # ######################################## mini-batches training
        Xi, zi = X_train[batch_idxs], z_train[batch_idxs]

        # ######################################## Feed-forward:
        a[0] = Xi  # input activation is the mini-batch of X_train

        for idx in range(len(a)):  # 0, 1, 2, 3, 4, 5

            if 0 <= idx < len(a)-2:  # 0, 1, 2, 3, 4
                zs[idx+1] = a[idx] @ weights[idx+1] + weight_biases[idx+1]
                a[idx + 1] = act_function(zs[idx+1])

            elif idx == len(a)-2:  # output | idx: 5
                zs[idx+1] = a[idx] @ weights[idx+1] + weight_biases[idx+1]
                a[idx + 1] = out_act_function(zs[idx+1])

        # ######################################## Back-propagation:
        cost = cost_function(y_true=zi, y_hat=a[-1])
        i = 0  # delete
        costs[i] = cost

        for idx in range(1, len(a)):
            w_gradient, wb_gradient = None, None

            if idx == 1:  # -1
                idx = 1  # delete it
                cost_prime = cost_function_prime(y_true=zi, y_hat=a[-idx])
                z_prime = out_act_function_prime(zs[-idx])
                w_gradient = cost_prime @ z_prime @ a[-(idx+1)]
                wb_gradient = cost_prime @ z_prime

                # regularization "l2"
                if lmbd > 0:
                    w_gradient = w_gradient + lmbd * weights[-idx]

                # updating weights and biases
                weights[-idx] = weights[-idx] - eta * w_gradient
                weight_biases[-idx] = weight_biases[-idx] - eta * wb_gradient

            else:  # -2, -3, -4, -5
                idx = 2  # delete it
                previous_w_gradient = w_gradient
                previous_wb_gradient = wb_gradient
                w_gradient =  # ????
                wb_gradient =  # ????

                # regularization "l2"
                if lmbd > 0:
                    w_gradient = w_gradient + lmbd * weights[-idx]

                # updating weights and biases
                weights[-idx] = weights[-idx] - eta * w_gradient
                weight_biases[-idx] = weight_biases[-idx] - eta * wb_gradient

    print(f'Epoch {i + 1}/{epochs}  |   Cost: {cost}', end='\r')
