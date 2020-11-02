# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from Project2.package.metrics import r2, mse
from Project2.package.import_data import terrain_data
from sklearn.neural_network import MLPRegressor
from Project2.package.deep_neural_network import MLP


X_train, X_test, z_train, z_test = \
    terrain_data(file='Project2/data/SRTM_data_Norway_1.tif', slice_size=15,
                 test_size=0.2, shuffle=True, stratify=None, scale_X=True,
                 scale_z=False, random_state=10)

# model = MLPRegressor(hidden_layer_sizes=(50,), activation='logistic',
#                      solver='sgd', alpha=0.0, batch_size=len(X_train),
#                      learning_rate='constant', learning_rate_init=0.1,
#                      max_iter=100, shuffle=True, random_state=10,
#                      verbose=True, n_iter_no_change=100)
# model.fit(X_train, z_train.ravel())

model = MLP(hidden_layer_sizes=[50, 10], epochs=100, batch_size=len(X_train),
            eta0=0.01, learning_rate='constant', decay=0.0, lmbd=0.0,
            bias0=0.01, init_weights='xavier', act_function='sigmoid',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True)

model.fit(X_train, z_train)

z_hat = model.predict(X_test)
mse_test = mse(y_true=z_test, y_hat=z_hat)
print(f"MSE test: {mse_test}")
