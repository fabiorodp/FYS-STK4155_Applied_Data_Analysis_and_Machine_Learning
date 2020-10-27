# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
partA.py
~~~~~~~~~~

A script to perform the exercise Part A of Project 2.
"""

from Project1.package.create_dataset import CreateData
from Project1.package.linear_models import OLS, Ridge
from Project2.package.cost_functions import mse
from Project2.package.gradient_descent import MiniSGD
from Project2.package.grid_search import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# ################################################## re-cap Project 1
# ################################################## best OLS and Ridge
degree = [10, 11]
terrain_file = 'Project2/data/SRTM_data_Norway_1.tif'
labels = ['Testing MSE for OLS model:', 'Testing MSE for Ridge model:']
models = [OLS(random_state=10), Ridge(lambda_=0.0001, random_state=10)]

for d, l, model in zip(degree, labels, models):
    cd = CreateData(random_state=10)
    X, z = cd.fit(nr_samples=15,
                  degree=d,
                  terrain_file='Project2/data/SRTM_data_Norway_1.tif')

    # splitting data
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

    # scaling X data
    scaler = StandardScaler(with_mean=False,
                            with_std=True)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)

    # # scaling z data
    # scale = StandardScaler()
    # scale.fit(z_train)
    # z_train = scale.transform(z_train)
    # z_test = scale.transform(z_test)

    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)

    print(f'{l}', mse(y_true=z_test, y_hat=y_hat))

    # Testing MSE for OLS model: 1.116780111855389
    # Testing MSE for Ridge model: 1.4407904605093536
# ##################################################

# ################################## pre-processing GeoTIF Image
# ################################## for SGD and Mini-SGD
cd = CreateData(random_state=10)
X, z = cd.fit(nr_samples=15,
              degree=10,
              terrain_file='Project2/data/SRTM_data_Norway_1.tif')

# splitting data
X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

# # scaling X data
# scaler = StandardScaler(with_mean=False,
#                         with_std=True)
# scaler.fit(X_train)
# X_test = scaler.transform(X_test)
# X_train = scaler.transform(X_train)
#
# # scaling z data
# scale = StandardScaler()
# scale.fit(z_train)
# z_train = scale.transform(z_train)
# z_test = scale.transform(z_test)

# ################################################## grid-search SGD
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
epochs = [100, 500, 1000, 5000, 10000]
batch_sizes, lambdas = 1, 0

gs = GridSearch(model=MiniSGD(random_state=10),
                params='ETASxEPOCHS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)

# ################################################## grid-search MiniSGD
# ################################################## tuning time performance
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
batch_sizes = [1, 5, 10, 15, 20, 25]
epochs = 5000
lambdas = 0

gs = GridSearch(model=MiniSGD(random_state=10),
                params='ETASxBATCHES',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)

# ################################################## grid-search MiniSGD
# ################################################## regularization l2 - Ridge
# ################################################## tuning metric performance
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
lambdas = [10 ** -2, 0.0015, 10 ** -3, 0.00015, 0]
epochs = 5000       # Benchmark metrics achieved on previous Grid-Search
batch_sizes = 5     # Benchmark metrics achieved on previous Grid-Search

gs = GridSearch(model=MiniSGD(regularization='l2', random_state=10),
                params='ETASxLAMBDAS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)
