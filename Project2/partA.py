# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
partA.py
~~~~~~~~~~

A script to perform the exercise Part A of Project 2.
"""

from Project2.package.linear_models import OLS, Ridge, LassoModel
from Project2.package.create_dataset import CreateData
from Project2.package.gradient_descent import MiniSGD
from Project2.package.grid_search import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

"""SGDRegressor(max_iter=epochs, penalty='l2', eta0=eta,
                alpha=lambda_, shuffle=True, fit_intercept=False,
                random_state=seed, learning_rate='constant')"""


# ################################## generating data from GeoTIF image
data = CreateData(random_state=10)

X, y = data.fit(nr_samples=20,
                degree=5,
                terrain_file='Project2/data/SRTM_data_Norway_1.tif',
                plot=False)

# ################################################## splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=None, random_state=10)

# ################################################## grid-search SGD
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
epochs = [50, 100, 500, 1000, 5000]
batch_sizes, lambdas = 1, 0

gs = GridSearch(model=MiniSGD(random_state=10),
                params='ETASxEPOCHS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)

# ################################################## grid-search MiniSGD
# ################################################## tuning time performance
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
batch_sizes = [1, 5, 10, 15, 20, 25]
epochs = 100
lambdas = 0

gs = GridSearch(model=MiniSGD(random_state=10),
                params='ETASxBATCHES',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
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

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)

# ################################################## grid-search MiniSGD
# ################################################## regularization l1 - Lasso
# ################################################## tuning metric performance
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
lambdas = [10 ** -2, 0.0015, 10 ** -3, 0.00015, 0]
epochs = 5000       # Benchmark metrics achieved on previous Grid-Search
batch_sizes = 5     # Benchmark metrics achieved on previous Grid-Search

gs = GridSearch(model=MiniSGD(regularization='l1', random_state=10),
                params='ETASxLAMBDAS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, batch_sizes=batch_sizes,
       plot_results=True, verbose=True)
