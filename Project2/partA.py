# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
partA.py
~~~~~~~~~~

A script to perform the exercise Part A of Project 2.
"""

from Project2.package.create_dataset import CreateData
from Project2.package.linear_models import OLS, Ridge, LassoModel
from Project2.package.gradient_descent import BGD, SGD
from Project2.package.grid_search import GridSearch
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

"""SGDRegressor(max_iter=epochs, penalty='l2', eta0=eta,
                alpha=lambda_, shuffle=True, fit_intercept=False,
                random_state=seed, learning_rate='constant')"""


# ################################## generating data from GeoTIF image
data = CreateData(random_state=10)
X, y = data.fit(nr_samples=20, degree=5,
                terrain_file='data/SRTM_data_Norway_1.tif',
                plot=False)

# ################################################## splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=None, random_state=10)

# ################################################## grid-search OLS
etas = [0.15, 10 ** -1, 0.015, 10 ** -2, 0.0015]
epochs = [50, 100, 250, 500, 750]
lambdas = 0

gs = GridSearch(model=SGD(random_state=10), params='ETAxEPOCHS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, plot_results=True,
       verbose=True)

# ################################################## grid-search Ridge
etas = [0.15, 10 ** -1, 0.015, 10 ** -2, 0.0015]
lambdas = [10 ** -2, 0.0015, 10 ** -3, 0.00015, 0]
epochs = 500  # Benchmark metrics achieved on OLS model

# Lasso's regularization
gs = GridSearch(model=SGD(regularization='l2', random_state=10),
                params='ETAxLAMBDA', random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, plot_results=True,
       verbose=True)

# ################################################## grid-search Lasso
etas = [0.15, 10 ** -1, 0.015, 10 ** -2, 0.0015]
lambdas = [10 ** -2, 0.0015, 10 ** -3, 0.00015, 0]
epochs = 500  # Benchmark metrics achieved on OLS model

# Lasso's regularization
gs = GridSearch(model=SGD(regularization='l1', random_state=10),
                params='ETAxLAMBDA', random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
       lambdas=lambdas, etas=etas, epochs=epochs, plot_results=True,
       verbose=True)
