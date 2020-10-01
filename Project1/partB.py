# FYS-STK4155 - H2020 - UiO
# Project 1 - Part B
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


from package.create_dataset import CreateData
from package.linear_models import OLS, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from package.studies import GridSearch, BiasVarianceTradeOff

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# choosing parameters
nr_samples = np.arange(10, 50, 5)
poly_degrees = np.arange(2, 10)

# calling GridSearch
gs = GridSearch(data=CreateData, model=OLS, random_state=10)

# getting results from GridSearch
mse_train, mse_test, r2_train, r2_test = \
    gs.run(nr_samples=nr_samples,
           poly_degrees=poly_degrees,
           lambda_=1, test_size=0.2, scale=True, terrain=None,
           plot_results=True, print_results=True)

# plotting result for nr_samples=20 and poly_degrees from 2 to 9
plt.plot(poly_degrees, mse_train[2, :], "--", label='MSE Train')
plt.plot(poly_degrees, mse_test[2, :], label='MSE Test')
plt.xlabel("Complexity: Polynomial degrees")
plt.ylabel("MSE scores")
plt.title("MSE train and MSE test, with nr_sample=12, for "
          "different poly_degrees")
plt.legend()
plt.show()

# plotting bias-variance trade-off
ff = BiasVarianceTradeOff(data=CreateData, model=OLS,
                          random_state=10)

error, bias, variance = \
    ff.run(nr_samples=20, poly_degrees=poly_degrees, lambda_=1,
           n_boostraps=1, test_size=0.2, scale=True, terrain=None,
           verboose=True, plot=True)
