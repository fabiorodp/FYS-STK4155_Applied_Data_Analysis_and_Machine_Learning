# FYS-STK4155 - H2020 - UiO
# Project 1 - Part F and G
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from package.create_dataset import CreateData
from package.linear_models import OLS, Ridge, LassoModel
from package.studies import GridSearch
from package.studies import BiasVarianceTradeOff
from package.studies import CrossValidationKFolds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from imageio import imread
from sklearn.linear_model import LinearRegression


cd = CreateData(random_state=10)
X, z = cd.fit(
    nr_samples=900, degree=5,
    terrain_file="data/SRTM_data_Norway_2.tif",
    plot=True)

X_train, X_test, z_train, z_test = \
    train_test_split(
        X, z, test_size=0.2, random_state=10)

scaler = StandardScaler(with_mean=False, with_std=True)
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

lr = OLS(random_state=10)
lr.fit(X_train, z_train)

z_hat = lr.predict(X_train)
z_tilde = lr.predict(X_test)

print(mean_squared_error(z_train, z_hat))
print(mean_squared_error(z_test, z_tilde))

# ########################################### GridSearch:

nr_samples = np.arange(5, 50, 5)
poly_degrees = np.arange(2, 10)

# calling GridSearch
gs = GridSearch(data=CreateData, model=OLS, random_state=10)

# getting results from GridSearch
mse_train, mse_test, r2_train, r2_test = \
    gs.run(nr_samples=nr_samples,
           poly_degrees=poly_degrees,
           lambda_=None, test_size=0.2, scale=True,
           terrain="data/SRTM_data_Norway_1.tif",
           plot_results=True, print_results=True)

# plotting result for nr_samples=20 and poly_degrees from 2 to 9
plt.plot(poly_degrees[:-1], mse_train[2, :-1], "--", label='MSE Train')
plt.plot(poly_degrees[:-1], mse_test[2, :-1], label='MSE Test')
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
    ff.run(nr_samples=8, poly_degrees=poly_degrees,
           lambda_=None, n_boostraps=100, test_size=0.2,
           scale=True,
           terrain="data/SRTM_data_Norway_1.tif",
           verboose=True, plot=True)
