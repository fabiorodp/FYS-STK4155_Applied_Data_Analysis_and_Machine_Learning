# FYS-STK4155 - H2020 - UiO
# Project 1 - Part D
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from package.create_dataset import CreateData
from package.linear_models import Ridge
from package.studies import GridSearch
from package.studies import BiasVarianceTradeOff
from package.studies import CrossValidationKFolds


nr_samples = 20
poly_degrees = np.arange(2, 12)
lambda_ = [0, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5, 1]

# ################################################### GridSearch:
gs = GridSearch(data=CreateData, model=Ridge, random_state=10)

mse_train, mse_test, r2_train, r2_test = \
    gs.run(nr_samples=nr_samples, poly_degrees=poly_degrees,
           lambda_=lambda_, test_size=0.2, scale=True,
           terrain=None, plot_results=True, print_results=True)

# plotting result for nr_samples=20 and poly_degrees from 2 to 12:
plt.plot(poly_degrees, mse_train[2, :], "--", label='MSE Train')
plt.plot(poly_degrees, mse_test[2, :], label='MSE Test')
plt.xlabel("Complexity: Polynomial degrees")
plt.ylabel("MSE scores")
plt.title("MSE train and MSE test metrics.")
plt.legend()
plt.show()

# ########## plotting bias-variance trade-off with bootstrapping:
ff = BiasVarianceTradeOff(data=CreateData, model=Ridge,
                          random_state=10)

error, bias, variance = \
    ff.run(nr_samples=20, poly_degrees=poly_degrees,
           lambda_=0.001, n_boostraps=100, test_size=0.2,
           scale=True, terrain=None, verboose=True,
           plot=True)

# #################################### k-folds cross-validation:
k = np.arange(5, 11)
kf_avg = []

for i in k:
    kf_cv = CrossValidationKFolds(data=CreateData, model=Ridge,
                                  random_state=10)

    kf_avg.append(kf_cv.run(
        nr_samples=20, poly_degrees=poly_degrees, lambda_=0.001,
        k=i, shuffle=True, plot=False))

# ######################## printing testing MSE for each k-fold:
print("Testing MSE for our k-folds CV: ", kf_avg)

# ############################## plotting lines for each k-fold:
signs = ["-", "--", "-.", "x", "+", "|"]
labels = ["5-fold Testing MSE", "6-fold Testing MSE",
          "7-fold Testing MSE", "8-fold Testing MSE",
          "9-fold Testing MSE", "10-fold Testing MSE"]

for e, sign, label in zip(kf_avg, signs, labels):
    plt.plot(poly_degrees, e, sign, label=label)

plt.ylabel("Training MSE")
plt.xlabel("Complexity: Polynomial degrees")
plt.title("Our {}-folds cross-validation".format(k))
plt.legend()
plt.show()

# heat-map with poly_degrees x k-folds testing MSE results:
sns.heatmap(data=kf_avg, xticklabels=poly_degrees,
            yticklabels=k, annot=True)
plt.xlabel("Polynomial degrees")
plt.ylabel("k-folds")
plt.show()
