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
from sklearn.metrics import mean_squared_error, r2_score
from imageio import imread
from sklearn.linear_model import LinearRegression

# Load the terrain
terrain1 = imread('data/SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# ########################################################## OLS:
# ############### Extracting a smaller region for our study:
# ########################################### GridSearch:
nr_samples = np.arange(15, 50, 5)
poly_degrees = np.arange(2, 12)

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
plt.plot(poly_degrees, mse_train[0, :], "--", label='MSE Train')
plt.plot(poly_degrees, mse_test[0, :], label='MSE Test')
plt.xlabel("Complexity: Polynomial degrees")
plt.ylabel("MSE scores")
plt.title("MSE train and MSE test, with nr_sample=15, for "
          "different poly_degrees")
plt.legend()
plt.show()

# ########################### plotting bias-variance trade-off:
ff = BiasVarianceTradeOff(data=CreateData, model=OLS,
                          random_state=10)

error, bias, variance = \
    ff.run(nr_samples=15, poly_degrees=poly_degrees,
           lambda_=None, n_boostraps=100, test_size=0.2,
           scale=True,
           terrain="data/SRTM_data_Norway_1.tif",
           verboose=True, plot=True)

# #################################### k-folds cross-validation:
k = np.arange(5, 11)
kf_avg = []

for i in k:
    kf_cv = CrossValidationKFolds(data=CreateData, model=OLS,
                                  random_state=10)

    kf_avg.append(kf_cv.run(
        nr_samples=15, poly_degrees=poly_degrees,
        terrain="data/SRTM_data_Norway_1.tif",
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
plt.title("Behavior of k-folds' MSE-scores as polynomial "
          "degree increases")
plt.show()

# ########################################################## Ridge:

