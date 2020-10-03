# FYS-STK4155 - H2020 - UiO
# Project 1 - Part C
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import matplotlib.pyplot as plt
from package.create_dataset import CreateData
from package.linear_models import OLS
from package.studies import CrossValidationKFolds
from package.studies import CrossValidationSKlearn
from sklearn.linear_model import LinearRegression

k = np.arange(5, 11)
poly_degrees = np.arange(2, 9)
nr_samples = 20

kf_avg = []
skkf_avf = []

for i in k:
    kf_cv = CrossValidationKFolds(data=CreateData, model=OLS,
                                  random_state=10)

    kf_avg.append(kf_cv.run(
        nr_samples=20, poly_degrees=np.arange(2, 9),
        k=i, shuffle=True, plot=False))


    sk_kf_cv = CrossValidationSKlearn(
        data=CreateData, model=LinearRegression(fit_intercept=False),
        random_state=10)

    skkf_avf.append(
        sk_kf_cv.run(nr_samples=20, poly_degrees=np.arange(2, 9),
                     k=i, shuffle=True, plot=False))

print("Testing MSE for our k-folds CV: ", kf_avg)
print("\n Testing MSE for SkLearn k-folds CV: ", skkf_avf)

plt.plot(poly_degrees, kf_avg[0], "-", label="5-fold Testing MSE")
plt.plot(poly_degrees, kf_avg[1], "--", label="6-fold Testing MSE")
plt.plot(poly_degrees, kf_avg[2], "-.", label="7-fold Testing MSE")
plt.plot(poly_degrees, kf_avg[3], "x", label="8-fold Testing MSE")
plt.plot(poly_degrees, kf_avg[4], "+", label="9-fold Testing MSE")
plt.plot(poly_degrees, kf_avg[5], "|", label="10-fold Testing MSE")
plt.ylabel("Training MSE")
plt.xlabel("Complexity: Polynomial degrees")
plt.title("Our {}-folds cross-validation".format(k))
plt.legend()
plt.show()

plt.plot(poly_degrees, skkf_avf[0], "-", label="5-fold Testing MSE")
plt.plot(poly_degrees, skkf_avf[1], "--", label="6-fold Testing MSE")
plt.plot(poly_degrees, skkf_avf[2], "-.", label="7-fold Testing MSE")
plt.plot(poly_degrees, skkf_avf[3], "x", label="8-fold Testing MSE")
plt.plot(poly_degrees, skkf_avf[4], "+", label="9-fold Testing MSE")
plt.plot(poly_degrees, skkf_avf[5], "|", label="10-fold Testing MSE")
plt.ylabel("Training MSE")
plt.xlabel("Complexity: Polynomial degrees")
plt.title("Sklearn {}-folds cross-validation".format(k))
plt.legend()
plt.show()
