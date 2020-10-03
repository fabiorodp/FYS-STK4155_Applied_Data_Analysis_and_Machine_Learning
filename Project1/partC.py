# FYS-STK4155 - H2020 - UiO
# Project 1 - Part C
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from package.create_dataset import CreateData
from package.linear_models import OLS
from package.studies import CrossValidationKFolds
from package.studies import CrossValidationSKlearn
from sklearn.linear_model import LinearRegression

k = np.arange(5, 11)
poly_degrees = np.arange(2, 9)
nr_samples = 20

# #################################### k-folds cross-validation:
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

# ######################## printing testing MSE for each k-fold:
print("Testing MSE for our k-folds CV: ", kf_avg)
print("\n Testing MSE for SkLearn k-folds CV: ", skkf_avf)

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

# ############ plotting Sci-Kit learn k-folds cross-validation:
for e, sign, label in zip(skkf_avf, signs, labels):
    plt.plot(poly_degrees, e, sign, label=label)

plt.ylabel("Training MSE")
plt.xlabel("Complexity: Polynomial degrees")
plt.title("Sklearn {}-folds cross-validation".format(k))
plt.legend()
plt.show()

# heat-map with poly_degrees x k-folds testing MSE results:
sns.heatmap(data=kf_avg, xticklabels=poly_degrees,
            yticklabels=k, annot=True)
plt.xlabel("Polynomial degrees")
plt.ylabel("k-folds")
plt.show()
