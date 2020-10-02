# FYS-STK4155 - H2020 - UiO
# Project 1 - Part A
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from package.create_dataset import CreateData
from package.linear_models import OLS
from package.accuracies import r2, mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


for degree in np.arange(2, 6):

    # generating and plotting datasets
    cd = CreateData(random_state=10)
    X, z = cd.fit(nr_samples=12, degree=degree, plot=False)

    # splitting dataset
    X_train, X_test, z_train, z_test = \
        train_test_split(X, z, test_size=0.2, random_state=10)

    # scaling dataset
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)

    # calling OLS
    ols = OLS(random_state=10)
    ols.fit(X_train, z_train)

    # predicting train dataset
    z_hat = ols.predict(X_train)

    # predicting test dataset
    z_tilde = ols.predict(X_test)

    # printing coefficients (betas)
    print("Betas for polynomial of {} degree: "
          .format(degree.ravel()), ols.coef_.ravel(), "\n")

    # print confidence interval for betas
    print("CI for betas: ",
          ols.coef_confidence_interval(percentile=0.95)
          .ravel(), "\n")

    # evaluating training MSE and R2-score
    print("Training MSE: ",
          mse(z_train, z_hat))

    print("Training R2-score: ",
          r2(z_train, z_hat), "\n")

    # evaluating test MSE and R2-score
    print("Test MSE: ",
          mse(z_test, z_tilde))

    print("Test R2-score: ",
          r2(z_test, z_tilde), "\n")
