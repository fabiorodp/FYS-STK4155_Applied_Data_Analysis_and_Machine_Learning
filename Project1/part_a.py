# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
from DesignMatrix import design_matrix
from FrankeFunction import franke_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegressionTechniques
from Metrics import r2, mse


# seeding the random variables
np.random.seed(10)

# defining the number of samples (n) and
# the polynomial degrees
n, degree = 100, 2

# generating the explanatory variables x and y
x = np.random.random(n)                 # shape (n,)
y = np.random.random(n)                 # shape (n,)

# generating repeated values matrix x, y
# to obtain all possible combinations
x, y = np.meshgrid(x, y)                # shape (n,n)
X = design_matrix(x, y, degree)         # shape (n*n, p)

# generating the response variables (z),
# from the explanatory variables,
# using FrankeFunction
z = np.ravel(franke_function(x, y))     # shape (n*n,)

# generation a random normally distributed noise
epsilon = np.random.normal(0, 1, n*n)

# adding the noise to FrankeFunction
z += epsilon

# splitting the data in test and training data
X_train, X_test, z_train, z_test = \
    train_test_split(X, z, test_size=0.2)

# scaling X_train and X_test
scaler = StandardScaler()
X_test_scaled = scaler.fit(X_train).transform(X_test)
X_train_scaled = scaler.fit(X_train).transform(X_train)

# del unnecessary objects stored
# performance purposes
del x, y, n, degree, X, z, epsilon, X_test, X_train, scaler

# finding betas by OLS
# call the LinearRegressionTechniques
# and fit data in the OLS model
lr = LinearRegressionTechniques(technique_name="OLS")
lr.fit(X_train_scaled, z_train)

# getting coefficients (betas)
print("Beta_hat :", lr.coef_())

# accessing the confidence interval of the coefficients
# print(lr.coef_confidence_interval())

# predicting the response variable
# and accessing the accuracies
z_hat = lr.predict()
print("Training R2: ", r2(z_train, z_hat))
print("Training MSE: ", mse(z_train, z_hat))

z_tilde = lr.predict(X_test_scaled)
print("Testing R2: ", r2(z_test, z_tilde))
print("Testing MSE: ", mse(z_test, z_tilde))

# # # # # # # # # # # # # # # # SKLEARN - Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

lr1 = LinearRegression()
lr1.fit(X_train_scaled, z_train)

print("###### SKLEARN")
print("Beta_hat: ", lr1.coef_)

z_hat = lr1.predict(X_train_scaled)
print("R2 score: ", r2_score(z_train, z_hat))
print("MSE: ", mean_squared_error(z_train, z_hat))

z_tilde = lr1.predict(X_test_scaled)
print("R2 score: ", r2_score(z_test, z_tilde))
print("MSE: ", mean_squared_error(z_test, z_tilde))
