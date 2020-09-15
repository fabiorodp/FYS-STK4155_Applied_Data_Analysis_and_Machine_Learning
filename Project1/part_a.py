# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from HelperFunctions import create_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegressionTechniques
from Metrics import r2, mse


def part_a(nr_samples, poly_degrees, seed):
    for poly_degree in poly_degrees:
        X, z = create_data(nr_samples, poly_degree, seed)

        # splitting the data in test and training data
        X_train, X_test, z_train, z_test = \
            train_test_split(X, z, test_size=0.2)

        # scaling X_train and X_test
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # del unnecessary objects stored
        # performance purposes
        del X, z, X_test, X_train, scaler

        # finding betas by OLS
        # call the LinearRegressionTechniques
        # and fit data in the OLS model
        lr = LinearRegressionTechniques(technique_name="OLS")
        lr.fit(X_train_scaled, z_train)

        # getting coefficients (betas)
        print("Beta_hat :", lr.coef_OLS)

        # accessing the confidence interval of the coefficients
        # significance level = 95%
        print("CI: ", lr.coef_CI_OLS)

        # predicting the response variable
        # and accessing the accuracies
        z_train_predict = lr.y_predict
        print("Training R2: ", r2(z_train, z_train_predict))
        print("Training MSE: ", mse(z_train, z_train_predict))

        z_test_predict = lr.predict(X_test_scaled)
        print("Testing R2: ", r2(z_test, z_test_predict))
        print("Testing MSE: ", mse(z_test, z_test_predict))

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


if __name__ == "__main__":
    part_a(nr_samples=5, poly_degrees=[2], seed=10)
