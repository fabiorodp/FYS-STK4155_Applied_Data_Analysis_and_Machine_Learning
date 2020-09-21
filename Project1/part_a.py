# FYS-STK4155 - H2020 - UiO
# Project 1 - part a
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from package.Create_data import CreateData
from package.LinearRegression import OlsModel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


for degree in [2, 3, 4, 5]:

    # generating polynomial with the data
    cd = CreateData(
        model_name="FrankeFunction",
        nr_samples=100,
        degree=degree,
        seed=10
    )
    cd.fit()
    X, z = cd.get()

    # splitting data
    X_train, X_test, z_train, z_test = \
        train_test_split(X, z, test_size=0.2, random_state=10)

    # scaling data
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)

    # calling OlsModel
    model = OlsModel(seed=10)

    # fitting LR model
    model.fit(X_train, z_train)

    # predicting
    z_predict_train = model.predict(X_train)
    z_predict_test = model.predict(X_test)

    # printing coefficients (betas)
    print("Betas for polynomial of {} degree: ".format(degree),
          model.coef_, "\n")

    # print confidence interval for betas
    print("CI for betas: ",
          model.coef_confidence_interval(percentile=0.95), "\n")

    # evaluating training MSE and R2-score
    print("Training MSE: ",
          mean_squared_error(z_train, z_predict_train))
    print("Training R2-score: ",
          r2_score(z_train, z_predict_train), "\n")

    # evaluating test MSE and R2-score
    print("Test MSE: ",
          mean_squared_error(z_test, z_predict_test))
    print("Test R2-score: ",
          r2_score(z_test, z_predict_test), "\n")
