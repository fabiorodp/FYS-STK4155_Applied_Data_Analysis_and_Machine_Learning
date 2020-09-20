# FYS-STK4155 - H2020 - UiO
# Project 1 - test Metrics
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from Create_data import CreateData
from Pre_processing import PreProcessing
from LinearRegression import OlsModel
from sklearn.linear_model import LinearRegression
import pytest

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_split_scale():
    """Testing if the coefficients are equal from sklearn and our model."""
    for degree in [2, 3, 4, 5]:
        # creating data
        cd = CreateData(model_name="FrankeFunction", seed=10, nr_samples=100, degree=degree)
        cd.fit()
        X, y = cd.get()

        pp = PreProcessing(test_size=0.2, seed=10, split=True,
                           scale=True)
        pp.fit(X, y)
        X_train_pp, X_test_pp, z_train_pp, z_test_pp =\
            pp.get()

        X_train, X_test, z_train, z_test = \
            train_test_split(X, y, test_size=0.2, random_state=10)

        scaler = StandardScaler(with_mean=False, with_std=True).fit(X_train)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

        # fitting sklearn LR model
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        lr.fit(X_train, z_train)
        sklearn_betas = lr.coef_.tolist()
        sklearn_y_predict = lr.predict(X_train)

        # fitting our LR model
        oslmodel = OlsModel()
        oslmodel.fit(X_train_pp, z_train_pp)
        oslmodel_betas = oslmodel.coef_.tolist()
        oslmodel_y_predict = oslmodel.predict(X_train_pp)

        assert sklearn_betas == pytest.approx(oslmodel_betas)
        assert sklearn_y_predict == pytest.approx(oslmodel_y_predict)
