# FYS-STK4155 - H2020 - UiO
# Project 1 - test Linear Regression
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from Create_data import CreateData
from LinearRegression import OlsModel, RidgeModel, LassoModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pytest


def test_OslModel():
    """Testing if the coefficients are equal from sklearn and our model."""
    for degree in [2, 3, 4, 5]:
        # creating data
        cd = CreateData(model_name="FrankeFunction", seed=10, nr_samples=100, degree=degree)
        cd.fit()
        X, y = cd.get()

        # fitting sklearn LR model
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        lr.fit(X, y)
        sklearn_betas = lr.coef_.tolist()
        sklearn_y_predict = lr.predict(X)

        # fitting our LR model
        oslmodel = OlsModel()
        oslmodel.fit(X, y)
        oslmodel_betas = oslmodel.coef_.tolist()
        oslmodel_y_predict = oslmodel.predict(X)

        assert sklearn_betas == pytest.approx(oslmodel_betas)
        assert sklearn_y_predict == pytest.approx(oslmodel_y_predict)


def test_Lasso():
    """Testing if the coefficients are equal from sklearn and our model."""
    # creating data
    cd = CreateData(model_name="FrankeFunction", seed=10, nr_samples=5, degree=2)
    cd.fit()
    X, y = cd.get()

    # fitting sklearn Lasso model
    lr = Lasso(alpha=1, fit_intercept=False, random_state=10)
    lr.fit(X, y)
    sklearn_betas = lr.coef_.tolist()
    sklearn_y_predict = lr.predict(X)

    # fitting our LS model
    lsmodel = LassoModel(alpha=1, fit_intercept=False, random_state=10)
    lsmodel.fit(X, y)
    lsmodel_betas = lsmodel.coef_.tolist()
    lsmodel_y_predict = lsmodel.predict(X)

    assert sklearn_betas == pytest.approx(lsmodel_betas)
    assert sklearn_y_predict == pytest.approx(lsmodel_y_predict)
