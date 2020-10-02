# FYS-STK4155 - H2020 - UiO
# Project 1 - Part C
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from package.create_dataset import CreateData
from package.linear_models import OLS
from package.studies import CrossValidationKFolds
from package.studies import CrossValidationSKlearn
from sklearn.linear_model import LinearRegression


k = np.arange(5, 11)
poly_degrees = np.arange(2, 10)
nr_samples = 20

kf_avg = []
skkf_avf = []

for i in k:
    kf_cv = CrossValidationKFolds(data=CreateData, model=OLS,
                                  random_state=10)

    kf_avg.append(kf_cv.run(nr_samples=20, poly_degrees=np.arange(2, 10),
                            k=i, shuffle=True, plot=True))

    sk_kf_cv = CrossValidationSKlearn(
        data=CreateData, model=LinearRegression(fit_intercept=False),
        random_state=10)

    skkf_avf.append(sk_kf_cv.run(nr_samples=20, poly_degrees=np.arange(2, 10),
                                 k=i, shuffle=True, plot=True))
