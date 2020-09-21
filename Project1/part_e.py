# FYS-STK4155 - H2020 - UiO
# Project 1 - part e
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from sklearn.linear_model import Lasso
from package.Create_data import CreateData
from package.LinearRegression import LassoModel
from package.Bootstrap import Bootstrap, BootstrapMLextend
from package.CrossValidation import CrossValidation


# bootstrapping as function of lambda
bp = Bootstrap(CreateData(degree=5, nr_samples=100),
               LassoModel(), n_boostraps=100, seed=10,
               function_of="lambda")
bp.fit(complexities=[-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1,
                     1.5, 2, 2.5, 3], verbose=True)
bp.plot()

# bootstrapping as function of the number of samples
bp = Bootstrap(CreateData(degree=5), LassoModel(alpha=1),
               n_boostraps=100, seed=10,
               function_of="nr_samples")
bp.fit(complexities=[i for i in range(10, 300, 50)], verbose=True)
bp.plot()

# bootstrapping as function of the polynomial degree
bp = Bootstrap(CreateData(nr_samples=100), LassoModel(alpha=1),
               n_boostraps=100, seed=10,
               function_of="poly_degrees")
bp.fit(complexities=np.arange(2, 10), verbose=True)
bp.plot()

# calling model that perform bootstrap bias-variance trade-off
# as function of poly-degree
bt = BootstrapMLextend(
    CreateData=CreateData(nr_samples=100), ML_Model=Lasso(alpha=1),
    function_of="poly_degrees", num_rounds=100,
    loss="mse", random_seed=10)
bt.fit(complexities=np.arange(2, 10))
bt.plot()

# calling model that perform cross validation, MSE study
# as function of poly-degree
bt = CrossValidation(
    CreateData=CreateData(nr_samples=100), ML_Model=Lasso(alpha=1),
    function_of="poly_degrees", random_seed=10)

bt.fit(complexities=np.arange(2, 10), k=5)
bt.plot()
