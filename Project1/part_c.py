# FYS-STK4155 - H2020 - UiO
# Project 1 - part c
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from package.Create_data import CreateData
from sklearn.linear_model import LinearRegression
from package.Bootstrap import Bootstrap, BootstrapMLextend
from package.CrossValidation import CrossValidation
from package.LinearRegression import OlsModel


# calling model that generates data
cd = CreateData(
    model_name="FrankeFunction", seed=10,
    nr_samples=150, degree=None
)

# bootstrapping as function of the polynomial degree
bp = Bootstrap(CreateData=cd, ML_Model=OlsModel(),
               n_boostraps=100, seed=10,
               function_of="poly_degrees")
bp.fit(complexities=np.arange(2, 10), verbose=False)
bp.plot()

# calling model that perform bootstrap bias-variance trade-off
# as function of poly-degree
bt = BootstrapMLextend(
    CreateData=cd, ML_Model=LinearRegression(),
    function_of="poly_degrees", num_rounds=100,
    loss="mse", random_seed=10)

bt.fit(complexities=np.arange(2, 10))
bt.plot()

# calling model that perform cross validation, MSE study
# as function of poly-degree
bt = CrossValidation(
    CreateData=cd, ML_Model=LinearRegression(),
    function_of="poly_degrees", random_seed=10)

bt.fit(complexities=np.arange(2, 10), k=5)
bt.plot()
