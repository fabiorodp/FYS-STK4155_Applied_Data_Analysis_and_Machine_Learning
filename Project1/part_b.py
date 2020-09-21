# FYS-STK4155 - H2020 - UiO
# Project 1 - part b
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


import numpy as np
from package.Create_data import CreateData
from package.LinearRegression import OlsModel
from package.Bootstrap import Bootstrap
from package.AccuracyStudies import AccuracyStudies


# # # # # # # # # # Plotting Complexity[polydegree] x MSE

# AccuracyStudies as function of nr_samples with MSE metric
acc1 = AccuracyStudies(CreateData(degree=5),
                       OlsModel(), function_of="nr_samples",
                       metric="MSE")
acc1.fit(complexities=np.arange(60, 300, 30), verbose=False)
acc1.plot()
acc1.print_best_metric()

# AccuracyStudies as function of poly_degrees with MSE metric
acc2 = AccuracyStudies(CreateData(nr_samples=120),
                       OlsModel(), function_of="poly_degrees",
                       metric="MSE")
acc2.fit(complexities=np.arange(2, 15), verbose=False)
acc2.plot()
acc2.print_best_metric()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # Plotting Bootstrap study

# bootstrapping as function of the number of samples
bp = Bootstrap(CreateData(degree=5), OlsModel(),
               n_boostraps=100, seed=10,
               function_of="nr_samples")
bp.fit(complexities=np.arange(60, 300, 30), verbose=False)
bp.plot()

# bootstrapping as function of the polynomial degree
bp = Bootstrap(CreateData(nr_samples=120), OlsModel(),
               n_boostraps=100, seed=10,
               function_of="poly_degrees")
bp.fit(complexities=np.arange(2, 15), verbose=False)
bp.plot()
