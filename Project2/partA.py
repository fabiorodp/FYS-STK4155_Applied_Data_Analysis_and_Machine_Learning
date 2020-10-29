# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

"""
partA.py
~~~~~~~~~~

A script to perform the exercise Part A of Project 2.
"""

import pandas as pd
from Project1.package.create_dataset import CreateData
from Project1.package.linear_models import OLS, Ridge
from Project2.package.cost_functions import mse
from Project2.package.gradient_descent import MiniSGDM
from Project2.package.grid_search import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ################################################## re-cap Project 1
# ################################################## best OLS and Ridge
degree = [10, 11]
terrain_file = 'Project2/data/SRTM_data_Norway_1.tif'
labels = ['Testing MSE for OLS model:', 'Testing MSE for Ridge model:']
models = [OLS(random_state=10), Ridge(lambda_=0.0001, random_state=10)]

for d, l, model in zip(degree, labels, models):
    cd = CreateData(random_state=10)
    X, z = cd.fit(nr_samples=15,
                  degree=d,
                  terrain_file='Project2/data/SRTM_data_Norway_1.tif')

    # splitting data
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

    # scaling X data
    scaler = StandardScaler(with_mean=False,
                            with_std=True)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)

    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)

    print(f'{l}', mse(y_true=z_test, y_hat=y_hat))

    # Testing MSE for OLS model: 1.116780111855389
    # Testing MSE for Ridge model: 1.4407904605093536
# ##################################################

# ################################## pre-processing GeoTIF Image
# ################################## for SGD and Mini-SGDM
cd = CreateData(random_state=10)
X, z = cd.fit(nr_samples=15,
              degree=10,
              terrain_file='Project2/data/SRTM_data_Norway_1.tif')

# splitting data
X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

# ################################################## grid-search SGD
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
epochs = [100, 500, 1000, 5000, 10000]
batch_sizes, gammas, lambdas, decays = 1.0, 0.0, 0.0, 0.0

gs = GridSearch(model=MiniSGDM(random_state=10),
                params='ETASxEPOCHS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       epochs=epochs, etas=etas, batch_sizes=batch_sizes, lambdas=lambdas,
       decays=decays, gammas=gammas, plot_results=True, verbose=True)

# ################################################## grid-search Mini-SGDM
# ################################################## tuning time & performance
etas = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [1, 5, 10, 15, 20]
gammas, lambdas, decays, epochs = 0.0, 0.0, 0.0, 5000

gs = GridSearch(model=MiniSGDM(random_state=10),
                params='ETASxBATCHES',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       epochs=epochs, etas=etas, batch_sizes=batch_sizes, lambdas=lambdas,
       decays=decays, gammas=gammas, plot_results=True, verbose=True)

# ################################################## tuning eta-decay
etas = [0.01, 0.005, 0.001, 0.0005]
decays = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
gammas, lambdas, epochs, batch_sizes = 0.0, 0.0, 5000, 10

gs = GridSearch(model=MiniSGDM(random_state=10),
                params='ETASxDECAYS',
                random_state=10)

gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
       epochs=epochs, etas=etas, batch_sizes=batch_sizes, lambdas=lambdas,
       decays=decays, gammas=gammas, plot_results=True, verbose=True)

# ################################################## tuning gamma
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lmbd, decay, epochs, batch_size, eta0 = 0, 0.01, 5000, 10, 0.005
testing_mse = []

for gamma in gammas:
    model = MiniSGDM(
        batch_size=batch_size, epochs=epochs, eta0=eta0, decay=decay,
        lambda_=lmbd, gamma=gamma, regularization='l2', random_state=10)
    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)
    testing_mse.append(mse(y_true=z_test, y_hat=y_hat))

df = pd.DataFrame(testing_mse, columns=['With Momentum gamma'])
df['Bench-marked score from before'] = [3.4 for _ in testing_mse]
df.index = gammas
sns.lineplot(data=df)
plt.xlabel(f'Gammas ($\gamma$)')
plt.ylabel(f'Testing MSE scores')
plt.title('Gammas X Testing MSE scores')
plt.legend()
plt.show()

# ################################################## tuning regularization l2
lambdas = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5]
gamma, decay, epochs, batch_size, eta0 = 0.7, 0, 5000, 10, 0.005
testing_mse = []

for lmbd in lambdas:
    model = MiniSGDM(
        batch_size=batch_size, epochs=epochs, eta0=eta0, decay=decay,
        lambda_=lmbd, gamma=gamma, regularization='l2', random_state=10)
    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)
    testing_mse.append(mse(y_true=z_test, y_hat=y_hat))

df = pd.DataFrame(testing_mse, columns=['With regularization l2'])
df['Bench-marked score from before'] = [3.39 for _ in testing_mse]
df.index = lambdas
sns.lineplot(data=df)
plt.xlabel(f'Lambdas (l2 regularization)')
plt.ylabel(f'Testing MSE scores')
plt.title('l2 regularization lambdas X Testing MSE scores')
plt.legend()
plt.show()
