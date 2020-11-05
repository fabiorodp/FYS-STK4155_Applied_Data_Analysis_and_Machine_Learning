# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project1.package.create_dataset import CreateData
from Project1.package.linear_models import OLS, Ridge
from Project2.package.metrics import mse
from Project2.package.gradient_descent import MiniSGDM
from Project2.package.studies import SearchParametersMiniSGDM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ################################################## re-cap Project 1
# ################################################## best OLS and Ridge
degree = [10, 11]
terrain_file = 'data/SRTM_data_Norway_1.tif'
labels = ['Testing MSE for OLS model:', 'Testing MSE for Ridge model:']
models = [OLS(random_state=10), Ridge(lambda_=0.0001, random_state=10)]

for d, l, model in zip(degree, labels, models):
    cd = CreateData(random_state=10)
    X, z = cd.fit(nr_samples=15,
                  degree=d,
                  terrain_file='data/SRTM_data_Norway_1.tif')

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
              terrain_file='data/SRTM_data_Norway_1.tif')

# splitting data
X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.2, shuffle=True, stratify=None, random_state=10)

# ################################################## grid-search SGD
etas = [0.02, 0.01, 0.005, 0.001, 0.0005]
epochs = [100, 500, 1000, 5000, 10000]
batch_sizes, gammas, lambdas, decays = 1.0, 0.0, 0.0, 0.0

gs = SearchParametersMiniSGDM(params='ETASxEPOCHS', random_state=10)

r2_train, mse_train, r2_test, mse_test, elapsed, models = \
    gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
           model=MiniSGDM, epochs=epochs, etas=etas, batch_sizes=batch_sizes,
           lambdas=lambdas, decays=decays, gammas=gammas, plot_results=True,
           verbose=True)

df = pd.DataFrame(mse_test)
df.columns = etas
df.index = epochs
sns.lineplot(data=df)
plt.ylabel("Testing MSE scores")
plt.ylim([0, 20])
plt.xlabel(f"Number of Epochs")
plt.title("(Eta X Epochs) X Testing MSE scores")
plt.legend([f'$\eta$ = {i}' for i in etas[::-1]])
plt.show()

# ################################################## grid-search Mini-SGDM
# ################################################## tuning time & performance
etas = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [1, 5, 10, 15, 20]
gammas, lambdas, decays, epochs = 0.0, 0.0, 0.0, 1000

gs = SearchParametersMiniSGDM(params='ETASxBATCHES', random_state=10)

r2_train, mse_train, r2_test, mse_test, elapsed, models = \
    gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
           model=MiniSGDM, epochs=epochs, etas=etas, batch_sizes=batch_sizes,
           lambdas=lambdas, decays=decays, gammas=gammas, plot_results=True,
           verbose=True)

df = pd.DataFrame(mse_test)
df.columns = etas
df.index = batch_sizes
sns.lineplot(data=df)
plt.ylabel("Testing MSE scores")
plt.ylim([0, 30])
plt.xlabel(f"Batch size")
plt.title("(Eta X Batch size) X Testing MSE scores")
plt.legend([f'$\eta$ = {i}' for i in etas[::-1]])
plt.show()

# ################################################## tuning eta-decay
etas = [0.35, 0.3, 0.25, 0.2, 0.01]
decays = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
gammas, lambdas, epochs, batch_sizes = 0.0, 0.0, 1000, 10

gs = SearchParametersMiniSGDM(params='ETASxDECAYS', random_state=10)

r2_train, mse_train, r2_test, mse_test, elapsed, models = \
    gs.run(X_train=X_train, X_test=X_test, y_train=z_train, y_test=z_test,
           model=MiniSGDM, epochs=epochs, etas=etas, batch_sizes=batch_sizes,
           lambdas=lambdas, decays=decays, gammas=gammas, plot_results=True,
           verbose=True)

df = pd.DataFrame(mse_test)
df.columns = etas
df.index = decays
sns.lineplot(data=df)
plt.ylabel("Testing MSE scores")
plt.ylim([0, 30])
plt.xlabel(f"Decays")
plt.title("(Eta X Decays) X Testing MSE scores")
plt.legend([f'$\eta$ = {i}' for i in etas[::-1]])
plt.show()

# ################################################## tuning gamma
gammas = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
lmbd, decay, epochs, batch_size, eta0 = 0, 0.001, 1000, 10, 0.3
testing_mse = []

for gamma in gammas:
    model = MiniSGDM(
        batch_size=batch_size, epochs=epochs, eta0=eta0, decay=decay,
        lambda_=lmbd, gamma=gamma, regularization='l2', random_state=10)
    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)
    mse_test = mse(y_true=z_test, y_hat=y_hat)
    print(f'Gamma: {gamma}   |   MSE test: {mse_test}')
    testing_mse.append(mse_test)

df = pd.DataFrame(testing_mse, columns=['With Momentum gamma'])
df['Bench-marked score from before'] = [4.32 for _ in testing_mse]
df.index = gammas
sns.lineplot(data=df)
plt.xlabel(f'Gammas ($\gamma$)')
plt.ylabel(f'Testing MSE scores')
plt.title('Gammas X Testing MSE scores')
plt.legend()
plt.show()

# ################################################## tuning regularization l2
lambdas = [0, 10**-9, 10**-6, 10**-4]
gamma, decay, epochs, batch_size, eta0 = 0.3, 0.001, 1000, 10, 0.3
testing_mse = []

for lmbd in lambdas:
    model = MiniSGDM(
        batch_size=batch_size, epochs=epochs, eta0=eta0, decay=decay,
        lambda_=lmbd, gamma=gamma, regularization='l2', random_state=10)
    model.fit(X_train, z_train)
    y_hat = model.predict(X_test)
    mse_test = mse(y_true=z_test, y_hat=y_hat)
    print(f'Lambdas: {lmbd}   |   MSE test: {mse_test}')
    testing_mse.append(mse_test)

df = pd.DataFrame(testing_mse, columns=['With regularization l2'])
df['Bench-marked score from before'] = [4.1814 for _ in testing_mse]
df.index = lambdas
sns.lineplot(data=df)
plt.xlabel(f'Lambdas $\lambda$={lambdas}')
plt.ylabel(f'Testing MSE scores')
plt.title('l2 regularization lambdas X Testing MSE scores')
plt.legend()
plt.show()
