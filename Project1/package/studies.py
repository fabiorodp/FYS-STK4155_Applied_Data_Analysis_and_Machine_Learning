# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GridSearch:

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, lambda_=1,
            test_size=0.2, scale=True, terrain=None,
            plot_results=False, print_results=False):

        mse_train = np.zeros(shape=(len(nr_samples), len(poly_degrees)))
        mse_test = np.zeros(shape=(len(nr_samples), len(poly_degrees)))
        r2_train = np.zeros(shape=(len(nr_samples), len(poly_degrees)))
        r2_test = np.zeros(shape=(len(nr_samples), len(poly_degrees)))

        for n_idx, n in enumerate(nr_samples):

            for d_idx, d in enumerate(poly_degrees):

                X, z = self.data.fit(
                    nr_samples=n,
                    degree=d,
                    terrain_file=terrain
                )

                X_train, X_test, z_train, z_test = \
                    self._split_scale(X, z, scale, test_size)

                self.ML_model.fit(X_train, z_train)

                z_hat = self.ML_model.predict(X_train)
                z_tilde = self.ML_model.predict(X_test)

                mse_train[n_idx, d_idx] = mean_squared_error(z_train, z_hat)
                mse_test[n_idx, d_idx] = mean_squared_error(z_test, z_tilde)

                r2_train[n_idx, d_idx] = r2_score(z_train, z_hat)
                r2_test[n_idx, d_idx] = r2_score(z_test, z_tilde)

        if print_results is True:
            self._print_best_results(nr_samples, poly_degrees, mse_train,
                                     mse_test, r2_train, r2_test)

        if plot_results is True:
            self._plot_heat_map(nr_samples, mse_train, poly_degrees,
                                mse_test, r2_train, r2_test)

        return mse_train, mse_test, r2_train, r2_test

    def _split_scale(self, X, z, scale, test_size):

        X_train, X_test, z_train, z_test = \
            None, None, None, None

        # splitting dataset
        X_train, X_test, z_train, z_test = \
            train_test_split(
                X, z, test_size=test_size,
                random_state=self.random_state)

        if scale is True:
            # scaling dataset
            scaler = StandardScaler(with_mean=False,
                                    with_std=True)
            scaler.fit(X_train)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        return X_train, X_test, z_train, z_test

    @staticmethod
    def _print_best_results(nr_samples, poly_degrees, mse_train,
                            mse_test, r2_train, r2_test):

        print("Best MSE for training: ", np.min(mse_train),
              "\n with Nr_samples={} and poly_degree={}".format(
                  nr_samples[np.argwhere(mse_train == np.min(mse_train))[0][0]],
                  poly_degrees[np.argwhere(mse_train == np.min(mse_train))[0][1]])
              )

        print("Best MSE for testing: ", np.min(mse_test),
              "\n with Nr_samples={} and poly_degree={}".format(
                  nr_samples[np.argwhere(mse_test == np.min(mse_test))[0][0]],
                  poly_degrees[np.argwhere(mse_test == np.min(mse_test))[0][1]])
              )

        print("Best R2-score for training: ", np.max(r2_train),
              "\n with Nr_samples={} and poly_degree={}".format(
                  nr_samples[np.argwhere(r2_train == np.max(r2_train))[0][0]],
                  poly_degrees[np.argwhere(r2_train == np.max(r2_train))[0][1]])
              )

        print("Best R2-score for testing: ", np.max(r2_test),
              "\n with Nr_samples={} and poly_degree={}".format(
                  nr_samples[np.argwhere(r2_test == np.max(r2_test))[0][0]],
                  poly_degrees[np.argwhere(r2_test == np.max(r2_test))[0][1]])
              )

    @staticmethod
    def _plot_heat_map(nr_samples, mse_train, poly_degrees,
                       mse_test, r2_train, r2_test):

        sns.heatmap(data=mse_train, xticklabels=poly_degrees,
                    yticklabels=nr_samples, annot=True)
        plt.title("MSE values for Training")
        plt.xlabel("Polynomial Degrees")
        plt.ylabel("Number of samples")
        plt.show()

        sns.heatmap(data=mse_test, xticklabels=poly_degrees,
                    yticklabels=nr_samples, annot=True)
        plt.title("MSE values for Testing")
        plt.xlabel("Polynomial Degrees")
        plt.ylabel("Number of samples")
        plt.show()

        sns.heatmap(data=r2_train, xticklabels=poly_degrees,
                    yticklabels=nr_samples, annot=True)
        plt.title("R2-score values for Training")
        plt.xlabel("Polynomial Degrees")
        plt.ylabel("Number of samples")
        plt.show()

        sns.heatmap(data=r2_test, xticklabels=poly_degrees,
                    yticklabels=nr_samples, annot=True)
        plt.title("R2-score values for Testing")
        plt.xlabel("Polynomial Degrees")
        plt.ylabel("Number of samples")
        plt.show()


class PlotCOMPLEXITYxMSE:

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, lambda_=1,
            test_size=0.2, scale=True, terrain=None):

        mse_train, mse_test, r2_train, r2_test = \
            None, None, None, None

        if isinstance(poly_degrees, int) is False:
            function_of = "d"

            mse_train = np.zeros(shape=len(poly_degrees))
            mse_test = np.zeros(shape=len(poly_degrees))
            r2_train = np.zeros(shape=len(poly_degrees))
            r2_test = np.zeros(shape=len(poly_degrees))

            for idx, i in enumerate(poly_degrees):
                X, z = self.data.fit(
                    nr_samples=nr_samples,
                    degree=i,
                    terrain_file=terrain
                )

                X_train, X_test, z_train, z_test = \
                    self._split_scale(X, z, scale, test_size)

                self.ML_model.fit(X_train, z_train)

                z_predict_train = self.ML_model.predict(X_train)
                z_predict_test = self.ML_model.predict(X_test)

                mse_train[idx] = mse(z_train, z_predict_train)
                mse_test[idx] = mse(z_test, z_predict_test)

                r2_train[idx] = r2(z_train, z_predict_train)
                r2_test[idx] = r2(z_test, z_predict_test)

            self._plot_COMPLEXITY_x_MSE(
                poly_degrees, mse_train, mse_test)

        if isinstance(nr_samples, int) is False:
            function_of = "n"

            mse_train = np.zeros(shape=len(nr_samples))
            mse_test = np.zeros(shape=len(nr_samples))
            r2_train = np.zeros(shape=len(nr_samples))
            r2_test = np.zeros(shape=len(nr_samples))

            for idx, i in enumerate(nr_samples):
                X, z = self.data.fit(
                    nr_samples=i,
                    degree=poly_degrees,
                    terrain_file=terrain
                )

                X_train, X_test, z_train, z_test = \
                    self._split_scale(X, z, scale, test_size)

                self.ML_model.fit(X_train, z_train)

                z_predict_train = self.ML_model.predict(X_train)
                z_predict_test = self.ML_model.predict(X_test)

                mse_train[idx] = mse(z_train, z_predict_train)
                mse_test[idx] = mse(z_test, z_predict_test)

                r2_train[idx] = r2(z_train, z_predict_train)
                r2_test[idx] = r2(z_test, z_predict_test)

            self._plot_COMPLEXITY_x_MSE(
                poly_degrees, mse_train, mse_test)

    def _split_scale(self, X, z, scale, test_size):

        X_train, X_test, z_train, z_test = \
            None, None, None, None

        # splitting dataset
        X_train, X_test, z_train, z_test = \
            train_test_split(
                X, z, test_size=test_size,
                random_state=self.random_state)

        if scale is True:
            # scaling dataset
            scaler = StandardScaler(with_mean=False,
                                    with_std=True)
            scaler.fit(X_train)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        return X_train, X_test, z_train, z_test

    @staticmethod
    def _plot_COMPLEXITY_x_MSE(complexity, mse_train, mse_test):
        """Plotting MSE train and MSE test for different
        poly_degrees."""

        plt.plot(complexity, mse_train, "--", label='MSE Train')
        plt.plot(complexity, mse_test, label='MSE Test')
        plt.xlabel("Complexity: Polynomial degrees")
        plt.ylabel("MSE scores")
        plt.title("MSE train and MSE test, with nr_sample=148, for "
                  "different poly_degrees")
        plt.legend()
        plt.show()


class BiasVarianceTradeOff:

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, lambda_=1, n_boostraps=1,
            test_size=0.2, scale=True, terrain=None, ylim=(0, 1)):

        if isinstance(poly_degrees, int) is False:

            error = np.zeros(shape=len(poly_degrees))
            bias = np.zeros(shape=len(poly_degrees))
            variance = np.zeros(shape=len(poly_degrees))
            z_test, z_tilde = None, None

            for idx, i in enumerate(poly_degrees):
                X, z = self.data.fit(
                    nr_samples=nr_samples,
                    degree=i,
                    terrain_file=terrain
                )

                X_train, X_test, z_train, z_test = \
                    self._split_scale(X, z, scale, test_size)

                z_tilde = np.empty((z_test.shape[0], n_boostraps))

                for i in range(n_boostraps):
                    x_, y_ = resample(X_train, z_train)
                    self.ML_model.fit(x_, y_)
                    z_tilde[:, i] = self.ML_model.predict(X_test).ravel()

                error[idx] = np.mean(np.mean((z_test - z_tilde) ** 2, axis=1, keepdims=True))
                bias[idx] = np.mean((z_test - np.mean(z_tilde, axis=1, keepdims=True)) ** 2)
                variance[idx] = np.mean(np.var(z_tilde, axis=1, keepdims=True))

            self._plot(ylim, poly_degrees, error, bias, variance)

            return error, bias, variance, z_test, z_tilde

    def _split_scale(self, X, z, scale, test_size):

        X_train, X_test, z_train, z_test = \
            None, None, None, None

        # splitting dataset
        X_train, X_test, z_train, z_test = \
            train_test_split(
                X, z, test_size=test_size,
                random_state=self.random_state)

        if scale is True:
            # scaling dataset
            scaler = StandardScaler(with_mean=False,
                                    with_std=True)
            scaler.fit(X_train)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        return X_train, X_test, z_train, z_test

    def _plot(self, ylim, complexities, error, bias, variance):
        plt.plot(complexities, error, label='Error/MSE')
        plt.plot(complexities, bias, "--", label='bias')
        plt.plot(complexities, variance, label='Variance')
        plt.ylabel("Metrics: [Error/MSE, bias^2 and variance]")
        plt.xlabel("Polynomial degrees")
        plt.title("Bias-variance tradeoff")
        plt.ylim(ylim)
        plt.legend()
        plt.show()
