# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GridSearch:
    """Class to find the best parameters for nr_samples,
    poly_degrees or lambdas."""

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, lambda_=None,
            test_size=0.2, scale=True, terrain=None,
            plot_results=False, print_results=False):

        if lambda_ is not None:
            mse_train = np.zeros(shape=(len(lambda_), len(poly_degrees)))
            mse_test = np.zeros(shape=(len(lambda_), len(poly_degrees)))
            r2_train = np.zeros(shape=(len(lambda_), len(poly_degrees)))
            r2_test = np.zeros(shape=(len(lambda_), len(poly_degrees)))

            for n_idx, n in enumerate(lambda_):
                for d_idx, d in enumerate(poly_degrees):

                    X, z = self.data.fit(
                        nr_samples=nr_samples,
                        degree=d,
                        terrain_file=terrain
                    )

                    X_train, X_test, z_train, z_test = \
                        self._split_scale(X, z, scale, test_size)

                    self.ML_model.set_lambda(new_lambda=n)
                    self.ML_model.fit(X_train, z_train)

                    z_hat = self.ML_model.predict(X_train)
                    z_tilde = self.ML_model.predict(X_test)

                    mse_train[n_idx, d_idx] = mean_squared_error(z_train, z_hat)
                    mse_test[n_idx, d_idx] = mean_squared_error(z_test, z_tilde)

                    r2_train[n_idx, d_idx] = r2_score(z_train, z_hat)
                    r2_test[n_idx, d_idx] = r2_score(z_test, z_tilde)

            if print_results is True:
                parameter_1st = poly_degrees
                parameter_2nd = lambda_

                self._print_best_results(
                    parameter_1st=parameter_1st, parameter_2nd=parameter_2nd,
                    mse_train=mse_train, mse_test=mse_test,
                    r2_train=r2_train, r2_test=r2_test)

            if plot_results is True:
                x_legend = "Polynomial degrees"
                y_legend = "Lambdas"
                x_label = poly_degrees
                y_label = lambda_
                annot = True

                self._plot_heat_map(
                    x_legend=x_legend, y_legend=y_legend,
                    x_label=x_label, y_label=y_label, annot=annot,
                    mse_train=mse_train, mse_test=mse_test,
                    r2_train=r2_train, r2_test=r2_test)

            return mse_train, mse_test, r2_train, r2_test

        else:
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
                parameter_1st = poly_degrees
                parameter_2nd = nr_samples

                self._print_best_results(
                    parameter_1st=parameter_1st, parameter_2nd=parameter_2nd,
                    mse_train=mse_train, mse_test=mse_test,
                    r2_train=r2_train, r2_test=r2_test)

            if plot_results is True:
                x_legend = "Polynomial degrees"
                y_legend = "Number of samples"
                x_label = poly_degrees
                y_label = nr_samples
                annot = True

                self._plot_heat_map(
                    x_legend=x_legend, y_legend=y_legend,
                    x_label=x_label, y_label=y_label, annot=annot,
                    mse_train=mse_train, mse_test=mse_test,
                    r2_train=r2_train, r2_test=r2_test)

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
    def _print_best_results(parameter_1st, parameter_2nd, mse_train,
                            mse_test, r2_train, r2_test):

        metrics = ["MSE", "MSE", "R2", "R2"]

        titles = ["Best MSE for training: ", "Best MSE for testing: ",
                  "Best R2-score for training: ",
                  "Best R2-score for testing: "]

        datas = [mse_train, mse_test, r2_train, r2_test]

        for metric, title, data in zip(metrics, titles, datas):
            if metric == "MSE":
                print(title, np.min(data),
                      "\n in Poly_degree={} and Parameter2={}".format(
                          parameter_1st[np.argwhere(
                              data == np.min(data))[0][1]],

                          parameter_2nd[np.argwhere(
                              data == np.min(data))[0][0]])
                      )

            elif metric == "R2":
                print(title, np.max(data),
                      "\n in Poly_degree={} and Parameter2={}".format(
                          parameter_1st[np.argwhere(
                              data == np.max(data))[0][1]],

                          parameter_2nd[np.argwhere(
                              data == np.max(data))[0][0]])
                      )

    @staticmethod
    def _plot_heat_map(x_legend, y_legend, x_label, y_label, annot,
                       mse_train, mse_test, r2_train, r2_test):

        datas = [mse_train, mse_test, r2_train, r2_test]

        titles = ["MSE values for Training", "MSE values for Testing",
                  "R2-score values for Training", "R2-score values for Testing"]

        for data, title in zip(datas, titles):
            sns.heatmap(data=data, xticklabels=x_label,
                        yticklabels=y_label, annot=annot)
            plt.title(title)
            plt.xlabel(x_legend)
            plt.ylabel(y_legend)
            plt.show()


class BiasVarianceTradeOff:
    """Class to decompose the bias-variance trade-off."""

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, lambda_=None,
            n_boostraps=100, test_size=0.2, scale=True,
            terrain=None, verboose=False, plot=False):

        error = np.zeros(shape=len(poly_degrees))
        bias = np.zeros(shape=len(poly_degrees))
        variance = np.zeros(shape=len(poly_degrees))

        for idx, c in enumerate(poly_degrees):
            X, z = self.data.fit(
                nr_samples=nr_samples,
                degree=c,
                terrain_file=terrain
            )

            X_train, X_test, z_train, z_test = \
                self._split_scale(X, z, scale, test_size)

            z_tilde = np.empty((z_test.shape[0], n_boostraps))

            for i in range(n_boostraps):
                x_, y_ = resample(X_train, z_train)

                if lambda_ is not None:
                    self.ML_model.set_lambda(new_lambda=lambda_)

                self.ML_model.fit(x_, y_)
                z_tilde[:, i] = self.ML_model.predict(X_test).ravel()

            # storing error
            error[idx] = np.mean(
                np.mean((z_test - z_tilde) ** 2, axis=1, keepdims=True))

            # storing bias2
            bias[idx] = np.mean(
                (z_test - np.mean(z_tilde, axis=1, keepdims=True)) ** 2)

            # storing variance
            variance[idx] = np.mean(
                np.var(z_tilde, axis=1, keepdims=True))

            if verboose is True:
                self._verboose(idx=idx, complexity=c, error=error,
                               bias=bias, variance=variance)

        if plot is True:
            self._plot(poly_degrees, error, bias, variance)

        return error, bias, variance

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

    def _plot(self, complexities, error, bias, variance):
        plt.plot(complexities, error, label='Error')
        plt.plot(complexities, bias, "--", label='bias^2')
        plt.plot(complexities, variance, label='Variance')
        plt.ylabel("Metrics: [Error, Bias^2 and Variance]")
        plt.xlabel("Polynomial degrees")
        plt.title("Bias-variance tradeoff")
        plt.legend()
        plt.show()

    def _verboose(self, idx, complexity, error, bias, variance):
        print('Complexity: ', complexity)
        print('Error: ', error[idx])
        print('Bias^2: ', bias[idx])
        print('Var: ', variance[idx])
        print('{} >= {} + {} = {}'
              .format(error[idx], bias[idx], variance[idx],
                      bias[idx] + variance[idx]))


class CrossValidationKFolds:
    """Class to perform the k-fold cross-validation."""

    def __init__(self, data, model, random_state=None):
        """
        Constructor to initialize the class.

        :param data: class object:  To generate the data-set.
        :param model: class object: To perform the fitting
                                    and predictions.
        :param random_state: float: Number of the seed.
        """
        self.data = data(random_state=random_state)
        self.ML_model = model(random_state=random_state)
        self.random_state = random_state
        np.random.seed(self.random_state)

    def run(self, nr_samples, poly_degrees, terrain=None,
            k=5, lambda_=None, shuffle=True, plot=False):

        avg_mse = []
        for degree in poly_degrees:

            # generating data-sets
            X, z = self.data.fit(nr_samples=nr_samples, degree=degree,
                                 terrain_file=terrain)

            # scaling
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaler.fit(X)
            X_std = scaler.transform(X)

            # getting k-fold with the sample indexes
            kfold_sample_idxs = self._kfold(k=k, X=X_std,
                                            shuffle=shuffle)

            mse = []
            for element in kfold_sample_idxs:
                # separating the test fold from the training data
                z_test_CV = z[element, :]
                X_test_CV = X_std[element, :]

                # making the training data without the test fold
                copy_z = np.delete(z, element)
                copy_X = np.delete(X_std, element, axis=0)

                # fitting, predicting and getting the MSE for each fold
                if lambda_ is not None:
                    self.ML_model.set_lambda(new_lambda=lambda_)

                self.ML_model.fit(copy_X, copy_z)
                z_tilde_CV = self.ML_model.predict(X_test_CV)
                mse.append(mean_squared_error(z_test_CV, z_tilde_CV))

            # appending the average MSE
            avg_mse.append(np.mean(mse))

        if plot is True:
            self._plot(complexities=poly_degrees, avg_mse=avg_mse, k=k)

        return avg_mse

    @staticmethod
    def _kfold(X, k, shuffle=True):
        """
        Divide the data-set in k number of folds, shuffling the
        samples and picking the indexes without replacement.

        :param X: nd-array: Design matrix.
        :param k: int: Number of folds.
        :param shuffle: bool: Shuffle the samples before divide
                              the folds.

        :return: list of nd-array: A list of nd-arrays with k-fold
                                   containing the indexes of the
                                   divided samples.
        """
        fold_size = int(X.shape[0] / k)
        list_of_idx = np.arange(X.shape[0])

        if shuffle is True:
            np.random.shuffle(list_of_idx)

        kfold_sample_idxs = []
        for _ in range(k):
            if list_of_idx.shape[0] <= fold_size:
                list_of_idx_sliced = list_of_idx[:fold_size]
                kfold_sample_idxs.append(list_of_idx_sliced)

            else:
                list_of_idx_sliced = list_of_idx[:fold_size]
                kfold_sample_idxs.append(list_of_idx_sliced)
                list_of_idx = np.delete(list_of_idx,
                                        np.arange(fold_size))

        return kfold_sample_idxs

    @staticmethod
    def _plot(complexities, avg_mse, k):
        plt.plot(complexities, avg_mse, "--", label='MSE')
        plt.ylabel("Training MSE")
        plt.xlabel("Complexity: Polynomial degrees")
        plt.title("Our {}-folds cross-validation".format(k))
        plt.legend()
        plt.show()


class CrossValidationSKlearn:
    """Class to perform sklearn k-fold cross-validation."""

    def __init__(self, data, model, random_state=None):
        self.data = data(random_state=random_state)
        self.ML_model = model
        self.random_state = random_state

    def run(self, nr_samples, poly_degrees, k,
            shuffle=True, plot=False):

        avg_mse = []
        for degree in poly_degrees:

            # generating data-sets
            X, z = self.data.fit(nr_samples=nr_samples, degree=degree)

            # scaling
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaler.fit(X)
            X_std = scaler.transform(X)

            kfold = KFold(n_splits=k, random_state=self.random_state,
                          shuffle=shuffle)

            estimated_mse_folds = \
                cross_val_score(self.ML_model, X_std, z,
                                scoring='neg_mean_squared_error',
                                cv=kfold)

            # cross_val_score return an array containing the
            # estimated negative mse for every fold.
            # we have to the the mean of every array to
            # get an estimate of the mse of the model
            avg_mse.append(np.mean(-estimated_mse_folds))

        if plot is True:
            self._plot(complexities=poly_degrees, avg_mse=avg_mse, k=k)

        return avg_mse

    @staticmethod
    def _plot(complexities, avg_mse, k):
        plt.plot(complexities, avg_mse, "--", label='mse')
        plt.ylabel("Training MSE")
        plt.xlabel("Complexity: Polynomial degrees")
        plt.title("SKLearn {}-folds cross-validation".format(k))
        plt.legend()
        plt.show()
