# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from Project2.package.metrics import r2, mse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time


class SearchParametersMiniSGDM:
    """Class to find the best parameters for ETAs as function of
     Batch-sizes, epochs or lambdas."""

    @staticmethod
    def _plot_heatmaps(params, param1, param2, r2_train, mse_train, r2_test,
                       mse_test, elapsed):

        datas = [mse_train, mse_test, r2_train, r2_test, elapsed]

        titles = ["MSE values for Training",
                  "MSE values for Testing",
                  "R2-score values for Training",
                  "R2-score values for Testing",
                  "Time elapsed (in sec) for training the model"]

        for data, title in zip(datas, titles):
            sns.heatmap(data=data, xticklabels=param1,
                        yticklabels=param2, annot=True,
                        annot_kws={"size": 8.5}, fmt=".2f")

            if params == 'ETASxLAMBDAS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Regularization $\lambda$ = {param2}")

            elif params == 'ETASxEPOCHS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Epochs = {param2}")

            elif params == 'ETASxBATCHES':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Batch sizes = {param2}")

            elif params == 'ETASxDECAYS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Decays = {param2}")

            elif params == 'ETASxGAMMAS':
                plt.xlabel(f"Learning rate $\eta$ = {param1}")
                plt.ylabel(f"Gamma $\gamma$ = {param2}")

            plt.title(title)
            plt.show()

    @staticmethod
    def _verbose(params, p1, p2, r2_train_value, mse_train_value,
                 r2_test_value, mse_test_value, elapsed):

        if params == 'ETASxLAMBDAS':
            print(f'Eta: {p1}   |   Lambda: {p2}')

        elif params == 'ETASxEPOCHS':
            print(f'Eta {p1}    |   Epochs: {p2}')

        elif params == 'ETASxBATCHES':
            print(f'Eta {p1}    |   Batch size:{p2}')

        elif params == 'ETASxDECAYS':
            print(f'Eta {p1}    |   Decay: {p2}')

        elif params == 'ETASxGAMMAS':
            print(f'Eta {p1}    |   Gamma:{p2}')

        print(f'R2 train: {r2_train_value}  |   MSE Train: {mse_train_value}')
        print(f'R2 test: {r2_test_value}    |   MSE test: {mse_test_value}')
        print(f'Time elapsed for training {elapsed} s.')

    def __init__(self, params='ETASxEPOCHS', random_state=None):
        """
        Constructor for the class.

        :param params: string: Parameter for the Grid-Search study.
        :param random_state: int: Seed for the experiment.
        """

        if random_state is not None:
            np.random.seed(random_state)

        self.models = None
        self.params = params
        self.random_state = random_state

    def run(self, X_train, X_test, y_train, y_test, model, epochs, etas,
            batch_sizes, lambdas, decays, gammas, plot_results=False,
            verbose=False):
        """
        Run the experiment to search the best parameters.
        """
        param1, param2, md = None, None, None

        if self.params == 'ETASxLAMBDAS':
            param1, param2 = etas, lambdas

        elif self.params == 'ETASxEPOCHS':
            param1, param2 = etas, epochs

        elif self.params == 'ETASxBATCHES':
            param1, param2 = etas, batch_sizes

        elif self.params == 'ETASxDECAYS':
            param1, param2 = etas, decays

        elif self.params == 'ETASxGAMMAS':
            param1, param2 = etas, gammas

        mse_train = np.zeros(shape=(len(param2), len(param1)))
        mse_test = np.zeros(shape=(len(param2), len(param1)))
        r2_train = np.zeros(shape=(len(param2), len(param1)))
        r2_test = np.zeros(shape=(len(param2), len(param1)))
        elapsed = np.zeros(shape=(len(param2), len(param1)))
        self.models = np.empty(shape=(len(param2), len(param1)), dtype=object)

        for c_idx, p1 in enumerate(param1):
            for r_idx, p2 in enumerate(param2):

                # set parameters
                if self.params == 'ETASxLAMBDAS':
                    md = model(
                        batch_size=batch_sizes, epochs=epochs, eta0=p1,
                        decay=decays, lambda_=p2, gamma=gammas,
                        regularization=None, random_state=self.random_state)

                elif self.params == 'ETASxEPOCHS':
                    md = model(
                        batch_size=batch_sizes, epochs=p2, eta0=p1,
                        decay=decays, lambda_=lambdas, gamma=gammas,
                        regularization=None, random_state=self.random_state)

                elif self.params == 'ETASxBATCHES':
                    md = model(
                        batch_size=p2, epochs=epochs, eta0=p1,
                        decay=decays, lambda_=lambdas, gamma=gammas,
                        regularization=None, random_state=self.random_state)

                elif self.params == 'ETASxDECAYS':
                    md = model(
                        batch_size=batch_sizes, epochs=epochs, eta0=p1,
                        decay=p2, lambda_=lambdas, gamma=gammas,
                        regularization=None, random_state=self.random_state)

                elif self.params == 'ETASxGAMMAS':
                    md = model(
                        batch_size=batch_sizes, epochs=epochs, eta0=p1,
                        decay=decays, lambda_=lambdas, gamma=p2,
                        regularization=None, random_state=self.random_state)

                # train model
                time0 = time()
                md.fit(X=X_train, z=y_train)
                time1 = time()
                elapsed_value = time1 - time0
                elapsed[r_idx, c_idx] = elapsed_value

                # assess model
                y_hat = md.predict(X=X_train)
                r2_train_value = r2(y_train, y_hat)
                r2_train[r_idx, c_idx] = r2_train_value
                mse_train_value = mse(y_train, y_hat)
                mse_train[r_idx, c_idx] = mse_train_value

                y_tilde = md.predict(X=X_test)
                r2_test_value = r2(y_test, y_tilde)
                r2_test[r_idx, c_idx] = r2_test_value
                mse_test_value = mse(y_test, y_tilde)
                mse_test[r_idx, c_idx] = mse_test_value

                if verbose is True:
                    self._verbose(params=self.params, p1=p1, p2=p2,
                                  r2_train_value=r2_train_value,
                                  mse_train_value=mse_train_value,
                                  r2_test_value=r2_test_value,
                                  mse_test_value=mse_test_value,
                                  elapsed=elapsed_value)

                self.models[r_idx, c_idx] = md

        if plot_results is True:
            self._plot_heatmaps(params=self.params, param1=param1,
                                param2=param2, r2_train=r2_train,
                                mse_train=mse_train, r2_test=r2_test,
                                mse_test=mse_test, elapsed=elapsed)

        return r2_train, mse_train, r2_test, mse_test, elapsed, self.models


class SearchParametersDNN:
    def __init__(self, params, random_state=None):
        self.params = params
        self.random_state = random_state
        self.mse_train = None
        self.mse_test = None
        self.r2_train = None
        self.r2_test = None
        self.elapsed = None
        self.models = None
        self.score = None
        self.prt1 = None
        self.prt2 = None
        self.prt3 = None

        if random_state is not None:
            np.random.seed(random_state)

    def _LAYERSxNEURONS(self, X_train, X_test, z_train, z_test, model,
                        epochs, batch_size, etas, learning_rate, decays,
                        lmbds, bias0, init_weights, act_function,
                        output_act_function, cost_function, random_state,
                        verbose, layers, neurons, hidden_layers):

        self.prt1, self.prt2, self.prt3 = layers, neurons, None
        self._init_parameters()

        for c_idx, p1 in enumerate(self.prt1):
            for r_idx, p2 in enumerate(self.prt2):
                h_layers = [p2] * p1

                if verbose is True:
                    print(f'Hidden layers: {h_layers}')

                md = model(
                    hidden_layers=h_layers, epochs=epochs,
                    batch_size=batch_size, eta0=etas,
                    learning_rate=learning_rate, decay=decays, lmbd=lmbds,
                    bias0=bias0, init_weights=init_weights,
                    act_function=act_function,
                    output_act_function=output_act_function,
                    cost_function=cost_function, random_state=random_state,
                    verbose=verbose)

                self._fit_and_assess(
                    X_train=X_train, X_test=X_test, z_train=z_train,
                    z_test=z_test, md=md, r_idx=r_idx, c_idx=c_idx,
                    d_idx=None, verbose=verbose)

        datas = [self.mse_test, self.mse_train, self.r2_test, self.r2_train]
        titles = ["Testing MSE scores", "Training MSE scores",
                  "Testing R2 scores", "Training R2 scores"]

        # plotting heat-map
        for d, t in zip(datas, titles):
            df = pd.DataFrame(d)
            df.columns = layers
            df.index = neurons
            sns.heatmap(data=df, xticklabels=layers, yticklabels=neurons,
                        annot=True, annot_kws={"size": 8.5}, fmt=".2f")
            plt.ylabel("Number of neurons in each hidden-layer")
            plt.xlabel(f"Number of hidden-layers")
            plt.title(t)
            plt.show()

    def _ETASxDECAYS(self, X_train, X_test, z_train, z_test, model,
                     epochs, batch_size, etas, learning_rate, decays,
                     lmbds, bias0, init_weights, act_function,
                     output_act_function, cost_function, random_state,
                     verbose, layers, neurons, hidden_layers):

        self.prt1, self.prt2, self.prt3 = etas, decays, None
        self._init_parameters()

        for c_idx, p1 in enumerate(self.prt1):
            for r_idx, p2 in enumerate(self.prt2):

                if verbose is True:
                    print(f'Eta: {p1}    |    Decays: {p2}')

                md = model(
                    hidden_layers=hidden_layers, epochs=epochs,
                    batch_size=batch_size, eta0=p1,
                    learning_rate=learning_rate, decay=p2, lmbd=lmbds,
                    bias0=bias0, init_weights=init_weights,
                    act_function=act_function,
                    output_act_function=output_act_function,
                    cost_function=cost_function, random_state=random_state,
                    verbose=verbose)

                self._fit_and_assess(
                    X_train=X_train, X_test=X_test, z_train=z_train,
                    z_test=z_test, md=md, r_idx=r_idx, c_idx=c_idx,
                    d_idx=None, verbose=verbose)

        datas = [self.mse_test, self.mse_train]
        titles = ["Testing MSE scores", "Training MSE scores at 1000 epoch"]

        # plotting heat-map
        for d, t in zip(datas, titles):
            df = pd.DataFrame(d)
            df.columns = etas
            df.index = decays
            sns.heatmap(data=df, xticklabels=etas, yticklabels=decays,
                        annot=True, annot_kws={"size": 8.5}, fmt=".2f")
            plt.ylabel("Decay values")
            plt.xlabel(f"Learning rate values")
            plt.title(t)
            plt.show()

    def _ETASxDECAYSxLAMBDAS(self, X_train, X_test, z_train, z_test, model,
                             epochs, batch_size, etas, learning_rate, decays,
                             lmbds, bias0, init_weights, act_function,
                             output_act_function, cost_function, random_state,
                             verbose, layers, neurons, hidden_layers):

        self.prt1, self.prt2, self.prt3 = etas, decays, lmbds
        self._init_parameters()

        for c_idx, p1 in enumerate(self.prt1):
            for r_idx, p2 in enumerate(self.prt2):
                for d_idx, p3 in enumerate(self.prt3):

                    if verbose is True:
                        print(f'Eta: {p1}    |    '
                              f'Decays: {p2}    |    '
                              f'Lambda{p3}')

                    md = model(
                        hidden_layers=hidden_layers, epochs=epochs,
                        batch_size=batch_size, eta0=p1,
                        learning_rate=learning_rate, decay=p2, lmbd=p3,
                        bias0=bias0, init_weights=init_weights,
                        act_function=act_function,
                        output_act_function=output_act_function,
                        cost_function=cost_function,
                        random_state=random_state,
                        verbose=verbose)

                    self._fit_and_assess(
                        X_train=X_train, X_test=X_test, z_train=z_train,
                        z_test=z_test, md=md, r_idx=r_idx, c_idx=c_idx,
                        d_idx=d_idx, verbose=verbose)

    def _NEURONSxETAS(self, X_train, X_test, z_train, z_test, model, epochs,
                      batch_size, etas, learning_rate, decays, lmbds, bias0,
                      init_weights, act_function, output_act_function,
                      cost_function, random_state, verbose, layers, neurons,
                      hidden_layers):

        self.prt1, self.prt2, self.prt3 = neurons, etas, None
        self._init_parameters()

        for c_idx, p1 in enumerate(self.prt1):
            for r_idx, p2 in enumerate(self.prt2):
                h_layers = [p1]*layers

                if verbose is True:
                    print(f'Hidden layers: {h_layers}')

                md = model(
                    hidden_layers=h_layers, epochs=epochs,
                    batch_size=batch_size, eta0=p2,
                    learning_rate=learning_rate, decay=decays, lmbd=lmbds,
                    bias0=bias0, init_weights=init_weights,
                    act_function=act_function,
                    output_act_function=output_act_function,
                    cost_function=cost_function, random_state=random_state,
                    verbose=verbose)

                self._fit_and_assess(
                    X_train=X_train, X_test=X_test, z_train=z_train,
                    z_test=z_test, md=md, r_idx=r_idx, c_idx=c_idx,
                    d_idx=None, verbose=verbose)

    def _init_parameters(self):
        if self.prt3 is not None:
            self.mse_train = np.empty(shape=(len(self.prt3),
                                             len(self.prt2), len(self.prt1)))
            self.mse_test = np.empty(shape=(len(self.prt3),
                                            len(self.prt2), len(self.prt1)))
            self.r2_train = np.empty(shape=(len(self.prt3),
                                            len(self.prt2), len(self.prt1)))
            self.r2_test = np.empty(shape=(len(self.prt3),
                                           len(self.prt2), len(self.prt1)))
            self.elapsed = np.empty(shape=(len(self.prt3),
                                           len(self.prt2), len(self.prt1)))
            self.score = np.empty(shape=(len(self.prt3),
                                         len(self.prt2), len(self.prt1)))
            self.models = np.empty(shape=(len(self.prt3), len(self.prt2),
                                          len(self.prt1)),
                                   dtype=object)

        else:
            self.mse_train = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.mse_test = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.r2_train = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.r2_test = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.elapsed = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.score = np.empty(shape=(len(self.prt2), len(self.prt1)))
            self.models = np.empty(shape=(len(self.prt2), len(self.prt1)),
                                   dtype=object)

    def _fit_and_assess(self, X_train, X_test, z_train, z_test, md,
                        r_idx, c_idx, d_idx, verbose):
        # fitting model
        time0 = time()
        md.fit(X=X_train, y=z_train)
        time1 = time()
        elapsed_value = time1 - time0

        if verbose is True:
            print(f'Fitting time: {elapsed_value}')

        # assess training
        z_hat = md.predict(X=X_train)
        r2_train_value = r2(z_train, z_hat)
        mse_train_value = mse(z_train, z_hat)

        # assess testing
        z_tilde = md.predict(X=X_test)
        r2_test_value = r2(z_test, z_tilde)
        mse_test_value = mse(z_test, z_tilde)

        # store results
        if self.prt3 is not None:
            self.r2_train[d_idx, r_idx, c_idx] = r2_train_value
            self.r2_test[d_idx, r_idx, c_idx] = r2_test_value
            self.mse_train[d_idx, r_idx, c_idx] = mse_train_value
            self.mse_test[d_idx, r_idx, c_idx] = mse_test_value
            self.elapsed[d_idx, r_idx, c_idx] = elapsed_value
            self.models[d_idx, r_idx, c_idx] = md
            self.score[d_idx, r_idx, c_idx] = md.costs[-1]

        else:
            self.r2_train[r_idx, c_idx] = r2_train_value
            self.r2_test[r_idx, c_idx] = r2_test_value
            self.mse_train[r_idx, c_idx] = mse_train_value
            self.mse_test[r_idx, c_idx] = mse_test_value
            self.elapsed[r_idx, c_idx] = elapsed_value
            self.models[r_idx, c_idx] = md
            self.score[r_idx, c_idx] = md.costs[-1]

    def run(self, X_train, X_test, z_train, z_test, model, epochs, batch_size,
            etas, learning_rate, decays, lmbds, bias0, init_weights,
            act_function, output_act_function, cost_function,
            random_state=None, verbose=False, layers=None, neurons=None,
            hidden_layers=None):
        """
        Run the experiment to search the best parameters.
        """
        if self.params == 'LAYERSxNEURONS':
            self._LAYERSxNEURONS(
                X_train=X_train, X_test=X_test, z_train=z_train,
                z_test=z_test, model=model, epochs=epochs,
                batch_size=batch_size, etas=etas,
                learning_rate=learning_rate, decays=decays,
                lmbds=lmbds, bias0=bias0,
                init_weights=init_weights,
                act_function=act_function,
                output_act_function=output_act_function,
                cost_function=cost_function,
                random_state=random_state, verbose=verbose,
                layers=layers, neurons=neurons,
                hidden_layers=hidden_layers)

        elif self.params == 'ETASxDECAYS':
            self._ETASxDECAYS(
                X_train=X_train, X_test=X_test, z_train=z_train,
                z_test=z_test, model=model, epochs=epochs,
                batch_size=batch_size, etas=etas,
                learning_rate=learning_rate, decays=decays,
                lmbds=lmbds, bias0=bias0,
                init_weights=init_weights,
                act_function=act_function,
                output_act_function=output_act_function,
                cost_function=cost_function,
                random_state=random_state, verbose=verbose,
                layers=layers, neurons=neurons,
                hidden_layers=hidden_layers)

        elif self.params == 'ETASxDECAYSxLAMBDAS':
            self._ETASxDECAYSxLAMBDAS(
                X_train=X_train, X_test=X_test, z_train=z_train,
                z_test=z_test, model=model, epochs=epochs,
                batch_size=batch_size, etas=etas,
                learning_rate=learning_rate, decays=decays,
                lmbds=lmbds, bias0=bias0,
                init_weights=init_weights,
                act_function=act_function,
                output_act_function=output_act_function,
                cost_function=cost_function,
                random_state=random_state, verbose=verbose,
                layers=layers, neurons=neurons,
                hidden_layers=hidden_layers)

        elif self.params == 'NEURONSxETAS':
            self._NEURONSxETAS(
                X_train=X_train, X_test=X_test, z_train=z_train,
                z_test=z_test, model=model, epochs=epochs,
                batch_size=batch_size, etas=etas,
                learning_rate=learning_rate, decays=decays,
                lmbds=lmbds, bias0=bias0,
                init_weights=init_weights,
                act_function=act_function,
                output_act_function=output_act_function,
                cost_function=cost_function,
                random_state=random_state, verbose=verbose,
                layers=layers, neurons=neurons,
                hidden_layers=hidden_layers)

    def plot_TRAINxTEST_MSE(self, column_idx, indexes,
                            title="MSE as a function of number of neurons "
                                  "for each hidden-layer.",
                            ylabel="Training and Testing MSE scores",
                            xlabel="Number of neurons for each hidden-layer",
                            ylim=[0, 15]):

        df = pd.DataFrame((self.mse_test[:, column_idx],
                           self.mse_train[:, column_idx])).transpose()
        df.columns = ["MSE test", "MSE train"]
        df.index = indexes
        sns.lineplot(data=df)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.grid()
        plt.show()

    def plot_cost_analysis(self, column_idx, row_idx,
                           title="Epochs x Accuracy score",
                           ylabel="Accuracy scores",
                           xlabel="Epoch numbers",
                           ylim=[0, 15]):

        md = self.models[row_idx, column_idx]
        plt.plot(np.arange(md.epochs), md.costs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.grid()
        plt.show()

    def argbest(self, metric_arr):
        """Gives the array's index of the cell containing the
        minimum value."""
        if metric_arr == 'mse_train':
            return np.unravel_index(np.argmin(self.mse_train, axis=None),
                                    self.mse_train.shape)
        elif metric_arr == 'mse_test':
            return np.unravel_index(np.argmin(self.mse_test, axis=None),
                                    self.mse_test.shape)
        elif metric_arr == 'r2_train':
            return np.unravel_index(np.argmax(self.r2_train, axis=None),
                                    self.r2_train.shape)
        elif metric_arr == 'r2_test':
            return np.unravel_index(np.argmax(self.r2_test, axis=None),
                                    self.r2_test.shape)
        elif metric_arr == 'loss_score':
            return np.unravel_index(np.argmin(self.score, axis=None),
                                    self.score.shape)
        elif metric_arr == 'time':
            return np.unravel_index(np.argmin(self.elapsed, axis=None),
                                    self.elapsed.shape)
        else:
            raise ValueError("metric_arr not defined correctly.")

    def heat_map_3d(self, metric_arr, title='', ylabel='', xlabel='',
                    diff_3td_prt=None):
        """Plotting a heat-map which contains the best 3rd parameter (aka
        smallest value for MSE, score and time, and largest value for
        R2-score) or another defined by the user.

        :param metric_arr: str: Name of the metric array the user wants to
                                plot the heat-map.
        :param title: str: Title for the heat-map.
        :param ylabel: str: Label for the y-axis.
        :param xlabel: str: Label for the x-axis.
        :param diff_3td_prt: int: Index of the 3rd parameter to plot the
                                  heat-map. When it is None, it will find the
                                  best 3rd parameter.
        """
        idx = self.argbest(metric_arr=metric_arr)
        p3, p2, p1, a = idx[0], idx[1], idx[2], None

        print(f"Array's best (or defined by the user) indexes for"
              f" {metric_arr}: {idx}")

        if metric_arr == 'mse_train':
            a = self.mse_train

        elif metric_arr == 'mse_test':
            a = self.mse_test

        elif metric_arr == 'r2_train':
            a = self.r2_train

        elif metric_arr == 'r2_test':
            a = self.r2_test

        elif metric_arr == 'loss_score':
            a = self.score

        elif metric_arr == 'time':
            a = self.elapsed

        else:
            raise ValueError('Parameter metric_arr not defined correctly.')

        if diff_3td_prt is not None:
            print(f'The value for {metric_arr} is {a[diff_3td_prt, p2, p1]}')
            print(f'With parameters: {self.prt3[diff_3td_prt]}, '
                  f'{self.prt2[p2]}, {self.prt1[p1]}')

            sns.heatmap(data=a[diff_3td_prt],
                        yticklabels=self.prt2,
                        xticklabels=self.prt1,
                        annot=True,
                        annot_kws={"size": 8.5}, fmt=".2f")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.tight_layout()
            plt.show()

        else:
            print(f'The best value for {metric_arr} is: {a[idx]}')
            print(f'With parameters: {self.prt3[p3]}, '
                  f'{self.prt2[p2]}, {self.prt1[p1]}')

            sns.heatmap(data=a[p3],
                        yticklabels=self.prt2,
                        xticklabels=self.prt1,
                        annot=True,
                        annot_kws={"size": 8.5}, fmt=".2f")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.tight_layout()
            plt.show()

    def heat_map_2d(self, metric_arr, title='', ylabel='', xlabel=''):
        """
        Plotting a heat-map which contains the metric array given by the
        user.

        :param metric_arr: str: Name of the metric array the user wants to
                                plot the heat-map.
        :param title: str: Title for the heat-map.
        :param ylabel: str: Label for the y-axis.
        :param xlabel: str: Label for the x-axis.
        """
        if metric_arr == 'mse_train':
            a = self.mse_train

        elif metric_arr == 'mse_test':
            a = self.mse_test

        elif metric_arr == 'r2_train':
            a = self.r2_train

        elif metric_arr == 'r2_test':
            a = self.r2_test

        elif metric_arr == 'loss_score':
            a = self.score

        elif metric_arr == 'time':
            a = self.elapsed

        else:
            raise ValueError('Parameter metric_arr not defined correctly.')

        sns.heatmap(data=a,
                    yticklabels=self.prt2,
                    xticklabels=self.prt1,
                    annot=True,
                    annot_kws={"size": 8.5}, fmt=".2f")

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.show()
