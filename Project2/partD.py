# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Project2.package.import_data import MNIST, breast_cancer
from Project2.package.deep_neural_network import MLP
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def classification_study(X_train, X_test, y_train, y_test, y_train_encoded,
                         y_test_encoded, n_layers, n_neurons, etas,
                         lmbds, random_state, epoch0, epochs, batch_size,
                         act_function, cost_function):

    acc_train = np.empty(shape=(len(etas), len(n_neurons)))
    acc_test = np.empty(shape=(len(etas), len(n_neurons)))
    elapsed_time = np.empty(shape=(len(etas), len(n_neurons)))
    models = np.empty(shape=(len(etas), len(n_neurons)), dtype=object)

    # searching best parameters
    for c_idx, n in enumerate(n_neurons):
        for r_idx, e in enumerate(etas):
            md = MLP(hidden_layers=[n] * n_layers, epochs=epoch0,
                     batch_size=batch_size, eta0=e,
                     learning_rate='constant', decay=0.0, lmbd=0.0,
                     bias0=0.01, init_weights='normal',
                     act_function=act_function,
                     output_act_function=act_function,
                     cost_function=cost_function, random_state=random_state,
                     verbose=False)

            # fitting
            t0 = time.time()
            md.fit(X_train, y_train_encoded)
            t1 = time.time()

            # predicting
            y_hat = md.predict_class(X_train)
            y_tilde = md.predict_class(X_test)
            acc_hat = accuracy_score(y_train, y_hat)
            acc_tilde = accuracy_score(y_test, y_tilde)

            # saving results
            acc_train[r_idx, c_idx] = acc_hat
            acc_test[r_idx, c_idx] = acc_tilde
            elapsed_time[r_idx, c_idx] = t1 - t0
            models[r_idx, c_idx] = md

            # verbose
            print(f'Accuracy train: {acc_hat}')
            print(f'Accuracy test: {acc_tilde}')

    # getting best arg for acc_train
    best_acc_train = \
        np.unravel_index(np.argmax(acc_train, axis=None), acc_train.shape)

    # getting best arg for acc_test
    best_acc_test = \
        np.unravel_index(np.argmax(acc_test, axis=None), acc_test.shape)

    # printing best testing metric and its localization
    best_n_neuros = n_neurons[best_acc_test[1]]
    best_eta = etas[best_acc_test[0]]

    print(f"Best testing Accuracy is {acc_test[best_acc_test]} at "
          f"indexes {best_acc_test}, with parameters:\n"
          f"n_neurons: {best_n_neuros}\n"
          f"Learning rate (eta): {best_eta}")

    # plotting heat-maps for different metrics
    datas = [acc_train, acc_test, elapsed_time]
    heat_maps = ["Acc_train", "Acc_test", "Time"]

    title = [f"Training Acc-Score X (Neurons x Etas) for "
             f"{n_layers} hidden-layer",
             f"Testing Acc-Score X (Neurons x Etas) for "
             f"{n_layers} hidden-layer",
             f"Elapsed time for training in seconds."]

    for d, h, t in zip(datas, heat_maps, title):
        sns.heatmap(data=d,
                    yticklabels=etas,
                    xticklabels=n_neurons,
                    annot=True,
                    annot_kws={"size": 8.5}, fmt=".2f")

        plt.title(t)
        plt.ylabel('Learning rates')
        plt.xlabel('Number of neurons')
        plt.tight_layout()
        plt.show()

    # plotting loss-error, Train MSE and Test MSE for a given eta.
    plt.plot(n_neurons, acc_train[best_acc_test[0]], "-.",
             label="Training Accuracy")
    plt.plot(n_neurons, acc_test[best_acc_test[0]], "--",
             label="Testing Accuracy")
    plt.ylabel("Accuracy scores")
    plt.xlabel("Number of neurons in the hidden-layer.")
    plt.title("Training and Testing Accuracies as a function of nr. "
              "of neurons")
    plt.legend()
    plt.grid()
    plt.show()

    # Tuning Eta, epoch, lambda
    acc_train_1 = np.empty(shape=(len(lmbds), len(epochs), len(etas)))
    acc_test_1 = np.empty(shape=(len(lmbds), len(epochs), len(etas)))
    elapsed_time_1 = np.empty(shape=(len(lmbds), len(epochs), len(etas)))
    models_1 = np.empty(shape=(len(lmbds), len(epochs), len(etas)),
                        dtype=object)

    for c_idx, e in enumerate(etas):
        for r_idx, ep in enumerate(epochs):
            for d_idx, l in enumerate(lmbds):
                md1 = MLP(hidden_layers=[best_n_neuros] * n_layers,
                          epochs=ep, batch_size=batch_size, eta0=e,
                          learning_rate='constant', decay=0.0, lmbd=l,
                          bias0=0.01, init_weights='normal',
                          act_function=act_function,
                          output_act_function=act_function,
                          cost_function=cost_function,
                          random_state=random_state,
                          verbose=False)

                # fitting
                t0 = time.time()
                md1.fit(X_train, y_train_encoded)
                t1 = time.time()

                # predicting
                y_hat = md1.predict_class(X_train)
                y_tilde = md1.predict_class(X_test)
                acc_hat = accuracy_score(y_train, y_hat)
                acc_tilde = accuracy_score(y_test, y_tilde)

                # saving results
                acc_train_1[d_idx, r_idx, c_idx] = acc_hat
                acc_test_1[d_idx, r_idx, c_idx] = acc_tilde
                elapsed_time_1[d_idx, r_idx, c_idx] = t1 - t0
                models_1[d_idx, r_idx, c_idx] = md1

                # verbose
                print(f'Accuracy train: {acc_hat}')
                print(f'Accuracy test: {acc_tilde}')

    # getting best arg for acc_train
    best_acc_train_1 = \
        np.unravel_index(np.argmax(acc_train_1, axis=None), acc_train_1.shape)

    # getting best arg for acc_test
    best_acc_test_1 = \
        np.unravel_index(np.argmax(acc_test_1, axis=None), acc_test_1.shape)

    # printing best testing metric and its localization
    best_eta_1 = etas[best_acc_test_1[2]]
    best_epoch_1 = epochs[best_acc_test_1[1]]
    best_lambda_1 = lmbds[best_acc_test_1[0]]

    # printing best metric and its localization
    print(f"Best testing MSE is {acc_test_1[best_acc_test_1]} at "
          f"indexes {best_acc_test_1}, with parameters:\n"
          f"Eta: {best_eta_1}\n"
          f"Epoch: {best_epoch_1}\n"
          f"Lambda: {best_lambda_1}\n"
          f"Training time: {elapsed_time_1[best_acc_test_1]}")

    # searching Eta Vs epochs:
    mse_tr_d = []
    mse_te_d = []
    training_time_d = []

    for ep in epochs:
        md2 = MLP(hidden_layers=[best_n_neuros]*n_layers,
                  epochs=ep, batch_size=batch_size,
                  eta0=best_eta_1, learning_rate='constant', decay=0.0,
                  lmbd=0.0, bias0=0.01, init_weights='normal',
                  act_function=act_function, output_act_function=act_function,
                  cost_function=cost_function, random_state=random_state,
                  verbose=False)

        # fitting
        t0 = time.time()
        md2.fit(X_train, y_train_encoded)
        t1 = time.time()

        # predicting
        y_hat = md2.predict_class(X_train)
        y_tilde = md2.predict_class(X_test)
        acc_hat_2 = accuracy_score(y_train, y_hat)
        acc_tilde_2 = accuracy_score(y_test, y_tilde)

        # saving results
        mse_tr_d.append(acc_hat_2)
        mse_te_d.append(acc_tilde_2)
        training_time_d.append(t1-t0)

        # verbose
        print(f'Accuracy train: {acc_hat_2}')
        print(f'Accuracy test: {acc_tilde_2}')

    plt.plot(epochs, mse_tr_d, label='Accuracy Train')
    plt.plot(epochs, mse_te_d, "--", label='Accuracy Test')
    plt.ylabel("Accuracy scores")
    plt.xlabel("Epoch values")
    plt.title("Training and Testing Accuracies Vs Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # searching (Best Eta and epoch) Vs Lambdas;
    mse_te_l = []
    mse_tr_l = []
    training_time_l = []

    for l in lmbds:
        md3 = MLP(hidden_layers=[best_n_neuros]*n_layers,
                  epochs=best_epoch_1, batch_size=batch_size,
                  eta0=best_eta_1, learning_rate='constant', decay=0.0,
                  lmbd=l, bias0=0.01, init_weights='normal',
                  act_function=act_function, output_act_function=act_function,
                  cost_function=cost_function, random_state=random_state,
                  verbose=False)

        # fitting
        t0 = time.time()
        md3.fit(X_train, y_train_encoded)
        t1 = time.time()

        # predicting
        y_hat = md3.predict_class(X_train)
        y_tilde = md3.predict_class(X_test)
        acc_hat_3 = accuracy_score(y_train, y_hat)
        acc_tilde_3 = accuracy_score(y_test, y_tilde)

        # saving results
        mse_tr_l.append(acc_hat_3)
        mse_te_l.append(acc_tilde_3)
        training_time_l.append(t1-t0)

    plt.plot(lmbds, mse_tr_l, label='Accuracy Train')
    plt.plot(lmbds, mse_te_l, "--", label='Accuracy Test')
    plt.ylabel("Accuracy scores")
    plt.xlabel("Lambdas values")
    plt.title("(Training & Testing) Accuracies Vs Lambdas")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # MNIST data-set
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = \
        MNIST(test_size=0.2, shuffle=True, stratify=None, scale_X=True,
              verbose=False, plot=False, random_state=10)

    # studying 1 hidden layer, softmax, accuracy_score, MNIST data-set
    n_layers = 1
    n_neurons = [5, 10, 50, 100, 200, 300, 400, 500]
    etas = [0.2, 0.1, 0.09, 0.05, 0.01]
    epochs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    lmbds = [0, 10**-9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -1, 0.5]
    random_state = 10
    epoch0 = 50
    batch_size = 50
    act_function = 'softmax'
    cost_function = 'accuracy_score'

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 2 hidden layers, softmax, accuracy_score, MNIST data-set
    n_layers = 2
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 1 hidden layer, sigmoid, accuracy_score, MNIST data-set
    n_layers = 1
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    epochs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    lmbds = [0, 10 ** -9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -1, 0.5]
    random_state = 10
    epoch0 = 50
    batch_size = 50
    act_function = 'sigmoid'
    cost_function = 'accuracy_score'

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 2 hidden layers, sigmoid, accuracy_score, MNIST data-set
    n_layers = 2
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # Brest Cancer data-set
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = \
        breast_cancer(test_size=0.2, shuffle=True, stratify=None,
                      scale_X=True, random_state=10)

    # studying 1 hidden layer, softmax, accuracy_score, Brest Cancer data-set
    n_layers = 1
    n_neurons = [5, 10, 50, 100, 200, 300, 400, 500]
    etas = [0.2, 0.1, 0.09, 0.05, 0.01]
    epochs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    lmbds = [0, 10 ** -9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -1, 0.5]
    random_state = 10
    epoch0 = 50
    batch_size = 50
    act_function = 'softmax'
    cost_function = 'accuracy_score'

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 2 hidden layers, softmax, accuracy_score, Brest Cancer data-set
    n_layers = 2
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 1 hidden layer, sigmoid, accuracy_score, Brest Cancer data-set
    n_layers = 1
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    epochs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    lmbds = [0, 10 ** -9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -1, 0.5]
    random_state = 10
    epoch0 = 50
    batch_size = 50
    act_function = 'sigmoid'
    cost_function = 'accuracy_score'

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)

    # studying 2 hidden layers, sigmoid, accuracy_score, Brest Cancer data-set
    n_layers = 2
    etas = [0.2, 0.1, 0.09, 0.05, 0.01, 0.005]
    n_neurons = [5, 10, 25, 50, 75, 100, 250, 500]

    classification_study(X_train=X_train, X_test=X_test, y_train=y_train,
                         y_test=y_test, y_train_encoded=y_train_encoded,
                         y_test_encoded=y_test_encoded, n_layers=n_layers,
                         n_neurons=n_neurons, etas=etas,
                         lmbds=lmbds, random_state=random_state,
                         epoch0=epoch0, epochs=epochs, batch_size=batch_size,
                         act_function=act_function,
                         cost_function=cost_function)
