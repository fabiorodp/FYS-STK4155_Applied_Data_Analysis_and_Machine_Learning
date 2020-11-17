# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Project2.package.import_data import breast_cancer
from Project2.package.gradient_descent import LR
from sklearn.metrics import accuracy_score


def classification_study(X_train, X_test, y_train, y_test,
                         etas, lmbds, random_state, epochs, batch_size):

    acc_train = np.empty(shape=(len(etas), len(epochs)))
    acc_test = np.empty(shape=(len(etas), len(epochs)))
    elapsed_time = np.empty(shape=(len(etas), len(epochs)))
    models = np.empty(shape=(len(etas), len(epochs)), dtype=object)

    # searching best parameters
    for c_idx, n in enumerate(epochs):
        for r_idx, e in enumerate(etas):
            md = LR(batch_size=batch_size, epochs=n, eta0=e, decay=0.0,
                    lambda_=0.0, gamma=0.0, regularization="l2",
                    random_state=random_state)

            # fitting
            t0 = time.time()
            md.fit(X_train, y_train)
            t1 = time.time()

            # predicting
            y_hat = md.predict(X_train)
            y_tilde = md.predict(X_test)
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

    # getting best arg for acc_test
    best_acc_test = \
        np.unravel_index(np.argmax(acc_test, axis=None), acc_test.shape)

    # printing best testing metric and its localization
    best_epoch = epochs[best_acc_test[1]]
    best_eta = etas[best_acc_test[0]]

    print(f"Best testing Accuracy is {acc_test[best_acc_test]} at "
          f"indexes {best_acc_test}, with parameters:\n"
          f"Epoch: {best_epoch}\n"
          f"Learning rate (eta): {best_eta}")

    # plotting heat-maps for different metrics
    datas = [acc_train, acc_test, elapsed_time]
    heat_maps = ["Acc_train", "Acc_test", "Time"]

    title = [f"Training Acc-Score X (Epochs x Etas)",
             f"Testing Acc-Score X (Epochs x Etas)",
             f"Elapsed time for training in seconds."]

    for d, h, t in zip(datas, heat_maps, title):
        sns.heatmap(data=d,
                    yticklabels=etas,
                    xticklabels=epochs,
                    annot=True,
                    annot_kws={"size": 8.5}, fmt=".2f")

        plt.title(t)
        plt.ylabel('Learning rates')
        plt.xlabel('Number of epochs')
        plt.tight_layout()
        plt.show()

    # plotting Train and Test accuracies for a given eta.
    plt.plot(epochs, acc_train[best_acc_test[0]], "-.",
             label="Training Accuracy")
    plt.plot(epochs, acc_test[best_acc_test[0]], "--",
             label="Testing Accuracy")
    plt.ylabel("Accuracy scores")
    plt.xlabel("Number of epochs.")
    plt.title("Training and Testing Accuracies as a function of nr. "
              "of epochs")
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
                md1 = LR(batch_size=batch_size, epochs=ep, eta0=e, decay=0.0,
                         lambda_=l, gamma=0.0, regularization="l2",
                         random_state=random_state)

                # fitting
                t0 = time.time()
                md1.fit(X_train, y_train)
                t1 = time.time()

                # predicting
                y_hat = md1.predict(X_train)
                y_tilde = md1.predict(X_test)
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

    # getting best arg for acc_test
    best_acc_test_1 = \
        np.unravel_index(np.argmax(acc_test_1, axis=None), acc_test_1.shape)

    # printing best testing metric and its localization
    best_eta_1 = etas[best_acc_test_1[2]]
    best_epoch_1 = epochs[best_acc_test_1[1]]
    best_lambda_1 = lmbds[best_acc_test_1[0]]

    # printing best metric and its localization
    print(f"Best testing accuracy is {acc_test_1[best_acc_test_1]} at "
          f"indexes {best_acc_test_1}, with parameters:\n"
          f"Eta: {best_eta_1}\n"
          f"Epoch: {best_epoch_1}\n"
          f"Lambda: {best_lambda_1}\n"
          f"Training time: {elapsed_time_1[best_acc_test_1]}")

    # searching (Best Eta and epoch) Vs Lambdas;
    mse_te_l = []
    mse_tr_l = []
    training_time_l = []

    for l in lmbds:
        md3 = LR(batch_size=batch_size, epochs=best_epoch_1, eta0=best_eta_1,
                 decay=0.0, lambda_=l, gamma=0.0, regularization="l2",
                 random_state=random_state)

        # fitting
        t0 = time.time()
        md3.fit(X_train, y_train)
        t1 = time.time()

        # predicting
        y_hat = md3.predict(X_train)
        y_tilde = md3.predict(X_test)
        acc_hat_3 = accuracy_score(y_train, y_hat)
        acc_tilde_3 = accuracy_score(y_test, y_tilde)

        # saving results
        mse_tr_l.append(acc_hat_3)
        mse_te_l.append(acc_tilde_3)
        training_time_l.append(t1 - t0)

    plt.plot(lmbds, mse_tr_l, label='Accuracy Train')
    plt.plot(lmbds, mse_te_l, "--", label='Accuracy Test')
    plt.ylabel("Accuracy scores")
    plt.xlabel("Lambdas values")
    plt.title("(Training & Testing) Accuracies Vs Lambdas")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # ##############################
    # ############################## Breast cancer data-set
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = \
        breast_cancer(test_size=0.2, shuffle=True, stratify=None,
                      scale_X=True, random_state=10)

    # parameters
    etas = [3, 2, 1, 0.1, 0.05, 0.01]
    epochs = [5, 10, 25, 50, 100, 250, 500]
    lmbds = [0, 10**-9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -1, 0.5]
    random_state = 10
    batch_size = 25

    # calling the study
    classification_study(X_train, X_test, y_train, y_test,
                         etas, lmbds, random_state, epochs, batch_size)
