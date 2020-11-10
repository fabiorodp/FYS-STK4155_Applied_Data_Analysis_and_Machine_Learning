# UiO: FYS-STK4155 - H20
# Project 2
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Project2.package.metrics import mse
from Project2.package.import_data import terrain_data
from Project2.package.deep_neural_network import MLP
from Project2.package.studies import SearchParametersDNN


def one_hidden_layer_sigmoid(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Printing best metric and its localization;
        3. Plotting heat-maps for different metrics;
        4. Plotting Train MSE x Test MSE for a given eta;
        5. Analysing obtained costs for each epoch when batch_size is the
           length of samples;
        6. Analysing obtained costs for each epoch when batch_size is 100;
        7. Tuning eta x decay and lambda;
        8. Searching Eta Vs Decays;
        9. Searching Eta and Decay=0 Vs Lambdas;
    """
    n_neurons = [10, 20, 30, 40, 50, 75, 100, 125, 150]
    etas = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='normal', act_function='sigmoid',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=False, layers=1, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # plotting heat-maps for different metrics
    heat_maps = ["r2_train", "r2_test", "mse_train", "mse_test",
                 "loss_score", "time"]
    title = ["Training R2-Score X (Neurons x Etas) for 1 hidden-layer",
             "Testing R2-Score X (Neurons x Etas) for 1 hidden-layer",
             "Training MSE-Score X (Neurons x Etas) for 1 hidden-layer",
             "Testing MSE-Score X (Neurons x Etas) for 1 hidden-layer",
             "Last loss-Error (MSE) obtained by the training.",
             "Elapsed time for training in seconds."]

    for h, t in zip(heat_maps, title):
        gs.heat_map_2d(metric_arr=h,
                       title=t,
                       ylabel='Learning rate $\eta$ values',
                       xlabel='Number of Neurons in each hidden layer')

    # plotting loss-error, Train MSE and Test MSE for a given eta.
    plt.plot(n_neurons, gs.score[1], "-.", label="Loss-error")
    plt.plot(n_neurons, gs.mse_test[1], "--", label="MSE test")
    plt.plot(n_neurons, gs.mse_train[1], label="MSE train")
    plt.ylabel("MSE scores")
    plt.xlabel("Number of neurons in the hidden-layer.")
    plt.title("Loss-error, Training MSE and Testing MSE as a function of nr. "
              "of neurons")
    plt.ylim([0, 50])
    plt.legend()
    plt.grid()
    plt.show()

    # analysing batch_size=100
    md = MLP(hidden_layers=[100], epochs=500,
             batch_size=100, eta0=0.9,
             learning_rate='constant', decay=0.0, lmbd=0.0,
             bias0=0.01, init_weights='normal',
             act_function='sigmoid', output_act_function='identity',
             cost_function='mse', random_state=10, verbose=True)

    md.fit(X=X_train, y=z_train)

    # plotting loss x epochs for batch_size=100
    plt.plot(np.arange(500), md.costs, "-.", label="Batch_size = 100")
    # batch_size = length of training samples
    plt.plot(np.arange(500), gs.models[(1, 6)].costs, "--",
             label="Batch_size = length of training samples")

    plt.ylabel("Cost/Loss (MSE) scores for every training epoch")
    plt.xlabel("Epochs / Number of interactions")
    plt.title("Loss (MSE) scores as a function of epochs")
    plt.legend()
    plt.grid()
    plt.ylim([0, 600])
    plt.show()

    # tuning eta x decay and lambda
    etas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5,
              10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    lmbds = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5,
             10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='normal', act_function='sigmoid',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[100])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")

    # plotting heat-plot with the best lambda
    gs1.heat_map_3d(metric_arr='mse_test',
                    title='Testing MSE Vs ($\lambda$=0, decay, $\eta$)',
                    ylabel='Decay Values',
                    xlabel='Learning rate $\eta$ values')

    # bar-plotting selected values
    p_idxs = np.array([(p3, p2, p1)
                       for p3 in lmbds[:3]
                       for p1 in etas[:3]
                       for p2 in decays[:3]], dtype=tuple)

    t_mse = np.hstack(
        [gs1.mse_test[0, :3, 0], gs1.mse_test[0, :3, 1],
         gs1.mse_test[0, :3, 2],
         gs1.mse_test[1, :3, 0], gs1.mse_test[1, :3, 1],
         gs1.mse_test[1, :3, 2],
         gs1.mse_test[2, :3, 0], gs1.mse_test[2, :3, 1],
         gs1.mse_test[2, :3, 2]])

    tr_mse = np.hstack(
        [gs1.mse_train[0, :3, 0], gs1.mse_train[0, :3, 1],
         gs1.mse_train[0, :3, 2],
         gs1.mse_train[1, :3, 0], gs1.mse_train[1, :3, 1],
         gs1.mse_train[1, :3, 2],
         gs1.mse_train[2, :3, 0], gs1.mse_train[2, :3, 1],
         gs1.mse_train[2, :3, 2]])

    df = pd.DataFrame(t_mse, columns=['MSE test'])
    df.index = p_idxs
    df['MSE train'] = tr_mse
    sns.catplot(kind="bar", data=df.transpose())
    plt.title(
        f'(Training and testing MSE scores) X ($\lambda$, decay, $\eta$)')
    plt.ylabel('Training and Testing MSE scores')
    plt.xlabel("('l2' regularization $\lambda$, decay, Learning rate $\eta$) "
               "values")
    plt.xticks(rotation=90)
    plt.show()

    # searching Eta=0.9 Vs Decays:
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]

    mse_te = []
    mse_tr = []
    loss_error = []

    for d in decays:
        md1 = MLP(hidden_layers=[100], epochs=500,
                  batch_size=len(X_train), eta0=0.9,
                  learning_rate='decay', decay=d, lmbd=0.0,
                  bias0=0.01, init_weights='normal',
                  act_function='sigmoid', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md1.fit(X=X_train, y=z_train)
        y_hat = md1.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md1.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md1.costs[-1])

    plt.plot(decays, mse_tr, label='MSE Train')
    plt.plot(decays, mse_te, "--", label='MSE Test')
    plt.plot(decays, loss_error, "-.", label='Loss-error in the last '
                                             'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Decay values")
    plt.title("Loss-error, Training and Testing MSE Vs Decays for $\eta$=0.9")
    plt.legend()
    plt.grid()
    plt.ylim([0, 13])
    plt.show()

    # searching Eta=0.9 and Decay=0 Vs Lambdas;
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    mse_te = []
    mse_tr = []
    loss_error = []

    for l in lmbds:
        md3 = MLP(hidden_layers=[100], epochs=500,
                  batch_size=len(X_train), eta0=0.9,
                  learning_rate='constant', decay=0.0, lmbd=l,
                  bias0=0.01, init_weights='normal',
                  act_function='sigmoid', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md3.fit(X=X_train, y=z_train)
        y_hat = md3.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md3.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md3.costs[-1])

    plt.plot(lmbds, mse_tr, label='MSE Train')
    plt.plot(lmbds, mse_te, "--", label='MSE Test')
    plt.plot(lmbds, loss_error, "-.", label='Loss-error in the last '
                                            'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Lambdas values")
    plt.title("Loss_error, Training & Testing MSE Vs Lambdas")
    plt.legend()
    plt.grid()
    plt.show()


def two_hidden_layer_sigmoid(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Printing best metric and its localization;
        3. Plotting heat-maps for different metrics;
        4. Plotting Train MSE x Test MSE for a given eta;
        5. Tuning eta x decay and lambda;
        6. Tuning Eta=0.08 Vs Decays;
        7. Tuning Eta=0.08 and Decay=0 Vs Lambdas;
    """
    n_neurons = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    etas = [0.1, 0.09, 0.08, 0.05]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=1000, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='normal', act_function='sigmoid',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # plotting heat-maps for different metrics
    heat_maps = ["r2_train", "r2_test", "mse_train", "mse_test",
                 "loss_score", "time"]
    title = ["Training R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Training MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Last loss-Error (MSE) obtained by the training for 2 hidden-layer",
             "Elapsed time for training in seconds for 2 hidden-layer"]

    for h, t in zip(heat_maps, title):
        gs.heat_map_2d(metric_arr=h,
                       title=t,
                       ylabel='Learning rate $\eta$ values',
                       xlabel='Number of Neurons in each hidden layers')

    # plotting loss-error, Train MSE and Test MSE for a given eta.
    plt.plot(n_neurons, gs.score[2], "-.", label="Loss-error")
    plt.plot(n_neurons, gs.mse_test[2], "--", label="MSE test")
    plt.plot(n_neurons, gs.mse_train[2], label="MSE train")
    plt.ylabel("MSE scores")
    plt.xlabel("Number of neurons in the hidden-layers.")
    plt.title("Training MSE and Testing MSE as a function of nr. of neurons")
    plt.ylim([0, 50])
    plt.legend()
    plt.grid()
    plt.show()

    # analysing batch_size=100
    md = MLP(hidden_layers=[500, 500], epochs=500,
             batch_size=100, eta0=0.08,
             learning_rate='constant', decay=0.0, lmbd=0.0,
             bias0=0.01, init_weights='normal',
             act_function='sigmoid', output_act_function='identity',
             cost_function='mse', random_state=10, verbose=True)

    md.fit(X=X_train, y=z_train)

    # plotting loss x epochs for batch_size=100
    plt.plot(np.arange(500), md.costs, "-.", label="Batch_size = 100")
    # batch_size = length of training samples
    plt.plot(np.arange(500), gs.models[(1, 6)].costs, "--",
             label="Batch_size = length of training samples")

    plt.ylabel("Cost/Loss (MSE) scores for every training epoch")
    plt.xlabel("Epochs / Number of interactions")
    plt.title("Loss (MSE) scores as a function of epochs")
    plt.legend()
    plt.grid()
    plt.ylim([0, 600])
    plt.show()

    # tuning eta x decay and lambda
    etas = [0.1, 0.09, 0.08, 0.05]
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='normal', act_function='sigmoid',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[500, 500])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")

    # tuning Eta=0.08 Vs Decays:
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]

    mse_te = []
    mse_tr = []
    loss_error = []

    for d in decays:
        md1 = MLP(hidden_layers=[500, 500], epochs=500,
                  batch_size=len(X_train), eta0=0.08,
                  learning_rate='decay', decay=d, lmbd=0.0,
                  bias0=0.01, init_weights='normal',
                  act_function='sigmoid', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md1.fit(X=X_train, y=z_train)
        y_hat = md1.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md1.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md1.costs[-1])

    plt.plot(decays, mse_tr, label='MSE Train')
    plt.plot(decays, mse_te, "--", label='MSE Test')
    plt.plot(decays, loss_error, "-.", label='Loss-error in the last '
                                             'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Decay values")
    plt.title("Training and Testing MSE Vs Decays for $\eta$=0.08")
    plt.legend()
    plt.grid()
    plt.ylim([0, 20])
    plt.show()

    # tuning Eta=0.08 and Decay=0 Vs Lambdas;
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    mse_te = []
    mse_tr = []
    loss_error = []

    for l in lmbds:
        md2 = MLP(hidden_layers=[500, 500], epochs=500,
                  batch_size=len(X_train), eta0=0.08,
                  learning_rate='constant', decay=0.0, lmbd=l,
                  bias0=0.01, init_weights='normal',
                  act_function='sigmoid', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md2.fit(X=X_train, y=z_train)
        y_hat = md2.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md2.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md2.costs[-1])

    plt.plot(lmbds, mse_tr, label='MSE Train')
    plt.plot(lmbds, mse_te, "--", label='MSE Test')
    plt.plot(lmbds, loss_error, "-.", label='Loss-error in the last '
                                            'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Lambdas values")
    plt.title("Training & Testing MSE Vs Lambdas for $\eta$=0.08 and decay=0")
    plt.legend()
    plt.ylim([0, 12])
    plt.grid()
    plt.show()


def two_hidden_layer_tanh(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Printing best metric and its localization;
        3. Plotting heat-maps for different metrics;
        4. Plotting Train MSE x Test MSE for a given eta;
        5. Tuning eta x decay and lambda;
        6. Tuning Eta=0.05 Vs Decays;
        7. Tuning Eta=0.05 and Decay=0 Vs Lambdas;
    """
    n_neurons = [5, 10, 25, 50, 100, 150, 200, 300, 400, 500]
    etas = [0.3, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='normal', act_function='tanh',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # plotting heat-maps for different metrics
    heat_maps = ["r2_train", "r2_test", "mse_train", "mse_test",
                 "loss_score", "time"]
    title = ["Training R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Training MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Last loss-error (MSE) obtained by training for 2 hidden-layer",
             "Elapsed time for training in seconds for 2 hidden-layer"]

    for h, t in zip(heat_maps, title):
        gs.heat_map_2d(metric_arr=h,
                       title=t,
                       ylabel='Learning rate $\eta$ values',
                       xlabel='Number of Neurons in each hidden layers')

    # plotting loss-error, Train MSE and Test MSE for a given eta.
    plt.plot(n_neurons, gs.score[4], "-.", label="Loss-error")
    plt.plot(n_neurons, gs.mse_test[4], "--", label="MSE test")
    plt.plot(n_neurons, gs.mse_train[4], label="MSE train")
    plt.ylabel("MSE scores")
    plt.xlabel("Number of neurons in the hidden-layers.")
    plt.title(
        "Training MSE and Testing MSE as a function of nr. of neurons")
    plt.ylim([0, 50])
    plt.legend()
    plt.grid()
    plt.show()

    # analysing batch_size=100
    md = MLP(hidden_layers=[400, 400], epochs=500,
             batch_size=100, eta0=0.04,
             learning_rate='constant', decay=0.0, lmbd=0.0,
             bias0=0.01, init_weights='normal',
             act_function='tanh', output_act_function='identity',
             cost_function='mse', random_state=10, verbose=True)

    md.fit(X=X_train, y=z_train)

    # plotting loss x epochs for batch_size=100
    plt.plot(np.arange(500), md.costs, "-.", label="Batch_size = 100")
    # batch_size = length of training samples
    plt.plot(np.arange(500), gs.models[(4, 8)].costs, "--",
             label="Batch_size = length of training samples")

    plt.ylabel("Cost/Loss (MSE) scores for every training epoch")
    plt.xlabel("Epochs / Number of interactions")
    plt.title("Loss (MSE) scores as a function of epochs")
    plt.legend()
    plt.grid()
    plt.ylim([0, 600])
    plt.show()

    # tuning eta x decay and lambda
    etas = [0.1, 0.07, 0.06, 0.05, 0.04, 0.03]
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='normal', act_function='tanh',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[400, 400])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")

    # tuning Eta=0.05 Vs Decays:
    decays = [0, 10 ** -8, 10 ** -6, 10 ** -5]

    mse_te = []
    mse_tr = []
    loss_error = []

    for d in decays:
        md1 = MLP(hidden_layers=[400, 400], epochs=500,
                  batch_size=len(X_train), eta0=0.04,
                  learning_rate='decay', decay=d, lmbd=0.0,
                  bias0=0.01, init_weights='normal',
                  act_function='tanh', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md1.fit(X=X_train, y=z_train)
        y_hat = md1.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md1.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md1.costs[-1])

    plt.plot(decays, mse_tr, label='MSE Train')
    plt.plot(decays, mse_te, "--", label='MSE Test')
    plt.plot(decays, loss_error, "-.", label='Loss-error in the last '
                                             'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Decay values")
    plt.title("Training and Testing MSE Vs Decays for $\eta$=0.04")
    plt.legend()
    plt.grid()
    plt.ylim([0, 20])
    plt.show()

    # tuning Eta=0.04 and Decay=1e-6 Vs Lambdas;
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    mse_te = []
    mse_tr = []
    loss_error = []

    for l in lmbds:
        md2 = MLP(hidden_layers=[400, 400], epochs=500,
                  batch_size=len(X_train), eta0=0.04,
                  learning_rate='decay', decay=1e-6, lmbd=l,
                  bias0=0.01, init_weights='normal',
                  act_function='tanh', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md2.fit(X=X_train, y=z_train)
        y_hat = md2.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md2.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md2.costs[-1])

    plt.plot(lmbds, mse_tr, label='MSE Train')
    plt.plot(lmbds, mse_te, "--", label='MSE Test')
    plt.plot(lmbds, loss_error, "-.", label='Loss-error in the last '
                                            'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Lambdas values")
    plt.title("Training & Testing MSE Vs lambds")
    plt.legend()
    plt.ylim([0, 12])
    plt.grid()
    plt.show()


def two_hidden_layer_relu(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Printing best metric and its localization;
        3. Plotting heat-maps for different metrics;
        4. Plotting Train MSE x Test MSE for a given eta;
        5. Tuning eta x decay and lambda;
        6. Tuning Eta=0.05 Vs Decays;
        7. Tuning Eta=0.05 and Decay=0 Vs Lambdas;
    """
    n_neurons = [7, 8, 9, 10, 14, 15]
    etas = [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005,
            0.0004, 0.0003, 0.0002, 0.0001]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='normal', act_function='relu',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # plotting heat-maps for different metrics
    heat_maps = ["r2_train", "r2_test", "mse_train", "mse_test",
                 "loss_score", "time"]
    title = ["Training R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing R2-Score X (Neurons x Etas) for 2 hidden-layer",
             "Training MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Testing MSE-Score X (Neurons x Etas) for 2 hidden-layer",
             "Last loss-Error (MSE) obtained by the training for 2 hidden-layer",
             "Elapsed time for training in seconds for 2 hidden-layer"]

    for h, t in zip(heat_maps, title):
        gs.heat_map_2d(metric_arr=h,
                       title=t,
                       ylabel='Learning rate $\eta$ values',
                       xlabel='Number of Neurons in each hidden layers')

    # plotting loss-error, Train MSE and Test MSE for a given eta.
    plt.plot(n_neurons[:-1], gs.score[5][:-1], "-.", label="Loss-error")
    plt.plot(n_neurons[:-1], gs.mse_test[5][:-1], "--", label="MSE test")
    plt.plot(n_neurons[:-1], gs.mse_train[5][:-1], label="MSE train")
    plt.ylabel("MSE scores")
    plt.xlabel("Number of neurons in the hidden-layers.")
    plt.title(
        "Training MSE and Testing MSE as a function of nr. of neurons")
    plt.ylim([0, 50])
    plt.legend()
    plt.grid()
    plt.show()

    # analysing batch_size=100
    md = MLP(hidden_layers=[9, 9], epochs=500,
             batch_size=100, eta0=0.0005,
             learning_rate='constant', decay=0.0, lmbd=0.0,
             bias0=0.01, init_weights='normal',
             act_function='relu', output_act_function='identity',
             cost_function='mse', random_state=10, verbose=True)

    md.fit(X=X_train, y=z_train)

    # plotting loss x epochs for batch_size=100
    plt.plot(np.arange(500), md.costs, "-.", label="Batch_size = 100")
    # batch_size = length of training samples
    plt.plot(np.arange(500), gs.models[(5, 2)].costs, "--",
             label="Batch_size = length of training samples")

    plt.ylabel("Cost/Loss (MSE) scores for every training epoch")
    plt.xlabel("Epochs / Number of interactions")
    plt.title("Loss (MSE) scores as a function of epochs")
    plt.legend()
    plt.grid()
    plt.ylim([0, 600])
    plt.show()

    # tuning eta x decay and lambda
    etas = [0.0006, 0.0005, 0.0004]
    decays = [0, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4]
    lmbds = [0, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='normal', act_function='relu',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[9, 9])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")

    # tuning Eta=0.0005 Vs Decays:
    decays = [0, 10 ** -7, 10 ** -6, 10 ** -5]

    mse_te = []
    mse_tr = []
    loss_error = []

    for d in decays:
        md1 = MLP(hidden_layers=[9, 9], epochs=500,
                  batch_size=len(X_train), eta0=0.0005,
                  learning_rate='decay', decay=d, lmbd=0.0,
                  bias0=0.01, init_weights='normal',
                  act_function='relu', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md1.fit(X=X_train, y=z_train)
        y_hat = md1.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md1.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md1.costs[-1])

    plt.plot(decays, mse_tr, label='MSE Train')
    plt.plot(decays, mse_te, "--", label='MSE Test')
    plt.plot(decays, loss_error, "-.", label='Loss-error in the last '
                                             'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Decay values")
    plt.title("Training and Testing MSE Vs Decays")
    plt.legend()
    plt.grid()
    plt.ylim([0, 100])
    plt.show()

    # tuning Eta=0.0005 and Decay=1e-5 Vs Lambdas;
    lmbds = [0, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99]

    mse_te = []
    mse_tr = []
    loss_error = []

    for l in lmbds:
        md2 = MLP(hidden_layers=[9, 9], epochs=500,
                  batch_size=len(X_train), eta0=0.0005,
                  learning_rate='decay', decay=1e-6, lmbd=l,
                  bias0=0.01, init_weights='normal',
                  act_function='relu', output_act_function='identity',
                  cost_function='mse', random_state=10, verbose=True)

        md2.fit(X=X_train, y=z_train)
        y_hat = md2.predict(X_train)
        mse_tr.append(mse(y_true=z_train, y_hat=y_hat))
        y_tilde = md2.predict(X_test)
        mse_te.append(mse(y_true=z_test, y_hat=y_tilde))
        loss_error.append(md2.costs[-1])

    plt.plot(lmbds, mse_tr, label='MSE Train')
    plt.plot(lmbds, mse_te, "--", label='MSE Test')
    plt.plot(lmbds, loss_error, "-.", label='Loss-error in the last '
                                            'training epoch')
    plt.ylabel("MSE scores")
    plt.xlabel("Lambdas values")
    plt.title("Training & Testing MSE Vs lambds")
    plt.legend()
    plt.ylim([0, 100])
    plt.grid()
    plt.show()


def two_hidden_layer_relu_xavier(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Tuning eta x decay and lambda;
    """
    n_neurons = [7, 9, 15, 50, 75, 100, 250, 500]
    etas = [0.001, 0.00075, 0.0005, 0.0004]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='xavier', act_function='relu',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # tuning eta x decay and lambda
    etas = [0.001, 0.00075, 0.0005, 0.0004]
    decays = [0, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    lmbds = [0, 10 ** -2, 10 ** -1, 0.5, 0.9, 0.99]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='xavier', act_function='relu',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[500, 500])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")


def two_hidden_layer_tanh_xavier(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Tuning eta x decay and lambda;
    """
    n_neurons = [5, 10, 25, 50, 100, 150, 200, 300, 400, 500]
    etas = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='xavier', act_function='tanh',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # tuning eta x decay and lambda
    etas = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 0.5]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='xavier', act_function='tanh',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[500, 500])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")


def two_hidden_layer_sigmoid_xavier(X_train, X_test, z_train, z_test):
    """
    In this test we study:
        1. Searching best parameters for 'NEURONSxETAS';
        2. Tuning eta x decay x lambda;
    """
    n_neurons = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    etas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    gs = SearchParametersDNN(params='NEURONSxETAS', random_state=10)

    # searching best parameters for 'NEURONSxETAS'
    gs.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
           model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
           learning_rate='constant', decays=0.0, lmbds=0.0, bias0=0.01,
           init_weights='xavier', act_function='sigmoid',
           output_act_function='identity', cost_function='mse',
           random_state=10, verbose=True, layers=2, neurons=n_neurons,
           hidden_layers=None)

    # printing best metric and its localization
    print(f"Best testing MSE is {gs.mse_test[gs.argbest('mse_test')]} at "
          f"indexes {gs.argbest('mse_test')}, with parameters:\n"
          f"n_neurons: {gs.prt1[gs.argbest('mse_test')[1]]}\n"
          f"Learning rate (eta): {gs.prt2[gs.argbest('mse_test')[0]]}")

    # tuning eta x decay and lambda
    etas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    decays = [0, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    lmbds = [0, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 0.5]

    gs1 = SearchParametersDNN(params='ETASxDECAYSxLAMBDAS', random_state=10)

    # searching best parameters for ETASxDECAYSxLAMBDAS'
    gs1.run(X_train=X_train, X_test=X_test, z_train=z_train, z_test=z_test,
            model=MLP, epochs=500, batch_size=len(X_train), etas=etas,
            learning_rate='decay', decays=decays, lmbds=lmbds, bias0=0.01,
            init_weights='xavier', act_function='sigmoid',
            output_act_function='identity', cost_function='mse',
            random_state=10, verbose=True, layers=None, neurons=None,
            hidden_layers=[300, 300])

    # printing best metric and its localization
    print(f"Best testing MSE is {gs1.mse_test[gs1.argbest('mse_test')]} at "
          f"indexes {gs1.argbest('mse_test')}, with parameters:\n"
          f"Eta: {gs1.prt1[gs1.argbest('mse_test')[2]]}\n"
          f"Decay: {gs1.prt2[gs1.argbest('mse_test')[1]]}\n"
          f"Lambda: {gs1.prt3[gs1.argbest('mse_test')[0]]}")


if __name__ == '__main__':
    # ########## Importing data
    X_train, X_test, z_train, z_test = \
        terrain_data(file='Project2/data/SRTM_data_Norway_1.tif',
                     slice_size=15, test_size=0.2, shuffle=True,
                     stratify=None, scale_X=True,
                     scale_z=False, random_state=10)

    # 1st study - 1 hidden-layer - sigmoid
    one_hidden_layer_sigmoid(X_train, X_test, z_train, z_test)

    # 2nd study - 2 hidden-layers - sigmoid
    two_hidden_layer_sigmoid(X_train, X_test, z_train, z_test)

    # 3rd study - 2 hidden-layers - tanh
    two_hidden_layer_tanh(X_train, X_test, z_train, z_test)

    # 4th study - 2 hidden-layers - relu
    two_hidden_layer_relu(X_train, X_test, z_train, z_test)

    # 5th study - 2 hidden-layers - relu and xavier
    two_hidden_layer_relu_xavier(X_train, X_test, z_train, z_test)

    # 6th study - 2 hidden-layers - tanh and xavier
    two_hidden_layer_relu_xavier(X_train, X_test, z_train, z_test)

    # 7th study - 2 hidden-layers - sigmoid and xavier
    two_hidden_layer_relu_xavier(X_train, X_test, z_train, z_test)

# model = MLPRegressor(hidden_layer_sizes=(50,), activation='logistic',
#                      solver='sgd', alpha=0.0, batch_size=len(X_train),
#                      learning_rate='constant', learning_rate_init=0.1,
#                      max_iter=100, shuffle=True, random_state=10,
#                      verbose=True, n_iter_no_change=100)
# model.fit(X_train, z_train.ravel())
