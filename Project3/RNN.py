# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.studies import RNN_CV_UNITxEPOCH
import matplotlib.pyplot as plt


# ########### reading OLHCV data
data = pd.read_csv(
    filepath_or_buffer='Project3/data/PETR4_1D_OLHCV.csv',
    sep=';'
)

# ########### engineering features
freqs = [5, 10, 20]
for f in freqs:
    data = bollinger_bands(data, freq=f, std_value=2.0, column_base='close')
    data = ema(data, freq=f, column_base='close')

# ########### dropping Nan values
data.dropna(inplace=True)

units = [15, 50, 100, 150, 200, 400, 600, 800, 1000, 1200]
epochs = [50]

# ########### training and CV predicting target high
mse_train_h_s1, mse_test_h_s1, mae_train_h_s1, mae_test_h_s1 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=1, batch_size=1, activation='sigmoid',
                      random_state=10, verbose=1)

mse_train_h_s2, mse_test_h_s2, mae_train_h_s2, mae_test_h_s2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=2, batch_size=1, activation='sigmoid',
                      random_state=10, verbose=1)

mse_train_h_t1, mse_test_h_t1, mae_train_h_t1, mae_test_h_t1 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=1, batch_size=1, activation='tanh',
                      random_state=10, verbose=1)

mse_train_h_t2, mse_test_h_t2, mae_train_h_t2, mae_test_h_t2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=2, batch_size=1, activation='tanh',
                      random_state=10, verbose=1)

mse_train_h_r2, mse_test_h_r2, mae_train_h_r2, mae_test_h_r2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=2, batch_size=2, activation='relu',
                      random_state=10, verbose=1)

mse_train_h_r3, mse_test_h_r3, mae_train_h_r3, mae_test_h_r3 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high',
                      rolling=60, n_hidden_layers=3, batch_size=1,
                      activation='relu', random_state=10, verbose=1)
# ###########

# ########### plotting comparison for highest
plt.plot(units, mse_test_h_s1, '.-',
         label='Testing MSE for Sigmoid with 1 hidden-layer')
plt.plot(units, mse_test_h_s2, '.-',
         label='Testing MSE for Sigmoid with 2 hidden-layer')
plt.plot(units, mse_test_h_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.plot(units, mse_test_h_t2, '--',
         label='Testing MSE for Tanh with 2 hidden-layer')
plt.plot(units, mse_test_h_r2,
         label='Testing MSE for ReLu with 2 hidden-layer')
plt.plot(units, mse_test_h_r3,
         label='Testing MSE for ReLu with 3 hidden-layer')
plt.title('Comparison among models by their validated MSE')
plt.ylabel('Validated MSE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0.4, 1.5])
plt.show()

plt.plot(units, mse_train_h_t1,
         '--',
         label='Train MSE for Tanh with 1 hidden-layer')
plt.plot(units, mse_test_h_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.title('Train Vs Test MSE for predicting the highest of '
          'the next day.')
plt.ylabel('Training and Validated MSE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0, 2])
plt.show()

plt.plot(units, mae_test_h_s1, '.-',
         label='Testing MSE for Sigmoid with 1 hidden-layer')
plt.plot(units, mae_test_h_s2, '.-',
         label='Testing MSE for Sigmoid with 2 hidden-layer')
plt.plot(units, mae_test_h_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.plot(units, mae_test_h_t2, '--',
         label='Testing MSE for Tanh with 2 hidden-layer')
plt.plot(units, mae_test_h_r2,
         label='Testing MSE for ReLu with 2 hidden-layer')
plt.plot(units, mae_test_h_r3,
         label='Testing MSE for ReLu with 3 hidden-layer')
plt.title('Comparison among models by their validated MAE')
plt.ylabel('Validated MAE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0.3, 1.5])
plt.show()

plt.plot(units, mae_train_h_t1,
         '--',
         label='Train MAE for Tanh with 1 hidden-layer')
plt.plot(units, mae_test_h_t1, '--',
         label='Testing MAE for Tanh with 1 hidden-layer')
plt.title('Train Vs Test MAE for predicting the highest of '
          'the next day.')
plt.ylabel('Training and Validated MAE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0, 2])
plt.show()
# ###########

# ########### training and CV predicting target low
mse_train_l_s1, mse_test_l_s1, mae_train_l_s1, mae_test_l_s1 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                      n_hidden_layers=1, batch_size=1, activation='sigmoid',
                      random_state=10, verbose=1)

mse_train_l_s2, mse_test_l_s2, mae_train_l_s2, mae_test_l_s2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                      n_hidden_layers=2, batch_size=1, activation='sigmoid',
                      random_state=10, verbose=1)

mse_train_l_t1, mse_test_l_t1, mae_train_l_t1, mae_test_l_t1 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                      n_hidden_layers=1, batch_size=1, activation='tanh',
                      random_state=10, verbose=1)

mse_train_l_t2, mse_test_l_t2, mae_train_l_t2, mae_test_l_t2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                      n_hidden_layers=2, batch_size=1, activation='tanh',
                      random_state=10, verbose=1)

mse_train_l_r2, mse_test_l_r2, mae_train_l_r2, mae_test_l_r2 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                      n_hidden_layers=2, batch_size=2, activation='relu',
                      random_state=10, verbose=1)

mse_train_l_r3, mse_test_l_r3, mae_train_l_r3, mae_test_l_r3 = \
    RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='low',
                      rolling=60, n_hidden_layers=3, batch_size=1,
                      activation='relu', random_state=10, verbose=1)
# ###########

# ########### plotting comparison for lowest
plt.plot(units, mse_test_l_s1, '.-',
         label='Testing MSE for Sigmoid with 1 hidden-layer')
plt.plot(units, mse_test_l_s2, '.-',
         label='Testing MSE for Sigmoid with 2 hidden-layer')
plt.plot(units, mse_test_l_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.plot(units, mse_test_l_t2, '--',
         label='Testing MSE for Tanh with 2 hidden-layer')
plt.plot(units, mse_test_l_r2,
         label='Testing MSE for ReLu with 2 hidden-layer')
plt.plot(units, mse_test_l_r3,
         label='Testing MSE for ReLu with 3 hidden-layer')
plt.title('Comparison among models by their validated MSE')
plt.ylabel('Validated MSE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0.4, 1.5])
plt.show()

plt.plot(units, mse_train_l_t1,
         '--',
         label='Train MSE for Tanh with 1 hidden-layer')
plt.plot(units, mse_test_l_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.title('Train Vs Test MSE for predicting the lowest of '
          'the next day.')
plt.ylabel('Training and Validated MSE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0, 2])
plt.show()

plt.plot(units, mae_test_l_s1, '.-',
         label='Testing MSE for Sigmoid with 1 hidden-layer')
plt.plot(units, mae_test_l_s2, '.-',
         label='Testing MSE for Sigmoid with 2 hidden-layer')
plt.plot(units, mae_test_l_t1, '--',
         label='Testing MSE for Tanh with 1 hidden-layer')
plt.plot(units, mae_test_l_t2, '--',
         label='Testing MSE for Tanh with 2 hidden-layer')
plt.plot(units, mae_test_l_r2,
         label='Testing MSE for ReLu with 2 hidden-layer')
plt.plot(units, mae_test_l_r3,
         label='Testing MSE for ReLu with 3 hidden-layer')
plt.title('Comparison among models by their validated MAE')
plt.ylabel('Validated MAE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0.3, 1.5])
plt.show()

plt.plot(units, mae_train_l_t1,
         '--',
         label='Train MAE for Tanh with 1 hidden-layer')
plt.plot(units, mae_test_l_t1, '--',
         label='Testing MAE for Tanh with 1 hidden-layer')
plt.title('Train Vs Test MAE for predicting the lowest of '
          'the next day.')
plt.ylabel('Training and Validated MAE')
plt.xlabel('Number of Units')
plt.legend()
plt.grid()
plt.ylim([0, 2])
plt.show()
# ###########
