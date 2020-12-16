# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.studies import LSTM_CV_UNITxEPOCH


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

units = [50, 100, 150, 200, 400, 600, 800, 1000, 1200]
epochs = [50]

# ########### training and CV predicting target high
mse_train_h_s1, mse_test_h_s1, mae_train_h_s1, mae_test_h_s1, y_pred_h_s1 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=1, batch_size=1,
                       recurrent_activation='hard_sigmoid',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_h_s2, mse_test_h_s2, mae_train_h_s2, mae_test_h_s2, y_pred_h_s2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='hard_sigmoid',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_h_t1, mse_test_h_t1, mae_train_h_t1, mae_test_h_t1, y_pred_h_t1 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=1, batch_size=1,
                       recurrent_activation='tanh',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_h_t2, mse_test_h_t2, mae_train_h_t2, mae_test_h_t2, y_pred_h_t2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='tanh',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_h_r2, mse_test_h_r2, mae_train_h_r2, mae_test_h_r2, y_pred_h_r2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='relu',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_h_r3, mse_test_h_r3, mae_train_h_r3, mae_test_h_r3, y_pred_h_r3 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                       n_hidden_layers=3, batch_size=1,
                       recurrent_activation='relu',
                       activation='tanh',
                       random_state=10, verbose=1)
# ###########

# ########### training and CV predicting target low
mse_train_l_s1, mse_test_l_s1, mae_train_l_s1, mae_test_l_s1, y_pred_l_s1 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=1, batch_size=1,
                       recurrent_activation='hard_sigmoid',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_l_s2, mse_test_l_s2, mae_train_l_s2, mae_test_l_s2, y_pred_l_s2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='hard_sigmoid',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_l_t1, mse_test_l_t1, mae_train_l_t1, mae_test_l_t1, y_pred_l_t1 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=1, batch_size=1,
                       recurrent_activation='tanh',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_l_t2, mse_test_l_t2, mae_train_l_t2, mae_test_l_t2, y_pred_l_t2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='tanh',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_l_r2, mse_test_l_r2, mae_train_l_r2, mae_test_l_r2, y_pred_l_r2 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=2, batch_size=1,
                       recurrent_activation='relu',
                       activation='tanh',
                       random_state=10, verbose=1)

mse_train_l_r3, mse_test_l_r3, mae_train_l_r3, mae_test_l_r3, y_pred_l_r3 = \
    LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='low', rolling=60,
                       n_hidden_layers=3, batch_size=1,
                       recurrent_activation='relu',
                       activation='tanh',
                       random_state=10, verbose=1)
# ###########
