# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.studies import CV_UNITxEPOCH


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

units = [50, 80, 90, 95, 100, 105, 110, 120, 150, 200, 300]
epochs = [50, 100, 500, 1000]

avg_mse_train_s1, avg_mse_test_s1, avg_mae_train_s1, avg_mae_test_s1 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=1, batch_size=1, activation='sigmoid',
                  random_state=10, verbose=1)

avg_mse_train_s2, avg_mse_test_s2, avg_mae_train_s2, avg_mae_test_s2 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=2, batch_size=1, activation='sigmoid',
                  random_state=10, verbose=1)

avg_mse_train_t1, avg_mse_test_t1, avg_mae_train_t1, avg_mae_test_t1 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=1, batch_size=1, activation='tanh',
                  random_state=10, verbose=1)

avg_mse_train_t2, avg_mse_test_t2, avg_mae_train_t2, avg_mae_test_t2 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=2, batch_size=1, activation='tanh',
                  random_state=10, verbose=1)

avg_mse_train_r1, avg_mse_test_r1, avg_mae_train_r1, avg_mae_test_r1 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=1, batch_size=1, activation='relu',
                  random_state=10, verbose=1)

avg_mse_train_r2, avg_mse_test_r2, avg_mae_train_r2, avg_mae_test_r2 = \
    CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=25,
                  n_hidden_layers=2, batch_size=1, activation='relu',
                  random_state=10, verbose=1)
