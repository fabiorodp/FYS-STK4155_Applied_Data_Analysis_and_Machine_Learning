# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.trade_systems import BB_strategy


# ########### reading 1D OLHCV data
data_1D = pd.read_csv(
    filepath_or_buffer='data/PETR4_1D_OLHCV.csv',
    sep=';'
)

# ########### engineering features
freqs = [5, 10, 20]
for f in freqs:
    data_1D = \
        bollinger_bands(data_1D, freq=f, std_value=2.0, column_base='close')
    data_1D = ema(data_1D, freq=f, column_base='close')

# ########### dropping Nan values
data_1D.dropna(inplace=True)

# ########### reading 15min OLHCV data
data_15min = pd.read_csv(
    filepath_or_buffer='data/PETR4_15min_OLHCV.csv',
    sep=';'
)

# ########### Bollinger Bands algorithm trading strategy
BB_all_valid_trades, BB_all_triggered_trades = \
    BB_strategy(data_1D=data_1D, data_15min=data_15min)

BB_all_valid_trades = pd.DataFrame(BB_all_valid_trades)
BB_all_triggered_trades = pd.DataFrame(BB_all_triggered_trades)

profit = 0
for t in BB_all_valid_trades['position']:
    profit += sum(t)

print(f'The profit achieved by Bollinger Bands algorithm '
      f'trading strategy was R${profit} per share of PETR4.')
# ###########

# ########### Our algorithm trading strategy using RNN or LSTM

# ###########
