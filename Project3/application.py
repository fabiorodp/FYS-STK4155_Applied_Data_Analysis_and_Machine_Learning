# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.trade_systems import algo_trading
from Project3.package.engineering_features import get_predicted_features


# ########### reading 1D OLHCV data
data_1D = pd.read_csv(
    filepath_or_buffer='Project3/data/PETR4_1D_OLHCV.csv',
    sep=';'
)

# ########### reading 15min OLHCV data
data_15min = pd.read_csv(
    filepath_or_buffer='Project3/data/PETR4_15min_OLHCV.csv',
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

# ########### getting predicted features from RNN
y_hat_highest, y_hat_lowest = \
    get_predicted_features(
        data_1D=data_1D, units=[800], epochs=[10], rolling=60, out_csv=None)

# ########### adding predicted features from RNN
period_1D = data_1D[-37:-1]
period_1D['Predicted High'] = y_hat_highest
period_1D['Predicted Low'] = y_hat_lowest

# ########### Bollinger Bands algorithm trading strategy
BB_all_valid_trades, BB_all_triggered_trades = \
    algo_trading(data_1D=period_1D, data_15min=data_15min, strategy='5 BB')

BB_all_valid_trades = pd.DataFrame(BB_all_valid_trades)
BB_all_triggered_trades = pd.DataFrame(BB_all_triggered_trades)

BB_profit = 0
for t in BB_all_valid_trades['position']:
    BB_profit += sum(t)

print(f'The profit achieved by Bollinger Bands algorithm '
      f'trading strategy was R${BB_profit} per share of PETR4.')
# ###########

# ########### Our algorithm trading strategy
our_all_valid_trades, our_all_triggered_trades = \
    algo_trading(data_1D=period_1D, data_15min=data_15min, strategy='Our')

our_all_valid_trades = pd.DataFrame(our_all_valid_trades)
our_all_triggered_trades = pd.DataFrame(our_all_triggered_trades)

our_profit = 0
for t in our_all_valid_trades['position']:
    our_profit += sum(t)

print(f'The profit achieved by our algorithm '
      f'trading strategy was R${our_profit} per share of PETR4.')
# ###########
