# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
import os

# setting parent directory to be accessed
os.chdir('..')


def get_instrument(ticker='PETR4', in_folder='data/',
                   out_folder='data/'):
    """
    This is a function to extract all negotiations of a ticker.

    Parameters:
    ===================
    :param ticker: str: The financial instrument symbol.
    :param in_folder: str: The Path where the data files are stored.
    :param out_folder: str: The Path where the json data file will be stored.
    """
    for filename in os.listdir(in_folder):
        print(filename)
        if filename.endswith(".zip") and \
                filename.startswith('TradeIntraday_'):
            # importing data-set file
            data = pd.read_csv(in_folder + filename, compression='zip',
                               sep=';', header=0, dtype=np.str)

            # removing trades that were not for the selected ticker
            drop_idxs = data['TckrSymb'][data['TckrSymb'] != ticker].index
            data.drop(drop_idxs, axis=0, inplace=True)

            # dropping row indexes with 'TradgSsnId' == 2
            # because they are cancelled trades
            drop_idxs = data['TradgSsnId'][data['TradgSsnId'] == 2].index
            data.drop(drop_idxs, axis=0, inplace=True)

            # dropping unnecessary columns
            data.drop(['TckrSymb', 'RptDt', 'UpdActn', 'TradId',
                       'TradgSsnId'], axis=1, inplace=True)

            # fixing data and time
            data["DateTime"] = data['TradDt'] + ' ' + data['NtryTm']

            # dropping unnecessary columns
            data.drop(['NtryTm', 'TradDt'], axis=1, inplace=True)

            # converting data type
            data["DateTime"] = pd.to_datetime(data["DateTime"],
                                              format='%Y-%m-%d %H%M%f')

            # replacing "," to "." in price
            data.columns = ["Price", "Volume", "DateTime"]
            data["Price"] = data["Price"].str.replace(',', '.')

            # fixing dtypes
            data["Price"] = data["Price"].astype(np.float64)
            data["Volume"] = data["Volume"].astype(np.int64)

            # dropping old index
            data.reset_index(inplace=True, drop='index')

            # creating csv data file
            data.to_csv(
                f'{out_folder}/{ticker}_{filename[14:-6]}.csv', sep=';',
                index_label=False)


def create_candles(file, candles_type='time', candles_periodicity='15min'):
    """
    This is a function to create candles data based on the ticker

    Parameters:
    ===================
    :param file: str: The file containing all negotiations of a ticker.
    :param candles_type: str: The type of the candle system.
    :param candles_periodicity: str or int: The periodicity of the candles.
    """
    if candles_type == 'time':
        df = pd.read_csv(file, sep=';')
        df.set_index(pd.DatetimeIndex(df['DateTime']), inplace=True)
        time_candle = df.Price.resample(candles_periodicity).ohlc()
        grouped = df.groupby(pd.Grouper(freq=candles_periodicity)).sum()
        time_candle['volume'] = grouped.Volume
        return time_candle

    elif candles_type == 'tick':
        pass


if __name__ == '__main__':
    get_instrument(ticker='PETR4', in_folder='data/', out_folder='data/PETR4_data')
