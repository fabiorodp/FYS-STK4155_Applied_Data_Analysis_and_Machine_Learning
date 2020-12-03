# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
import altair as alt
import os

# setting parent directory to be accessed
# os.chdir('..')


def bollinger_bands(data, freq=5, std_value=2.0, column_base='close'):
    """
    Function to calculate the upper, lower and middle Bollinger Bands (BBs).

    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param std_value: float: Value of the Standard Deviation.
    :param column_base: srt: Name of the column that will be the base of
                             the computations.

    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the upper, lower and middle Bollinger Bands (BBs).
    """
    data[f'{freq} SMA'] = data[column_base].rolling(window=freq).mean()
    data['STD'] = data[column_base].rolling(window=freq).std()
    data[f'{freq} BB Upper'] = data[f'{freq} SMA'] + (data['STD'] * std_value)
    data[f'{freq} BB Lower'] = data[f'{freq} SMA'] - (data['STD'] * std_value)
    data.drop('STD', axis=1, inplace=True)

    return data


def ema(data, freq=5, column_base='close'):
    """
    Function to calculate the Exponential Moving Average (EMA).

    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                         the computations.

    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Exponential Moving Average (EMA).
    """
    data[f'{freq} EMA'] = \
        data[column_base].ewm(span=freq, adjust=False).mean()

    return data


def sma(data, freq=5, column_base='close'):
    """
    Function to calculate the Simple Moving Average (SMA).

    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                     the computations.

    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Simple Moving Average (SMA).
    """
    data[f'{freq} SMA'] = data[column_base].rolling(window=freq).mean()

    return data


def lwma(data, freq=5, column_base='close'):
    """
    Function to calculate the Linearly Weighted Moving Average (LWMA).

    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                     the computations.

    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Linearly Weighted Moving Average (LWMA).

    Notes:
    ===================
    Using the .apply() method we pass our own function (a lambda function)
    to compute the dot product of weights and prices in our rolling window
    (prices in the window will be multiplied by the corresponding weight,
    then summed), then dividing it by the sum of the weights.
    """
    weights = np.arange(1, freq + 1)

    data[f'{freq} LWMA'] = \
        data[column_base].rolling(freq).apply(
            lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    return data


def ema2(data, freq=5, alpha=2.0, column_base='close'):
    """
    Function to calculate the Exponential Moving Average (EMA).

    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param alpha: float: The weight factor.
    :param column_base: srt: Name of the column that will be the base of
                         the computations.

    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Exponential Moving Average (EMA).
    """
    alpha = alpha / (freq + 1.0)
    beta = 1 - alpha

    data['SMA'] = data[column_base].rolling(window=freq).mean()

    # First value is a simple SMA
    data[freq - 1, 'SMA'] = np.mean(data[:freq - 1, column_base])

    # Calculating first EMA
    data[freq, f'{freq} EMA2'] = (data[freq, column_base] * alpha) + (
            data[freq - 1, 'SMA'] * beta)

    # Calculating the rest of EMA
    for i in range(freq + 1, len(data)):
        try:
            data[i, f'{freq} EMA2'] = (data[i, column_base] * alpha) + (
                    data[i - 1, f'{freq} EMA2'] * beta)

        except IndexError:
            pass

    data.drop('SMA', axis=1, inplace=True)

    return data
