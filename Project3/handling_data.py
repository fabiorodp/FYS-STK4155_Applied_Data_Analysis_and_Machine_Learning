# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
from Project3.package.read_data import get_instrument, create_candles


# function to separate all negotiations of a single ticker from all raw data
get_instrument(
    ticker='PETR4',
    in_folder='Project3/data/raw/',
    out_folder='Project3/data/PETR4/'
)

# generating data in a given periodicity=1d
create_candles(
    ticker='PETR4',
    candles_periodicity='1D',
    in_folder='Project3/data/PETR4/',
    out_folder='Project3/data/'
)

# generating data in a given periodicity=15min
create_candles(
    ticker='PETR4',
    candles_periodicity='15min',
    in_folder='Project3/data/PETR4/',
    out_folder='Project3/data/'
)
