import pandas as pd
import numpy as np
import os

csv_file = 'Project3/data/PETR4_20201127.csv'
candles_type = 'tick'
candles_periodicity = 5000

df = pd.read_csv(csv_file, sep=';')
df.set_index(pd.DatetimeIndex(df['DateTime']), inplace=True)

ticks = df.ix[:, ['Price', 'Volume']]


time_candle = df.Price.resample('5min').ohlc()
grouped = df.groupby(pd.Grouper(freq='5min')).sum()

d = {'date-time': [], 'Open': [], 'Low': [], 'High': [],
     'Volume': [], 'Close': []}

for i in range(1, df.shape[0], candles_periodicity):
    d['date-time'].append(df.iloc[i+candles_periodicity, 2])
    d['Open'].append(df.iloc[i, 0])
    d['Low'].append(df.iloc[i - candles_periodicity:i, 0].min())
    d['High'].append(df.iloc[i - candles_periodicity:i, 0].max())
    d['Volume'].append(df.iloc[i - candles_periodicity:i, 1].sum())
    d['Close'].append(df.iloc[i, 0])


import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web


# Make function for calls to Yahoo Finance
def get_adj_close(ticker, start, end):
    """
    A function that takes ticker symbols, starting period, ending period
    as arguments and returns with a Pandas DataFrame of the Adjusted Close
    Prices for the tickers from Yahoo Finance
    """
    start = start
    end = end
    info = web.DataReader(
        ticker, data_source='yahoo', start=start, end=end)['Adj Close']
    return pd.DataFrame(info)


# Get Adjusted Closing Prices for Facebook, Tesla and Amazon between 2016-2020
fb = get_adj_close('fb', '1/2/2016', '27/11/2020')
tesla = get_adj_close('tsla', '1/2/2016', '27/11/2020')
amazon = get_adj_close('amzn', '1/2/2016', '27/11/2020')

# Calculate 30 Day Moving Average, Std Deviation, Upper Band and Lower Band
for item in (fb, tesla, amazon):
    item['30 Day MA'] = item['Adj Close'].rolling(window=20).mean()

    # set .std(ddof=0) for population std instead of sample
    item['30 Day STD'] = item['Adj Close'].rolling(window=20).std()
    item['Upper Band'] = item['30 Day MA'] + (item['30 Day STD'] * 2)
    item['Lower Band'] = item['30 Day MA'] - (item['30 Day STD'] * 2)

# Simple 30 Day Bollinger Band for Facebook (2016-2017)
fb[['Adj Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(
    figsize=(12, 6))
plt.title('30 Day Bollinger Band for Facebook')
plt.ylabel('Price (USD)')
plt.show()


sinal, trade = [], []
for row in range(0, len(fb)):
    if fb.iloc[row, 0] < fb.iloc[row, 4]:
        sinal.append('buy')
        trade.append(fb.iloc[row, 0])

    elif fb.iloc[row, 0] > fb.iloc[row, 3]:
        sinal.append('sell')
        trade.append(fb.iloc[row, 0]*-1)

profit = []
cum = []
for i in range(0, len(trade)):
    if (trade[i] <= 0) and (sum(cum) <= 0):
        cum.append(trade[i])
    elif (trade[i] >= 0) and (sum(cum) < 0):
        profit.append(-1 * sum(cum) - trade[i] * len(cum))
        cum = [trade[i]]
    elif (trade[i] >= 0) and (sum(cum) >= 0):
        cum.append(trade[i])
    elif (trade[i] <= 0) and (sum(cum) > 0):
        profit.append(-1 * sum(cum) - trade[i] * len(cum))
        cum = [trade[i]]
