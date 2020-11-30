import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf


class StockExchangeData:

    @staticmethod
    def _create_tick_candles(df, n_trades):
        d = {'time': [], 'Open': [], 'Low': [], 'High': [],
             'Volume': [], 'Close': []}

        for i in range(n_trades, df.shape[0], n_trades):
            """
            Error: when it updates the cycles it counts since 
            the beginning
            """
            d['time'].append(df.iloc[i, 2])
            d['Open'].append(df.iloc[i - n_trades, 0])
            d['Low'].append(df.iloc[i - n_trades:i, 0].min())
            d['High'].append(df.iloc[i - n_trades:i, 0].max())
            d['Volume'].append(df.iloc[i - n_trades:i, 1].sum())
            d['Close'].append(df.iloc[i, 0])

        return pd.DataFrame(d)

    def __init__(self, random_state=None):
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, nr_trades, ticker="PETR4", degree=None, path=None,
            sep=None, plot_cadles=False, plot_pairplot=False):

        # importing data-set file
        data = pd.read_csv(path, sep=sep, header=0, dtype=np.str)

        # removing trades that were not for the selected ticker
        drop_idxs = data['TckrSymb'][data['TckrSymb'] != ticker].index
        data.drop(drop_idxs, axis=0, inplace=True)

        # dropping row indexes with 'TradgSsnId' == 2
        # because they are cancelled trades:
        drop_idxs = data['TradgSsnId'][data['TradgSsnId'] == 2].index
        data.drop(drop_idxs, axis=0, inplace=True)

        # dropping unnecessary columns:
        data.drop(['TckrSymb', 'RptDt', 'UpdActn', 'TradId',
                   'TradgSsnId'], axis=1, inplace=True)

        # fixing data and time:
        data["date-time"] = data['TradDt'] + ' ' + data['NtryTm']

        # dropping unnecessary columns:
        data.drop(['NtryTm', 'TradDt'], axis=1, inplace=True)

        # converting data type:
        data["date-time"] = pd.to_datetime(data["date-time"],
                                           format='%Y-%m-%d %H%M%f')

        # replacing "," to "." in price:
        data.columns = ["price", "volume", "date-time"]
        data["price"] = data["price"].str.replace(',', '.')

        # fixing dtypes:
        data["price"] = data["price"].astype(np.float64)
        data["volume"] = data["volume"].astype(np.int64)

        # generating tick-candles:
        Xz = self._create_tick_candles(df=data, n_trades=nr_trades)
        Xz.set_index('time', inplace=True)

        # plotting candles:
        if plot_cadles == True:
            mpf.plot(data=Xz, type='candle', style='yahoo', volume=True)
            plt.show()

        # checking distribution and linearity
        if plot_pairplot == True:
            sns.pairplot(data=Xz)
            plt.show()

        # setting up design matrix X and targets z:
        Xz = Xz.to_numpy()
        X = Xz[:-1, :-1]
        z = Xz[1:, -1][:, np.newaxis]

        return X, z


def create_tick_candles(df, n_trades):
    d = {'time': [], 'Open': [], 'Low': [], 'High': [],
         'Volume': [], 'Close': []}

    for i in range(n_trades, df.shape[0], n_trades):
        """Error: when it updates the cycles it counts since the beginning"""
        d['time'].append(df.iloc[i, 2])
        d['Open'].append(df.iloc[i - n_trades, 0])
        d['Low'].append(df.iloc[i - n_trades:i, 0].min())
        d['High'].append(df.iloc[i - n_trades:i, 0].max())
        d['Volume'].append(df.iloc[i - n_trades:i, 1].sum())
        d['Close'].append(df.iloc[i, 0])

    return pd.DataFrame(d)


import pandas as pd
import numpy as np
import package.helper as h
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
from package.linear_models import OLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ###################################### dataset engineering:

# importing data-set file
data = pd.read_csv('data/data.txt', sep=";", header=0,
                   dtype=np.str)

# removing trades that were not for ticker WINV20:
drop_idxs = \
    data['TckrSymb'][data['TckrSymb'] != "WINV20"].index

data.drop(drop_idxs, axis=0, inplace=True)

# dropping row indexes with 'TradgSsnId' == 2
# because they are cancelled trades:
drop_idxs = \
    data['TradgSsnId'][data['TradgSsnId'] == 2].index

data.drop(drop_idxs, axis=0, inplace=True)

# dropping unnecessary columns:
data.drop(['TckrSymb', 'RptDt', 'UpdActn', 'TradId',
           'TradgSsnId'], axis=1, inplace=True)

# fixing data and time:
data["date-time"] = data['TradDt'] + ' ' + data['NtryTm']

# dropping unnecessary columns:
data.drop(['NtryTm', 'TradDt'], axis=1, inplace=True)

# converting data type:
data["date-time"] = pd.to_datetime(data["date-time"],
                                   format='%Y-%m-%d %H%M%f')

# replacing "," to "." in price:
data.columns = ["price", "volume", "date-time"]
data["price"] = data["price"].str.replace(',', '.')

# fixing dtypes:
data["price"] = data["price"].astype(np.float64)
data["volume"] = data["volume"].astype(np.int64)

# generating tick-candles:
Xz = h.create_tick_candles(df=data, n_trades=5000)
Xz.set_index('time', inplace=True)

# plotting candles:
mpf.plot(data=Xz, type='candle', style='yahoo', volume=True)

# checking distribution and linearity
sns.pairplot(data=Xz)
plt.show()

# setting up design matrix X and targets z:
Xz = Xz.to_numpy()
X = Xz[:-1, :-1]
z = Xz[1:, -1][:, np.newaxis]

#### ?????? testar transformar em polynomial

X_train, X_test, z_train, z_test = \
    train_test_split(
        X, z, test_size=0.2, random_state=10)

# scaler = StandardScaler(with_mean=False, with_std=True)
# scaler.fit(X_train)
# X_test = scaler.transform(X_test)
# X_train = scaler.transform(X_train)

# sns.pairplot(data=X_train)
# plt.show()

lr = OLS(random_state=10)
lr.fit(X_train, z_train)

z_hat = lr.predict(X_train)
z_tilde = lr.predict(X_test)

print(mean_squared_error(z_train, z_hat))
print(mean_squared_error(z_test, z_tilde))

plt.scatter(x=np.linspace(-100, 100, 88), y=(z_test - z_tilde))
plt.show()
