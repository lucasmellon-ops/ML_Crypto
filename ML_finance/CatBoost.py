import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
import Indicators

import warnings
warnings.filterwarnings('ignore')

class ML:
    def __init__(self, dataset, model_init):
        self.dataset = dataset
        self.model_init = model_init

    def _add_indicators(self, test=False):
        dataset = self.dataset.copy()

        dataset['ROC10'] = Indicators.ROC(dataset['Close'], 10)
        dataset['ROC30'] = Indicators.ROC(dataset['Close'], 30)
        dataset['ROC50'] = Indicators.ROC(dataset['Close'], 50)

        def MOM(df, n):
            return pd.Series(df.diff(n), name='Momentum_' + str(n))

        for n in [10, 30, 50, 100, 150]:
            dataset[f'MOM{n}'] = MOM(dataset['Close'], n)

        for n in [10, 30, 50, 100, (200 if test else 100)]:
            dataset[f'RSI{n}'] = Indicators.RSI(dataset['Close'], n)
            dataset[f'%K{n}'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], n)
            dataset[f'%D{n}'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], n)

        if not test:
            dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100)
        else:
            for n in [10, 20, 55, 90, 155]:
                dataset[f'SMA{n}'] = Indicators.SMA(dataset['Close'], n) / dataset['Close']

            for n in [20, 100, 200, 400]:
                dataset[f'EMA{n}'] = Indicators.EMA(dataset['Close'], n) / dataset['Close']

            for column in ['Low', 'High', 'Open']:
                dataset[column] = dataset[column] / dataset['Close']

            dataset.drop(['Volume', 'Low', 'High', 'Open'], axis=1, inplace=True)

        dataset = dataset.dropna(axis=0)
        return dataset

    def real(self):
        dataset = self._add_indicators(test=False)
        dataset['positions'] = np.where(dataset['Close'] > dataset['Close'].shift(1), 0, 1)
        dataset.drop(['Close2'], axis=1, inplace=True)
        
        model = self.model_init
        Y = dataset["positions"]
        X = dataset.drop('positions', axis=1)
        model.fit(X, Y)
        return dataset

    def test(self):
        return self._add_indicators(test=True)

def pred_real(model, df):
    dataset = ML(df, model).real()
    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()
    return dataset

def pred_test(model, df, columns):
    dataset = ML(df, model).test()
    dataset = dataset.iloc[:, columns]
    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()
    return dataset

