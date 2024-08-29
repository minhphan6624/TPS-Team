import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split


def process_data(data, lags):
    df = pd.read_csv(data)

    print(df.head())


if __name__ == '__main__':
    data = 'data/scats_data.csv'
    lags = 5
    process_data(data, lags)
