import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def original_process(train, lags):
    attr = 'Lane 1 Flow (Veh/15 Minutes)'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train = []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])

    train = np.array(train)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train, scaler