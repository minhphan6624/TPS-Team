import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_data(train, test, lags):

    flow_columns = [f'V{i:02}' for i in range(96)]  # V00 to V95
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # Flatten the flow columns into a single flow series
    flow1 = df1[flow_columns].values.flatten()
    flow2 = df2[flow_columns].values.flatten()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow1.reshape(-1, 1))
    flow1 = scaler.transform(flow1.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(flow2.reshape(-1, 1)).reshape(1, -1)[0]

    # Create lagged sequences
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler