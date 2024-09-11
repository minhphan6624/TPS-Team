import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def process_data(file, lags):
    
    # Load the data
    df = pd.read_csv(file)

    # Convert the "15 Minutes" column to datetime
    df["15 Minutes"] = pd.to_datetime(df["15 Minutes"], dayfirst=True)

    # Extract useful components from the datetime (day, month, year, hour, minute)
    df['year'] = df["15 Minutes"].dt.year
    df['month'] = df["15 Minutes"].dt.month
    df['day'] = df["15 Minutes"].dt.day
    df['hour'] = df["15 Minutes"].dt.hour
    df['minute'] = df["15 Minutes"].dt.minute

    # Select the target column and numerical features (assuming "Lane 1 Flow (Veh/15 Minutes)" is your target)
    y = df["Lane 1 Flow (Veh/15 Minutes)"].values
    
    # Select the numerical features for X
    X = df[['hour', 'minute']].sum(axis=1).values

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1, 1))
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)

    # Create lagged sequences
    X_lagged, y_lagged = [], []
    
    for i in range(lags, len(X_scaled)):
        X_lagged.append(X_scaled[i - lags: i])
        y_lagged.append(y_scaled[i])

    X_lagged = np.array(X_lagged)
    y_lagged = np.array(y_lagged)

    # Split the lagged data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_lagged, y_lagged, test_size=0.33, random_state=42)

    return X_train, y_train, X_test, y_test, scaler_y

def process_data_alt(file, lags):
    
    # Load the data
    df = pd.read_csv(file)

    # Convert the "15 Minutes" column to datetime
    df["15 Minutes"] = pd.to_datetime(df["15 Minutes"], dayfirst=True)

    # Extract useful components from the datetime (day, month, year, hour, minute)
    df['year'] = df["15 Minutes"].dt.year
    df['month'] = df["15 Minutes"].dt.month
    df['day'] = df["15 Minutes"].dt.day
    df['hour'] = df["15 Minutes"].dt.hour
    df['minute'] = df["15 Minutes"].dt.minute

    # Select the target column and numerical features (assuming "Lane 1 Flow (Veh/15 Minutes)" is your target)
    y = df["Lane 1 Flow (Veh/15 Minutes)"].values
    
    # Select the numerical features for X
    X = df[['hour', 'minute']].sum(axis=1).values

    # Reshape X to be 2D (required by MinMaxScaler)
    X = X.reshape(-1, 1)

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1, 1))
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)

    # Create lagged sequences
    X_lagged, y_lagged = [], []
    
    for i in range(lags, len(X_scaled)):
        X_lagged.append(X_scaled[i - lags: i])
        y_lagged.append(y_scaled[i])

    X_lagged = np.array(X_lagged)
    y_lagged = np.array(y_lagged)

    # Split the lagged data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_lagged, y_lagged, test_size=0.33, random_state=42)

    return X_train, y_train, X_test, y_test, scaler_y, scaler_X, X, y