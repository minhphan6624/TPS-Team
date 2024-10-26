import sys

sys.dont_write_bytecode = True

from utilities import logger
from utilities.time import *

from tcn import TCN
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime

import training.data as data
from train import MODELS, TEST_CSV_DIRECTION

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL_DIR = "./saved_models"
NEW_MODEL_DIR = "./saved_new_models"
CSV_DIR = "../training_data/new_traffic_flows"

# key value (scats_num) -> model instance
lstm_models = {}

def init():
    # Load all lstm models from MODEL_DIR/*_lstm.keras
    for model_name in os.listdir(MODEL_DIR):
        if "lstm" in model_name:
            model_path = f"{MODEL_DIR}/{model_name}"
            scats_num = model_name.split("_")[0]
            lstm_models[scats_num] = load_model(model_path)

    print("LSTM Models loaded successfully, list size -> ", len(lstm_models))

def plot_results(y_true, y_pred):
    d = "2016-10-1 00:00"
    x = pd.date_range(d, periods=96, freq="15min")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label="True Data")
    ax.plot(x, y_pred, label="Model")

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel("Flow")

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def predict_traffic_flow(datetime_input, direction_input, model_path, data_path):
  # Load the model
    if "tcn" in model_path.lower():
        model = load_model(model_path, custom_objects={"TCN": TCN})
    else:
        model = load_model(model_path)

    # Load and preprocess the data
    df = pd.read_csv(data_path, encoding="utf-8").fillna(0)
    attr = "Lane 1 Flow (Veh/15 Minutes)"
    direction_attr = "direction"  # Direction column

    # Normalize traffic flow
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # One-hot encode the entire direction column for 8 possible directions
    encoder = OneHotEncoder(
        sparse_output=False, categories=[["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]
    )
    direction_encoded = encoder.fit_transform(df[direction_attr].values.reshape(-1, 1))

    # Combine flow and direction features (1 for flow + 8 for directions = 9 features)
    features = np.hstack([flow.reshape(-1, 1), direction_encoded])

    index = get_date_time_index(df, datetime_input)
    
    # One-hot encode the input direction
    direction_categories = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
    if direction_input not in direction_categories:
        raise ValueError(
            f"Invalid direction input. Valid directions are {direction_categories}"
        )

    direction_onehot = encoder.transform([[direction_input]])

    # Prepare the input for prediction
    lags = 4

    if index < lags:
        #raise ValueError("Not enough historical data for the given time.")
        return 0

    # Extract the last `lags` timesteps of features (flow + direction)
    X_pred = features[index - lags : index].reshape(
        1, lags, 9
    )  # 9 features (flow + directions)

    # Overwrite the direction feature in the input with the one-hot encoded direction
    for i in range(lags):
        X_pred[0, i, 1:] = direction_onehot

    # Check if the model is SAES and flatten input only if it is
    if "saes" in model_path.lower():
        # Flatten the input for SAES model (expects 36 features)
        X_pred_flat = X_pred.reshape(1, -1)  # Flatten to (1, 36)
        # Make prediction for SAES
        predicted = model.predict(X_pred_flat)
    else:
        # Keep input shape as (1, 4, 9) for other models
        predicted = model.predict(X_pred)

    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]

    return predicted

def predict_flow_lstm_optimized(scats_num, date_time, direction):
    if scats_num not in lstm_models:
        print(f"Model for scats_num {scats_num} not found!")
        return

    model = lstm_models[scats_num]

    csv_path = CSV_DIR + "/" + scats_num + "_" + "trafficflow.csv"

    def lstm_flow(datetime_input, direction_input, model, data_path):
        df = pd.read_csv(data_path, encoding="utf-8").fillna(0)
        attr = "Lane 1 Flow (Veh/15 Minutes)"
        direction_attr = "direction"

        scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
        flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

        encoder = OneHotEncoder(sparse_output=False, categories=[["N", "S", "E", "W", "NE", "NW", "SE", "SW"]])

        direction_encoded = encoder.fit_transform(df[direction_attr].values.reshape(-1, 1))
        features = np.hstack([flow.reshape(-1, 1), direction_encoded])

        index = get_date_time_index(df, datetime_input)
        
        direction_onehot = encoder.transform([[direction_input]])
        # Prepare the input for prediction
        X_pred = features[index - 4 : index].reshape(1, 4, 9)

        for i in range(4):
            X_pred[0, i, 1:] = direction_onehot

        predicted = model.predict(X_pred)

        return scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]

    predicted_flow = lstm_flow(date_time, direction, model, csv_path)

    print(f"Predicted traffic flow at {date_time} in direction {direction}: {predicted_flow:.2f} vehicles per 15 minutes")
    print("----------------------------------------")

    return predicted_flow

def predict_flow(scats_num, date_time, direction, model_type):
    model_path = MODEL_DIR + "/" + scats_num + "_" + model_type + ".keras"
    csv_path = CSV_DIR + "/" + scats_num + "_" + "trafficflow.csv"

    print(model_path)
    print(csv_path)
    predicted_flow = predict_traffic_flow(date_time, direction, model_path, csv_path)
    print(
        f"Predicted traffic flow at {date_time} in direction {direction}: {predicted_flow:.2f} vehicles per 15 minutes"
    )
    print("----------------------------------------")

    return predicted_flow

def predict_new_model(scats_num, date_time, direction, model_type="lstm"):
    try:
        # Define paths
        model_path = f"{NEW_MODEL_DIR}/{scats_num}_{model_type}.keras"
        csv_path = f"{CSV_DIR}/{scats_num}_trafficflow.csv"
        scaler_path = f"{NEW_MODEL_DIR}/{scats_num}_{model_type}_scalers.npz"

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scalers file not found: {scaler_path}")

        # Load the model
        model = load_model(model_path)

        # Load the historical data
        df = pd.read_csv(csv_path, encoding="utf-8").fillna(0)

        # Load scalers and encoder
        saved_data = np.load(scaler_path, allow_pickle=True)
        flow_scaler = saved_data['flow_scaler'].item()
        temporal_scaler = saved_data['temporal_scaler'].item()
        direction_encoder = saved_data['direction_encoder'].item()

        # Convert input datetime string to datetime object
        target_datetime = pd.to_datetime(date_time, format='%d/%m/%Y %H:%M')
        
        # Extract temporal features for target datetime
        temporal_features = np.array([[
            target_datetime.hour,
            target_datetime.minute,
            target_datetime.dayofweek,
            target_datetime.day,
            target_datetime.month
        ]])
        
        # Scale temporal features
        scaled_temporal = temporal_scaler.transform(temporal_features)
        
        # Encode direction
        direction_encoded = direction_encoder.transform([[direction]])
        
        # Process historical data
        df['datetime'] = pd.to_datetime(df['15 Minutes'], dayfirst=True)
        df = df.sort_values('datetime')
        
        # Find the last 4 flow values before target_datetime
        mask = df['datetime'] < target_datetime
        recent_flows = df[mask].tail(4)['Lane 1 Flow (Veh/15 Minutes)'].values
        
        if len(recent_flows) < 4:
            print(f"Not enough historical data for {scats_num} {direction} at {date_time}")
            return 0
        
        # Scale the historical flows
        scaled_flows = flow_scaler.transform(recent_flows.reshape(-1, 1)).reshape(-1)
        
        # Create the input sequence
        X_pred = np.zeros((1, 4, 14))  # 14 features
        
        # Fill in historical values and features
        for i in range(4):
            X_pred[0, i, 0] = scaled_flows[i]  # Flow
            X_pred[0, i, 1:6] = scaled_temporal[0]  # Temporal features
            X_pred[0, i, 6:] = direction_encoded[0]  # Direction

        if model_type.lower() == "saes":
            # Flatten the input from (1, 4, 14) to (1, 56)
            X_pred = X_pred.reshape(1, -1)

        # Make prediction
        predicted = model.predict(X_pred, verbose=1)
        predicted_flow = flow_scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]

        print(f"[{model_type}] Predicted traffic flow for scats {scats_num} at {date_time} in direction {direction}: {predicted_flow:.2f} vehicles per 15 minutes")
        return predicted_flow

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None


def main():

    date_time = "25/10/2006 01:00"
    direction = "W"
    scats_num = "2000"

    predict_new_model(scats_num,date_time,direction, "saes")
    predict_new_model(scats_num,date_time,direction, "lstm")
    predict_new_model(scats_num,date_time,direction, "gru")
    predict_new_model(scats_num,date_time,direction, "cnn")


    '''
    # Load Keras models and predict traffic flow including directions
    for model_name in MODELS:
        model_path = f"./saved_models/{model_name}.keras"
        print(model_path)
        cpredict(model_path, TEST_CSV_DIRECTION)'''


def cpredict(model_path, data_path):
    model_name = get_model_name(model_path)

    print(f"-------------- {model_name} --------------")

    date_time = "20/10/2024 11:30"  # Specify the time input for prediction
    direction_input = "W"  # Specify the direction input for prediction

    predicted_flow = predict_traffic_flow(
        date_time, direction_input, model_path, data_path
    )

    print(
        f"Predicted traffic flow at {date_time} in direction {direction_input}: {predicted_flow:.2f} vehicles per 15 minutes"
    )

    print("----------------------------------------")


def original_predict(model_path, train_csv):
    lags = 4

    model = load_model(model_path)
    print("Model loaded successfully!")

    X_train, y_train, scaler = data.original_process(train_csv, lags)

    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(1, -1)[0]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Make predictions
    predicted = model.predict(X_train)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    print("Size of y_train -> ", len(predicted))

    plot_results(y_train[:96], predicted[:96])

    # 96 -> 1 day
    print("Predicted 97 -> ", predicted[:96])
    print("Predicted Array -> ", predicted)


def get_model_name(model_path):
    return model_path.split("/")[-1].split(".")[0].upper()


if __name__ == "__main__":
    main()
