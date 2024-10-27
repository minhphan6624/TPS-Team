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
all_models = {}

def init():
    count = 0

    # Load all lstm models from NEW_MODEL_DIR, key value (scats_num) -> model instance
    for model_name in os.listdir(NEW_MODEL_DIR):
        file_split = model_name.split(".")

        file_name = file_split[0]
        file_ext = file_split[1]

        if file_ext != "keras":
            continue

        count += 1

        # Load Model
        scats_split = model_name.split("_")

        scats_num = scats_split[0]
        model_type = scats_split[1].replace(".keras", "")

        model_path = f"{NEW_MODEL_DIR}/{model_name}"
        model = load_model(model_path)

        # Load Traffic Flow CSV
        csv_path = f"{CSV_DIR}/{scats_num}_trafficflow.csv"
        df = pd.read_csv(csv_path, encoding="utf-8").fillna(0)

        # Load Scalers
        scaler_path = f"{NEW_MODEL_DIR}/{scats_num}_{model_type}_scalers.npz"
        saved_data = np.load(scaler_path, allow_pickle=True)

        all_models[file_name] = {
            "model": model,
            "flow_csv": df,
            "scaler": saved_data
        }

        logger.log(f"[{count} of 160] Loaded model, scalers and flow for {model_type} -> {scats_num}")

    print("All models loaded successfully, list size -> ", len(all_models))

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

def predict_new_model(scats_num, date_time, direction, model_type="lstm"):
    try:
        # Load the model data
        model_data = all_models[scats_num + "_" + model_type]

        if model_data is None:
            raise FileNotFoundError(f"Model not found for scats {scats_num} and type {model_type}")

        model = model_data["model"]
        df = model_data["flow_csv"].copy()  # Create a copy to avoid modifying original
        saved_data = model_data["scaler"]

        # Load the saved data
        flow_scaler = saved_data['flow_scaler'].item()
        temporal_scaler = saved_data['temporal_scaler'].item()
        direction_encoder = saved_data['direction_encoder'].item()

        # Process historical data
        df['datetime'] = pd.to_datetime(df['15 Minutes'], dayfirst=True)
        
        # Add dummy direction if less than 4 directions
        unique_directions = df['direction'].unique()

        # If we have less than 4 directions, add a dummy direction
        if len(unique_directions) < 4:
            first_direction_data = df[df['direction'] == unique_directions[0]].copy()
            first_direction_data['direction'] = 'D'
            df = pd.concat([df, first_direction_data])

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
        
        df = df.sort_values('datetime')
        
        # Find the last 4 flow values before target_datetime
        mask = df['datetime'] <= target_datetime
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
        predicted = model.predict(X_pred, verbose=0)
        predicted_flow = flow_scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]

        print(f"[{model_type}] Predicted traffic flow for scats {scats_num} at {date_time} in direction {direction}: {predicted_flow:.2f} vehicles per 15 minutes")
        return predicted_flow

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None


def main():

    init()

    date_time = "1/10/2006 00:00"
    direction = "S"
    scats_num = "3126"

    #predict_new_model(scats_num,date_time,direction, "saes")
    predict_new_model(scats_num,date_time,direction, "lstm")
    #predict_new_model(scats_num,date_time,direction, "gru")
    #predict_new_model(scats_num,date_time,direction, "cnn")


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
