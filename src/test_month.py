import sys

sys.dont_write_bytecode = True

from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from utilities import logger
from predict import (
    predict_new_model,
    init,
)  # Assuming predict_new_model is in predict_traffic_flow.py

# Initialize the model directory and CSV directory
MODEL_DIR = "./saved_models"
NEW_MODEL_DIR = "./saved_new_models"
CSV_DIR = "../training_data/new_traffic_flows"


def predict_for_date_range(scats_num, direction, model_type, start_date, end_date):
    """Predict traffic flow for each day from start_date to end_date, for each hour."""
    current_date = start_date
    results = []

    while current_date <= end_date:
        for hour in range(24):  # Iterate through all 24 hours in a day
            date_time = (
                current_date.strftime("%d/%m/%Y") + f" {hour:02d}:00"
            )  # Format date as 'dd/mm/yyyy HH:00'
            prediction = predict_new_model(scats_num, date_time, direction, model_type)
            results.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "prediction": prediction,
                }
            )

        current_date += timedelta(days=1)  # Move to the next day

    return results


def plot_comparison(oct_predictions, nov_predictions):
    """Plot the predictions for October vs November."""
    # Convert predictions to DataFrame for easy plotting
    df_oct = pd.DataFrame(oct_predictions)
    df_nov = pd.DataFrame(nov_predictions)

    plt.figure(figsize=(12, 6))

    # Plot October predictions
    plt.plot(
        df_oct["date"] + " " + df_oct["hour"].astype(str) + ":00",
        df_oct["prediction"],
        label="October 2006",
        color="b",
    )

    # Plot November predictions
    plt.plot(
        df_nov["date"] + " " + df_nov["hour"].astype(str) + ":00",
        df_nov["prediction"],
        label="November 2006",
        color="r",
    )

    plt.xticks(rotation=45)
    plt.xlabel("Date and Hour")
    plt.ylabel("Predicted Traffic Flow (Vehicles per 15 Minutes)")
    plt.title("Traffic Flow Predictions for October vs November 2006")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Initialize the models
    init()

    # SCATS site and direction to predict for
    scats_num = "2000"
    direction = "W"
    model_type = "lstm"  # Can be changed to 'gru', 'cnn', or 'saes'

    # Date ranges for October and November 2006
    start_oct = datetime(2006, 10, 1)
    end_oct = datetime(2006, 10, 31)

    start_nov = datetime(2006, 11, 1)
    end_nov = datetime(2006, 11, 30)

    # Get predictions for October
    print("Predicting for October 2006...")
    oct_predictions = predict_for_date_range(
        scats_num, direction, model_type, start_oct, end_oct
    )

    # Get predictions for November
    print("Predicting for November 2006...")
    nov_predictions = predict_for_date_range(
        scats_num, direction, model_type, start_nov, end_nov
    )

    # Plot comparison
    plot_comparison(oct_predictions, nov_predictions)


if __name__ == "__main__":
    main()
