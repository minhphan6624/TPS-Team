import sys

sys.dont_write_bytecode = True

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from utilities import logger
from predict import (
    predict_new_model,
    init,
)
from matplotlib.backends.backend_pdf import (
    PdfPages,
)  # Import PdfPages to save plots in PDF


def predict_for_day(scats_num, direction, model_type, specific_date):
    """Predict traffic flow for two different times in a specific day (00:00 and 12:00)."""
    results = []
    hours = [0, 12]  # Predictions for 00:00 and 12:00 (every 12 hours)

    for hour in hours:
        date_time = (
            specific_date.strftime("%d/%m/%Y") + f" {hour:02d}:00"
        )  # Format: dd/mm/yyyy HH:00
        prediction = predict_new_model(scats_num, date_time, direction, model_type)
        results.append(
            {
                "date": specific_date.strftime("%Y-%m-%d"),
                "hour": hour,
                "prediction": prediction,
            }
        )

    return results


def predict_for_month(scats_num, direction, model_type, start_date, end_date):
    """Predict traffic flow for the entire month by iterating through each day."""
    current_date = start_date
    monthly_predictions = []

    while current_date <= end_date:
        daily_predictions = predict_for_day(
            scats_num, direction, model_type, current_date
        )
        monthly_predictions.extend(
            daily_predictions
        )  # Collect predictions for each day
        current_date += timedelta(days=1)  # Move to the next day

    return monthly_predictions


def plot_predictions(ax, predictions, model_type, month, color):
    """Plot the predictions for the entire month on the given subplot axes."""
    df = pd.DataFrame(predictions)

    # Combine date and hour to create labels for the x-axis
    x_labels = df["date"] + " " + df["hour"].astype(str) + ":00"

    # Plot predictions
    ax.plot(
        x_labels,
        df["prediction"],
        label=f"{model_type.upper()} Predictions for {month}",
        color=color,
    )

    # Set up clean x-axis ticks (fewer ticks)
    ax.set_xticks(np.arange(0, len(x_labels), 2))  # Show fewer ticks (every 2 entries)
    ax.set_xticklabels(
        x_labels[::2], rotation=45, ha="right", fontsize=6
    )  # Rotate and format labels

    ax.set_xlabel("Date and Hour")
    ax.set_ylabel("Predicted Traffic Flow (Vehicles per 15 Minutes)")
    ax.set_title(f"{model_type.upper()} Predictions for {month} 2006")
    ax.legend(
        loc="upper right"
    )  # Ensure the legend is in the upper-right corner for all plots


def main():
    # Initialize the models
    init()

    # SCATS site and direction to predict for
    scats_num = "2000"
    direction = "W"

    # Define the list of model types to evaluate
    model_types = ["lstm", "gru", "cnn", "saes"]

    # Date ranges for October and November 2006
    start_oct = datetime(2006, 10, 1)
    end_oct = datetime(2006, 10, 31)

    start_nov = datetime(2006, 11, 1)
    end_nov = datetime(2006, 11, 30)

    # Open a single PDF file to save all plots
    pdf_filename = "traffic_flow_predictions_oct_nov_2006.pdf"
    with PdfPages(pdf_filename) as pdf:
        fig, axes = plt.subplots(
            2, 4, figsize=(20, 10)
        )  # 2 rows and 4 columns for 8 total plots
        axes = axes.flatten()  # Flatten axes to easily iterate over

        for i, model_type in enumerate(model_types):
            # Predict for the entire month of October
            print(f"Predicting for October 2006 using {model_type.upper()}...")
            oct_predictions = predict_for_month(
                scats_num, direction, model_type, start_oct, end_oct
            )

            # Predict for the entire month of November
            print(f"Predicting for November 2006 using {model_type.upper()}...")
            nov_predictions = predict_for_month(
                scats_num, direction, model_type, start_nov, end_nov
            )

            # Plot predictions for October in the first row (color blue)
            plot_predictions(axes[i], oct_predictions, model_type, "October", "b")

            # Plot predictions for November in the second row (color red)
            plot_predictions(axes[i + 4], nov_predictions, model_type, "November", "r")

        plt.tight_layout()  # Adjust subplots to fit into the figure area
        pdf.savefig(fig)  # Save the entire figure with subplots to the PDF
        plt.close()

    print(f"Predictions saved to {pdf_filename}")


if __name__ == "__main__":
    main()
