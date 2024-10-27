import sys

sys.dont_write_bytecode = True

from training.data import process_temporal_data_test
from train import TEST_CSV_DIRECTION, LAG

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def test():
    # Load the test data
    _, X_test, _, y_test, _, _, _ = process_temporal_data_test(TEST_CSV_DIRECTION, LAG)

    # Load the trained models
    models = {
        "lstm": tf.keras.models.load_model("./saved_new_models/lstm.keras"),
        "gru": tf.keras.models.load_model("./saved_new_models/gru.keras"),
        "saes": tf.keras.models.load_model("./saved_new_models/saes.keras"),
        "cnn": tf.keras.models.load_model("./saved_new_models/cnn.keras"),
    }

    # Dictionary to store metrics for each model
    metrics = {}

    with PdfPages("traffic_flow_predictions.pdf") as pdf:
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 10)
        )  # Create a 2x2 grid for subplots
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, (model_name, model) in enumerate(models.items()):
            # Prepare input for each model
            if model_name == "saes":
                # Flatten the input for the SAE model
                X_test_model = X_test.reshape(X_test.shape[0], -1)
            else:
                # Keep input as original for CNN, LSTM, and GRU models
                X_test_model = X_test

            # Predict with the model
            y_pred = model.predict(X_test_model)

            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)  # RMSE calculation
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Store metrics
            metrics[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

            # Plot the predictions against the true values (limit to 20 entries)
            axes[i].plot(y_test[:20], label="True Values", color="b")
            axes[i].plot(
                y_pred[:20],
                label=f"{model_name} Predictions",
                color="r",
                linestyle="--",
            )
            axes[i].set_title(f"Traffic Flow Predictions - {model_name.upper()}")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Traffic Flow (Vehicles per 15 Minutes)")
            axes[i].legend()

        plt.tight_layout()
        pdf.savefig(fig)  # Save the entire figure with subplots to the PDF
        plt.close()

    # Display the results
    for model_name, model_metrics in metrics.items():
        print(f"Model: {model_name}")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("----------------------------------------")


if __name__ == "__main__":
    test()
