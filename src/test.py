from training.data import original_process_test
from train import TEST_CSV_DIRECTION, LAG

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np
import tensorflow as tf


def test():
    _, X_test, _, y_test, _, _ = original_process_test(TEST_CSV_DIRECTION, LAG)

    models = {
        "lstm": tf.keras.models.load_model("./saved_models/lstm.keras"),
        "gru": tf.keras.models.load_model("./saved_models/gru.keras"),
        "saes": tf.keras.models.load_model("./saved_models/saes.keras"),
        "cnn": tf.keras.models.load_model("./saved_models/cnn.keras"),
    }

    # Dictionary to store metrics for each model
    metrics = {}

    for model_name, model in models.items():
        # Predict with the model
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Store metrics
        metrics[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    # Display the results
    for model_name, model_metrics in metrics.items():
        print(f"Model: {model_name}")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("----------------------------------------")


if __name__ == "__main__":
    test()
