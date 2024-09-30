import sys


sys.dont_write_bytecode = True

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from data import original_process
import model as model
from keras.models import Model
from keras.callbacks import EarlyStopping
from pathlib import Path

warnings.filterwarnings("ignore")


EPOCHS = 300
BATCH_SIZE = 256
LAG = 4
SCATS_CSV_DIR = "../../data/traffic_flows"
TEST_CSV = f"{SCATS_CSV_DIR}/970_N_trafficflow.csv"
MODELS = {
    "lstm": model.get_lstm([LAG, 64, 64, 1]),
    "gru": model.get_gru([LAG, 64, 64, 1]),
    "saes": model.get_saes([LAG, 128, 64, 32, 1]),
    "tcn": model.get_tcn([LAG, 128, 64, 32, 1]),
}


def get_early_stopping_callback():
    return EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Verbose setting, 1 for output when early stopping kicks in
        mode="min",  # 'min' for minimizing loss, 'max' for maximizing metric, 'auto' decides automatically
        restore_best_weights=True,  # Restores model weights from the epoch with the best validation loss
    )


def train_model(model, X_train, y_train, name, config, print_loss):
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

    # Set up EarlyStopping callback

    # Train the model with EarlyStopping
    hist = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[get_early_stopping_callback()],
    )

    # if model exists, delete
    if os.path.exists("saved_models/" + str(name) + ".keras"):
        os.remove("saved_models/" + str(name) + ".keras")

    model.save("saved_models/" + name + ".keras")
    if print_loss:
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv("saved_models/" + name + "_loss.csv", encoding="utf-8", index=False)


def train_saes(models, X_train, y_train, name, config, print_loss):
    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            prev_model = models[i - 1]

            input_tensor = prev_model.layers[0].input
            hidden_layer_output = prev_model.get_layer("hidden").output

            hidden_layer_model = Model(inputs=input_tensor, outputs=hidden_layer_output)

            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

        m.fit(
            temp,
            y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05,
            callbacks=[get_early_stopping_callback()],
        )

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer("hidden").get_weights()
        saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config, print_loss)


def train_models(model_types, model_prefix, csv, print_loss):
    config = {"batch": BATCH_SIZE, "epochs": EPOCHS}

    X_train, y_train, _ = original_process(csv, LAG)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_train_saes = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    for model_type in model_types:
        model_name = (
            model_prefix + model_type if model_prefix != None or "" else model_type
        )

        model_instance = MODELS.get(model_type)

        if model_type == "saes":
            train_saes(
                model_instance, X_train_saes, y_train, model_name, config, print_loss
            )
        else:
            train_model(
                model_instance, X_train, y_train, model_name, config, print_loss
            )


def train_scats(model_types):
    for path in Path(SCATS_CSV_DIR).iterdir():
        if path.is_file():
            name = Path(path).name
            scats_data = name.split("_")
            scats_number = scats_data[0]
            scats_direction = scats_data[1]
            print(
                f"------------  SCATS site: {scats_number} | Direction: {scats_direction}  ------------"
            )

            model_prefix = str.format("{0}_{1}_", scats_number, scats_direction)
            print(model_types)
            train_models(model_types, model_prefix, path, False)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        help="Model names (e.g. lstm gru tcn)",
        nargs="+",  # This allows multiple model names
        default=["lstm"],  # Default to a list containing "lstm"
    )

    # Add scats argument
    parser.add_argument(
        "--scats", help="Check if --scats is present", action="store_true"
    )

    parser.add_argument(
        "--loss", help="Check if --scats is present", action="store_true"
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.scats:
        train_scats(args.model)
    else:
        train_models(args.model, None, TEST_CSV, True)


if __name__ == "__main__":
    main(sys.argv)
