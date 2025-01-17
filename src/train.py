import sys
sys.dont_write_bytecode = True

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping
from pathlib import Path
from training.model import get_lstm, get_gru, get_saes, get_cnn
from training.data import process_temporal_data

warnings.filterwarnings("ignore")

# Hyperparameters
EPOCHS = 600
BATCH_SIZE = 256
LAG = 4
SCATS_CSV_DIR = "../training_data/traffic_flows"
TEST_CSV = f"{SCATS_CSV_DIR}/970_N_trafficflow.csv"
SCATS_CSV_DIR_DIRECTION = "../training_data/new_traffic_flows"
TEST_CSV_DIRECTION = f"{SCATS_CSV_DIR_DIRECTION}/970_trafficflow.csv"

MODEL_DIR = "./saved_test_models/"

# Models with input shape reflecting 14 features
# (1 for flow + 5 for temporal + 8 for direction)
MODELS = {
    "lstm": get_lstm([LAG, 64, 64, 1]),
    "gru": get_gru([LAG, 64, 64, 1]),
    "saes": get_saes([LAG, 128, 64, 32, 1]),
    "cnn": get_cnn([LAG, 128, 1]),
}

class ModelTrainer:
    def __init__(self):
        self.flow_scaler = None
        self.temporal_scaler = None
        self.direction_encoder = None
    
    def get_early_stopping_callback(self):
        return EarlyStopping(
            monitor="loss",
            patience=50,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

    def train_model(self, model, X_train, y_train, name, config, print_loss):
        model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

        model_path = MODEL_DIR + str(name) + ".keras"
        model_loss_path = MODEL_DIR + name + "_loss.csv"
        scaler_path = MODEL_DIR + name + "_scalers.npz"

        # Train the model
        hist = model.fit(
            X_train,
            y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05,
        )

        # Delete existing model if it exists
        if os.path.exists(model_path):
            os.remove(model_path)

        # Save loss history if requested
        if print_loss:
            df = pd.DataFrame.from_dict(hist.history)
            df.to_csv(model_loss_path, encoding="utf-8", index=False)

        # Save the model
        model.save(model_path)
        
        # Save the scalers and encoder
        if self.flow_scaler is not None:
            np.savez(
                scaler_path,
                flow_scaler=self.flow_scaler,
                temporal_scaler=self.temporal_scaler,
                direction_encoder=self.direction_encoder
            )

    def train_saes(self, models, X_train, y_train, name, config, print_loss):
        # Flatten the X_train for the SAES model (now 14 features * LAG)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        temp = X_train_flat
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
            )
            models[i] = m

        # Train the final SAES model
        saes = models[-1]
        for i in range(len(models) - 1):
            weights = models[i].get_layer("hidden").get_weights()
            saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

        self.train_model(saes, X_train_flat, y_train, name, config, print_loss)

    def train_models(self, model_types, model_prefix, csv, print_loss):
        config = {"batch": BATCH_SIZE, "epochs": EPOCHS}

        # Read the CSV file
        df = pd.read_csv(csv, encoding="utf-8").fillna(0)
        
        # Process data including temporal features
        X_train, y_train, self.flow_scaler, self.temporal_scaler, self.direction_encoder = process_temporal_data(df, LAG)

        # For non-SAES models, reshape to (samples, timesteps, features)
        num_features = X_train.shape[2]  # Should be 14
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

        # For SAES, flatten the features
        X_train_saes = np.reshape(X_train, (X_train.shape[0], -1))

        for model_type in model_types:
            model_name = model_prefix + model_type if model_prefix else model_type
            model_instance = MODELS.get(model_type)

            if model_type == "saes":
                self.train_saes(
                    model_instance, X_train_saes, y_train, model_name, config, print_loss
                )
            else:
                self.train_model(
                    model_instance, X_train_reshaped, y_train, model_name, config, print_loss
                )

    def train_one_model(self, one_model):
        one_model_data = one_model.split("_")

        scat_number = one_model_data[0]
        model_type = one_model_data[1]

        print(f"Training one model: {scat_number} {model_type}")

        config = {"batch": BATCH_SIZE, "epochs": EPOCHS}

        # Load in traffic flow data
        csv_path = f"{SCATS_CSV_DIR_DIRECTION}/{scat_number}_trafficflow.csv"
        df = pd.read_csv(csv_path, encoding="utf-8").fillna(0)

        X_train, y_train, self.flow_scaler, self.temporal_scaler, self.direction_encoder = process_temporal_data(df, LAG)

        # For non-SAES models, reshape to (samples, timesteps, features)
        num_features = X_train.shape[2]  # Should be 14
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

        # For SAES, flatten the features
        X_train_saes = np.reshape(X_train, (X_train.shape[0], -1))

        model_name = f"{scat_number}_{model_type}"
        model_instance = MODELS.get(model_type)

        if model_type == "saes":
            self.train_saes(model_instance, X_train_saes, y_train, model_name, config, False)
        else:
            self.train_model(model_instance, X_train_reshaped, y_train, model_name, config, False)

    def train_scats(self, model_types):
        for path in Path(SCATS_CSV_DIR_DIRECTION).iterdir():
            if path.is_file():
                name = Path(path).name
                scats_data = name.split("_")
                scats_number = scats_data[0]
                print(f"------------  SCATS site: {scats_number}  ------------")

                model_prefix = str.format(
                    "{0}_",
                    scats_number,
                )
                print(model_types)
                self.train_models(model_types, model_prefix, path, False)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model names (e.g. lstm gru tcn)",
        nargs="+",
        default=["lstm"],
    )
    parser.add_argument(
        "--scats",
        help="Check if --scats is present",
        action="store_true"
    )
    parser.add_argument(
        "--loss",
        help="Save loss history",
        action="store_true"
    )
    parser.add_argument(
        "--one_model",
        help="Train just one scat model",
    )

    args = parser.parse_args()
    trainer = ModelTrainer()

    if args.one_model:
        trainer.train_one_model(args.one_model)
    elif args.scats:
        trainer.train_scats(args.model)
    else:
        trainer.train_models(args.model, None, TEST_CSV_DIRECTION, args.loss)

if __name__ == "__main__":
    main(sys.argv)