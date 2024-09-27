"""
Train the NN model.
"""
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

warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    
    hist = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
    )

    # if model exists, delete
    if os.path.exists("saved_models/" + name + ".keras"):
        os.remove("saved_models/" + name + ".keras")

    model.save("saved_models/" + name + ".keras")
    df = pd.DataFrame.from_dict(hist.history)
    #df.to_csv("saved_models/" + name + " loss.csv", encoding="utf-8", index=False)


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(
                input=p.input, output=p.get_layer("hidden").output
            )
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

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer("hidden").get_weights()
        saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    lag = 4
    config = {"batch": 256, "epochs": 600}

    train_csv = '../../data/traffic_flows/970_E_trafficflow.csv'

    X_train, y_train, _ = original_process(train_csv, lag)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    m = model.get_lstm([lag, 64, 64, 1])
    train_model(m, X_train, y_train, 'lstm', config)


if __name__ == "__main__":
    main(sys.argv)