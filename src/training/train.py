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
from keras.layers import Input

warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config):
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
    df.to_csv("saved_models/" + name + "_loss.csv", encoding="utf-8", index=False)

def train_saes(models, X_train, y_train, name, config):
    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            prev_model = models[i - 1]

            input_tensor = prev_model.layers[0].input
            hidden_layer_output = prev_model.get_layer('hidden').output

            hidden_layer_model = Model(inputs=input_tensor, outputs=hidden_layer_output)
            
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name", default="lstm")
    args = parser.parse_args()

    lag = 4
    config = {"batch": 256, "epochs": 600}

    train_csv = '../../data/traffic_flows/970_E_trafficflow.csv'

    X_train, y_train, _ = original_process(train_csv, lag)

    if args.model != "saes":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    if args.model == "lstm":
        m = model.get_lstm([lag, 64, 64, 1])
        train_model(m, X_train, y_train, 'lstm', config)
    elif args.model == "gru":
        m = model.get_gru([lag, 64, 64, 1])
        train_model(m, X_train, y_train, 'gru', config)
    elif args.model == "saes":
        m = model.get_saes([lag, 128, 64, 32, 1])
        train_saes(m, X_train, y_train, 'saes', config)

if __name__ == "__main__":
    main(sys.argv)
