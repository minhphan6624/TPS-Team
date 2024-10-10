from keras.layers import (
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    SimpleRNN,
    Dense,
    Dropout,
    Activation,
)
from keras.models import Sequential


# Define LSTM model with 9 features
def get_lstm(input_shape, units):
    model = Sequential()
    model.add(
        LSTM(units["lstm_units_1"], input_shape=input_shape, return_sequences=True)
    )  # 9 features
    model.add(LSTM(units["lstm_units_2"]))
    model.add(Dropout(0.2))
    model.add(Dense(units["dense_units"], activation="sigmoid"))

    return model


# Define GRU model with 9 features
def get_gru(input_shape, units):
    model = Sequential()
    model.add(
        GRU(units["gru_units_1"], input_shape=input_shape, return_sequences=True)
    )  # 9 features
    model.add(GRU(units["gru_units_2"]))
    model.add(Dropout(0.2))
    model.add(Dense(units["dense_units"], activation="sigmoid"))

    return model


# Define SAE model
def get_sae(input_dim, layers):
    model = Sequential()
    model.add(Dense(layers["hidden_1"], input_shape=(input_dim,), name="hidden"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(layers["output"], activation="sigmoid"))

    return model


# Define stacked autoencoder (SAE) model
def get_saes(input_dim, layers):
    sae1 = get_sae(
        input_dim, {"hidden_1": layers["hidden_1"], "output": layers["hidden_2"]}
    )
    sae2 = get_sae(
        layers["hidden_1"],
        {"hidden_1": layers["hidden_2"], "output": layers["hidden_3"]},
    )
    sae3 = get_sae(
        layers["hidden_2"],
        {"hidden_1": layers["hidden_3"], "output": layers["hidden_4"]},
    )

    saes = Sequential()
    saes.add(Dense(layers["hidden_1"], input_shape=(input_dim,), name="hidden1"))
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers["hidden_2"], name="hidden2"))
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers["hidden_3"], name="hidden3"))
    saes.add(Activation("sigmoid"))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers["output"], activation="sigmoid"))

    return saes


# Define CNN model with 9 features
def get_cnn(input_shape, units):
    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units["dense_1"], activation="relu"))
    model.add(
        Dense(units["dense_2"], activation="sigmoid")
    )  # Or 'linear' for regression
    return model
