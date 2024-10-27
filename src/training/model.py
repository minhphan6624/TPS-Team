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
    BatchNormalization
)
from keras.regularizers import l2
from keras.models import Sequential
import tensorflow as tf

def get_lstm(units):
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 14), return_sequences=True))  # 14 features
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    return model

def get_gru(units):
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 14), return_sequences=True))  # 14 features
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    return model


def _get_sae(inputs, hidden, output, dropout_rate=0.3):
    model = Sequential()
    
    model.add(Dense(
        hidden,
        input_shape=(inputs,),
        kernel_regularizer=l2(0.01),
        name='hidden'
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(
        output,
        kernel_regularizer=l2(0.01),
        activation='sigmoid'
    ))
    
    return model

def get_saes(layers, dropout_rate=0.3):
    # Individual autoencoders
    sae1 = _get_sae(56, layers[1], layers[-1], dropout_rate)
    sae2 = _get_sae(layers[1], layers[2], layers[-1], dropout_rate)
    sae3 = _get_sae(layers[2], layers[3], layers[-1], dropout_rate)
    
    # Combined stacked autoencoder
    saes = Sequential()
    
    # First hidden layer
    saes.add(Dense(
        layers[1],
        input_shape=(56,),
        kernel_regularizer=l2(0.01),
        name='hidden1'
    ))
    saes.add(BatchNormalization())
    saes.add(Activation('relu'))
    
    # Second hidden layer
    saes.add(Dense(
        layers[2],
        kernel_regularizer=l2(0.01),
        name='hidden2'
    ))
    saes.add(BatchNormalization())
    saes.add(Activation('relu'))
    
    # Third hidden layer
    saes.add(Dense(
        layers[3],
        kernel_regularizer=l2(0.01),
        name='hidden3'
    ))
    saes.add(BatchNormalization())
    saes.add(Activation('relu'))
    
    saes.add(Dropout(dropout_rate))
    
    # Output layer
    saes.add(Dense(
        layers[4],
        kernel_regularizer=l2(0.01),
        activation='sigmoid'
    ))
    
    return [sae1, sae2, sae3, saes]

def old_get_cnn(units):
    model = Sequential()
    model.add(Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(units[0], 14)  # 14 features
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units[1], activation='relu'))
    model.add(Dense(units[2], activation='sigmoid'))
    return model

def get_cnn(units):
    model = Sequential()
    # First Conv Block
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', input_shape=(units[0], 14)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    # Second Conv Block
    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(units[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units[2], activation='sigmoid'))
    return model