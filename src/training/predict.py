import sys
sys.dont_write_bytecode = True

from keras.models import load_model
from data import process_data_alt
import numpy as np

def main():
    # Load in keras model
    model = load_model("saved_models/lstm.keras")
    print("Model loaded successfully!")

    # Define model file and lag settings
    file = "../../data/traffic_flows/970_E_trafficflow.csv"
    lag = 4
    
    # Process the data to get the test set
    X_train, y_train, X_test, y_test, scaler_y, scaler_X, X, y = process_data_alt(file, lag)

    # Select a singular piece of data (e.g., the first test sample)
    singular_raw_sample = X_test[900]

    # print size of X_test
    print(X_test.shape)

    # Now, let's reverse the processing of X_test[51] to see the original input
    original_input = scaler_X.inverse_transform(singular_raw_sample)  # Reverse the scaling

    print("Making a prediction with the following input:")
    print(original_input)

    # Reshape it to match the LSTM input requirements (1 sample, timesteps, features)
    singular_raw_sample = singular_raw_sample.reshape(1, lag, 1)

    # Make a prediction
    prediction = model.predict(singular_raw_sample)

    # Inverse transform the prediction
    prediction = scaler_y.inverse_transform(prediction)

    print("The prediction is: ", prediction)

if __name__ == "__main__":
    main()