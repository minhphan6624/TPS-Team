import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split


def process_data(data, lags):
    df = pd.read_csv(data)

    # Get unique scats
    unique_scats = df['SCATS Number'].unique()

    print(unique_scats, len(unique_scats))

    scat = unique_scats[0]

    # Filter the data for that scat number
    scat_data = df[df['SCATS Number'] == scat]

    # Get the traffic flow columns
    traffic_flow = scat_data.filter(regex=r'^V\d+$').columns

    # Reshape the data from wide to long
    scat_data_long = pd.melt(
        scat_data,
        id_vars=['Date'],
        value_vars=traffic_flow,
        var_name='Time Period',
        value_name='Traffic Flow'
    )

    print(scat_data)

    # # For each scat
    # for scat in unique_scats:

    #     # Filter the data for that scat number
    #     scat_data = df[df['SCATS Number'] == scat]

    #     # Get the traffic flow columns
    #     traffic_flow = scat_data.filter(like='V').columns

    #     # Reshape the data from wide to long
    #     scat_data = pd.melt(
    #         scat_data,
    #         id_vars=['Date', 'Time'],
    #         value_vars=traffic_flow
    #     )


if __name__ == '__main__':
    data = 'data/scats_data.csv'
    lags = 5
    process_data(data, lags)
