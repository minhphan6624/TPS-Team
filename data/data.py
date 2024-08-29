import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from datetime import datetime 


def process_data(data, lags):
    df = pd.read_csv(data)

    # Get unique scats
    unique_scats = df['SCATS Number'].unique()

    # print(unique_scats, len(unique_scats))

    scat = unique_scats[0]

    # Filter the data for that scat number
    scat_data = df[df['SCATS Number'] == scat]

    # Get the date and time columns

    date = scat_data['Date']

    # Get the traffic flow columns
    traffic_flow = scat_data.filter(regex=r'^V\d+$').columns


    # Reshape the data from wide to long
    scat_data_long = pd.melt(
        scat_data,
        id_vars=['Date'],
        value_vars=traffic_flow,
        var_name='Time Period',
        value_name='Lane 1 Flow (Veh/15 Minutes)'
    )

    # Take Time Period interval and change to numerical value return as str
    scat_data_long['Time'] = ((scat_data_long['Time Period'].str.replace("V", "").astype(int) + 1)).astype(str)

    # Create new 15 Minutes column and combine data and time
    scat_data_long['15 Minutes'] = scat_data_long['Date'] + " " + scat_data_long['Time']

    # restructure df to only have 5 Minutes and Lane 1 Flow (Veh/5 Minutes)
    scat_data_long = scat_data_long[['15 Minutes', 'Lane 1 Flow (Veh/15 Minutes)']]

    scat_data_long.to_csv('scat_data_970.csv', index=False)
    # print(scat_data_long.head())

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
    data = 'scats_data.csv'
    lags = 5
    process_data(data, lags)
