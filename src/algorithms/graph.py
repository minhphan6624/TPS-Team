# System Imports
import math

# Library Imports
import pandas as pd

# Project Imports
import utilities.logger as logger

# Constant Variables
LAT_OFFSET = 0.00155
LONG_OFFSET = 0.00125

# Global Variables
df = None

def init():
    load_data()
    # generate_graph()

def load_data():
    global df, scat_df, position_df
    # Load in the 'scats_data.csv' file
    file_location = "../training_data/scats_data.csv"

    df = pd.read_csv(file_location)

    # Load in the 'scats_site_listing.csv' file
    file_location = "../training_data/scats_site_listing.csv"

    scat_df = pd.read_csv(file_location)

    # Load in the 'traffic_count_locations.csv' file
    file_location = "../training_data/traffic_count_locations.csv"

    position_df = pd.read_csv(file_location)

    position_df = position_df.drop_duplicates(subset=["TFM_DESC"])
    position_df["TFM_DESC"] = position_df["TFM_DESC"].str.upper().apply(format_tfm_desc)

    # Fix location names.
    df["Location"] = df["Location"].replace(
        {
            "HIGH STREET_RD": "HIGH_STREET_RD",
            "STUDLEY PARK_RD": "STUDLEY_PARK_RD",
            "MONT ALBERT_RD": "MONT_ALBERT_RD",
            "MAROONDAH_HWY": "WHITEHORSE_RD", # Maroondah Hwy is now Whitehorse Rd
            "VICTORIA_ST E": "BARKERS_RD E", # Victoria St turns into Barkers Rd
        },
        regex=True,
    )

def format_tfm_desc(text):
    # Split the text into words
    words = text.split()
    
    # Join the first two words with an underscore
    if len(words) >= 2:
        words[0:2] = ['_'.join(words[0:2])]
    
    # Join the words back together
    return ' '.join(words)

def generate_graph():
    global df

    organised_df = df[["SCATS Number", "Location", "NB_LONGITUDE", "NB_LATITUDE"]]
    organised_df = organised_df.drop_duplicates(subset=["Location"])

    # add missing cases
    # new_row = pd.DataFrame({'SCATS Number': [4051], 'Location': ['BULLEEN_RD N'], 'NB_LONGITUDE': [145.07045], 'NB_LATITUDE': [-37.79262]})
    # organised_df = pd.concat([organised_df, new_row], ignore_index=True)
    # new_row = pd.DataFrame({'SCATS Number': [4030], 'Location': ['DONCASTER_RD NE'], 'NB_LONGITUDE': [145.06394], 'NB_LATITUDE': [-37.793440000000004]})
    # organised_df = pd.concat([organised_df, new_row], ignore_index=True)
    # new_row = pd.DataFrame({'SCATS Number': [4030], 'Location': ['BURKE_RD N'], 'NB_LONGITUDE': [145.06394], 'NB_LATITUDE': [-37.793440000000004]})
    # organised_df = pd.concat([organised_df, new_row], ignore_index=True)
    # new_row = pd.DataFrame({'SCATS Number': [4262], 'Location': ['CHURCH_ST NE'], 'NB_LONGITUDE': [145.01628], 'NB_LATITUDE': [-37.82]})
    # organised_df = pd.concat([organised_df, new_row], ignore_index=True)
    # new_row = pd.DataFrame({'SCATS Number': [4262], 'Location': ['BURWOOD_RD E'], 'NB_LONGITUDE': [145.01628], 'NB_LATITUDE': [-37.82]})
    # organised_df = pd.concat([organised_df, new_row], ignore_index=True)


    graph = {}

    # organised_df.to_csv("organised_df.csv")

    for index, row in organised_df.iterrows():
        # get the scat number, location, longitude and latitude
        scat = int(row["SCATS Number"])
        location = row["Location"]
        longitude = row["NB_LONGITUDE"]
        latitude = row["NB_LATITUDE"]

        # get the road and direction from the location
        location_split = location.split(" ")
        road = location_split[0]
        direction = location_split[1]

        # get the opposite direction
        opposite_direction = get_opposite_direction(direction)

        # create a search string to find the opposite direction in locations
        search_str = f"{road} {opposite_direction}".lower()

        # Search the unique dataframe for a 'Location' that contains the first location and direction
        first_loc_df = organised_df[
            (organised_df["Location"].str.lower().str.contains(search_str))
            & (organised_df["SCATS Number"] != scat)
        ]

        closest_scat = None
        min_distance = float("inf")

        # Find the closest SCAT based on longitude and latitude
        for _, row in first_loc_df.iterrows():
            # calculate the distance between the scat and the row
            dist = calculate_distance(scat, row["SCATS Number"])
            # if this is the scat with the closest distance
            if dist < min_distance or closest_scat is None:
                # if the scat is 4035 and the direction is W then skip
                if scat == 4035 and direction == "W":
                    continue

                # check if the scat is the closest scat in terms of long and lat depending on direction
                if direction == "N":
                    if row["NB_LATITUDE"] > latitude:
                        closest_scat = row["SCATS Number"]
                        min_distance = dist
                elif direction == "S":
                    if row["NB_LATITUDE"] < latitude:
                        closest_scat = row["SCATS Number"]
                        min_distance = dist
                elif direction == "E":
                    if row["NB_LONGITUDE"] > longitude:
                        closest_scat = row["SCATS Number"]
                        min_distance = dist
                elif direction == "W":
                    if row["NB_LONGITUDE"] < longitude:
                        closest_scat = row["SCATS Number"]
                        min_distance = dist
                else:
                    # if row["NB_LONGITUDE"] > longitude and row["NB_LATITUDE"] > latitude:
                    closest_scat = row["SCATS Number"]
                    min_distance = dist
            
        entry = f"{closest_scat}_{opposite_direction}"

        if closest_scat is not None:
            if graph.get(scat) is None:
                graph[scat] = [entry]
            else:
                if entry not in graph[scat]:
                    graph[scat].append(entry)
            

    logger.log(graph)
    logger.log("[+] Graph generated successfully")

    return graph


def generate_graph_old():
    global df

    # get unique values of 'Location' column
    locations = df["Location"]

    # get longitude and latitude
    longitudes = df["NB_LONGITUDE"]
    latitudes = df["NB_LATITUDE"]

    # get scats number
    scats_numbers = df["SCATS Number"]

    # Compute a separate dataframe which is all unique rows by Location
    unique_df = df.drop_duplicates(subset=["Location"])

    graph = {}

    for index, scat in enumerate(scats_numbers):
        location_split = locations[index].split(" ")

        longitude = longitudes[index]
        latitude = latitudes[index]

        intersection = int(scat)

        direction = location_split[1]

        opposite_direction = get_opposite_direction(direction)

        search_str = f"{location_split[0]} {opposite_direction}".lower()

        # Search the unique dataframe for a 'Location' that contains the first location and direction
        first_loc_df = unique_df[
            (unique_df["Location"].str.lower().str.contains(search_str))
            & (unique_df["SCATS Number"] != scat)
        ]

        closest_scat = None
        min_distance = float("inf")

        # Find the closest SCAT based on longitude and latitude
        for _, row in first_loc_df.iterrows():

            # calculate the euclidean distance between the scat and the row
            dist = math.sqrt(
                (row["NB_LONGITUDE"] - longitude) ** 2
                + (row["NB_LATITUDE"] - latitude) ** 2
            )

            if dist < min_distance:
                min_distance = dist
                if scat == 4035 and direction == "W":
                # scat 4035 doesn't have a W direction on map
                    continue
                else:
                    closest_scat = row["SCATS Number"]

        entry = f"{closest_scat}_{opposite_direction}"

        if closest_scat is not None:
            if graph.get(intersection) is None:
                graph[intersection] = [entry]
            else:
                if entry not in graph[intersection]:
                    graph[intersection].append(entry)

    logger.log(graph)
    logger.log("[+] Graph generated successfully")

    return graph


def get_opposite_direction(direction):

    opposites = {
        "N": "S",
        "S": "N",
        "E": "W",
        "W": "E",
        "NE": "SW",
        "SW": "NE",
        "NW": "SE",
        "SE": "NW",
    }

    return opposites.get(direction, None)


# Get all SCAT numbers
def get_all_scats():
    global df

    return df["SCATS Number"].unique()

def does_scat_exist(scat_number):
    global df

    if scat_number == '' or scat_number is None:
        return False

    logger.log(f"Checking if SCAT number {scat_number} exists in the dataset...")

    return int(scat_number) in df["SCATS Number"].values

def get_coords_by_scat(scat_number):
    global df

    scat_number = int(scat_number)

    # get all rows with the SCAT number
    rows = df[df["SCATS Number"] == scat_number]

    # get all locations from rows
    directions = rows[['Location', 'NB_LONGITUDE', 'NB_LATITUDE']].copy()
    directions = directions.drop_duplicates(subset=['Location'])
    directions['direction'] = directions['Location'].str.split(" ", expand=True)[1]
    directions = directions[['direction', 'NB_LONGITUDE', 'NB_LATITUDE']]

    latitude = 0
    longitude = 0

    for index, row in directions.iterrows():
       #find the location that contains N
        if "N" in row['direction'] and longitude == 0:
            longitude = row['NB_LONGITUDE']
        elif "E" in row['direction'] and latitude == 0:
            latitude = row['NB_LATITUDE']

    # if the directions dataframe doesn't have a N direction then get the S direction's longitude
    if longitude == 0:
        if directions['direction'].str.contains('S').any():
            longitude = directions[directions['direction'].str.contains('S')]['NB_LONGITUDE'].iloc[0]
        else:
            longitude = directions['NB_LONGITUDE'].iloc[0]
    
    # if the directions dataframe doesn't have a E direction then get the W direction's latitude
    if latitude == 0:
        if directions['direction'].str.contains('W').any() > 1:
            latitude = directions[directions['direction'].str.contains('W')]['NB_LATITUDE'].iloc[0]
        else:
            latitude = directions['NB_LATITUDE'].iloc[0] 

    # print(scat_number, latitude, longitude)

    # add offsets to the latitude and longitude
    latitude = latitude + LAT_OFFSET
    longitude = longitude + LONG_OFFSET

    # hard coded offsets for specific scat numbers
    # if LAT_OFFSET and LONG_OFFSET are changed these would need to be recalculated
    if scat_number == 4335:
        latitude = latitude - 0.00026 # down
        longitude = longitude - 0.0005 # to left
    elif scat_number == 4030:
        latitude = latitude + 0.00062 # up
        longitude = longitude + 0.00018 # to right
    elif scat_number == 4051:
        latitude = latitude + 0.00002 # up
        longitude = longitude - 0.00035 #to left
    elif scat_number == 3126:
        longitude = longitude + 0.0001 # to right
    elif scat_number == 3662:
        latitude = latitude - 0.00015 # down
    elif scat_number == 4324:
        longitude = longitude + 0.00005 # to right

    return latitude, longitude


def calculate_speed(start, flow):
    # get the flow of the cars at the start node
    velocity = 32
    q = 1500
    A = q/(velocity**2)
    B = - 2*velocity*A

    # calculate the speed of the cars at the start node
    speed = abs((-B + math.sqrt(B**2 - 4*A*flow))/(2*A))

    return speed

def calculate_distance(start, end):
    start_lat, start_long = get_coords_by_scat(start)
    end_lat, end_long = get_coords_by_scat(end)

    # 0.01 of a degree is 1km
    # degree difference x 100
    a = abs(start_lat - end_lat) * 100
    b = abs(start_long - end_long) * 100

    # c^2 = a^2 +b^2
    c = math.sqrt(a**2 + b**2)

    return c
