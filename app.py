from tkinter import *
import pandas as pd
import json
import math

import bfs as bfs

class EntryObject:
    def __init__(self, location, potential_scats, closest_scat):
        self.location = location
        self.potential_scats = potential_scats
        self.closest_scat = closest_scat
        
    def __eq__(self, other):
        if not isinstance(other, EntryObject):
            return False
        
        # Compare the location directly
        if self.location != other.location:
            return False
        
        # Check if the closest scat is the same
        if self.closest_scat != other.closest_scat:
            return False
        
        # Compare the potential_scats arrays element-wise
        return (self.potential_scats == other.potential_scats).all()
    
    def __repr__(self):
        return json.dumps({
            'location': self.location,
            'potential_scats': self.potential_scats.tolist(),
            'closest_scat': self.closest_scat
        })
    
    def __str__(self):
        return self.__repr__()
class App:

    def __init__(self, linux):
        self.linux = linux
        self.graph = self.generate_graph()
        # self.path = path
        self.start_entry = ""
        self.end_entry = ""

        # create the window
        self.window = Tk()
        self.window.title("TrafficPredictionSystem")

        # create fields
        # start input
        Label(self.window, text="Start SCAT Number:").grid(row=0, column=0, padx=10, pady=10)
        self.start_entry = Entry(self.window)
        self.start_entry.grid(row=0, column=1, padx=10, pady=10)
        # end input
        Label(self.window, text="End SCAT Number:").grid(row=1, column=0, padx=10, pady=10)
        self.end_entry = Entry(self.window)
        self.end_entry.grid(row=1, column=1, padx=10, pady=10)
        # submit button
        submit_button = Button(self.window, text="Submit", command=self.submit)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20)

    def generate_graph(self):
        # Load in the 'scats_data.csv' file
        file_location = (
            "./data/scats_data.csv" if self.linux else "data\\scats_data.csv"
        )

        df = pd.read_csv(file_location)

        # Fix location names.
        df['Location'] = df['Location'].replace({
            'HIGH STREET_RD': 'HIGH_STREET_RD',
            'STUDLEY PARK_RD': 'STUDLEY_PARK_RD',
            'MONT ALBERT_RD': 'MONT_ALBERT_RD'
        }, regex=True)

        # get unique values of 'Location' column
        locations = df['Location']

        # get longitude and latitude
        longitudes = df['NB_LONGITUDE']
        latitudes = df['NB_LATITUDE']

        # get scats number
        scats_numbers = df['SCATS Number']

        # Check locations and scats_numbers length is the same
        print(f"Locations: {len(locations)}")
        print(f"Longitudes: {len(longitudes)}")

        # Compute a seperate dataframe which is all unique rows by Location
        unique_df = df.drop_duplicates(subset=['Location'])

        graph = {}

        for index,scat in enumerate(scats_numbers):
            location_split = locations[index].split(" ")

            longitude = longitudes[index]
            latitude = latitudes[index]

            # WARRIGAL_RD N of HIGH STREET_RD

            intersection = str(scat)
            direction = location_split[1]
            first_loc = location_split[0] + "_" + direction

            opposite_direction = self.get_opposite_direction(direction)

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
                dist = math.sqrt(
                    (row["NB_LONGITUDE"] - longitude) ** 2
                    + (row["NB_LATITUDE"] - latitude) ** 2
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_scat = row["SCATS Number"]

            entry = EntryObject(
                first_loc, first_loc_df["SCATS Number"].values, closest_scat
            )

            # Check if graph[first_loc] is an empty list
            if graph.get(intersection) == None:
                graph[intersection] = [entry]
            else:
                # Check if the entry is already in the list
                if entry not in graph[intersection]:
                    graph[intersection].append(entry)

            # print('Added edge from {} to {}'.format(first_loc, intersection))

        print(graph)

        print("[+] Graph generated successfully")

        return graph

    def get_opposite_direction(self, direction):
        # Define a dictionary mapping each direction to its opposite
        opposites = {
            'N': 'S', 'S': 'N',
            'E': 'W', 'W': 'E',
            'NE': 'SW', 'SW': 'NE',
            'NW': 'SE', 'SE': 'NW'
        }

        # Return the opposite direction using the dictionary
        return opposites.get(direction, None)

    def run(self):
        # run app
        self.window.mainloop()

    def get_scat_numbers(self):
        start = self.start_entry.get()
        end = self.end_entry.get()
        return start, end

    def submit(self):
        start, end = self.get_scat_numbers()
        print(f"Start SCAT Number: {start}")
        print(f"End SCAT Number: {end}")

        # calls bfs function
        path = bfs.bfs(self.graph, start, end)

        if path:
            print('Path from {} to {}:'.format(start, end))
            print(' -> '.join(path))
        else:
            print('No path found from {} to {}'.format(start, end))        
