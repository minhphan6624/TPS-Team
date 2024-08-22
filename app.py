# from tkinter import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWebEngineWidgets, QtCore
from pathlib import Path
import pandas as pd
import json
import math
import folium as folium
import sys


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

class MainWindow(QMainWindow):
    def __init__(self,linux):
        self.linux = linux
        self.app = QApplication(sys.argv)
        super(MainWindow, self).__init__()

        self.df = None

        self.graph = self.generate_graph()
        self.start_entry = ""
        self.end_entry = ""

        self.positions = []

        # pyqt window 
        self.setWindowTitle("TrafficPredictionSystem")
        self.setGeometry(160, 70, 1200, 700)

        self.setCentralWidget(self.make_window())        

    def make_window(self):
        # create map
        map = folium.Map(location=(-37.86703, 145.09159), zoom_start=13, tiles='CartoDB Positron')
        # only show street names

        # Unique long/lats for each location
        longitudes = self.df['NB_LONGITUDE'].unique() 
        latitudes = self.df['NB_LATITUDE'].unique()

        self.lat_offset = 0.0015
        self.long_offset = 0.0013

        path = bfs.bfs(self.graph, int(3682), int(4030))

        # Loop through path array and draw line between each pair of points
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            start_lat, start_long = self.get_coords_by_scat(start)
            end_lat, end_long = self.get_coords_by_scat(end)

            folium.PolyLine([(start_lat, start_long), (end_lat, end_long)], color="red", weight=2.5, opacity=1).add_to(map)

        # Print marker at start and end of path
        start_lat, start_long = self.get_coords_by_scat(path[0])

        #folium.Marker(location=(start_lat, start_long), popup=f"Start: {path[0]}").add_to(map)

        end_lat, end_long = self.get_coords_by_scat(path[-1])

        #folium.Marker(location=(end_lat, end_long), popup=f"End: {path[-1]}").add_to(map)
       
        # 
        # 


        """
        for index, row in enumerate(longitudes):
            if index > 500:
                break

            if latitudes[index] != 0 and row != 0:
                modified_lat = latitudes[index] + self.lat_offset
                modified_long = row + self.long_offset

                folium.Marker(location=(modified_lat, modified_long), popup=f"{latitudes[index]}, {row}").add_to(map)
        """

        map.save("map1.html")

        # read from file (probably could move this to just use the map return from above^)
        with open('map1.html', 'r') as file:  # r to open file in READ mode
            map_html = file.read()

        # create map in window
        map_view = QtWebEngineWidgets.QWebEngineView(self)
        map_view.setHtml(map_html)

        return map_view
    
    def get_coords_by_scat(self, scat_number):
        # Get the row with the SCAT number
        row = self.df[self.df['SCATS Number'] == scat_number]

        # Get the latitude and longitude from the row
        latitude = row['NB_LATITUDE'].values[0] + self.lat_offset
        longitude = row['NB_LONGITUDE'].values[0] + self.long_offset

        return latitude, longitude

    def generate_graph(self):
        # Load in the 'scats_data.csv' file
        file_location = (
            "./data/scats_data.csv" if self.linux else "data\\scats_data.csv"
        )

        self.df = pd.read_csv(file_location)

        # Fix location names.
        self.df['Location'] = self.df['Location'].replace({
            'HIGH STREET_RD': 'HIGH_STREET_RD',
            'STUDLEY PARK_RD': 'STUDLEY_PARK_RD',
            'MONT ALBERT_RD': 'MONT_ALBERT_RD'
        }, regex=True)

        # get unique values of 'Location' column
        locations = self.df['Location']

        # get longitude and latitude
        longitudes = self.df['NB_LONGITUDE']
        latitudes = self.df['NB_LATITUDE']

        # get scats number
        scats_numbers = self.df['SCATS Number']

        # Check locations and scats_numbers length is the same
        print(f"Locations: {len(locations)}")
        print(f"Longitudes: {len(longitudes)}")

        # Compute a seperate dataframe which is all unique rows by Location
        unique_df = self.df.drop_duplicates(subset=['Location'])

        graph = {}
           

        for index,scat in enumerate(scats_numbers):
            location_split = locations[index].split(" ")

            longitude = longitudes[index]
            latitude = latitudes[index]

            # WARRIGAL_RD N of HIGH STREET_RD

            intersection = int(scat)
            direction = location_split[1]

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

            entry = closest_scat

            if entry is not None:
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

        # Convert graph to be an object

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
        # self.window.mainloop()
        self.show()
        self.app.exec()
    
    def get_scat_numbers(self):
        start = self.start_entry.get()
        end = self.end_entry.get()
        return start, end

    def submit(self):
        start, end = self.get_scat_numbers()
        print(f"Start SCAT Number: {start}")
        print(f"End SCAT Number: {end}")

        # calls bfs function
        path = bfs.bfs(self.graph, int(start), int(end))

        print("Path is below...")
        print(path)