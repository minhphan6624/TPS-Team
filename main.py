from tkinter import Tk
import pandas as pd
from bfs import bfs
from app import App


def main():
    # Load in the 'scats_data.csv' file 
    df = pd.read_csv('data\\scats_data.csv')

    # Print column names
    print(df.columns)

    # get unique values of 'Location' column
    locations = df['Location']

    # get scats number
    scats_numbers = df['SCATS Number']

    # Check locations and scats_numbers length is the same
    print(len(locations))

    print(len(scats_numbers))

    # Replace HIGH STREET_RD with 'HIGH_STREET_RD'
    locations = [location.replace('HIGH STREET_RD', 'HIGH_STREET_RD') for location in locations]

    # Same for STUDLEY PARK_RD
    locations = [location.replace('STUDLEY PARK_RD', 'STUDLEY_PARK_RD') for location in locations]

    # Same for MONT ALBERT_RD
    locations = [location.replace('MONT ALBERT_RD', 'MONT_ALBERT_RD') for location in locations]
    
    print(locations)

    graph = {}


    for index,scat in enumerate(scats_numbers):
            location_split = locations[index].split(' ')

            # WARRIGAL_RD N of HIGH STREET_RD

            intersection = str(scat)
            direction = location_split[1]
            first_loc = location_split[0] + "_" + direction

            # Check if graph[first_loc] is an empty list
            if graph.get(intersection) == None:
                graph[intersection] = [first_loc]
            else:
                if first_loc in graph[intersection]:
                    pass
                else:
                    graph[intersection].append(first_loc)

            print('Added edge from {} to {}'.format(first_loc, intersection))

    print(graph.keys())


    app = App(graph)
    app.run()

    # get input values
    start, end = app.get_scat_numbers()

    # calls bfs function
    path = bfs(graph, start, end)

    if path:
        print('Path from {} to {}:'.format(start, end))
        print(' -> '.join(path))
    else:
        print('No path found from {} to {}'.format(start, end))


if __name__ == '__main__':
    main()