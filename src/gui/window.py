# Library Imports
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt5 import QtWebEngineWidgets, QtCore, QtWidgets
from folium import plugins, IFrame
import qdarktheme
import folium as folium

# System Imports
import sys

# Project Imports
import algorithms.bfs as bfs
import algorithms.astar as astar
import algorithms.graph as graph_maker
import utilities.logger as logger
import predict as prediction_module
import main as main

from utilities.time import round_to_nearest_15_minutes

# Constants
WINDOW_TITLE = "TrafficPredictionSystem"
WINDOW_SIZE = (1200, 700)
WINDOW_LOCATION = (160, 70)

# Global variables
graph = None
map_widget = None


def update_map(html):
    global map_widget

    map_widget.setHtml(html, QtCore.QUrl(""))


def create_marker(scat, map_obj):
    html = f"""
        <h4>Scat Number: {scat}</h4>
        
        """
    iframe = folium.IFrame(html=html, width=150, height=100)
    popup = folium.Popup(iframe, max_width=200)

    folium.Marker(
        graph_maker.get_coords_by_scat(int(scat)),
        popup=popup,
        tooltip=f"Scat {scat}",
        icon=folium.Icon(color="green"),
    ).add_to(map_obj)


def run_pathfinding(start, end, time):
    global graph

    logger.log(f"Running pathfinding algorithm from {start} to {end}")

    graph = graph_maker.generate_graph()

    map_obj = folium.Map(
        location=(-37.86703, 145.09159), zoom_start=13, tiles="CartoDB Positron"
    )

    logger.log(f"Using start and end node [{start}, {end}]")

    paths = astar.astar(graph, start, int(end), time)

    if paths is None or len(paths) == 0:
        logger.log("No paths found.")
        return

    # Define a list of distinct colors for different paths
    path_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkblue']
    
    # add start and end markers on the map with the displayed scat number
    create_marker(start, map_obj)
    create_marker(end, map_obj)

    # Draw each path with a different color
    for path_index, path_info in enumerate(paths):
        color = path_colors[path_index % len(path_colors)]  # Cycle through colors if more paths than colors
        
        print(path_info)
        
        logger.log(f"\nDrawing Path {path_index + 1} in {color}")
        
        # Draw the path segments
        for i in range(len(path_info['path']) - 1):
            current = path_info['path'][i]
            next_node = path_info['path'][i + 1]

            start_lat, start_long = graph_maker.get_coords_by_scat(current)
            end_lat, end_long = graph_maker.get_coords_by_scat(next_node)

            logger.log(f"Visited: {current} -> {next_node}")
            
            # Create the path line with the current color
            folium.PolyLine(
                [(start_lat, start_long), (end_lat, end_long)],
                color=color,
                weight=2.5 if path_index == 0 else 2.0,
                opacity=1.0 if path_index == 0 else 0.8,
                popup=f'Path {path_index + 1}',
                tooltip=f'Path {path_index + 1} - Segment: {current} → {next_node}'
            ).add_to(map_obj)
            
        # Add a summary for this path
        logger.log(f"Path {path_index + 1} - {len(path_info['path'])} nodes, Color: {color}")

    update_map(map_obj._repr_html_())


def make_menu():
    logger.log("Creating menu...")

    prediction_module.init()

    # Create a widget for the menu
    menu_widget = QWidget()

    # Create a layout for the menu
    menu_layout = QVBoxLayout()
    menu_widget.setLayout(menu_layout)

    # Title at the top middle
    title = QLabel(f"Traffic Prediction System v{main.VERSION}")
    title.setStyleSheet(
        "font-size: 20px; font-weight: bold; color: white; background-color: #333; padding: 5px;"
    )
    title.setAlignment(QtCore.Qt.AlignCenter)
    menu_layout.addWidget(title)

    # Two textboxes, one for "Start Scats Number", one for "End Scats Number"
    start_scats = QtWidgets.QLineEdit()
    start_scats.setPlaceholderText("Start Scats Number")
    menu_layout.addWidget(start_scats)

    end_scats = QtWidgets.QLineEdit()
    end_scats.setPlaceholderText("End Scats Number")
    menu_layout.addWidget(end_scats)

    time_select = QtWidgets.QTimeEdit()
    menu_layout.addWidget(time_select)

    # Button to run pathfinding algorithm
    run_button = QPushButton("Run Pathfinding")
    run_button.clicked.connect(
        lambda: run_pathfinding(
            start_scats.text(),
            end_scats.text(),
            round_to_nearest_15_minutes(time_select.text()),
        )
    )
    menu_layout.addWidget(run_button)

    # Add a stretcher to push buttons to the top
    menu_layout.addStretch()

    # Set the size and position of the menu
    menu_widget.setFixedWidth(int(WINDOW_SIZE[0] * 0.3))  # 20% of window width
    menu_widget.setFixedHeight(WINDOW_SIZE[1])  # 100% of window height

    return menu_widget


def create_map():
    global graph, map_widget

    logger.log("Creating map...")

    # create map
    map_obj = folium.Map(
        location=(-37.86703, 145.09159), zoom_start=13, tiles="CartoDB Positron"
    )
    map_widget = QtWebEngineWidgets.QWebEngineView()

    # Get all scat numbers and long lats
    scats = graph_maker.get_all_scats()

    logger.log(f"Creating nodes...")
    for scat in scats:
        # create map markers for the scats
        create_marker(scat, map_obj)

    return map_obj


def make_window():
    global graph, map_widget

    logger.log("Creating window...")

    # Create main widget and layout
    main_widget = QWidget()
    main_layout = QHBoxLayout()
    main_widget.setLayout(main_layout)
    main_layout.setSpacing(0)  # Set spacing to zero

    update_map(create_map()._repr_html_())

    map_widget.page().setBackgroundColor(QtCore.Qt.transparent)

    # Add map and menu to layout
    main_layout.addWidget(make_menu())
    main_layout.addWidget(map_widget)

    return main_widget


def run():
    global app, graph

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setGeometry(
        WINDOW_LOCATION[0], WINDOW_LOCATION[1], WINDOW_SIZE[0], WINDOW_SIZE[1]
    )

    graph_maker.load_data()
    window.setCentralWidget(make_window())

    logger.log("Window created.")

    window.show()
    app.exec()
