
# Traffic Flow Prediction System (TFPS)

## Overview
The **Traffic Flow Prediction System (TFPS)** is a machine learning-based solution designed to predict traffic flow and recommend optimal routes for urban areas. This project focuses on the city of Boroondara, Melbourne, utilizing historical SCATS (Sydney Coordinated Adaptive Traffic System) data to provide actionable insights for traffic management and navigation.

By combining multiple machine learning models and advanced pathfinding algorithms, the TFPS offers accurate traffic flow predictions and generates efficient route recommendations tailored to real-time conditions. The project features a user-friendly graphical interface for visualization and interaction.

## Features
- **Traffic Flow Prediction:** Uses models like LSTM, GRU, CNN, and Stacked Autoencoders for short-term traffic forecasting.
- **Optimal Route Finder:** Implements the A* algorithm enhanced with traffic predictions to calculate up to five efficient routes, considering congestion, travel time, and distance.
- **Interactive User Interface:** Built with PyQt5 and Folium for real-time map visualization, including traffic heatmaps and alternate routes.
- **Data Preprocessing:** Handles large datasets to normalize, clean, and structure traffic data for machine learning.

## System Architecture
The system consists of the following key components:
1. **Data Preprocessing:**
   - Cleans and formats raw SCATS data into training-ready datasets.
   - Extracts and normalizes temporal features to enhance model performance.
2. **Predictive Models:**
   - LSTM, GRU: Captures temporal dependencies in traffic flow.
   - CNN: Processes time-series data to detect patterns.
   - Stacked Autoencoders: Reduces dimensionality and predicts traffic trends.
3. **Pathfinding Module:**
   - A* algorithm calculates optimal routes based on traffic predictions.
   - Incorporates penalties for path diversity and traffic conditions.
4. **Graphical User Interface (GUI):**
   - Displays interactive maps with traffic heatmaps.
   - Allows users to specify origin, destination, and prediction model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/minhphan6624/TrafficPredictionSystem.git
   cd TrafficPredictionSystem
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
   You can use `-l` or `--linux` to ensure linux compatibility
4. (Optional) Train the models:
   ```bash
   python train.py --model model_name
   ```
   You can choose "lstm", "gru", "saes" or "tcn" as arguments. The .keras weight file was saved in the model folder. You can also do "all" to train all models.

## Usage
1. Launch the application and select the desired prediction model (e.g., LSTM, GRU).
2. Input the origin and destination SCATS site numbers, along with the desired time interval.
3. Click "Run Pathfinding" to generate predicted traffic flow and recommended routes.
4. View results on the interactive map, including traffic heatmaps and route details.

## Dataset
The project uses SCATS traffic flow data from October 2006, provided by VicRoads. The dataset includes:
- Traffic volume recorded at 15-minute intervals across multiple intersections.
- Geographic coordinates (latitude, longitude) for each SCATS site.

## Preprocessing steps:
- Merge directional data into unified datasets for each SCATS site.
- Normalize traffic flow and temporal features using MinMaxScaler.
- Encode categorical directional data via one-hot encoding.

## Technologies
- **Programming Languages:** Python
- **Libraries:** TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy
- **Visualization Tools:** Folium, PyQt5
- **Algorithms:** A* search
- **Models:** LSTM, GRU, CNN, Stacked Autoencoders

## Results
- Models achieved high accuracy for October 2006 data, demonstrating robust prediction capabilities.
- The A* algorithm effectively incorporated traffic predictions to recommend efficient routes.
- User-friendly GUI enhanced interaction and visualization of traffic flow and routes.

## Contributors
- Quang Minh Phan
- Daniel Paolone
- Nicola Ng
- Jeremy De Jong

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Swinburne University of Technology
- VicRoads for providing traffic datasets
- Open-source libraries and frameworks that enabled this project
