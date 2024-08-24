# COS30018 – Intelligent Systems

## Project Assignment – Option A - (Topic 2)

## Running the Project

```
<your python alias> main.py
```

You can use `-l` or `--linux` to ensure linux compatibility

## Training the models

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The `.h5` weight file was saved in the model folder.

### Summary

### Traffic Flow Prediction System

- **Due:** 11:59 pm 27/10/2024 (End of Week 12)
- **Contributes:** 50% of your final result
- **Group Assignment:** Group of 2-4 students

This project requires your group to implement and demonstrate a traffic flow prediction system, TFPS. Accurate and timely traffic flow information is important for traffic authorities (such as VicRoads) to identify congested areas and implement traffic management policies to reduce congestion. It is also critical for route guidance systems (such as Google Maps) to calculate the best routes for their users considering the traffic conditions potentially experienced along each route. Thanks to recent advancements in sensor technologies, traffic authorities can collect massive amounts of traffic data to enable accurate predictions of traffic information (such as traffic flow, speed, travel time, etc.). Your group will be required to develop a solution to utilise historical traffic data for traffic flow prediction. There are many versions of this problem, but we will start with the basic one and suggest several extensions to enable you to do extra research and achieve a higher mark.

The basic system will consider a small dataset from VicRoads for the city of Boroondara that contains only the traffic flow data (the number of cars passing an intersection every 15 minutes). Your task is to develop a system TFPS that takes an input traffic flow dataset in a predefined format (to be specified by you) and train a machine learning model to allow relevant traffic conditions at specified times to be predicted.

### Constraints:

The TFPS can use any machine learning technique or combinations of them. You should take advantage of existing libraries such as PyTorch, TensorFlow, Keras, Theano, etc. At the very least, your TFPS system will need to include the techniques implemented in this project: [TrafficFlowPrediction](https://github.com/xiaochus/TrafficFlowPrediction) and adapt their code to work with the VicRoads dataset. Your TFPS system will also need to implement at least one other technique (to be approved by the tutor) to train another machine learning (ML) model and give a comprehensive comparison between different models. Note that one of the techniques implemented in the TrafficFlowPrediction project is the Stacked Autoencoder (SAE). However, their implementation is unconventional and may not be considered a proper SAE model. After learning about SAE networks, you should be able to examine the implementation in the TrafficFlowPrediction project and adapt the implementation to follow a standard SAE model. Your tutor will discuss this in the tutorial.

- **Minimum Requirements:**
  - The TFPS will have to be able to train ML models using the Boroondara dataset and give meaningful predictions based on these models.
  - A GUI will be available for user input, parameter settings, and visualisation (and a configuration file for the defaults).

### System Requirements:

- **Core Functionality:** One core functionality of the TFPS system is to provide route guidance to drivers.
  - **Basic Version:** The basic version of the route guidance is for the Boroondara area. A user can specify the origin and destination of their trip as the SCATS site number (e.g., origin O = 2000 [intersection WARRIGAL_RD/TOORAK_RD] and destination D = 3002 [intersection DENMARK_ST/BARKERS_RD]). The system then returns up to five (5) routes from O to D with the estimated travel time along each route. To simplify the calculation, you can make a number of assumptions: (i) The speed limit on every link will be the same and set at 60km/h; (ii) the travel time between two SCATS sites A and B can be approximated by a simple expression based on the volume at the SCATS site A and the distance between A and B (you can also learn more about the [Fundamental Diagram](https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow)); and (iii) there is an average delay of 30 seconds to pass each controlled intersection. Note that the objective is not to better Google Maps but to utilise the AI techniques you have learned (e.g., machine learning for forecasting traffic volume, graph-based search for optimal paths) to solve a real-world problem.
  - **Extension Option 1:** The basic version of the system only deals with a small area of Melbourne (the city of Boroondara) and uses a very small dataset (traffic volume data for the month of October 2006). Under this extension, you can look at more comprehensive datasets from VicRoads for the whole of Victoria in multiple years. Data processing for a large amount of data will be a challenge.
  - **Extension Option 2:** Dealing with VicRoads datasets may not be straightforward. You may want to consider other sources of open data. Two popular data sources are: [PeMS Data Information](https://github.com/mas-dse-c6sander/DSE_Cohort2_Traffic_Capstone/wiki/PeMS-Data-Information) and [Highways England Network Journey Time and Traffic Flow Data](https://www.data.gov.uk/dataset/9562c512-4a0b-45ee-b6ad-afc0f99b841f/highways-england-network-journey-time-and-traffic-flow-data). Under this extension, you can choose one of these datasets or a good data source you have access to (and get your tutor’s approval) and extend this project to deal with the selected network and dataset.
  - **Extension Option 3:** Visualising your system predictions and route recommendations. Inspired by Google Maps Traffic, can you build something similar using open-source resources such as OpenStreetMap?

### Project Requirements:

- **Source Code:** Maintained on a Git-based VCS (GitHub/Bitbucket/GitLab/…). You must provide read-only access to the tutor/lecturer.
- **Running Illustrative Demo:** A working prototype (please refer to Marking Scheme for details on functionality that needs to be implemented).
- **Project Report:** (8-10 pages) that includes the following sections:
  - Cover Page (with team details) and a Table of Contents (TOC)
  - Introduction
  - Overall system architecture
  - Implemented interaction protocols
  - Implemented search/optimization techniques
  - Scenarios/examples to demonstrate how the system works
  - Some critical analysis of the implementation
  - Summary/Conclusion
- **Presentation:** (Video, 6-8 minutes)

### Marking Scheme:

| Requirements                                                                                                                                                                                                      | Mark        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Task 1: Adapt the open-source Traffic Flow Prediction with Neural Networks project to work successfully with the Boroondara dataset.                                                                              | 10          |
| Task 2: Implement new ML technique(s) for traffic flow prediction and carry out experiments to compare different ML models.                                                                                       | 20          |
| Task 3: Complete the Basic version                                                                                                                                                                                | 20          |
| GUI: for a user to perform basic tasks with the TFPS system such as predicting the volume at a SCATS site or finding alternative routes from O to D                                                               | 10          |
| Project Report                                                                                                                                                                                                    | 10          |
| Project Presentation (Video, 6-8 minutes)                                                                                                                                                                         | 10          |
| **Total**                                                                                                                                                                                                         | 80          |
| **Research Component:** (can be done by the whole team, a sub-team, or an individual). Select one of the Extensions and get your tutor’s approval then complete it very well.                                     | Up to 20    |
| **Grand Total**                                                                                                                                                                                                   | **100/100** |
| **Penalties:**                                                                                                                                                                                                    |
| You need to follow good programming practices (e.g., well-designed, well-structured code with clear and helpful comments). Failure to do so will get a penalty.                                                   | Up to -20   |
| You need to demonstrate the progress you make every week to your tutor by following a clear project plan the team has agreed with your tutor after the team has been formed. Failure to do so will get a penalty. | Up to -50   |

**NOTE:** Individual marks will be proportionally adjusted based on each team member’s overall contribution to the project as indicated in the ‘Who did what’ declaration.

### Submission:

- At least one member of the team must submit the entire project (code + report) as a .zip file to Canvas by 11:59 pm on 27/10/2024. Create a single zip file with your code and a working version of your system. Standard late penalties apply – 10% for each day late, more than 5 days late is 0%.
- You must also provide your tutor with read-only access to your Git repository within 1 week of forming teams.
- The video (6-8 minutes duration) should be submitted to Canvas by at least one member of the team by 11:59 pm on Tuesday 29/10/2024.

### Appendix:

- (If applicable, append any additional material or instructions here.)
