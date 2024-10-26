# Project Imports
import utilities.logger as logger
import predict as prediction_module
import algorithms.graph as graph_maker

# Library Imports
import heapq
import random

PATH_COST = 30

heuristic_dict = {}
flow_dict = {}

def heuristic_function(nodeStart, nodeEnd, date_time, model):
    global overall_time, overall_distance

    print(f"Calculating heuristic cost for NodeStart -> {nodeStart}, NodeEnd -> {nodeEnd}")

    end_scat = nodeEnd.split("_")[0]
    end_direction = nodeEnd.split("_")[1]

    # New predict
    flow = prediction_module.predict_new_model(end_scat, date_time, end_direction, model)

    # add flow by scat to dictionary
    flow_dict[end_scat] = flow

    if "_" in nodeStart:
        nodeStart = nodeStart.split("_")[0]

    distance = graph_maker.calculate_distance(nodeStart, end_scat)
    speed = graph_maker.calculate_speed(nodeStart, flow)

    heuristic_dict[f"{nodeStart}_{end_scat}"] = {"distance": distance, "speed": speed}

    return distance / speed


def parse_node(node_str):
    return int(node_str.split("_")[0])


def astar(graph, start_node, end_node, date_time, num_paths=5, model = "lstm"):
    open_set = []
    closed_set = set()
    found_paths = []

    parent = {}
    
    g_score = {start_node: 0}
    f_score = {start_node: 0}
    
    heapq.heappush(open_set, start_node)
    
    while open_set and len(found_paths) < num_paths:
        current_node = heapq.heappop(open_set)

        #logger.log(f"Visiting: {current_node}")
        
        if parse_node(current_node) == end_node:
            logger.log(f"Found path {len(found_paths) + 1}!")
            
            # Reconstruct the current path
            path = []
            temp_node = current_node
            
            while temp_node:
                path.append(parse_node(temp_node))
                temp_node = parent.get(temp_node)
                
            path.reverse()
            
            # Calculate metrics for this path
            overall_time = 0
            overall_distance = 0
            
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                
                overall_distance += heuristic_dict[f"{start}_{end}"]["distance"]
                
                if i != 0 and i != len(path) - 1:
                    overall_time += 0.00833333  # add 30 seconds for traffic light delay
                
                overall_time += heuristic_dict[f"{start}_{end}"]["distance"] / heuristic_dict[f"{start}_{end}"]["speed"]
            
            found_paths.append({
                'path': path,
                'distance': round(overall_distance, 2),
                'time': round(overall_time * 60, 2)
            })
            
            # Don't stop here - continue searching for alternative paths
            closed_set.add(current_node)
            continue
        
        closed_set.add(current_node)
        neighbors = graph.get(parse_node(current_node), [])

        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
                
            tentative_g_score = g_score[current_node] + PATH_COST
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Store the parent relationship
                parent[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_function(current_node, neighbor, date_time, model)
                
                if neighbor not in [node for node in open_set]:
                    heapq.heappush(open_set, neighbor)
    
    if not found_paths:
        logger.log("No paths found")
        return None
    
    found_paths.sort(key=lambda x: x['time'])
    
    for i, path_info in enumerate(found_paths):
        logger.log(f"Path {i + 1}:")
        logger.log(f"Nodes: {path_info['path']}")
        logger.log(f"Distance: {path_info['distance']} km")
        logger.log(f"Time: {path_info['time']} minutes")
    
    return found_paths