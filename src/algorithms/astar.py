# Project Imports
import utilities.logger as logger
import predict as prediction_module
import algorithms.graph as graph_maker

# Library Imports
import heapq
import random

PATH_COST = 1

heuristic_dict = {}
flow_dict = {}

def heuristic_function(nodeStart, nodeEnd, date_time, model):
    global overall_time, overall_distance

    print(f"Calculating heuristic cost for NodeStart -> {nodeStart}, NodeEnd -> {nodeEnd}")

    end_scat = nodeEnd.split("_")[0]
    end_direction = nodeEnd.split("_")[1]

    # Old predict
    #flow = prediction_module.predict_flow(end_scat, date_time, end_direction, model)

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


def astar(graph, start_node, end_node, date_time, num_paths=5, model="lstm"):
    found_paths = []
    path_penalties = {}  # Store penalties for used edges
    attempts = 0
    max_attempts = 3  # Prevent infinite loops if 5 paths don't exist
    
    while len(found_paths) < num_paths and attempts < max_attempts:
        # Reinitialize search parameters
        open_set = []
        closed_set = set()
        parent = {}
        g_score = {start_node: 0}
        f_score = {start_node: 0}
        
        heapq.heappush(open_set, (f_score[start_node], start_node))
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            
            if parse_node(current_node) == end_node:
                # Path found
                logger.log(f"Found path {len(found_paths) + 1}!")
                # Reconstruct the current path
                path = []
                temp_node = current_node
                
                while temp_node:
                    path.append(parse_node(temp_node))
                    temp_node = parent.get(temp_node)
                    
                path.reverse()
                
                # Calculate metrics
                overall_time = 0
                overall_distance = 0
                
                for i in range(len(path) - 1):
                    start = path[i]
                    end = path[i + 1]
                    
                    overall_distance += heuristic_dict[f"{start}_{end}"]["distance"]
                    
                    if i != 0 and i != len(path) - 1:
                        overall_time += 0.00833333  # traffic light delay
                    
                    overall_time += heuristic_dict[f"{start}_{end}"]["distance"] / heuristic_dict[f"{start}_{end}"]["speed"]
                
                if path not in [path_info['path'] for path_info in found_paths]:
                    logger.log("Path unique - adding to list")
                    found_paths.append({
                        'path': path,
                        'distance': round(overall_distance, 2),
                        'time': round(overall_time * 60, 2)
                    })
                    
                    # Add penalties to edges in the found path
                    penalty_factor = 0.5 * (attempts + 1)  # Increase penalties with each attempt
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        if edge not in path_penalties:
                            path_penalties[edge] = 0
                        path_penalties[edge] += PATH_COST * penalty_factor
                    
                    if len(found_paths) >= num_paths:
                        break
                
                # Don't break here - continue searching for more paths
                continue

            closed_set.add(current_node)
            neighbors = graph.get(parse_node(current_node), [])

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate edge penalty
                edge = (parse_node(current_node), parse_node(neighbor))
                edge_penalty = path_penalties.get(edge, 0)
                    
                tentative_g_score = g_score[current_node] + edge_penalty
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    parent[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    
                    # Add edge penalty to heuristic calculation
                    h_score = heuristic_function(current_node, neighbor, date_time, model) + edge_penalty
                    f_score[neighbor] = g_score[neighbor] + h_score
                    
                    if neighbor not in [node for _, node in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        attempts += 1
        # Increase penalties for next attempt if we haven't found enough paths
        for edge in path_penalties:
            path_penalties[edge] *= 1.5
    
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