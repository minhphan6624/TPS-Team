# Project Imports
import utilities.logger as logger
import predict as prediction_module
import algorithms.graph as graph_maker

# Library Imports
import heapq

PATH_COST = 30

heuristic_dict = {}


def heuristic_function(nodeStart, nodeEnd, time):
    global overall_time, overall_distance

    print(
        f"Calculating heuristic cost for NodeStart -> {nodeStart}, NodeEnd -> {nodeEnd}"
    )

    end_scat = nodeEnd.split("_")[0]
    end_direction = nodeEnd.split("_")[1]

    flow = prediction_module.predict_flow(end_scat, time, end_direction, "gru")

    if "_" in nodeStart:
        nodeStart = nodeStart.split("_")[0]

    distance = graph_maker.calculate_distance(nodeStart, end_scat)
    speed = graph_maker.calculate_speed(nodeStart, flow)

    heuristic_dict[f"{nodeStart}_{end_scat}"] = {"distance": distance, "speed": speed}

    return distance / speed


def parse_node(node_str):
    return int(node_str.split("_")[0])


def astar(graph, start_node, end_node, start_time):
    open_set = []
    closed_set = set()

    parent = {}

    start_with_direction = f"{start_node}_N"

    print("Got start with direction -> ", start_with_direction)

    g_score = {start_node: 0}
    f_score = {start_node: 0}

    heapq.heappush(open_set, (f_score[start_node], start_node))

    while open_set:
        current_f, current_node = heapq.heappop(open_set)

        logger.log(f"Visiting: {current_node}")

        if parse_node(current_node) == end_node:
            logger.log("Found the end node!")

            path = []

            while current_node:
                path.append(parse_node(current_node))
                current_node = parent.get(current_node)

            path.reverse()

            overall_time = 0
            overall_distance = 0

            # loop over path
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]

                overall_distance += heuristic_dict[f"{start}_{end}"]["distance"]

                if i != 0 and i != len(path) - 1:
                    overall_time += 0.00833333  # add 30 seconds for traffic light delay

                overall_time += (
                    heuristic_dict[f"{start}_{end}"]["distance"]
                    / heuristic_dict[f"{start}_{end}"]["speed"]
                )

            logger.log(f"Distance: {round(overall_distance,2)} km")
            logger.log(f"Time: {round(overall_time*60,2)} minutes")

            return path

        closed_set.add(current_node)

        neighbors = graph.get(parse_node(current_node), [])
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_node] + PATH_COST

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                parent[neighbor] = current_node
                g_score[neighbor] = tentative_g_score

                # TODO This will need to be updated to account for the time already spent driving/traversing previous nodes
                f_score[neighbor] = g_score[neighbor] + heuristic_function(
                    current_node, neighbor, start_time
                )

                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    logger.log("No path found")
    return None
