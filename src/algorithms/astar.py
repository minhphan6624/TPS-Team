# Project Imports
import utilities.logger as logger

# Library Imports
import heapq

def heuristic_function(a, b):
    print(f'Calculating heuristic for {a} and {b}')


    return 30

def parse_node(node_str):
    # Extract the numeric part of the node
    return int(node_str.split('_')[0])

def astar(graph, start_node, end_node):
    open_set = []
    closed_set = set()
    
    parent = {}
    
    g_score = {start_node: 0}
    f_score = {start_node: heuristic_function(start_node, end_node)}
    
    heapq.heappush(open_set, (f_score[start_node], start_node))
    
    while open_set:
        current_f, current_node = heapq.heappop(open_set)
        
        logger.log(f'Visiting: {current_node}')
        
        if parse_node(current_node) == end_node:
            logger.log('Found the end node!')
            path = []
            while current_node:
                path.append(parse_node(current_node))
                current_node = parent.get(current_node)
            path.reverse()
            return path
        
        closed_set.add(current_node)
        
        neighbors = graph.get(parse_node(current_node), [])
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current_node] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                parent[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_function(parse_node(neighbor), end_node)
                
                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    logger.log('No path found')
    return None