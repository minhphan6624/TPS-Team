from collections import deque

def bfs(graph, start_node, end_node):
    # Initialize a queue for BFS and add the start node
    queue = deque([start_node])
    # Keep track of visited nodes to avoid cycles
    visited = set()
    visited.add(start_node)
    # Keep track of the parent of each node to reconstruct the path
    parent = {start_node: None}

    while queue:
        # Dequeue a node from the front of the queue
        current_node = queue.popleft()
        print('Visiting:', current_node)

        # Check if we've reached the end node
        if current_node == end_node:
            print('Found the end node!')
            # Reconstruct the path from end_node to start_node
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()
            return path

        # Get all adjacent nodes of the current node
        neighbors = graph.get(current_node, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                # If the neighbor hasn't been visited, add it to the queue and mark it as visited
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current_node

    return None  # If no path found