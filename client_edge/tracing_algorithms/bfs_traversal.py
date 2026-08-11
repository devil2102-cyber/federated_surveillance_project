import networkx as nx
from typing import List, Any

def bfs_find_cluster(graph: nx.Graph, start_node_id: Any, max_depth: int = 2) -> List[Any]:
    """Finds a local transmission cluster using Breadth-First Search."""
    if start_node_id not in graph:
        return []
        
    cluster = []
    # Queue stores tuples of (node, depth)
    queue = [(start_node_id, 0)]
    visited = set([start_node_id])
    
    while queue:
        current_node, depth = queue.pop(0)
        cluster.append(current_node)
        
        if depth < max_depth:
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    
    return cluster
