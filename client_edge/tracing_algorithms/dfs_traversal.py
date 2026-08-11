import networkx as nx
from typing import List, Any

def dfs_trace_source(graph: nx.Graph, current_node_id: Any, max_depth: int = 5) -> List[Any]:
    """Deep historical source tracing using Depth-First Search."""
    if current_node_id not in graph:
        return []
        
    path = []
    visited = set()
    
    def _dfs(node, depth):
        if depth > max_depth or node in visited:
            return
        
        visited.add(node)
        path.append(node)
        
        for neighbor in graph.neighbors(node):
            _dfs(neighbor, depth + 1)
            
    _dfs(current_node_id, 0)
    return path
