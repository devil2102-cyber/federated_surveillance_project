import networkx as nx
import math

def calculate_mle_source(graph: nx.Graph, infected_nodes: list) -> int:
    """
    Calculates Maximum-Likelihood Estimation for the source of an outbreak.
    Simplified approach based on minimizing path distances to infected nodes.
    """
    if not graph.nodes or not infected_nodes:
        return None
        
    best_source = None
    max_likelihood = -float('inf')
    
    for candidate in graph.nodes:
        try:
            total_distance = 0
            for inf_node in infected_nodes:
                if candidate == inf_node:
                    continue
                # Calculate shortest path weighted by interaction duration
                distance = nx.shortest_path_length(graph, source=candidate, target=inf_node, weight='weight')
                total_distance += distance
            
            # The smaller the total distance to all infected, the higher the likelihood of being the source
            likelihood = -total_distance if total_distance > 0 else float('inf')
                
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_source = candidate
                
        except nx.NetworkXNoPath:
            continue
            
    return best_source
