"""
Optimized helper functions for batch BFS computation.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Union
import networkx as nx
import igraph as ig


def get_all_neighborhoods_batch_nx(G: nx.Graph, max_hops: int, nodes_to_process: List[int]) -> Dict[int, Dict[int, Set[int]]]:
    """
    Compute neighborhoods for all nodes in a single pass using multi-source BFS.
    This is optimized for dense graphs where neighborhoods overlap significantly.
    
    Args:
        G: NetworkX graph
        max_hops: Maximum number of hops
        nodes_to_process: List of nodes to get neighborhoods for
        
    Returns:
        Dict mapping each node to its neighborhoods at each hop level
    """
    neighborhoods = {node: defaultdict(set) for node in nodes_to_process}
    
    # For small graphs or sparse graphs, fall back to individual BFS
    if len(nodes_to_process) < 10 or G.number_of_edges() < len(nodes_to_process) * 5:
        # Use original implementation for sparse graphs
        from NEExT.helper_functions import get_nodes_x_hops_away
        for node in nodes_to_process:
            neighborhoods[node] = get_nodes_x_hops_away(G, node, max_hops)
        return neighborhoods
    
    # Multi-source BFS for dense graphs
    # Track distances from all source nodes simultaneously
    distances = defaultdict(lambda: defaultdict(lambda: float('inf')))
    
    # Initialize queue with all source nodes
    queue = deque()
    for node in nodes_to_process:
        queue.append((node, node, 0))  # (current_node, source_node, distance)
        distances[node][node] = 0
    
    # Process BFS from all sources simultaneously
    while queue:
        current, source, dist = queue.popleft()
        
        if dist >= max_hops:
            continue
        
        for neighbor in G.neighbors(current):
            # Check if we've found a shorter path to this neighbor from this source
            if distances[source][neighbor] > dist + 1:
                distances[source][neighbor] = dist + 1
                queue.append((neighbor, source, dist + 1))
                
                # Add to appropriate hop level
                neighborhoods[source][dist + 1].add(neighbor)
    
    # Convert defaultdicts to regular dicts
    result = {}
    for node in nodes_to_process:
        result[node] = dict(neighborhoods[node])
    
    return result


def get_all_neighborhoods_batch_ig(G: ig.Graph, max_hops: int, nodes_to_process: List[int]) -> Dict[int, Dict[int, List[int]]]:
    """
    Compute neighborhoods for all nodes using igraph's efficient methods.
    
    Args:
        G: iGraph graph
        max_hops: Maximum number of hops
        nodes_to_process: List of nodes to get neighborhoods for
        
    Returns:
        Dict mapping each node to its neighborhoods at each hop level
    """
    neighborhoods = {}
    
    # Use igraph's efficient neighborhood function
    for node in nodes_to_process:
        neighborhoods[node] = {}
        
        # Get all nodes within max_hops
        all_neighbors = set(G.neighborhood(node, order=max_hops))
        all_neighbors.discard(node)  # Remove the node itself
        
        # Now separate by hop distance
        for hop in range(1, max_hops + 1):
            if hop == 1:
                # Direct neighbors
                neighborhoods[node][hop] = list(G.neighbors(node))
            else:
                # Get nodes at exactly this hop distance
                nodes_at_hop = set(G.neighborhood(node, order=hop))
                nodes_at_prev_hop = set(G.neighborhood(node, order=hop-1))
                neighborhoods[node][hop] = list(nodes_at_hop - nodes_at_prev_hop)
    
    return neighborhoods