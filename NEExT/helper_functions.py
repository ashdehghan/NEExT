import networkx as nx
import igraph as ig
from typing import List, Set, Dict, Union
from functools import lru_cache


def divide_chunks(list, chunks):
    for i in range(0, len(list), chunks):
        yield list[i:i + chunks]


def get_numb_of_nb_x_hops_away(G: Union[nx.Graph, ig.Graph], node: int, max_hop_length: int) -> List[int]:
    """
    Compute the number of neighbors x hops away from a given node.
    Supports both NetworkX and iGraph backends.

    Args:
        G: Graph object (NetworkX or iGraph)
        node: Source node ID
        max_hop_length: Maximum hop distance to consider

    Returns:
        List[int]: List where index i contains number of nodes i+1 hops away
    """
    hop_dict = get_nodes_x_hops_away(G, node, max_hop_length)
    return [len(hop_dict.get(i, set())) for i in range(1, max_hop_length + 1)]


def get_nodes_x_hops_away(G: Union[nx.Graph, ig.Graph], node: int, max_hop_length: int) -> Dict[int, Set[int]]:
    """This function should now only be used for single-node queries."""
    if isinstance(G, nx.Graph):
        seen = {node}
        current_set = {node}
        hop_dict = {}
        
        for hop in range(1, max_hop_length + 1):
            boundary = nx.node_boundary(G, current_set)
            next_hop = boundary - seen
            
            if not next_hop:
                break
                
            hop_dict[hop] = next_hop
            seen.update(boundary)
            current_set = boundary
            
        return hop_dict
    else:
        raise ValueError("For iGraph, use get_all_neighborhoods_ig instead")


def get_all_neighborhoods_nx(G: nx.Graph, max_hop_length: int) -> Dict[int, Dict[int, Set[int]]]:
    """Compute ALL neighborhoods for ALL nodes in a NetworkX graph efficiently."""
    result = {}
    
    # Process each node
    for node in G.nodes():
        seen = {node}
        current_set = {node}
        node_neighborhoods = {}
        
        # For each hop distance
        for hop in range(1, max_hop_length + 1):
            # Get nodes exactly hop steps away using node_boundary
            boundary = nx.node_boundary(G, current_set)
            next_hop = boundary - seen
            
            if not next_hop:  # No more nodes to explore
                break
                
            node_neighborhoods[hop] = next_hop
            seen.update(boundary)
            current_set = boundary
            
        result[node] = node_neighborhoods
    
    return result


def get_all_neighborhoods_ig(G: ig.Graph, max_hop_length: int) -> Dict[int, Dict[int, Set[int]]]:
    """Compute ALL neighborhoods for ALL nodes in ONE batch operation."""
    result = {}
    
    for node in range(G.vcount()):
        node_neighborhoods = {}
        prev_neighbors = {node}
        
        for hop in range(1, max_hop_length + 1):
            # Get neighbors of all nodes in previous hop
            curr_neighbors = set()
            for prev_node in prev_neighbors:
                curr_neighbors.update(G.neighbors(prev_node))
            
            # Remove already seen nodes
            hop_neighbors = curr_neighbors - prev_neighbors
            if not hop_neighbors:
                break
            
            node_neighborhoods[hop] = hop_neighbors
            prev_neighbors.update(curr_neighbors)
        
        result[node] = node_neighborhoods
    
    return result


def get_specific_in_community_degree(G, node_id, community_partition: List[List[int]],
                                     community_id: int) -> int:
    """
    This method will compute the community degree of a node for a specific community.

    Returns an integer, which is the in-community degree of the node for the specified community.
    """
    neighbors = list(G.neighbors(node_id))
    return len([n for n in neighbors if n in community_partition[community_id]])


def get_all_in_community_degrees(G, node_id, community_partition: List[List[int]]) -> List[int]:
    """
    This method will compute the community degree of a node for each of the communities.

    Returns a list of integers,
    where each integer is the in-community degree of the node for that community.
    """
    in_community_degrees = []
    for i, community in enumerate(community_partition):
        in_community_degrees.append(
            get_specific_in_community_degree(G, node_id, community_partition, i)
            )

    return in_community_degrees


def get_own_in_community_degree(G, node_id, community_partition: List[List[int]]) -> int:
    """
    This method will compute the community degree of a node for the community it is in.

    Returns an integer, which is the in-community degree of the node for its community.
    """
    for i, community in enumerate(community_partition):
        if node_id in community:
            return get_specific_in_community_degree(G, node_id, community_partition, i)


def get_specific_community_volume(G, community_partition: List[List[int]],
                                  community_id: int) -> int:
    """
    This method will compute the volume of a specific community in the graph.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns an integer, which is the volume of the community.
    """
    return sum([G.degree[node] for node in community_partition[community_id]])


def get_all_community_volumes(G, community_partition: List[List[int]]) -> List[int]:
    """
    This method will compute the volume of each community in the graph.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns a list of integers, where each integer is the volume of the community.
    """

    community_volumes = []
    for i in range(len(community_partition)):
        community_volumes.append(
            get_specific_community_volume(G, community_partition, i)
            )

    return community_volumes


def get_own_community_volume(G, node_id: int, community_partition: List[List[int]]) -> int:
    """
    This method will compute the volume of the community the node is in.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns an integer, which is the volume of the community.
    """
    for i, community in enumerate(community_partition):
        if node_id in community:
            return get_specific_community_volume(G, community_partition, i)
