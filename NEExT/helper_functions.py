import networkx as nx
from typing import List, Set


def divide_chunks(list, chunks):
    for i in range(0, len(list), chunks):
        yield list[i:i + chunks]


def get_numb_of_nb_x_hops_away(G, node, max_hop_length) -> List[int]:
    """
    This method will compute the number of neighbors x hops away from
    a given node.

    Returns a list of integers, where each integer is the number of nodes at that hop length.
    """
    seen = set()
    seen.add(node)
    current_set = {node}
    hop_list = []
    for _ in range(max_hop_length):
        boundary = nx.node_boundary(G, current_set)
        current_set = boundary - seen
        hop_list.append(len(current_set))
        seen.update(boundary)

    return [len(n) for n in get_nodes_x_hops_away(G, node, max_hop_length)]


def get_nodes_x_hops_away(G, node, max_hop_length) -> List[Set]:
    """
    This method will get the of neighbors of 1 to max_hop_length hops away from
    a given node.

    Returns a list of sets, where each set is the nodes at a given hop length.
    """
    seen = set()
    seen.add(node)
    current_set = {node}
    hop_list = []
    for _ in range(max_hop_length):
        boundary = nx.node_boundary(G, current_set)
        current_set = boundary - seen
        hop_list.append(current_set)
        seen.update(boundary)

    return hop_list


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
