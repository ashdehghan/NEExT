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
