import networkx as nx

import sys
sys.path.append("../")
from NEExT.helper_functions import get_nodes_x_hops_away, get_numb_of_nb_x_hops_away, \
    get_specific_in_community_degree, get_all_in_community_degrees, \
    get_own_in_community_degree, get_specific_community_volume, \
    get_all_community_volumes, get_own_community_volume
from _test_format import run_list_of_tests


def test_get_nodes_x_hops_away():
    G = nx.cycle_graph(6)
    return get_nodes_x_hops_away(G, 0, 3) == [{1, 5}, {2, 4}, {3}]


def test_get_numb_of_nb_x_hops_away():
    G = nx.cycle_graph(6)
    return get_numb_of_nb_x_hops_away(G, 0, 3) == [2, 2, 1]


def test_in_degree_calculators():
    G = nx.cycle_graph(6)
    partition = [[0, 1, 2], [3, 4, 5]]
    assert get_specific_in_community_degree(G, 0, partition, 0) == 1, "Error in get_specific_in_community_degree"
    assert get_specific_in_community_degree(G, 0, partition, 1) == 1, "Error in get_specific_in_community_degree"
    assert get_own_in_community_degree(G, 0, partition) == 1, "Error in get_own_in_community_degree"
    assert get_all_in_community_degrees(G, 0, partition) == [1, 1], "Error in get_all_in_community_degrees"

    assert get_specific_in_community_degree(G, 1, partition, 0) == 2, "Error in get_specific_in_community_degree"
    assert get_specific_in_community_degree(G, 1, partition, 1) == 0, "Error in get_specific_in_community_degree"
    assert get_own_in_community_degree(G, 1, partition) == 2, "Error in get_own_in_community_degree"
    assert get_all_in_community_degrees(G, 1, partition) == [2, 0], "Error in get_all_in_community_degrees"


def test_community_volume_calculators():
    G = nx.cycle_graph(6)
    partition = [[0, 1, 2], [3, 4, 5]]
    assert get_specific_community_volume(G, partition, 0) == 6, "Error in get_specific_community_volume"
    assert get_specific_community_volume(G, partition, 1) == 6, "Error in get_specific_community_volume"
    assert get_own_community_volume(G, 0, partition) == 6, "Error in get_own_community_volume"
    assert get_all_community_volumes(G, partition) == [6, 6], "Error in get_all_community_volumes"


if __name__ == "__main__":
    run_list_of_tests([test_get_nodes_x_hops_away, test_get_numb_of_nb_x_hops_away,
                       test_in_degree_calculators, test_community_volume_calculators])
