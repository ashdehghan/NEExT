import sys
sys.path.append("../")
from NEExT.helper_functions import get_nodes_x_hops_away, get_numb_of_nb_x_hops_away

from _test_format import run_list_of_tests

import networkx as nx


def test_get_nodes_x_hops_away():
    G = nx.cycle_graph(6)
    return get_nodes_x_hops_away(G, 0, 3) == [{1, 5}, {2, 4}, {3}]


def test_get_numb_of_nb_x_hops_away():
    G = nx.cycle_graph(6)
    return get_numb_of_nb_x_hops_away(G, 0, 3) == [2, 2, 1]


if __name__ == "__main__":
    run_list_of_tests([test_get_nodes_x_hops_away, test_get_numb_of_nb_x_hops_away])
