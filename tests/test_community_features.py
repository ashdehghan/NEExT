import sys
sys.path.append("../")
from NEExT import feature_engine
from NEExT.graph_object import Graph_Object

# from _test_format import run_list_of_tests
import networkx as nx


def test_compute_community_aware_features():
    G = nx.cycle_graph(6)
    g_obj = Graph_Object()
    g_obj.graph = G
    g_obj.graph_id = 0
    g_obj.graph_node_source = "all"

    feature_engine.compute_community_aware_features(g_obj, 3, 'anomaly_score_CADA')
    feature_engine.compute_community_aware_features(g_obj, 3, 'normalized_anomaly_score_CADA')
    feature_engine.compute_community_aware_features(g_obj, 3, 'participation_coefficient')
    feature_engine.compute_community_aware_features(g_obj, 3, 'normalized_within_module_degree')
    feature_engine.compute_community_aware_features(g_obj, 3, 'participation_coefficient')

    return True


if __name__ == "__main__":
    test_compute_community_aware_features()
