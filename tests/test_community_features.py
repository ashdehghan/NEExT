import sys
sys.path.append("../")
from NEExT.feature_engine import Feature_Engine
from NEExT.graph_object import Graph_Object

# from _test_format import run_list_of_tests
import networkx as nx

feat_eng = Feature_Engine(None)


def test_compute_community_aware_features():
    G = nx.cycle_graph(6)
    g_obj = Graph_Object()
    g_obj.graph = G
    g_obj.graph_id = 0
    g_obj.graph_node_source = "all"

    feat_eng.compute_community_aware_features(g_obj, 3, 'anomaly_score_CADA')
    feat_eng.compute_community_aware_features(g_obj, 3, 'normalized_anomaly_score_CADA')
    print(g_obj.feature_collection['features'])


if __name__ == "__main__":
    test_compute_community_aware_features()
