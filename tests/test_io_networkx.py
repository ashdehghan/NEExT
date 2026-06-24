"""Offline tests for loading NetworkX graphs into NEExT.

Migrated from the former root-level ``test_networkx.py`` demo script into the
pytest suite. Uses only NetworkX's built-in deterministic generators, so it runs
fully offline with no network access.
"""

import networkx as nx
import numpy as np

from NEExT import NEExT
from NEExT.collections import GraphCollection


def test_load_from_networkx_basic():
    """Load a list of NetworkX graphs and verify the resulting collection."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    g1 = nx.karate_club_graph()  # 34 nodes, connected
    g1.graph["label"] = 0
    g2 = nx.complete_graph(10)  # 10 nodes, connected
    g2.graph["label"] = 1

    collection = nxt.load_from_networkx(
        nx_graphs=[g1, g2],
        reindex_nodes=True,
        filter_largest_component=False,
    )

    assert len(collection.graphs) == 2
    assert collection.get_total_node_count() == g1.number_of_nodes() + g2.number_of_nodes()
    assert sorted(g.graph_label for g in collection.graphs) == [0, 1]


def test_compute_features_on_networkx_loaded_graphs():
    """Structural node features compute on NetworkX-loaded graphs."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    g1 = nx.karate_club_graph()
    g1.graph["label"] = 0
    g2 = nx.cycle_graph(8)
    g2.graph["label"] = 1

    collection = nxt.load_from_networkx(
        nx_graphs=[g1, g2],
        reindex_nodes=True,
        filter_largest_component=False,
    )

    features = nxt.compute_node_features(
        graph_collection=collection,
        feature_list=["degree_centrality", "clustering_coefficient"],
        feature_vector_length=2,
        show_progress=False,
    )

    assert len(features.feature_columns) > 0
    # One feature row per node across both graphs.
    assert features.features_df.shape[0] == collection.get_total_node_count()


def test_mixed_networkx_and_dict_loading():
    """A GraphCollection accepts NetworkX graphs mixed with dict-format graphs."""
    g = nx.cycle_graph(8)
    g.graph["label"] = 1

    dict_graph = {
        "graph_id": 100,
        "graph_label": 0,
        "nodes": list(range(10)),
        "edges": [(i, (i + 1) % 10) for i in range(10)],
        "node_attributes": {},
        "edge_attributes": {},
    }

    collection = GraphCollection(graph_type="networkx")
    collection.add_graphs([g, dict_graph], reindex_nodes=True)

    assert len(collection.graphs) == 2
