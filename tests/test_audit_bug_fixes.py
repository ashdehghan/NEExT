"""Regression tests for the bugs surfaced by the 2026-08 performance audit."""

import random

import pandas as pd

from NEExT import NEExT


def triangle_plus_isolated_nodes_df():
    """A 3-node triangle plus one isolated node (node 3)."""
    edges = pd.DataFrame({"src_node_id": [0, 1, 2], "dest_node_id": [1, 2, 0]})
    nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "lbl": [0, 1, 0, 1]})
    return edges, nodes


def load_collection(graph_type: str, filter_largest_component: bool = False):
    edges, nodes = triangle_plus_isolated_nodes_df()
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges,
        nodes_df=nodes,
        graph_type=graph_type,
        filter_largest_component=filter_largest_component,
        reindex_nodes=False,
    )
    return nxt, collection


# ---------------------------------------------------------------------------
# BUG-1: load_centrality on igraph crashed with KeyError when the graph has
# isolated vertices (nx.Graph(edgelist) drops them in the conversion fallback).
# ---------------------------------------------------------------------------


def test_load_centrality_igraph_isolated_node_no_crash():
    nxt, collection = load_collection("igraph")

    features = nxt.compute_node_features(collection, feature_list=["load_centrality"], feature_vector_length=2, show_progress=False)

    assert set(features.features_df["node_id"]) == {0, 1, 2, 3}


def test_load_centrality_backend_parity_with_isolated_node():
    nxt_ig, collection_ig = load_collection("igraph")
    nxt_nx, collection_nx = load_collection("networkx")

    features_ig = nxt_ig.compute_node_features(
        collection_ig, feature_list=["load_centrality"], feature_vector_length=2, show_progress=False, normalize_features=False
    )
    features_nx = nxt_nx.compute_node_features(
        collection_nx, feature_list=["load_centrality"], feature_vector_length=2, show_progress=False, normalize_features=False
    )

    merged_ig = features_ig.features_df.sort_values("node_id").reset_index(drop=True)
    merged_nx = features_nx.features_df.sort_values("node_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged_ig, merged_nx)


# ---------------------------------------------------------------------------
# BUG-3: node sampling reseeded and consumed Python's global random module on
# every collection load (call-order-dependent results, silent global reseed).
# Sampling now uses self-contained random.Random instances.
# ---------------------------------------------------------------------------


def _load_with_rate(rate: float):
    edges, nodes = triangle_plus_isolated_nodes_df()
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type="networkx", node_sample_rate=rate)


def test_loading_does_not_touch_global_random_state():
    for rate in (1.0, 0.5):
        random.seed(999)
        expected_draw = random.random()
        random.seed(999)
        _load_with_rate(rate)
        assert random.random() == expected_draw, f"global random stream perturbed at rate {rate}"


def test_sampling_reproducible_regardless_of_global_random_usage():
    collection_a = _load_with_rate(0.5)
    collection_a.sample_nodes(random_seed=42)
    samples_a = [list(g.sampled_nodes) for g in collection_a.graphs]

    random.seed(0)
    random.random()  # interleaved global-random consumption must not matter

    collection_b = _load_with_rate(0.5)
    collection_b.sample_nodes(random_seed=42)
    samples_b = [list(g.sampled_nodes) for g in collection_b.graphs]

    assert samples_a == samples_b


def test_full_rate_sampling_returns_all_nodes():
    collection = _load_with_rate(1.0)
    for graph in collection.graphs:
        assert graph.sampled_nodes == graph.nodes


# ---------------------------------------------------------------------------
# BUG-2: framework egonet entry points dropped the source backend, leaving
# EgonetCollection.graph_type at its "networkx" default even for igraph
# sources (the Leiden path is igraph-only, so it was always mislabeled).
# ---------------------------------------------------------------------------


def test_k_hop_egonet_collection_reports_source_graph_type():
    for graph_type in ("igraph", "networkx"):
        nxt, collection = load_collection(graph_type, filter_largest_component=True)
        egonets = nxt.compute_k_hop_egonets(collection, k_hop=1, egonet_feature_target="lbl")
        assert egonets.graph_type == graph_type
        assert egonets.describe()["graph_type"] == graph_type


def test_leiden_egonet_collection_reports_igraph():
    nxt, collection = load_collection("igraph", filter_largest_component=True)
    egonets = nxt.compute_leiden_egonets(collection, egonet_feature_target="lbl")
    assert egonets.graph_type == "igraph"


def test_load_centrality_backend_parity_connected_graph():
    """Guard the no-regression claim: connected graphs match across backends."""
    edges = pd.DataFrame({"src_node_id": [0, 1, 2, 3], "dest_node_id": [1, 2, 3, 0]})
    nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "lbl": [0, 1, 0, 1]})

    frames = {}
    for graph_type in ("igraph", "networkx"):
        nxt = NEExT()
        nxt.set_log_level("WARNING")
        collection = nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type=graph_type)
        features = nxt.compute_node_features(
            collection, feature_list=["load_centrality"], feature_vector_length=2, show_progress=False, normalize_features=False
        )
        frames[graph_type] = features.features_df.sort_values("node_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(frames["igraph"], frames["networkx"])
