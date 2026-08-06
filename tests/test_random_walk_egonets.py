"""Tests for random-walk egonet construction (compute_random_walk_egonets).

Covers: determinism and self-contained randomness, membership/weight
invariants, restart locality, size controls, re-entrancy, backend parity,
weighted incidence-matrix behavior, provenance, and an end-to-end pipeline.
"""

import random

import numpy as np
import pandas as pd

from NEExT import NEExT
from NEExT.collections import EgonetCollection


def two_communities_df(n_per_side: int = 12):
    """Two dense cliques joined by one bridge edge — walks should stay home."""
    left = list(range(n_per_side))
    right = list(range(n_per_side, 2 * n_per_side))
    edges = [(u, v) for i, u in enumerate(left) for v in left[i + 1 :]]
    edges += [(u, v) for i, u in enumerate(right) for v in right[i + 1 :]]
    edges.append((left[-1], right[0]))
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": left + right, "lbl": [0] * n_per_side + [1] * n_per_side})
    return edges_df, nodes_df


def load_collection(graph_type: str = "networkx"):
    edges_df, nodes_df = two_communities_df()
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    return nxt, nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type)


def build(nxt, collection, **kwargs):
    defaults = {"walk_length": 8, "n_walks": 50, "restart_prob": 0.15, "egonet_feature_target": "lbl", "random_seed": 7}
    defaults.update(kwargs)
    return nxt.compute_random_walk_egonets(collection, **defaults)


def assert_collection_is_single_generation(egonets: EgonetCollection, node_count: int):
    assert len(egonets.graphs) == node_count
    assert sorted(g.graph_id for g in egonets.graphs) == list(range(node_count))
    assert len(egonets.graph_id_node_array) == sum(len(g.nodes) for g in egonets.graphs)
    assert set(egonets.egonet_to_graph_node_mapping) == {g.graph_id for g in egonets.graphs}


def test_deterministic_given_seed():
    nxt, collection = load_collection()
    a = build(nxt, collection, random_seed=7)
    b = build(nxt, load_collection()[1], random_seed=7)
    for ga, gb in zip(a.graphs, b.graphs):
        assert sorted(ga.node_mapping) == sorted(gb.node_mapping)
        assert ga.node_weights == gb.node_weights


def test_does_not_touch_global_random_state():
    random.seed(123)
    expected_py = random.random()
    np.random.seed(123)
    expected_np = np.random.random()

    random.seed(123)
    np.random.seed(123)
    nxt, collection = load_collection()
    build(nxt, collection)
    assert random.random() == expected_py
    assert np.random.random() == expected_np


def test_center_always_member_and_weights_normalized():
    nxt, collection = load_collection()
    egonets = build(nxt, collection, min_visits=5)
    for egonet in egonets.graphs:
        _, center = egonets.egonet_to_graph_node_mapping[egonet.graph_id]
        assert center in egonet.node_mapping
        weights = egonet.node_weights
        assert weights is not None
        assert set(weights) == set(egonet.node_mapping.values())
        assert all(w > 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # The center accumulates walk starts + restarts: it should carry the
        # largest mass with restart_prob well above zero.
        assert weights[egonet.node_mapping[center]] == max(weights.values())


def test_restart_locality():
    """Higher restart probability keeps walks nearer the center."""
    nxt, collection = load_collection()
    tight = build(nxt, collection, restart_prob=0.5)
    loose = build(nxt, load_collection()[1], restart_prob=0.0)
    tight_mean = np.mean([len(g.nodes) for g in tight.graphs])
    loose_mean = np.mean([len(g.nodes) for g in loose.graphs])
    assert tight_mean <= loose_mean


def test_walks_respect_community_boundary():
    """From deep inside one clique, the far community is barely visited."""
    nxt, collection = load_collection()
    egonets = build(nxt, collection, n_walks=100, min_visits=2)
    labels = {v: collection.graphs[0].node_attributes[v]["lbl"] for v in collection.graphs[0].nodes}
    for egonet in egonets.graphs:
        _, center = egonets.egonet_to_graph_node_mapping[egonet.graph_id]
        members = list(egonet.node_mapping)
        same_side = sum(1 for m in members if labels[m] == labels[center])
        assert same_side / len(members) > 0.5


def test_min_visits_and_max_size():
    nxt, collection = load_collection()
    capped = build(nxt, collection, max_egonet_size=5)
    for egonet in capped.graphs:
        assert len(egonet.nodes) <= 5
        _, center = capped.egonet_to_graph_node_mapping[egonet.graph_id]
        assert center in egonet.node_mapping

    thresholded = build(nxt, load_collection()[1], min_visits=10_000)
    for egonet in thresholded.graphs:
        _, center = thresholded.egonet_to_graph_node_mapping[egonet.graph_id]
        assert list(egonet.node_mapping) == [center]


def test_weight_by_visits_false_gives_uniform_membership():
    nxt, collection = load_collection()
    egonets = build(nxt, collection, weight_by_visits=False)
    assert all(egonet.node_weights is None for egonet in egonets.graphs)


def test_repeat_call_rebuilds_collection():
    nxt, collection = load_collection()
    egonets = EgonetCollection(graph_type=collection.graph_type, egonet_feature_target="lbl", skip_features=["lbl"])
    egonets.compute_random_walk_egonets(collection, walk_length=5, n_walks=20, random_seed=7)
    egonets.compute_random_walk_egonets(collection, walk_length=8, n_walks=20, random_seed=7)
    assert_collection_is_single_generation(egonets, node_count=24)


def test_backend_parity_given_seed():
    nxt_nx, coll_nx = load_collection("networkx")
    nxt_ig, coll_ig = load_collection("igraph")
    a = build(nxt_nx, coll_nx)
    b = build(nxt_ig, coll_ig)
    assert a.graph_type == "networkx" and b.graph_type == "igraph"
    for ga, gb in zip(a.graphs, b.graphs):
        assert sorted(ga.node_mapping) == sorted(gb.node_mapping)
        assert ga.node_weights == gb.node_weights


def test_validation_rejects_bad_params():
    nxt, collection = load_collection()
    for kwargs in (
        {"walk_length": 0},
        {"n_walks": 0},
        {"restart_prob": 1.0},
        {"restart_prob": -0.1},
        {"min_visits": 0},
        {"max_egonet_size": 0},
    ):
        try:
            build(nxt, collection, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_weight_remapping_survives_reindex():
    nxt, collection = load_collection()
    egonets = build(nxt, collection)
    egonet = egonets.graphs[0]
    reindexed = egonet.reindex_nodes()
    assert reindexed.node_weights is not None
    assert abs(sum(reindexed.node_weights.values()) - 1.0) < 1e-9
    assert set(reindexed.node_weights) == set(reindexed.node_mapping.values())


def test_provenance_fields_populated():
    nxt, collection = load_collection()
    egonets = build(nxt, collection)
    for egonet in egonets.graphs:
        assert egonet.node_mapping
        assert egonet.original_graph_id == collection.graphs[0].graph_id
        assert egonet.original_node_id in egonet.node_mapping


def test_end_to_end_pipeline_with_weighted_embeddings():
    nxt, collection = load_collection()
    egonets = build(nxt, collection, min_visits=2)
    features = nxt.compute_node_features(
        egonets, feature_list=["degree_centrality", "clustering_coefficient"], feature_vector_length=2, show_progress=False
    )
    embeddings = nxt.compute_graph_embeddings(egonets, features, embedding_algorithm="approx_wasserstein", embedding_dimension=4)
    assert len(embeddings.embeddings_df) == len(egonets.graphs)
    assert embeddings.embeddings_df.notna().all().all()


def test_incidence_matrix_uses_weights_only_when_present():
    from NEExT.embeddings.graph_embeddings import GraphEmbeddings

    nxt, collection = load_collection()
    walk = build(nxt, collection)
    khop = nxt.compute_k_hop_egonets(collection, k_hop=1, egonet_feature_target="lbl", random_seed=7)

    for egonets, expect_weighted in ((walk, True), (khop, False)):
        features = nxt.compute_node_features(egonets, feature_list=["degree_centrality"], feature_vector_length=1, show_progress=False)
        ge = GraphEmbeddings(graph_collection=egonets, features=features, embedding_algorithm="approx_wasserstein", embedding_dimension=2)
        incidence, _, _ = ge._prepare_incidence_matrix(features.features_df)
        values = incidence.data
        if expect_weighted:
            assert not np.allclose(values, 1.0)
            row_sums = np.asarray(incidence.sum(axis=1)).ravel()
            assert np.allclose(row_sums, 1.0)
        else:
            assert np.allclose(values, 1.0)
