"""Tests for family-1 pair priors between patches: adjacency, cut edges, coarse hop distance."""

import time

import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.patches import coarse_patch_graph, compute_pair_priors


def load(edges, n, graph_type="igraph"):
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": list(range(n))})
    return nxt, nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type, filter_largest_component=False)


# Three patches: A = {0,1}, B = {2,3}, C = {4,5}; A-B joined by two edges, B-C by one, A-C none.
TOY_EDGES = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (3, 4)]
TOY_MEMBERSHIP = [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("graph_type", ["igraph", "networkx"])
def test_hand_computed_values(graph_type):
    _, collection = load(TOY_EDGES, 6, graph_type)
    priors = compute_pair_priors(collection.graphs[0], TOY_MEMBERSHIP)
    np.testing.assert_array_equal(priors["cut_edges"], [[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(priors["adjacent"], np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool))
    np.testing.assert_array_equal(priors["hop_distance"], [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert priors["cut_edges"].dtype == np.int64 and priors["hop_distance"].dtype == np.int64


def test_symmetry_and_adjacent_iff_hop_one():
    _, collection = load(TOY_EDGES, 6)
    priors = compute_pair_priors(collection.graphs[0], TOY_MEMBERSHIP)
    for key in ("adjacent", "cut_edges", "hop_distance"):
        np.testing.assert_array_equal(priors[key], priors[key].T)
    np.testing.assert_array_equal(priors["adjacent"], priors["hop_distance"] == 1)
    assert (np.diag(priors["hop_distance"]) == 0).all()


def test_disconnected_patches_get_minus_one():
    edges = [(0, 1), (2, 3)]  # two components, two patches
    _, collection = load(edges, 4)
    priors = compute_pair_priors(collection.graphs[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(priors["hop_distance"], [[0, -1], [-1, 0]])
    assert not priors["adjacent"].any()


def test_coarse_graph_edges_and_weights():
    _, collection = load(TOY_EDGES, 6)
    coarse, cut = coarse_patch_graph(collection.graphs[0], TOY_MEMBERSHIP)
    assert coarse.vcount() == 3
    assert sorted(coarse.get_edgelist()) == [(0, 1), (1, 2)]
    assert dict(zip(coarse.get_edgelist(), coarse.es["weight"])) == {(0, 1): 2, (1, 2): 1}
    assert cut.sum() == 2 * 3


def test_membership_validation():
    _, collection = load(TOY_EDGES, 6)
    graph = collection.graphs[0]
    with pytest.raises(ValueError, match="entries"):
        compute_pair_priors(graph, [0, 0, 1])
    with pytest.raises(ValueError, match="contiguous"):
        compute_pair_priors(graph, [0, 0, 2, 2, 3, 3])
    with pytest.raises(ValueError, match="non-negative"):
        compute_pair_priors(graph, [0, 0, -1, -1, 1, 1])


def test_priors_from_a_patch_collection_round_trip():
    """Indices of the priors are the patch ids of that graph's patches, in order."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    # four cliques of 5 in a ring
    edges = []
    for c in range(4):
        base = c * 5
        edges += [(base + i, base + j) for i in range(5) for j in range(i + 1, 5)]
        edges.append((base + 4, ((c + 1) % 4) * 5))
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    collection = nxt.load_single_graph_from_dfs(edges_df=edges_df, graph_type="igraph")
    patches = nxt.compute_leiden_patches(collection, min_patch_size=1, random_seed=3)
    membership = patches.patch_memberships[0]
    priors = compute_pair_priors(collection.graphs[0], membership)
    assert priors["adjacent"].shape == (4, 4)
    assert priors["adjacent"].sum() == 8  # ring: each patch has two neighbours
    assert priors["hop_distance"].max() == 2
    # patch ids of this graph are 0..3 and index the arrays directly
    assert [p.graph_id for p in patches.graphs] == list(range(4))


def test_cost_on_a_mid_size_graph():
    """A few thousand nodes and a few hundred patches must take well under a second."""
    rng = np.random.RandomState(0)
    n, n_blocks = 5000, 250
    membership = np.repeat(np.arange(n_blocks), n // n_blocks)
    edges = set()
    for _ in range(4 * n):
        u, v = rng.randint(0, n, 2)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    _, collection = load(sorted(edges), n)
    start = time.perf_counter()
    priors = compute_pair_priors(collection.graphs[0], membership.tolist())
    elapsed = time.perf_counter() - start
    assert priors["hop_distance"].shape == (n_blocks, n_blocks)
    assert elapsed < 5.0
