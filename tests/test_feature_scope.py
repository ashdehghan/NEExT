"""Tests for feature_scope="local"|"global" on compute_node_features.

"local" (default) computes features within each graph of the collection —
unchanged behavior. "global" computes features once on the source
GraphCollection an EgonetCollection was built from and projects each source
node's vector onto every egonet it is a member of.

Covers: default parity, per-member equality with a direct source computation,
local/global divergence, k_hop=0 semantics, end-to-end embeddings, error
paths, backend parity, random-walk-bag compatibility, column naming, and the
sampled-source coverage guard.
"""

import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.features.structural_node_features import StructuralNodeFeatures


def two_communities_df(n_per_side: int = 10):
    """Two dense cliques joined by one bridge edge."""
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


def build_khop(nxt, collection, k_hop: int = 1):
    return nxt.compute_k_hop_egonets(collection, k_hop=k_hop, egonet_feature_target="lbl", random_seed=7)


FEATURES = ["degree_centrality", "page_rank"]


def compute(nxt, collection, scope=None, **kwargs):
    defaults = {"feature_list": list(FEATURES), "feature_vector_length": 1, "show_progress": False}
    defaults.update(kwargs)
    if scope is not None:
        defaults["feature_scope"] = scope
    return nxt.compute_node_features(collection, **defaults)


def test_default_scope_is_local_and_unchanged():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    default = compute(nxt, egonets)
    explicit = compute(nxt, egonets, scope="local")
    direct = StructuralNodeFeatures(graph_collection=egonets, feature_list=list(FEATURES), feature_vector_length=1, show_progress=False).compute()
    pd.testing.assert_frame_equal(default.features_df, explicit.features_df)
    pd.testing.assert_frame_equal(default.features_df, direct.features_df)
    assert list(default.features_df.columns[:2]) == ["node_id", "graph_id"]


@pytest.mark.parametrize("normalize", [True, False])
def test_global_matches_direct_source_computation_per_member(normalize):
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    projected = compute(nxt, egonets, scope="global", normalize_features=normalize)
    source = compute(nxt, collection, normalize_features=normalize)

    src_lookup = source.features_df.set_index("node_id")
    checked = 0
    for egonet in egonets.graphs:
        rows = projected.features_df[projected.features_df["graph_id"] == egonet.graph_id]
        assert len(rows) == len(egonet.node_mapping)
        for orig, internal in egonet.node_mapping.items():
            got = rows.loc[rows["node_id"] == internal, projected.feature_columns].to_numpy()[0]
            expected = src_lookup.loc[orig, projected.feature_columns].to_numpy()
            assert np.allclose(got, expected.astype(float))
            checked += 1
    assert checked == sum(len(g.node_mapping) for g in egonets.graphs)


def test_local_and_global_genuinely_differ():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    local = compute(nxt, egonets, scope="local")
    global_ = compute(nxt, egonets, scope="global")
    assert local.features_df.shape == global_.features_df.shape
    assert not np.allclose(
        local.features_df[local.feature_columns].to_numpy(),
        global_.features_df[global_.feature_columns].to_numpy(),
    )


def test_khop_zero_singletons_carry_true_features_in_global_mode():
    nxt, collection = load_collection()
    singletons = build_khop(nxt, collection, k_hop=0)
    assert all(len(g.nodes) == 1 for g in singletons.graphs)

    global_ = compute(nxt, singletons, scope="global")
    source = compute(nxt, collection)
    src_lookup = source.features_df.set_index("node_id")
    for egonet in singletons.graphs:
        row = global_.features_df[global_.features_df["graph_id"] == egonet.graph_id]
        expected = src_lookup.loc[egonet.original_node_id, global_.feature_columns].to_numpy()
        assert np.allclose(row[global_.feature_columns].to_numpy()[0], expected.astype(float))

    # local mode on singleton bags: every bag looks identical (degenerate constants)
    local = compute(nxt, singletons, scope="local")
    for col in local.feature_columns:
        assert local.features_df[col].nunique() == 1


def test_global_end_to_end_embedding():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    features = compute(nxt, egonets, scope="global")
    embeddings = nxt.compute_graph_embeddings(egonets, features, embedding_algorithm="approx_wasserstein", embedding_dimension=2)
    assert len(embeddings.embeddings_df) == len(egonets.graphs)
    assert embeddings.embeddings_df.notna().all().all()


def test_global_on_plain_graph_collection_raises():
    nxt, collection = load_collection()
    with pytest.raises(ValueError, match="EgonetCollection"):
        compute(nxt, collection, scope="global")


def test_global_without_source_ref_raises():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    egonets.source_graph_collection = None
    with pytest.raises(ValueError, match="source_graph_collection"):
        compute(nxt, egonets, scope="global")


def test_invalid_scope_string_raises():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    with pytest.raises(ValueError, match="feature_scope"):
        compute(nxt, egonets, scope="globl")


def test_backend_parity_nx_vs_igraph_global():
    nxt_nx, coll_nx = load_collection("networkx")
    nxt_ig, coll_ig = load_collection("igraph")
    a = compute(nxt_nx, build_khop(nxt_nx, coll_nx), scope="global")
    b = compute(nxt_ig, build_khop(nxt_ig, coll_ig), scope="global")
    merged = a.features_df.merge(b.features_df, on=["node_id", "graph_id"], suffixes=("_nx", "_ig"))
    assert len(merged) == len(a.features_df)
    for col in a.feature_columns:
        assert np.allclose(merged[f"{col}_nx"], merged[f"{col}_ig"], atol=1e-6)


def test_random_walk_bags_keep_node_weights():
    from NEExT.embeddings.graph_embeddings import GraphEmbeddings

    nxt, collection = load_collection()
    bags = nxt.compute_random_walk_egonets(collection, walk_length=8, n_walks=50, egonet_feature_target="lbl", random_seed=7)
    features = compute(nxt, bags, scope="global")
    assert all(egonet.node_weights is not None for egonet in bags.graphs)

    ge = GraphEmbeddings(graph_collection=bags, features=features, embedding_algorithm="approx_wasserstein", embedding_dimension=2)
    incidence, _, _ = ge._prepare_incidence_matrix(features.features_df)
    row_sums = np.asarray(incidence.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0)


def test_vector_length_column_naming_preserved():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    features = compute(nxt, egonets, scope="global", feature_vector_length=3)
    expected = ["node_id", "graph_id"] + [f"{f}_{i}" for f in FEATURES for i in range(3)]
    assert list(features.features_df.columns) == expected
    assert features.feature_columns == expected[2:]


def test_global_hard_fails_on_sampled_source():
    nxt, collection = load_collection()
    egonets = build_khop(nxt, collection)
    source_graph = collection.graphs[0]
    source_graph.sampled_nodes = source_graph.nodes[: len(source_graph.nodes) // 2]
    try:
        with pytest.raises(ValueError, match="no feature row"):
            compute(nxt, egonets, scope="global")
    finally:
        source_graph.sampled_nodes = None
