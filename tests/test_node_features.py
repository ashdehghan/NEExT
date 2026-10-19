"""Tests for NodeFeatures (native node attributes as features).

Covers: alignment of rows with node ids when the node table is not in node order,
the normalize_features flag, "all", missing attributes and non-numeric columns,
label exclusion by the caller, combination with structural features, and the
patch (egonet) path where attributes are inserted in parent-dict order.
"""

import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.features import NodeFeatures


def attributed_graph(shuffle=True, n=12):
    """Path 0-1-...-n-1 with attributes a = node id, b = 10 - node id, and a label; node table unsorted."""
    edges_df = pd.DataFrame({"src_node_id": list(range(n - 1)), "dest_node_id": list(range(1, n))})
    nodes_df = pd.DataFrame(
        {"node_id": list(range(n)), "a": [float(i) for i in range(n)], "b": [10.0 - i for i in range(n)], "lbl": [i % 2 for i in range(n)]}
    )
    if shuffle:
        nodes_df = nodes_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    return edges_df, nodes_df


def load(edges_df, nodes_df, graph_type="igraph"):
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    return nxt, nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type)


@pytest.mark.parametrize("graph_type", ["igraph", "networkx"])
def test_rows_align_with_node_ids_when_table_is_unsorted(graph_type):
    nxt, collection = load(*attributed_graph(shuffle=True), graph_type=graph_type)
    features = NodeFeatures(collection, feature_list=["a", "b"], normalize_features=False).compute()
    df = features.features_df
    assert list(df.columns) == ["node_id", "graph_id", "a", "b"]
    assert df["node_id"].tolist() == list(range(12))
    np.testing.assert_allclose(df["a"], np.arange(12, dtype=float))
    np.testing.assert_allclose(df["b"], 10.0 - np.arange(12))


def test_normalize_flag_is_honoured():
    nxt, collection = load(*attributed_graph())
    raw = NodeFeatures(collection, feature_list=["a", "b"], normalize_features=False).compute().features_df
    scaled = NodeFeatures(collection, feature_list=["a", "b"], normalize_features=True).compute().features_df
    assert raw["a"].std() > 1.5
    assert abs(scaled["a"].mean()) < 1e-9 and abs(scaled["a"].std(ddof=0) - 1.0) < 1e-9
    assert abs(scaled["b"].mean()) < 1e-9


def test_all_selects_every_attribute_including_label_so_caller_must_exclude_it():
    nxt, collection = load(*attributed_graph())
    features = NodeFeatures(collection, feature_list=["all"], normalize_features=False).compute()
    assert features.feature_columns == ["a", "b", "lbl"]
    # the pattern data loaders must use: everything except node_id and the label
    label = "lbl"
    feature_list = [c for c in features.feature_columns if c != label]
    selected = NodeFeatures(collection, feature_list=feature_list, normalize_features=False).compute()
    assert selected.feature_columns == ["a", "b"]


def test_missing_attribute_column_raises():
    nxt, collection = load(*attributed_graph())
    with pytest.raises(ValueError, match="missing"):
        NodeFeatures(collection, feature_list=["a", "zzz"]).compute()


def test_node_without_attributes_raises():
    edges_df, nodes_df = attributed_graph(shuffle=False)
    nxt, collection = load(edges_df, nodes_df)
    graph = collection.graphs[0]
    del graph.node_attributes[4]
    with pytest.raises(ValueError, match="no attributes"):
        NodeFeatures(collection, feature_list=["a"]).compute()


def test_non_numeric_attribute_raises():
    edges_df, nodes_df = attributed_graph()
    nodes_df["name"] = ["w" + str(i) for i in nodes_df["node_id"]]
    nxt, collection = load(edges_df, nodes_df)
    with pytest.raises(ValueError, match="not numeric"):
        NodeFeatures(collection, feature_list=["a", "name"]).compute()


def test_combines_with_structural_features_without_nans():
    nxt, collection = load(*attributed_graph())
    native = NodeFeatures(collection, feature_list=["a", "b"]).compute()
    structural = nxt.compute_node_features(collection, feature_list=["degree_centrality"], feature_vector_length=2, show_progress=False)
    combined = structural + native
    assert len(combined.features_df) == 12
    assert combined.feature_columns == ["degree_centrality_0", "degree_centrality_1", "a", "b"]
    assert combined.features_df.notna().all().all()


def test_patch_path_is_aligned_and_label_free():
    """Patches store attributes in parent-dict order; rows must still line up with internal node ids."""
    nxt, collection = load(*attributed_graph(shuffle=True, n=20))
    patches = nxt.compute_leiden_patches(collection, egonet_feature_target="lbl", min_patch_size=1, random_seed=1)
    features = NodeFeatures(patches, feature_list=["all"], normalize_features=False).compute()
    assert features.feature_columns == ["a", "b"]  # lbl was stripped from patch attributes
    df = features.features_df
    assert len(df) == 20
    for patch in patches.graphs:
        rows = df[df["graph_id"] == patch.graph_id].set_index("node_id")
        for orig, internal in patch.node_mapping.items():
            assert rows.loc[internal, "a"] == float(orig)
            assert rows.loc[internal, "b"] == 10.0 - orig
    embeddings = nxt.compute_graph_embeddings(patches, features, embedding_algorithm="approx_wasserstein", embedding_dimension=2)
    assert len(embeddings.embeddings_df) == len(patches.graphs)
