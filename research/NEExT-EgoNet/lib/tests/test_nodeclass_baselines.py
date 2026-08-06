"""karateclub adapter + structural-baseline invariants for lib.nodeclass.baselines."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.nodeclass import center_structural_features, degree_only, kc_embed, to_networkx
from NEExT import NEExT


def _collection(nxt, graph_type="networkx"):
    rng = np.random.RandomState(3)
    n = 40
    edges = pd.DataFrame({"src_node_id": range(n - 1), "dest_node_id": range(1, n)})
    extra = pd.DataFrame({"src_node_id": rng.randint(0, n, 30), "dest_node_id": rng.randint(0, n, 30)})
    edges = pd.concat([edges, extra[extra["src_node_id"] != extra["dest_node_id"]]], ignore_index=True)
    nodes = pd.DataFrame({"node_id": range(n), "label": rng.randint(0, 2, n)})
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type=graph_type)


def test_to_networkx_invariants():
    nxt = NEExT(log_level="WARNING")
    G = to_networkx(_collection(nxt))
    assert sorted(G.nodes) == list(range(40))
    assert not G.is_directed()
    assert all(u != v for u, v in G.edges)


def test_kc_embed_deepwalk_shape_and_determinism():
    pytest.importorskip("karateclub")
    nxt = NEExT(log_level="WARNING")
    G = to_networkx(_collection(nxt))
    df1, record = kc_embed("deepwalk", G, seed=42)
    df2, _ = kc_embed("deepwalk", G, seed=42)
    assert len(df1) == 40 and record["n_features"] == 16
    assert list(df1["node_id"]) == list(range(40))
    pd.testing.assert_frame_equal(df1, df2)


def test_center_structural_features_node_keyed():
    nxt = NEExT(log_level="WARNING")
    collection = _collection(nxt)
    df = center_structural_features(nxt, collection, feature_vector_length=2, n_jobs=1)
    assert "node_id" in df.columns and len(df) == 40
    assert df.drop(columns="node_id").notna().all().all()


def test_degree_only_matches_graph():
    nxt = NEExT(log_level="WARNING")
    collection = _collection(nxt, graph_type="igraph")
    df = degree_only(collection)
    assert list(df.columns) == ["node_id", "degree", "core_number"]
    g = collection.graphs[0].G
    assert df["degree"].tolist() == list(g.degree())
