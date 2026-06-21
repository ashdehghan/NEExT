"""Tests for the pure-PyTorch GNN graph embeddings (NEExT/embeddings/gnn_embeddings.py).

These exercise the core embedding contract independent of the Workbench. They
skip cleanly when torch is not installed.
"""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

from NEExT.embeddings import GNNEmbeddings  # noqa: E402
from NEExT.framework import NEExT  # noqa: E402

ARCHITECTURES = ["GCN", "GraphSAGE", "GIN"]


def _tiny_collection_and_features(graph_type="networkx", n_graphs=5, seed_offset=0):
    """Build a small graph collection plus node features via the public API."""
    nxt = NEExT(log_level="ERROR")
    graphs = []
    for k in range(n_graphs):
        if k % 2 == 0:
            g = nx.erdos_renyi_graph(7 + k, 0.45, seed=k + seed_offset)
        else:
            g = nx.barabasi_albert_graph(7 + k, 2, seed=k + seed_offset)
        g.graph["label"] = k % 2
        graphs.append(g)
    collection = nxt.load_from_networkx(graphs, graph_type=graph_type)
    features = nxt.compute_node_features(collection, feature_list=["page_rank", "degree_centrality"])
    return nxt, collection, features


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_gnn_embeddings_contract_per_architecture(architecture):
    """Each architecture returns the standard Embeddings contract with finite values."""
    _, collection, features = _tiny_collection_and_features()
    dim = 4

    embeddings = GNNEmbeddings(
        graph_collection=collection,
        features=features,
        architecture=architecture,
        embedding_dimension=dim,
        random_state=42,
    ).compute()

    df = embeddings.embeddings_df
    expected_cols = ["graph_id"] + [f"emb_{i}" for i in range(dim)]
    assert list(df.columns) == expected_cols
    assert len(df) == len(collection.graphs)
    assert embeddings.embedding_columns == [f"emb_{i}" for i in range(dim)]
    assert embeddings.embedding_name == f"gnn_{architecture.lower()}"
    values = df[embeddings.embedding_columns].to_numpy()
    assert np.isfinite(values).all()


def test_gnn_embeddings_via_framework_api():
    """Core-API parity: gnn is selectable through compute_graph_embeddings."""
    nxt, collection, features = _tiny_collection_and_features()
    embeddings = nxt.compute_graph_embeddings(
        collection,
        features,
        embedding_algorithm="gnn",
        architecture="GCN",
        embedding_dimension=3,
    )
    assert list(embeddings.embeddings_df.columns) == ["graph_id", "emb_0", "emb_1", "emb_2"]
    assert len(embeddings.embeddings_df) == len(collection.graphs)


def test_gnn_embeddings_igraph_backend():
    """Backend parity: adjacency built from graph.nodes/edges works for igraph too."""
    _, collection, features = _tiny_collection_and_features(graph_type="igraph")
    embeddings = GNNEmbeddings(
        graph_collection=collection,
        features=features,
        architecture="GraphSAGE",
        embedding_dimension=4,
    ).compute()
    df = embeddings.embeddings_df
    assert len(df) == len(collection.graphs)
    assert np.isfinite(df[embeddings.embedding_columns].to_numpy()).all()


def test_gnn_embeddings_dimension_can_exceed_feature_count():
    """A GNN learns an arbitrary output dim independent of the input feature count."""
    _, collection, features = _tiny_collection_and_features()
    n_features = len(features.feature_columns)
    dim = n_features + 4  # deliberately larger than the input feature dimension
    embeddings = GNNEmbeddings(
        graph_collection=collection,
        features=features,
        architecture="GIN",
        embedding_dimension=dim,
    ).compute()
    assert embeddings.embeddings_df.shape[1] == 1 + dim
    assert np.isfinite(embeddings.embeddings_df[embeddings.embedding_columns].to_numpy()).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_gnn_embeddings_single_graph_with_isolated_node(architecture):
    """Single graph containing an isolated node must not produce NaNs/Infs."""
    nxt = NEExT(log_level="ERROR")
    g = nx.path_graph(5)
    g.add_node(99)  # isolated node (no edges)
    g.graph["label"] = 0
    # Keep the isolated node: skip largest-component filtering.
    collection = nxt.load_from_networkx([g], filter_largest_component=False)
    features = nxt.compute_node_features(collection, feature_list=["page_rank", "degree_centrality"])

    embeddings = GNNEmbeddings(
        graph_collection=collection,
        features=features,
        architecture=architecture,
        embedding_dimension=3,
    ).compute()
    df = embeddings.embeddings_df
    assert len(df) == 1
    assert np.isfinite(df[embeddings.embedding_columns].to_numpy()).all()


def test_gnn_embeddings_rejects_unknown_architecture():
    _, collection, features = _tiny_collection_and_features(n_graphs=2)
    with pytest.raises(ValueError):
        GNNEmbeddings(
            graph_collection=collection,
            features=features,
            architecture="NotARealGNN",
            embedding_dimension=2,
        )
