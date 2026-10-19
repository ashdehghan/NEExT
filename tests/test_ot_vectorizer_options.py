"""Exact LOT and Sinkhorn: the ground metric and the per-graph point cap are NEExT options, passed to vectorizers."""

import networkx as nx
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.embeddings import GraphEmbeddings


def _bags():
    G = nx.connected_caveman_graph(5, 6)
    edges_df = pd.DataFrame(list(G.edges()), columns=["src_node_id", "dest_node_id"])
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(edges_df=edges_df)
    egonets = nxt.compute_k_hop_egonets(collection, k_hop=2)
    features = nxt.compute_node_features(egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=2, show_progress=False)
    return egonets, features


class _Recorder:
    """Stands in for a vectorizer class and records the keyword arguments it is built with."""

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self

    def fit_transform(self, incidence, vectors):
        import numpy as np

        return np.zeros((incidence.shape[0], 2))


@pytest.mark.parametrize("algorithm, vectorizer", [("wasserstein", "WassersteinVectorizer"), ("sinkhornvectorizer", "SinkhornVectorizer")])
def test_metric_and_cap_reach_the_vectorizer(monkeypatch, algorithm, vectorizer):
    egonets, features = _bags()
    calls = []
    embedder = GraphEmbeddings(egonets, features, embedding_algorithm=algorithm, embedding_dimension=2, metric="cosine", max_distribution_size=7)
    monkeypatch.setattr(embedder.vectorizers, vectorizer, _Recorder(calls))
    embedder.compute()
    assert calls[0]["metric"] == "cosine"
    if algorithm == "wasserstein":
        assert calls[0]["max_distribution_size"] == 7


def test_defaults_are_euclidean_and_uncapped(monkeypatch):
    egonets, features = _bags()
    calls = []
    embedder = GraphEmbeddings(egonets, features, embedding_algorithm="wasserstein", embedding_dimension=2)
    monkeypatch.setattr(embedder.vectorizers, "WassersteinVectorizer", _Recorder(calls))
    embedder.compute()
    largest_bag = max(len(g.nodes) for g in egonets.graphs)
    assert calls[0]["metric"] == "euclidean"
    assert calls[0]["max_distribution_size"] == largest_bag


def test_exact_lot_runs_end_to_end():
    egonets, features = _bags()
    nxt = NEExT(log_level="WARNING")
    embeddings = nxt.compute_graph_embeddings(egonets, features, embedding_algorithm="wasserstein", embedding_dimension=3)
    assert embeddings.embeddings_df.shape == (len(egonets.graphs), 4)
