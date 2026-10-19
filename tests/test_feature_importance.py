"""Tests for FeatureImportance: held-out protocol and the unsupervised ordering."""

import networkx as nx
import pytest

from NEExT.framework import NEExT
from NEExT.ml_models import FeatureImportance, MLModels

FEATURES = ["page_rank", "degree_centrality", "closeness_centrality", "clustering_coefficient"]


def _collection_and_features(n_graphs=24):
    nxt = NEExT(log_level="ERROR")
    graphs = []
    for k in range(n_graphs):
        if k % 2 == 0:
            g = nx.erdos_renyi_graph(20, 0.2, seed=k)
        else:
            g = nx.barabasi_albert_graph(20, 2, seed=k)
        g.graph["label"] = k % 2
        graphs.append(g)
    collection = nxt.load_from_networkx(graphs)
    features = nxt.compute_node_features(collection, feature_list=FEATURES, show_progress=False)
    return collection, features


@pytest.mark.parametrize("algorithm", ["supervised_greedy", "supervised_fast"])
def test_holdout_graphs_never_reach_supervised_selection(monkeypatch, algorithm):
    collection, features = _collection_and_features()
    seen_graph_ids = set()
    original_init = MLModels.__init__

    def recording_init(self, graph_collection, embeddings, *args, **kwargs):
        seen_graph_ids.update(graph.graph_id for graph in graph_collection.graphs)
        seen_graph_ids.update(embeddings.embeddings_df["graph_id"].tolist())
        original_init(self, graph_collection, embeddings, *args, **kwargs)

    monkeypatch.setattr(MLModels, "__init__", recording_init)

    importance = FeatureImportance(
        graph_collection=collection,
        features=features,
        algorithm=algorithm,
        n_iterations=2,
        n_jobs=1,
        parallel_backend="thread",
        holdout_size=0.25,
    )
    results = importance.compute()

    assert len(importance.heldout_graph_ids) == 6
    assert seen_graph_ids
    assert seen_graph_ids.isdisjoint(importance.heldout_graph_ids)
    assert seen_graph_ids == set(importance.train_graph_ids)
    assert len(results["heldout_performance"]) == len(features.feature_columns)
    assert results["heldout_performance"].between(0.0, 1.0).all()


def test_without_holdout_keeps_previous_output():
    collection, features = _collection_and_features()
    results = FeatureImportance(
        graph_collection=collection,
        features=features,
        algorithm="supervised_fast",
        n_iterations=2,
        n_jobs=1,
        parallel_backend="thread",
    ).compute()

    assert "heldout_performance" not in results.columns
    assert sorted(results["feature_name"]) == sorted(features.feature_columns)


def test_unsupervised_order_is_deterministic_and_complete():
    collection, features = _collection_and_features()

    def run(seed):
        return FeatureImportance(
            graph_collection=collection,
            features=features,
            algorithm="unsupervised",
            random_state=seed,
            unsupervised_n_pairs=100,
        ).compute()

    first = run(7)
    second = run(7)

    assert first["feature_name"].tolist() == second["feature_name"].tolist()
    assert sorted(first["feature_name"]) == sorted(features.feature_columns)
    assert first["max_abs_correlation"].iloc[0] == 0.0
    assert first["variance"].iloc[0] == first["variance"].max()


def test_node_sampling_is_reproducible_with_seed():
    collection, _ = _collection_and_features()
    collection.node_sample_rate = 0.5

    collection.sample_nodes(random_seed=3)
    first = [list(graph.sampled_nodes) for graph in collection.graphs]
    collection.sample_nodes(random_seed=3)
    second = [list(graph.sampled_nodes) for graph in collection.graphs]

    assert first == second
