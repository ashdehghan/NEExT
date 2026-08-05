"""Evaluation-harness invariants, including the oracle-leakage regression test."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.containment import build_bags, evaluate_node_oracle, shared_splits
from NEExT import NEExT


def test_shared_splits_identical_partitions():
    y = np.array([0, 1] * 40)
    a = [(i, tuple(tr), tuple(te)) for i, tr, te in shared_splits(y, n_splits=5)]
    b = [(i, tuple(tr), tuple(te)) for i, tr, te in shared_splits(y, n_splits=5)]
    assert a == b
    # stratification holds per split
    for _, tr, te in shared_splits(y, n_splits=3):
        assert 0 < y[list(te)].mean() < 1


def _random_label_collection(nxt, n=200, n_pos=30, seed=5):
    """ER graph with labels assigned at random — structurally unlearnable."""
    rng = np.random.default_rng(seed)
    import networkx as nx

    G = nx.gnm_random_graph(n, n * 4, seed=seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_pos, replace=False)] = 1
    edges = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    nodes = pd.DataFrame({"node_id": range(n), "is_anomaly": labels})
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type="igraph")


def test_oracle_does_not_leak_labels():
    """Regression for the audit-found leak: with RANDOM labels there is no
    structural signal, so a leak-free oracle must score near chance. The old
    implementation (trained on all centers, including test bags) memorized
    labels and scored far above chance on exactly this setup."""
    nxt = NEExT(log_level="ERROR")
    collection = _random_label_collection(nxt)
    bags = build_bags(nxt, collection, label_column="is_anomaly", k_hop=1)
    result = evaluate_node_oracle(nxt, collection, bags, n_splits=5)
    aucs = [r["roc_auc"] for r in result["metrics_rows"] if r["status"] == "ok"]
    assert aucs, "expected evaluable splits"
    assert float(np.mean(aucs)) < 0.65, f"oracle scored {np.mean(aucs):.3f} on random labels - leakage"


def test_oracle_degenerate_training_reported():
    """All-positive centers in training -> per-split degenerate status, not a crash."""
    nxt = NEExT(log_level="ERROR")
    import networkx as nx

    G = nx.path_graph(40)
    edges = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    nodes = pd.DataFrame({"node_id": range(40), "is_anomaly": [1] * 25 + [0] * 15})
    collection = nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type="igraph")
    bags = build_bags(nxt, collection, label_column="is_anomaly", k_hop=1)
    result = evaluate_node_oracle(nxt, collection, bags, n_splits=3)
    statuses = {r["status"] for r in result["metrics_rows"]}
    assert statuses <= {"ok", "degenerate_oracle_training", "degenerate"}
