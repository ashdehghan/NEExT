"""Region-holdout splits: separation property, determinism, retention accounting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.nodeclass.separated_splits import adjacency_from_edges, multi_source_distances, region_holdout_splits


def ring_graph(n=200):
    src = list(range(n))
    dst = [(i + 1) % n for i in range(n)]
    return adjacency_from_edges(n, src, dst)


def node_table(n=200):
    rng = np.random.RandomState(0)
    return pd.DataFrame({"node_id": np.arange(n), "y": rng.randint(0, 3, n)})


def test_multi_source_distances_ring():
    adj = ring_graph(10)
    d = multi_source_distances(adj, [0])
    assert d[0] == 0 and d[5] == 5 and d[9] == 1


def test_separation_property():
    adj = ring_graph()
    table = node_table()
    for B in (0, 2, 5):
        for name, idx_train, idx_test, stats in region_holdout_splits(adj, table, buffer_hops=B, n_repeats=3):
            test_ids = table["node_id"].to_numpy()[idx_test]
            dist = multi_source_distances(adj, test_ids.tolist())
            train_ids = table["node_id"].to_numpy()[idx_train]
            assert (dist[train_ids] > B).all(), f"{name}: train center within buffer {B}"
            assert len(np.intersect1d(idx_train, idx_test)) == 0


def test_determinism():
    adj = ring_graph()
    table = node_table()
    a = [(n, t.tolist(), s.tolist()) for n, t, s, _ in region_holdout_splits(adj, table, buffer_hops=2, n_repeats=2, seed=7)]
    b = [(n, t.tolist(), s.tolist()) for n, t, s, _ in region_holdout_splits(adj, table, buffer_hops=2, n_repeats=2, seed=7)]
    assert a == b


def test_retention_accounting():
    adj = ring_graph()
    table = node_table()
    for _, idx_train, idx_test, stats in region_holdout_splits(adj, table, buffer_hops=3, n_repeats=2):
        assert stats["n_test"] == len(idx_test)
        assert stats["n_train"] == len(idx_train)
        assert stats["n_test"] + stats["n_train"] + stats["n_dropped_buffer"] == len(table)


def test_larger_buffer_smaller_train():
    adj = ring_graph()
    table = node_table()
    trains = {}
    for B in (0, 4, 10):
        sizes = [len(t) for _, t, _, _ in region_holdout_splits(adj, table, buffer_hops=B, n_repeats=3, seed=3)]
        trains[B] = np.mean(sizes)
    assert trains[0] >= trains[4] >= trains[10]
    assert trains[10] < trains[0]


def test_test_frac_respected():
    adj = ring_graph()
    table = node_table()
    for _, _, idx_test, _ in region_holdout_splits(adj, table, buffer_hops=0, test_frac=0.25, n_repeats=2):
        assert abs(len(idx_test) - 50) <= 2
