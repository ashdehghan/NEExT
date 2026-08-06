"""Node-table + evaluation invariants for lib.nodeclass.

Split identity is THE property the benchmark rests on: every representation
evaluated on the same canonical node table must see byte-identical
partitions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.containment.evaluate import shared_splits
from lib.nodeclass import (
    build_node_table,
    evaluate_node_representation,
    filter_rare_classes,
    majority_floor,
    permutation_floor,
    sample_centers,
)
from NEExT import NEExT


def _labeled_collection(nxt, n=60, n_classes=3, seed=7):
    rng = np.random.RandomState(seed)
    edges = pd.DataFrame({"src_node_id": range(n - 1), "dest_node_id": range(1, n)})  # path: connected
    extra = pd.DataFrame({"src_node_id": rng.randint(0, n, 40), "dest_node_id": rng.randint(0, n, 40)})
    edges = pd.concat([edges, extra[extra["src_node_id"] != extra["dest_node_id"]]], ignore_index=True)
    nodes = pd.DataFrame({"node_id": range(n), "label": rng.randint(0, n_classes, n)})
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes)


def _random_rep(node_ids, dim=4, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.normal(size=(len(node_ids), dim)), columns=[f"r_{j}" for j in range(dim)])
    df.insert(0, "node_id", node_ids)
    return df


def test_node_table_and_rare_class_filter():
    nxt = NEExT(log_level="WARNING")
    collection = _labeled_collection(nxt)
    table, report = build_node_table(collection, "label")
    assert list(table.columns) == ["node_id", "y", "y_raw"]
    assert report["n_classes"] == 3
    assert sorted(table["y"].unique()) == [0, 1, 2]

    # Make class 2 rare, filter, codes must survive un-renumbered
    rare = table[table["y"] != 2].head(55)
    rare = pd.concat([rare, table[table["y"] == 2].head(3)], ignore_index=True)
    kept, freport = filter_rare_classes(rare, min_count=10)
    assert freport["n_dropped_classes"] == 1
    assert freport["n_dropped_nodes"] == 3
    assert set(kept["y"]) <= {0, 1}


def test_sample_centers_stratified_and_canonical_order():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    sampled = sample_centers(table, n_centers=30, seed=13)
    assert len(sampled) <= 30
    assert list(sampled["node_id"]) == sorted(sampled["node_id"])
    # deterministic
    again = sample_centers(table, n_centers=30, seed=13)
    pd.testing.assert_frame_equal(sampled, again)


def test_split_identity_across_representations():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    y = table["y"].to_numpy()
    splits_a = [(i, tr.copy(), te.copy()) for i, tr, te in shared_splits(y, n_splits=3)]
    splits_b = [(i, tr.copy(), te.copy()) for i, tr, te in shared_splits(y, n_splits=3)]
    for (_, tr_a, te_a), (_, tr_b, te_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


def test_evaluate_multiclass_metrics_and_predictions():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    rep = _random_rep(table["node_id"])
    out = evaluate_node_representation("random", rep, table, n_splits=3)
    assert out["status"] == "ok"
    rows = out["metrics_rows"]
    assert len(rows) == 3
    for col in ("accuracy", "f1_macro", "auc_ovr_macro"):
        assert all(0.0 <= r[col] <= 1.0 for r in rows)
    assert "roc_auc" not in rows[0]  # multiclass: no binary metrics
    preds = out["node_predictions"]
    assert {"node_id", "split", "y_true", "y_pred", "p_true", "p_0", "p_1", "p_2"} <= set(preds.columns)


def test_evaluate_binary_adds_rank_metrics():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt, n_classes=2), "label")
    out = evaluate_node_representation("random", _random_rep(table["node_id"]), table, n_splits=3)
    assert all(("roc_auc" in r and "pr_auc" in r) for r in out["metrics_rows"])


def test_evaluate_handles_noncontiguous_codes():
    """After rare-class filtering y codes may have gaps; XGBoost needs 0..C-1."""
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    gapped = table[table["y"] != 1].reset_index(drop=True)  # codes {0, 2}
    out = evaluate_node_representation("random", _random_rep(gapped["node_id"]), gapped, n_splits=3)
    assert out["status"] == "ok"
    assert set(out["node_predictions"]["y_pred"].unique()) <= {0, 2}


def test_merge_coverage_tripwire():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    partial = _random_rep(table["node_id"][:-5])  # missing 5 nodes
    with pytest.raises(AssertionError):
        evaluate_node_representation("partial", partial, table, n_splits=3)


def test_degenerate_single_class():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    one = table[table["y"] == 0].reset_index(drop=True)
    out = evaluate_node_representation("random", _random_rep(one["node_id"]), one, n_splits=3)
    assert out["status"] == "degenerate"


def test_permutation_floor_severs_signal():
    """A label-leaking representation scores high normally, ~chance under permutation."""
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    leaky = _random_rep(table["node_id"], dim=1, seed=0)
    leaky["r_0"] = table["y"].to_numpy() + 0.01 * leaky["r_0"]

    honest = evaluate_node_representation("leaky", leaky, table, n_splits=3)
    permuted = permutation_floor("permuted", leaky, table, n_splits=3)
    assert permuted["status"] == "ok"
    rows = permuted["metrics_rows"]
    assert len(rows) == 3
    assert set(rows[0]) == set(honest["metrics_rows"][0])  # same row schema
    honest_acc = np.mean([r["accuracy"] for r in honest["metrics_rows"]])
    permuted_acc = np.mean([r["accuracy"] for r in rows])
    assert honest_acc > 0.9
    assert permuted_acc < 0.6


def test_majority_floor_matches_share():
    nxt = NEExT(log_level="WARNING")
    table, _ = build_node_table(_labeled_collection(nxt), "label")
    out = majority_floor(table, n_splits=3)
    rows = out["metrics_rows"]
    assert len(rows) == 3
    share = rows[0]["majority_share"]
    assert all(abs(r["accuracy"] - share) < 0.15 for r in rows)
