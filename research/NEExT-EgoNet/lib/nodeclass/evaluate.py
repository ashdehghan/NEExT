"""Node-level evaluation: repeated stratified hold-out, multi-class metrics.

Reuses `containment.evaluate.shared_splits` — the one split protocol — so
pairing across representations is structural: the canonical node table fixes
y and its row order, identical for every method.

Primary metrics: accuracy + macro-F1 (all datasets) + macro-OVR-AUC; binary
datasets additionally get ROC-AUC and PR-AUC (average precision), the
headline for the imbalanced outlier tasks.

Contract mirrors containment: {metrics_rows, node_predictions, status}, so
`containment.runio.write_run` persists it unchanged (node_predictions rides
in the bag_predictions slot).
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from ..containment.evaluate import DEFAULT_SPLITS, DEFAULT_TEST_SIZE, shared_splits


def _merge_matrix(rep_df: pd.DataFrame, node_table: pd.DataFrame):
    """Join a node_id-keyed representation onto the canonical node table.

    Hard-fails on partial coverage or NaNs — silent row loss here would
    de-pair the shared splits, which is exactly the class of bug the
    containment audit existed to catch.
    """
    merged = node_table[["node_id", "y"]].merge(rep_df, on="node_id", how="left", validate="one_to_one")
    feature_cols = [c for c in merged.columns if c not in ("node_id", "y")]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    if len(merged) != len(node_table):
        raise AssertionError("Representation lost node rows in the merge")
    if np.isnan(X).any():
        missing = merged.loc[np.isnan(X).any(axis=1), "node_id"].tolist()[:5]
        raise AssertionError(f"Representation has NaNs / missing nodes (first: {missing})")
    return X, merged["y"].to_numpy(), merged["node_id"].to_numpy(), feature_cols


def evaluate_node_representation(
    rep_name: str,
    rep_df: pd.DataFrame,
    node_table: pd.DataFrame,
    n_splits: int = DEFAULT_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 42,
) -> dict:
    """Evaluate one node representation. Returns {metrics_rows, node_predictions, status}."""
    X, y, node_ids, feature_cols = _merge_matrix(rep_df, node_table)
    classes = np.unique(y)
    n_classes = len(classes)

    base = {
        "representation": rep_name,
        "n_nodes": len(y),
        "n_classes": n_classes,
        "n_features": len(feature_cols),
        "majority_share": round(float(np.bincount(y).max() / len(y)), 4),
    }
    if n_classes < 2:
        return {
            "metrics_rows": [{**base, "split": "", "status": "degenerate"}],
            "node_predictions": pd.DataFrame(),
            "status": "degenerate",
        }

    binary = n_classes == 2
    eval_metric = "logloss" if binary else "mlogloss"
    # XGBoost requires contiguous 0..C-1 labels; rare-class filtering keeps the
    # original codes, which may have gaps. Train encoded, report decoded.
    y_enc = np.searchsorted(classes, y)
    rows, pred_frames = [], []
    t0 = time.time()
    for i, idx_train, idx_test in shared_splits(y, n_splits=n_splits, test_size=test_size, seed=seed):
        model = XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric=eval_metric, random_state=seed, n_jobs=-1
        )
        model.fit(X[idx_train], y_enc[idx_train])
        proba = model.predict_proba(X[idx_test])
        y_test = y[idx_test]
        y_pred = classes[np.argmax(proba, axis=1)]

        if binary:
            # metrics on encoded {0,1} labels: proba[:, 1] is P(classes[1])
            auc_ovr = roc_auc_score(y_enc[idx_test], proba[:, 1])
        else:
            auc_ovr = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
        row = {
            **base,
            "split": i,
            "status": "ok",
            "accuracy": round(float(np.mean(y_pred == y_test)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "auc_ovr_macro": round(float(auc_ovr), 4),
        }
        if binary:
            row["roc_auc"] = round(float(roc_auc_score(y_enc[idx_test], proba[:, 1])), 4)
            row["pr_auc"] = round(float(average_precision_score(y_enc[idx_test], proba[:, 1])), 4)
        rows.append(row)

        pred = pd.DataFrame(
            {
                "node_id": node_ids[idx_test],
                "representation": rep_name,
                "split": i,
                "y_true": y_test,
                "y_pred": y_pred,
                "p_true": np.round(proba[np.arange(len(y_test)), np.searchsorted(classes, y_test)], 5),
            }
        )
        for j, c in enumerate(classes):
            pred[f"p_{c}"] = np.round(proba[:, j], 5)
        pred_frames.append(pred)

    elapsed = time.time() - t0
    for row in rows:
        row["eval_seconds"] = round(elapsed, 2)

    return {
        "metrics_rows": rows,
        "node_predictions": pd.concat(pred_frames, ignore_index=True),
        "status": "ok",
    }


def majority_floor(
    node_table: pd.DataFrame,
    n_splits: int = DEFAULT_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 42,
) -> dict:
    """Train-majority constant predictor under the same shared splits."""
    y = node_table["y"].to_numpy()
    base = {
        "representation": "majority",
        "n_nodes": len(y),
        "n_classes": int(node_table["y"].nunique()),
        "n_features": 0,
        "majority_share": round(float(np.bincount(y).max() / len(y)), 4),
    }
    rows = []
    for i, idx_train, idx_test in shared_splits(y, n_splits=n_splits, test_size=test_size, seed=seed):
        majority = np.bincount(y[idx_train]).argmax()
        y_test = y[idx_test]
        y_pred = np.full_like(y_test, majority)
        rows.append(
            {
                **base,
                "split": i,
                "status": "ok",
                "accuracy": round(float(np.mean(y_pred == y_test)), 4),
                "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                "auc_ovr_macro": 0.5,
            }
        )
    return {"metrics_rows": rows, "node_predictions": pd.DataFrame(), "status": "ok"}


def permutation_floor(
    rep_name: str,
    rep_df: pd.DataFrame,
    node_table: pd.DataFrame,
    n_splits: int = DEFAULT_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 42,
) -> dict:
    """Label-permutation floor: the full pipeline with the label association severed.

    Trains the same classifier on the same representation under the same shared
    splits, but with TRAINING labels shuffled (per-split seed), then scores
    against the true test labels. Unlike `majority_floor`, this floor inherits
    any leakage or overfitting artifact of the pipeline itself — accuracy above
    chance here means the harness is broken, not that the representation works.
    """
    X, y, _, feature_cols = _merge_matrix(rep_df, node_table)
    classes = np.unique(y)
    n_classes = len(classes)
    base = {
        "representation": rep_name,
        "n_nodes": len(y),
        "n_classes": n_classes,
        "n_features": len(feature_cols),
        "majority_share": round(float(np.bincount(y).max() / len(y)), 4),
    }
    if n_classes < 2:
        return {
            "metrics_rows": [{**base, "split": "", "status": "degenerate"}],
            "node_predictions": pd.DataFrame(),
            "status": "degenerate",
        }

    binary = n_classes == 2
    eval_metric = "logloss" if binary else "mlogloss"
    y_enc = np.searchsorted(classes, y)
    rows = []
    t0 = time.time()
    for i, idx_train, idx_test in shared_splits(y, n_splits=n_splits, test_size=test_size, seed=seed):
        y_train_perm = np.random.RandomState(seed + i).permutation(y_enc[idx_train])
        model = XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric=eval_metric, random_state=seed, n_jobs=-1
        )
        model.fit(X[idx_train], y_train_perm)
        proba = model.predict_proba(X[idx_test])
        y_test = y[idx_test]
        y_pred = classes[np.argmax(proba, axis=1)]
        if binary:
            auc_ovr = roc_auc_score(y_enc[idx_test], proba[:, 1])
        else:
            auc_ovr = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
        rows.append(
            {
                **base,
                "split": i,
                "status": "ok",
                "accuracy": round(float(np.mean(y_pred == y_test)), 4),
                "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                "auc_ovr_macro": round(float(auc_ovr), 4),
            }
        )
    elapsed = time.time() - t0
    for row in rows:
        row["eval_seconds"] = round(elapsed, 2)
    return {"metrics_rows": rows, "node_predictions": pd.DataFrame(), "status": "ok"}


def summarize_node_metrics(metrics_rows: list) -> pd.DataFrame:
    """Aggregate per-split rows to mean±std per representation."""
    df = pd.DataFrame([r for r in metrics_rows if r.get("status") == "ok"])
    if df.empty:
        return pd.DataFrame()
    spec = {
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "f1_macro_mean": ("f1_macro", "mean"),
        "f1_macro_std": ("f1_macro", "std"),
        "auc_ovr_macro_mean": ("auc_ovr_macro", "mean"),
        "n_nodes": ("n_nodes", "first"),
        "n_features": ("n_features", "first"),
    }
    if "pr_auc" in df.columns:
        spec["roc_auc_mean"] = ("roc_auc", "mean")
        spec["pr_auc_mean"] = ("pr_auc", "mean")
    return df.groupby("representation").agg(**spec).round(4).reset_index()
