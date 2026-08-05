"""Bag-level evaluation: repeated stratified hold-out, rank metrics.

Containment is a (possibly heavily imbalanced) binary problem, so ROC-AUC and
PR-AUC (average precision) are the primary metrics; accuracy at 0.5 is kept
for continuity with the PoC. Splits are seeded identically across
representations so every method sees the same train/test partitions.

Degenerate cells (all bags positive — saturation — or all negative) are not
errors: they are the phenomenon. They come back as a row with status
"degenerate" and the label balance, and feed the saturation report.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DEFAULT_SPLITS = 10
DEFAULT_TEST_SIZE = 0.3


def evaluate_representation(
    rep_name: str,
    rep_df: pd.DataFrame,
    bag_table: pd.DataFrame,
    n_splits: int = DEFAULT_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 42,
) -> dict:
    """Evaluate one representation. Returns {metrics_rows, bag_scores, status}."""
    merged = rep_df.merge(bag_table[["graph_id", "y_contains"]], on="graph_id")
    feature_cols = [c for c in merged.columns if c not in ("graph_id", "y_contains")]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    y = merged["y_contains"].to_numpy()
    graph_ids = merged["graph_id"].to_numpy()

    positive_rate = float(y.mean())
    base = {
        "representation": rep_name,
        "n_bags": len(y),
        "positive_rate": round(positive_rate, 4),
        "n_features": len(feature_cols),
    }

    if len(np.unique(y)) < 2:
        return {
            "metrics_rows": [{**base, "split": "", "status": "degenerate", "roc_auc": "", "pr_auc": "", "accuracy": ""}],
            "bag_scores": pd.DataFrame(),
            "status": "degenerate",
        }

    rows = []
    score_sum = pd.Series(0.0, index=graph_ids)
    score_count = pd.Series(0, index=graph_ids)
    t0 = time.time()
    for i in range(n_splits):
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed + i, shuffle=True, stratify=y
        )
        model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", random_state=seed, n_jobs=-1)
        model.fit(X[idx_train], y[idx_train])
        scores = model.predict_proba(X[idx_test])[:, 1]
        y_test = y[idx_test]
        rows.append(
            {
                **base,
                "split": i,
                "status": "ok",
                "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
                "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
                "accuracy": round(float(np.mean((scores >= 0.5) == y_test)), 4),
            }
        )
        test_ids = graph_ids[idx_test]
        score_sum.loc[test_ids] += scores
        score_count.loc[test_ids] += 1

    elapsed = time.time() - t0
    for row in rows:
        row["eval_seconds"] = round(elapsed, 2)

    seen = score_count > 0
    bag_scores = pd.DataFrame(
        {
            "graph_id": score_sum.index[seen],
            "representation": rep_name,
            "mean_test_score": (score_sum[seen] / score_count[seen]).round(5).to_numpy(),
            "n_test_appearances": score_count[seen].to_numpy(),
        }
    )
    return {"metrics_rows": rows, "bag_scores": bag_scores, "status": "ok"}


def summarize_metrics(metrics_rows: list) -> pd.DataFrame:
    """Aggregate per-split rows to mean±std per representation."""
    df = pd.DataFrame([r for r in metrics_rows if r.get("status") == "ok"])
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("representation").agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        positive_rate=("positive_rate", "first"),
        n_bags=("n_bags", "first"),
    )
    return agg.round(4).reset_index()
