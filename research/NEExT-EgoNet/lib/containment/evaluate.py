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
# Below this many minority-class bags a cell is saturation-degenerate: the
# stratified splitter can't even guarantee both classes in every split, and
# rank metrics on a handful of minority test bags are noise, not signal.
MIN_MINORITY_BAGS = 10


def shared_splits(y: np.ndarray, n_splits: int = DEFAULT_SPLITS, test_size: float = DEFAULT_TEST_SIZE, seed: int = 42):
    """The one split protocol every evaluation path must use.

    Yields (split_index, idx_train, idx_test) with seeds `seed+i`, stratified
    on y. Pairing across representations (and the node oracle) is structural:
    identical y in identical row order => identical partitions.
    """
    for i in range(n_splits):
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed + i, shuffle=True, stratify=y
        )
        yield i, idx_train, idx_test


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

    minority = int(np.bincount(y, minlength=2).min())
    if len(np.unique(y)) < 2 or minority < MIN_MINORITY_BAGS:
        return {
            "metrics_rows": [{**base, "split": "", "status": "degenerate", "roc_auc": "", "pr_auc": "", "accuracy": ""}],
            "bag_scores": pd.DataFrame(),
            "status": "degenerate",
        }

    rows = []
    score_sum = pd.Series(0.0, index=graph_ids)
    score_count = pd.Series(0, index=graph_ids)
    t0 = time.time()
    for i, idx_train, idx_test in shared_splits(y, n_splits=n_splits, test_size=test_size, seed=seed):
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


def evaluate_node_oracle(
    nxt,
    source_collection,
    bags,
    feature_list=("all",),
    feature_vector_length: int = 3,
    n_jobs: int = -1,
    n_splits: int = DEFAULT_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 42,
) -> dict:
    """Full-graph-cost quality reference, evaluated WITHOUT label leakage.

    Structural features for every source-graph node are computed once (they
    are label-free and shared across splits). Then, per shared split: an
    XGBoost node classifier is trained on the y_center labels of TRAINING-bag
    centers only, every node is scored, and each TEST bag is scored as the
    max score over its members EXCLUDING training centers — their labels were
    revealed during training, and "detecting" a bag through a member whose
    label the model memorized is recall of knowns, not detection (the
    regression test constructs exactly this failure with random labels).
    Test bags whose members are all training centers are excluded from that
    split's oracle metrics (count reported as n_excluded_bags).

    Splits whose training centers are single-class are reported with status
    `degenerate_oracle_training` — the honest outcome in saturated cells,
    where almost every center is positive.
    """
    src_features = nxt.compute_node_features(
        source_collection,
        feature_list=list(feature_list),
        feature_vector_length=feature_vector_length,
        show_progress=False,
        n_jobs=n_jobs,
    )
    fdf = src_features.features_df.set_index("node_id")
    cols = src_features.feature_columns

    table = bags.table.sort_values("graph_id").reset_index(drop=True)
    y = table["y_contains"].to_numpy()
    y_center = table["y_center"].to_numpy()
    centers = table["center_node"].to_numpy()
    graph_ids = table["graph_id"].to_numpy()
    members_of = {egonet.graph_id: list(egonet.node_mapping.keys()) for egonet in bags.egonets.graphs}

    base = {
        "representation": "node_oracle",
        "n_bags": len(y),
        "positive_rate": round(float(y.mean()), 4),
        "n_features": len(cols),
    }
    minority = int(np.bincount(y, minlength=2).min())
    if len(np.unique(y)) < 2 or minority < MIN_MINORITY_BAGS:
        return {
            "metrics_rows": [{**base, "split": "", "status": "degenerate", "roc_auc": "", "pr_auc": "", "accuracy": ""}],
            "bag_scores": pd.DataFrame(),
            "status": "degenerate",
        }

    rows = []
    score_sum = pd.Series(0.0, index=graph_ids)
    score_count = pd.Series(0, index=graph_ids)
    t0 = time.time()
    for i, idx_train, idx_test in shared_splits(y, n_splits=n_splits, test_size=test_size, seed=seed):
        train_center_labels = y_center[idx_train]
        if len(np.unique(train_center_labels)) < 2:
            rows.append(
                {**base, "split": i, "status": "degenerate_oracle_training", "roc_auc": "", "pr_auc": "", "accuracy": ""}
            )
            continue
        model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", random_state=seed, n_jobs=-1)
        model.fit(fdf.loc[centers[idx_train], cols].values, train_center_labels)
        node_scores = pd.Series(model.predict_proba(fdf[cols].values)[:, 1], index=fdf.index)

        known = set(centers[idx_train].tolist())
        kept_ids, scores, y_test = [], [], []
        for j in idx_test:
            unknown = [m for m in members_of[graph_ids[j]] if m not in known]
            if not unknown:
                continue
            kept_ids.append(graph_ids[j])
            scores.append(float(node_scores.loc[unknown].max()))
            y_test.append(y[j])
        scores, y_test = np.array(scores), np.array(y_test)
        if len(np.unique(y_test)) < 2:
            rows.append(
                {**base, "split": i, "status": "degenerate_oracle_training", "roc_auc": "", "pr_auc": "", "accuracy": ""}
            )
            continue
        rows.append(
            {
                **base,
                "split": i,
                "status": "ok",
                "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
                "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
                "accuracy": round(float(np.mean((scores >= 0.5) == y_test)), 4),
                "n_excluded_bags": len(idx_test) - len(kept_ids),
            }
        )
        score_sum.loc[kept_ids] += scores
        score_count.loc[kept_ids] += 1

    elapsed = time.time() - t0
    for row in rows:
        row["eval_seconds"] = round(elapsed, 2)

    seen = score_count > 0
    bag_scores = pd.DataFrame(
        {
            "graph_id": score_sum.index[seen],
            "representation": "node_oracle",
            "mean_test_score": (score_sum[seen] / score_count[seen]).round(5).to_numpy(),
            "n_test_appearances": score_count[seen].to_numpy(),
        }
    )
    ok = any(r["status"] == "ok" for r in rows)
    return {"metrics_rows": rows, "bag_scores": bag_scores, "status": "ok" if ok else "degenerate_oracle_training"}


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
