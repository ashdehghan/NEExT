"""Per-bag vector representations for the containment classifier.

Every function returns a DataFrame with a `graph_id` column (the bag id) and
feature columns. All *fair-fight* representations (wasserstein, pooled_*,
size_only) see ONLY the egonet subgraph; the node_oracle is a full-graph-cost
quality reference and is labeled as such wherever it is reported.

The structural Features object computed on the EgonetCollection is shared:
compute it once, feed it to both `wasserstein_embedding` and
`pooled_features`.
"""

import numpy as np
import pandas as pd

POOL_STATS = ("mean", "max", "p90")


def wasserstein_embedding(nxt, egonets, features, dimension: int = 16, seed: int = 42) -> pd.DataFrame:
    """NEExT approx_wasserstein graph embedding of each bag (the PoC pipeline)."""
    embeddings = nxt.compute_graph_embeddings(
        egonets,
        features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=dimension,
        random_state=seed,
    )
    return embeddings.embeddings_df.copy()


def pooled_features(features, stats=POOL_STATS) -> pd.DataFrame:
    """Statistic-pooled member features per bag: mean / max / 90th percentile.

    The max pool is the tail-sensitive baseline — an anomaly is a tail event,
    and a distributional embedding may average it away while a max sees it.
    """
    fdf = features.features_df
    cols = features.feature_columns
    grouped = fdf.groupby("graph_id")[cols]
    parts = []
    for stat in stats:
        if stat == "mean":
            part = grouped.mean()
        elif stat == "max":
            part = grouped.max()
        elif stat.startswith("p"):
            part = grouped.quantile(int(stat[1:]) / 100.0)
        else:
            raise ValueError(f"Unknown pool stat: {stat}")
        parts.append(part.add_suffix(f"_{stat}"))
    return pd.concat(parts, axis=1).reset_index()


def size_only(bag_table: pd.DataFrame) -> pd.DataFrame:
    """The confound baseline: nothing but bag size. Every claim must beat this."""
    return bag_table[["graph_id", "n_nodes", "n_edges"]].copy()


def node_oracle_scores(
    nxt,
    source_collection,
    bags,
    feature_list=("all",),
    feature_vector_length: int = 3,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Full-graph-cost quality reference (NOT a fair fight).

    Computes structural features for EVERY node of the source graph, trains a
    node-level classifier on the labels of *training-bag centers only* (the
    caller's split discipline: bags inherit their center's split), and scores
    each bag as the max member score. Returned as a single-column
    representation `oracle_score` keyed by bag graph_id, plus the fitted
    node-score lookup for reuse.
    """
    from xgboost import XGBClassifier

    src_features = nxt.compute_node_features(
        source_collection,
        feature_list=list(feature_list),
        feature_vector_length=feature_vector_length,
        show_progress=False,
        n_jobs=n_jobs,
    )
    fdf = src_features.features_df.set_index("node_id")
    cols = src_features.feature_columns

    egonets = bags.egonets
    source_graphs = {g.graph_id: g for g in source_collection.graphs}

    def bag_members(egonet):
        return list(egonet.node_mapping.keys())

    # Node-level training set: centers only (each node is a center at most once).
    label_col_rows = []
    for _, row in bags.table.iterrows():
        label_col_rows.append({"node_id": row["center_node"], "y": row["y_center"], "graph_id": row["graph_id"]})
    centers = pd.DataFrame(label_col_rows)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, eval_metric="logloss", random_state=seed, n_jobs=max(n_jobs, 1) if n_jobs != -1 else -1
    )
    model.fit(fdf.loc[centers["node_id"], cols].values, centers["y"].values)
    node_scores = pd.Series(model.predict_proba(fdf[cols].values)[:, 1], index=fdf.index)

    rows = []
    for egonet in egonets.graphs:
        members = bag_members(egonet)
        rows.append({"graph_id": egonet.graph_id, "oracle_score": float(node_scores.loc[members].max())})
    return pd.DataFrame(rows)


def representation_matrix(name: str, df: pd.DataFrame) -> tuple:
    """(X, graph_ids, feature_names) from a representation DataFrame."""
    cols = [c for c in df.columns if c != "graph_id"]
    return df[cols].to_numpy(dtype=np.float64), df["graph_id"].to_numpy(), cols
