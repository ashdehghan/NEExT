"""Per-bag vector representations for the containment classifier.

Every function returns a DataFrame with a `graph_id` column (the bag id) and
feature columns. All representations here are *fair fights*: they see ONLY
the egonet subgraph. The full-graph node oracle lives in
`evaluate.evaluate_node_oracle`, which owns its per-split training
discipline (it must never be evaluated as a precomputed representation —
that was the audit-found leak).

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


def representation_matrix(name: str, df: pd.DataFrame) -> tuple:
    """(X, graph_ids, feature_names) from a representation DataFrame."""
    cols = [c for c in df.columns if c != "graph_id"]
    return df[cols].to_numpy(dtype=np.float64), df["graph_id"].to_numpy(), cols
