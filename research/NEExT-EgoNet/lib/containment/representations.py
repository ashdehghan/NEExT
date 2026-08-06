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


def pooled_features(features, stats=POOL_STATS, egonets=None) -> pd.DataFrame:
    """Statistic-pooled member features per bag.

    Supported stats: "mean", "max", "min", and percentiles "pNN". Max/p90 are
    the top-tail baselines; min/p10 exist because bottom-extremum anomalies
    (e.g. dangling tails) are invisible to top-of-distribution pools — the
    2026-08-05 audit follow-up.

    When `egonets` is given and its graphs carry `node_weights` (random-walk
    bags), the mean becomes visit-weighted; extremum/percentile stats stay
    unweighted (an extremum is an extremum regardless of visit mass).
    """
    fdf = features.features_df
    cols = features.feature_columns
    weights = None
    if egonets is not None:
        weights_by_graph = {g.graph_id: g.node_weights for g in egonets.graphs if getattr(g, "node_weights", None)}
        if weights_by_graph:
            weights = np.array(
                [weights_by_graph.get(gid, {}).get(nid, np.nan) for gid, nid in zip(fdf["graph_id"], fdf["node_id"])]
            )

    grouped = fdf.groupby("graph_id")[cols]
    parts = []
    for stat in stats:
        if stat == "mean":
            if weights is not None:
                weighted = fdf[cols].mul(weights, axis=0)
                weighted["graph_id"] = fdf["graph_id"]
                wsum = pd.Series(weights, index=fdf.index).groupby(fdf["graph_id"]).sum()
                part = weighted.groupby("graph_id")[cols].sum().div(wsum, axis=0)
            else:
                part = grouped.mean()
        elif stat == "max":
            part = grouped.max()
        elif stat == "min":
            part = grouped.min()
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
