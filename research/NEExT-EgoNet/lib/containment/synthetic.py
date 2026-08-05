"""Planted-anomaly synthetic graph generator for containment experiments.

Produces (edges_df, nodes_df) pairs ready for NEExT.load_single_graph_from_dfs,
with a binary `is_anomaly` node label. Three anomaly types with distinct
structural signatures:

  - hub:    degree outlier — extra edges to random nodes up to a multiple of
            the graph's median degree.
  - clique: planted-clique member — anomalies grouped into fully connected
            cliques (dense local neighborhood).
  - tail:   dangling node — all edges removed, one edge to a random node
            (degree-1, zero clustering).

Everything is seeded; the config dict returned alongside the frames records
every generator parameter for the run's config.json.
"""

from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

FAMILIES = ("er", "ba")
ANOMALY_TYPES = ("hub", "clique", "tail")


def make_synthetic(
    family: str,
    anomaly: str,
    n: int = 3000,
    prevalence: float = 0.02,
    seed: int = 7,
    mean_degree: float = 8.0,
    ba_m: int = 4,
    hub_degree_factor: float = 8.0,
    clique_size: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build a base graph, plant anomalies, return (edges_df, nodes_df, config)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if anomaly not in ANOMALY_TYPES:
        raise ValueError(f"anomaly must be one of {ANOMALY_TYPES}, got {anomaly!r}")

    rng = np.random.default_rng(seed)
    if family == "er":
        G = nx.gnm_random_graph(n, int(round(n * mean_degree / 2)), seed=seed)
    else:
        G = nx.barabasi_albert_graph(n, ba_m, seed=seed)

    n_anom = max(1, int(round(prevalence * n)))
    anomaly_nodes = rng.choice(n, size=n_anom, replace=False)

    if anomaly == "hub":
        target_degree = int(round(hub_degree_factor * np.median([d for _, d in G.degree()])))
        for v in anomaly_nodes:
            candidates = rng.permutation(n)
            for u in candidates:
                if G.degree(v) >= target_degree:
                    break
                if u != v and not G.has_edge(v, u):
                    G.add_edge(v, int(u))
    elif anomaly == "clique":
        for start in range(0, n_anom, clique_size):
            group = anomaly_nodes[start : start + clique_size]
            for i, u in enumerate(group):
                for v in group[i + 1 :]:
                    G.add_edge(int(u), int(v))
    else:  # tail
        for v in anomaly_nodes:
            G.remove_edges_from(list(G.edges(v)))
            others = rng.integers(0, n, size=8)
            anchor = next(int(u) for u in others if u != v)
            G.add_edge(int(v), anchor)

    edges_df = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    is_anom = np.zeros(n, dtype=int)
    is_anom[anomaly_nodes] = 1
    nodes_df = pd.DataFrame({"node_id": range(n), "is_anomaly": is_anom})

    config = {
        "family": family,
        "anomaly": anomaly,
        "n": n,
        "prevalence": prevalence,
        "n_anomalies": n_anom,
        "seed": seed,
        "mean_degree": mean_degree,
        "ba_m": ba_m,
        "hub_degree_factor": hub_degree_factor,
        "clique_size": clique_size,
    }
    return edges_df, nodes_df, config
