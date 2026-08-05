"""Realistic synthetic networks for the score-landscape study.

Design rationale (2026-08-05): the phase-1 plants (hub/clique/tail on ER/BA)
are blatant — the full-graph oracle scored ~1.0 on all of them, and a
homogeneous background makes any deviation glow. Landscape mapping needs
(a) a textured background with honest false-glow (communities + power-law
degrees) and (b) anomalies that are relational/distributional rather than
single-feature outliers. Three graded networks:

  N1 calibration  — ER + degree-1 tails (deliberately easy; bug detector).
  N2 fraud rings  — LFR background + quasi-cliques at densities 0.9/0.6/0.3
                    mixed in ONE graph (a subtlety dial on a single map).
  N3 infiltrators — LFR background + degree-preserving random rewires
                    (degree unchanged; only the neighborhood *shape* is wrong).

All generators return connected graphs relabeled 0..n'-1 BEFORE dataframes are
built, so the loader's largest-component filter drops nothing and node ids in
the collection match `nodes_df` exactly. Plant metadata (ring ids/densities)
rides along in `nodes_df` extra columns; strip to [node_id, is_anomaly] before
loading if features must stay structural-only.
"""

from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .synthetic import make_synthetic


def _largest_component_relabel(G: nx.Graph) -> Tuple[nx.Graph, dict]:
    """Keep the largest component, relabel to 0..n'-1; return (graph, old->new)."""
    component = max(nx.connected_components(G), key=len)
    H = G.subgraph(component).copy()
    mapping = {old: new for new, old in enumerate(sorted(H.nodes()))}
    return nx.relabel_nodes(H, mapping), mapping


def lfr_background(n: int = 1500, seed: int = 7, mu: float = 0.1) -> nx.Graph:
    """LFR benchmark graph (power-law degrees + communities); SBM fallback.

    Tries a few seeds; LFR generation can fail to converge for some seeds.
    The fallback is a random-partition (planted-communities) graph with
    power-law-ish community sizes — less standard, still textured.
    """
    for attempt in range(5):
        try:
            G = nx.LFR_benchmark_graph(
                n,
                tau1=2.5,
                tau2=1.5,
                mu=mu,
                average_degree=8,
                max_degree=50,
                min_community=30,
                max_community=150,
                seed=seed + attempt,
            )
            G.remove_edges_from(nx.selfloop_edges(G))
            return nx.Graph(G)  # strip community attrs / multiedges defensively
        except Exception:
            continue
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(min(150, max(30, rng.pareto(2.0) * 40))))
    sizes[-1] = max(30, n - sum(sizes[:-1]))
    return nx.random_partition_graph(sizes, 0.15, 0.004, seed=seed)


def make_calibration_tails(n: int = 1500, prevalence: float = 0.01, seed: int = 7):
    """N1: ER + degree-1 tails via the phase-1 generator, connectivity-cleaned."""
    edges_df, nodes_df, config = make_synthetic("er", "tail", n=n, prevalence=prevalence, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(nodes_df["node_id"])
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    labels = dict(zip(nodes_df["node_id"], nodes_df["is_anomaly"]))
    G, mapping = _largest_component_relabel(G)
    nodes_df = pd.DataFrame(
        {"node_id": sorted(G.nodes()), "is_anomaly": [labels[old] for old in sorted(mapping, key=mapping.get)]}
    )
    edges_df = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    config.update({"network": "calibration_tails", "n_kept": G.number_of_nodes()})
    return edges_df, nodes_df, config


def make_fraud_rings(
    n: int = 1500,
    seed: int = 7,
    ring_size: int = 10,
    ring_densities: Tuple[float, ...] = (0.9, 0.6, 0.3),
):
    """N2: LFR background + one quasi-clique per density, mixed in one graph.

    Ring members are sampled across the whole graph (a fraud ring is its own
    group, not a subset of one community) and wired to reach internal edge
    density >= rho. Members keep their organic edges.
    """
    rng = np.random.default_rng(seed)
    G = lfr_background(n=n, seed=seed)
    G, _ = _largest_component_relabel(G)
    n_kept = G.number_of_nodes()

    ring_of = {}
    density_of = {}
    available = rng.permutation(n_kept).tolist()
    for ring_id, rho in enumerate(ring_densities):
        members = [available.pop() for _ in range(ring_size)]
        pairs = [(u, v) for i, u in enumerate(members) for v in members[i + 1 :]]
        target_edges = int(np.ceil(rho * len(pairs)))
        existing = [p for p in pairs if G.has_edge(*p)]
        missing = [p for p in pairs if not G.has_edge(*p)]
        rng.shuffle(missing)
        for u, v in missing[: max(0, target_edges - len(existing))]:
            G.add_edge(u, v)
        for m in members:
            ring_of[m] = ring_id
            density_of[m] = rho

    nodes = sorted(G.nodes())
    nodes_df = pd.DataFrame(
        {
            "node_id": nodes,
            "is_anomaly": [int(v in ring_of) for v in nodes],
            "ring_id": [ring_of.get(v, -1) for v in nodes],
            "ring_density": [density_of.get(v, 0.0) for v in nodes],
        }
    )
    edges_df = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    config = {
        "network": "fraud_rings",
        "n": n,
        "n_kept": n_kept,
        "seed": seed,
        "ring_size": ring_size,
        "ring_densities": list(ring_densities),
        "n_anomalies": len(ring_of),
    }
    return edges_df, nodes_df, config


def make_infiltrators(n: int = 1500, prevalence: float = 0.01, seed: int = 7):
    """N3: LFR background + degree-preserving random rewires.

    Each infiltrator keeps its exact degree but every edge is redirected to a
    uniformly random non-neighbor — neighbors that don't know each other,
    egonets that mix communities. Purely relational anomaly.
    """
    rng = np.random.default_rng(seed)
    G = lfr_background(n=n, seed=seed)
    G, _ = _largest_component_relabel(G)
    n_kept = G.number_of_nodes()

    n_anom = max(1, int(round(prevalence * n_kept)))
    infiltrators = rng.choice(n_kept, size=n_anom, replace=False)
    infiltrator_set = set(int(v) for v in infiltrators)
    for v in infiltrators:
        v = int(v)
        degree = G.degree(v)
        G.remove_edges_from(list(G.edges(v)))
        candidates = rng.permutation(n_kept)
        for u in candidates:
            u = int(u)
            if G.degree(v) >= degree:
                break
            if u != v and not G.has_edge(v, u):
                G.add_edge(v, u)

    G, mapping = _largest_component_relabel(G)  # rewiring can strand nodes
    nodes = sorted(G.nodes())
    old_of = {new: old for old, new in mapping.items()}
    nodes_df = pd.DataFrame(
        {"node_id": nodes, "is_anomaly": [int(old_of[v] in infiltrator_set) for v in nodes]}
    )
    edges_df = pd.DataFrame(G.edges(), columns=["src_node_id", "dest_node_id"])
    config = {
        "network": "infiltrators",
        "n": n,
        "n_kept": G.number_of_nodes(),
        "seed": seed,
        "prevalence": prevalence,
        "n_anomalies": int(nodes_df["is_anomaly"].sum()),
    }
    return edges_df, nodes_df, config
