"""Generator invariants: prevalence, and each anomaly type's structural signature."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.containment import make_synthetic


def _to_graph(edges_df, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    return G


def test_prevalence_and_determinism():
    e1, n1, cfg = make_synthetic("er", "hub", n=500, prevalence=0.02, seed=3)
    e2, n2, _ = make_synthetic("er", "hub", n=500, prevalence=0.02, seed=3)
    assert cfg["n_anomalies"] == 10
    assert n1["is_anomaly"].sum() == 10
    assert e1.equals(e2) and n1.equals(n2)


def test_hub_degrees_are_outliers():
    edges, nodes, cfg = make_synthetic("er", "hub", n=500, prevalence=0.02, seed=5, hub_degree_factor=8.0)
    G = _to_graph(edges, 500)
    anoms = nodes.loc[nodes["is_anomaly"] == 1, "node_id"]
    normal_median = np.median([d for v, d in G.degree() if v not in set(anoms)])
    assert all(G.degree(v) >= 5 * normal_median for v in anoms)


def test_clique_members_have_clique_neighbors():
    # Groups are formed in sampling order (not node-id order), so test the
    # grouping-independent invariant: every full-group anomaly has at least
    # clique_size-1 anomalous neighbors (its clique mates).
    edges, nodes, cfg = make_synthetic("ba", "clique", n=500, prevalence=0.032, seed=5, clique_size=8)
    G = _to_graph(edges, 500)
    anom_set = set(nodes.loc[nodes["is_anomaly"] == 1, "node_id"])
    assert len(anom_set) == 16  # two full cliques of 8
    for v in anom_set:
        assert sum(1 for u in G.neighbors(v) if u in anom_set) >= 7


def test_tail_nodes_are_degree_one():
    edges, nodes, cfg = make_synthetic("er", "tail", n=500, prevalence=0.02, seed=5)
    G = _to_graph(edges, 500)
    anoms = nodes.loc[nodes["is_anomaly"] == 1, "node_id"]
    assert all(G.degree(v) == 1 for v in anoms)
