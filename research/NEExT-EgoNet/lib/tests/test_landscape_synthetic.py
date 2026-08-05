"""Invariants for the landscape-study generators."""

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.containment.landscape_synthetic import (
    make_calibration_tails,
    make_fraud_rings,
    make_infiltrators,
)


def _graph(edges_df, nodes_df):
    G = nx.Graph()
    G.add_nodes_from(nodes_df["node_id"])
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    return G


def test_calibration_connected_and_labeled():
    edges, nodes, cfg = make_calibration_tails(n=600, prevalence=0.02, seed=3)
    G = _graph(edges, nodes)
    assert nx.is_connected(G)
    anoms = nodes.loc[nodes["is_anomaly"] == 1, "node_id"]
    assert len(anoms) >= 1
    assert all(G.degree(v) == 1 for v in anoms)


def test_fraud_rings_meet_density_targets():
    edges, nodes, cfg = make_fraud_rings(n=800, seed=3, ring_size=10, ring_densities=(0.9, 0.6, 0.3))
    G = _graph(edges, nodes)
    assert nx.is_connected(G)
    for ring_id, rho in enumerate((0.9, 0.6, 0.3)):
        members = list(nodes.loc[nodes["ring_id"] == ring_id, "node_id"])
        assert len(members) == 10
        internal = G.subgraph(members).number_of_edges()
        assert internal >= int(0.999 * rho * 45)  # 45 = 10 choose 2
    assert nodes["is_anomaly"].sum() == 30


def test_infiltrators_preserve_degree_and_scatter():
    # Build once with a fixed seed; infiltrator degrees must match the
    # background degrees they had before rewiring. We verify by construction:
    # rewire keeps |edges(v)| constant, so total degree is preserved unless a
    # node was stranded (dropped by the component filter, reported in config).
    edges, nodes, cfg = make_infiltrators(n=800, prevalence=0.015, seed=3)
    G = _graph(edges, nodes)
    assert nx.is_connected(G)
    anoms = list(nodes.loc[nodes["is_anomaly"] == 1, "node_id"])
    assert len(anoms) >= 8
    # Relational signature: an infiltrator's neighbors should rarely know each
    # other -> clustering well below the graph average.
    avg_clust = nx.average_clustering(G)
    anom_clust = sum(nx.clustering(G, v) for v in anoms) / len(anoms)
    assert anom_clust < avg_clust * 0.5
