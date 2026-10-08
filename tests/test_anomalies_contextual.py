"""Degree-preserving, collateral-free contextual anomalies: contextual_regime_swap and contextual_combination.

Both base graphs are rings of 4-regular blocks (type A = circulant, high clustering; type B = random 4-regular) with
port links, so every block has the same degree sequence whatever its type. Anomalies flip whole blocks.
"""

import networkx as nx
import numpy as np
import pytest

from NEExT.generators import AnomalyConfig, AnomalyInjector, combination_rule_graph, regime_graph


def _ego_signature(G, v):
    """The 1-hop ego-net of v as a set of edges (induced subgraph on v and its neighbours)."""
    nodes = set(G.neighbors(v)) | {v}
    return frozenset(frozenset(e) for e in G.subgraph(nodes).edges())


def _inject(G, kind, fraction=0.03, seed=3):
    return AnomalyInjector(seed=seed).inject(G, AnomalyConfig(anomaly_type=kind, fraction=fraction))


@pytest.mark.parametrize(
    "build,kind",
    [
        (lambda: regime_graph(n_blocks=60, seed=1), "contextual_regime_swap"),
        (lambda: combination_rule_graph(n_periods=20, seed=1), "contextual_combination"),
    ],
)
def test_degree_sequence_preserved_and_no_collateral(build, kind):
    G = build()
    before_deg = dict(G.degree())
    before_ego = {v: _ego_signature(G, v) for v in G}
    _inject(G, kind)
    assert dict(G.degree()) == before_deg
    anomalies = {v for v in G if G.nodes[v]["is_outlier"] == 1}
    assert anomalies
    changed = {v for v in G if _ego_signature(G, v) != before_ego[v]}
    assert changed <= anomalies  # no non-anomalous node's 1-hop ego-net changes
    assert G.graph["anomaly_report"]["collateral"] == []


def test_block_degree_sequence_is_type_independent():
    for G in (regime_graph(n_blocks=20, seed=2), combination_rule_graph(n_periods=4, seed=2)):
        for b in G.graph["blocks"]:
            assert sorted(d for _, d in G.degree(b["nodes"])) == [4] * 18 + [5] * 2


def test_regime_graph_clustering_by_regime_and_swap_reaches_other_regime():
    G = regime_graph(n_blocks=60, seed=4)
    c = nx.clustering(G)
    h = np.mean([c[n] for n in G if G.nodes[n]["regime"] == "H"])
    low = np.mean([c[n] for n in G if G.nodes[n]["regime"] == "L"])
    assert h > 0.4 and low < 0.15
    _inject(G, "contextual_regime_swap", fraction=0.05, seed=5)
    c2 = nx.clustering(G)
    for b in G.graph["blocks"]:
        if b["anomalous"]:
            own = h if b["regime"] == "H" else low
            other = low if b["regime"] == "H" else h
            m = np.mean([c2[n] for n in b["nodes"]])
            assert abs(m - other) < abs(m - own)  # the flipped block looks like the other regime
            nb = len(G.graph["blocks"])
            i = b["index"]
            assert G.graph["blocks"][(i - 1) % nb]["regime"] == G.graph["blocks"][(i + 1) % nb]["regime"] == b["regime"]


def test_realised_counts_reported_and_on_target():
    G = regime_graph(n_blocks=60, seed=6)  # 1200 nodes, 3% -> 2 blocks
    _inject(G, "contextual_regime_swap", fraction=0.03, seed=7)
    rep = G.graph["anomaly_report"]
    assert rep["requested_blocks"] == rep["realised_blocks"] == 2
    assert rep["realised"] == sum(G.nodes[n]["is_outlier"] for n in G) == 40
    H = combination_rule_graph(n_periods=20, seed=6)  # 2000 nodes, 3% -> 3 blocks
    _inject(H, "contextual_combination", fraction=0.03, seed=7)
    rep = H.graph["anomaly_report"]
    assert rep["realised_blocks"] == rep["requested_blocks"] == 3
    assert rep["realised"] == sum(H.nodes[n]["is_outlier"] for n in H) == 60


def test_combination_rule_holds_and_anomalies_break_it():
    G = combination_rule_graph(n_periods=40, seed=7)
    _inject(G, "contextual_combination", fraction=0.03, seed=8)
    rule = [b for b in G.graph["blocks"] if b["role"] == "rule"]
    anom = [b for b in rule if b["anomalous"]]
    norm = [b for b in rule if not b["anomalous"]]
    assert anom
    assert all((b["type"] == "A") != (b["context_counts"][0] == b["context_counts"][1]) for b in anom)
    assert all((b["type"] == "A") == (b["context_counts"][0] == b["context_counts"][1]) for b in norm)
    # per-band means match up to the stratification granularity (few anomalous blocks per graph)
    ma = np.array([b["context_counts"] for b in anom]).mean(0)
    mn = np.array([b["context_counts"] for b in norm]).mean(0)
    assert np.all(np.abs(ma - mn) <= 0.5)


def test_deterministic_under_seed():
    def build():
        G = combination_rule_graph(n_periods=10, seed=9)
        _inject(G, "contextual_combination", fraction=0.05, seed=10)
        H = regime_graph(n_blocks=40, seed=9)
        _inject(H, "contextual_regime_swap", fraction=0.05, seed=10)
        return sorted(G.edges()), sorted(H.edges()), [H.nodes[n]["is_outlier"] for n in sorted(H)]

    assert build() == build()
