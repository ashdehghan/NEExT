"""Tests for the synthetic homophily-reach generator (mechanics only)."""

import numpy as np
import pandas as pd
import pytest

from lib.nodeclass.synthetic import LABEL_COLUMN, rgg_homophily_graph


def small(**kw):
    defaults = dict(n=1200, mean_degree=8.0, wavelength=0.15, coupling=0.0, seed=7)
    defaults.update(kw)
    return rgg_homophily_graph(**defaults)


def degree_series(edges, n):
    return pd.concat([edges.src_node_id, edges.dest_node_id]).value_counts().reindex(range(n)).fillna(0)


def eta2(vals, y):
    overall = vals.mean()
    ss_b = sum(len(g) * (g.mean() - overall) ** 2 for _, g in vals.groupby(y))
    ss_t = ((vals - overall) ** 2).sum()
    return ss_b / ss_t


def edge_homophily(edges, labels):
    return (labels.reindex(edges.src_node_id).to_numpy() == labels.reindex(edges.dest_node_id).to_numpy()).mean()


def test_deterministic_per_seed():
    e1, n1 = small()
    e2, n2 = small()
    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(n1, n2)
    e3, _ = small(seed=8)
    assert len(e3) != len(e1) or not e3.equals(e1)


def test_mean_degree_near_target():
    edges, nodes = small()
    md = 2 * len(edges) / len(nodes)
    assert 6.5 <= md <= 9.5


def test_no_self_loops_or_duplicates():
    edges, _ = small()
    assert (edges.src_node_id != edges.dest_node_id).all()
    assert not edges.duplicated().any()
    assert (edges.src_node_id < edges.dest_node_id).all()


def test_zero_coupling_leaves_degree_dark():
    edges, nodes = small(coupling=0.0)
    deg = degree_series(edges, len(nodes))
    e2 = eta2(np.log1p(deg), nodes[LABEL_COLUMN])
    assert e2 < 0.02


def test_coupling_brightens_degree():
    edges, nodes = small(coupling=0.3, label_noise=0.0)
    deg = degree_series(edges, len(nodes))
    e2 = eta2(np.log1p(deg), nodes[LABEL_COLUMN])
    assert e2 > 0.05


def test_homophily_present_and_monotone_in_wavelength():
    h = {}
    for wl in (0.08, 0.30):
        edges, nodes = small(wavelength=wl, label_noise=0.0)
        y = nodes.set_index("node_id")[LABEL_COLUMN]
        p = y.value_counts(normalize=True)
        h[wl] = edge_homophily(edges, y) - float((p**2).sum())
    assert h[0.30] > h[0.08] > 0.0


def test_most_nodes_connected():
    edges, nodes = rgg_homophily_graph(n=400, mean_degree=10, wavelength=0.2, seed=3)
    deg = degree_series(edges, len(nodes))
    assert (deg > 0).mean() > 0.95


def test_label_range_and_noise():
    _, nodes = small(n_classes=5)
    assert set(nodes[LABEL_COLUMN].unique()) <= {0, 1, 2, 3, 4}
    counts = nodes[LABEL_COLUMN].value_counts(normalize=True)
    assert counts.max() < 0.35


def test_invalid_coupling_rejected():
    with pytest.raises(ValueError):
        small(coupling=0.9)
