"""GraphWave (batched Chebyshev, from scratch) vs a dense eigendecomposition."""


import networkx as nx
import numpy as np
import scipy.sparse as sp


from NEExT.node_embeddings.graphwave import _charfn, auto_scales, cheb_coefficients, graphwave_embed, heat_wavelets_batched
from NEExT.node_embeddings.graph_utils import normalized_laplacian_csr, nx_to_csr


def _dense_wavelets(G, scales):
    indptr, indices, _ = nx_to_csr(G)
    L = normalized_laplacian_csr(indptr, indices).toarray()
    w, U = np.linalg.eigh(L)
    return [U @ np.diag(np.exp(-s * w)) @ U.T for s in scales]


def test_chebyshev_wavelets_match_dense():
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    scales = [1.0, 10.0]
    dense = _dense_wavelets(G, scales)
    indptr, indices, _ = nx_to_csr(G)
    L = normalized_laplacian_csr(indptr, indices)
    Lt = (L - sp.identity(n, format="csr")).tocsr()
    order = 50
    coeffs = [cheb_coefficients(s, order) for s in scales]
    cols = np.arange(n)
    psis = heat_wavelets_batched(Lt, cols, scales, coeffs, batch=7)
    for s in range(2):
        assert np.max(np.abs(psis[s] - dense[s])) < 1e-6
    # batched (batch=7) == one column at a time
    single = heat_wavelets_batched(Lt, cols, scales, coeffs, batch=1)
    for s in range(2):
        assert np.allclose(psis[s], single[s])


def test_auto_scale_order_is_sufficient_at_24k_nodes():
    # the reference's auto scales at n ~ 24.5k; the chosen order must reproduce
    # exp(-s x) on [0, 2] to 1e-6 (checked on the scalar function)
    scales = auto_scales(24492)
    order = int(max(30, np.ceil(2.0 * scales.max())))
    x = np.linspace(0, 2, 2001)
    y = x - 1.0
    for s in scales:
        c = cheb_coefficients(s, order)
        T0, T1 = np.ones_like(y), y.copy()
        approx = 0.5 * c[0] * T0 + c[1] * T1
        for j in range(2, order + 1):
            T2 = 2 * y * T1 - T0
            approx += c[j] * T2
            T0, T1 = T1, T2
        assert np.max(np.abs(approx - np.exp(-s * x))) < 1e-6


def test_charfn_and_subset_consistency():
    G = nx.barbell_graph(6, 4)
    full, rec = graphwave_embed(G, scales=[1.0, 5.0], time_pnts=np.linspace(0, 10, 5))
    assert full.shape == (16, 2 * 5 * 2) and rec["n_features"] == 20
    sub, _ = graphwave_embed(G, nodes=[3, 9, 12], scales=[1.0, 5.0], time_pnts=np.linspace(0, 10, 5))
    assert np.allclose(sub, full[[3, 9, 12]])
    # symmetric nodes of the barbell get identical features; clique vs path differ
    assert np.allclose(full[1], full[2]) and np.allclose(full[1], full[14])
    assert not np.allclose(full[1], full[7])
    # |phi(t)| <= 1 and phi(0) = 1
    assert full[:, 0].max() <= 1 + 1e-12 and np.allclose(full[:, 0], 1.0)
    # numba kernel vs numpy reference
    psi = np.random.RandomState(0).randn(12, 3)
    t = np.array([0.0, 0.5, 2.0])
    ref = np.concatenate(
        [np.cos(np.outer(t, psi[:, a])).mean(1).tolist() + np.sin(np.outer(t, psi[:, a])).mean(1).tolist() for a in range(3)]
    ).reshape(3, 6)
    assert np.allclose(_charfn(psi, t), ref)
