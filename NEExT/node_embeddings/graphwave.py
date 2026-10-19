"""GraphWave from scratch (Donnat, Zitnik, Hallac & Leskovec, KDD 2018),
following the authors' reference implementation (snap-stanford/graphwave):

* heat wavelets psi_a = g_s(L) delta_a with g_s(lambda) = exp(-s lambda) on the
  *normalized* Laplacian L = I - D^{-1/2} A D^{-1/2} (spectrum in [0, 2]);
* scales: 'auto' -> J = 2 scales linspace(s_min, s_max) with
  s_min = -ln(0.95) sqrt(0.5 / l1), s_max = -ln(0.80) sqrt(0.5 / l1), and the
  reference's cheap approximation l1 ~ 1/n of the smallest non-zero eigenvalue;
* embedding of node a at scale s: the empirical characteristic function of the
  wavelet coefficients, phi_a(t) = (1/n) sum_m exp(i t psi_a(m)), sampled at
  t in linspace(0, 100, 25) -> [Re, Im] pairs, concatenated over scales:
  2 * 25 * J = 100 features.

The wavelets are computed by a Chebyshev expansion of g_s on [0, 2] applied
to *batches* of impulse columns (so memory is O(n * batch), never O(n^2)), and
only for the requested `nodes` — each column is independent, so restricting
to the evaluation centres is exact. The reference's order-30 expansion is
raised to max(30, 2 * s_max) because exp(-s lambda) at s ~ 25 needs it; the
accuracy check against a dense eigendecomposition lives in the tests.
"""

import time

import numpy as np
from numba import njit, prange

from .graph_utils import normalized_laplacian_csr, nx_to_csr

ETA_MAX = 0.95
ETA_MIN = 0.80


def auto_scales(n, n_filters=2, l1=None):
    """Reference 'auto' scale rule (l1 defaults to the 1/n approximation)."""
    if l1 is None:
        l1 = 1.0 / n
    s_max = -np.log(ETA_MIN) * np.sqrt(0.5 / l1)
    s_min = -np.log(ETA_MAX) * np.sqrt(0.5 / l1)
    return np.linspace(s_min, s_max, n_filters)


def cheb_coefficients(scale, order):
    """Chebyshev coefficients c_0..c_order of g(x) = exp(-scale x) on [0, 2]
    (x = y + 1, y in [-1, 1]); g(L) ~ c_0/2 I + sum_{j>=1} c_j T_j(L - I)."""
    m = order + 1
    k = np.arange(m)
    theta = np.pi * (k + 0.5) / m
    x = np.cos(theta) + 1.0
    g = np.exp(-scale * x)
    j = np.arange(order + 1)
    return (2.0 / m) * (np.cos(np.outer(j, theta)) @ g)


def heat_wavelets_batched(Lt, cols, scales, coeffs, batch=256):
    """psi[:, cols] for every scale via the Chebyshev recursion on batches.

    Lt = L - I (scipy CSR). Returns list over scales of float64[n, len(cols)]."""
    n = Lt.shape[0]
    out = [np.empty((n, len(cols)), dtype=np.float64) for _ in scales]
    order = len(coeffs[0]) - 1
    for start in range(0, len(cols), batch):
        sel = cols[start : start + batch]
        b = len(sel)
        T0 = np.zeros((n, b), dtype=np.float64)
        T0[sel, np.arange(b)] = 1.0
        T1 = Lt @ T0
        acc = [0.5 * c[0] * T0 + c[1] * T1 for c in coeffs]
        for j in range(2, order + 1):
            T2 = 2.0 * (Lt @ T1) - T0
            for s, c in enumerate(coeffs):
                acc[s] += c[j] * T2
            T0, T1 = T1, T2
        for s in range(len(scales)):
            out[s][:, start : start + b] = acc[s]
    return out


@njit(parallel=True, cache=True)
def _charfn(psi, t_grid):
    """Empirical characteristic function per column: [Re(t_0..), Im(t_0..)]."""
    n, m = psi.shape
    T = len(t_grid)
    out = np.empty((m, 2 * T), dtype=np.float64)
    for a in prange(m):
        for ti in range(T):
            t = t_grid[ti]
            re = 0.0
            im = 0.0
            for i in range(n):
                v = t * psi[i, a]
                re += np.cos(v)
                im += np.sin(v)
            out[a, ti] = re / n
            out[a, T + ti] = im / n
    return out


def graphwave_embed(G, nodes=None, scales="auto", n_filters=2, time_pnts=None, order=None, batch=256, seed=42, dim=None):
    """GraphWave features for `nodes` (default: all). `seed`/`dim` are accepted
    for registry uniformity only: the method is deterministic and its width is
    2 * len(time_pnts) * n_filters (reported in the record).

    Returns (feat float64[len(nodes), width], record)."""
    t0 = time.perf_counter()
    indptr, indices, _ = nx_to_csr(G)
    n = len(indptr) - 1
    if nodes is None:
        cols = np.arange(n)
    else:
        cols = np.asarray(nodes, dtype=np.int64)
    if time_pnts is None:
        time_pnts = np.linspace(0, 100, 25)
    time_pnts = np.asarray(time_pnts, dtype=np.float64)
    if isinstance(scales, str) and scales == "auto":
        scales = auto_scales(n, n_filters=n_filters)
    scales = np.asarray(scales, dtype=np.float64)
    if order is None:
        order = int(max(30, np.ceil(2.0 * scales.max())))
    L = normalized_laplacian_csr(indptr, indices)
    import scipy.sparse as sp

    Lt = (L - sp.identity(n, format="csr")).tocsr()
    coeffs = [cheb_coefficients(s, order) for s in scales]
    psis = heat_wavelets_batched(Lt, cols, scales, coeffs, batch=batch)
    t1 = time.perf_counter()
    feats = np.concatenate([_charfn(psi, time_pnts) for psi in psis], axis=1)
    t2 = time.perf_counter()
    record = {
        "scales": [float(s) for s in scales],
        "n_filters": int(n_filters),
        "n_time_pnts": int(len(time_pnts)),
        "cheb_order": int(order),
        "n_features": int(feats.shape[1]),
        "n_columns": int(len(cols)),
        "t_wavelets": t1 - t0,
        "t_charfn": t2 - t1,
    }
    return feats, record
