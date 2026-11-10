"""Flat CSR graph utilities shared by the from-scratch node embedders.

Every embedder in this package consumes the undirected `networkx.Graph`
produced by `NEExT.node_embeddings.to_simple_networkx` (consecutive 0..n-1 ids,
connected, no self loops) and converts it once to a CSR adjacency
(`indptr`, `indices`) so the hot loops can run in numba.
"""

import numpy as np
from numba import njit


def nx_to_csr(G):
    """Undirected nx.Graph -> (indptr int64[n+1], indices int32[nnz], degree int32[n]).

    Neighbour lists are sorted; the karateclub contract (0..n-1 ids, connected,
    no self loops) is asserted rather than trusted.
    """
    import networkx as nx

    n = G.number_of_nodes()
    if sorted(G.nodes) != list(range(n)):
        raise AssertionError("Node ids are not consecutive 0..n-1")
    if any(u == v for u, v in G.edges):
        raise AssertionError("Graph has self loops")
    if n > 1 and not nx.is_connected(G):
        raise AssertionError("Graph is not connected")
    src, dst = [], []
    for u, v in G.edges:
        src.append(u)
        dst.append(v)
        src.append(v)
        dst.append(u)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    degree = np.bincount(src, minlength=n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=indptr[1:])
    return indptr, dst.astype(np.int32), degree


@njit(cache=True)
def bfs_ball_sizes(indptr, indices, source, k_max, stamp, stamp_val, queue):
    """|B_k(source)| for k = 1..k_max (cumulative ball sizes, excluding the source)."""
    out = np.zeros(k_max, dtype=np.int64)
    head = 0
    tail = 0
    queue[tail] = source
    tail += 1
    stamp[source] = stamp_val
    depth = 0
    frontier_end = tail
    total = 0
    while head < tail and depth < k_max:
        # process the whole current frontier
        while head < frontier_end:
            u = queue[head]
            head += 1
            for j in range(indptr[u], indptr[u + 1]):
                v = indices[j]
                if stamp[v] != stamp_val:
                    stamp[v] = stamp_val
                    queue[tail] = v
                    tail += 1
        depth += 1
        total = tail - 1
        out[depth - 1] = total
        frontier_end = tail
    for d in range(depth, k_max):
        out[d] = total
    return out


def sample_ball_sizes(indptr, indices, k_max=8, n_sample=300, seed=42):
    """Mean |B_k| over a seeded sample of source nodes, k = 1..k_max."""
    n = len(indptr) - 1
    rng = np.random.RandomState(seed)
    sources = np.arange(n) if n <= n_sample else rng.choice(n, n_sample, replace=False)
    stamp = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    acc = np.zeros(k_max, dtype=np.float64)
    for i, s in enumerate(sources):
        acc += bfs_ball_sizes(indptr, indices, int(s), k_max, stamp, i, queue)
    return acc / len(sources)


def choose_k_star(indptr, indices, k_cap=6, min_ring=1.0, n_sample=300, seed=42):
    r"""struc2vec OPT3 layer cap: the largest k <= k_cap whose mean ring
    R_k = B_k \ B_{k-1} still holds at least `min_ring` nodes on average — i.e.
    every layer that carries ring information for a typical node, capped for
    cost (the reference default is every layer up to the diameter).

    Returns (k_star, ball_curve) with ball_curve[k-1] = mean |B_k|.
    """
    curve = sample_ball_sizes(indptr, indices, k_max=k_cap, n_sample=n_sample, seed=seed)
    rings = np.diff(np.concatenate([[0.0], curve]))
    k_star = 1
    for k in range(1, k_cap + 1):
        if rings[k - 1] >= min_ring:
            k_star = k
        else:
            break
    return int(k_star), curve


def normalized_laplacian_csr(indptr, indices):
    """L = I - D^{-1/2} A D^{-1/2} as a scipy CSR (float64)."""
    import scipy.sparse as sp

    n = len(indptr) - 1
    degree = np.diff(indptr).astype(np.float64)
    dinv = 1.0 / np.sqrt(degree)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    data = -dinv[rows] * dinv[indices]
    A = sp.csr_matrix((data, indices.astype(np.int64), indptr), shape=(n, n))
    return (sp.identity(n, format="csr") + A).tocsr()
