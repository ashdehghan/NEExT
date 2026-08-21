"""Role2Vec from scratch (Ahmed, Rossi, Lee, Willke, Zhou, Kong & Eldardiry,
"Learning Role-based Graph Embeddings", arXiv:1802.02896, 2018).

The original algorithm: (1) a structural feature vector x(v) per node, (2) a
function Phi mapping x(v) to a discrete *type* by logarithmic binning, (3)
uniform random walks on the graph whose node sequences are rewritten as type
sequences ("attributed random walks"), (4) skip-gram over the type sequences,
(5) every node's embedding is the embedding of its type — nodes of the same
type share a vector by construction.

Features: the paper's default is graphlet/motif counts up to 4 nodes (via
PGD); we use the 3-node graphlet orbit counts (degree, triangles, open wedges
centred at v) plus the k-core number, computed with networkx. Binning: vertical
logarithmic binning with fraction `alpha` (the ReFeX scheme role2vec adopts):
the lowest alpha-fraction of values get bin 0, the lowest alpha-fraction of the
remainder bin 1, and so on; ties share a bin. karateclub's `Role2Vec` (WL-hash
documents + Doc2Vec) is a different estimator and is not used here.
"""

import time

import numpy as np

from .graph_utils import nx_to_csr
from .walks import train_skipgram, uniform_walks


def structural_features(G):
    """[degree, triangles, open wedges centred at v, core number] as int64[n, 4]."""
    import networkx as nx

    n = G.number_of_nodes()
    deg = np.array([G.degree(v) for v in range(n)], dtype=np.int64)
    tri = nx.triangles(G)
    tri = np.array([tri[v] for v in range(n)], dtype=np.int64)
    wedges = deg * (deg - 1) // 2 - tri
    core = nx.core_number(G)
    core = np.array([core[v] for v in range(n)], dtype=np.int64)
    return np.stack([deg, tri, wedges, core], axis=1)


def log_bin(values, alpha=0.5):
    """Vertical logarithmic binning of one feature column (ReFeX / role2vec).

    Sorted unique values are consumed in increasing order; bin b takes values
    until at least alpha * (remaining nodes) are covered, then b += 1."""
    values = np.asarray(values)
    n = len(values)
    uniq, counts = np.unique(values, return_counts=True)
    bins_of_value = np.empty(len(uniq), dtype=np.int64)
    remaining = n
    b = 0
    taken = 0
    for i in range(len(uniq)):
        bins_of_value[i] = b
        taken += counts[i]
        if taken >= alpha * remaining and i < len(uniq) - 1:
            remaining -= taken
            taken = 0
            b += 1
    return bins_of_value[np.searchsorted(uniq, values)]


def node_types(features, alpha=0.5):
    """Phi: bin every feature column, then one type id per distinct bin tuple.
    Returns (type_of int32[n], n_types)."""
    binned = np.stack([log_bin(features[:, j], alpha) for j in range(features.shape[1])], axis=1)
    _, inverse = np.unique(binned, axis=0, return_inverse=True)
    type_of = inverse.reshape(-1).astype(np.int32)
    return type_of, int(type_of.max()) + 1


def role2vec_embed(G, dim=16, seed=42, n_walks=10, walk_length=80, window=10, epochs=1, alpha=0.5):
    """Full Role2Vec pipeline on an nx.Graph (to_networkx contract).

    Returns (emb float32[n, dim], record); record carries n_types and timings."""
    t0 = time.perf_counter()
    indptr, indices, degree = nx_to_csr(G)
    len(degree)
    feats = structural_features(G)
    type_of, n_types = node_types(feats, alpha=alpha)
    t1 = time.perf_counter()
    walks = uniform_walks(indptr, indices, n_walks=n_walks, length=walk_length, seed=seed)
    type_walks = type_of[walks]
    t2 = time.perf_counter()
    type_emb = train_skipgram(type_walks, n_types, dim=dim, window=window, seed=seed, epochs=epochs)
    emb = type_emb[type_of]
    t3 = time.perf_counter()
    record = {
        "n_types": n_types,
        "alpha": alpha,
        "features": ["degree", "triangles", "wedges", "core_number"],
        "n_walks": n_walks,
        "walk_length": walk_length,
        "window": window,
        "epochs": epochs,
        "dim": dim,
        "seed": seed,
        "t_features": t1 - t0,
        "t_walks": t2 - t1,
        "t_skipgram": t3 - t2,
    }
    return emb, record
