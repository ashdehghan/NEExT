"""struc2vec (from scratch) against the paper's own sanity cases."""


import networkx as nx
import numpy as np
import pytest


from NEExT.node_embeddings.struc2vec import (
    build_layers,
    candidate_pairs,
    compute_rings,
    dtw_compressed,
    struc2vec_embed,
    struc2vec_walks,
    structural_distances,
)
from NEExT.node_embeddings.graph_utils import nx_to_csr


def _dtw_uncompressed(a, b):
    """Brute-force DTW on the expanded sequences with the paper's cost."""
    la, lb = len(a), len(b)
    D = np.full((la + 1, lb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = max(a[i - 1], b[j - 1]) / min(a[i - 1], b[j - 1]) - 1.0
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[la, lb]


def _compress(seq):
    vals, cnts = np.unique(np.asarray(seq), return_counts=True)
    return vals.astype(np.int32), cnts.astype(np.int32)


def test_dtw_compressed_matches_definition():
    # Identical sequences -> 0; the compressed cost equals uncompressed DTW
    # whenever every run is matched as a block (the OPT1 approximation), which is
    # exact for sequences with all-distinct values.
    rng = np.random.RandomState(0)
    for _ in range(50):
        a = np.sort(rng.choice(np.arange(1, 40), size=rng.randint(1, 8), replace=False))
        b = np.sort(rng.choice(np.arange(1, 40), size=rng.randint(1, 8), replace=False))
        av, ac = _compress(a)
        bv, bc = _compress(b)
        assert np.isclose(dtw_compressed(av, ac, bv, bc), _dtw_uncompressed(a, b))
    av, ac = _compress([2, 2, 3, 5])
    assert dtw_compressed(av, ac, av, ac) == 0.0
    # a run of count 3 vs count 1 of the same degree costs 0 (ratio 1)
    assert dtw_compressed(np.array([4], np.int32), np.array([3], np.int32), np.array([4], np.int32), np.array([1], np.int32)) == 0.0


def _mirrored_karate():
    """Two Zachary copies bridged by one edge between node 0 and its mirror;
    swapping the halves is an automorphism, so f_k(u, mirror(u)) must be 0."""
    K = nx.karate_club_graph()
    n = K.number_of_nodes()
    G = nx.Graph()
    G.add_nodes_from(range(2 * n))
    for u, v in K.edges:
        G.add_edge(u, v)
        G.add_edge(u + n, v + n)
    G.add_edge(0, n)
    return G, n


def test_mirror_nodes_have_zero_structural_distance():
    G, n = _mirrored_karate()
    indptr, indices, degree = nx_to_csr(G)
    k_star = 3
    rings = compute_rings(indptr, indices, degree, k_star)
    pair_u, pair_v = candidate_pairs(degree, None)  # all pairs (OPT2 off)
    f = structural_distances(*rings, k_star, pair_u, pair_v)
    assert f.shape == (len(pair_u), k_star + 1)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(pair_u, pair_v))}
    for u in range(n):
        row = f[idx[(u, u + n)]]
        finite = row[np.isfinite(row)]
        assert len(finite) >= 1 and np.all(finite == 0.0)
    # f_k is non-decreasing in k wherever defined
    fin = np.isfinite(f)
    for p in range(len(pair_u)):
        vals = f[p][fin[p]]
        assert np.all(np.diff(vals) >= -1e-12)


def test_barbell_roles_separate():
    G = nx.barbell_graph(8, 6)  # two 8-cliques joined by a 6-node path
    indptr, indices, degree = nx_to_csr(G)
    rings = compute_rings(indptr, indices, degree, 2)
    pair_u, pair_v = candidate_pairs(degree, None)
    f = structural_distances(*rings, 2, pair_u, pair_v)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(pair_u, pair_v))}
    # interior clique nodes are structurally identical to each other
    assert f[idx[(1, 2)], 2] == 0.0 and f[idx[(1, 17 + 2)] if (1, 19) in idx else idx[(1, 19)], 2] == 0.0
    # symmetric path nodes (8 and 13) are identical; clique node vs path centre is not
    assert f[idx[(8, 13)], 2] == 0.0
    assert f[idx[(1, 10)], 0] > 0.0
    layers = build_layers(G.number_of_nodes(), 2, pair_u, pair_v, f)
    walks = struc2vec_walks(layers, 2, n_walks=2, length=15, seed=1)
    assert walks.shape == (G.number_of_nodes() * 2, 15)
    assert walks.min() >= 0 and walks.max() < G.number_of_nodes()


def test_embedding_recovers_mirrors_and_is_deterministic():
    pytest.importorskip("gensim")
    G, n = _mirrored_karate()
    kw = {"dim": 16, "seed": 11, "k_star": 3, "n_candidates": None, "n_walks": 20, "walk_length": 80, "epochs": 5}
    emb1, rec1 = struc2vec_embed(G, **kw)
    emb2, _ = struc2vec_embed(G, **kw)
    assert emb1.shape == (2 * n, 16) and np.isfinite(emb1).all()
    assert np.array_equal(emb1, emb2)
    # Structurally identical nodes (f_k == 0 at every defined layer) are
    # interchangeable for struc2vec, so the nearest neighbour must be the mirror
    # OR a structurally identical node (22 of the 68 nodes have >= 2 such twins).
    indptr, indices, degree = nx_to_csr(G)
    rings = compute_rings(indptr, indices, degree, 3)
    pair_u, pair_v = candidate_pairs(degree, None)
    f = structural_distances(*rings, 3, pair_u, pair_v)
    twins = {u: set() for u in range(2 * n)}
    for a, b, row in zip(pair_u, pair_v, f):
        fin = row[np.isfinite(row)]
        if np.all(fin == 0.0):
            twins[int(a)].add(int(b))
            twins[int(b)].add(int(a))
    X = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    mirror = np.r_[np.arange(n) + n, np.arange(n)]
    ok = np.mean([nn[u] == mirror[u] or int(nn[u]) in twins[u] for u in range(2 * n)])
    assert ok >= 0.65, f"NN is mirror/twin for only {ok:.2f} of nodes"
    cos_mirror = np.mean([S[u, mirror[u]] for u in range(2 * n)])
    cos_any = np.mean(S[np.isfinite(S)])
    assert cos_mirror >= 0.9 and cos_mirror > cos_any + 0.4, (cos_mirror, cos_any)
    assert rec1["k_star"] == 3 and rec1["n_pairs"] == (2 * n) * (2 * n - 1) // 2


def test_opt2_candidates_cover_every_node():
    G = nx.barbell_graph(8, 6)
    _, _, degree = nx_to_csr(G)
    pair_u, pair_v = candidate_pairs(degree, 5, seed=3)
    assert np.all(pair_u < pair_v)
    touched = np.zeros(G.number_of_nodes(), bool)
    touched[pair_u] = True
    touched[pair_v] = True
    assert touched.all()
    # same-degree nodes are preferred: every clique node pairs with another clique node
    cl = set(range(8)) | set(range(14, 22))
    for u in cl:
        partners = set(pair_v[pair_u == u]) | set(pair_u[pair_v == u])
        assert partners & cl
