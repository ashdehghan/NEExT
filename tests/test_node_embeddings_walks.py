"""graph_utils CSR/ball-size helpers and walks/skip-gram invariants."""


import networkx as nx
import numpy as np
import pytest


from NEExT.node_embeddings.graph_utils import choose_k_star, normalized_laplacian_csr, nx_to_csr, sample_ball_sizes
from NEExT.node_embeddings.walks import LazyCorpus, node2vec_walks, train_skipgram, uniform_walks


def test_nx_to_csr_sorted_and_symmetric():
    G = nx.karate_club_graph()
    indptr, indices, degree = nx_to_csr(G)
    assert indptr[-1] == 2 * G.number_of_edges()
    for u in G.nodes:
        nbrs = indices[indptr[u] : indptr[u + 1]]
        assert list(nbrs) == sorted(G.neighbors(u))
        assert degree[u] == G.degree(u)


def test_nx_to_csr_rejects_bad_graphs():
    G = nx.path_graph(5)
    G.add_edge(2, 2)
    with pytest.raises(AssertionError):
        nx_to_csr(G)
    H = nx.Graph()
    H.add_edges_from([(0, 1), (3, 4)])
    with pytest.raises(AssertionError):
        nx_to_csr(H)


def test_ball_sizes_path_and_clique():
    indptr, indices, _ = nx_to_csr(nx.complete_graph(10))
    curve = sample_ball_sizes(indptr, indices, k_max=3, n_sample=10)
    assert np.allclose(curve, [9, 9, 9])
    k_star, _ = choose_k_star(indptr, indices, k_cap=6)
    assert k_star == 1  # ring 2 of a clique is empty
    indptr, indices, _ = nx_to_csr(nx.path_graph(101))
    curve = sample_ball_sizes(indptr, indices, k_max=4, n_sample=101)
    # interior nodes see 2k nodes within k hops; ends see k -> mean slightly below 2k
    assert curve[0] < 2 and curve[3] < 8 and curve[3] > 7.5
    k_star, _ = choose_k_star(indptr, indices, k_cap=6)
    assert k_star == 6  # every ring of a long path is populated -> the cap
    # star graph: ring 1 populated for every node, ring 2 populated only for leaves
    indptr, indices, _ = nx_to_csr(nx.star_graph(20))
    k_star, curve = choose_k_star(indptr, indices, k_cap=6, n_sample=21)
    assert k_star == 2 and np.isclose(curve[1], 20.0)


def test_normalized_laplacian_spectrum():
    G = nx.karate_club_graph()
    indptr, indices, _ = nx_to_csr(G)
    L = normalized_laplacian_csr(indptr, indices).toarray()
    ref = nx.normalized_laplacian_matrix(G, nodelist=range(34), weight=None).toarray()
    assert np.allclose(L, ref)
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-10 and w.max() <= 2 + 1e-10


def test_uniform_walks_valid_and_deterministic():
    G = nx.karate_club_graph()
    indptr, indices, _ = nx_to_csr(G)
    w1 = uniform_walks(indptr, indices, n_walks=3, length=10, seed=7)
    w2 = uniform_walks(indptr, indices, n_walks=3, length=10, seed=7)
    w3 = uniform_walks(indptr, indices, n_walks=3, length=10, seed=8)
    assert w1.shape == (34 * 3, 10) and w1.dtype == np.int32
    assert np.array_equal(w1, w2) and not np.array_equal(w1, w3)
    assert list(w1[:34, 0]) == list(range(34))
    for row in w1:
        for a, b in zip(row[:-1], row[1:]):
            assert G.has_edge(int(a), int(b))


def test_lazy_corpus_restartable():
    walks = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    corpus = LazyCorpus(walks, 3)
    assert list(corpus) == [["0", "1", "2"], ["2", "1", "0"]]
    assert list(corpus) == [["0", "1", "2"], ["2", "1", "0"]]  # second pass identical


def test_train_skipgram_bit_exact_and_row_order():
    pytest.importorskip("gensim")
    G = nx.karate_club_graph()
    indptr, indices, _ = nx_to_csr(G)
    walks = uniform_walks(indptr, indices, n_walks=5, length=20, seed=1)
    e1 = train_skipgram(walks, 34, dim=8, window=3, seed=3)
    e2 = train_skipgram(walks, 34, dim=8, window=3, seed=3)
    e3 = train_skipgram(walks, 34, dim=8, window=3, seed=4)
    assert e1.shape == (34, 8) and e1.dtype == np.float32
    assert np.array_equal(e1, e2) and not np.array_equal(e1, e3)
    assert np.isfinite(e1).all()


def test_node2vec_walks_transition_probabilities():
    # prev=0 --> cur=1 on a 4-node graph: 1's neighbours are 0 (return), 2 (common
    # neighbour of 0 and 1), 3 (outward). Expected: proportional to 1/p, 1, 1/q.
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    indptr, indices, _ = nx_to_csr(G)
    for p, q in [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25)]:
        walks = node2vec_walks(indptr, indices, n_walks=4000, length=3, p=p, q=q, seed=5)
        rows = walks[(walks[:, 0] == 0) & (walks[:, 1] == 1)]
        counts = np.bincount(rows[:, 2], minlength=4).astype(float)
        expect = np.array([1 / p, 0.0, 1.0, 1 / q])
        expect /= expect.sum()
        assert len(rows) > 500
        assert np.allclose(counts / counts.sum(), expect, atol=0.05), (p, q, counts / counts.sum(), expect)
    w1 = node2vec_walks(indptr, indices, n_walks=3, length=10, p=0.5, q=2.0, seed=9)
    w2 = node2vec_walks(indptr, indices, n_walks=3, length=10, p=0.5, q=2.0, seed=9)
    assert np.array_equal(w1, w2)
    # p = q = 1 matches the uniform walker's distribution shape (adjacency-valid)
    for row in w1:
        for a, b in zip(row[:-1], row[1:]):
            assert G.has_edge(int(a), int(b))
