"""Role2Vec (original, from scratch): typing and embedding invariants."""


import networkx as nx
import numpy as np
import pytest


from NEExT.node_embeddings.role2vec import log_bin, node_types, role2vec_embed, structural_features


def test_log_bin_fractions_and_ties():
    vals = np.arange(100)
    b = log_bin(vals, alpha=0.5)
    # bin 0 = lowest half, bin 1 = next quarter, ... monotone, ties share bins
    assert (b[:50] == 0).all() and (b[50:75] == 1).all() and b[75] == 2
    assert np.all(np.diff(b) >= 0)
    assert len(set(log_bin(np.zeros(10)))) == 1
    # a tied value group is consumed whole: bin 0 = {1, 5} (6 nodes >= 5), bin 1 = {9}
    v = np.array([1, 1, 1, 1, 5, 5, 9, 9, 9, 9])
    assert log_bin(v, alpha=0.5).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_types_on_barbell():
    G = nx.barbell_graph(6, 4)
    feats = structural_features(G)
    assert feats.shape == (16, 4)
    type_of, n_types = node_types(feats, alpha=0.5)
    # interior clique nodes (0..4; node 5 is the connector) share one type,
    # path interior nodes (7, 8) another
    assert len(set(type_of[0:5])) == 1 and len(set(type_of[[7, 8]])) == 1
    assert type_of[0] != type_of[7]
    assert n_types >= 2


def _mirrored_karate():
    K = nx.karate_club_graph()
    n = K.number_of_nodes()
    G = nx.Graph()
    G.add_nodes_from(range(2 * n))
    for u, v in K.edges:
        G.add_edge(u, v)
        G.add_edge(u + n, v + n)
    G.add_edge(0, n)
    return G, n


def test_embedding_mirror_identity_and_determinism():
    pytest.importorskip("gensim")
    G, n = _mirrored_karate()
    emb1, rec = role2vec_embed(G, dim=8, seed=5, n_walks=5, walk_length=30)
    emb2, _ = role2vec_embed(G, dim=8, seed=5, n_walks=5, walk_length=30)
    assert emb1.shape == (2 * n, 8) and np.isfinite(emb1).all()
    assert np.array_equal(emb1, emb2)
    # mirrors have identical features -> identical type -> identical vector
    # (node 0 / n differ by the bridge edge and are exempt)
    for u in range(1, n):
        assert np.array_equal(emb1[u], emb1[u + n])
    assert 2 <= rec["n_types"] < 2 * n
