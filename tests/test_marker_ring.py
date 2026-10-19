"""SyntheticPresets.marker_ring: the compositional bed of track T2 (brainstorming/jepa-synthetic-compositional/design.md)."""

from collections import Counter

import networkx as nx
import pytest

from NEExT.generators import SyntheticPresets

SPARSE = ((3, 13), (6, 10))
PERIODIC = ((3, 3, 3, 3, 3, 3), (2, 4, 2, 4, 2, 4))


def ring_gaps(blocks, n):
    blocks = sorted(blocks)
    return sorted(((blocks[(i + 1) % len(blocks)] - blocks[i]) % n) or n for i in range(len(blocks)))


@pytest.mark.parametrize("spacings,n_blocks", [(SPARSE, 16), (PERIODIC, 18)])
def test_degree_sequence_and_size_identical_across_classes(spacings, n_blocks):
    graphs = SyntheticPresets(seed=3).marker_ring(n_per_class=20, n_blocks=n_blocks, spacings=spacings)
    seqs = {tuple(sorted(d for _, d in G.degree())) for G in graphs}
    assert len(seqs) == 1
    assert all(G.number_of_nodes() == 6 * n_blocks and G.number_of_edges() == 10 * n_blocks for G in graphs)
    assert all(nx.is_connected(G) for G in graphs)


@pytest.mark.parametrize("spacings,n_blocks", [(SPARSE, 16), (PERIODIC, 18)])
def test_marker_spacing_and_counts(spacings, n_blocks):
    graphs = SyntheticPresets(seed=4).marker_ring(n_per_class=30, n_blocks=n_blocks, spacings=spacings)
    assert Counter(G.graph["label"] for G in graphs) == {0: 30, 1: 30}
    for G in graphs:
        sp = spacings[G.graph["label"]]
        assert ring_gaps(G.graph["marker_blocks"], n_blocks) == sorted(sp)
        assert len(G.graph["marker_blocks"]) == len(sp)
        assert sum(G.nodes[v]["is_marker"] for v in G) == 6 * len(sp)


def test_block_types():
    G = SyntheticPresets(seed=5).marker_ring(n_per_class=1, spacings=SPARSE)[0]
    for b in range(16):
        nodes = [v for v in G if G.nodes[v]["block"] == b]
        sub = G.subgraph(nodes)
        assert all(d == 3 for _, d in sub.degree())
        if G.nodes[nodes[0]]["is_marker"]:
            assert nx.is_bipartite(sub) and sum(nx.triangles(sub).values()) == 0
        else:
            assert sum(nx.triangles(sub).values()) == 6  # two triangles, each counted at 3 nodes


def test_start_positions_vary_and_deterministic():
    a = SyntheticPresets(seed=6).marker_ring(n_per_class=40, spacings=SPARSE)
    b = SyntheticPresets(seed=6).marker_ring(n_per_class=40, spacings=SPARSE)
    assert [G.graph["marker_blocks"] for G in a] == [G.graph["marker_blocks"] for G in b]
    assert len({G.graph["marker_blocks"][0] for G in a}) > 5


def test_bad_spacing_rejected():
    with pytest.raises(ValueError):
        SyntheticPresets(seed=1).marker_ring(n_per_class=1, n_blocks=16, spacings=((3, 3),))


def test_jitter_zero_reproduces_noise_free_output():
    a = SyntheticPresets(seed=3).marker_ring(n_per_class=20)
    b = SyntheticPresets(seed=3).marker_ring(n_per_class=20, jitter_prob=0.0)
    assert [G.graph["marker_blocks"] for G in a] == [G.graph["marker_blocks"] for G in b]
    assert all(G.graph["n_jittered"] == 0 for G in b)


def test_jitter_moves_markers_by_one_and_keeps_invariants():
    clean = SyntheticPresets(seed=4).marker_ring(n_per_class=50)
    noisy = SyntheticPresets(seed=4).marker_ring(n_per_class=50, jitter_prob=0.5)
    moved = 0
    for G0, G1 in zip(clean, noisy):
        assert G0.graph["label"] == G1.graph["label"]
        assert len(G1.graph["marker_blocks"]) == len(G0.graph["marker_blocks"])
        assert sorted(d for _, d in G1.degree()) == sorted(d for _, d in G0.degree())
        assert G1.number_of_nodes() == G0.number_of_nodes() and G1.number_of_edges() == G0.number_of_edges()
        moved += G1.graph["n_jittered"]
        assert sum(G1.nodes[v]["is_marker"] for v in G1) == 6 * len(G1.graph["marker_blocks"])
    assert moved > 0


def test_jitter_prob_out_of_range_rejected():
    with pytest.raises(ValueError):
        SyntheticPresets(seed=0).marker_ring(jitter_prob=1.5)
