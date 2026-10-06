"""Tests for patch tiling: EgonetCollection.compute_leiden_patches and the pure helpers in NEExT.patches.

A patch is one subgraph per Leiden community (not one per node, which is what
compute_leiden_egonets builds). Covers: the partition property, one-hop expansion,
min-size merging (including a chain of merges), labels against hand computation,
determinism and seed isolation, membership round-trip, repeat-call rebuild,
multi-graph collections, the framework wrapper, downstream features and
embeddings, and that compute_leiden_egonets is untouched.
"""

import random
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.collections import EgonetCollection
from NEExT.patches import compute_patch_label, merge_small_communities, relabel_contiguous

# ---------------------------------------------------------------------------
# Toy graphs
# ---------------------------------------------------------------------------


def clique_edges(nodes):
    return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]


def cliques_in_a_ring(n_cliques=4, size=6, labels_per_clique=None):
    """n_cliques cliques joined into a ring by one edge each; node label = clique index (or given)."""
    edges, labels = [], []
    for c in range(n_cliques):
        base = c * size
        members = list(range(base, base + size))
        edges += clique_edges(members)
        edges.append((base + size - 1, ((c + 1) % n_cliques) * size))
        labels += [c if labels_per_clique is None else labels_per_clique[c]] * size
    n = n_cliques * size
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": list(range(n)), "lbl": labels})
    return edges_df, nodes_df


def load(edges_df, nodes_df, graph_type="igraph"):
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    return nxt, nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type)


def tile(nxt, collection, **kwargs):
    defaults = {"egonet_feature_target": "lbl", "min_patch_size": 1, "random_seed": 7}
    defaults.update(kwargs)
    return nxt.compute_leiden_patches(collection, **defaults)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_relabel_contiguous_preserves_order():
    assert relabel_contiguous([5, 2, 5, 9, 2]) == [1, 0, 1, 2, 0]


def test_merge_small_into_most_connected_neighbour():
    # community 0 = {0,1}, community 1 = {2,3,4,5}, community 2 = {6,7,8,9}
    # 0 has 1 edge to community 1 and 2 edges to community 2 -> merges into 2
    edges = [(0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (0, 2), (0, 6), (1, 7)]
    membership = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    merged, log = merge_small_communities(edges, membership, min_size=3)
    assert log == [(0, 2)]
    assert merged == [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]


def test_merge_chain_rechecks_absorbing_community():
    # singletons 0,1,2 in a path 0-1-2, plus a big community {3,4,5,6} attached to 2.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 5)]
    membership = [0, 1, 2, 3, 3, 3, 3]
    merged, log = merge_small_communities(edges, membership, min_size=3)
    # smallest first: 0 -> 1 (its only neighbour); then the singleton 2 (size 1 < size 2 of {0,1}) has one
    # edge each to community 1 and community 3, ties go to the lowest id -> 2 -> 1; {0,1,2} has size 3 and stays
    assert log == [(0, 1), (2, 1)]
    assert merged == [0, 0, 0, 1, 1, 1, 1]


def test_merge_leaves_isolated_small_community():
    edges = [(0, 1), (2, 3), (3, 4)]
    membership = [0, 0, 1, 1, 1]
    merged, log = merge_small_communities(edges, membership, min_size=3)
    assert log == []
    assert merged == [0, 0, 1, 1, 1]


def test_merge_ties_pick_lowest_neighbour_id():
    # community 2 = {4} has one edge each to community 0 and community 1 -> lowest id wins
    edges = [(0, 1), (2, 3), (4, 0), (4, 2)]
    membership = [0, 0, 1, 1, 2]
    merged, log = merge_small_communities(edges, membership, min_size=2)
    assert log == [(2, 0)]
    assert merged == [0, 0, 1, 1, 0]


def test_patch_label_schemes():
    labels = [1, 1, 2, 2, 3]
    label, fraction, hist = compute_patch_label(labels, "majority")
    assert label == 1 and fraction == pytest.approx(0.4) and hist == {1: 2, 2: 2, 3: 1}
    assert compute_patch_label(labels, "centre", centre_label=3)[0] == 3
    assert compute_patch_label(labels, "purity", purity_threshold=0.5)[0] == 0
    assert compute_patch_label([1, 1, 1, 2], "purity", purity_threshold=0.75)[0] == 1
    assert compute_patch_label(labels, "distribution")[0] == 1
    with pytest.raises(ValueError):
        compute_patch_label(labels, "centre")
    with pytest.raises(ValueError):
        compute_patch_label(labels, "mode")
    assert compute_patch_label([], "majority") == (None, 0.0, {})


# ---------------------------------------------------------------------------
# compute_leiden_patches
# ---------------------------------------------------------------------------


def test_requires_igraph_backend():
    nxt, collection = load(*cliques_in_a_ring(), graph_type="networkx")
    with pytest.raises(NotImplementedError, match="iGraph"):
        tile(nxt, collection)


def test_patches_partition_the_node_set_and_round_trip():
    nxt, collection = load(*cliques_in_a_ring(4, 6))
    patches = tile(nxt, collection)
    graph = collection.graphs[0]
    membership = patches.patch_memberships[graph.graph_id]

    assert len(patches.graphs) == 4
    assert len(membership) == len(graph.nodes)
    seen = Counter()
    for patch in patches.graphs:
        assert patch.original_graph_id == graph.graph_id
        for orig, internal in patch.node_mapping.items():
            assert membership[orig] == patch.graph_id
            assert internal in patch.nodes
            seen[orig] += 1
    assert seen == Counter(dict.fromkeys(graph.nodes, 1))
    assert len(patches.graph_id_node_array) == len(graph.nodes)
    assert sorted(set(membership)) == list(range(4))
    assert set(patches.egonet_to_graph_node_mapping) == {p.graph_id for p in patches.graphs}


def test_one_hop_expansion_overlaps_only_at_boundaries():
    nxt, collection = load(*cliques_in_a_ring(4, 6))
    plain = tile(nxt, collection)
    expanded = tile(nxt, collection, expand_one_hop=True)
    # memberships are the partition either way
    assert expanded.patch_memberships == plain.patch_memberships
    membership = plain.patch_memberships[0]
    for patch in expanded.graphs:
        community = {n for n, m in enumerate(membership) if m == patch.graph_id}
        members = set(patch.node_mapping)
        assert community <= members
        # each clique has exactly two boundary edges in the ring -> two extra nodes
        assert len(members - community) == 2
    assert len(expanded.graph_id_node_array) == 24 + 4 * 2


def test_min_patch_size_merges_in_the_collection():
    # 4 cliques of 6 plus one pendant pair hanging off clique 0 (likely its own community)
    edges_df, nodes_df = cliques_in_a_ring(4, 6)
    extra = pd.DataFrame({"src_node_id": [0, 24], "dest_node_id": [24, 25]})
    edges_df = pd.concat([edges_df, extra], ignore_index=True)
    nodes_df = pd.concat([nodes_df, pd.DataFrame({"node_id": [24, 25], "lbl": [9, 9]})], ignore_index=True)
    nxt, collection = load(edges_df, nodes_df)
    patches = tile(nxt, collection, min_patch_size=4)
    sizes = [len(p.nodes) for p in patches.graphs]
    assert min(sizes) >= 4
    membership = patches.patch_memberships[0]
    assert sorted(set(membership)) == list(range(len(patches.graphs)))
    assert sum(sizes) == 26
    # the pendant pair ends up with clique 0
    assert membership[24] == membership[25] == membership[0]


def test_labels_match_hand_computation():
    # clique 2 gets mixed labels: 4 of label 2, 2 of label 7
    edges_df, nodes_df = cliques_in_a_ring(4, 6)
    nodes_df.loc[[12, 13], "lbl"] = 7
    nxt, collection = load(edges_df, nodes_df)

    majority = tile(nxt, collection, patch_label="majority")
    for patch in majority.graphs:
        members = sorted(patch.node_mapping)
        clique = members[0] // 6
        if clique == 2:
            assert patch.graph_label == 2
            assert majority.patch_label_fraction[patch.graph_id] == pytest.approx(4 / 6)
            assert majority.patch_label_histogram[patch.graph_id] == {2: 4, 7: 2}
        else:
            assert patch.graph_label == clique
            assert majority.patch_label_fraction[patch.graph_id] == 1.0
    # integer node labels arrive as floats from the loader (pydantic coerces int64 into Union[float, int, str]);
    # the patch label is whatever the node table carried, so only integrality is checked here
    assert all(float(p.graph_label).is_integer() for p in majority.graphs)

    purity = tile(nxt, collection, patch_label="purity", purity_threshold=0.8)
    assert sorted(p.graph_label for p in purity.graphs) == [0, 1, 1, 1]

    distribution = tile(nxt, collection, patch_label="distribution")
    assert sorted(p.graph_label for p in distribution.graphs) == [0, 1, 2, 3]

    centre = tile(nxt, collection, patch_label="centre")
    for patch in centre.graphs:
        _, centre_node = centre.egonet_to_graph_node_mapping[patch.graph_id]
        assert patch.graph_label == int(nodes_df.loc[centre_node, "lbl"])


def test_centre_is_highest_degree_member():
    edges_df, nodes_df = cliques_in_a_ring(2, 5)
    # add a hub inside clique 1: node 5 gets two extra pendant nodes
    extra = pd.DataFrame({"src_node_id": [5, 5], "dest_node_id": [10, 11]})
    edges_df = pd.concat([edges_df, extra], ignore_index=True)
    nodes_df = pd.concat([nodes_df, pd.DataFrame({"node_id": [10, 11], "lbl": [1, 1]})], ignore_index=True)
    nxt, collection = load(edges_df, nodes_df)
    patches = tile(nxt, collection, min_patch_size=1)
    centres = {patches.egonet_to_graph_node_mapping[p.graph_id][1] for p in patches.graphs}
    assert 5 in centres


def test_no_label_target_gives_none_labels():
    nxt, collection = load(*cliques_in_a_ring())
    patches = nxt.compute_leiden_patches(collection, random_seed=1)
    assert all(p.graph_label is None for p in patches.graphs)
    assert patches.patch_label_fraction == {} and patches.patch_label_histogram == {}


def ambiguous_er_collection(seed=123, n=300, p=0.03):
    rng = np.random.RandomState(seed)
    upper = np.triu(rng.rand(n, n) < p, k=1)
    src, dst = np.nonzero(upper)
    edges_df = pd.DataFrame({"src_node_id": src, "dest_node_id": dst})
    nodes_df = pd.DataFrame({"node_id": list(range(n)), "lbl": rng.randint(0, 3, n)})
    return load(edges_df, nodes_df)


def test_same_seed_same_partition_different_seed_differs():
    nxt, collection = ambiguous_er_collection()
    a = tile(nxt, collection, random_seed=1, min_patch_size=1).patch_memberships[0]
    b = tile(nxt, collection, random_seed=1, min_patch_size=1).patch_memberships[0]
    c = tile(nxt, collection, random_seed=2, min_patch_size=1).patch_memberships[0]
    assert a == b
    assert a != c


def test_leiden_seeding_does_not_consume_global_random():
    nxt, collection = ambiguous_er_collection()
    random.seed(99)
    expected = [random.random() for _ in range(3)]
    random.seed(99)
    tile(nxt, collection, random_seed=1, min_patch_size=1)
    assert [random.random() for _ in range(3)] == expected


def test_partition_is_reproducible_in_a_fresh_process():
    code = (
        "import numpy as np, pandas as pd\n"
        "from NEExT import NEExT\n"
        "rng = np.random.RandomState(123); n=300\n"
        "upper = np.triu(rng.rand(n, n) < 0.03, k=1); src, dst = np.nonzero(upper)\n"
        "nxt = NEExT(); nxt.set_log_level('WARNING')\n"
        "c = nxt.load_single_graph_from_dfs(pd.DataFrame({'src_node_id': src, 'dest_node_id': dst}), graph_type='igraph')\n"
        "print(nxt.compute_leiden_patches(c, random_seed=1, min_patch_size=1).patch_memberships[0])\n"
    )
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout for _ in range(2)}
    assert len(outputs) == 1


def test_repeat_call_rebuilds_collection():
    nxt, collection = load(*cliques_in_a_ring(4, 6))
    patches = EgonetCollection(graph_type="igraph", egonet_feature_target="lbl")
    patches.compute_leiden_patches(collection, min_patch_size=1)
    patches.compute_leiden_patches(collection, min_patch_size=1)
    assert len(patches.graphs) == 4
    assert sorted(p.graph_id for p in patches.graphs) == [0, 1, 2, 3]
    assert len(patches.graph_id_node_array) == 24
    assert set(patches.patch_label_fraction) == {0, 1, 2, 3}


def test_two_source_graphs_get_global_patch_ids_and_per_graph_memberships():
    edges_a, nodes_a = cliques_in_a_ring(3, 5)
    edges_b, nodes_b = cliques_in_a_ring(2, 5)
    nodes_b["node_id"] += 100
    edges_b += 100
    edges_df = pd.concat([edges_a, edges_b], ignore_index=True)
    node_graph = pd.DataFrame({"node_id": list(nodes_a["node_id"]) + list(nodes_b["node_id"]), "graph_id": [0] * 15 + [1] * 10})
    node_features = pd.concat([nodes_a, nodes_b], ignore_index=True)
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    collection = nxt.graph_io.load_from_dfs(edges_df=edges_df, node_graph_df=node_graph, node_features_df=node_features, graph_type="igraph")
    patches = tile(nxt, collection)
    assert len(patches.graphs) == 5
    assert [p.graph_id for p in patches.graphs] == [0, 1, 2, 3, 4]
    assert [p.original_graph_id for p in patches.graphs] == [0, 0, 0, 1, 1]
    assert set(patches.patch_memberships) == {0, 1}
    assert len(patches.patch_memberships[0]) == 15 and len(patches.patch_memberships[1]) == 10
    assert sorted(set(patches.patch_memberships[1])) == [0, 1]


def test_framework_wrapper_reports_backend_and_skips_label_from_attributes():
    nxt, collection = load(*cliques_in_a_ring())
    patches = tile(nxt, collection)
    assert patches.graph_type == "igraph"
    assert all("lbl" not in attrs for p in patches.graphs for attrs in p.node_attributes.values())


def test_downstream_features_and_embeddings_run_on_patches():
    nxt, collection = load(*cliques_in_a_ring(4, 6))
    patches = tile(nxt, collection)
    local = nxt.compute_node_features(patches, feature_list=["degree_centrality", "page_rank"], feature_vector_length=2, show_progress=False)
    global_ = nxt.compute_node_features(
        patches, feature_list=["degree_centrality", "page_rank"], feature_vector_length=2, show_progress=False, feature_scope="global"
    )
    assert len(local.features_df) == 24 and len(global_.features_df) == 24
    embeddings = nxt.compute_graph_embeddings(patches, local, embedding_algorithm="approx_wasserstein", embedding_dimension=2)
    assert len(embeddings.embeddings_df) == 4
    assert embeddings.embeddings_df.notna().all().all()


def test_compute_leiden_egonets_unchanged():
    """Golden check: the per-node builder still yields one egonet per node with the same memberships."""
    nxt, collection = load(*cliques_in_a_ring(4, 6))
    random.seed(0)
    egonets = nxt.compute_leiden_egonets(collection, egonet_feature_target="lbl")
    assert len(egonets.graphs) == 24
    for egonet in egonets.graphs:
        centre = egonets.egonet_to_graph_node_mapping[egonet.graph_id][1]
        assert set(egonet.node_mapping) == set(range((centre // 6) * 6, (centre // 6) * 6 + 6))
        assert egonet.graph_label == centre // 6
