"""Feature-stage dedupe: grouping correctness and bit-exactness vs the naive path."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.nodeclass import build_node_bags
from lib.nodeclass.dedupe import dedupe_node_features, member_set_groups

from NEExT import NEExT


def dense_toy_collection(nxt):
    """A 12-node graph dense enough that k=3 egonets all saturate to the
    full component (guaranteed duplicate member sets)."""
    edges = []
    for i in range(12):
        edges.append((i, (i + 1) % 12))
        edges.append((i, (i + 2) % 12))
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": range(12), "y": [i % 3 for i in range(12)]})
    return nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )


FEATURE_KW = dict(feature_list=["degree_centrality", "closeness_centrality"], feature_vector_length=1, show_progress=False, n_jobs=1)


def test_groups_cover_all_bags():
    nxt = NEExT(log_level="ERROR")
    coll = dense_toy_collection(nxt)
    bags = build_node_bags(nxt, coll, list(range(12)), "y", method="k_hop", k_hop=3, seed=1)
    groups = member_set_groups(bags.egonets)
    all_ids = sorted(gid for gids in groups.values() for gid in gids)
    assert all_ids == sorted(g.graph_id for g in bags.egonets.graphs)
    # saturation: every k=3 bag is the whole 12-node graph -> one group
    assert len(groups) == 1


def test_exactness_vs_naive():
    nxt = NEExT(log_level="ERROR")
    coll = dense_toy_collection(nxt)
    bags = build_node_bags(nxt, coll, list(range(12)), "y", method="k_hop", k_hop=2, seed=1)

    naive = nxt.compute_node_features(bags.egonets, **FEATURE_KW)
    deduped = dedupe_node_features(nxt, bags.egonets, **FEATURE_KW)

    a = naive.features_df.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    b = deduped.features_df.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)], check_like=True)
    assert naive.feature_columns == deduped.feature_columns


def test_no_duplicates_passthrough():
    nxt = NEExT(log_level="ERROR")
    coll = dense_toy_collection(nxt)
    bags = build_node_bags(nxt, coll, list(range(12)), "y", method="k_hop", k_hop=1, seed=1)
    groups = member_set_groups(bags.egonets)
    if len(groups) == len(bags.egonets.graphs):
        feats = dedupe_node_features(nxt, bags.egonets, **FEATURE_KW)
        assert len(feats.features_df) > 0


def test_batched_matches_naive():
    from lib.nodeclass.dedupe import khop_features_batched
    from lib.nodeclass import build_node_bags as bnb

    nxt = NEExT(log_level="ERROR")
    coll = dense_toy_collection(nxt)
    centers = list(range(12))
    bags = bnb(nxt, coll, centers, "y", method="k_hop", k_hop=2, seed=1)
    naive = nxt.compute_node_features(bags.egonets, **FEATURE_KW)

    feats, table = khop_features_batched(
        nxt, coll, centers, "y", k_hop=2, seed=1, feature_kwargs=FEATURE_KW, batch_size=5
    )
    # graph ids differ (renumbered); compare per-center blocks via the tables
    naive_map = bags.table.set_index("node_id")["graph_id"].to_dict()
    batched_map = table.set_index("node_id")["graph_id"].to_dict()
    a_df, b_df = naive.features_df, feats.features_df
    for center in centers:
        a = a_df[a_df.graph_id == naive_map[center]].drop(columns="graph_id").sort_values("node_id").reset_index(drop=True)
        b = b_df[b_df.graph_id == batched_map[center]].drop(columns="graph_id").sort_values("node_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)], check_like=True)
    assert sorted(table.node_id) == sorted(centers)
    assert table.graph_id.is_unique
