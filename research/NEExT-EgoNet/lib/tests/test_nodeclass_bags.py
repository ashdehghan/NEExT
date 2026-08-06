"""Exact-centers contract + re-keying + reach stats for lib.nodeclass.bags."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.nodeclass import build_node_bags, egonet_rep_to_node_frame, khop_reach
from NEExT import NEExT

HAS_WALKS = hasattr(NEExT(log_level="WARNING"), "compute_random_walk_egonets")


def _path_collection(nxt, n=12, graph_type="networkx"):
    edges = pd.DataFrame({"src_node_id": range(n - 1), "dest_node_id": range(1, n)})
    nodes = pd.DataFrame({"node_id": range(n), "label": [0] * (n - 1) + [1]})
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes, graph_type=graph_type)


def test_exact_centers_khop():
    nxt = NEExT(log_level="WARNING")
    centers = [0, 3, 7, 11]
    bags = build_node_bags(nxt, _path_collection(nxt), centers, "label", method="k_hop", k_hop=1)
    assert set(bags.table["node_id"]) == set(centers)
    assert len(bags.table) == len(centers)
    sizes = bags.table.set_index("node_id")["n_nodes"]
    assert sizes[0] == 2 and sizes[3] == 3  # path graph k=1


@pytest.mark.skipif(not HAS_WALKS, reason="compute_random_walk_egonets not on this branch")
def test_exact_centers_random_walk():
    nxt = NEExT(log_level="WARNING")
    centers = [0, 5, 11]
    bags = build_node_bags(
        nxt, _path_collection(nxt), centers, "label", method="random_walk", walk_length=5, n_walks=30, min_visits=1
    )
    assert set(bags.table["node_id"]) == set(centers)


def test_rep_rekeying_roundtrip():
    nxt = NEExT(log_level="WARNING")
    centers = [2, 6, 9]
    bags = build_node_bags(nxt, _path_collection(nxt), centers, "label", method="k_hop", k_hop=1)
    rep = pd.DataFrame({"graph_id": bags.table["graph_id"], "feat": bags.table["graph_id"] * 10.0})
    node_rep = egonet_rep_to_node_frame(rep, bags.table)
    assert list(node_rep.columns) == ["node_id", "feat"]
    assert set(node_rep["node_id"]) == set(centers)
    # each center's feature is its own egonet's value
    merged = node_rep.merge(bags.table, on="node_id")
    assert np.allclose(merged["feat"], merged["graph_id"] * 10.0)


def test_khop_reach_matches_bruteforce():
    nxt = NEExT(log_level="WARNING")
    collection = _path_collection(nxt, graph_type="igraph")
    centers = [0, 5, 11]
    reach = khop_reach(collection, centers, k=2)
    # path graph: node 0 reaches {0,1,2}=3; node 5 reaches 5; node 11 reaches 3
    assert reach["median"] == 3.0
    assert reach["max"] == 5
    assert reach["n_centers"] == 3
