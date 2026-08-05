"""Bag labeling invariants on a hand-computable path graph.

Path 0-1-2-3-4-5 with node 5 positive:
  k=1 bags containing 5: centers {4, 5}
  k=2 bags containing 5: centers {3, 4, 5}
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.containment import build_bags
from NEExT import NEExT


def _path_collection(nxt):
    edges = pd.DataFrame({"src_node_id": [0, 1, 2, 3, 4], "dest_node_id": [1, 2, 3, 4, 5]})
    nodes = pd.DataFrame({"node_id": range(6), "label": [0, 0, 0, 0, 0, 1]})
    return nxt.load_single_graph_from_dfs(edges_df=edges, nodes_df=nodes)


def test_containment_labels_k1_and_k2():
    nxt = NEExT(log_level="WARNING")
    collection = _path_collection(nxt)
    for k, expected_centers in [(1, {4, 5}), (2, {3, 4, 5})]:
        bags = build_bags(nxt, _path_collection(nxt), label_column="label", k_hop=k)
        table = bags.table.set_index("center_node")
        assert set(table.index) == set(range(6))
        assert set(table.index[table["y_contains"] == 1]) == expected_centers
        assert set(table.index[table["y_center"] == 1]) == {5}
    assert collection is not None


def test_bag_sizes_match_khop():
    nxt = NEExT(log_level="WARNING")
    bags = build_bags(nxt, _path_collection(nxt), label_column="label", k_hop=1)
    sizes = bags.table.set_index("center_node")["n_nodes"]
    assert sizes[0] == 2 and sizes[1] == 3 and sizes[5] == 2


def test_center_sampling_caps_bag_count():
    nxt = NEExT(log_level="WARNING")
    bags = build_bags(nxt, _path_collection(nxt), label_column="label", k_hop=1, n_centers=3, seed=13)
    assert len(bags.table) == 3
