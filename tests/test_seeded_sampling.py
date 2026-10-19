"""Loader node sampling and Leiden ego-nets are reproducible for a given seed."""

import networkx as nx
import pandas as pd

from NEExT import NEExT


def _edges_df():
    G = nx.connected_caveman_graph(6, 8)
    return pd.DataFrame(list(G.edges()), columns=["src_node_id", "dest_node_id"])


def test_loader_node_sampling_is_seeded():
    nxt = NEExT(log_level="WARNING")
    draws = [
        nxt.load_single_graph_from_dfs(edges_df=_edges_df(), node_sample_rate=0.3, random_seed=seed).graphs[0].sampled_nodes for seed in (5, 5, 6)
    ]
    assert draws[0] == draws[1]
    assert draws[0] != draws[2]


def test_leiden_egonets_are_seeded():
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(edges_df=_edges_df(), graph_type="igraph")
    runs = [nxt.compute_leiden_egonets(collection, resolution=1.0, random_seed=3) for _ in range(2)]
    for a, b in zip(runs[0].graphs, runs[1].graphs):
        assert sorted(a.node_mapping) == sorted(b.node_mapping)
