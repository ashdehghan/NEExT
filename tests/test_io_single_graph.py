"""Offline tests for loading a single graph without a node-graph mapping."""

import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.collections import EgonetCollection


def path_graph_edges(node_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src_node_id": list(range(node_count - 1)),
            "dest_node_id": list(range(1, node_count)),
        }
    )


def test_load_single_graph_edges_only():
    """Edges alone load into a one-graph collection with inferred nodes."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    collection = nxt.load_single_graph_from_dfs(edges_df=path_graph_edges(8))

    assert len(collection.graphs) == 1
    graph = collection.graphs[0]
    assert len(graph.nodes) == 8
    assert len(graph.edges) == 7
    assert graph.source_to_internal_node_id is not None
    assert set(graph.source_to_internal_node_id) == set(range(8))


def test_load_single_graph_with_node_and_edge_attributes():
    """Extra nodes/edges columns become node and edge attributes."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    edges_df = path_graph_edges(4)
    edges_df["weight"] = [0.1, 0.2, 0.3]
    nodes_df = pd.DataFrame({"node_id": [0, 1, 2, 3], "role": [1, 0, 1, 0]})

    collection = nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df)

    graph = collection.graphs[0]
    assert all("role" in attributes for attributes in graph.node_attributes.values())
    assert all("weight" in attributes for attributes in graph.edge_attributes.values())


def test_load_single_graph_isolated_node_requires_nodes_df():
    """A nodes table adds isolated nodes that edges alone cannot describe."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    nodes_df = pd.DataFrame({"node_id": [0, 1, 2, 3, 4]})
    collection = nxt.load_single_graph_from_dfs(
        edges_df=path_graph_edges(4),
        nodes_df=nodes_df,
        filter_largest_component=False,
    )

    assert len(collection.graphs[0].nodes) == 5


def test_load_single_graph_validation_errors():
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    with pytest.raises(ValueError, match="src_node_id"):
        nxt.load_single_graph_from_dfs(edges_df=pd.DataFrame({"a": [1], "b": [2]}))

    with pytest.raises(ValueError, match="node_id"):
        nxt.load_single_graph_from_dfs(
            edges_df=path_graph_edges(3),
            nodes_df=pd.DataFrame({"name": ["a"]}),
        )

    with pytest.raises(ValueError, match="unique node_id"):
        nxt.load_single_graph_from_dfs(
            edges_df=path_graph_edges(3),
            nodes_df=pd.DataFrame({"node_id": [0, 1, 1, 2]}),
        )

    with pytest.raises(ValueError, match="not present in nodes_df"):
        nxt.load_single_graph_from_dfs(
            edges_df=path_graph_edges(4),
            nodes_df=pd.DataFrame({"node_id": [0, 1]}),
        )


def test_single_graph_to_egonet_collection():
    """Loader output feeds k-hop egonet decomposition end-to-end."""
    nxt = NEExT()
    nxt.set_log_level("WARNING")

    collection = nxt.load_single_graph_from_dfs(edges_df=path_graph_edges(6))
    egonets = EgonetCollection(graph_type=collection.graph_type)
    egonets.compute_k_hop_egonets(collection, k_hop=1)

    assert len(egonets.graphs) == 6
    assert set(egonets.egonet_to_graph_node_mapping) == set(range(6))
