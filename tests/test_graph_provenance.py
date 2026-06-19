from NEExT.collections import EgonetCollection, GraphCollection
from NEExT.features import StructuralNodeFeatures
from NEExT.graphs import Graph


def test_reindex_nodes_records_source_to_internal_mapping():
    graph = Graph(graph_id=7, nodes=[30, 10, 20], edges=[(30, 10), (10, 20)], graph_type="networkx")
    graph.initialize_graph()

    reindexed = graph.reindex_nodes()

    assert reindexed.nodes == [0, 1, 2]
    assert reindexed.edges == [(2, 0), (0, 1)]
    assert reindexed.source_graph_id == 7
    assert reindexed.source_to_internal_node_id == {10: 0, 20: 1, 30: 2}
    assert reindexed.dropped_source_node_ids == []


def test_largest_component_filter_records_dropped_source_nodes():
    graph = Graph(graph_id=1, nodes=[1, 2, 3, 4, 5], edges=[(1, 2), (2, 3), (4, 5)], graph_type="networkx")
    graph.initialize_graph()

    filtered = graph.filter_largest_component()

    assert filtered.nodes == [0, 1, 2]
    assert filtered.source_to_internal_node_id == {1: 0, 2: 1, 3: 2}
    assert filtered.dropped_source_node_ids == [4, 5]
    assert filtered.drop_reasons_by_source_node_id == {
        4: "not_in_largest_connected_component",
        5: "not_in_largest_connected_component",
    }


def test_graph_collection_exports_complete_node_mapping_records():
    collection = GraphCollection(graph_type="networkx")
    collection.add_graphs(
        [
            {
                "graph_id": "g1",
                "nodes": [10, 20, 30],
                "edges": [(10, 20)],
            }
        ],
        reindex_nodes=True,
        filter_largest_component=True,
    )

    mapping = collection.export_node_mapping_records()

    assert list(mapping.columns) == [
        "source_graph_id",
        "source_node_id",
        "internal_graph_id",
        "internal_node_id",
        "included",
        "drop_reason",
    ]
    assert mapping.to_dict(orient="records") == [
        {
            "source_graph_id": "g1",
            "source_node_id": 10,
            "internal_graph_id": "g1",
            "internal_node_id": 0,
            "included": True,
            "drop_reason": None,
        },
        {
            "source_graph_id": "g1",
            "source_node_id": 20,
            "internal_graph_id": "g1",
            "internal_node_id": 1,
            "included": True,
            "drop_reason": None,
        },
        {
            "source_graph_id": "g1",
            "source_node_id": 30,
            "internal_graph_id": None,
            "internal_node_id": None,
            "included": False,
            "drop_reason": "not_in_largest_connected_component",
        },
    ]


def test_feature_output_contract_remains_node_id_graph_id_then_features():
    collection = GraphCollection(graph_type="networkx")
    collection.add_graphs(
        [{"graph_id": 1, "nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]}],
        reindex_nodes=True,
        filter_largest_component=True,
    )

    features = StructuralNodeFeatures(
        collection,
        ["degree_centrality"],
        feature_vector_length=2,
        normalize_features=False,
        show_progress=False,
    ).compute()

    assert list(features.features_df.columns) == ["node_id", "graph_id", "degree_centrality_0", "degree_centrality_1"]
    assert set(features.features_df["node_id"]) == {0, 1, 2}


def test_egonet_node_mapping_behavior_is_unchanged():
    collection = GraphCollection(graph_type="networkx")
    collection.add_graphs(
        [{"graph_id": 1, "nodes": [10, 20, 30], "edges": [(10, 20), (20, 30)]}],
        reindex_nodes=True,
        filter_largest_component=True,
    )
    egonets = EgonetCollection(graph_type="networkx")
    egonets.compute_k_hop_egonets(collection, k_hop=1, sample_fraction=1.0, random_seed=13)

    assert egonets.egonet_to_graph_node_mapping
    for egonet in egonets.graphs:
        assert egonet.node_mapping
        assert egonet.original_graph_id == 1


def test_load_from_dfs_scopes_edges_by_graph_id_without_contamination():
    """Prepared datasets number nodes locally per graph (every graph starts at 0), so node IDs are
    only unique *within* a graph. ``load_from_dfs`` must scope each graph's edges by the ``graph_id``
    column, not by node-ID membership — otherwise graphs steal each other's edges."""
    import pandas as pd

    from NEExT.io import GraphIO

    # Two graphs, BOTH using local node ids {0, 1, 2}, but different edges.
    node_graph_df = pd.DataFrame({"node_id": [0, 1, 2, 0, 1, 2], "graph_id": [0, 0, 0, 1, 1, 1]})
    edges_df = pd.DataFrame(
        {"graph_id": [0, 0, 1], "src_node_id": [0, 1, 0], "dest_node_id": [1, 2, 2]}
    )

    collection = GraphIO().load_from_dfs(
        edges_df=edges_df,
        node_graph_df=node_graph_df,
        graph_type="networkx",
        reindex_nodes=False,
        filter_largest_component=False,
    )
    by_id = {graph.graph_id: graph for graph in collection.graphs}
    as_sets = lambda graph: {frozenset(edge) for edge in graph.edges}
    # Each graph gets exactly its own edges — no leakage in either direction.
    assert as_sets(by_id[0]) == {frozenset((0, 1)), frozenset((1, 2))}
    assert as_sets(by_id[1]) == {frozenset((0, 2))}

    # Contrast: without a graph_id column, the legacy node-ID-membership fallback contaminates —
    # graph 0 (nodes 0,1,2) wrongly pulls in graph 1's (0,2) edge. This documents the bug the
    # graph_id-aware path fixes, and guards the backward-compatible fallback.
    legacy = GraphIO().load_from_dfs(
        edges_df=edges_df.drop(columns=["graph_id"]),
        node_graph_df=node_graph_df,
        graph_type="networkx",
        reindex_nodes=False,
        filter_largest_component=False,
    )
    legacy_graph0 = {graph.graph_id: graph for graph in legacy.graphs}[0]
    assert frozenset((0, 2)) in {frozenset(edge) for edge in legacy_graph0.edges}
