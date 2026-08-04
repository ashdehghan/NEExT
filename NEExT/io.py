from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from NEExT.collections import GraphCollection


class GraphIO:
    """
    Input/Output class for reading and writing graph data.

    This class provides methods to read graph data from various file formats
    and create a GraphCollection instance.
    """

    def __init__(self, logger=None):
        """Initialize GraphIO with optional logger."""
        self.logger = logger

    def read_from_csv(
        self,
        edges_path: Union[str, Path],
        node_graph_mapping_path: Union[str, Path],
        graph_label_path: Optional[Union[str, Path]] = None,
        node_features_path: Optional[Union[str, Path]] = None,
        edge_features_path: Optional[Union[str, Path]] = None,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0,
    ) -> GraphCollection:
        """
        Read graph data from CSV files and create a GraphCollection.

        Args:
            edges_path: Path to edges CSV file (src_node_id, dest_node_id)
            node_graph_mapping_path: Path to node-graph mapping CSV file (node_id, graph_id)
            graph_label_path: Optional path to graph labels CSV file (graph_id, graph_label)
            node_features_path: Optional path to node features CSV file
            edge_features_path: Optional path to edge features CSV file
            graph_type: Backend to use ("networkx" or "igraph"). Defaults to "networkx"
            reindex_nodes: Whether to reindex nodes to start from 0 (default: True)
            filter_largest_component: Whether to keep only the largest connected
                                    component of each graph (default: True)
            node_sample_rate: Rate at which to sample nodes from each graph (default: 1.0).
                            Must be between 0 and 1.

        Returns:
            GraphCollection: Collection of graphs created from the CSV data
        """
        # Read required CSV files
        edges_df = pd.read_csv(edges_path)
        node_graph_df = pd.read_csv(node_graph_mapping_path)

        # Validate required columns
        if not {"src_node_id", "dest_node_id"}.issubset(edges_df.columns):
            raise ValueError("edges.csv must contain 'src_node_id' and 'dest_node_id' columns")
        if not {"node_id", "graph_id"}.issubset(node_graph_df.columns):
            raise ValueError("node_graph_mapping.csv must contain 'node_id' and 'graph_id' columns")

        # Read graph labels if provided
        graph_labels_df = None
        if graph_label_path:
            graph_labels_df = pd.read_csv(graph_label_path)
            if not {"graph_id", "graph_label"}.issubset(graph_labels_df.columns):
                raise ValueError("graph_labels.csv must contain 'graph_id' and 'graph_label' columns")

        # Read optional feature files
        node_features_df = None
        if node_features_path:
            node_features_df = pd.read_csv(node_features_path)
            if "node_id" not in node_features_df.columns:
                raise ValueError("node_features.csv must contain 'node_id' column")

        edge_features_df = None
        if edge_features_path:
            edge_features_df = pd.read_csv(edge_features_path)
            if not {"src_node_id", "dest_node_id"}.issubset(edge_features_df.columns):
                raise ValueError("edge_features.csv must contain 'src_node_id' and 'dest_node_id' columns")

        # Validate node_sample_rate
        if not 0.0 < node_sample_rate <= 1.0:
            raise ValueError("node_sample_rate must be between 0 and 1")

        # Organize data by graph
        graphs_data = self._organize_graph_data(edges_df, node_graph_df, node_features_df, edge_features_df, graph_labels_df)

        # Create GraphCollection and add graphs
        collection = GraphCollection(graph_type=graph_type, node_sample_rate=node_sample_rate)
        collection.add_graphs(
            graph_data_list=graphs_data,
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate,
        )

        return collection

    def load_from_dfs(
        self,
        edges_df: pd.DataFrame,
        node_graph_df: pd.DataFrame,
        graph_labels_df: Optional[pd.DataFrame] = None,
        node_features_df: Optional[pd.DataFrame] = None,
        edge_features_df: Optional[pd.DataFrame] = None,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0,
    ) -> GraphCollection:
        # Validate required columns
        if not {"src_node_id", "dest_node_id"}.issubset(edges_df.columns):
            raise ValueError("edges_df must contain 'src_node_id' and 'dest_node_id' columns")
        if not {"node_id", "graph_id"}.issubset(node_graph_df.columns):
            raise ValueError("node_graph_df must contain 'node_id' and 'graph_id' columns")

        if graph_labels_df is not None:
            if not {"graph_id", "graph_label"}.issubset(graph_labels_df.columns):
                raise ValueError("graph_labels_df must contain 'graph_id' and 'graph_label' columns")

        # Read optional feature files
        if node_features_df is not None:
            if "node_id" not in node_features_df.columns:
                raise ValueError("node_features_df must contain 'node_id' column")

        if edge_features_df is not None:
            if not {"src_node_id", "dest_node_id"}.issubset(edge_features_df.columns):
                raise ValueError("edge_features_df must contain 'src_node_id' and 'dest_node_id' columns")

        # Validate node_sample_rate
        if not 0.0 < node_sample_rate <= 1.0:
            raise ValueError("node_sample_rate must be between 0 and 1")

        # Organize data by graph
        graphs_data = self._organize_graph_data(
            edges_df,
            node_graph_df,
            node_features_df,
            edge_features_df,
            graph_labels_df,
        )

        # Create GraphCollection and add graphs
        collection = GraphCollection(graph_type=graph_type, node_sample_rate=node_sample_rate)
        collection.add_graphs(
            graph_data_list=graphs_data,
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate,
        )

        return collection

    def load_single_graph_from_dfs(
        self,
        edges_df: pd.DataFrame,
        nodes_df: Optional[pd.DataFrame] = None,
        graph_id: Union[int, str] = 0,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0,
    ) -> GraphCollection:
        """
        Load a single graph from DataFrames without a node-graph mapping.

        All nodes belong to one graph. The node set comes from nodes_df when
        provided, otherwise from the unique edge endpoints (isolated nodes
        therefore require nodes_df).

        Args:
            edges_df: Edge list with 'src_node_id' and 'dest_node_id' columns;
                extra columns are treated as edge features
            nodes_df: Optional node table with a unique 'node_id' column; extra
                columns are treated as node features
            graph_id: Identifier assigned to the single graph
            graph_type: Type of graph representation ("networkx" or "igraph")
            reindex_nodes: Whether to reindex nodes starting from 0
            filter_largest_component: Whether to keep only the largest component
            node_sample_rate: Rate at which to sample nodes (1.0 = all nodes)

        Returns:
            GraphCollection containing the single graph
        """
        if not {"src_node_id", "dest_node_id"}.issubset(edges_df.columns):
            raise ValueError("edges_df must contain 'src_node_id' and 'dest_node_id' columns")

        if nodes_df is not None:
            if "node_id" not in nodes_df.columns:
                raise ValueError("nodes_df must contain 'node_id' column")
            if nodes_df.empty:
                raise ValueError("nodes_df must contain at least one node")
            if nodes_df["node_id"].duplicated().any():
                raise ValueError("nodes_df must contain unique node_id values")
            node_ids = list(nodes_df["node_id"])
            known_node_ids = set(node_ids)
            unknown_endpoints = (set(edges_df["src_node_id"]) | set(edges_df["dest_node_id"])) - known_node_ids
            if unknown_endpoints:
                raise ValueError("edges_df contains endpoints that are not present in nodes_df")
        else:
            node_ids = list(dict.fromkeys(pd.concat([edges_df["src_node_id"], edges_df["dest_node_id"]])))
            if not node_ids:
                raise ValueError("edges_df must contain at least one edge when nodes_df is not provided")

        node_graph_df = pd.DataFrame({"node_id": node_ids, "graph_id": graph_id})

        node_features_df = None
        if nodes_df is not None:
            node_feature_columns = [column for column in nodes_df.columns if column != "node_id"]
            if node_feature_columns:
                node_features_df = nodes_df.loc[:, ["node_id"] + node_feature_columns].copy()

        edge_features_df = None
        edge_feature_columns = [column for column in edges_df.columns if column not in {"src_node_id", "dest_node_id"}]
        if edge_feature_columns:
            edge_features_df = edges_df.loc[:, ["src_node_id", "dest_node_id"] + edge_feature_columns].copy()

        return self.load_from_dfs(
            edges_df=edges_df.loc[:, ["src_node_id", "dest_node_id"]],
            node_graph_df=node_graph_df,
            node_features_df=node_features_df,
            edge_features_df=edge_features_df,
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate,
        )

    def _organize_graph_data(
        self,
        edges_df: pd.DataFrame,
        node_graph_df: pd.DataFrame,
        node_features_df: Optional[pd.DataFrame],
        edge_features_df: Optional[pd.DataFrame],
        graph_labels_df: Optional[pd.DataFrame],
    ) -> List[Dict]:
        """
        Organizes the data from DataFrames into a list of graph dictionaries.

        Args:
            edges_df (pd.DataFrame): DataFrame containing edge information
            node_graph_df (pd.DataFrame): DataFrame containing node-to-graph mapping
            node_features_df (Optional[pd.DataFrame]): DataFrame containing node features
            edge_features_df (Optional[pd.DataFrame]): DataFrame containing edge features
            graph_labels_df (Optional[pd.DataFrame]): DataFrame containing graph labels

        Returns:
            List[Dict]: List of dictionaries containing organized graph data
        """
        # Group nodes by graph_id (preserves per-graph node order; O(n)).
        graph_nodes = {graph_id: list(group["node_id"]) for graph_id, group in node_graph_df.groupby("graph_id", sort=False)}

        # Create graph labels dictionary if available
        graph_labels = {}
        if graph_labels_df is not None:
            graph_labels = dict(zip(graph_labels_df["graph_id"], graph_labels_df["graph_label"]))

        # When the inputs carry a 'graph_id' column, scope edges/features per graph by that column
        # (correct even when node IDs are only unique *within* a graph, and O(n)). When 'graph_id' is
        # absent, fall back to node-ID membership (preserves behaviour for globally-unique-ID inputs).
        edges_by_graph = {gid: group for gid, group in edges_df.groupby("graph_id", sort=False)} if "graph_id" in edges_df.columns else None
        node_features_by_graph = (
            {gid: group for gid, group in node_features_df.groupby("graph_id", sort=False)}
            if node_features_df is not None and "graph_id" in node_features_df.columns
            else None
        )
        edge_features_by_graph = (
            {gid: group for gid, group in edge_features_df.groupby("graph_id", sort=False)}
            if edge_features_df is not None and "graph_id" in edge_features_df.columns
            else None
        )

        # For the no-graph_id fallback with globally-unique node IDs (the
        # documented contract for that input shape), a single node->graph map
        # replaces the per-graph full-frame isin() scans; groupby preserves the
        # original row order within each graph, matching the isin() filters.
        # Non-unique IDs keep the legacy per-graph scan behaviour.
        node_to_graph = None
        if node_graph_df["node_id"].is_unique:
            node_to_graph = dict(zip(node_graph_df["node_id"], node_graph_df["graph_id"]))
        if edges_by_graph is None and node_to_graph is not None:
            src_graph = edges_df["src_node_id"].map(node_to_graph)
            dst_graph = edges_df["dest_node_id"].map(node_to_graph)
            same_graph = (src_graph == dst_graph) & src_graph.notna()
            edges_by_graph = {gid: group for gid, group in edges_df[same_graph].groupby(src_graph[same_graph], sort=False)}
        if node_features_by_graph is None and node_features_df is not None and node_to_graph is not None:
            nf_graph = node_features_df["node_id"].map(node_to_graph)
            nf_mask = nf_graph.notna()
            node_features_by_graph = {gid: group for gid, group in node_features_df[nf_mask].groupby(nf_graph[nf_mask], sort=False)}
        if edge_features_by_graph is None and edge_features_df is not None and node_to_graph is not None:
            ef_src = edge_features_df["src_node_id"].map(node_to_graph)
            ef_dst = edge_features_df["dest_node_id"].map(node_to_graph)
            ef_mask = (ef_src == ef_dst) & ef_src.notna()
            edge_features_by_graph = {gid: group for gid, group in edge_features_df[ef_mask].groupby(ef_src[ef_mask], sort=False)}

        # Create graph data dictionaries
        graphs_data = []
        for graph_id, nodes in graph_nodes.items():
            node_set = set(nodes)

            # Get edges for this graph
            if edges_by_graph is not None:
                graph_edges = edges_by_graph.get(graph_id, edges_df.iloc[0:0])
            else:
                graph_edges = edges_df[(edges_df["src_node_id"].isin(node_set)) & (edges_df["dest_node_id"].isin(node_set))]
            edges = list(zip(graph_edges["src_node_id"], graph_edges["dest_node_id"]))

            # Initialize graph data
            graph_data = {
                "graph_id": graph_id,
                "graph_label": graph_labels.get(graph_id),
                "nodes": nodes,
                "edges": edges,
                "node_attributes": {},
                "edge_attributes": {},
            }

            # Add node features if available
            if node_features_df is not None:
                if node_features_by_graph is not None:
                    node_features = node_features_by_graph.get(graph_id, node_features_df.iloc[0:0])
                else:
                    node_features = node_features_df[node_features_df["node_id"].isin(node_set)]
                feature_cols = [col for col in node_features.columns if col not in ("node_id", "graph_id")]
                # Row-wise build over the frame's common-dtype matrix: same cell
                # types iterrows() produced (it also reads frame.values), without
                # the per-row Series construction.
                cols = list(node_features.columns)
                nid_idx = cols.index("node_id")
                feature_idx = [cols.index(col) for col in feature_cols]
                node_attr_dst = graph_data["node_attributes"]
                for row_values in node_features.to_numpy():
                    node_attr_dst[row_values[nid_idx]] = dict(zip(feature_cols, row_values[feature_idx]))

            # Add edge features if available
            if edge_features_df is not None:
                if edge_features_by_graph is not None:
                    edge_features = edge_features_by_graph.get(graph_id, edge_features_df.iloc[0:0])
                else:
                    edge_features = edge_features_df[
                        (edge_features_df["src_node_id"].isin(node_set)) & (edge_features_df["dest_node_id"].isin(node_set))
                    ]
                feature_cols = [col for col in edge_features.columns if col not in ("src_node_id", "dest_node_id", "graph_id")]
                cols = list(edge_features.columns)
                src_idx = cols.index("src_node_id")
                dst_idx = cols.index("dest_node_id")
                feature_idx = [cols.index(col) for col in feature_cols]
                edge_attr_dst = graph_data["edge_attributes"]
                for row_values in edge_features.to_numpy():
                    edge_key = (row_values[src_idx], row_values[dst_idx])
                    edge_attr_dst[edge_key] = dict(zip(feature_cols, row_values[feature_idx]))

            graphs_data.append(graph_data)

        return graphs_data

    def load_from_networkx(
        self,
        nx_graphs: List,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0,
    ) -> GraphCollection:
        """
        Create a GraphCollection from a list of NetworkX graphs.

        Args:
            nx_graphs (List): List of NetworkX graph objects
            graph_type (str): Backend to use ("networkx" or "igraph"). Defaults to "networkx"
            reindex_nodes (bool): Whether to reindex nodes to start from 0 (default: True)
            filter_largest_component (bool): Whether to keep only the largest connected
                                           component of each graph (default: True)
            node_sample_rate (float): Rate at which to sample nodes from each graph (default: 1.0).
                                    Must be between 0 and 1.

        Returns:
            GraphCollection: Collection of graphs created from the NetworkX graphs
        """
        import networkx as nx

        # Validate inputs
        if not all(isinstance(g, nx.Graph) for g in nx_graphs):
            raise ValueError("All items in nx_graphs must be NetworkX Graph objects")

        if not 0.0 < node_sample_rate <= 1.0:
            raise ValueError("node_sample_rate must be between 0 and 1")

        # Create GraphCollection and add graphs directly
        collection = GraphCollection(graph_type=graph_type, node_sample_rate=node_sample_rate)
        collection.add_graphs(
            graph_data_list=nx_graphs,
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate,
        )

        return collection
