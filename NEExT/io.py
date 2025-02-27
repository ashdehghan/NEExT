from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
from pydantic import BaseModel, Field
from collections import defaultdict
from .graph_collection import GraphCollection

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
        node_sample_rate: float = 1.0
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
        if not {'src_node_id', 'dest_node_id'}.issubset(edges_df.columns):
            raise ValueError("edges.csv must contain 'src_node_id' and 'dest_node_id' columns")
        if not {'node_id', 'graph_id'}.issubset(node_graph_df.columns):
            raise ValueError("node_graph_mapping.csv must contain 'node_id' and 'graph_id' columns")

        # Read graph labels if provided
        graph_labels_df = None
        if graph_label_path:
            graph_labels_df = pd.read_csv(graph_label_path)
            if not {'graph_id', 'graph_label'}.issubset(graph_labels_df.columns):
                raise ValueError("graph_labels.csv must contain 'graph_id' and 'graph_label' columns")

        # Read optional feature files
        node_features_df = None
        if node_features_path:
            node_features_df = pd.read_csv(node_features_path)
            if 'node_id' not in node_features_df.columns:
                raise ValueError("node_features.csv must contain 'node_id' column")

        edge_features_df = None
        if edge_features_path:
            edge_features_df = pd.read_csv(edge_features_path)
            if not {'src_node_id', 'dest_node_id'}.issubset(edge_features_df.columns):
                raise ValueError("edge_features.csv must contain 'src_node_id' and 'dest_node_id' columns")

        # Validate node_sample_rate
        if not 0.0 < node_sample_rate <= 1.0:
            raise ValueError("node_sample_rate must be between 0 and 1")

        # Organize data by graph
        graphs_data = self._organize_graph_data(
            edges_df,
            node_graph_df,
            node_features_df,
            edge_features_df,
            graph_labels_df
        )

        # Create GraphCollection and add graphs
        collection = GraphCollection(graph_type=graph_type, node_sample_rate=node_sample_rate)
        collection.add_graphs(
            graph_data_list=graphs_data, 
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate
        )
        
        return collection

    def _organize_graph_data(
        self,
        edges_df: pd.DataFrame,
        node_graph_df: pd.DataFrame,
        node_features_df: Optional[pd.DataFrame],
        edge_features_df: Optional[pd.DataFrame],
        graph_labels_df: Optional[pd.DataFrame]
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
        # Group nodes by graph_id
        graph_nodes = defaultdict(list)
        for _, row in node_graph_df.iterrows():
            graph_nodes[row['graph_id']].append(row['node_id'])

        # Create graph labels dictionary if available
        graph_labels = {}
        if graph_labels_df is not None:
            graph_labels = dict(zip(graph_labels_df['graph_id'], graph_labels_df['graph_label']))

        # Create graph data dictionaries
        graphs_data = []
        for graph_id, nodes in graph_nodes.items():
            # Get edges for this graph
            graph_edges = edges_df[
                (edges_df['src_node_id'].isin(nodes)) &
                (edges_df['dest_node_id'].isin(nodes))
            ]
            edges = list(zip(graph_edges['src_node_id'], graph_edges['dest_node_id']))

            # Initialize graph data
            graph_data = {
                "graph_id": graph_id,
                "graph_label": graph_labels.get(graph_id),
                "nodes": nodes,
                "edges": edges,
                "node_attributes": {},
                "edge_attributes": {}
            }

            # Add node features if available
            if node_features_df is not None:
                node_features = node_features_df[node_features_df['node_id'].isin(nodes)]
                feature_cols = [col for col in node_features.columns if col != 'node_id']
                for _, row in node_features.iterrows():
                    graph_data["node_attributes"][row['node_id']] = {
                        col: row[col] for col in feature_cols
                    }

            # Add edge features if available
            if edge_features_df is not None:
                edge_features = edge_features_df[
                    (edge_features_df['src_node_id'].isin(nodes)) &
                    (edge_features_df['dest_node_id'].isin(nodes))
                ]
                feature_cols = [col for col in edge_features.columns 
                              if col not in ['src_node_id', 'dest_node_id']]
                for _, row in edge_features.iterrows():
                    edge_key = (row['src_node_id'], row['dest_node_id'])
                    graph_data["edge_attributes"][edge_key] = {
                        col: row[col] for col in feature_cols
                    }

            graphs_data.append(graph_data)

        return graphs_data
