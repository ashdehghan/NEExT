from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set, Union, Literal
import numpy as np
from .graph import Graph
import pandas as pd
import random

class GraphCollection(BaseModel):
    """
    A collection class that manages multiple Graph instances.
    
    This class provides functionality to store and manage multiple graphs,
    supporting both NetworkX and iGraph backends.

    Attributes:
        graphs (List[Graph]): List of Graph instances
        graph_type (str): Backend graph library to use ("networkx" or "igraph")
        graph_id_node_array (Optional[List[int]]): Array mapping each node to its graph ID
        node_sample_rate (float): Rate at which to sample nodes from each graph (default: 1.0)
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like numpy arrays
    }

    graphs: List[Graph] = Field(default_factory=list)
    graph_type: Literal["networkx", "igraph"] = "networkx"
    graph_id_node_array: Optional[List[int]] = Field(default=None, exclude=True)
    node_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    def add_graphs(
        self,
        graph_data_list: List[Dict],
        graph_type: Optional[Literal["networkx", "igraph"]] = None,
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: Optional[float] = None
    ) -> None:
        """
        Creates Graph instances from a list of dictionaries containing graph data.

        Args:
            graph_data_list (List[Dict]): List of dictionaries containing graph data.
                Each dictionary should have:
                - graph_id (required): Unique identifier for the graph
                - graph_label (optional): Label for the graph
                - nodes (required): List of node IDs
                - edges (required): List of (source, target) edge tuples
                - node_attributes (optional): Dictionary of node attributes
                - edge_attributes (optional): Dictionary of edge attributes
            graph_type (str, optional): Backend to use ("networkx" or "igraph").
                                      If None, uses the collection's default.
            reindex_nodes (bool): Whether to reindex nodes to start from 0 (default: True)
            filter_largest_component (bool): Whether to keep only the largest connected 
                                           component of each graph (default: True)
            node_sample_rate (float, optional): Rate at which to sample nodes from each graph.
                                              If None, uses the collection's default.
                                              Must be between 0 and 1.
        """
        # Use instance graph_type if none provided
        graph_type = graph_type or self.graph_type
        
        # Use instance node_sample_rate if none provided
        node_sample_rate = node_sample_rate if node_sample_rate is not None else self.node_sample_rate
        
        # Clear existing graph_id_node_array
        self.graph_id_node_array = []

        for graph_data in graph_data_list:
            # Extract required fields
            graph_id = graph_data["graph_id"]
            nodes = graph_data["nodes"]
            edges = graph_data["edges"]

            # Sample nodes if rate is less than 1
            if node_sample_rate < 1.0:
                # Calculate number of nodes to sample
                num_nodes = len(nodes)
                num_sample = max(1, int(num_nodes * node_sample_rate))
                
                # Randomly sample nodes
                sampled_nodes = random.sample(nodes, num_sample)
                sampled_nodes_set = set(sampled_nodes)
                
                # Filter edges to only include sampled nodes
                edges = [(src, dst) for src, dst in edges 
                        if src in sampled_nodes_set and dst in sampled_nodes_set]
                
                # Update nodes list
                nodes = sampled_nodes
                
                # Filter node and edge attributes
                node_attributes = {k: v for k, v in graph_data.get("node_attributes", {}).items()
                                 if k in sampled_nodes_set}
                edge_attributes = {k: v for k, v in graph_data.get("edge_attributes", {}).items()
                                 if k[0] in sampled_nodes_set and k[1] in sampled_nodes_set}
            else:
                # Use original attributes
                node_attributes = graph_data.get("node_attributes", {})
                edge_attributes = graph_data.get("edge_attributes", {})

            graph_label = graph_data.get("graph_label")

            # Create Graph instance
            graph = Graph(
                graph_id=graph_id,
                graph_label=graph_label,
                nodes=nodes,
                edges=edges,
                node_attributes=node_attributes,
                edge_attributes=edge_attributes,
                graph_type=graph_type
            )
            
            # Initialize the graph backend
            graph.initialize_graph()
            
            # Apply processing steps if requested
            if filter_largest_component:
                graph = graph.filter_largest_component()
                # Re-initialize after filtering
                graph.initialize_graph()
            elif reindex_nodes:  # Only reindex if not already done by filter_largest_component
                graph = graph.reindex_nodes()
                # Re-initialize after reindexing
                graph.initialize_graph()

            self.graphs.append(graph)
            
            # Update graph_id_node_array with this graph's nodes
            self.graph_id_node_array.extend([graph_id] * len(graph.nodes))

    def get_graph_by_id(self, graph_id: int) -> Optional[Graph]:
        """
        Retrieves a graph from the collection by its ID.

        Args:
            graph_id (int): The ID of the graph to retrieve

        Returns:
            Optional[Graph]: The Graph instance if found, None otherwise
        """
        for graph in self.graphs:
            if graph.graph_id == graph_id:
                return graph
        return None
    
    def get_total_node_count(self) -> int:
        """
        Get the total number of nodes across all graphs in the collection.
        
        Returns:
            int: Total number of nodes
        """
        return len(self.graph_id_node_array) if self.graph_id_node_array else sum(len(g.nodes) for g in self.graphs)
    
    def rebuild_graph_id_node_array(self) -> None:
        """
        Rebuild the graph_id_node_array from the current graphs.
        This is useful after modifying the graphs in the collection.
        """
        self.graph_id_node_array = []
        for graph in self.graphs:
            self.graph_id_node_array.extend([graph.graph_id] * len(graph.nodes))

    def describe(self) -> dict:
        """
        Get basic information about the graph collection.
        
        Returns:
            dict: Dictionary containing collection information
                - num_graphs: Number of graphs in collection
                - graph_type: Backend being used
                - has_labels: Whether graphs have labels
        """
        info = {
            "num_graphs": len(self.graphs),
            "graph_type": self.graph_type,
            "has_labels": any(g.graph_label is not None for g in self.graphs)
        }
        return info

    def get_labels(self) -> pd.DataFrame:
        """
        Get a DataFrame with graph IDs and their labels.
        
        Returns:
            pd.DataFrame: DataFrame with graph_id and label columns
        """
        graph_ids = []
        graph_labels = []
        
        for graph in self.graphs:
            if graph.graph_label is not None:
                graph_ids.append(graph.graph_id)
                graph_labels.append(graph.graph_label)
        
        return pd.DataFrame({
            "graph_id": graph_ids,
            "label": graph_labels
        })
