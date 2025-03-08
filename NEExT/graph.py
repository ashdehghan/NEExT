from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Union, Optional, Set, Tuple
import networkx as nx
import igraph as ig
import random

class Graph(BaseModel):
    """
    A graph representation class that supports both NetworkX and iGraph backends.
    
    This class provides a unified interface for working with graphs, allowing users
    to choose between NetworkX and iGraph as the underlying graph implementation.
    It uses Pydantic for data validation and type checking.

    Attributes:
        graph_id (int): Unique identifier for the graph
        graph_label (Optional[int]): Label for the graph (e.g., for classification)
        nodes (List[int]): List of node identifiers
        edges (List[tuple]): List of edge tuples (source, target)
        node_attributes (Dict[int, Dict]): Dictionary mapping nodes to their attributes
        edge_attributes (Dict[tuple, Dict]): Dictionary mapping edges to their attributes
        graph_type (str): Backend graph library to use ("networkx" or "igraph")
        G (Union[nx.Graph, ig.Graph]): The actual graph instance (set automatically)
        vertex_map (Optional[Dict[int, int]]): Mapping from internal to original node IDs
        reverse_vertex_map (Optional[Dict[int, int]]): Mapping from original to internal node IDs
        sampled_nodes (Optional[List[int]]): List of sampled nodes for feature computation
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types like nx.Graph
    }

    graph_id: int
    graph_label: Optional[int] = None
    nodes: List[int]
    edges: List[tuple[int, int]]
    node_attributes: Dict[int, Dict[str, Union[float, int, str]]] = Field(default_factory=dict)
    edge_attributes: Dict[tuple[int, int], Dict[str, Union[float, int, str]]] = Field(default_factory=dict)
    graph_type: Literal["networkx", "igraph"] = "networkx"
    G: Optional[Union[nx.Graph, ig.Graph]] = Field(default=None, exclude=True)
    sampled_nodes: Optional[List[int]] = Field(default=None, exclude=True)

    def initialize_graph(self):
        """Initialize the graph with the specified backend."""
        if self.graph_type == "networkx":
            self.G = nx.Graph()
            self.G.add_nodes_from(self.nodes)
            self.G.add_edges_from(self.edges)
            
            # Add attributes
            for node, attrs in self.node_attributes.items():
                nx.set_node_attributes(self.G, {node: attrs})
            for edge, attrs in self.edge_attributes.items():
                nx.set_edge_attributes(self.G, {edge: attrs})

        elif self.graph_type == "igraph":
            # Ensure nodes are consecutive integers starting from 0
            node_set = set(self.nodes)
            if not all(isinstance(n, int) for n in node_set):
                raise ValueError("All node IDs must be integers")
            
            min_node = min(node_set)
            max_node = max(node_set)
            if min_node < 0:
                raise ValueError("Node IDs must be non-negative")
            
            # Create graph with correct number of vertices
            self.G = ig.Graph(n=max_node + 1)
            
            # Add edges ensuring all node IDs are valid
            if self.edges:
                edge_list = [(src, dst) for src, dst in self.edges]
                self.G.add_edges(edge_list)

    def get_node_mapping(self):
        """Return the mapping between original and internal node IDs."""
        if self.graph_type == "igraph":
            return self.vertex_map
        return {node: node for node in self.nodes}  # Identity mapping for networkx

    def get_original_id(self, internal_id):
        """Convert internal node ID back to original ID."""
        if self.graph_type == "igraph":
            return self.reverse_vertex_map[internal_id]
        return internal_id

    def reindex_nodes(self) -> 'Graph':
        """Reindex nodes to be consecutive integers starting from 0."""
        # Create mapping from old to new indices
        unique_nodes = sorted(set(self.nodes))
        node_mapping = {old: new for new, old in enumerate(unique_nodes)}
        
        # Create new edges with mapped node IDs
        new_edges = [(node_mapping[src], node_mapping[dst]) for src, dst in self.edges]
        
        # Map node attributes
        new_node_attrs = {node_mapping[node]: attrs 
                         for node, attrs in self.node_attributes.items()}
        
        # Map edge attributes
        new_edge_attrs = {(node_mapping[src], node_mapping[dst]): attrs 
                         for (src, dst), attrs in self.edge_attributes.items()}
        
        # Create new graph with mapped IDs
        return Graph(
            graph_id=self.graph_id,
            graph_label=self.graph_label,
            nodes=list(range(len(unique_nodes))),  # Consecutive integers from 0
            edges=new_edges,
            node_attributes=new_node_attrs,
            edge_attributes=new_edge_attrs,
            graph_type=self.graph_type
        )
    
    def filter_largest_component(self) -> 'Graph':
        """
        Filter the graph to keep only the largest connected component.
        
        Returns:
            Graph: A new Graph instance containing only the largest connected component
        """
        if self.graph_type == "networkx":
            # Find largest connected component
            if not nx.is_connected(self.G):
                largest_cc = max(nx.connected_components(self.G), key=len)
                subgraph = self.G.subgraph(largest_cc).copy()
            else:
                # Already connected
                return self
                
            # Extract nodes and edges from subgraph
            nodes = list(subgraph.nodes())
            edges = list(subgraph.edges())
            
            # Extract node and edge attributes
            node_attrs = {n: {k: v for k, v in subgraph.nodes[n].items()} 
                         for n in nodes if subgraph.nodes[n]}
            
            edge_attrs = {e: {k: v for k, v in subgraph.edges[e].items()} 
                         for e in edges if subgraph.edges[e]}
            
        else:  # igraph
            # Find largest connected component
            components = self.G.clusters()
            if len(components) > 1:
                largest_cc_idx = components.sizes().index(max(components.sizes()))
                subgraph = self.G.subgraph(components[largest_cc_idx])
            else:
                # Already connected
                return self
                
            # Extract nodes and edges from subgraph
            nodes = [v.index for v in subgraph.vs]
            edges = [(e.source, e.target) for e in subgraph.es]
            
            # Extract node and edge attributes
            node_attrs = {}
            for v in subgraph.vs:
                attrs = {attr: v[attr] for attr in v.attributes() if attr != 'name'}
                if attrs:
                    node_attrs[v.index] = attrs
            
            edge_attrs = {}
            for e in subgraph.es:
                attrs = {attr: e[attr] for attr in e.attributes()}
                if attrs:
                    edge_attrs[(e.source, e.target)] = attrs
        
        # Create new Graph instance
        filtered_graph = Graph(
            graph_id=self.graph_id,
            graph_label=self.graph_label,
            nodes=nodes,
            edges=edges,
            node_attributes=node_attrs,
            edge_attributes=edge_attrs,
            graph_type=self.graph_type
        )
        
        # Reindex nodes to be consecutive
        return filtered_graph.reindex_nodes()

    def get_graph_info(self):
        """
        Returns basic information about the graph.

        Returns:
            dict: A dictionary containing:
                - graph_id: The unique identifier of the graph
                - graph_type: The backend being used ("networkx" or "igraph")
                - num_nodes: Total number of nodes
                - num_edges: Total number of edges
                - graph_label: The label of the graph (if any)
        """
        return {
            "graph_id": self.graph_id,
            "graph_type": self.graph_type,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "graph_label": self.graph_label,
        }

    def sample_nodes(self, sample_rate: float = 1.0, random_seed: Optional[int] = None) -> List[int]:
        """
        Sample a subset of nodes based on the given sample rate.
        
        Args:
            sample_rate (float): Fraction of nodes to sample (between 0 and 1)
            random_seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            List[int]: List of sampled node IDs
        """
        if sample_rate >= 1.0:
            self.sampled_nodes = self.nodes
            return self.nodes
            
        if random_seed is not None:
            random.seed(random_seed)
            
        num_nodes = len(self.nodes)
        num_samples = max(1, int(num_nodes * sample_rate))  # Ensure at least 1 node is sampled
        self.sampled_nodes = random.sample(self.nodes, num_samples)
        return self.sampled_nodes