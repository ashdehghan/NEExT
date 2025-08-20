"""
Random walk neighborhood sampling strategy.

This module implements random walk with restart sampling for ego-centric
neighborhoods. This technique preserves local graph structure while
providing computational efficiency for large graphs.
"""

from typing import List, Union, Set
import numpy as np
import networkx as nx
from .base import NodeSampler

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class RandomWalkSampler(NodeSampler):
    """
    Random walk with restart neighborhood sampler.
    
    Performs multiple random walks starting from the ego node, with a probability
    of restarting at the ego node at each step. This maintains ego-centric
    neighborhoods while exploring diverse structural patterns.
    
    References:
        - GraphSAGE: Inductive Representation Learning on Large Graphs
        - Personal PageRank and random walks with restart
    """
    
    def sample_neighborhood(self, 
                          graph: Union[nx.Graph, 'ig.Graph'], 
                          ego_node: int,
                          walk_length: int = 10,
                          num_walks: int = 5,
                          restart_prob: float = 0.15,
                          max_nodes: int = None,
                          random_seed: int = None,
                          **kwargs) -> List[int]:
        """
        Sample neighborhood using random walks with restart.
        
        Args:
            graph: NetworkX or iGraph graph object
            ego_node: The central node for neighborhood sampling
            walk_length: Maximum length of each random walk
            num_walks: Number of random walks to perform
            restart_prob: Probability of restarting at ego node at each step
            max_nodes: Maximum number of nodes to include (if None, no limit)
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of node IDs in the sampled neighborhood (including ego_node)
        """
        self.validate_graph(graph)
        self.validate_ego_node(graph, ego_node)
        
        # Validate parameters
        if walk_length < 1:
            raise ValueError("walk_length must be positive")
        if num_walks < 1:
            raise ValueError("num_walks must be positive")
        if not 0 <= restart_prob <= 1:
            raise ValueError("restart_prob must be between 0 and 1")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Track visit frequencies for intelligent truncation
        visit_counts = {ego_node: 0}
        
        # Perform random walks
        for walk_idx in range(num_walks):
            current_node = ego_node
            visit_counts[current_node] = visit_counts.get(current_node, 0) + 1
            
            for step in range(walk_length):
                # Check for restart
                if np.random.random() < restart_prob:
                    current_node = ego_node
                    visit_counts[current_node] = visit_counts.get(current_node, 0) + 1
                    continue
                
                # Get neighbors of current node
                neighbors = self._get_neighbors(graph, current_node)
                
                if not neighbors:
                    # Dead end - restart at ego node
                    current_node = ego_node
                    visit_counts[current_node] = visit_counts.get(current_node, 0) + 1
                    continue
                
                # Randomly select next node
                current_node = np.random.choice(neighbors)
                visit_counts[current_node] = visit_counts.get(current_node, 0) + 1
                
                # Stop early if we've reached max_nodes
                if max_nodes is not None and len(visit_counts) >= max_nodes:
                    break
            
            # Stop early if we've reached max_nodes
            if max_nodes is not None and len(visit_counts) >= max_nodes:
                break
        
        # Intelligent truncation based on visit frequency
        if max_nodes is not None and len(visit_counts) > max_nodes:
            # Sort nodes by visit frequency (descending), breaking ties by node ID for determinism
            sorted_nodes = sorted(visit_counts.items(), key=lambda x: (-x[1], x[0]))
            # Keep top max_nodes most visited nodes, ensuring ego node is always included
            if ego_node not in [node for node, _ in sorted_nodes[:max_nodes]]:
                # Replace least visited non-ego node with ego node
                selected_nodes = [ego_node] + [node for node, _ in sorted_nodes[:max_nodes-1] if node != ego_node]
            else:
                selected_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            neighborhood = sorted(selected_nodes)
        else:
            neighborhood = sorted(list(visit_counts.keys()))
        
        return neighborhood
    
    def _get_neighbors(self, graph: Union[nx.Graph, 'ig.Graph'], node: int) -> List[int]:
        """Get neighbors of a node for both NetworkX and iGraph."""
        if isinstance(graph, nx.Graph):
            return list(graph.neighbors(node))
        elif IGRAPH_AVAILABLE and isinstance(graph, ig.Graph):
            return graph.neighbors(node)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    def adaptive_parameters(self, 
                          graph: Union[nx.Graph, 'ig.Graph'], 
                          ego_node: int,
                          target_size: int = 50) -> dict:
        """
        Automatically determine good parameters for random walk sampling.
        
        Args:
            graph: NetworkX or iGraph graph object
            ego_node: The central node for neighborhood sampling
            target_size: Target neighborhood size
            
        Returns:
            Dictionary with recommended parameters
        """
        self.validate_graph(graph)
        self.validate_ego_node(graph, ego_node)
        
        # Get ego node degree
        if isinstance(graph, nx.Graph):
            ego_degree = graph.degree(ego_node)
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        elif IGRAPH_AVAILABLE and isinstance(graph, ig.Graph):
            ego_degree = graph.degree(ego_node)
            avg_degree = np.mean(graph.degree())
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
        
        # Adaptive parameter calculation
        if ego_degree == 0:
            # Isolated node
            return {
                'walk_length': 1,
                'num_walks': 1,
                'restart_prob': 1.0,
                'max_nodes': 1
            }
        
        # Base parameters
        base_walk_length = max(3, min(10, int(np.log2(target_size))))
        base_num_walks = max(3, min(15, target_size // 5))
        
        # Adjust based on ego degree relative to average
        degree_ratio = ego_degree / max(avg_degree, 1)
        
        if degree_ratio > 2.0:
            # High-degree node - shorter walks, more restarts
            walk_length = max(2, base_walk_length // 2)
            restart_prob = 0.25
        elif degree_ratio < 0.5:
            # Low-degree node - longer walks, fewer restarts
            walk_length = base_walk_length * 2
            restart_prob = 0.1
        else:
            # Average degree node
            walk_length = base_walk_length
            restart_prob = 0.15
        
        return {
            'walk_length': walk_length,
            'num_walks': base_num_walks,
            'restart_prob': restart_prob,
            'max_nodes': target_size
        }