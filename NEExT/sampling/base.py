"""
Base classes and enums for neighborhood sampling strategies.
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import networkx as nx

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class SamplingStrategy(Enum):
    """Available neighborhood sampling strategies."""
    K_HOP = "k_hop"
    RANDOM_WALK = "random_walk"
    # Future strategies can be added here:
    # PAGERANK = "pagerank"
    # COMMUNITY_AWARE = "community_aware"


class NodeSampler(ABC):
    """
    Abstract base class for neighborhood sampling strategies.
    
    All sampling strategies should inherit from this class and implement
    the sample_neighborhood method.
    """
    
    @abstractmethod
    def sample_neighborhood(self, 
                          graph: Union[nx.Graph, 'ig.Graph'], 
                          ego_node: int, 
                          **kwargs) -> List[int]:
        """
        Sample a neighborhood around the ego node.
        
        Args:
            graph: NetworkX or iGraph graph object
            ego_node: The central node for neighborhood sampling
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of node IDs in the sampled neighborhood (including ego_node)
        """
        pass
    
    def validate_graph(self, graph: Union[nx.Graph, 'ig.Graph']) -> None:
        """Validate that the graph is supported by this sampler."""
        if isinstance(graph, nx.Graph):
            return  # NetworkX always supported
        elif IGRAPH_AVAILABLE and isinstance(graph, ig.Graph):
            return  # iGraph supported if available
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    def validate_ego_node(self, graph: Union[nx.Graph, 'ig.Graph'], ego_node: int) -> None:
        """Validate that ego_node exists in the graph."""
        if isinstance(graph, nx.Graph):
            if ego_node not in graph:
                raise ValueError(f"Ego node {ego_node} not found in graph")
        elif IGRAPH_AVAILABLE and isinstance(graph, ig.Graph):
            if ego_node < 0 or ego_node >= graph.vcount():
                raise ValueError(f"Ego node {ego_node} not found in graph")
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")


def get_sampler(strategy: Union[str, SamplingStrategy]) -> NodeSampler:
    """
    Factory function to get the appropriate sampler for a strategy.
    
    Args:
        strategy: Sampling strategy (string or enum)
        
    Returns:
        NodeSampler instance for the specified strategy
    """
    # Import here to avoid circular imports
    from .k_hop_sampler import KHopSampler
    from .random_walk_sampler import RandomWalkSampler
    
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = SamplingStrategy(strategy)
        except ValueError:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Return appropriate sampler
    if strategy == SamplingStrategy.K_HOP:
        return KHopSampler()
    elif strategy == SamplingStrategy.RANDOM_WALK:
        return RandomWalkSampler()
    else:
        raise ValueError(f"Sampler not implemented for strategy: {strategy}")