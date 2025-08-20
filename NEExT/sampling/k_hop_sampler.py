"""
K-hop neighborhood sampling strategy.

This module implements the traditional k-hop neighborhood sampling,
which includes ALL nodes within k-hop distance of the ego node.
This is the existing behavior from NEExT, extracted for modularity.
"""

from typing import List, Union
import networkx as nx
from .base import NodeSampler

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class KHopSampler(NodeSampler):
    """
    Traditional k-hop neighborhood sampler.
    
    Includes all nodes within k-hop distance of the ego node.
    This preserves the original NEExT behavior for backwards compatibility.
    """
    
    def sample_neighborhood(self, 
                          graph: Union[nx.Graph, 'ig.Graph'], 
                          ego_node: int,
                          k_hop: int = 1,
                          **kwargs) -> List[int]:
        """
        Sample k-hop neighborhood (all nodes within k-hop distance).
        
        Args:
            graph: NetworkX or iGraph graph object
            ego_node: The central node for neighborhood sampling
            k_hop: Maximum hop distance to include
            **kwargs: Additional parameters (ignored for backwards compatibility)
            
        Returns:
            List of node IDs in the k-hop neighborhood (including ego_node)
        """
        self.validate_graph(graph)
        self.validate_ego_node(graph, ego_node)
        
        if k_hop < 0:
            raise ValueError("k_hop must be non-negative")
        
        if k_hop == 0:
            return [ego_node]
        
        # Use existing helper function for consistency
        from ..helper_functions import get_nodes_x_hops_away
        
        # Get nodes at each hop level
        nodes_by_hop = get_nodes_x_hops_away(graph, ego_node, k_hop)
        
        # Flatten and combine all nodes
        neighborhood = [ego_node]  # Always include ego node
        
        for hop_level in range(1, k_hop + 1):
            if hop_level in nodes_by_hop:
                neighborhood.extend(nodes_by_hop[hop_level])
        
        # Remove duplicates and sort for consistency
        return sorted(list(set(neighborhood)))