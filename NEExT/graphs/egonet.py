from typing import Dict, Optional

from pydantic import Field

from NEExT.graphs import Graph


class Egonet(Graph):
    """
    Attributes:
        node_mapping Optional[Dict[int, int]]: Napping from internal nodes_id of an egonet to original graph nodes_id
    """
    node_mapping: Optional[Dict[int, int]] = Field(default_factory=dict)
    
    def reindex_nodes(self) -> 'Egonet':
        """Reindex nodes to be consecutive integers starting from 0."""
        # Create mapping from old to new indices
        unique_nodes, new_edges, new_node_attrs, new_edge_attrs = self._reindex_nodes()

        # Create new graph with mapped IDs
        return Egonet(
            graph_id=self.graph_id,
            graph_label=self.graph_label,
            nodes=list(range(len(unique_nodes))),  
            edges=new_edges,
            node_attributes=new_node_attrs,
            edge_attributes=new_edge_attrs,
            graph_type=self.graph_type,
            node_mapping=self.node_mapping,
        )
        
    def filter_largest_component(self) -> 'Egonet':
        """
        Filter the graph to keep only the largest connected component.
        
        Returns:
            Graph: A new Graph instance containing only the largest connected component
        """
        nodes, edges, node_attrs, edge_attrs = self._filter_largest_component()
        
        # Create new Graph instance
        filtered_graph = Egonet(
            graph_id=self.graph_id,
            graph_label=self.graph_label,
            nodes=nodes,
            edges=edges,
            node_attributes=node_attrs,
            edge_attributes=edge_attrs,
            graph_type=self.graph_type,
            node_mapping=self.node_mapping,
        )
        
        # Reindex nodes to be consecutive
        return filtered_graph.reindex_nodes()