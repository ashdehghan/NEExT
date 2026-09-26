from typing import Dict, Optional, Union

from pydantic import Field

from NEExT.graphs import Graph


class Egonet(Graph):
    """
    Attributes:
        node_mapping Optional[Dict[int, int]]: Mapping from original graph node IDs to internal egonet node IDs
        node_weights Optional[Dict[int, float]]: Optional membership weights keyed by INTERNAL egonet node ID
            (e.g. normalized random-walk visit frequencies). None means uniform membership; when present the
            weights are consumed as the egonet's distribution mass in graph embeddings.
    """

    original_graph_id: Optional[Union[int, str]] = Field(default=None)
    original_node_id: int = Field(default=None)
    node_mapping: Optional[Dict[int, int]] = Field(default_factory=dict)
    node_weights: Optional[Dict[int, float]] = Field(default=None)

    def reindex_nodes(self) -> "Egonet":
        """Reindex nodes to be consecutive integers starting from 0."""
        # Create mapping from old to new indices
        unique_nodes, new_edges, new_node_attrs, new_edge_attrs = self._reindex_nodes()

        # Compose old node_mapping with the reindex mapping so original→internal stays valid
        node_remap = {old: new for new, old in enumerate(sorted(set(self.nodes)))}
        new_node_mapping = {orig: node_remap[internal] for orig, internal in self.node_mapping.items() if internal in node_remap}

        # Recompose weights the same way; renormalize because dropped members take
        # their mass with them and the weights are read as a distribution.
        new_node_weights = None
        if self.node_weights is not None:
            kept = {node_remap[internal]: weight for internal, weight in self.node_weights.items() if internal in node_remap}
            total = sum(kept.values())
            new_node_weights = {internal: weight / total for internal, weight in kept.items()} if total > 0 else None

        # Create new graph with mapped IDs
        return Egonet(
            graph_id=self.graph_id,
            graph_label=self.graph_label,
            nodes=list(range(len(unique_nodes))),
            edges=new_edges,
            node_attributes=new_node_attrs,
            edge_attributes=new_edge_attrs,
            graph_type=self.graph_type,
            original_graph_id=self.original_graph_id,
            original_node_id=self.original_node_id,
            node_mapping=new_node_mapping,
            node_weights=new_node_weights,
        )

    def filter_largest_component(self) -> "Egonet":
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
            original_graph_id=self.original_graph_id,
            original_node_id=self.original_node_id,
            node_mapping=self.node_mapping,
            node_weights=self.node_weights,
        )

        # Reindex nodes to be consecutive
        return filtered_graph.reindex_nodes()
