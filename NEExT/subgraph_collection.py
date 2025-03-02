from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set, Tuple, Union, Literal, get_args

from NEExT.egonets import build_egonet, one_hop_algorithm, two_hop_algorithm
from NEExT.graph_collection import GraphCollection

EGONET_ALGORITHMS = Literal["one-hop", "two-hop"]

ALGORITHMS = {
    "one-hop": one_hop_algorithm,
    "two-hop": two_hop_algorithm,
}


class SubGraphCollection(GraphCollection):
    subgraph_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default=None)

    def create_subgraphs(
        self,
        graph_collection: GraphCollection,
        subgraph_target: str,
        egonet_algorithm: EGONET_ALGORITHMS = "one-hop",
    ):
        if egonet_algorithm not in get_args(EGONET_ALGORITHMS):
            raise ValueError(f'Specified egonet algorithm "{egonet_algorithm}" is not implemented!')

        self.graph_id_node_array = []
        self.subgraph_to_graph_node_mapping = {}
        subgraph_id = 0

        for graph_id, graph in enumerate(graph_collection.graphs):
            for node_id in range(graph.G.vcount()):
                subgraph_nodes = ALGORITHMS[egonet_algorithm](graph, node_id)

                subgraph_label = graph.node_attributes[node_id][subgraph_target]
                subgraph = build_egonet(
                    graph,
                    subgraph_nodes,
                    subgraph_id,
                    subgraph_label,
                    skip_features=[subgraph_target],
                )

                subgraph.initialize_graph()
                # Re-initialize after filtering
                self.graphs.append(subgraph)
                # Update graph_id_node_array with this graph's nodes
                self.graph_id_node_array.extend([node_id] * len(subgraph.nodes))

                self.subgraph_to_graph_node_mapping[subgraph_id] = (graph_id, node_id)
                subgraph_id += 1
