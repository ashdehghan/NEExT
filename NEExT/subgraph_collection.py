from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args

import networkx as nx
from pydantic import BaseModel, Field

from NEExT.graph import Graph
from NEExT.graph_collection import GraphCollection

EGONET_ALGORITHMS = Literal["one-hop", "two-hop"]

class SubGraphCollection(GraphCollection):
    subgraph_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default=None)
    available_algorithms: Dict[str, Callable] = Field(default=None)


    def model_post_init(self, __context):
        self.available_algorithms = {
            "one-hop": one_hop_algorithm,
            "two-hop": two_hop_algorithm,
        }

    def create_subgraphs(
        self,
        graph_collection: GraphCollection,
        egonet_target: str,
        egonet_algorithm: EGONET_ALGORITHMS = "one-hop",
        skip_features: Optional[List[str]] = None,
    ):
        if egonet_algorithm not in get_args(EGONET_ALGORITHMS):
            raise ValueError(f'Specified egonet algorithm "{egonet_algorithm}" is not implemented!')
        if skip_features is None:
            skip_features = []
        skip_features += [egonet_target]

        self.graph_id_node_array = []
        self.subgraph_to_graph_node_mapping = {}
        egonet_id = 0

        for graph_id, graph in enumerate(graph_collection.graphs):
            for node_id in range(graph.G.vcount()):
                egonet_nodes = self.available_algorithms[egonet_algorithm](graph, node_id)

                egonet_label = graph.node_attributes[node_id][egonet_target]
                egonet = self._build_egonet(
                    graph,
                    egonet_nodes,
                    egonet_id,
                    egonet_label,
                    skip_features=skip_features,
                )

                egonet.initialize_graph()
                # Re-initialize after filtering
                self.graphs.append(egonet)
                # Update graph_id_node_array with this graph's nodes
                self.graph_id_node_array.extend([node_id] * len(egonet.nodes))

                self.subgraph_to_graph_node_mapping[egonet_id] = (graph_id, node_id)
                egonet_id += 1

    def _build_egonet(
        self,
        graph: Graph,
        egonet_nodes: List[int],
        graph_id: int,
        graph_label: float,
        skip_features: List[str] = None,
    ):
        if skip_features is None:
            skip_features = []

        node_mapping = {n: i for i, n in enumerate(egonet_nodes)}
        sub_node_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in skip_features}
            for node_id, feature_dict in graph.node_attributes.items()
            if node_id in egonet_nodes
        }
        sub_edge_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in skip_features}
            for node_id, feature_dict in graph.edge_attributes.items()
            if node_id in egonet_nodes
        }

        G_sub = graph.G.subgraph(list(set(egonet_nodes)))
        nodes = list(range(G_sub.vcount()))
        edges = G_sub.get_edgelist()

        return Graph(
            graph_id=graph_id,
            graph_label=graph_label,
            nodes=nodes,
            edges=edges,
            node_attributes=sub_node_attributes,
            edge_attributes=sub_edge_attributes,
            graph_type="networkx" if isinstance(G_sub, nx.Graph) else "igraph",
            node_mapping=node_mapping,
        )

def one_hop_algorithm(graph: Graph, node_id: int) -> List[int]:
    subgraph_nodes = [node_id] + graph.G.neighbors(node_id)
    return list(set(subgraph_nodes))


def two_hop_algorithm(graph: Graph, node_id: int) -> List[int]:
    subgraph_nodes = [node_id] + graph.G.neighbors(node_id)
    for neigh in graph.G.neighbors(node_id):
        subgraph_nodes += graph.G.neighbors(neigh)
    return list(set(subgraph_nodes))