from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings
from NEExT.graphs import Egonet, Graph

EGONET_ALGORITHMS = Literal["k_hop_egonet", "leiden_egonet"]


class EgonetCollection(GraphCollection):
    """
    A collection class that is an extension of GraphCollection class.

    This class provides functionality to store and create instances of graphs within graphs.
    For example this can be used to create egonet for each node using k-hop neighborhood which can be treated as a graph for further analysis.

    Attributes:
        Attributes:
        graphs (List[Graph]): List of Graph instances
        graph_type (str): Backend graph library to use ("networkx" or "igraph")
        graph_id_node_array (Optional[List[int]]): Array mapping each node to its graph ID
        node_sample_rate (float): Rate at which to sample nodes from each graph (default: 1.0)
        subgraph_to_graph_node_mapping (Dict[int, Tuple[int, int]]): A dictionary mapping from a subgraph to (graph_id, node_id)
        available_algorithms: (Dict[str, Callable]): Dictionary containing all implemented egonet algorithms
    """

    subgraph_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default=None)
    available_algorithms: Dict[str, Callable] = Field(default=None)
    egonet_node_features: Embeddings = Field(default=None)
    skip_features: List[str] = Field(default=None)
    egonet_target: str = Field(default=None)

    def model_post_init(self, __context):
        self.available_algorithms = {"k_hop_egonet": self._k_hop_egonet, "leiden_egonet": self._leiden_egonet}

    def create_egonets_from_graphs(
        self,
        graph_collection: GraphCollection,
        egonet_target: str,
        egonet_algorithm: EGONET_ALGORITHMS = "k_hop_egonet",
        skip_features: Optional[List[str]] = None,
        max_hop_length: int = 1,
        n_iterations: int = 10,
        resolution: float = 1.0,
    ):
        """
        Creates an ego-net for each node in a graph_collection.

        Args:
            graph_collection (GraphCollection): Initial collection of graphs that is used to create ego-nets.
            egonet_target (str): Target variable that is used as an ego-net label
            egonet_algorithm (str, optional): Algorithm that is used to create egonets. By deafault one-hop is used.
            skip_features (List[str], optional): List of node features that should be skipped. This is a variables that is used to avoid possible data leakage issues. egonet_target byt default is added to this list.
        """
        if egonet_algorithm not in get_args(EGONET_ALGORITHMS):
            raise ValueError(f'Specified egonet algorithm "{egonet_algorithm}" is not implemented!')

        self.skip_features = [] if skip_features is None else skip_features
        self.skip_features += [egonet_target]
        self.egonet_target = egonet_target

        self.graph_id_node_array = []
        self.subgraph_to_graph_node_mapping = {}
        if egonet_algorithm == "k_hop_egonet":
            self._k_hop_egonet(graph_collection, max_hop_length)
        elif egonet_algorithm == "leiden_egonet":
            self._leiden_egonet(graph_collection, n_iterations, resolution)

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)

    def _k_hop_egonet(self, graph_collection: GraphCollection, max_hop_length: int):
        egonet_id = 0

        for graph_id, graph in enumerate(graph_collection.graphs):
            for node_id in range(graph.G.vcount()):
                egonet_nodes = k_hop_egonet(graph, node_id, k=max_hop_length)
                egonet_label = graph.node_attributes[node_id][self.egonet_target]

                egonet = self._build_egonet(
                    graph=graph,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                )

                egonet.initialize_graph()
                # Re-initialize after filtering
                self.graphs.append(egonet)
                # Update graph_id_node_array with this graph's nodes
                self.graph_id_node_array.extend([node_id] * len(egonet.nodes))

                self.subgraph_to_graph_node_mapping[egonet_id] = (graph_id, node_id)
                egonet_id += 1

    def _leiden_egonet(self, graph_collection: GraphCollection, n_iterations: int, resolution: float):
        egonet_id = 0

        for graph_id, graph in enumerate(graph_collection.graphs):
            community_detection = graph.G.community_leiden(objective_function="modularity", n_iterations=n_iterations, resolution=resolution)
            node_community_mapping = {k: v for k, v in enumerate(community_detection.membership)}

            for node_id in range(graph.G.vcount()):
                community_id = node_community_mapping[node_id]
                egonet_nodes = [n_id for n_id, com_id in node_community_mapping.items() if com_id == community_id]
                egonet_label = graph.node_attributes[node_id][self.egonet_target]

                egonet = self._build_egonet(
                    graph=graph,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
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
        egonet_id: int,
        egonet_label: float,
    ):
        """
        Build and ego-net instance as a graph

        Args:
            graph_collection (GraphCollection): Initial collection of graphs that is used to create ego-nets.
            egonet_target (str): Target variable that is used as an ego-net label
            egonet_algorithm (str, optional): Algorithm that is used to create egonets. By deafault one-hop is used.
            skip_features (List[str], optional): List of node features that should be skipped. This is a variables that is used to avoid possible data leakage issues. egonet_target byt default is added to this list.

        """
        node_mapping = {n: i for i, n in enumerate(egonet_nodes)}
        sub_node_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features}
            for node_id, feature_dict in graph.node_attributes.items()
            if node_id in egonet_nodes
        }
        sub_edge_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features}
            for node_id, feature_dict in graph.edge_attributes.items()
            if node_id in egonet_nodes
        }

        G_sub = graph.G.subgraph(list(set(egonet_nodes)))
        nodes = list(range(G_sub.vcount()))
        edges = G_sub.get_edgelist()

        return Egonet(
            graph_id=egonet_id,
            graph_label=egonet_label,
            nodes=nodes,
            edges=edges,
            node_attributes=sub_node_attributes,
            edge_attributes=sub_edge_attributes,
            graph_type="networkx" if isinstance(G_sub, nx.Graph) else "igraph",
            node_mapping=node_mapping,
        )

    def _create_egonet_features_df(self, graph_collection: GraphCollection):
        egonet_node_features_df = pd.DataFrame().from_dict(self.subgraph_to_graph_node_mapping, orient="index").reset_index()
        egonet_node_features_df.columns = ["subgraph_id", "graph_id", "node_id"]

        raw_features = defaultdict(dict)

        for graph in graph_collection.graphs:
            for node_id, features in graph.node_attributes.items():
                for feature, value in features.items():
                    if feature in self.skip_features:
                        continue
                    raw_features[graph.graph_id, node_id][feature] = value

        raw_features = (
            pd.DataFrame.from_dict(raw_features, orient="index").reset_index().rename(columns={"level_0": "graph_id", "level_1": "node_id"})
        )

        egonet_node_features_df = (
            (
                egonet_node_features_df.merge(raw_features, on=["graph_id", "node_id"])
                .drop(columns=["graph_id", "node_id"])
                .rename(columns={"subgraph_id": "graph_id"})
            )
            if len(raw_features > 0)
            else (
                egonet_node_features_df
                .drop(columns=["graph_id", "node_id"])
                .rename(columns={"subgraph_id": "graph_id"})
            )
        )
        return Embeddings(egonet_node_features_df, "egonet_node_features", [col for col in egonet_node_features_df.columns if col != "graph_id"])


def k_hop_egonet(graph: Egonet, node_id: int, k: int = 1) -> List[int]:
    return graph.G.neighborhood(node_id, order=k)
