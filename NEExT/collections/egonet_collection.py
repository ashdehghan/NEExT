from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings
from NEExT.graphs import Egonet, Graph


class EgonetCollection(GraphCollection):
    egonet_feature_target: Optional[str] = Field(default=None)
    skip_features: List[str] = Field(default_factory=list)
    egonet_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default_factory=dict)
    egonet_node_features: Embeddings = Field(default=None)

    def _build_egonet(
        self,
        graph: Graph,
        node_id: int,
        egonet_nodes: List[int],
        egonet_id: int,
        egonet_label: float,
    ):
        # build internal egonet node mapping and extract the features
        node_mapping = {n: i for i, n in enumerate(egonet_nodes)}
        egonet_node_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features + [self.egonet_feature_target]}
            for node_id, feature_dict in graph.node_attributes.items()
            if node_id in egonet_nodes
        }
        egonet_edge_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features + [self.egonet_feature_target]}
            for node_id, feature_dict in graph.edge_attributes.items()
            if node_id in egonet_nodes
        }
        # extract egonet subgraph
        G_egonet = graph.G.subgraph(list(set(egonet_nodes)))
        nodes = list(range(G_egonet.vcount()))
        edges = G_egonet.get_edgelist()

        egonet = Egonet(
            graph_id=egonet_id,
            graph_label=egonet_label,
            nodes=nodes,
            edges=edges,
            node_attributes=egonet_node_attributes,
            edge_attributes=egonet_edge_attributes,
            graph_type=graph.graph_type,
            node_mapping=node_mapping,
            original_graph_id=graph.graph_id,
            original_node_id=node_id,
        )
        egonet.initialize_graph()
        return egonet

    def compute_k_hop_egonets(self, graph_collection: GraphCollection, k_hop: int = 1):
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        for graph in graph_collection.graphs:
            for node_id in range(graph.G.vcount()):
                egonet_nodes = graph.G.neighborhood(node_id, order=k_hop)
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                )

                self.graphs.append(egonet)
                # Update graph_id_node_array with this graph's nodes
                self.graph_id_node_array.extend([node_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)

    def compute_leiden_egonets(self, graph_collection: GraphCollection, n_iterations: int = 10, resolution: float = 1.0):
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        for graph in graph_collection.graphs:
            community_detection = graph.G.community_leiden(objective_function="modularity", n_iterations=n_iterations, resolution=resolution)
            node_community_mapping = {k: v for k, v in enumerate(community_detection.membership)}

            for node_id in range(graph.G.vcount()):
                community_id = node_community_mapping[node_id]
                egonet_nodes = [n_id for n_id, com_id in node_community_mapping.items() if com_id == community_id]
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target]

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                )
                egonet.initialize_graph()
                self.graphs.append(egonet)
                # Update graph_id_node_array with this graph's nodes
                self.graph_id_node_array.extend([node_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)

    def _create_egonet_features_df(self, graph_collection: GraphCollection):
        """
        Creates a DataFrame containing node features for each egonet.

        This method extracts node features from the original graph collection and
        organizes them into a DataFrame where each row represents an egonet and
        its associated node features. It also handles the mapping between
        subgraph IDs, graph IDs, and node IDs.

        Args:
            graph_collection (GraphCollection): The original collection of graphs
                from which the egonets were derived.

        Returns:
            Embeddings: An Embeddings object containing the egonet node features DataFrame.
        """
        egonet_node_features_df = pd.DataFrame().from_dict(self.egonet_to_graph_node_mapping, orient="index").reset_index()
        egonet_node_features_df.columns = ["subgraph_id", "graph_id", "node_id"]

        raw_features = defaultdict(dict)

        for graph in graph_collection.graphs:
            for node_id, features in graph.node_attributes.items():
                for feature, value in features.items():
                    if feature in self.skip_features + [self.egonet_feature_target]:
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
            else (egonet_node_features_df.drop(columns=["graph_id", "node_id"]).rename(columns={"subgraph_id": "graph_id"}))
        )
        return Embeddings(egonet_node_features_df, "egonet_node_features", [col for col in egonet_node_features_df.columns if col != "graph_id"])
