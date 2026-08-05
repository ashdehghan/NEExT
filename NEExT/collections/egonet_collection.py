from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings
from NEExT.features import Features
from NEExT.graphs import Egonet, Graph
from NEExT.helper_functions import get_nodes_x_hops_away


class EgonetCollection(GraphCollection):
    """
    A collection of egonets derived from a GraphCollection.

    This class extends GraphCollection to specifically handle collections of
    egonets. Egonets are subgraphs centered around a specific node (the ego)
    and include its neighbors up to a certain distance (k-hop) or within a
    community.

    Attributes:
        egonet_feature_target (Optional[str]): The name of the node feature to
            be used as the target variable for the egonets. Defaults to None.
        skip_features (List[str]): A list of node feature names to be excluded
            when building egonets. Defaults to an empty list.
        egonet_to_graph_node_mapping (Dict[int, Tuple[int, int]]): A dictionary
            mapping egonet IDs to tuples of (original graph ID, original node ID).
            Defaults to an empty dictionary.
        egonet_node_features (Embeddings): An Embeddings object containing the
            features of a central node of each egonet. Defaults to None.
    """

    egonet_feature_target: Optional[str] = Field(default=None)
    skip_features: List[str] = Field(default_factory=list)
    egonet_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default_factory=dict)
    egonet_node_features: Embeddings = Field(default=None)

    def _build_parent_index(self, graph: Graph) -> dict:
        """
        Precompute per-parent-graph lookup structures shared by every egonet of
        that graph, so each egonet build touches only its own neighborhood
        instead of scanning the whole parent graph.

        The position maps preserve the parent dicts' iteration order so egonet
        contents come out identical to a full-scan filter of those dicts.
        """
        skip_keys = frozenset(self.skip_features + ([self.egonet_feature_target] if self.egonet_feature_target else []))
        node_attr_pos = {nid: i for i, nid in enumerate(graph.node_attributes)}
        edge_attr_by_node = {}
        if graph.edge_attributes:
            edge_attr_by_node = defaultdict(list)
            for pos, (edge, attrs) in enumerate(graph.edge_attributes.items()):
                edge_attr_by_node[edge[0]].append((pos, edge, attrs))
        index = {
            "skip_keys": skip_keys,
            "node_attr_pos": node_attr_pos,
            "edge_attr_by_node": edge_attr_by_node,
        }
        if graph.graph_type == "igraph":
            # Attribute-free clone: igraph's subgraph() copies every vertex/edge
            # attribute of the parent, all of which the egonet build discards.
            # Same topology in the same order -> identical subgraph structure.
            import igraph as ig

            bare = ig.Graph(n=graph.G.vcount())
            bare.add_edges(graph.G.get_edgelist())
            index["bare_graph"] = bare
        return index

    def _build_egonet(
        self,
        graph: Graph,
        node_id: int,
        egonet_nodes: List[int],
        egonet_id: int,
        egonet_label: float,
        parent_index: Optional[dict] = None,
    ) -> Egonet:
        """
        This method constructs an Egonet object from a given graph.
        It extracts the relevant subgraph, node and
        edge attributes, and creates the necessary mappings.

        Args:
            graph (Graph): The original graph from which to extract the egonet.
            node_id (int): The ID of the center node (ego) of the egonet.
            egonet_nodes (List[int]): The list of node IDs that belong to the egonet.
            egonet_id (int): The unique ID to assign to the egonet.
            egonet_label (float): The label to assign to the egonet.
            parent_index (Optional[dict]): Precomputed per-parent lookups from
                _build_parent_index; computed on the fly when omitted.

        Returns:
            Egonet: The constructed egonet object.

        """

        # Sort nodes for deterministic ordering
        egonet_nodes_sorted = sorted(set(egonet_nodes))  # Remove duplicates and sort for determinism

        # build internal egonet node mapping and extract the features
        # Use sorted nodes for deterministic mapping
        node_mapping = {n: i for i, n in enumerate(egonet_nodes_sorted)}
        if parent_index is None:
            parent_index = self._build_parent_index(graph)
        skip_keys = parent_index["skip_keys"]
        egonet_nodes_set = set(egonet_nodes_sorted)

        # node attributes: visit only egonet members, in the parent dict's order
        node_attr_pos = parent_index["node_attr_pos"]
        node_attrs_src = graph.node_attributes
        attr_nodes = [nid for nid in egonet_nodes_sorted if nid in node_attr_pos]
        attr_nodes.sort(key=node_attr_pos.__getitem__)
        egonet_node_attributes = {
            node_mapping[nid]: {key: value for key, value in node_attrs_src[nid].items() if key not in skip_keys} for nid in attr_nodes
        }

        # edge attributes: gather via the per-node index, restore parent dict order
        egonet_edge_attributes = {}
        edge_attr_by_node = parent_index["edge_attr_by_node"]
        if edge_attr_by_node:
            matches = []
            for nid in egonet_nodes_sorted:
                for entry in edge_attr_by_node.get(nid, ()):
                    if entry[1][1] in egonet_nodes_set:
                        matches.append(entry)
            matches.sort(key=lambda entry: entry[0])
            for _, (src, dst), attrs in matches:
                egonet_edge_attributes[(node_mapping[src], node_mapping[dst])] = {key: value for key, value in attrs.items() if key not in skip_keys}

        # extract egonet subgraph (nx: cheap view; igraph: attribute-free clone
        # so the C subgraph routine yields identical structure without copying
        # the parent's attributes)
        if graph.graph_type == "networkx":
            G_egonet = graph.G.subgraph(egonet_nodes_sorted)
            nodes = list(range(G_egonet.number_of_nodes()))
            edges = [(node_mapping[u], node_mapping[v]) for u, v in G_egonet.edges()]
        else:
            G_egonet = parent_index["bare_graph"].subgraph(egonet_nodes_sorted)
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

    def compute_k_hop_egonets(
        self,
        graph_collection: GraphCollection,
        k_hop: int = 1,
        nodes_to_sample: Optional[Dict[int, List[int]]] = None,
        sample_fraction: Optional[float] = 1.0,
        random_seed: int = 13,
    ):
        """
        Computes egonets based on k-hop neighborhood.

        This method iterates through each node in each graph of the input
        GraphCollection and creates an egonet centered around that node,
        including all nodes within k-hop distance.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            k_hop (int): The maximum distance (in hops) from the center node
                to include in the egonet. Defaults to 1.

        """
        rng = np.random.RandomState(random_seed)
        if nodes_to_sample is None:
            nodes_to_sample = {}

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        valid_nodes = {}
        # draw nodes to sample from for each graph
        for graph in graph_collection.graphs:
            nodes = graph.nodes
            random_nodes = rng.choice(nodes, int(len(nodes) * sample_fraction), replace=False).tolist()
            forced_nodes = nodes_to_sample.get(graph.graph_id, [])
            valid_nodes[graph.graph_id] = list(set(random_nodes + forced_nodes))

        for graph in graph_collection.graphs:
            parent_index = self._build_parent_index(graph)
            for node_id in valid_nodes[graph.graph_id]:
                if k_hop > 0:
                    egonet_nodes_dict = get_nodes_x_hops_away(graph.G, node_id, k_hop)
                    egonet_nodes = {node_id}
                    for v in egonet_nodes_dict.values():
                        egonet_nodes.update(v)
                else:
                    egonet_nodes = [node_id]
                # egonet_nodes = sorted(graph.G.neighborhood(node_id, order=k_hop))
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                    parent_index=parent_index,
                )

                self.graphs.append(egonet)
                # Update graph_id_node_array with this egonet's id
                self.graph_id_node_array.extend([egonet_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)

    def compute_leiden_egonets(self, graph_collection: GraphCollection, n_iterations: int = 10, resolution: float = 1.0):
        """
        Computes egonets based on Leiden community detection.

        This method iterates through each graph in the input GraphCollection,
        performs Leiden community detection, and then creates an egonet for
        each node, including all nodes in the same community.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            n_iterations (int): The number of iterations for the Leiden
                algorithm. Defaults to 10.
            resolution (float): The resolution parameter for the Leiden
                algorithm. Defaults to 1.0.

        """

        if any(g.graph_type == "networkx" for g in graph_collection.graphs):
            raise NotImplementedError("Leiden egonets require iGraph backend. Use graph_type='igraph' when loading graphs.")

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        for graph in graph_collection.graphs:
            community_detection = graph.G.community_leiden(objective_function="modularity", n_iterations=n_iterations, resolution=resolution)
            membership = community_detection.membership
            community_members = defaultdict(list)
            for n_id, com_id in enumerate(membership):
                community_members[com_id].append(n_id)
            parent_index = self._build_parent_index(graph)

            for node_id in range(graph.G.vcount()):
                egonet_nodes = community_members[membership[node_id]]
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                    parent_index=parent_index,
                )
                self.graphs.append(egonet)
                # Update graph_id_node_array with this egonet's id
                self.graph_id_node_array.extend([egonet_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)

    def _create_egonet_features_df(self, graph_collection: GraphCollection):
        """
        Creates a DataFrame containing node features for each egonet.

        This method extracts node features from the original graph collection and
        organizes them into a DataFrame where each row represents an egonet and
        its associated node features.

        Args:
            graph_collection (GraphCollection): The original collection of graphs
                from which the egonets were derived.

        Returns:
            Embeddings: An Embeddings object containing the egonet node features DataFrame.
        """
        egonet_node_features_df = pd.DataFrame().from_dict(self.egonet_to_graph_node_mapping, orient="index").reset_index()
        egonet_node_features_df.columns = ["subgraph_id", "graph_id", "node_id"]

        skip_keys = frozenset(self.skip_features + ([self.egonet_feature_target] if self.egonet_feature_target else []))
        raw_features = {}

        for graph in graph_collection.graphs:
            graph_id = graph.graph_id
            for node_id, features in graph.node_attributes.items():
                kept = {feature: value for feature, value in features.items() if feature not in skip_keys}
                if kept:
                    raw_features[graph_id, node_id] = kept

        raw_features = (
            pd.DataFrame.from_dict(raw_features, orient="index").reset_index().rename(columns={"level_0": "graph_id", "level_1": "node_id"})
        )

        egonet_node_features_df = (
            (
                egonet_node_features_df.merge(raw_features, on=["graph_id", "node_id"])
                .drop(columns=["graph_id", "node_id"])
                .rename(columns={"subgraph_id": "graph_id"})
            )
            if raw_features is not None and len(raw_features) > 0
            else (egonet_node_features_df.drop(columns=["graph_id", "node_id"]).rename(columns={"subgraph_id": "graph_id"}))
        )
        return Embeddings(egonet_node_features_df, "egonet_node_features", [col for col in egonet_node_features_df.columns if col != "graph_id"])

    def compute_egonet_positional_features(
        self,
        strategy: Literal["distance", "inv_distance", "inv_exp_distance"],
        one_hot_encode: bool = False,
    ):
        """
        Compute egonet positional features that can be used to encode central node
        position in the egonet. The positional features have to be added independently
        to features before embedding if you want to include it.
        """

        node_ids: List[int] = []
        graph_ids: List[int] = []
        positions: List[float] = []
        for egonet in self.graphs:
            _, central_node = self.egonet_to_graph_node_mapping[egonet.graph_id]
            mapped_central = egonet.node_mapping[central_node]

            if egonet.graph_type == "igraph":
                egonet_positions = egonet.G.distances(mapped_central)[0]
            else:
                lengths = nx.single_source_shortest_path_length(egonet.G, mapped_central)
                egonet_positions = [lengths.get(n, float("inf")) for n in egonet.nodes]
            node_ids.extend(egonet.nodes)
            graph_ids.extend([egonet.graph_id] * len(egonet.nodes))
            positions.extend(egonet_positions)

        df_position = pd.DataFrame({"node_id": node_ids, "graph_id": graph_ids, "egonet_position": positions})
        if strategy == "inv_distance":
            df_position["egonet_position"] = 1 / (df_position["egonet_position"] + 1)
        elif strategy == "inv_exp_distance":
            df_position["egonet_position"] = 1 / np.exp(df_position["egonet_position"] + 1)

        if one_hot_encode and strategy == "distance":
            df_position = pd.get_dummies(df_position, columns=["egonet_position"], dtype=np.int8)
        elif not one_hot_encode and strategy == "distance":
            df_position["egonet_position"] = MinMaxScaler().fit_transform(df_position[["egonet_position"]])

        positional_features = Features(df_position, [i for i in df_position.columns if i not in ["node_id", "graph_id"]])
        return positional_features
