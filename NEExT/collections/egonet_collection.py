from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args
import logging

import networkx as nx
from NEExT.helper_functions import get_nodes_x_hops_away
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings
from NEExT.features import Features
from NEExT.graphs import Egonet, Graph
from NEExT.sampling.base import SamplingStrategy, get_sampler

# Set up logger
logger = logging.getLogger(__name__)


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

    def _build_egonet(
        self,
        graph: Graph,
        node_id: int,
        egonet_nodes: List[int],
        egonet_id: int,
        egonet_label: float,
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

        Returns:
            Egonet: The constructed egonet object.

        """

        # Ensure egonet_nodes is a list and sort for deterministic ordering
        if isinstance(egonet_nodes, np.ndarray):
            egonet_nodes = egonet_nodes.tolist()
        egonet_nodes_set = set(egonet_nodes)  # Convert to set for O(1) lookups
        egonet_nodes_sorted = sorted(egonet_nodes_set)  # Remove duplicates and sort for determinism
        
        # build internal egonet node mapping and extract the features
        # Use sorted nodes for deterministic mapping
        node_mapping = {n: i for i, n in enumerate(egonet_nodes_sorted)}
        egonet_node_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features + [self.egonet_feature_target]}
            for node_id, feature_dict in graph.node_attributes.items()
            if node_id in egonet_nodes_set
        }
        egonet_edge_attributes = {
            node_mapping[node_id]: {key: value for key, value in feature_dict.items() if key not in self.skip_features + [self.egonet_feature_target]}
            for node_id, feature_dict in graph.edge_attributes.items()
            if node_id in egonet_nodes_set
        }
        # extract egonet subgraph
        if graph.graph_type == "networkx":
            G_egonet = graph.G.subgraph(egonet_nodes_sorted)
            nodes = list(range(G_egonet.number_of_nodes()))
            edges = list(G_egonet.edges())
            
        else:
            G_egonet = graph.G.subgraph(egonet_nodes_sorted)
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
        # NEW: Neighborhood sampling parameters (backwards compatible)
        sampling_strategy: Union[str, SamplingStrategy] = SamplingStrategy.K_HOP,
        walk_length: int = 10,
        num_walks: int = 5,
        restart_prob: float = 0.15,
        max_nodes_per_egonet: Optional[int] = None
    ):
        """
        Computes egonets based on neighborhood sampling strategies.

        This method iterates through each node in each graph of the input
        GraphCollection and creates an egonet centered around that node.
        The neighborhood can be sampled using different strategies.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            k_hop (int): For k-hop strategy: maximum distance from center node.
                For random walk: used as hint for walk_length if not specified.
                Defaults to 1.
            nodes_to_sample (Optional[Dict[int, List[int]]]): Dictionary mapping 
                graph IDs to lists of specific nodes to sample. If None, samples
                from all nodes based on sample_fraction.
            sample_fraction (Optional[float]): Fraction of nodes to sample as 
                ego centers (0.0-1.0). Defaults to 1.0 (all nodes).
            random_seed (int): Random seed for reproducibility. Defaults to 13.
            sampling_strategy (Union[str, SamplingStrategy]): Neighborhood sampling
                strategy. Options: 'k_hop' or 'random_walk'. Defaults to 'k_hop'.
            walk_length (int): For random walk: maximum length of each walk.
                Defaults to 10.
            num_walks (int): For random walk: number of walks per ego node.
                Defaults to 5.
            restart_prob (float): For random walk: probability of restarting at
                ego node at each step. Defaults to 0.15.
            max_nodes_per_egonet (Optional[int]): Maximum number of nodes per
                egonet. If None, no limit is applied.

        """
        np.random.seed(random_seed)
        if nodes_to_sample is None:
            nodes_to_sample = {}

        # Get the appropriate sampler
        sampler = get_sampler(sampling_strategy)

        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        valid_nodes = {}
        # draw nodes to sample from for each graph
        for graph in graph_collection.graphs:
            nodes = graph.nodes
            random_nodes = np.random.choice(nodes, int(len(nodes)* sample_fraction), replace=False).tolist()
            forced_nodes = nodes_to_sample.get(graph.graph_id, [])
            valid_nodes[graph.graph_id] = list(set(random_nodes + forced_nodes))

        for graph in graph_collection.graphs:
            for node_id in valid_nodes[graph.graph_id]:
                # Use the sampler to get neighborhood nodes
                sampler_kwargs = {
                    'k_hop': k_hop,
                    'walk_length': walk_length,
                    'num_walks': num_walks, 
                    'restart_prob': restart_prob,
                    'max_nodes': max_nodes_per_egonet,
                    'random_seed': random_seed
                }
                
                try:
                    egonet_nodes = sampler.sample_neighborhood(
                        graph=graph.G, 
                        ego_node=node_id,
                        **sampler_kwargs
                    )
                except (ValueError, IndexError, KeyError) as e:
                    # Specific errors related to sampling parameters or graph structure
                    logger.warning(f"Sampling failed for node {node_id} due to {type(e).__name__}: {e}. Falling back to k-hop sampling.")
                    if k_hop > 0: 
                        egonet_nodes_dict = get_nodes_x_hops_away(graph.G, node_id, k_hop)
                        egonet_nodes = [node_id] 
                        for v in egonet_nodes_dict.values():
                            egonet_nodes.extend(list(v))
                    else: 
                        egonet_nodes = [node_id]
                except Exception as e:
                    # Unexpected errors - log with higher severity but still fallback
                    logger.error(f"Unexpected error during sampling for node {node_id}: {type(e).__name__}: {e}. Falling back to k-hop sampling.")
                    if k_hop > 0: 
                        egonet_nodes_dict = get_nodes_x_hops_away(graph.G, node_id, k_hop)
                        egonet_nodes = [node_id] 
                        for v in egonet_nodes_dict.values():
                            egonet_nodes.extend(list(v))
                    else: 
                        egonet_nodes = [node_id]
                
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
        its associated node features.

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
            if raw_features is not None and len(raw_features) > 0
            else (egonet_node_features_df.drop(columns=["graph_id", "node_id"]).rename(columns={"subgraph_id": "graph_id"}))
        )
        return Embeddings(egonet_node_features_df, "egonet_node_features", [col for col in egonet_node_features_df.columns if col != "graph_id"])

    def compute_egonet_positionaL_features(
        self,
        strategy: Literal["distance", "inv_distance", 'inv_exp_distance'],
        one_hot_encode: bool = False,
    ):
        """
        Compute egonet positional features that can be used to encode central node
        position in the egonet. The positional features have to be added independently
        to features before embedding if you want to include it.
        """        

        df_position = []
        for egonet in self.graphs:
            _, central_node = self.egonet_to_graph_node_mapping[egonet.graph_id]

            d = pd.DataFrame()
            d['node_id'] = egonet.nodes
            d['graph_id'] = egonet.graph_id
            d['egonet_position'] = egonet.G.distances(egonet.node_mapping[central_node])[0]
            if strategy == 'inv_distance':
                d['egonet_position'] = 1 / (d['egonet_position'] + 1)
            elif strategy == 'inv_exp_distance':
                d['egonet_position'] = 1 / np.exp(d['egonet_position'] + 1)
            df_position.append(d)
            
        df_position = pd.concat(df_position, axis=0, ignore_index=True)

        if one_hot_encode and strategy == 'distance':
            df_position = pd.get_dummies(df_position, columns=['egonet_position'], dtype=np.int8)
        elif not one_hot_encode and strategy == 'distance':
            df_position['egonet_position'] = MinMaxScaler().fit_transform(df_position[['egonet_position']])

        positional_features = Features(df_position, [i for i in df_position.columns if i not in ['node_id', 'graph_id']])
        return positional_features