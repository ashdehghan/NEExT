import logging
import random
import time
from collections import defaultdict
from typing import Any, List, Literal, Optional

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from NEExT.collections import GraphCollection
from NEExT.features import Features
from NEExT.helper_functions import get_all_neighborhoods_ig, get_all_neighborhoods_nx

# Set up logger
logger = logging.getLogger(__name__)

_NEEXT_OWNED_JOBLIB_KWARGS = {"n_jobs", "backend", "return_as"}


class StructuralNodeFeatureConfig(BaseModel):
    """Configuration for node feature computation"""

    feature_list: List[str]
    feature_vector_length: int
    normalize_features: bool = True
    show_progress: bool = Field(default=True)
    n_jobs: int = -1
    parallel_backend: Literal["loky", "threading"] = "loky"
    profile_features: bool = False
    joblib_kwargs: Optional[dict[str, Any]] = None
    suffix: str = ""

    @field_validator("joblib_kwargs")
    @classmethod
    def validate_joblib_kwargs(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if value is None:
            return None

        owned_keys = sorted(_NEEXT_OWNED_JOBLIB_KWARGS.intersection(value))
        if owned_keys:
            raise ValueError(
                "joblib_kwargs cannot include NEExT-owned joblib arguments: "
                f"{', '.join(owned_keys)}. Use StructuralNodeFeatures arguments for these controls."
            )

        return dict(value)


class StructuralNodeFeatures:
    """
    A class for computing node features across a collection of graphs.

    This class provides parallel computation of various node features for all nodes
    in a graph collection. It supports both NetworkX and iGraph backends and includes
    multiple feature computation methods like PageRank, centrality measures, and
    structural embeddings.

    Features are computed with neighborhood aggregation, meaning for each feature,
    we compute values for the node itself and its neighborhood up to a specified depth.

    Attributes:
        graph_collection (GraphCollection): Collection of graphs to process
        config (StructuralNodeFeatureConfig): Configuration for feature computation
        available_features (Dict): Dictionary mapping feature names to computation methods

    Example:
        >>> node_features = StructuralNodeFeatures(
        ...     graph_collection=collection,
        ...     feature_list=["page_rank", "degree_centrality"],
        ...     feature_vector_length=3,
        ...     n_jobs=-1
        ... )
        >>> features_df = node_features.compute()
    """

    def __init__(
        self,
        graph_collection: GraphCollection,
        feature_list: List[str],
        feature_vector_length: int = 3,
        normalize_features: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
        parallel_backend: Literal["loky", "threading"] = "loky",
        profile_features: bool = False,
        joblib_kwargs: Optional[dict[str, Any]] = None,
        suffix: str = "",
    ):
        """Initialize the NodeFeatures processor."""
        self.graph_collection = graph_collection

        # Define available features
        self.available_features = {
            "page_rank": self._compute_page_rank,
            "degree_centrality": self._compute_degree_centrality,
            "closeness_centrality": self._compute_closeness_centrality,
            "betweenness_centrality": self._compute_betweenness_centrality,
            "eigenvector_centrality": self._compute_eigenvector_centrality,
            "clustering_coefficient": self._compute_clustering_coefficient,
            "local_efficiency": self._compute_local_efficiency,
            "lsme": self._compute_lsme,
            "load_centrality": self._compute_load_centrality,
            "basic_expansion": self._compute_basic_expansion,
            "betastar": self._compute_betastar,
        }

        # Store the raw feature_list; resolution and validation will occur in compute()
        self.config = StructuralNodeFeatureConfig(
            feature_list=list(feature_list),  # Store a copy
            feature_vector_length=feature_vector_length,
            normalize_features=normalize_features,
            show_progress=show_progress,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            profile_features=profile_features,
            joblib_kwargs=joblib_kwargs,
            suffix=suffix,
        )
        self.features_df = None

        # Cache for neighborhoods to avoid redundant computation
        self._neighborhood_cache = {}

    def _precompute_neighborhoods(self, graph):
        """Precompute neighborhoods for all nodes in the graph.

        This optimization caches neighborhoods to avoid redundant computation
        across different feature methods.
        """
        # Check if already cached for this graph
        if graph.graph_id in self._neighborhood_cache:
            return self._neighborhood_cache[graph.graph_id]

        # Determine which nodes to process
        nodes_to_process = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)

        # Compute neighborhoods based on graph type
        if isinstance(graph.G, nx.Graph):
            neighborhoods = get_all_neighborhoods_nx(graph.G, self.config.feature_vector_length, nodes_to_process)
        else:  # igraph
            neighborhoods = get_all_neighborhoods_ig(graph.G, self.config.feature_vector_length, nodes_to_process)

        # Cache the result
        self._neighborhood_cache[graph.graph_id] = neighborhoods
        return neighborhoods

    def _compute_structural_feature(self, graph, feature_func_nx, feature_func_ig=None, feature_name=None) -> pd.DataFrame:
        """Optimized structural feature computation."""
        # Compute base features for all nodes ONCE (needed for neighborhood computations)
        if isinstance(graph.G, nx.Graph):
            features = feature_func_nx(graph.G)
            # Get neighborhoods only for sampled nodes
            nodes_to_process = graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes
            neighborhoods = get_all_neighborhoods_nx(graph.G, self.config.feature_vector_length, nodes_to_process)
        else:  # igraph
            features = feature_func_ig(graph.G) if feature_func_ig else feature_func_nx(nx.Graph(graph.G.get_edgelist()))
            features = dict(enumerate(features)) if isinstance(features, (list, np.ndarray)) else features
            # Get neighborhoods only for sampled nodes
            nodes_to_process = graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes
            neighborhoods = get_all_neighborhoods_ig(graph.G, self.config.feature_vector_length, nodes_to_process)

        # Prepare feature matrix only for nodes we're processing
        n_nodes = len(nodes_to_process)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((n_nodes, n_hops))

        # Fill in base features
        feature_matrix[:, 0] = [features[node] for node in nodes_to_process]

        # Vectorized computation of neighborhood features
        for i, node in enumerate(nodes_to_process):
            node_neighborhoods = neighborhoods[node]
            for hop in range(1, n_hops):
                if hop in node_neighborhoods and node_neighborhoods[hop]:
                    hop_features = [features[n] for n in node_neighborhoods[hop]]
                    feature_matrix[i, hop] = np.mean(hop_features)

        # Create DataFrame
        columns = [f"{feature_name}_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes_to_process
        df["graph_id"] = graph.graph_id

        return df[["node_id", "graph_id"] + columns]

    def _compute_page_rank(self, graph) -> pd.DataFrame:
        """Compute PageRank features for all nodes (supports networkx and igraph)."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((len(nodes), n_hops))

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        if isinstance(G, nx.Graph):
            page_rank = nx.pagerank(G)
        else:  # igraph
            page_rank = dict(enumerate(G.pagerank()))

        # Fill feature matrix using pre-computed neighborhoods
        for i, node in enumerate(nodes):
            feature_matrix[i, 0] = page_rank[node]
            node_neighborhoods = neighborhoods.get(node, {})
            for hop in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(hop, [])
                if hop_nodes:
                    feature_matrix[i, hop] = np.mean([page_rank[n] for n in hop_nodes])
                else:
                    feature_matrix[i, hop] = 0.0

        columns = [f"page_rank_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id
        return df[["node_id", "graph_id"] + columns]

    def _compute_degree_centrality(self, graph) -> pd.DataFrame:
        """Compute degree centrality features for all nodes (supports networkx and igraph)."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((len(nodes), n_hops))

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        if isinstance(G, nx.Graph):
            deg_cent = nx.degree_centrality(G)
        else:  # igraph
            n = G.vcount()
            degs = G.degree()
            if n > 1:
                deg_cent = {i: deg / float(n - 1) for i, deg in enumerate(degs)}
            else:
                deg_cent = {i: -1 for i in range(n)}

        # Fill feature matrix using pre-computed neighborhoods
        for i, node in enumerate(nodes):
            feature_matrix[i, 0] = deg_cent[node]
            node_neighborhoods = neighborhoods.get(node, {})
            for hop in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(hop, [])
                if hop_nodes:
                    feature_matrix[i, hop] = np.mean([deg_cent[n] for n in hop_nodes])
                else:
                    feature_matrix[i, hop] = 0.0

        columns = [f"degree_centrality_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id
        return df[["node_id", "graph_id"] + columns]

    def _compute_closeness_centrality(self, graph) -> pd.DataFrame:
        """Compute closeness centrality features for all nodes (supports networkx and igraph)."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((len(nodes), n_hops))

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        if isinstance(G, nx.Graph):
            clo_cent = nx.closeness_centrality(G)
        else:  # igraph
            clo_list = G.closeness()
            clo_cent = dict(enumerate(clo_list))

        # Fill feature matrix using pre-computed neighborhoods
        for i, node in enumerate(nodes):
            val = clo_cent[node]
            feature_matrix[i, 0] = val if not np.isnan(val) else -1
            node_neighborhoods = neighborhoods.get(node, {})
            for hop in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(hop, [])
                if hop_nodes:
                    vals = [clo_cent[n] if not np.isnan(clo_cent[n]) else -1 for n in hop_nodes]
                    feature_matrix[i, hop] = np.mean(vals)
                else:
                    feature_matrix[i, hop] = 0.0

        columns = [f"closeness_centrality_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id
        return df[["node_id", "graph_id"] + columns]

    def _compute_eigenvector_centrality(self, graph) -> pd.DataFrame:
        """Compute eigenvector centrality features for all nodes."""

        def eigenvector_with_fallback(G):
            try:
                # Try computing eigenvector centrality with increased iterations
                return nx.eigenvector_centrality(G, max_iter=2000, tol=1e-4)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                # Fallback: use degree centrality as an approximation
                logger.warning(f"Eigenvector centrality failed to converge for graph {graph.graph_id}. Using degree centrality as fallback.")
                return nx.degree_centrality(G)

        return self._compute_structural_feature(
            graph, eigenvector_with_fallback, lambda G: dict(enumerate(G.eigenvector_centrality())), feature_name="eigenvector_centrality"
        )

    def _compute_load_centrality(self, graph) -> pd.DataFrame:
        """Compute load centrality features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph, nx.load_centrality, None, feature_name="load_centrality"  # No iGraph equivalent; triggers NetworkX conversion fallback
        )

    def _compute_basic_expansion(self, graph) -> pd.DataFrame:
        """
        Compute basic expansion features for each node.

        The expansion embedding captures how quickly the neighborhood of a node grows.
        For each node v, computes the vector E(v) = (n₁/d, n₂/(n₁·d), ..., nₖ/(nₖ₋₁·d))
        where:
        - nᵢ is the number of nodes at distance i from v
        - d is the average degree of the graph

        Args:
            graph: Graph object to process

        Returns:
            pd.DataFrame: DataFrame containing expansion features for each node
        """
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        feature_vectors = []

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        # Calculate average degree
        if isinstance(G, nx.Graph):
            avg_degree = (2 * len(G.edges)) / len(G.nodes)
        else:  # igraph
            avg_degree = (2 * G.ecount()) / G.vcount()

        for node in nodes:
            # Use pre-computed neighborhoods
            node_neighborhoods = neighborhoods.get(node, {})
            nr_neighbors = [len(node_neighborhoods.get(i, [])) for i in range(1, self.config.feature_vector_length + 1)]

            # Calculate expansion ratios
            vector = []
            vector.append(nr_neighbors[0] / avg_degree)  # First hop

            # Subsequent hops
            for i in range(1, len(nr_neighbors)):
                if nr_neighbors[i - 1] == 0:
                    vector.append(0)
                else:
                    vector.append(nr_neighbors[i] / (avg_degree * nr_neighbors[i - 1]))

            feature_vectors.append(vector)

        # Create DataFrame
        df = pd.DataFrame(feature_vectors)
        df.columns = [f"basic_expansion_{i}" for i in range(len(df.columns))]
        df.insert(0, "node_id", nodes)
        df.insert(0, "graph_id", graph.graph_id)
        return df

    def _compute_betweenness_centrality(self, graph) -> pd.DataFrame:
        """Compute betweenness centrality features for all nodes (supports networkx and igraph)."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((len(nodes), n_hops))

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        if isinstance(G, nx.Graph):
            bet_cent = nx.betweenness_centrality(G)
        else:  # igraph
            bet_list = G.betweenness()
            # Normalize to match NetworkX output: NetworkX divides by (n-1)*(n-2)/2
            n = G.vcount()
            if n > 2:
                norm = 2.0 / ((n - 1) * (n - 2))
            else:
                norm = 1.0
            bet_cent = {i: v * norm for i, v in enumerate(bet_list)}

        # Fill feature matrix using pre-computed neighborhoods
        for i, node in enumerate(nodes):
            val = bet_cent[node]
            feature_matrix[i, 0] = val if not np.isnan(val) else -1
            node_neighborhoods = neighborhoods.get(node, {})
            for hop in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(hop, [])
                if hop_nodes:
                    vals = [bet_cent[n] if not np.isnan(bet_cent[n]) else -1 for n in hop_nodes]
                    feature_matrix[i, hop] = np.mean(vals)
                else:
                    feature_matrix[i, hop] = 0.0

        columns = [f"betweenness_centrality_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id
        return df[["node_id", "graph_id"] + columns]

    def _compute_clustering_coefficient(self, graph) -> pd.DataFrame:
        """Compute clustering coefficient features for all nodes (supports networkx and igraph)."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((len(nodes), n_hops))

        # Get pre-computed neighborhoods
        neighborhoods = self._precompute_neighborhoods(graph)

        if isinstance(G, nx.Graph):
            clust = nx.clustering(G)
        else:  # igraph
            clust_list = G.transitivity_local_undirected(mode="zero")
            clust = dict(enumerate(clust_list))

        # Fill feature matrix using pre-computed neighborhoods
        for i, node in enumerate(nodes):
            val = clust[node]
            feature_matrix[i, 0] = val if not np.isnan(val) else 0.0
            node_neighborhoods = neighborhoods.get(node, {})
            for hop in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(hop, [])
                if hop_nodes:
                    vals = [clust[n] if not np.isnan(clust[n]) else 0.0 for n in hop_nodes]
                    feature_matrix[i, hop] = np.mean(vals)
                else:
                    feature_matrix[i, hop] = 0.0

        columns = [f"clustering_coefficient_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id
        return df[["node_id", "graph_id"] + columns]

    def _compute_local_efficiency(self, graph) -> pd.DataFrame:
        """Compute local efficiency features for all nodes in the graph."""

        # For NetworkX, compute local efficiency using the correct formula:
        # E_local(v) = (1 / (k*(k-1))) * sum(1/d(j,h)) for all pairs j,h in neighbors(v)
        def nx_local_efficiency(G):
            efficiency = {}
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                k = len(neighbors)
                if k < 2:
                    efficiency[node] = 0.0
                    continue

                # Create subgraph of neighbors
                subgraph = G.subgraph(neighbors)

                # Sum of inverse distances for all pairs
                inv_dist_sum = 0.0
                for i in range(k):
                    for j in range(i + 1, k):
                        try:
                            d = nx.shortest_path_length(subgraph, neighbors[i], neighbors[j])
                            inv_dist_sum += 1.0 / d
                        except nx.NetworkXNoPath:
                            pass

                n_pairs = k * (k - 1) / 2.0
                efficiency[node] = inv_dist_sum / n_pairs if n_pairs > 0 else 0.0

            return efficiency

        # For iGraph, compute local efficiency using the correct formula
        def ig_local_efficiency(G):
            result = {}
            for node in range(G.vcount()):
                neighbors = G.neighbors(node)
                k = len(neighbors)
                if k < 2:
                    result[node] = 0.0
                    continue

                # Create subgraph of neighbors
                subgraph = G.subgraph(neighbors)
                if subgraph.ecount() == 0:
                    result[node] = 0.0
                    continue

                # Compute pairwise shortest path lengths
                inv_dist_sum = 0.0
                try:
                    sp_matrix = subgraph.shortest_paths()
                    for i in range(len(sp_matrix)):
                        for j in range(i + 1, len(sp_matrix)):
                            d = sp_matrix[i][j]
                            if d > 0 and d != float("inf"):
                                inv_dist_sum += 1.0 / d
                except Exception:
                    result[node] = 0.0
                    continue

                n_pairs = k * (k - 1) / 2.0
                result[node] = inv_dist_sum / n_pairs if n_pairs > 0 else 0.0

            return result

        return self._compute_structural_feature(graph, nx_local_efficiency, ig_local_efficiency, feature_name="local_efficiency")

    def _compute_lsme(self, graph) -> pd.DataFrame:
        """Compute LSME (Local Spectral Method Embedding) features for all nodes."""
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_nodes = len(nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((n_nodes, n_hops))

        # Get all neighborhoods at once for efficiency
        if isinstance(G, nx.Graph):
            neighborhoods = get_all_neighborhoods_nx(G, n_hops, nodes)
        else:  # igraph
            neighborhoods = get_all_neighborhoods_ig(G, n_hops, nodes)

        # Compute LSME for each node
        for i, node in enumerate(nodes):
            # Get node neighborhoods
            node_neighborhoods = neighborhoods[node]

            # Compute base feature (hop 0)
            if isinstance(G, nx.Graph):
                feature_matrix[i, 0] = G.degree[node]
            else:  # igraph
                feature_matrix[i, 0] = G.degree(node)

            # Compute features for each hop
            for hop in range(1, n_hops):
                if hop in node_neighborhoods and node_neighborhoods[hop]:
                    if isinstance(G, nx.Graph):
                        hop_features = [G.degree[n] for n in node_neighborhoods[hop]]
                    else:  # igraph
                        hop_features = [G.degree(n) for n in node_neighborhoods[hop]]
                    feature_matrix[i, hop] = np.mean(hop_features)

        # Create DataFrame
        columns = [f"lsme_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id

        return df[["node_id", "graph_id"] + columns]

    def _compute_betastar(self, graph) -> pd.DataFrame:
        """
        Betastar community aware node feature

        https://arxiv.org/pdf/2311.04730
        """
        n_hops = self.config.feature_vector_length
        G = graph.G
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        n_nodes = len(nodes)
        feature_matrix = np.zeros((n_nodes, n_hops))
        nodes_in_community = defaultdict(list)

        # Use local RNG to avoid mutating global random state
        local_rng = random.Random(13)

        # get neighborhood mapping
        if isinstance(G, nx.Graph):
            neighborhoods = get_all_neighborhoods_nx(G, n_hops, nodes)
            community_detection = nx.community.louvain_communities(G, seed=13)
            node_community_mapping = {node: i for i, community in enumerate(community_detection) for node in community}
        else:  # igraph
            # Use local RNG for deterministic community detection
            ig.set_random_number_generator(local_rng)
            neighborhoods = get_all_neighborhoods_ig(G, n_hops, nodes)

            # generate communities using leiden
            community_detection = G.community_leiden(objective_function="modularity", n_iterations=10, resolution=1.0)
            node_community_mapping = {k: v for k, v in enumerate(community_detection.membership)}

        for k, v in node_community_mapping.items():
            nodes_in_community[v].append(k)

        for k, v in nodes_in_community.items():
            nodes_in_community[k] = sorted(v)

        if isinstance(G, nx.Graph):
            degrees = {k: v for k, v in G.degree}
        else:
            degrees = G.degree()

        # Build node_to_idx mapping for aggregation
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for i, node in enumerate(nodes):
            if degrees[node] == 0:
                feature_matrix[i, 0] = 0.0
                continue

            if isinstance(G, nx.Graph):
                G_a = G.subgraph(nodes_in_community[node_community_mapping[node]])
                degrees_A = {k: v for k, v in G_a.degree}
                G_num_nodes = G.number_of_nodes()
                G_a_num_nodes = G_a.number_of_nodes()

                beta_star = 2 * (degrees_A[node] / degrees[node] - (G_a_num_nodes - degrees[node]) / G_num_nodes)
            else:
                G_a = community_detection.subgraph(node_community_mapping[node])
                degrees_A = G_a.degree()
                G_num_nodes = G.vcount()
                G_a_num_nodes = G_a.vcount()

                nodes_id_in_subgraph = {n: idx for idx, n in enumerate(nodes_in_community[node_community_mapping[node]])}
                beta_star = 2 * (degrees_A[nodes_id_in_subgraph[node]] / degrees[node] - (G_a_num_nodes - degrees[node]) / G_num_nodes)
            feature_matrix[i, 0] = beta_star

        for i, node in enumerate(nodes):
            node_neighborhoods = neighborhoods.get(node, {})
            for k in range(1, n_hops):
                hop_nodes = node_neighborhoods.get(k, [])
                if hop_nodes:
                    hop_values = [feature_matrix[node_to_idx[n], 0] for n in hop_nodes if n in node_to_idx]
                    feature_matrix[i, k] = np.mean(hop_values) if hop_values else 0.0

        columns = [f"betastar_{i}" for i in range(n_hops)]
        df = pd.DataFrame(feature_matrix, columns=columns)
        df["node_id"] = nodes
        df["graph_id"] = graph.graph_id

        return df[["node_id", "graph_id"] + columns]

    def _normalize_features(self):
        """Normalize all feature columns using StandardScaler."""
        if not self.features_df.empty:
            # Get feature columns (exclude node_id and graph_id)
            feature_cols = [col for col in self.features_df.columns if col not in ["node_id", "graph_id"]]

            if feature_cols:
                # Initialize scaler
                scaler = StandardScaler()

                # Fit and transform the feature columns
                self.features_df[feature_cols] = scaler.fit_transform(self.features_df[feature_cols])

    def compute(self) -> Features:
        feature_dfs = []
        profile_records = []

        # Resolve and validate feature list here, after custom metrics might be registered
        resolved_feature_list = []
        original_feature_list = self.config.feature_list

        if any(f.lower() == "all" for f in original_feature_list):
            # Add all available features
            resolved_feature_list.extend(self.available_features.keys())
            # Add any other explicitly mentioned features that aren't "all"
            for feature_name in original_feature_list:
                if feature_name.lower() != "all" and feature_name not in resolved_feature_list:
                    resolved_feature_list.append(feature_name)
        else:
            resolved_feature_list = list(original_feature_list)

        # Validate resolved list
        for feature in resolved_feature_list:
            if feature not in self.available_features:
                raise ValueError(f"Unknown feature: {feature}. Available features: {list(self.available_features.keys())}")

        graphs = self.graph_collection.graphs
        if self.config.show_progress:
            graphs = tqdm(graphs, desc="Computing structural node features")

        if not resolved_feature_list:  # Check if the list is empty after resolution
            for graph in graphs:
                _df = pd.DataFrame()
                _df["node_id"] = graph.nodes
                _df["graph_id"] = graph.graph_id
                feature_dfs.append(_df)

            features_df = pd.concat(feature_dfs, axis=0, ignore_index=True)
            feature_columns = []
        else:
            if self.config.n_jobs == 1:
                results = [self._compute_graph_node_features(graph, resolved_feature_list) for graph in graphs]
            else:
                joblib_kwargs = self.config.joblib_kwargs or {}
                results = Parallel(n_jobs=self.config.n_jobs, backend=self.config.parallel_backend, **joblib_kwargs)(
                    delayed(self._compute_graph_node_features)(graph, resolved_feature_list) for graph in graphs
                )

            if self.config.profile_features:
                for graph_df, graph_profile_records in results:
                    feature_dfs.append(graph_df)
                    profile_records.extend(graph_profile_records)
                self._log_feature_profile_records(profile_records)
            else:
                feature_dfs = results

            if not feature_dfs:
                raise ValueError("No features were computed. Check if the feature list is empty or if there are no graphs.")

            features_df = pd.concat(feature_dfs, ignore_index=True)

            feature_columns = [col for col in features_df.columns if col not in ["node_id", "graph_id"]]

        if self.config.normalize_features:
            self.features_df = features_df
            self._normalize_features()
            features_df = self.features_df

        return Features(features_df, feature_columns)

    def _compute_graph_node_features(self, graph, resolved_feature_list: List[str]):
        # Clear neighborhood cache for previous graphs to save memory
        # Keep only current graph if it exists
        if len(self._neighborhood_cache) > 1:
            current_cache = self._neighborhood_cache.get(graph.graph_id)
            self._neighborhood_cache.clear()
            if current_cache is not None:
                self._neighborhood_cache[graph.graph_id] = current_cache

        graph_features = []
        profile_records = []
        for feature_name in resolved_feature_list:  # Use the resolved list
            # This check is now redundant due to validation in compute(), but kept for safety
            if feature_name not in self.available_features:
                raise ValueError(f"Unknown feature: {feature_name}")

            start = time.perf_counter()
            feature_df = self.available_features[feature_name](graph)
            duration_s = time.perf_counter() - start
            graph_features.append(feature_df)
            if self.config.profile_features:
                profile_records.append(
                    self._build_feature_profile_record(
                        graph=graph,
                        feature_name=feature_name,
                        feature_df=feature_df,
                        duration_s=duration_s,
                    )
                )

        if not graph_features:  # Handle case where no features were computed for this graph
            graph_df = pd.DataFrame({"node_id": graph.nodes, "graph_id": graph.graph_id})
            return (graph_df, profile_records) if self.config.profile_features else graph_df

        graph_df = graph_features[0]
        for df in graph_features[1:]:
            graph_df = graph_df.merge(df, on=["node_id", "graph_id"])
        if self.config.suffix:
            graph_df.columns = [f"{col}_{self.config.suffix}" if col not in ["node_id", "graph_id"] else col for col in graph_df.columns]

        return (graph_df, profile_records) if self.config.profile_features else graph_df

    def _build_feature_profile_record(self, graph, feature_name: str, feature_df: pd.DataFrame, duration_s: float) -> dict[str, Any]:
        sampled_nodes = graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes
        backend = "sequential" if self.config.n_jobs == 1 else self.config.parallel_backend
        return {
            "graph_id": graph.graph_id,
            "feature": feature_name,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "sampled_node_count": len(sampled_nodes),
            "backend": backend,
            "n_jobs": self.config.n_jobs,
            "duration_s": duration_s,
            "output_rows": len(feature_df),
            "output_columns": len(feature_df.columns),
        }

    def _log_feature_profile_records(self, profile_records: list[dict[str, Any]]) -> None:
        for record in profile_records:
            logger.info(
                "node_feature_profile graph_id=%s feature=%s node_count=%s edge_count=%s sampled_node_count=%s backend=%s "
                "n_jobs=%s duration_s=%.6f output_rows=%s output_columns=%s",
                record["graph_id"],
                record["feature"],
                record["node_count"],
                record["edge_count"],
                record["sampled_node_count"],
                record["backend"],
                record["n_jobs"],
                record["duration_s"],
                record["output_rows"],
                record["output_columns"],
                extra={"node_feature_profile": record},
            )

    def register_metric(self, name: str, func):
        """
        Register a custom metric function to this instance.
        The function should take a graph object and return a DataFrame in the expected format.
        """
        self.available_features[name] = func
