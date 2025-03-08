from typing import List, Dict, Optional, Union, Literal
import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
from pydantic import BaseModel, Field
from .graph_collection import GraphCollection
from .helper_functions import get_nodes_x_hops_away, get_all_neighborhoods_ig, get_all_neighborhoods_nx
from sklearn.preprocessing import StandardScaler
import logging
from tqdm.auto import tqdm
from .features import Features

# Set up logger
logger = logging.getLogger(__name__)

class NodeFeatureConfig(BaseModel):
    """Configuration for node feature computation"""
    feature_list: List[str]
    feature_vector_length: int
    normalize_features: bool = True
    show_progress: bool = Field(default=True)

class NodeFeatures:
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
        config (NodeFeatureConfig): Configuration for feature computation
        available_features (Dict): Dictionary mapping feature names to computation methods

    Example:
        >>> node_features = NodeFeatures(
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
        show_progress: bool = True
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
            "lsme": self._compute_lsme
        }
        
        # Handle "all" feature option
        if len(feature_list) == 1 and feature_list[0].lower() == "all":
            feature_list = list(self.available_features.keys())
        
        # Validate features
        for feature in feature_list:
            if feature not in self.available_features:
                raise ValueError(f"Unknown feature: {feature}. Available features: {list(self.available_features.keys())}")
        
        self.config = NodeFeatureConfig(
            feature_list=feature_list,
            feature_vector_length=feature_vector_length,
            normalize_features=normalize_features,
            show_progress=show_progress
        )
        self.features_df = None

    def _precompute_neighborhoods(self, graph):
        """Precompute neighborhoods for all nodes in the graph."""
        neighborhoods = {}
        for node in graph.nodes:
            neighborhoods[node] = get_nodes_x_hops_away(
                graph.G, node, self.config.feature_vector_length
            )
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
        df['node_id'] = nodes_to_process
        df['graph_id'] = graph.graph_id
        
        return df[['node_id', 'graph_id'] + columns]

    def _compute_page_rank(self, graph) -> pd.DataFrame:
        """Compute PageRank features for all nodes."""
        return self._compute_structural_feature(
            graph,
            lambda G: nx.pagerank(G),
            lambda G: dict(enumerate(G.pagerank())),
            "page_rank"
        )

    def _compute_degree_centrality(self, graph) -> pd.DataFrame:
        """Compute degree centrality features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph,
            nx.degree_centrality,
            lambda G: {i: deg/float(G.vcount()-1) for i, deg in enumerate(G.degree())},  # Normalize by n-1
            feature_name="degree_centrality"
        )

    def _compute_closeness_centrality(self, graph) -> pd.DataFrame:
        """Compute closeness centrality features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph,
            nx.closeness_centrality,
            lambda G: dict(enumerate(G.closeness())),  # G.closeness() is correct
            feature_name="closeness_centrality"
        )

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
            graph,
            eigenvector_with_fallback,
            lambda G: dict(enumerate(G.eigenvector_centrality())),
            feature_name="eigenvector_centrality"
        )

    def _compute_load_centrality(self, graph) -> pd.DataFrame:
        """Compute load centrality features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph,
            nx.load_centrality,
            lambda G: dict(enumerate(G.betweenness())),  # iGraph uses betweenness as equivalent
            feature_name="load_centrality"
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
        nodes = graph.nodes
        feature_vectors = []
        
        # Calculate average degree
        if isinstance(G, nx.Graph):
            avg_degree = (2 * len(G.edges)) / len(G.nodes)
        else:  # igraph
            avg_degree = (2 * G.ecount()) / G.vcount()
            
        for node in nodes:
            # Get number of neighbors at each distance
            neighbors = get_nodes_x_hops_away(G, node, self.config.feature_vector_length)
            nr_neighbors = [len(neighbors.get(i, [])) for i in range(1, self.config.feature_vector_length + 1)]
            
            # Calculate expansion ratios
            vector = []
            vector.append(nr_neighbors[0] / avg_degree)  # First hop
            
            # Subsequent hops
            for i in range(1, len(nr_neighbors)):
                if nr_neighbors[i-1] == 0:
                    vector.append(0)
                else:
                    vector.append(nr_neighbors[i] / (avg_degree * nr_neighbors[i-1]))
            
            feature_vectors.append(vector)

        # Create DataFrame
        df = pd.DataFrame(feature_vectors)
        df.columns = [f"basic_expansion_{i}" for i in range(len(df.columns))]
        df.insert(0, "node_id", nodes)
        df.insert(0, "graph_id", graph.graph_id)
        return df

    def _compute_feature_parallel(self, feature_name: str) -> pd.DataFrame:
        """
        Compute a specific feature for all graphs in parallel.
        
        Args:
            feature_name: Name of the feature to compute
            
        Returns:
            pd.DataFrame: DataFrame containing the computed feature for all nodes
        """
        graphs = self.graph_collection.graphs
        total_graphs = len(graphs)
        
        # Compute feature for each graph in parallel
        feature_dfs = []
        
        if self.config.show_progress:
            # For process backend, we need to use a different approach
            # since tqdm doesn't work well with ProcessPoolExecutor
            if self.config.parallel_backend == "process":
                # Create a progress bar for this feature's graphs
                graph_pbar = tqdm(
                    total=total_graphs,
                    desc=f"Computing {feature_name}",
                    leave=False,
                    position=1
                )
                
                # Process graphs in parallel with manual progress tracking
                for graph in graphs:
                    feature_df = self.available_features[feature_name](graph)
                    feature_dfs.append(feature_df)
                    graph_pbar.update(1)
                
                graph_pbar.close()
            else:
                # For thread backend, we can use tqdm with executor.map
                feature_func = self.available_features[feature_name]
                feature_dfs = list(executor.map(feature_func, tqdm(
                    graphs,
                    desc=f"Computing {feature_name}",
                    leave=False,
                    position=1,
                    disable=not self.config.show_progress
                )))
        else:
            # No progress bar, just use executor.map
            feature_func = self.available_features[feature_name]
            feature_dfs = list(executor.map(feature_func, graphs))
        
        # Concatenate results
        if feature_dfs:
            return pd.concat(feature_dfs, ignore_index=True)
        return pd.DataFrame()

    def _compute_betweenness_centrality(self, graph) -> pd.DataFrame:
        """Compute betweenness centrality features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph,
            nx.betweenness_centrality,
            lambda G: dict(enumerate(G.betweenness())),
            feature_name="betweenness_centrality"
        )

    def _compute_clustering_coefficient(self, graph) -> pd.DataFrame:
        """Compute clustering coefficient features for all nodes in the graph."""
        return self._compute_structural_feature(
            graph,
            nx.clustering,
            lambda G: dict(enumerate(G.transitivity_local_undirected(mode="zero"))),
            feature_name="clustering_coefficient"
        )

    def _compute_local_efficiency(self, graph) -> pd.DataFrame:
        """Compute local efficiency features for all nodes in the graph."""
        # For NetworkX, we need to compute local efficiency manually
        def nx_local_efficiency(G):
            efficiency = {}
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                if len(neighbors) < 2:
                    efficiency[node] = 0.0
                    continue
                    
                # Create subgraph of neighbors
                subgraph = G.subgraph(neighbors)
                if len(subgraph.edges()) == 0:
                    efficiency[node] = 0.0
                    continue
                    
                # Compute efficiency as inverse of average shortest path length
                path_lengths = []
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        try:
                            path_length = nx.shortest_path_length(subgraph, neighbors[i], neighbors[j])
                            path_lengths.append(path_length)
                        except nx.NetworkXNoPath:
                            pass
                            
                if path_lengths:
                    avg_path_length = sum(path_lengths) / len(path_lengths)
                    efficiency[node] = 1.0 / avg_path_length if avg_path_length > 0 else 0.0
                else:
                    efficiency[node] = 0.0
                    
            return efficiency
        
        # For iGraph, we can use built-in methods
        def ig_local_efficiency(G):
            result = {}
            for i, node in enumerate(range(G.vcount())):
                neighbors = G.neighbors(node)
                if len(neighbors) < 2:
                    result[i] = 0.0
                    continue
                    
                # Create subgraph of neighbors
                subgraph = G.subgraph(neighbors)
                if subgraph.ecount() == 0:
                    result[i] = 0.0
                    continue
                    
                # Compute efficiency using average path length
                try:
                    avg_path = subgraph.average_path_length(directed=False)
                    result[i] = 1.0 / avg_path if avg_path > 0 else 0.0
                except:
                    result[i] = 0.0
                    
            return result
        
        return self._compute_structural_feature(
            graph,
            nx_local_efficiency,
            ig_local_efficiency,
            feature_name="local_efficiency"
        )

    def _compute_lsme(self, graph) -> pd.DataFrame:
        """Compute LSME (Local Spectral Method Embedding) features for all nodes."""
        G = graph.G
        n_nodes = len(graph.nodes)
        n_hops = self.config.feature_vector_length
        feature_matrix = np.zeros((n_nodes, n_hops))
        
        # Get all neighborhoods at once for efficiency
        if isinstance(G, nx.Graph):
            neighborhoods = get_all_neighborhoods_nx(G, n_hops)
        else:  # igraph
            neighborhoods = get_all_neighborhoods_ig(G, n_hops)
        
        # Compute LSME for each node
        for i, node in enumerate(graph.nodes):
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
        df['node_id'] = graph.nodes
        df['graph_id'] = graph.graph_id
        
        return df[['node_id', 'graph_id'] + columns]

    def _normalize_features(self):
        """Normalize all feature columns using StandardScaler."""
        if not self.features_df.empty:
            # Get feature columns (exclude node_id and graph_id)
            feature_cols = [col for col in self.features_df.columns 
                           if col not in ['node_id', 'graph_id']]
            
            if feature_cols:
                # Initialize scaler
                scaler = StandardScaler()
                
                # Fit and transform the feature columns
                self.features_df[feature_cols] = scaler.fit_transform(
                    self.features_df[feature_cols]
                )

    def compute(self) -> Features:
        """Compute all requested features for all graphs."""
        feature_dfs = []
        
        # Process each graph sequentially
        graphs = self.graph_collection.graphs
        if self.config.show_progress:
            graphs = tqdm(graphs, desc="Computing node features")
        
        for graph in graphs:
            # Compute each feature for this graph
            graph_features = []
            for feature_name in self.config.feature_list:
                if feature_name not in self.available_features:
                    raise ValueError(f"Unknown feature: {feature_name}")
                
                feature_df = self.available_features[feature_name](graph)
                graph_features.append(feature_df)
            
            # Merge all features for this graph
            if graph_features:
                graph_df = graph_features[0]
                for df in graph_features[1:]:
                    graph_df = graph_df.merge(df, on=['node_id', 'graph_id'])
                
                feature_dfs.append(graph_df)
        
        # Combine features from all graphs
        if not feature_dfs:
            raise ValueError("No features were computed. Check if the feature list is empty or if there are no graphs.")
            
        features_df = pd.concat(feature_dfs, ignore_index=True)
        
        # Get feature columns (excluding node_id and graph_id)
        feature_columns = [col for col in features_df.columns 
                        if col not in ['node_id', 'graph_id']]
        
        return Features(features_df, feature_columns)
