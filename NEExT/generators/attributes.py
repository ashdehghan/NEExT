"""Synthetic node and edge attribute generation."""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ._config import AttributeConfig


class AttributeGenerator:
    """Adds synthetic node and edge attributes to graphs.

    Strategies control how attribute values are distributed across nodes:
    - random_uniform / random_normal: IID features
    - community_correlated: Different distributions per community
    - label_informative: Features carry signal about the graph label
    - degree_correlated: Features correlated with node degree
    - custom: User-provided callable

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def add_node_attributes(self, G: nx.Graph, config: AttributeConfig) -> nx.Graph:
        """Add synthetic node attributes to a graph.

        Args:
            G: Input graph (modified in-place and returned).
            config: Attribute generation configuration.

        Returns:
            The graph with added node attributes.
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        k = config.n_features

        if config.strategy == "random_uniform":
            features = self._rng.uniform(0, 1, size=(n, k))

        elif config.strategy == "random_normal":
            features = self._rng.normal(0, 1, size=(n, k))

        elif config.strategy == "community_correlated":
            features = self._community_correlated_features(G, nodes, k, config.noise_level)

        elif config.strategy == "label_informative":
            features = self._label_informative_features(G, nodes, k, config.noise_level)

        elif config.strategy == "degree_correlated":
            features = self._degree_correlated_features(G, nodes, k, config.noise_level)

        elif config.strategy == "custom":
            G = config.custom_fn(G, self._rng)
            return G

        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")

        # Assign features to nodes
        for i, node in enumerate(nodes):
            for j in range(k):
                G.nodes[node][f"feature_{j}"] = float(features[i, j])

        return G

    def add_edge_attributes(
        self,
        G: nx.Graph,
        n_features: int = 1,
        strategy: str = "random_normal",
    ) -> nx.Graph:
        """Add synthetic edge attributes.

        Args:
            G: Input graph (modified in-place and returned).
            n_features: Number of edge features.
            strategy: "random_uniform" or "random_normal".

        Returns:
            The graph with added edge attributes.
        """
        edges = list(G.edges())
        m = len(edges)

        if strategy == "random_uniform":
            features = self._rng.uniform(0, 1, size=(m, n_features))
        elif strategy == "random_normal":
            features = self._rng.normal(0, 1, size=(m, n_features))
        else:
            raise ValueError(f"Unknown edge strategy: {strategy}")

        for i, (u, v) in enumerate(edges):
            for j in range(n_features):
                G.edges[u, v][f"edge_feature_{j}"] = float(features[i, j])

        return G

    def add_attributes_to_collection(
        self,
        graphs: List[nx.Graph],
        config: AttributeConfig,
    ) -> List[nx.Graph]:
        """Add synthetic node attributes to all graphs in a collection.

        Args:
            graphs: List of nx.Graph objects.
            config: Attribute generation configuration.

        Returns:
            The list of graphs with added attributes.
        """
        for G in graphs:
            self.add_node_attributes(G, config)
        return graphs

    # ── Private strategy implementations ──────────────────────────────

    def _community_correlated_features(
        self,
        G: nx.Graph,
        nodes: List[int],
        k: int,
        noise_level: float,
    ) -> np.ndarray:
        """Generate features with different distributions per community."""
        n = len(nodes)
        communities = self._detect_communities(G, nodes)
        unique_communities = sorted(set(communities.values()))
        n_communities = len(unique_communities)

        # Each community gets a distinct mean vector
        community_means = {c: self._rng.randn(k) * 2 for c in unique_communities}

        features = np.zeros((n, k))
        for i, node in enumerate(nodes):
            c = communities.get(node, 0)
            features[i] = community_means[c] + self._rng.normal(0, noise_level, size=k)

        return features

    def _label_informative_features(
        self,
        G: nx.Graph,
        nodes: List[int],
        k: int,
        noise_level: float,
    ) -> np.ndarray:
        """Generate features that carry signal about graph label."""
        n = len(nodes)
        label = G.graph.get("label", 0)

        # Use label to seed the mean vector (deterministic per label)
        label_rng = np.random.RandomState(int(hash(str(label))) % (2**31))
        mean_vector = label_rng.randn(k) * 2

        features = np.tile(mean_vector, (n, 1)) + self._rng.normal(0, noise_level, size=(n, k))
        return features

    def _degree_correlated_features(
        self,
        G: nx.Graph,
        nodes: List[int],
        k: int,
        noise_level: float,
    ) -> np.ndarray:
        """Generate features correlated with node degree."""
        n = len(nodes)
        degrees = np.array([G.degree(node) for node in nodes], dtype=float)
        max_deg = degrees.max() if degrees.max() > 0 else 1.0
        degrees_norm = degrees / max_deg

        features = np.zeros((n, k))
        for j in range(k):
            # Each feature is a different nonlinear transform of degree + noise
            weight = self._rng.uniform(0.5, 2.0)
            bias = self._rng.uniform(-1, 1)
            features[:, j] = weight * degrees_norm + bias + self._rng.normal(0, noise_level, size=n)

        return features

    def _detect_communities(self, G: nx.Graph, nodes: List[int]) -> Dict[int, int]:
        """Detect or retrieve community assignments for nodes."""
        # Check for SBM block attribute first
        if all("block" in G.nodes.get(n, {}) for n in nodes):
            return {n: G.nodes[n]["block"] for n in nodes}

        # Fall back to greedy modularity communities
        try:
            communities = nx.community.greedy_modularity_communities(G)
            mapping = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    mapping[node] = comm_id
            return mapping
        except Exception:
            # If community detection fails, assign all to community 0
            return {n: 0 for n in nodes}
