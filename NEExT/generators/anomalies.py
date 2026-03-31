"""Anomaly injection for synthetic graphs.

Injects structural and contextual anomalies into graphs and marks affected
nodes with ``G.nodes[n]['is_outlier'] = 1``.
"""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ._config import AnomalyConfig


class AnomalyInjector:
    """Injects anomalies into graphs for outlier detection experiments.

    Anomaly types:
    - structural_hub: Convert nodes into anomalous high-degree hubs.
    - structural_clique: Form dense clique among anomaly nodes.
    - structural_bridge: Nodes that span across detected communities.
    - structural_rewire: Rewire edges of selected nodes to unusual targets.
    - contextual_feature: Push feature values to distribution extremes.
    - contextual_mixed: Both structural + feature anomalies on same nodes.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def inject(self, G: nx.Graph, config: AnomalyConfig) -> nx.Graph:
        """Inject anomalies into a graph.

        All non-anomaly nodes get ``is_outlier = 0``, anomaly nodes get
        ``is_outlier = 1``.

        Args:
            G: Input graph (modified in-place and returned).
            config: Anomaly injection configuration.

        Returns:
            The graph with injected anomalies.
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        n_anomalies = max(1, int(n * config.fraction))

        # Initialize all nodes as non-outlier
        for node in nodes:
            G.nodes[node].setdefault("is_outlier", 0)

        # Select anomaly nodes
        anomaly_indices = self._rng.choice(n, size=min(n_anomalies, n), replace=False)
        anomaly_nodes = [nodes[i] for i in anomaly_indices]

        # Mark anomaly nodes
        for node in anomaly_nodes:
            G.nodes[node]["is_outlier"] = 1

        # Apply anomaly type
        handler = {
            "structural_hub": self._inject_hub,
            "structural_clique": self._inject_clique,
            "structural_bridge": self._inject_bridge,
            "structural_rewire": self._inject_rewire,
            "contextual_feature": self._inject_contextual_feature,
            "contextual_mixed": self._inject_mixed,
        }.get(config.anomaly_type)

        if handler is None:
            raise ValueError(f"Unknown anomaly_type: {config.anomaly_type}")

        handler(G, anomaly_nodes, config.severity)
        return G

    def inject_multiple(self, G: nx.Graph, configs: List[AnomalyConfig]) -> nx.Graph:
        """Apply multiple anomaly injections sequentially."""
        for config in configs:
            G = self.inject(G, config)
        return G

    # ── Structural anomaly implementations ────────────────────────────

    def _inject_hub(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Convert nodes into anomalous high-degree hubs."""
        all_nodes = list(G.nodes())
        for node in anomaly_nodes:
            # Connect to many random nodes (severity controls how many)
            n_new_edges = int(len(all_nodes) * min(severity * 0.1, 0.8))
            targets = self._rng.choice(all_nodes, size=min(n_new_edges, len(all_nodes)), replace=False)
            for t in targets:
                if t != node:
                    G.add_edge(node, t)

    def _inject_clique(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Form dense clique among anomaly nodes."""
        # Connect all anomaly nodes to each other
        for i, u in enumerate(anomaly_nodes):
            for v in anomaly_nodes[i + 1 :]:
                G.add_edge(u, v)

        # Additionally connect to random non-anomaly nodes based on severity
        non_anomaly = [n for n in G.nodes() if n not in set(anomaly_nodes)]
        if non_anomaly:
            n_extra = int(len(non_anomaly) * min(severity * 0.05, 0.5))
            for node in anomaly_nodes:
                targets = self._rng.choice(non_anomaly, size=min(n_extra, len(non_anomaly)), replace=False)
                for t in targets:
                    G.add_edge(node, t)

    def _inject_bridge(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Make anomaly nodes bridge across communities."""
        # Detect communities
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except Exception:
            # If community detection fails, just add random long-range edges
            communities = [set(G.nodes())]

        if len(communities) < 2:
            # Can't bridge if only one community; add random edges instead
            all_nodes = list(G.nodes())
            for node in anomaly_nodes:
                n_edges = int(severity * 3)
                targets = self._rng.choice(all_nodes, size=min(n_edges, len(all_nodes)), replace=False)
                for t in targets:
                    if t != node:
                        G.add_edge(node, t)
            return

        # Connect each anomaly node to multiple communities
        n_edges_per_community = max(1, int(severity))
        for node in anomaly_nodes:
            for comm in communities:
                comm_nodes = list(comm)
                if node in comm:
                    continue
                targets = self._rng.choice(comm_nodes, size=min(n_edges_per_community, len(comm_nodes)), replace=False)
                for t in targets:
                    G.add_edge(node, t)

    def _inject_rewire(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Rewire edges of anomaly nodes to unusual/random targets."""
        all_nodes = list(G.nodes())
        for node in anomaly_nodes:
            neighbors = list(G.neighbors(node))
            n_rewire = max(1, int(len(neighbors) * min(severity * 0.3, 0.9)))

            if not neighbors:
                continue

            rewire_indices = self._rng.choice(len(neighbors), size=min(n_rewire, len(neighbors)), replace=False)
            for idx in rewire_indices:
                old_neighbor = neighbors[idx]
                G.remove_edge(node, old_neighbor)
                # Connect to a random non-neighbor
                new_target = all_nodes[self._rng.randint(len(all_nodes))]
                if new_target != node:
                    G.add_edge(node, new_target)

    # ── Contextual anomaly implementations ────────────────────────────

    def _inject_contextual_feature(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Push feature values of anomaly nodes to distribution extremes."""
        # Find all feature keys
        sample_node = next(iter(G.nodes()))
        feature_keys = [k for k in G.nodes[sample_node] if k.startswith("feature_")]

        if not feature_keys:
            # No features to perturb — add some extreme features
            for node in G.nodes():
                G.nodes[node]["feature_0"] = float(self._rng.normal(0, 1))

            feature_keys = ["feature_0"]

        # Compute feature statistics from normal nodes
        normal_nodes = [n for n in G.nodes() if G.nodes[n].get("is_outlier", 0) == 0]
        for key in feature_keys:
            values = [G.nodes[n].get(key, 0.0) for n in normal_nodes]
            mean = np.mean(values) if values else 0.0
            std = np.std(values) if values else 1.0
            std = max(std, 1e-6)

            # Push anomaly nodes to extremes
            for node in anomaly_nodes:
                direction = self._rng.choice([-1, 1])
                G.nodes[node][key] = float(mean + direction * severity * std)

    def _inject_mixed(self, G: nx.Graph, anomaly_nodes: List[int], severity: float) -> None:
        """Apply both structural and contextual anomalies."""
        # Structural: rewire
        self._inject_rewire(G, anomaly_nodes, severity)
        # Contextual: perturb features
        self._inject_contextual_feature(G, anomaly_nodes, severity)
