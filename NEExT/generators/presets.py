"""Pre-built experiment configurations for common synthetic graph experiments."""

from typing import List, Optional

import networkx as nx

from ._config import AnomalyConfig, AttributeConfig, GeneratorSpec
from .anomalies import AnomalyInjector
from .attributes import AttributeGenerator
from .generator import GraphGenerator


class SyntheticPresets:
    """Pre-built experiment configurations that return ``List[nx.Graph]``.

    All presets produce graphs with ``G.graph['label']`` set, ready for
    ``nxt.load_from_networkx()``.

    Args:
        seed: Master random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def binary_classification_er_vs_ba(
        self,
        n_per_class: int = 50,
        n_nodes: int = 50,
    ) -> List[nx.Graph]:
        """Erdos-Renyi (label=0) vs Barabasi-Albert (label=1).

        These have very different degree distributions, so classification
        accuracy should be high.
        """
        gen = GraphGenerator(seed=self.seed)
        er_spec = GeneratorSpec(generator_type="erdos_renyi", params={"n": n_nodes, "p": 0.1}, label=0)
        ba_spec = GeneratorSpec(generator_type="barabasi_albert", params={"n": n_nodes, "m": 2}, label=1)
        return gen.generate_collection([(er_spec, n_per_class), (ba_spec, n_per_class)], shuffle=True)

    def multiclass_topology(
        self,
        n_per_class: int = 30,
        n_nodes: int = 50,
    ) -> List[nx.Graph]:
        """4-class topology experiment: ER vs BA vs WS vs Geometric."""
        gen = GraphGenerator(seed=self.seed)
        specs = [
            (GeneratorSpec(generator_type="erdos_renyi", params={"n": n_nodes, "p": 0.1}, label=0), n_per_class),
            (GeneratorSpec(generator_type="barabasi_albert", params={"n": n_nodes, "m": 2}, label=1), n_per_class),
            (GeneratorSpec(generator_type="watts_strogatz", params={"n": n_nodes, "k": 4, "p": 0.1}, label=2), n_per_class),
            (GeneratorSpec(generator_type="random_geometric", params={"n": n_nodes, "radius": 0.3}, label=3), n_per_class),
        ]
        return gen.generate_collection(specs, shuffle=True)

    def community_strength_gradient(
        self,
        n_per_level: int = 20,
        n_nodes: int = 90,
        n_communities: int = 3,
    ) -> List[nx.Graph]:
        """SBM graphs with varying community clarity (5 levels).

        Label corresponds to community strength level (0 = weakest, 4 = strongest).
        """
        gen = GraphGenerator(seed=self.seed)
        community_size = n_nodes // n_communities
        # p_in increases, p_out decreases across levels
        levels = [
            (0.10, 0.08),  # Level 0: almost random
            (0.20, 0.06),  # Level 1: slight communities
            (0.30, 0.04),  # Level 2: moderate
            (0.40, 0.02),  # Level 3: clear
            (0.50, 0.01),  # Level 4: very strong
        ]
        specs = []
        for label, (p_in, p_out) in enumerate(levels):
            spec = GeneratorSpec(
                generator_type="stochastic_block_model",
                params={"n_communities": n_communities, "community_size": community_size, "p_in": p_in, "p_out": p_out},
                label=label,
            )
            specs.append((spec, n_per_level))
        return gen.generate_collection(specs, shuffle=True)

    def scalability_test(
        self,
        sizes: Optional[List[int]] = None,
        n_per_size: int = 10,
        generator_type: str = "erdos_renyi",
    ) -> List[nx.Graph]:
        """Graphs of increasing sizes for benchmarking pipeline performance.

        Label encodes the size class index.
        """
        if sizes is None:
            sizes = [50, 100, 200, 500, 1000]
        gen = GraphGenerator(seed=self.seed)
        specs = []
        for label, n in enumerate(sizes):
            params = {"n": n, "p": min(0.1, 10.0 / n)}  # keep density roughly constant
            spec = GeneratorSpec(generator_type=generator_type, params=params, label=label)
            specs.append((spec, n_per_size))
        return gen.generate_collection(specs, shuffle=False)

    def variable_size_classification(
        self,
        n_per_class: int = 50,
        class_a_range: tuple = (20, 50),
        class_b_range: tuple = (80, 150),
    ) -> List[nx.Graph]:
        """Test size-invariance: two classes with different size distributions.

        Class A: small ER graphs. Class B: large BA graphs.
        """
        gen = GraphGenerator(seed=self.seed)
        specs = [
            (GeneratorSpec(generator_type="erdos_renyi", params={"p": 0.15}, label=0, n_range=class_a_range), n_per_class),
            (GeneratorSpec(generator_type="barabasi_albert", params={"m": 3}, label=1, n_range=class_b_range), n_per_class),
        ]
        return gen.generate_collection(specs, shuffle=True)

    def egonet_anomaly_detection(
        self,
        n_graphs: int = 5,
        n_nodes: int = 200,
        anomaly_fraction: float = 0.05,
        anomaly_type: str = "structural_hub",
    ) -> List[nx.Graph]:
        """Graphs with planted anomalous nodes for egonet-based outlier detection.

        Each graph is an SBM with injected anomalies. Anomalous nodes are marked
        with ``G.nodes[n]['is_outlier'] = 1``.
        """
        gen = GraphGenerator(seed=self.seed)
        injector = AnomalyInjector(seed=self.seed)
        config = AnomalyConfig(anomaly_type=anomaly_type, fraction=anomaly_fraction)

        spec = GeneratorSpec(
            generator_type="stochastic_block_model",
            params={"n_communities": 4, "community_size": n_nodes // 4, "p_in": 0.3, "p_out": 0.02},
            label=0,
        )
        graphs = gen.generate_many(spec, n_graphs)
        for G in graphs:
            injector.inject(G, config)
        return graphs
