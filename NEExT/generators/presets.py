"""Pre-built experiment configurations for common synthetic graph experiments."""

from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

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

    def marker_ring(
        self,
        n_per_class: int = 500,
        n_blocks: int = 16,
        spacings: Sequence[Tuple[int, ...]] = ((3, 13), (6, 10)),
        jitter_prob: float = 0.0,
    ) -> List[nx.Graph]:
        """Compositional bed: a ring of six-node 3-regular blocks whose class is decided by where the markers sit.

        Every block is either a spacer (the triangular prism, C3 x K2) or a marker (K3,3); both are 3-regular on six
        nodes. Block b is joined to block b+1 by one edge from its node 3 to the next block's node 0, so every block
        has the same degree sequence (two nodes of degree 4, four of degree 3) whatever its type, and all graphs
        of all classes have the same degree sequence and size. Class c places markers around the ring with the
        cyclic block spacings ``spacings[c]`` (which must sum to ``n_blocks``), from a random start block and in a
        random direction; the label is c. Classes with equal-length spacing tuples have equal marker counts.

        Nodes carry ``block`` (index along the ring) and ``is_marker`` (0/1); the graph carries ``label`` and
        ``marker_blocks`` (sorted block indices of the markers).

        Args:
            n_per_class: Graphs per class.
            n_blocks: Blocks in the ring.
            spacings: One cyclic spacing tuple per class, e.g. ((3, 13), (6, 10)) (sparse, 2 markers) or
                ((3, 3, 3, 3, 3, 3), (2, 4, 2, 4, 2, 4)) with ``n_blocks=18`` (periodic, 6 markers).
            jitter_prob: Structural noise. Each marker independently, with this probability, swaps places with the
                spacer one block to its left or right (direction at random; no move if that block is a marker), so
                spacings shift by one. Marker count, degree sequence and size are unchanged. Drawn from a separate
                generator seeded ``seed + 1``, so ``jitter_prob=0`` reproduces the noise-free output exactly.
                The graph carries ``n_jittered``.
        """
        if not 0.0 <= jitter_prob <= 1.0:
            raise ValueError(f"jitter_prob must be in [0, 1], got {jitter_prob}")
        jitter_rng = np.random.RandomState(self.seed + 1)
        for sp in spacings:
            if sum(sp) != n_blocks or min(sp) < 1:
                raise ValueError(f"spacing {sp} must be positive and sum to n_blocks={n_blocks}")
        rng = np.random.RandomState(self.seed)
        prism = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
        k33 = [(a, b) for a in (0, 1, 2) for b in (3, 4, 5)]
        graphs = []
        for label, sp in enumerate(spacings):
            for _ in range(n_per_class):
                start = int(rng.randint(n_blocks))
                steps = list(sp) if rng.rand() < 0.5 else list(reversed(sp))
                markers, pos = set(), start
                for step in steps:
                    markers.add(pos % n_blocks)
                    pos += step
                n_jittered = 0
                if jitter_prob > 0:
                    for m in sorted(markers):
                        if jitter_rng.rand() < jitter_prob:
                            target = (m + (1 if jitter_rng.rand() < 0.5 else -1)) % n_blocks
                            if target not in markers:
                                markers.remove(m)
                                markers.add(target)
                                n_jittered += 1
                G = nx.Graph()
                for b in range(n_blocks):
                    base = 6 * b
                    is_marker = int(b in markers)
                    for v in range(6):
                        G.add_node(base + v, block=b, is_marker=is_marker)
                    G.add_edges_from((base + u, base + v) for u, v in (k33 if is_marker else prism))
                    G.add_edge(base + 3, 6 * ((b + 1) % n_blocks))
                G.graph["label"] = label
                G.graph["marker_blocks"] = sorted(markers)
                G.graph["n_jittered"] = n_jittered
                graphs.append(G)
        order = rng.permutation(len(graphs))
        return [graphs[i] for i in order]
