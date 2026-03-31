"""Layer 2: Fluent compositional API for building custom graphs programmatically."""

import copy
from typing import Any, Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from .generator import GraphGenerator


class GraphBuilder:
    """Fluent builder for compositional graph construction.

    All mutation methods return ``self`` for chaining. Call ``.build()``
    to finalize and return the nx.Graph.

    Args:
        seed: Random seed. If None, uses non-deterministic randomness.
    """

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._G: Optional[nx.Graph] = None
        self._label: Union[int, float] = 0

    # ── Initializers ──────────────────────────────────────────────────

    def start_from(self, generator_type: str, **params: Any) -> "GraphBuilder":
        """Initialize from a Layer 1 generator.

        Args:
            generator_type: Name of a registered generator (e.g., "erdos_renyi").
            **params: Parameters for the generator.
        """
        from ._config import GeneratorSpec

        gen = GraphGenerator(seed=int(self._rng.randint(0, 2**31)))
        spec = GeneratorSpec(generator_type=generator_type, params=params, label=self._label)
        self._G = gen.generate_one(spec)
        return self

    def from_graph(self, G: nx.Graph) -> "GraphBuilder":
        """Initialize from an existing nx.Graph (deep copy)."""
        self._G = copy.deepcopy(G)
        return self

    def empty(self, n: int) -> "GraphBuilder":
        """Start from an empty graph with n isolated nodes."""
        self._G = nx.Graph()
        self._G.add_nodes_from(range(n))
        return self

    # ── Graph mutations ───────────────────────────────────────────────

    def plant_community(
        self,
        size: int,
        p_internal: float = 0.8,
        p_bridge: float = 0.05,
        attach_to: Optional[List[int]] = None,
    ) -> "GraphBuilder":
        """Plant a dense community subgraph with bridge edges to existing nodes.

        Args:
            size: Number of nodes in the new community.
            p_internal: Edge probability within the community.
            p_bridge: Probability of each possible bridge edge to existing nodes.
            attach_to: Specific existing nodes to bridge to. If None, bridges
                can connect to any existing node.
        """
        self._ensure_graph()
        existing_nodes = list(self._G.nodes())
        start_id = max(existing_nodes) + 1 if existing_nodes else 0
        new_nodes = list(range(start_id, start_id + size))
        self._G.add_nodes_from(new_nodes)

        # Internal edges
        for i, u in enumerate(new_nodes):
            for v in new_nodes[i + 1 :]:
                if self._rng.random() < p_internal:
                    self._G.add_edge(u, v)

        # Bridge edges
        targets = attach_to if attach_to is not None else existing_nodes
        if targets:
            for u in new_nodes:
                for v in targets:
                    if self._rng.random() < p_bridge:
                        self._G.add_edge(u, v)

        return self

    def attach_motif(
        self,
        motif_type: str,
        size: int = 5,
        attach_to: Union[str, int, List[int]] = "random",
        n_bridges: int = 1,
    ) -> "GraphBuilder":
        """Attach a small motif (clique, cycle, star, path, tree) to the graph.

        Args:
            motif_type: One of "clique", "cycle", "star", "path", "tree".
            size: Number of nodes in the motif.
            attach_to: Where to attach — "random" picks random existing node(s),
                or an int/list of specific node IDs.
            n_bridges: Number of bridge edges connecting motif to existing graph.
        """
        self._ensure_graph()
        existing_nodes = list(self._G.nodes())
        start_id = max(existing_nodes) + 1 if existing_nodes else 0

        # Create motif
        motif_builders = {
            "clique": lambda s: nx.complete_graph(s),
            "cycle": lambda s: nx.cycle_graph(s),
            "star": lambda s: nx.star_graph(s - 1),
            "path": lambda s: nx.path_graph(s),
            "tree": lambda s: nx.random_labeled_tree(s, seed=int(self._rng.randint(0, 2**31))),
        }
        if motif_type not in motif_builders:
            raise ValueError(f"Unknown motif_type '{motif_type}'. Choose from: {list(motif_builders.keys())}")

        motif = motif_builders[motif_type](size)
        mapping = {old: old + start_id for old in motif.nodes()}
        motif = nx.relabel_nodes(motif, mapping)

        self._G.add_nodes_from(motif.nodes())
        self._G.add_edges_from(motif.edges())

        # Create bridge edges
        if existing_nodes:
            motif_nodes = list(motif.nodes())
            if attach_to == "random":
                anchor_nodes = list(self._rng.choice(existing_nodes, size=min(n_bridges, len(existing_nodes)), replace=False))
            elif isinstance(attach_to, int):
                anchor_nodes = [attach_to]
            else:
                anchor_nodes = list(attach_to)

            for i in range(min(n_bridges, len(anchor_nodes))):
                motif_node = motif_nodes[i % len(motif_nodes)]
                self._G.add_edge(anchor_nodes[i], motif_node)

        return self

    def add_hub(
        self,
        degree: int,
        connect_to: Optional[List[int]] = None,
    ) -> "GraphBuilder":
        """Add a high-degree hub node.

        Args:
            degree: Number of edges from the hub to existing nodes.
            connect_to: Specific nodes to connect to. If None, picks random existing nodes.
        """
        self._ensure_graph()
        existing_nodes = list(self._G.nodes())
        hub_id = max(existing_nodes) + 1 if existing_nodes else 0
        self._G.add_node(hub_id)

        if connect_to is not None:
            targets = connect_to[:degree]
        elif existing_nodes:
            targets = list(self._rng.choice(existing_nodes, size=min(degree, len(existing_nodes)), replace=False))
        else:
            targets = []

        for t in targets:
            self._G.add_edge(hub_id, t)

        return self

    def bridge_subgraphs(
        self,
        node_set_a: List[int],
        node_set_b: List[int],
        n_bridges: int = 1,
    ) -> "GraphBuilder":
        """Connect two node groups with bridge edges.

        Args:
            node_set_a: First group of nodes.
            node_set_b: Second group of nodes.
            n_bridges: Number of bridge edges to add.
        """
        self._ensure_graph()
        for _ in range(n_bridges):
            a = node_set_a[self._rng.randint(len(node_set_a))]
            b = node_set_b[self._rng.randint(len(node_set_b))]
            self._G.add_edge(a, b)
        return self

    def rewire_edges(self, fraction: float = 0.1) -> "GraphBuilder":
        """Random edge rewiring: remove edges and add new random ones.

        Args:
            fraction: Fraction of edges to rewire (0.0 to 1.0).
        """
        self._ensure_graph()
        edges = list(self._G.edges())
        n_rewire = max(1, int(len(edges) * fraction))
        nodes = list(self._G.nodes())

        if len(nodes) < 2 or not edges:
            return self

        indices = self._rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)
        for idx in indices:
            u, v = edges[idx]
            if self._G.has_edge(u, v):
                self._G.remove_edge(u, v)
            # Add a new random edge
            new_u = nodes[self._rng.randint(len(nodes))]
            new_v = nodes[self._rng.randint(len(nodes))]
            if new_u != new_v:
                self._G.add_edge(new_u, new_v)

        return self

    def merge_graph(
        self,
        other: nx.Graph,
        bridge_edges: int = 0,
    ) -> "GraphBuilder":
        """Merge another graph into this one, optionally with bridge edges.

        Args:
            other: Graph to merge in (nodes are relabeled to avoid collisions).
            bridge_edges: Number of random bridge edges between the two graphs.
        """
        self._ensure_graph()
        existing_nodes = list(self._G.nodes())
        start_id = max(existing_nodes) + 1 if existing_nodes else 0

        mapping = {old: old + start_id for old in other.nodes()}
        relabeled = nx.relabel_nodes(other, mapping)

        self._G.add_nodes_from(relabeled.nodes(data=True))
        self._G.add_edges_from(relabeled.edges(data=True))

        if bridge_edges > 0 and existing_nodes:
            new_nodes = list(relabeled.nodes())
            for _ in range(bridge_edges):
                a = existing_nodes[self._rng.randint(len(existing_nodes))]
                b = new_nodes[self._rng.randint(len(new_nodes))]
                self._G.add_edge(a, b)

        return self

    def remove_edges(
        self,
        fraction: Optional[float] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> "GraphBuilder":
        """Remove edges to sparsify the graph.

        Args:
            fraction: Fraction of edges to remove randomly.
            edges: Specific edges to remove.
        """
        self._ensure_graph()
        if edges is not None:
            for e in edges:
                if self._G.has_edge(*e):
                    self._G.remove_edge(*e)
        elif fraction is not None:
            all_edges = list(self._G.edges())
            n_remove = max(1, int(len(all_edges) * fraction))
            indices = self._rng.choice(len(all_edges), size=min(n_remove, len(all_edges)), replace=False)
            for idx in indices:
                self._G.remove_edge(*all_edges[idx])
        return self

    def set_label(self, label: Union[int, float]) -> "GraphBuilder":
        """Set the graph classification label."""
        self._label = label
        return self

    def set_node_attribute(self, node: int, key: str, value: Any) -> "GraphBuilder":
        """Set an attribute on a specific node."""
        self._ensure_graph()
        if node not in self._G:
            raise ValueError(f"Node {node} not in graph")
        self._G.nodes[node][key] = value
        return self

    # ── Finalizers ────────────────────────────────────────────────────

    def build(self) -> nx.Graph:
        """Finalize and return the graph."""
        self._ensure_graph()
        G = self._G
        G.graph["label"] = self._label
        # Reset builder state
        self._G = None
        return G

    def build_many(
        self,
        recipe: Callable[["GraphBuilder"], nx.Graph],
        count: int,
        labels: Optional[List[Union[int, float]]] = None,
    ) -> List[nx.Graph]:
        """Batch-build graphs from a recipe function.

        Args:
            recipe: Function that takes a GraphBuilder and returns an nx.Graph
                by calling builder methods and .build().
            count: Number of graphs to build.
            labels: Optional list of labels, one per graph.

        Returns:
            List of nx.Graph objects.
        """
        graphs = []
        for i in range(count):
            builder = GraphBuilder(seed=int(self._rng.randint(0, 2**31)) if self._seed is not None else None)
            if labels is not None:
                builder._label = labels[i]
            G = recipe(builder)
            graphs.append(G)
        return graphs

    # ── Internal ──────────────────────────────────────────────────────

    def _ensure_graph(self) -> None:
        if self._G is None:
            raise RuntimeError("No graph initialized. Call start_from(), from_graph(), or empty() first.")
