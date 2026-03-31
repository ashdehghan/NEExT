"""Layer 1: Registry-based graph generator with extensible plugin architecture.

Built-in generators wrap NetworkX functions, but users can register custom
generators that use any library (igraph, graph-tool, etc.) via the adapter layer.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from ._adapters import adapt_output
from ._config import CollectionSpec, GeneratorSpec

logger = logging.getLogger(__name__)


def _erdos_renyi(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 100)
    p = params.get("p", 0.1)
    seed = params.get("_seed", None)
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _barabasi_albert(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 100)
    m = params.get("m", 2)
    seed = params.get("_seed", None)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _watts_strogatz(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 100)
    k = params.get("k", 4)
    p = params.get("p", 0.1)
    seed = params.get("_seed", None)
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def _stochastic_block_model(params: Dict[str, Any]) -> nx.Graph:
    seed = params.get("_seed", None)
    if "sizes" in params and "p_matrix" in params:
        sizes = params["sizes"]
        p_matrix = params["p_matrix"]
    else:
        # Convenience: derive from p_in, p_out, n_communities, community_size
        n_communities = params.get("n_communities", 3)
        community_size = params.get("community_size", 30)
        p_in = params.get("p_in", 0.3)
        p_out = params.get("p_out", 0.01)
        sizes = [community_size] * n_communities
        p_matrix = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]
    return nx.stochastic_block_model(sizes, p_matrix, seed=seed)


def _planted_partition(params: Dict[str, Any]) -> nx.Graph:
    l = params.get("l", 4)
    k = params.get("k", 25)
    p_in = params.get("p_in", 0.3)
    p_out = params.get("p_out", 0.01)
    seed = params.get("_seed", None)
    return nx.planted_partition_graph(l, k, p_in, p_out, seed=seed)


def _lfr_benchmark(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 250)
    tau1 = params.get("tau1", 3)
    tau2 = params.get("tau2", 1.5)
    mu = params.get("mu", 0.1)
    average_degree = params.get("average_degree", 5)
    min_community = params.get("min_community", 20)
    seed = params.get("_seed", None)
    max_iters = params.get("max_iters", 500)
    try:
        G = nx.LFR_benchmark_graph(
            n,
            tau1,
            tau2,
            mu,
            average_degree=average_degree,
            min_community=min_community,
            seed=seed,
            max_iters=max_iters,
        )
        # LFR assigns frozenset community attributes — convert to list for serialization
        for node in G.nodes():
            if "community" in G.nodes[node]:
                G.nodes[node]["community"] = sorted(G.nodes[node]["community"])
        return G
    except nx.ExceededMaxIterations:
        raise RuntimeError(
            f"LFR benchmark failed to converge with params n={n}, tau1={tau1}, tau2={tau2}, "
            f"mu={mu}, average_degree={average_degree}, min_community={min_community}. "
            "Try adjusting parameters (e.g., increase n, decrease min_community, or change tau values)."
        )


def _random_geometric(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 100)
    radius = params.get("radius", 0.2)
    seed = params.get("_seed", None)
    return nx.random_geometric_graph(n, radius, seed=seed)


def _powerlaw_cluster(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 100)
    m = params.get("m", 2)
    p = params.get("p", 0.1)
    seed = params.get("_seed", None)
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def _configuration_model(params: Dict[str, Any]) -> nx.Graph:
    seed = params.get("_seed", None)
    rng = np.random.RandomState(seed)
    if "degree_sequence" in params:
        deg_seq = list(params["degree_sequence"])
    else:
        # Auto-generate power-law degree sequence
        n = params.get("n", 100)
        exponent = params.get("exponent", 2.5)
        deg_seq = [max(1, int(x)) for x in rng.pareto(exponent - 1, n) + 1]
    # Ensure even sum
    if sum(deg_seq) % 2 != 0:
        deg_seq[rng.randint(len(deg_seq))] += 1
    G = nx.configuration_model(deg_seq, seed=seed)
    # Convert multigraph to simple graph
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _caveman(params: Dict[str, Any]) -> nx.Graph:
    l = params.get("l", 4)
    k = params.get("k", 10)
    return nx.caveman_graph(l, k)


def _connected_caveman(params: Dict[str, Any]) -> nx.Graph:
    l = params.get("l", 4)
    k = params.get("k", 10)
    return nx.connected_caveman_graph(l, k)


def _cycle(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 20)
    return nx.cycle_graph(n)


def _star(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 10)
    return nx.star_graph(n)


def _complete(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 10)
    return nx.complete_graph(n)


def _tree(params: Dict[str, Any]) -> nx.Graph:
    n = params.get("n", 50)
    seed = params.get("_seed", None)
    return nx.random_labeled_tree(n, seed=seed)


def _grid_2d(params: Dict[str, Any]) -> nx.Graph:
    m = params.get("m", 5)
    n = params.get("n", 5)
    G = nx.grid_2d_graph(m, n)
    # Relabel tuple nodes (i,j) to consecutive integers
    mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    return G


# Default built-in registry
_BUILTIN_GENERATORS: Dict[str, Callable] = {
    "erdos_renyi": _erdos_renyi,
    "barabasi_albert": _barabasi_albert,
    "watts_strogatz": _watts_strogatz,
    "stochastic_block_model": _stochastic_block_model,
    "planted_partition": _planted_partition,
    "lfr_benchmark": _lfr_benchmark,
    "random_geometric": _random_geometric,
    "powerlaw_cluster": _powerlaw_cluster,
    "configuration_model": _configuration_model,
    "caveman": _caveman,
    "connected_caveman": _connected_caveman,
    "cycle": _cycle,
    "star": _star,
    "complete": _complete,
    "tree": _tree,
    "grid_2d": _grid_2d,
}


class GraphGenerator:
    """Registry-based graph generator with extensible plugin architecture.

    Built-in generators wrap NetworkX functions, but custom generators can
    be registered to use any library. The adapter layer normalizes all
    output to nx.Graph.

    Args:
        seed: Master random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._generators: Dict[str, Callable] = dict(_BUILTIN_GENERATORS)
        self._output_formats: Dict[str, str] = {k: "networkx" for k in _BUILTIN_GENERATORS}

    def register_generator(
        self,
        name: str,
        fn: Callable,
        output_format: str = "networkx",
    ) -> None:
        """Register a custom generator function.

        Args:
            name: Unique name for the generator.
            fn: Callable that takes a params dict and returns a graph.
            output_format: Format the function returns. One of:
                "networkx", "igraph", "graph_tool", "adjacency_matrix", "edge_list".
        """
        self._generators[name] = fn
        self._output_formats[name] = output_format
        logger.info(f"Registered generator '{name}' (output_format={output_format})")

    @property
    def available_generators(self) -> List[str]:
        """List all registered generator names."""
        return list(self._generators.keys())

    def _derive_seed(self) -> int:
        """Derive a per-graph seed from the master RNG."""
        return int(self._rng.randint(0, 2**31))

    def generate_one(self, spec: GeneratorSpec) -> nx.Graph:
        """Generate a single graph from a GeneratorSpec.

        Args:
            spec: Generator specification.

        Returns:
            nx.Graph with G.graph['label'] set.
        """
        gen_type = spec.generator_type

        if gen_type == "custom":
            fn = spec.params.get("fn")
            if fn is None:
                raise ValueError("'custom' generator_type requires 'fn' in params")
            kwargs = {k: v for k, v in spec.params.items() if k != "fn"}
            kwargs["_seed"] = self._derive_seed()
            G = fn(kwargs)
            G = adapt_output(G, "networkx")
        elif gen_type in self._generators:
            params = dict(spec.params)
            params["_seed"] = self._derive_seed()

            # Handle variable-size graphs
            if spec.n_range is not None:
                n = self._rng.randint(spec.n_range[0], spec.n_range[1] + 1)
                params["n"] = n

            raw = self._generators[gen_type](params)
            output_format = self._output_formats.get(gen_type, "networkx")
            G = adapt_output(raw, output_format)
        else:
            raise ValueError(f"Unknown generator type '{gen_type}'. Available: {self.available_generators}")

        G.graph["label"] = spec.label
        return G

    def generate_many(self, spec: GeneratorSpec, count: int) -> List[nx.Graph]:
        """Generate multiple graphs from the same spec.

        Args:
            spec: Generator specification.
            count: Number of graphs to generate.

        Returns:
            List of nx.Graph objects.
        """
        return [self.generate_one(spec) for _ in range(count)]

    def generate_collection(
        self,
        specs: List[Tuple[GeneratorSpec, int]],
        shuffle: bool = True,
    ) -> List[nx.Graph]:
        """Generate a mixed collection from multiple specs.

        Args:
            specs: List of (GeneratorSpec, count) pairs.
            shuffle: Whether to shuffle the final list.

        Returns:
            List of nx.Graph objects.
        """
        graphs = []
        for spec, count in specs:
            graphs.extend(self.generate_many(spec, count))

        if shuffle:
            self._rng.shuffle(graphs)

        return graphs

    def to_graph_collection(self, graphs: List[nx.Graph], **kwargs) -> Any:
        """Convert generated graphs to a NEExT GraphCollection.

        Args:
            graphs: List of nx.Graph objects.
            **kwargs: Additional arguments passed to GraphIO.load_from_networkx().

        Returns:
            GraphCollection
        """
        from NEExT.io import GraphIO

        graph_io = GraphIO()
        return graph_io.load_from_networkx(graphs, **kwargs)
