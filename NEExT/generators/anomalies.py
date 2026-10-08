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
    - contextual_regime_swap: Flip whole blocks of a ``regime_graph`` to the other regime's block type (an A block
      inside a run of B blocks or the reverse). Degree sequence and every non-anomalous node's 1-hop ego-net
      unchanged (collateral-free); all nodes of a flipped block are labelled.
    - contextual_combination: Flip the block type of rule-bound blocks (needs a graph from
      ``combination_rule_graph``); the block becomes locally normal but breaks the context rule. Degree
      sequence unchanged; anomalies are whole blocks, chosen stratified by context cell.

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
        if config.anomaly_type in ("contextual_combination", "contextual_regime_swap"):
            for node in G.nodes():
                G.nodes[node].setdefault("is_outlier", 0)
            if config.anomaly_type == "contextual_combination":
                self._inject_combination(G, config.fraction)
            else:
                self._inject_regime_blocks(G, config.fraction)
            return G

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

    # ── Contextual anomalies that preserve the degree sequence ────────

    def _inject_regime_blocks(self, G: nx.Graph, fraction: float) -> None:
        """Flip whole blocks of a ``regime_graph`` to the other regime's block type, collateral-free.

        A block is eligible if both ring neighbours belong to its own regime; chosen blocks are pairwise
        non-adjacent. Only edges inside a flipped block change and every block keeps its degree sequence, so the
        1-hop ego-net (induced subgraph) of every node outside the flipped blocks is unchanged. All nodes of a
        flipped block are labelled ``is_outlier = 1``. Records ``G.graph['anomaly_report']`` with requested and
        realised counts (blocks and nodes) and an empty ``collateral`` list.
        """
        if any("block" not in G.nodes[n] for n in G.nodes()) or "regime" not in G.graph.get("blocks", [{}])[0]:
            raise ValueError("contextual_regime_swap needs a graph built with regime_graph")
        blocks = G.graph["blocks"]
        nb = len(blocks)
        n_flip = max(1, int(round(fraction * G.number_of_nodes() / G.graph["block_size"])))
        eligible = [i for i in range(nb) if blocks[(i - 1) % nb]["regime"] == blocks[i]["regime"] == blocks[(i + 1) % nb]["regime"]]
        chosen: List[int] = []
        for i in self._rng.permutation(eligible):
            if len(chosen) >= n_flip:
                break
            if any(min(abs(i - j), nb - abs(i - j)) <= 1 for j in chosen):
                continue
            chosen.append(int(i))
        for i in chosen:
            b = blocks[i]
            new_type = "B" if b["type"] == "A" else "A"
            _set_block_type(G, b, new_type, self._rng)
            b["type"] = new_type
            b["anomalous"] = True
            for n in b["nodes"]:
                G.nodes[n]["is_outlier"] = 1
                G.nodes[n]["block_type"] = new_type
        G.graph["anomaly_report"] = {
            "requested_blocks": int(n_flip),
            "realised_blocks": len(chosen),
            "requested_nodes": int(n_flip * G.graph["block_size"]),
            "realised": int(sum(len(blocks[i]["nodes"]) for i in chosen)),
            "collateral": [],
        }

    def _inject_combination(self, G: nx.Graph, fraction: float) -> None:
        """Flip the type of rule-bound blocks of a ``combination_rule_graph``, stratified by context cell."""
        if any("block" not in G.nodes[n] for n in G.nodes()):
            raise ValueError("contextual_combination needs a graph built with combination_rule_graph")
        blocks = G.graph["blocks"]
        bound = [b for b in blocks if b["role"] == "rule"]
        n_flip = max(1, int(round(fraction * G.number_of_nodes() / G.graph["block_size"])))
        cells: Dict[tuple, List[dict]] = {}
        for b in bound:
            cells.setdefault(tuple(b["context_counts"]), []).append(b)
        # stratified: allocate flips to cells in proportion to their size (largest remainders)
        sizes = {c: len(v) for c, v in cells.items()}
        quota = {c: n_flip * s / len(bound) for c, s in sizes.items()}
        alloc = {c: int(np.floor(q)) for c, q in quota.items()}
        for c in sorted(quota, key=lambda c: quota[c] - alloc[c], reverse=True)[: n_flip - sum(alloc.values())]:
            alloc[c] += 1
        G.graph["anomaly_report"] = {"requested_blocks": int(n_flip), "requested_nodes": int(n_flip * G.graph["block_size"])}
        for c, k in alloc.items():
            chosen = self._rng.choice(len(cells[c]), size=min(k, len(cells[c])), replace=False)
            for idx in chosen:
                b = cells[c][idx]
                new_type = "B" if b["type"] == "A" else "A"
                _set_block_type(G, b, new_type, self._rng)
                b["type"] = new_type
                b["anomalous"] = True
                for n in b["nodes"]:
                    G.nodes[n]["is_outlier"] = 1
                    G.nodes[n]["block_type"] = new_type
        flipped = [b for b in blocks if b.get("anomalous")]
        G.graph["anomaly_report"].update({"realised_blocks": len(flipped), "realised": sum(len(b["nodes"]) for b in flipped), "collateral": []})


# ── Base graphs for the degree-preserving contextual anomalies ────────


def _block_edges(nodes: List[int], kind: str, rng: np.random.RandomState) -> List[tuple]:
    """4-regular block: 'A' = circulant C_n(1, 2) (clustering 0.5), 'B' = random 4-regular (low clustering)."""
    n = len(nodes)
    if kind == "A":
        local = [(i, (i + s) % n) for i in range(n) for s in (1, 2)]
    else:
        g = nx.random_regular_graph(4, n, seed=int(rng.randint(0, 2**31 - 1)))
        local = list(g.edges())
    return [(nodes[a], nodes[b]) for a, b in local]


def _set_block_type(G: nx.Graph, block: dict, kind: str, rng: np.random.RandomState) -> None:
    nodes = block["nodes"]
    internal = [(a, b) for a, b in G.edges(nodes) if a in set(nodes) and b in set(nodes)]
    G.remove_edges_from(internal)
    G.add_edges_from(_block_edges(nodes, kind, rng))


def _ring_of_blocks(types: List[str], block_size: int, rng: np.random.RandomState, extra: Optional[List[dict]] = None) -> nx.Graph:
    """Ring of 4-regular blocks of the given types; port nodes 0 and block_size // 2 link consecutive blocks, so every
    block has the same degree sequence (two degree-5 ports, the rest degree 4) whatever its type."""
    n_blocks = len(types)
    blocks = []
    for i, t in enumerate(types):
        rec = {"index": i, "nodes": list(range(i * block_size, (i + 1) * block_size)), "type": t, "anomalous": False}
        if extra is not None:
            rec.update(extra[i])
        blocks.append(rec)
    G = nx.Graph(blocks=blocks, block_size=block_size)
    for b in blocks:
        G.add_nodes_from(b["nodes"], block=b["index"], block_type=b["type"], **{k: b[k] for k in ("regime", "block_role") if k in b})
        G.add_edges_from(_block_edges(b["nodes"], b["type"], rng))
    half = block_size // 2
    for i in range(n_blocks):
        G.add_edge(blocks[i]["nodes"][half], blocks[(i + 1) % n_blocks]["nodes"][0])
    return G


def regime_graph(n_blocks: int = 60, block_size: int = 20, run_length: int = 10, seed: int = 0) -> nx.Graph:
    """Ring of blocks in alternating regimes of ``run_length`` blocks: regime 'H' = type-A blocks (circulant
    C_n(1, 2), clustering 0.5), regime 'L' = type-B blocks (random 4-regular, low clustering). Every block has the
    same degree sequence. Node attributes ``block``, ``block_type``, ``regime``; ``G.graph['blocks']`` holds the
    per-block records (nodes, type, regime, anomalous)."""
    rng = np.random.RandomState(seed)
    regimes = ["H" if (i // run_length) % 2 == 0 else "L" for i in range(n_blocks)]
    types = ["A" if r == "H" else "B" for r in regimes]
    return _ring_of_blocks(types, block_size, rng, [{"regime": r} for r in regimes])


def combination_rule_graph(n_periods: int = 20, block_size: int = 20, seed: int = 0) -> nx.Graph:
    """Ring of blocks in periods of five: context, context, rule, context, context. Every block is 4-regular
    internally (type 'A' = circulant, high clustering; 'B' = random 4-regular); block port nodes 0 and
    block_size // 2 link to the neighbouring blocks, so every block has the same degree sequence whatever its type.

    Context blocks get a random type. A rule block's type is a non-linear function of its context:
    'A' iff the number of type-A blocks among its two adjacent blocks (band 1) equals the number among the two
    blocks at distance two (band 2); otherwise 'B'. The rule is non-monotone in the band means, so a linear model
    of context cannot express it. Node attributes: ``block``, ``block_role`` ('context' or 'rule'),
    ``block_type``. ``G.graph['blocks']`` holds per-block records (nodes, role, type, context_counts).
    """
    rng = np.random.RandomState(seed)
    n_blocks = 5 * n_periods
    roles = ["context", "context", "rule", "context", "context"] * n_periods
    types = [None] * n_blocks
    for i, r in enumerate(roles):
        if r == "context":
            types[i] = "A" if rng.rand() < 0.5 else "B"
    extra = []
    for i, r in enumerate(roles):
        counts = None
        if r == "rule":
            band1 = [types[(i - 1) % n_blocks], types[(i + 1) % n_blocks]]
            band2 = [types[(i - 2) % n_blocks], types[(i + 2) % n_blocks]]
            counts = (band1.count("A"), band2.count("A"))
            types[i] = "A" if counts[0] == counts[1] else "B"
        extra.append({"role": r, "block_role": r, "context_counts": counts})
    G = _ring_of_blocks(types, block_size, rng, extra)
    return G
