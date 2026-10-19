from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, List, Optional, Set, Union

import igraph as ig
import networkx as nx
import numpy as np


def divide_chunks(list, chunks):
    for i in range(0, len(list), chunks):
        yield list[i : i + chunks]


def build_adjacency_lists(G: Union[nx.Graph, ig.Graph]) -> Dict[int, np.ndarray]:
    """
    Build a node -> neighbor-array adjacency map for either backend.

    Random-walk simulation steps through this structure instead of the
    backend's own walk routines: igraph's random_walk draws from a
    process-global C RNG, which would break the library's self-contained
    randomness guarantee, and NetworkX has no walk routine at all. A plain
    dict of numpy arrays gives both backends identical, seedable behavior.

    Args:
        G: Graph object (NetworkX or iGraph)

    Returns:
        Dict[int, np.ndarray]: Sorted neighbor IDs per node.
    """
    if isinstance(G, nx.Graph):
        return {node: np.fromiter(sorted(G.neighbors(node)), dtype=np.int64) for node in G.nodes()}
    return {v.index: np.fromiter(sorted(G.neighbors(v.index)), dtype=np.int64) for v in G.vs}


def random_walk_visit_counts(
    adjacency: Dict[int, np.ndarray],
    start: int,
    n_steps: int,
    restart_prob: float,
    rng: np.random.RandomState,
) -> Dict[int, int]:
    """
    Visit counts of one random walk with restart, started at one node.

    The walk starts at `start` (counted as one visit) and takes `n_steps`
    steps. At every step it jumps back to `start` with probability
    `restart_prob`, otherwise it moves to a uniformly random neighbor; a walker
    on an isolated node jumps back to `start`. Every position reached is
    counted, so the visit frequencies converge to the personalized PageRank
    vector of `start` with teleport probability `restart_prob` as `n_steps`
    grows. All randomness comes from the caller's RNG instance.

    Args:
        adjacency: Output of build_adjacency_lists for the parent graph
        start: Center node ID the walk starts (and restarts) from
        n_steps: Number of steps of the walk
        restart_prob: Per-step probability of jumping back to `start`
        rng: Self-contained random state (never the global RNG)

    Returns:
        Dict[int, int]: Node ID -> number of visits (sums to n_steps + 1).
    """
    counts: Dict[int, int] = defaultdict(int)
    counts[start] = 1
    restart_draws = rng.random_sample(n_steps) < restart_prob if restart_prob > 0.0 else np.zeros(n_steps, dtype=bool)
    step_draws = rng.random_sample(n_steps)
    current = start
    for i in range(n_steps):
        neighbors = adjacency.get(current)
        if restart_draws[i] or neighbors is None or len(neighbors) == 0:
            current = start
        else:
            current = int(neighbors[int(step_draws[i] * len(neighbors))])
        counts[current] += 1
    return dict(counts)


def hop_distances(adjacency: Dict[int, np.ndarray], source: int, targets) -> Dict[int, int]:
    """
    Hop distance from `source` to each node in `targets`, by breadth-first search over `adjacency`
    (the output of build_adjacency_lists) that stops once every target is reached.

    Args:
        adjacency: Node -> neighbor array of the parent graph
        source: Start node
        targets: Nodes whose distance is wanted (all must be reachable from `source`)

    Returns:
        Dict[int, int]: Target node -> hop distance (source -> 0).
    """
    remaining = set(targets)
    found: Dict[int, int] = {}
    if source in remaining:
        found[source] = 0
        remaining.discard(source)
    seen = {source}
    frontier = [source]
    depth = 0
    while remaining and frontier:
        depth += 1
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
                    if neighbor in remaining:
                        found[neighbor] = depth
                        remaining.discard(neighbor)
        frontier = next_frontier
    if remaining:
        raise ValueError(f"{len(remaining)} target(s) unreachable from node {source}")
    return found


def get_numb_of_nb_x_hops_away(G: Union[nx.Graph, ig.Graph], node: int, max_hop_length: int) -> List[int]:
    """
    Compute the number of neighbors x hops away from a given node.
    Supports both NetworkX and iGraph backends.

    Args:
        G: Graph object (NetworkX or iGraph)
        node: Source node ID
        max_hop_length: Maximum hop distance to consider

    Returns:
        List[int]: List where index i contains number of nodes i+1 hops away
    """
    hop_dict = get_nodes_x_hops_away(G, node, max_hop_length)
    return [len(hop_dict.get(i, set())) for i in range(1, max_hop_length + 1)]


def get_nodes_x_hops_away(G: Union[nx.Graph, ig.Graph], node: int, max_hop_length: int) -> Dict[int, Set[int]]:
    """Efficiently get nodes at each hop distance from a given node, up to max_hop_length."""
    if isinstance(G, nx.Graph):
        hop_dict = defaultdict(set)
        visited = {node}
        queue = deque([(node, 0)])
        while queue:
            current, hop = queue.popleft()
            if hop == max_hop_length:
                continue
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    hop_dict[hop + 1].add(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
        return dict(hop_dict)
    else:
        hop_dict = {}
        bfs = G.bfsiter(node, advanced=True)
        for v, depth, _ in bfs:
            if depth == 0:
                continue
            if depth > max_hop_length:
                break
            hop_dict.setdefault(depth, set()).add(v.index)
        return hop_dict


def get_all_neighborhoods_nx(G, max_hops: int, nodes_to_process: Optional[List[int]] = None) -> Dict:
    """
    Get neighborhoods for all specified nodes in a NetworkX graph.

    Args:
        G: NetworkX graph
        max_hops: Maximum number of hops to consider
        nodes_to_process: List of nodes to get neighborhoods for. If None, process all nodes.

    Returns:
        Dict: Dictionary mapping each node to its neighborhood at each hop
    """
    if nodes_to_process is None:
        nodes_to_process = list(G.nodes())

    neighborhoods = {}
    for node in nodes_to_process:
        neighborhoods[node] = get_nodes_x_hops_away(G, node, max_hops)
    return neighborhoods


def get_all_neighborhoods_ig(G, max_hops: int, nodes_to_process: Optional[List[int]] = None) -> Dict:
    """
    Get neighborhoods for all specified nodes in an iGraph graph.

    Args:
        G: iGraph graph
        max_hops: Maximum number of hops to consider
        nodes_to_process: List of nodes to get neighborhoods for. If None, process all nodes.

    Returns:
        Dict: Dictionary mapping each node to its neighborhood at each hop
    """
    if nodes_to_process is None:
        nodes_to_process = list(range(G.vcount()))

    neighborhoods = {}
    for node in nodes_to_process:
        neighborhoods[node] = {}
        # Reuse the previous hop's ball instead of recomputing it, halving the
        # per-node BFS calls; the set construction sequence is unchanged so the
        # resulting difference sets (and their list order) are identical.
        prev = set(G.neighborhood(node, order=0))
        for hop in range(1, max_hops + 1):
            cur = set(G.neighborhood(node, order=hop))
            # Get nodes at exactly hop distance
            nodes_at_hop = cur - prev
            if nodes_at_hop:
                neighborhoods[node][hop] = list(nodes_at_hop)
            prev = cur
    return neighborhoods


def get_specific_in_community_degree(G, node_id, community_partition: List[List[int]], community_id: int) -> int:
    """
    This method will compute the community degree of a node for a specific community.

    Returns an integer, which is the in-community degree of the node for the specified community.
    """
    neighbors = list(G.neighbors(node_id))
    return len([n for n in neighbors if n in community_partition[community_id]])


def get_all_in_community_degrees(G, node_id, community_partition: List[List[int]]) -> List[int]:
    """
    This method will compute the community degree of a node for each of the communities.

    Returns a list of integers,
    where each integer is the in-community degree of the node for that community.
    """
    in_community_degrees = []
    for i, community in enumerate(community_partition):
        in_community_degrees.append(get_specific_in_community_degree(G, node_id, community_partition, i))

    return in_community_degrees


def get_own_in_community_degree(G, node_id, community_partition: List[List[int]]) -> int:
    """
    This method will compute the community degree of a node for the community it is in.

    Returns an integer, which is the in-community degree of the node for its community.
    """
    for i, community in enumerate(community_partition):
        if node_id in community:
            return get_specific_in_community_degree(G, node_id, community_partition, i)


def get_specific_community_volume(G, community_partition: List[List[int]], community_id: int) -> int:
    """
    This method will compute the volume of a specific community in the graph.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns an integer, which is the volume of the community.
    """
    return sum([G.degree[node] for node in community_partition[community_id]])


def get_all_community_volumes(G, community_partition: List[List[int]]) -> List[int]:
    """
    This method will compute the volume of each community in the graph.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns a list of integers, where each integer is the volume of the community.
    """

    community_volumes = []
    for i in range(len(community_partition)):
        community_volumes.append(get_specific_community_volume(G, community_partition, i))

    return community_volumes


def get_own_community_volume(G, node_id: int, community_partition: List[List[int]]) -> int:
    """
    This method will compute the volume of the community the node is in.
    The volume is the sum of all the degrees of the nodes in the community.

    Returns an integer, which is the volume of the community.
    """
    for i, community in enumerate(community_partition):
        if node_id in community:
            return get_specific_community_volume(G, community_partition, i)
