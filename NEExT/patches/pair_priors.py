"""Pairwise quantities between the patches of one graph (family 1: proximity).

Given a source graph and a membership vector (node -> patch index 0..P-1), compute dense P x P arrays:

- ``adjacent``: True where at least one source edge joins the two patches.
- ``cut_edges``: number of source edges joining the two patches (each edge counted once).
- ``hop_distance``: shortest-path distance between the patches on the coarse patch graph, whose
  vertices are the patches and whose edges join adjacent patches. 0 on the diagonal, -1 when no path.

All arrays are symmetric. The coarse graph is built from the graph's edge list, so either backend works.
"""

from typing import Dict, Sequence, Tuple

import igraph as ig
import numpy as np

from NEExT.graphs import Graph


def _check_membership(graph: Graph, membership: Sequence[int]) -> int:
    n = len(graph.nodes)
    if len(membership) != n:
        raise ValueError(f"membership has {len(membership)} entries but the graph has {n} nodes")
    if graph.nodes != list(range(n)):
        raise ValueError("pair priors need nodes indexed 0..n-1 (load with reindex_nodes=True)")
    membership_arr = np.asarray(membership, dtype=np.int64)
    if membership_arr.min() < 0:
        raise ValueError("membership values must be non-negative patch indices")
    n_patches = int(membership_arr.max()) + 1
    if set(membership_arr.tolist()) != set(range(n_patches)):
        raise ValueError("membership values must be contiguous 0..P-1")
    return n_patches


def cut_edge_matrix(graph: Graph, membership: Sequence[int]) -> np.ndarray:
    """P x P integer matrix of source edges between patches (symmetric, zero diagonal)."""
    n_patches = _check_membership(graph, membership)
    membership_arr = np.asarray(membership, dtype=np.int64)
    cut = np.zeros((n_patches, n_patches), dtype=np.int64)
    if graph.edges:
        edges = np.asarray(graph.edges, dtype=np.int64)
        a = membership_arr[edges[:, 0]]
        b = membership_arr[edges[:, 1]]
        off = a != b
        np.add.at(cut, (a[off], b[off]), 1)
        np.add.at(cut, (b[off], a[off]), 1)
    return cut


def coarse_patch_graph(graph: Graph, membership: Sequence[int]) -> Tuple[ig.Graph, np.ndarray]:
    """The coarse graph over patches: one vertex per patch, an edge where ``cut_edges > 0`` (weight = count).

    Returns:
        (coarse igraph.Graph with an edge attribute "weight", the cut-edge matrix).
    """
    cut = cut_edge_matrix(graph, membership)
    i_idx, j_idx = np.nonzero(np.triu(cut, k=1))
    coarse = ig.Graph(n=cut.shape[0], edges=list(zip(i_idx.tolist(), j_idx.tolist())))
    coarse.es["weight"] = cut[i_idx, j_idx].tolist()
    return coarse, cut


def compute_pair_priors(graph: Graph, membership: Sequence[int]) -> Dict[str, np.ndarray]:
    """Family-1 pair priors for the patches of one graph.

    Args:
        graph: Source graph with nodes 0..n-1.
        membership: Patch index per node (0..P-1), e.g. ``EgonetCollection.patch_memberships[graph_id]``.

    Returns:
        Dict with keys "adjacent" (bool), "cut_edges" (int64) and "hop_distance" (int64, -1 = no path),
        each a symmetric P x P array.
    """
    coarse, cut = coarse_patch_graph(graph, membership)
    hop = np.asarray(coarse.distances(), dtype=np.float64)
    hop[~np.isfinite(hop)] = -1
    return {
        "adjacent": cut > 0,
        "cut_edges": cut,
        "hop_distance": hop.astype(np.int64),
    }
