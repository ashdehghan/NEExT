"""Pure helpers for tiling a graph into patches from a community membership vector.

These functions take plain Python data (edge lists, membership lists, label lists)
so they can be tested by hand and reused by any tiling, not only Leiden.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

Label = Union[int, float, str]


def relabel_contiguous(membership: Sequence[int]) -> List[int]:
    """Relabel community ids to 0..P-1, preserving the order of the original ids."""
    ids = sorted(set(membership))
    remap = {old: new for new, old in enumerate(ids)}
    return [remap[m] for m in membership]


def _cut_edge_counts(edges: Sequence[Tuple[int, int]], membership: Sequence[int]) -> Dict[int, Dict[int, int]]:
    """Number of source edges between every pair of distinct communities (symmetric, nested dict)."""
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for u, v in edges:
        a, b = membership[u], membership[v]
        if a != b:
            counts[a][b] += 1
            counts[b][a] += 1
    return counts


def merge_small_communities(
    edges: Sequence[Tuple[int, int]],
    membership: Sequence[int],
    min_size: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Merge every community smaller than ``min_size`` into the neighbouring community it shares the most edges with.

    Communities are processed smallest first (ties by lowest id). After each merge the sizes and the
    cut-edge counts are updated and the absorbing community is re-checked, so a chain of merges ends
    only when no under-size community has a neighbour. An under-size community with no neighbour (a
    separate connected component) is left as it is. The returned membership is relabelled to 0..P-1.

    Args:
        edges: Source edge list over nodes 0..n-1.
        membership: Community id per node (any integers).
        min_size: Minimum number of nodes a community must have to stay unmerged.

    Returns:
        (new_membership, merge_log): the relabelled membership and the list of (absorbed_id, absorbing_id)
        pairs in the order they happened, in the ORIGINAL community ids.
    """
    membership = list(membership)
    sizes: Dict[int, int] = Counter(membership)
    counts = _cut_edge_counts(edges, membership)
    alias: Dict[int, int] = {c: c for c in sizes}  # original id -> current id
    merge_log: List[Tuple[int, int]] = []

    def resolve(c: int) -> int:
        while alias[c] != c:
            c = alias[c]
        return c

    while True:
        candidates = [c for c, s in sizes.items() if s < min_size and counts.get(c)]
        if not candidates:
            break
        small = min(candidates, key=lambda c: (sizes[c], c))
        neighbours = counts[small]
        target = min(neighbours, key=lambda k: (-neighbours[k], k))
        # fold `small` into `target`
        sizes[target] += sizes.pop(small)
        for k, w in neighbours.items():
            if k == target:
                continue
            counts[target][k] += w
            counts[k][target] += w
            del counts[k][small]
        del counts[target][small]
        del counts[small]
        alias[small] = target
        merge_log.append((small, target))

    new_membership = relabel_contiguous([resolve(m) for m in membership])
    return new_membership, merge_log


def compute_patch_label(
    labels: Sequence[Label],
    scheme: str = "majority",
    centre_label: Optional[Label] = None,
    purity_threshold: float = 0.8,
) -> Tuple[Optional[Label], float, Dict[Label, int]]:
    """Derive one label for a patch from the node labels of its members.

    Args:
        labels: Node labels of the patch's (pre-expansion) members.
        scheme: "majority" (most frequent label, ties broken by the lowest label), "centre" (the centre
            node's label), "purity" (1 if the majority fraction is at least ``purity_threshold`` else 0),
            or "distribution" (the histogram is the label; ``graph_label`` then holds the majority label).
        centre_label: Required for "centre".
        purity_threshold: Threshold for "purity".

    Returns:
        (label, majority_fraction, histogram).
    """
    histogram: Dict[Label, int] = dict(Counter(labels))
    if not histogram:
        return None, 0.0, histogram
    majority = min(histogram, key=lambda lab: (-histogram[lab], _sort_key(lab)))
    fraction = histogram[majority] / sum(histogram.values())
    if scheme in ("majority", "distribution"):
        label: Optional[Label] = majority
    elif scheme == "centre":
        if centre_label is None:
            raise ValueError("scheme='centre' needs centre_label")
        label = centre_label
    elif scheme == "purity":
        label = int(fraction >= purity_threshold)
    else:
        raise ValueError(f"Unknown patch label scheme {scheme!r}; use 'majority', 'centre', 'purity' or 'distribution'")
    return label, fraction, histogram


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Order numbers before strings so mixed label types still break ties deterministically."""
    if isinstance(value, str):
        return (1, value)
    return (0, value)
