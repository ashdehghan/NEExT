"""Region-holdout splits: train/test centers separated in graph distance.

The transductive-locality control (phase 3): under shared random splits,
nearby train and test centers can share most of their egonet members, so a
classifier may match test centers to neighbors instead of reading
structure. A region-holdout split removes that pathway: test centers are
grown as contiguous BFS regions, and every train center at graph distance
<= buffer_hops from ANY test center is discarded. With buffer_hops >= 2k,
no k-hop bag overlaps across the boundary.

Buffer semantics: distances are measured on the full graph from the test
CENTER set (multi-source BFS); a train center survives iff its distance to
the nearest test center exceeds buffer_hops. buffer_hops=0 keeps every
non-test center (regions still contiguous — a weaker control that only
removes random-split adjacency).
"""

from collections import deque
from typing import Iterator, Sequence

import numpy as np


def adjacency_from_edges(n_nodes: int, src: Sequence[int], dst: Sequence[int]) -> list:
    adj = [[] for _ in range(n_nodes)]
    for a, b in zip(src, dst):
        adj[a].append(b)
        adj[b].append(a)
    return adj


def multi_source_distances(adj: list, sources: Sequence[int]) -> np.ndarray:
    """Hop distance from the nearest source to every node (O(V+E))."""
    dist = np.full(len(adj), -1, dtype=np.int32)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _grow_test_regions(adj: list, center_ids: np.ndarray, n_test_target: int, n_regions: int, rng: np.random.RandomState) -> np.ndarray:
    """Grow contiguous BFS regions from random center seeds until the test
    pool reaches its target size. Returns the selected test center ids."""
    center_set = set(center_ids.tolist())
    test: set = set()
    seeds = rng.permutation(center_ids)
    seed_i = 0
    per_region = int(np.ceil(n_test_target / n_regions))
    for _ in range(n_regions):
        while seed_i < len(seeds) and int(seeds[seed_i]) in test:
            seed_i += 1
        if seed_i >= len(seeds) or len(test) >= n_test_target:
            break
        quota = min(per_region, n_test_target - len(test))
        # BFS outward from the seed, collecting centers until the region quota fills
        start = int(seeds[seed_i])
        seen = {start}
        q = deque([start])
        collected = 0
        while q and collected < quota:
            u = q.popleft()
            if u in center_set and u not in test:
                test.add(u)
                collected += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return np.fromiter(test, dtype=np.int64)


def region_holdout_splits(
    adj: list,
    node_table,
    buffer_hops: int,
    test_frac: float = 0.3,
    n_regions: int = 3,
    n_repeats: int = 10,
    seed: int = 42,
) -> Iterator[tuple]:
    """Yield (name, idx_train, idx_test, stats) region-holdout splits.

    Indices index into node_table row order (the same convention as
    lib.nodeclass.evaluate). stats records retained-center counts so
    shrinking train pools are visible, never silent.
    """
    center_ids = node_table["node_id"].to_numpy()
    id_to_row = {int(nid): i for i, nid in enumerate(center_ids)}
    n_test_target = int(round(test_frac * len(center_ids)))
    for r in range(n_repeats):
        rng = np.random.RandomState(seed + r)
        test_ids = _grow_test_regions(adj, center_ids, n_test_target, n_regions, rng)
        dist = multi_source_distances(adj, test_ids.tolist())
        test_rows = np.array([id_to_row[int(t)] for t in test_ids], dtype=np.int64)
        test_mask = np.zeros(len(center_ids), dtype=bool)
        test_mask[test_rows] = True
        center_dist = dist[center_ids]
        train_mask = (~test_mask) & (center_dist > buffer_hops)
        idx_train = np.nonzero(train_mask)[0]
        stats = {
            "buffer_hops": buffer_hops,
            "repeat": r,
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_dropped_buffer": int(((~test_mask) & ~train_mask).sum()),
        }
        yield (f"B{buffer_hops}_r{r}", idx_train, test_rows, stats)
