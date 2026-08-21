"""struc2vec from scratch (Ribeiro, Saverese & Figueiredo, KDD 2017).

Faithful to the paper's definitions, implemented on flat arrays with numba:

* R_k(u) = nodes at distance exactly k from u; s(R_k(u)) = its ordered degree
  sequence, stored compressed as (degree, count) pairs (OPT1).
* g = DTW with the paper's element cost d(a, b) = max(a, b)/min(a, b) - 1;
  on compressed sequences the cost of matching (a0, a1) with (b0, b1) is
  (max(a0, b0)/min(a0, b0) - 1) * max(a1, b1). We run exact DTW (the compressed
  sequences are short) where the reference code used FastDTW.
* f_k(u, v) = f_{k-1}(u, v) + g(s(R_k(u)), s(R_k(v))), defined only while both
  rings are non-empty (f_{-1} = 0).
* Multilayer context graph: layer k has edge weight w_k(u, v) = exp(-f_k(u, v));
  between layers, w(u_k, u_{k+1}) = log(Gamma_k(u) + e) and w(u_k, u_{k-1}) = 1,
  where Gamma_k(u) counts u's layer-k edges whose weight exceeds the layer's
  average edge weight.
* Walks start in layer 0; with probability `q_stay` the walker moves within the
  layer (neighbour drawn proportional to weight), otherwise it switches layer
  (up with probability w_up / (w_up + w_down); layer switches do not append to
  the walk). The reference implementation stays with probability 0.3.
* OPT2: each node is compared only with the 2*floor(log2 n) + 1 nodes of most
  similar degree. Within equal degree the reference used insertion order; we use
  a seeded random order (on a graph with 11 distinct degrees, insertion order
  would make every node pick the same few lowest-id landmarks).
* OPT3: the hierarchy is capped at k_star layers (`choose_k_star`: every layer
  whose mean ring still holds a node, never beyond `k_cap` = 6).
* Skip-gram (gensim, sg=1, hs=1) over the walks; the reference defaults are
  10 walks x 80, window 10, 5 epochs, dim 128 (we expose `dim`).
"""

import math
import time

import numpy as np
from numba import njit, prange

from .graph_utils import choose_k_star, nx_to_csr
from .walks import train_skipgram

# ---------------------------------------------------------------- rings


@njit(cache=True)
def _bfs_rings(indptr, indices, source, k_star, stamp, stamp_val, queue, depth_end):
    """BFS from source to depth k_star. Fills queue with visited nodes in BFS
    order (source first) and depth_end[d] = end index (exclusive) of ring d in
    queue, for d = 0..k_star. Returns number of rings actually reached."""
    head = 0
    tail = 0
    queue[tail] = source
    tail += 1
    stamp[source] = stamp_val
    depth_end[0] = 1
    depth = 0
    frontier_end = 1
    while depth < k_star:
        while head < frontier_end:
            u = queue[head]
            head += 1
            for j in range(indptr[u], indptr[u + 1]):
                v = indices[j]
                if stamp[v] != stamp_val:
                    stamp[v] = stamp_val
                    queue[tail] = v
                    tail += 1
        depth += 1
        depth_end[depth] = tail
        frontier_end = tail
        if depth_end[depth] == depth_end[depth - 1]:
            # empty ring: all deeper rings are empty too
            for d in range(depth + 1, k_star + 1):
                depth_end[d] = tail
            return depth
    return k_star


@njit(cache=True)
def _count_runs(sorted_vals):
    if len(sorted_vals) == 0:
        return 0
    c = 1
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] != sorted_vals[i - 1]:
            c += 1
    return c


@njit(cache=True)
def _rings_pass(indptr, indices, degree, k_star, fill, ring_offsets, ring_deg, ring_cnt):
    """Pass 1 (fill=False): ring_offsets[u*(k_star+1)+k+1] = compressed length of
    ring k of u (to be cumsummed by the caller). Pass 2 (fill=True): write the
    (degree, count) runs into ring_deg/ring_cnt at the offsets."""
    n = len(indptr) - 1
    stamp = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    depth_end = np.zeros(k_star + 1, dtype=np.int64)
    for u in range(n):
        _bfs_rings(indptr, indices, u, k_star, stamp, u, queue, depth_end)
        start = 0
        for k in range(k_star + 1):
            end = depth_end[k]
            size = end - start
            slot = u * (k_star + 1) + k
            if size == 0:
                if not fill:
                    ring_offsets[slot + 1] = 0
                start = end
                continue
            degs = np.empty(size, dtype=np.int32)
            for i in range(size):
                degs[i] = degree[queue[start + i]]
            degs.sort()
            if not fill:
                ring_offsets[slot + 1] = _count_runs(degs)
            else:
                pos = ring_offsets[slot]
                cur = degs[0]
                cnt = 1
                for i in range(1, size):
                    if degs[i] == cur:
                        cnt += 1
                    else:
                        ring_deg[pos] = cur
                        ring_cnt[pos] = cnt
                        pos += 1
                        cur = degs[i]
                        cnt = 1
                ring_deg[pos] = cur
                ring_cnt[pos] = cnt
            start = end


def compute_rings(indptr, indices, degree, k_star):
    """Compressed ordered degree sequences of R_k(u) for all u, k = 0..k_star.

    Returns (ring_offsets int64[n*(k_star+1)+1], ring_deg int32, ring_cnt int32);
    ring (u, k) occupies [ring_offsets[u*(k_star+1)+k], ring_offsets[u*(k_star+1)+k+1]).
    """
    n = len(indptr) - 1
    ring_offsets = np.zeros(n * (k_star + 1) + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int32)
    _rings_pass(indptr, indices, degree, k_star, False, ring_offsets, dummy, dummy)
    np.cumsum(ring_offsets, out=ring_offsets)
    ring_deg = np.empty(ring_offsets[-1], dtype=np.int32)
    ring_cnt = np.empty(ring_offsets[-1], dtype=np.int32)
    _rings_pass(indptr, indices, degree, k_star, True, ring_offsets, ring_deg, ring_cnt)
    return ring_offsets, ring_deg, ring_cnt


# ---------------------------------------------------------------- candidates (OPT2)


def candidate_pairs(degree, n_candidates, seed=42):
    """OPT2 candidate set: each node vs the `n_candidates` nodes of most similar
    degree (same degree first, in a seeded random order; then alternately the
    next-lower / next-higher degrees). Returns the symmetrised, deduplicated
    pair list as (pair_u, pair_v) int32 arrays with pair_u < pair_v.
    `n_candidates=None` returns all pairs (small graphs / tests only)."""
    n = len(degree)
    if n_candidates is None:
        iu, iv = np.triu_indices(n, k=1)
        return iu.astype(np.int32), iv.astype(np.int32)
    rng = np.random.RandomState(seed)
    order = np.lexsort((rng.permutation(n), degree))
    pos_of = np.empty(n, dtype=np.int64)
    pos_of[order] = np.arange(n)
    c = min(int(n_candidates), n - 1)
    cand = _window_candidates(order, pos_of, c)
    u = np.repeat(np.arange(n, dtype=np.int32), c)
    v = cand.reshape(-1).astype(np.int32)
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    keys = a.astype(np.int64) * n + b
    keys = np.unique(keys)
    return (keys // n).astype(np.int32), (keys % n).astype(np.int32)


@njit(cache=True)
def _window_candidates(order, pos_of, c):
    n = len(order)
    out = np.empty((n, c), dtype=np.int64)
    for u in range(n):
        p = pos_of[u]
        left = p - 1
        right = p + 1
        k = 0
        go_right = False
        while k < c:
            if go_right:
                if right < n:
                    out[u, k] = order[right]
                    right += 1
                    k += 1
                elif left >= 0:
                    out[u, k] = order[left]
                    left -= 1
                    k += 1
            else:
                if left >= 0:
                    out[u, k] = order[left]
                    left -= 1
                    k += 1
                elif right < n:
                    out[u, k] = order[right]
                    right += 1
                    k += 1
            go_right = not go_right
    return out


# ---------------------------------------------------------------- DTW and f_k


@njit(cache=True)
def dtw_compressed(av, ac, bv, bc):
    """Exact DTW between two compressed ordered degree sequences.

    Element cost between runs (a0, a1) and (b0, b1):
    (max(a0, b0) / min(a0, b0) - 1) * max(a1, b1)   [struc2vec OPT1]."""
    la = len(av)
    lb = len(bv)
    prev = np.empty(lb + 1, dtype=np.float64)
    cur = np.empty(lb + 1, dtype=np.float64)
    prev[0] = 0.0
    for j in range(1, lb + 1):
        prev[j] = np.inf
    for i in range(1, la + 1):
        cur[0] = np.inf
        a0 = av[i - 1]
        a1 = ac[i - 1]
        for j in range(1, lb + 1):
            b0 = bv[j - 1]
            b1 = bc[j - 1]
            if a0 >= b0:
                ratio = a0 / b0
            else:
                ratio = b0 / a0
            m = a1 if a1 >= b1 else b1
            cost = (ratio - 1.0) * m
            best = prev[j - 1]
            if prev[j] < best:
                best = prev[j]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = cost + best
        for j in range(lb + 1):
            prev[j] = cur[j]
    return prev[lb]


@njit(parallel=True, cache=True)
def _pair_distances(ring_offsets, ring_deg, ring_cnt, k_star, pair_u, pair_v):
    n_pairs = len(pair_u)
    f = np.empty((n_pairs, k_star + 1), dtype=np.float64)
    for p in prange(n_pairs):
        u = pair_u[p]
        v = pair_v[p]
        acc = 0.0
        alive = True
        for k in range(k_star + 1):
            if not alive:
                f[p, k] = np.inf
                continue
            su = u * (k_star + 1) + k
            sv = v * (k_star + 1) + k
            a0 = ring_offsets[su]
            a1 = ring_offsets[su + 1]
            b0 = ring_offsets[sv]
            b1 = ring_offsets[sv + 1]
            if a1 == a0 or b1 == b0:
                alive = False
                f[p, k] = np.inf
                continue
            acc += dtw_compressed(ring_deg[a0:a1], ring_cnt[a0:a1], ring_deg[b0:b1], ring_cnt[b0:b1])
            f[p, k] = acc
    return f


def structural_distances(ring_offsets, ring_deg, ring_cnt, k_star, pair_u, pair_v):
    """f_k(u, v) for every candidate pair and k = 0..k_star (inf once a ring is empty)."""
    return _pair_distances(ring_offsets, ring_deg, ring_cnt, k_star, pair_u, pair_v)


# ---------------------------------------------------------------- layers


def build_layers(n, k_star, pair_u, pair_v, f):
    """Per-layer weighted CSR adjacency (both directions) plus the inter-layer
    up-weights. Returns dict with arrays indexed [layer]."""
    layer_indptr = []
    layer_indices = []
    layer_cumw = []
    up_w = np.zeros((n, k_star + 1), dtype=np.float64)
    n_edges = []
    for k in range(k_star + 1):
        fk = f[:, k]
        keep = np.isfinite(fk)
        w = np.exp(-fk[keep])
        us = pair_u[keep]
        vs = pair_v[keep]
        src = np.concatenate([us, vs]).astype(np.int64)
        dst = np.concatenate([vs, us]).astype(np.int64)
        ww = np.concatenate([w, w])
        order = np.lexsort((dst, src))
        src, dst, ww = src[order], dst[order], ww[order]
        deg = np.bincount(src, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=indptr[1:])
        # cumulative weights per row for O(log d) sampling
        cumw = np.empty(len(ww), dtype=np.float64)
        for u in range(n):
            a, b = indptr[u], indptr[u + 1]
            if b > a:
                cumw[a:b] = np.cumsum(ww[a:b])
        avg = w.mean() if len(w) else 0.0
        gamma = np.bincount(src[ww > avg], minlength=n) if len(w) else np.zeros(n, dtype=np.int64)
        up_w[:, k] = np.log(gamma + math.e)
        layer_indptr.append(indptr)
        layer_indices.append(dst.astype(np.int32))
        layer_cumw.append(cumw)
        n_edges.append(int(len(w)))
    return {
        "indptr": np.stack(layer_indptr),  # (k_star+1, n+1)
        "indices": _ragged_stack(layer_indices),
        "cumw": _ragged_stack(layer_cumw),
        "up_w": up_w,
        "n_edges": n_edges,
    }


def _ragged_stack(arrays):
    """Stack ragged per-layer arrays into one 2-D array padded with zeros (layers
    differ in edge count; the CSR indptr bounds what is read)."""
    width = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), width), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


# ---------------------------------------------------------------- walks


@njit(cache=True)
def _struc2vec_walks_kernel(layer_indptr, layer_indices, layer_cumw, up_w, k_star, n_walks, length, q_stay, seed):
    n = layer_indptr.shape[1] - 1
    np.random.seed(seed)
    walks = np.empty((n * n_walks, length), dtype=np.int32)
    row = 0
    for _w in range(n_walks):
        for start in range(n):
            v = start
            layer = 0
            walks[row, 0] = v
            t = 1
            guard = 0
            while t < length:
                guard += 1
                lo = layer_indptr[layer, v]
                hi = layer_indptr[layer, v + 1]
                has_edges = hi > lo
                r = np.random.random()
                if r < q_stay and has_edges:
                    total = layer_cumw[layer, hi - 1]
                    x = np.random.random() * total
                    # searchsorted on the row's cumulative weights
                    a = lo
                    b = hi - 1
                    while a < b:
                        m = (a + b) // 2
                        if layer_cumw[layer, m] < x:
                            a = m + 1
                        else:
                            b = m
                    v = layer_indices[layer, a]
                    walks[row, t] = v
                    t += 1
                else:
                    can_up = False
                    if layer < k_star:
                        # reference: move up only if v has edges in the layer above
                        if layer_indptr[layer + 1, v + 1] > layer_indptr[layer + 1, v]:
                            can_up = True
                    wu = up_w[v, layer]
                    if layer == 0:
                        if can_up:
                            layer += 1
                        elif not has_edges:
                            # isolated in every layer: stay put
                            walks[row, t] = v
                            t += 1
                    else:
                        p_up = wu / (wu + 1.0) if can_up else 0.0
                        if np.random.random() < p_up:
                            layer += 1
                        else:
                            layer -= 1
                if guard > 50 * length:
                    # pathological layer ping-pong: pad by staying
                    while t < length:
                        walks[row, t] = v
                        t += 1
            row += 1
    return walks


def struc2vec_walks(layers, k_star, n_walks=10, length=80, q_stay=0.3, seed=42):
    return _struc2vec_walks_kernel(
        layers["indptr"], layers["indices"], layers["cumw"], layers["up_w"], int(k_star), int(n_walks), int(length), float(q_stay), int(seed)
    )


# ---------------------------------------------------------------- driver


def struc2vec_embed(
    G,
    dim=16,
    seed=42,
    k_star=None,
    k_cap=6,
    n_candidates="auto",
    n_walks=10,
    walk_length=80,
    window=10,
    epochs=5,
    q_stay=0.3,
):
    """Full struc2vec pipeline on an nx.Graph (to_networkx contract).

    Returns (emb float32[n, dim], record) where record carries k_star, the ball
    curve, candidate count, pair/edge counts and stage timings."""
    t0 = time.perf_counter()
    indptr, indices, degree = nx_to_csr(G)
    n = len(degree)
    if k_star is None:
        k_star, curve = choose_k_star(indptr, indices, k_cap=k_cap, seed=seed)
    else:
        curve = None
    if n_candidates == "auto":
        n_candidates = 2 * int(math.floor(math.log2(max(n, 2)))) + 1
    ring_offsets, ring_deg, ring_cnt = compute_rings(indptr, indices, degree, k_star)
    t1 = time.perf_counter()
    pair_u, pair_v = candidate_pairs(degree, n_candidates, seed=seed)
    f = structural_distances(ring_offsets, ring_deg, ring_cnt, k_star, pair_u, pair_v)
    t2 = time.perf_counter()
    layers = build_layers(n, k_star, pair_u, pair_v, f)
    walks = struc2vec_walks(layers, k_star, n_walks=n_walks, length=walk_length, q_stay=q_stay, seed=seed)
    t3 = time.perf_counter()
    emb = train_skipgram(walks, n, dim=dim, window=window, seed=seed, epochs=epochs)
    t4 = time.perf_counter()
    record = {
        "k_star": int(k_star),
        "ball_curve": None if curve is None else [float(x) for x in curve],
        "n_candidates": None if n_candidates is None else int(n_candidates),
        "n_pairs": int(len(pair_u)),
        "layer_edges": layers["n_edges"],
        "q_stay": q_stay,
        "n_walks": n_walks,
        "walk_length": walk_length,
        "window": window,
        "epochs": epochs,
        "dim": dim,
        "seed": seed,
        "t_rings": t1 - t0,
        "t_distances": t2 - t1,
        "t_walks": t3 - t2,
        "t_skipgram": t4 - t3,
    }
    return emb, record
