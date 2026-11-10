"""Random-walk corpora and a seeded skip-gram trainer shared by the embedders.

Walk kernels are numba (one int32 array, no Python lists of strings), the
corpus is streamed to gensim through a restartable iterable over one interned
string table, and Word2Vec is run single-threaded with an explicit hash
function so a seed reproduces an embedding bit-exactly (gensim's default
`hash()` of str keys depends on PYTHONHASHSEED).
"""

import zlib

import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def _uniform_walks_kernel(indptr, indices, n_walks, length, seed):
    n = len(indptr) - 1
    np.random.seed(seed)
    walks = np.empty((n * n_walks, length), dtype=np.int32)
    row = 0
    for _w in range(n_walks):
        for start in range(n):
            cur = start
            walks[row, 0] = cur
            for t in range(1, length):
                lo = indptr[cur]
                hi = indptr[cur + 1]
                if hi == lo:
                    # isolated node: stay (cannot happen on a connected graph)
                    walks[row, t] = cur
                    continue
                cur = indices[lo + np.random.randint(hi - lo)]
                walks[row, t] = cur
            row += 1
    return walks


@njit(cache=True)
def _has_edge(indptr, indices, u, v):
    lo = indptr[u]
    hi = indptr[u + 1] - 1
    while lo <= hi:
        m = (lo + hi) // 2
        x = indices[m]
        if x == v:
            return True
        if x < v:
            lo = m + 1
        else:
            hi = m - 1
    return False


@njit(cache=True)
def _node2vec_walks_kernel(indptr, indices, n_walks, length, p, q, seed):
    """Second-order node2vec walks by rejection sampling (exact transition
    probabilities; Grover & Leskovec 2016 biases 1/p for returning to the
    previous node, 1 for a common neighbour, 1/q for moving outward)."""
    n = len(indptr) - 1
    np.random.seed(seed)
    walks = np.empty((n * n_walks, length), dtype=np.int32)
    inv_p = 1.0 / p
    inv_q = 1.0 / q
    upper = max(inv_p, max(1.0, inv_q))
    row = 0
    for _w in range(n_walks):
        for start in range(n):
            prev = -1
            cur = start
            walks[row, 0] = cur
            for t in range(1, length):
                lo = indptr[cur]
                hi = indptr[cur + 1]
                if hi == lo:
                    walks[row, t] = cur
                    continue
                if prev < 0:
                    nxt = indices[lo + np.random.randint(hi - lo)]
                else:
                    while True:
                        nxt = indices[lo + np.random.randint(hi - lo)]
                        if nxt == prev:
                            alpha = inv_p
                        elif _has_edge(indptr, indices, prev, nxt):
                            alpha = 1.0
                        else:
                            alpha = inv_q
                        if np.random.random() * upper < alpha:
                            break
                prev = cur
                cur = nxt
                walks[row, t] = cur
            row += 1
    return walks


def node2vec_walks(indptr, indices, n_walks=10, length=80, p=1.0, q=1.0, seed=42):
    """node2vec biased walks, `n_walks` per node, int32[n*n_walks, length].
    p = q = 1 reduces to uniform first-order walks (DeepWalk)."""
    return _node2vec_walks_kernel(indptr, indices, int(n_walks), int(length), float(p), float(q), int(seed))


def uniform_walks(indptr, indices, n_walks=10, length=80, seed=42):
    """First-order uniform random walks, `n_walks` per node, as int32[n*n_walks, length].

    Walk `w*n + v` starts at node v (DeepWalk-style epochs over all nodes)."""
    return _uniform_walks_kernel(indptr, indices, int(n_walks), int(length), int(seed))


class LazyCorpus:
    """Restartable iterable of token lists for gensim, built from an int32 walk
    array and one interned string table (tokens[i] = str(i))."""

    def __init__(self, walks, n_tokens):
        self.walks = walks
        self.tokens = [str(i) for i in range(n_tokens)]

    def __iter__(self):
        tokens = self.tokens
        for row in self.walks:
            yield [tokens[i] for i in row]

    def __len__(self):
        return len(self.walks)


def _crc32_hash(s):
    return zlib.crc32(s.encode("utf-8"))


def train_skipgram(walks, n_keys, dim=16, window=10, seed=42, epochs=1, hs=1, negative=5, workers=1, min_count=0):
    """Skip-gram (gensim Word2Vec sg=1) over an int32 walk array.

    Returns float32[n_keys, dim] with row i = vector of token str(i). Every key
    must occur in the corpus (true for walk starts covering all nodes/types).
    """
    from gensim.models import Word2Vec

    corpus = LazyCorpus(walks, n_keys)
    model = Word2Vec(
        sentences=corpus,
        vector_size=int(dim),
        window=int(window),
        min_count=int(min_count),
        sg=1,
        hs=int(hs),
        negative=int(negative),
        workers=int(workers),
        seed=int(seed),
        epochs=int(epochs),
        hashfxn=_crc32_hash,
    )
    out = np.empty((n_keys, dim), dtype=np.float32)
    missing = 0
    for i in range(n_keys):
        key = str(i)
        if key in model.wv:
            out[i] = model.wv[key]
        else:
            missing += 1
            out[i] = 0.0
    if missing:
        raise AssertionError(f"{missing} of {n_keys} keys never appeared in the walk corpus")
    return out
