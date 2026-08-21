"""One entry point for every external node-embedding baseline.

`embed(method, G, seed, dim, nodes)` returns a node_id-keyed DataFrame plus a
record (family, parameters, bibkey, timings) so the runner can persist exactly
what was fit. Families:

* proximity — DeepWalk / node2vec: biased walks from `lib.nodeclass.walks`
  (numba rejection sampling; exact node2vec transition probabilities) + gensim
  skip-gram through `train_skipgram` (seeded, single-threaded, explicit hash
  function -> bit-exact reproducibility). pecanpy (Liu & Krishnan 2021) is
  wired as an optional `route="pecanpy"` cross-check only: its parallel numba
  walker is not reproducible run-to-run even at workers=1. Optional Walklets /
  NetMF via karateclub (GPLv3, installed --no-deps; see baselines.py).
* structural — struc2vec, the original Role2Vec and GraphWave implemented from
  the papers in this package (emb_struc2vec / emb_role2vec / emb_graphwave);
  karateclub's Role2Vec/GraphWave variants exposed as `*_kc` cross-checks.

Word2Vec objective per method follows its reference implementation: DeepWalk
hierarchical softmax; node2vec negative sampling (5); struc2vec hs=1 with the
gensim-era default negative=5 left on (as the reference code did); Role2Vec
likewise hs=1 with negative=5 (the `train_skipgram` defaults).
"""

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .emb_graphwave import graphwave_embed
from .emb_role2vec import role2vec_embed
from .emb_struc2vec import struc2vec_embed
from .graph_utils import nx_to_csr
from .walks import node2vec_walks, train_skipgram


@dataclass
class EmbedSpec:
    family: str
    fn: Callable
    params: dict = field(default_factory=dict)
    bibkey: str = ""
    fixed_width: bool = False  # True when the method has no learned bottleneck (width reported, `dim` ignored)
    needs: str = ""  # optional import the method depends on


# ---------------------------------------------------------------- pecanpy (DeepWalk / node2vec)


def pecanpy_walks(G, p=1.0, q=1.0, n_walks=10, walk_length=80, seed=42, workdir=None):
    """node2vec walks via pecanpy.SparseOTF (p=q=1 -> first-order DeepWalk walks).

    pecanpy reads edge lists from disk; the graph is written to `workdir`
    (default: system temp) and removed afterwards. Returns int32[n*n_walks, walk_length+1]
    (pecanpy counts `walk_length` steps after the start node)."""
    from pecanpy import pecanpy as pp

    n = G.number_of_nodes()
    workdir = workdir or tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=".edg", dir=workdir)
    os.close(fd)
    try:
        with open(path, "w") as fh:
            for u, v in G.edges:
                fh.write(f"{u}\t{v}\n")
        g = pp.SparseOTF(p=float(p), q=float(q), workers=1, verbose=False, random_state=int(seed))
        g.read_edg(path, weighted=False, directed=False)
        walks = g.simulate_walks(int(n_walks), int(walk_length))
    finally:
        os.remove(path)
    # pecanpy walks carry the start node plus `walk_length` steps
    L = len(walks[0])
    arr = np.empty((len(walks), L), dtype=np.int32)
    for i, w in enumerate(walks):
        if len(w) != L:
            raise AssertionError(f"pecanpy walk {i} has length {len(w)} != {L}")
        arr[i] = [int(x) for x in w]
    if arr.min() < 0 or arr.max() >= n:
        raise AssertionError("pecanpy walk contains an out-of-range node id")
    return arr


def _walk_embed(G, dim, seed, p, q, window, hs, negative, n_walks=10, walk_length=80, epochs=1, workdir=None, route="numba"):
    t0 = time.perf_counter()
    if route == "pecanpy":
        walks = pecanpy_walks(G, p=p, q=q, n_walks=n_walks, walk_length=walk_length, seed=seed, workdir=workdir)
    else:
        indptr, indices, _ = nx_to_csr(G)
        walks = node2vec_walks(indptr, indices, n_walks=n_walks, length=walk_length, p=p, q=q, seed=seed)
    t1 = time.perf_counter()
    emb = train_skipgram(walks, G.number_of_nodes(), dim=dim, window=window, seed=seed, epochs=epochs, hs=hs, negative=negative)
    t2 = time.perf_counter()
    record = {
        "p": p,
        "q": q,
        "n_walks": n_walks,
        "walk_length": walk_length,
        "window": window,
        "hs": hs,
        "negative": negative,
        "epochs": epochs,
        "route": route,
        "t_walks": t1 - t0,
        "t_skipgram": t2 - t1,
    }
    return emb, record


def deepwalk_embed(G, dim=16, seed=42, p=1.0, q=1.0, workdir=None, **kw):
    # Perozzi et al. 2014: uniform walks (p = q = 1), hierarchical softmax, window 10
    return _walk_embed(G, dim, seed, p=p, q=q, window=kw.pop("window", 10), hs=1, negative=0, workdir=workdir, **kw)


def node2vec_embed(G, dim=16, seed=42, p=1.0, q=1.0, workdir=None, **kw):
    # Grover & Leskovec 2016: biased walks, negative sampling (5), window 10
    return _walk_embed(G, dim, seed, p=p, q=q, window=kw.pop("window", 10), hs=0, negative=5, workdir=workdir, **kw)


# ---------------------------------------------------------------- karateclub cross-checks (optional)


def _kc_embed(cls_name, params, G, dim, seed):
    import karateclub

    np.random.seed(seed)
    p = dict(params)
    if "dimensions" in p and p["dimensions"] is None:
        p["dimensions"] = dim
    model = getattr(karateclub, cls_name)(seed=seed, **p)
    model.fit(G)
    return model.get_embedding(), {"class": cls_name, **p}


def walklets_embed(G, dim=16, seed=42, **kw):
    # dimensions * window_size columns; keep total = dim (dim must be divisible by 4)
    return _kc_embed("Walklets", {"dimensions": max(dim // 4, 1), "walk_number": 10, "walk_length": 80, "window_size": 4, "workers": 1}, G, dim, seed)


def netmf_embed(G, dim=16, seed=42, **kw):
    return _kc_embed("NetMF", {"dimensions": dim, "iteration": 10, "order": 2, "negative_samples": 1}, G, dim, seed)


def role2vec_kc_embed(G, dim=16, seed=42, **kw):
    return _kc_embed("Role2Vec", {"dimensions": dim, "walk_number": 10, "walk_length": 80, "workers": 1}, G, dim, seed)


def graphwave_kc_embed(G, dim=None, seed=42, **kw):
    return _kc_embed("GraphWave", {"sample_number": 100, "mechanism": "approximate"}, G, dim, seed)


# ---------------------------------------------------------------- registry

EMBEDDERS: dict[str, EmbedSpec] = {
    "deepwalk": EmbedSpec("proximity", deepwalk_embed, {"p": 1.0, "q": 1.0}, "perozzi2014deepwalk"),
    "node2vec_bfs": EmbedSpec("proximity", node2vec_embed, {"p": 0.25, "q": 4.0}, "grover2016node2vec"),
    "node2vec_dfs": EmbedSpec("proximity", node2vec_embed, {"p": 4.0, "q": 0.25}, "grover2016node2vec"),
    "walklets": EmbedSpec("proximity", walklets_embed, {}, "perozzi2017walklets", needs="karateclub"),
    "netmf": EmbedSpec("proximity", netmf_embed, {}, "qiu2018netmf", needs="karateclub"),
    "struc2vec": EmbedSpec("structural", struc2vec_embed, {}, "ribeiro2017struc2vec"),
    "role2vec": EmbedSpec("structural", role2vec_embed, {}, "ahmed2018role2vec"),
    "graphwave": EmbedSpec("structural", graphwave_embed, {}, "donnat2018graphwave", fixed_width=True),
    "role2vec_kc": EmbedSpec("structural", role2vec_kc_embed, {}, "rozemberczki2020karateclub", needs="karateclub"),
    "graphwave_kc": EmbedSpec("structural", graphwave_kc_embed, {}, "rozemberczki2020karateclub", fixed_width=True, needs="karateclub"),
}

FAMILY_OF = {name: spec.family for name, spec in EMBEDDERS.items()}


def available_methods(include_optional=False):
    out = []
    for name, spec in EMBEDDERS.items():
        if spec.needs == "karateclub" and not include_optional:
            continue
        out.append(name)
    return out


def embed(method: str, G, seed: int = 42, dim: int = 16, nodes: Optional[np.ndarray] = None, **overrides):
    """Fit `method` on the nx.Graph G (to_networkx contract).

    Returns (DataFrame[node_id, f"{method}_0" ...], record). For fixed-width
    methods (`graphwave*`) `nodes` restricts the rows computed (exact); every
    other method returns one row per node and `nodes` only subsets the frame."""
    if method not in EMBEDDERS:
        raise KeyError(f"unknown embedding method {method!r}; known: {sorted(EMBEDDERS)}")
    spec = EMBEDDERS[method]
    n = G.number_of_nodes()
    params = {**spec.params, **overrides}
    t0 = time.perf_counter()
    if method.startswith("graphwave") and not method.endswith("_kc"):
        emb, rec = spec.fn(G, nodes=nodes, seed=seed, dim=dim, **params)
        node_ids = np.arange(n) if nodes is None else np.asarray(nodes)
    else:
        emb, rec = spec.fn(G, dim=dim, seed=seed, **params)
        node_ids = np.arange(n)
        if emb.shape[0] != n:
            raise AssertionError(f"{method}: {emb.shape[0]} rows != {n} nodes")
        if nodes is not None:
            keep = np.asarray(nodes)
            emb = emb[keep]
            node_ids = keep
    wall = time.perf_counter() - t0
    emb = np.asarray(emb)
    if not np.isfinite(emb).all():
        raise AssertionError(f"{method}: non-finite embedding values")
    df = pd.DataFrame(emb, columns=[f"{method}_{j}" for j in range(emb.shape[1])])
    df.insert(0, "node_id", node_ids.astype(np.int64))
    record = {
        "method": method,
        "family": spec.family,
        "bibkey": spec.bibkey,
        "seed": int(seed),
        "dim": None if spec.fixed_width else int(dim),
        "n_features": int(emb.shape[1]),
        "fixed_width": spec.fixed_width,
        "wall_s": wall,
        **params,
        **rec,
    }
    return df, record
