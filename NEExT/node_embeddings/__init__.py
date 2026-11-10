"""Learned and spectral structural node embeddings, usable as NEExT node features.

struc2vec (Ribeiro et al., KDD 2017), Role2Vec (Ahmed et al., 2018) and GraphWave (Donnat et al., KDD 2018), implemented
from their papers on flat CSR arrays with numba. They were written for the EgoNEExT baseline comparison and are exposed
here as multi-column node features: ``compute_node_features(..., feature_list=["struc2vec"])`` adds one block of
columns ``struc2vec_0 .. struc2vec_{d-1}`` per node. struc2vec and Role2Vec train a skip-gram model and need the
``node-embeddings`` extra (gensim); GraphWave needs only numpy, scipy and numba.

Each method runs on the whole graph it is given. Inside NEExT that is the source graph (``feature_scope="global"``);
on an ego-net subgraph a learned embedding has little meaning, so the framework refuses the local scope for them.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

NODE_EMBEDDING_FEATURES = ("struc2vec", "role2vec", "graphwave")

# Defaults follow the EgoNEExT baseline runs (D = 16, seed 42; GraphWave's native 100 columns).
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "struc2vec": {"dim": 16, "seed": 42},
    "role2vec": {"dim": 16, "seed": 42},
    "graphwave": {},
}


def to_simple_networkx(graph) -> Tuple[nx.Graph, list]:
    """A NEExT Graph as a simple undirected nx.Graph on 0..n-1 (self loops and duplicate edges dropped).

    Returns the graph and the list mapping position i back to the NEExT node id. The embedders require a connected
    graph; load with ``filter_largest_component=True`` (the default).
    """
    nodes = list(graph.nodes)
    position = {node: i for i, node in enumerate(nodes)}
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((position[u], position[v]) for u, v in graph.edges if u != v)
    if G.number_of_nodes() > 1 and not nx.is_connected(G):
        raise ValueError(
            f"Graph {graph.graph_id} is not connected; struc2vec, Role2Vec and GraphWave need a connected graph. "
            "Load it with filter_largest_component=True."
        )
    return G, nodes


def _embed(name: str, G: nx.Graph, params: Dict[str, Any]) -> np.ndarray:
    if name == "struc2vec":
        from NEExT.node_embeddings.struc2vec import struc2vec_embed

        return struc2vec_embed(G, **params)[0]
    if name == "role2vec":
        from NEExT.node_embeddings.role2vec import role2vec_embed

        return role2vec_embed(G, **params)[0]
    if name == "graphwave":
        from NEExT.node_embeddings.graphwave import graphwave_embed

        return graphwave_embed(G, **params)[0]
    raise ValueError(f"Unknown node embedding {name!r}; available: {list(NODE_EMBEDDING_FEATURES)}")


def node_embedding_feature(name: str, params: Optional[Dict[str, Any]] = None) -> Callable[[Any], pd.DataFrame]:
    """A feature function for StructuralNodeFeatures: graph -> DataFrame(node_id, graph_id, <name>_0 ..)."""
    merged = dict(DEFAULT_PARAMS.get(name, {}))
    merged.update(params or {})

    def compute(graph) -> pd.DataFrame:
        G, nodes = to_simple_networkx(graph)
        emb = np.asarray(_embed(name, G, merged), dtype=np.float64)
        position = {node: i for i, node in enumerate(nodes)}
        out_nodes = list(graph.sampled_nodes) if graph.sampled_nodes is not None else nodes
        rows = emb[[position[n] for n in out_nodes]]
        df = pd.DataFrame(rows, columns=[f"{name}_{i}" for i in range(rows.shape[1])])
        df.insert(0, "graph_id", graph.graph_id)
        df.insert(0, "node_id", out_nodes)
        return df

    return compute
