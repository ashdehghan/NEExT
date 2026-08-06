"""Baseline node representations: karateclub embeddings + structural features.

karateclub (1.3.3, installed --no-deps against this repo's newer stack; every
method smoke-verified) requires an undirected networkx graph with consecutive
0..n-1 integer node ids — exactly what `load_single_graph_from_dfs` with
filter_largest_component=True produces. `to_networkx` asserts that contract
rather than trusting it.

Dimension fairness: dimensions=16 wherever the method exposes a true
bottleneck knob (matches the Wasserstein-16 egonet embedding). GraphWave's
width is 2*sample_number characteristic-function samples, not a learned
bottleneck — it runs at its natural 200 and n_features is reported in every
table. Walklets emits dimensions*window_size columns, so 4*4=16.

Determinism: workers=1 + a fixed seed (plus np.random.seed for gensim's
corpus shuffling) reproduced embeddings bit-exactly in the install smoke;
recorded as "best-effort" because gensim only guarantees it single-threaded.
"""

import numpy as np
import pandas as pd

KC_METHODS = {
    "deepwalk": ("DeepWalk", {"dimensions": 16, "walk_number": 10, "walk_length": 80, "window_size": 5, "workers": 1}),
    "node2vec": (
        "Node2Vec",
        {"dimensions": 16, "walk_number": 10, "walk_length": 80, "p": 1.0, "q": 1.0, "workers": 1},
    ),
    "node2vec_bfs": (
        "Node2Vec",
        {"dimensions": 16, "walk_number": 10, "walk_length": 80, "p": 0.25, "q": 4.0, "workers": 1},
    ),
    "walklets": ("Walklets", {"dimensions": 4, "walk_number": 10, "walk_length": 80, "window_size": 4, "workers": 1}),
    "graphwave": ("GraphWave", {"sample_number": 100, "mechanism": "approximate"}),
    "role2vec": ("Role2Vec", {"dimensions": 16, "walk_number": 10, "walk_length": 80, "workers": 1}),
}


def to_networkx(source_collection):
    """Export the (single) source graph as the undirected nx.Graph karateclub needs."""
    import networkx as nx

    graph = source_collection.graphs[0]
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    n = G.number_of_nodes()
    if sorted(G.nodes) != list(range(n)):
        raise AssertionError("Node ids are not consecutive 0..n-1 (karateclub contract)")
    if not nx.is_connected(G):
        raise AssertionError("Graph is not connected — was filter_largest_component applied?")
    return G


def kc_embed(method: str, G, seed: int = 42) -> "tuple[pd.DataFrame, dict]":
    """Fit one karateclub method on the full graph; node_id-keyed frame + record."""
    import karateclub

    cls_name, params = KC_METHODS[method]
    cls = getattr(karateclub, cls_name)
    np.random.seed(seed)  # gensim corpus shuffling reads global state
    model = cls(seed=seed, **params)
    model.fit(G)
    emb = model.get_embedding()

    if emb.shape[0] != G.number_of_nodes():
        raise AssertionError(f"{method}: embedding rows {emb.shape[0]} != nodes {G.number_of_nodes()}")
    df = pd.DataFrame(emb, columns=[f"{method}_{j}" for j in range(emb.shape[1])])
    df.insert(0, "node_id", np.arange(emb.shape[0]))
    record = {"method": method, "class": cls_name, "seed": seed, "n_features": emb.shape[1], **params}
    return df, record


def center_structural_features(nxt, source_collection, feature_list=("all",), feature_vector_length: int = 3, n_jobs: int = -1) -> pd.DataFrame:
    """The PoC winner: NEExT structural features computed on the FULL graph, per node."""
    features = nxt.compute_node_features(
        source_collection,
        feature_list=list(feature_list),
        feature_vector_length=feature_vector_length,
        show_progress=False,
        n_jobs=n_jobs,
    )
    return features.features_df[["node_id"] + features.feature_columns].copy()


def degree_only(source_collection) -> pd.DataFrame:
    """Trivial-information floor: degree + k-core number, nothing else."""
    graph = source_collection.graphs[0]
    nodes = sorted(graph.nodes)
    if nodes != list(range(len(nodes))):
        raise AssertionError("Node ids are not consecutive 0..n-1")
    if graph.graph_type == "igraph":
        degree = graph.G.degree()  # igraph vertex ids == node ids after reindex
        core = graph.G.coreness()
    else:
        import networkx as nx

        degree = [d for _, d in sorted(graph.G.degree())]
        core = [c for _, c in sorted(nx.core_number(graph.G).items())]
    return pd.DataFrame({"node_id": nodes, "degree": degree, "core_number": core})
