"""Ring-by-ring ego-net embeddings: one embedding block per hop ring around the centre, side by side.

A pooled ego-net embedding mixes the centre's near and far neighbours into one distribution. Here each egonet's members
are split by their hop distance from the centre in the parent graph (``Egonet.node_rings``): ring 0 is the centre
itself, ring 1 its neighbours, and so on. Every ring of every egonet is embedded with the chosen vectorizer, ring by
ring, and the blocks are concatenated, so a node is represented as [centre | ring 1 | ring 2 | ...]. Raising
``max_ring`` appends blocks and leaves the earlier ones unchanged.

Under ``approx_wasserstein`` a ring's block is a linear map of the (membership-weighted) mean feature vector of that
ring; the exact and Sinkhorn vectorizers keep the shape of each ring's distribution.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from NEExT.embeddings.embeddings import Embeddings
from NEExT.embeddings.graph_embeddings import DEFAULT_MAX_DISTRIBUTION_SIZE, DEFAULT_OT_METRIC, GraphEmbeddings
from NEExT.features import Features

if TYPE_CHECKING:  # EgonetCollection imports NEExT.embeddings, so only import it for type checkers
    from NEExT.collections import EgonetCollection


def member_rings(egonet_collection: "EgonetCollection") -> pd.DataFrame:
    """One row per egonet member: ``node_id`` (internal), ``graph_id`` (egonet) and ``ring`` (hop distance)."""
    rows = []
    for egonet in egonet_collection.graphs:
        if egonet.node_rings is None:
            raise ValueError(
                f"Egonet {egonet.graph_id} has no node_rings; ring embeddings need egonets built by "
                "compute_k_hop_egonets or compute_random_walk_egonets."
            )
        rows.extend((internal, egonet.graph_id, ring) for internal, ring in egonet.node_rings.items())
    return pd.DataFrame(rows, columns=["node_id", "graph_id", "ring"])


def compute_ring_embeddings(
    egonet_collection: "EgonetCollection",
    features: Features,
    max_ring: int,
    embedding_algorithm: str = "approx_wasserstein",
    embedding_dimension: Optional[int] = None,
    feature_columns: Optional[List[str]] = None,
    ring_weights: Optional[Dict[int, float]] = None,
    random_state: int = 42,
    memory_size: str = "4G",
    metric: str = DEFAULT_OT_METRIC,
    max_distribution_size: Optional[int] = DEFAULT_MAX_DISTRIBUTION_SIZE,
) -> Embeddings:
    """
    Embed every ring 0..max_ring of every egonet separately and concatenate the blocks.

    Args:
        egonet_collection: Egonets with ``node_rings`` (k-hop or random-walk construction).
        features: Member features for the egonets (local or global scope).
        max_ring: Deepest ring embedded; rings beyond it are ignored.
        embedding_algorithm: "approx_wasserstein", "wasserstein" or "sinkhornvectorizer".
        embedding_dimension: Dimension of each ring block (default and cap: the number of feature columns).
        feature_columns: Feature columns to use (default: all).
        ring_weights: Optional fixed multiplier per ring (ring -> weight, default 1.0), applied to that ring's block.
        random_state, memory_size, metric, max_distribution_size: Passed to the vectorizer (see GraphEmbeddings).

    Returns:
        Embeddings with columns ``emb_<i>_ring<r>`` for each ring block and ``ring<r>_size`` (number of members in
        the ring) for r = 0..max_ring. An egonet with an empty ring gets zeros in that ring's block and size 0.
    """
    if max_ring < 0:
        raise ValueError(f"max_ring must be >= 0, got {max_ring}")
    columns = list(feature_columns or features.feature_columns)
    dimension = min(embedding_dimension or len(columns), len(columns))
    weights = dict(ring_weights or {})

    rings = member_rings(egonet_collection)
    table = features.features_df.merge(rings, on=["node_id", "graph_id"], how="left")
    if table["ring"].isna().any():
        raise ValueError("Some feature rows have no ring: the features were not computed for these egonets")
    graph_ids = [egonet.graph_id for egonet in egonet_collection.graphs]
    out = pd.DataFrame({"graph_id": graph_ids})
    embedding_columns: List[str] = []

    for ring in range(max_ring + 1):
        ring_rows = table[table["ring"] == ring].drop(columns=["ring"]).reset_index(drop=True)
        block_columns = [f"emb_{i}_ring{ring}" for i in range(dimension)]
        sizes = ring_rows.groupby("graph_id").size()
        out[f"ring{ring}_size"] = out["graph_id"].map(sizes).fillna(0).astype(int)
        if ring_rows.empty:
            for col in block_columns:
                out[col] = 0.0
        else:
            ring_dimension = min(dimension, ring_rows["graph_id"].nunique())
            block = GraphEmbeddings(
                graph_collection=egonet_collection,
                features=Features(ring_rows, columns),
                embedding_algorithm=embedding_algorithm,
                embedding_dimension=ring_dimension,
                feature_columns=columns,
                random_state=random_state,
                memory_size=memory_size,
                suffix=f"ring{ring}",
                metric=metric,
                max_distribution_size=max_distribution_size,
            ).compute()
            out = out.merge(block.embeddings_df, on="graph_id", how="left")
            for col in block_columns:
                if col not in out.columns:  # fewer egonets with this ring than requested dimensions
                    out[col] = 0.0
            out[block_columns] = out[block_columns].fillna(0.0)
        out[block_columns] = out[block_columns] * float(weights.get(ring, 1.0))
        embedding_columns.extend(block_columns)

    size_columns = [f"ring{ring}_size" for ring in range(max_ring + 1)]
    out = out[["graph_id"] + embedding_columns + size_columns]
    return Embeddings(out, f"{embedding_algorithm}_rings", embedding_columns + size_columns)


def ring_importance_map(egonet_collection: "EgonetCollection", centre_egonet_id, ring_weights: Dict[int, float]) -> pd.DataFrame:
    """
    Per-member importance for one egonet, for drawing on the parent graph: ring weight x membership weight.

    Args:
        egonet_collection: Egonets with ``node_rings``.
        centre_egonet_id: graph_id of the egonet to map.
        ring_weights: Importance of each ring (ring -> weight; missing rings count 0).

    Returns:
        DataFrame with ``node_id`` (parent graph id), ``ring``, ``membership_weight`` (visit weight, or 1/|bag| for
        uniform bags) and ``importance`` (ring weight x membership weight, normalised to sum to 1 when positive).
    """
    egonet = next((g for g in egonet_collection.graphs if g.graph_id == centre_egonet_id), None)
    if egonet is None:
        raise ValueError(f"No egonet with graph_id {centre_egonet_id!r}")
    if egonet.node_rings is None:
        raise ValueError(f"Egonet {centre_egonet_id!r} has no node_rings")
    original = {internal: node for node, internal in egonet.node_mapping.items()}
    n = len(egonet.node_rings)
    rows = []
    for internal, ring in sorted(egonet.node_rings.items(), key=lambda item: (item[1], item[0])):
        membership = egonet.node_weights.get(internal, 0.0) if egonet.node_weights else 1.0 / n
        rows.append((original[internal], ring, membership, float(ring_weights.get(ring, 0.0)) * membership))
    df = pd.DataFrame(rows, columns=["node_id", "ring", "membership_weight", "importance"])
    total = df["importance"].sum()
    if total > 0:
        df["importance"] = df["importance"] / total
    return df


def draw_importance_map(graph, importance: pd.DataFrame, ax=None, layout_seed: int = 13, cmap: str = "viridis"):
    """
    Draw a centre's neighbourhood coloured by importance (needs matplotlib, the ``advanced`` extra).

    Args:
        graph: The parent NEExT Graph (its ``G`` is NetworkX or iGraph).
        importance: Output of ``ring_importance_map``.
        ax: Optional matplotlib axes.
        layout_seed: Seed for the spring layout.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib axes.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    members = importance["node_id"].tolist()
    G = graph.G if isinstance(graph.G, nx.Graph) else nx.Graph(graph.G.get_edgelist())
    sub = G.subgraph(members)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(sub, seed=layout_seed)
    values = importance.set_index("node_id").loc[list(sub.nodes()), "importance"].to_numpy()
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3)
    nodes = nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=values, cmap=cmap, node_size=80)
    centre = importance.loc[importance["ring"] == 0, "node_id"].tolist()
    nx.draw_networkx_nodes(sub, pos, nodelist=centre, ax=ax, node_color="none", edgecolors="red", node_size=160, linewidths=2)
    plt.colorbar(nodes, ax=ax, label="importance")
    ax.set_axis_off()
    return ax
