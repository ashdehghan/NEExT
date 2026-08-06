"""Egonet construction keyed to a fixed center list, for node classification.

Unlike `containment.bags.build_bags` (which samples centers internally and
derives binary OR-labels), the node-classification benchmark fixes the
centers up front: every construction (hop k=1, k=2, each walk variant)
receives the exact same pre-sampled center list via `nodes_to_sample` with
`sample_fraction=0.0` — identical centers by contract, not RNG coincidence.
Labels live in the canonical node table (`data.py`), not here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NodeBagSet:
    """An EgonetCollection plus its egonet->center-node size table."""

    egonets: object  # EgonetCollection
    table: pd.DataFrame  # graph_id (egonet id), node_id (center), n_nodes, n_edges


def _edge_count(egonet) -> int:
    return egonet.G.number_of_edges() if egonet.graph_type == "networkx" else egonet.G.ecount()


def build_node_bags(
    nxt,
    source_collection,
    centers,
    label_column: str,
    method: str = "k_hop",
    k_hop: int = 1,
    walk_length: int = 10,
    n_walks: int = 100,
    restart_prob: float = 0.15,
    weight_by_visits: bool = True,
    min_visits: int = 3,
    max_egonet_size: int = None,
    seed: int = 13,
) -> NodeBagSet:
    """Build one egonet per center node, for exactly the given centers.

    `label_column` is passed as egonet_feature_target so it never enters the
    structural features. Returns a NodeBagSet whose table covers precisely
    `centers` (asserted).
    """
    graph_id = source_collection.graphs[0].graph_id
    forced = {graph_id: list(centers)}

    if method == "random_walk":
        egonets = nxt.compute_random_walk_egonets(
            source_collection,
            walk_length=walk_length,
            n_walks=n_walks,
            restart_prob=restart_prob,
            weight_by_visits=weight_by_visits,
            min_visits=min_visits,
            max_egonet_size=max_egonet_size,
            egonet_feature_target=label_column,
            nodes_to_sample=forced,
            sample_fraction=0.0,
            random_seed=seed,
        )
    elif method == "k_hop":
        egonets = nxt.compute_k_hop_egonets(
            source_collection,
            k_hop=k_hop,
            egonet_feature_target=label_column,
            nodes_to_sample=forced,
            sample_fraction=0.0,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unknown bag construction method: {method}")

    rows = []
    for egonet in egonets.graphs:
        _, center = egonets.egonet_to_graph_node_mapping[egonet.graph_id]
        rows.append(
            {
                "graph_id": egonet.graph_id,
                "node_id": center,
                "n_nodes": len(egonet.nodes),
                "n_edges": _edge_count(egonet),
            }
        )
    table = pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True)

    got, want = set(table["node_id"]), set(centers)
    if got != want:
        raise AssertionError(f"Egonet centers != requested centers (missing {len(want - got)}, extra {len(got - want)})")
    return NodeBagSet(egonets=egonets, table=table)


def egonet_rep_to_node_frame(rep_df: pd.DataFrame, bag_table: pd.DataFrame) -> pd.DataFrame:
    """Re-key an egonet-level representation (graph_id column) to node_id.

    After this every representation in the benchmark — egonet-side or
    baseline — is a node_id-keyed frame with identical join semantics.
    """
    merged = rep_df.merge(bag_table[["graph_id", "node_id"]], on="graph_id", validate="one_to_one")
    cols = ["node_id"] + [c for c in rep_df.columns if c != "graph_id"]
    return merged[cols]


def khop_reach(source_collection, centers, k: int = 2) -> dict:
    """Cheap k-hop neighborhood-size stats over the centers (dense-graph guard).

    Uses igraph neighborhood_size when available; BFS via networkx otherwise.
    Costs seconds, not the minutes a full k=2 egonet build on a dense graph
    would — run it BEFORE deciding to build.
    """
    graph = source_collection.graphs[0]
    centers = list(centers)
    if graph.graph_type == "igraph":
        sizes = np.array(graph.G.neighborhood_size(vertices=centers, order=k))
    else:
        import networkx as nx

        sizes = np.array([len(nx.single_source_shortest_path_length(graph.G, c, cutoff=k)) for c in centers])
    n = len(graph.nodes)
    return {
        "k": k,
        "n_centers": len(centers),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "max": int(sizes.max()),
        "median_frac_of_graph": round(float(np.median(sizes)) / n, 4),
    }
