"""Egonet bag construction: k-hop egonets with containment labels.

A *bag* is a k-hop egonet treated as an unordered set of member nodes. Its
containment label is the OR over member labels:

    y_contains(B) = 1  iff  any node in B has the positive label

The center's own label (`y_center`) is kept for diagnostics — it is the label
the original PoC classified, and the contrast between the two questions is the
point of the study.

The label column is passed as `egonet_feature_target`, which keeps it out of
the egonet node attributes (no leakage into structural features), and member
labels are read from the *source* collection via each egonet's node_mapping.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BagSet:
    """An EgonetCollection plus its bag-level label/size table."""

    egonets: object  # EgonetCollection
    table: pd.DataFrame  # graph_id, center_node, y_contains, y_center, n_nodes, n_edges


def _edge_count(egonet) -> int:
    return egonet.G.number_of_edges() if egonet.graph_type == "networkx" else egonet.G.ecount()


def build_bags(
    nxt,
    source_collection,
    label_column: str,
    k_hop: int,
    n_centers: int = None,
    positive_value=1,
    seed: int = 13,
) -> BagSet:
    """Decompose a single-graph collection into k-hop bags with containment labels.

    Args:
        nxt: NEExT framework instance.
        source_collection: GraphCollection holding ONE source graph.
        label_column: node-attribute column with the class label.
        k_hop: egonet radius.
        n_centers: sample this many egonet centers (None = every node).
        positive_value: label value counted as positive.
        seed: center-sampling seed.
    """
    source_graphs = {g.graph_id: g for g in source_collection.graphs}
    total_nodes = sum(len(g.nodes) for g in source_graphs.values())
    sample_fraction = 1.0 if n_centers is None else min(1.0, n_centers / total_nodes)

    egonets = nxt.compute_k_hop_egonets(
        source_collection,
        k_hop=k_hop,
        egonet_feature_target=label_column,
        sample_fraction=sample_fraction,
        random_seed=seed,
    )

    rows = []
    for egonet in egonets.graphs:
        src_graph_id, center = egonets.egonet_to_graph_node_mapping[egonet.graph_id]
        src = source_graphs[src_graph_id]
        members = egonet.node_mapping.keys()
        member_positive = sum(1 for m in members if src.node_attributes[m][label_column] == positive_value)
        rows.append(
            {
                "graph_id": egonet.graph_id,
                "center_node": center,
                "y_contains": int(member_positive > 0),
                "y_center": int(src.node_attributes[center][label_column] == positive_value),
                "n_positive_members": member_positive,
                "n_nodes": len(egonet.nodes),
                "n_edges": _edge_count(egonet),
            }
        )
    return BagSet(egonets=egonets, table=pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True))
