"""Node-classification benchmark library (NEExT-EgoNet research).

Egonet embeddings as node-level features, evaluated against classic node
embeddings (karateclub) and per-node structural features. Sibling of
`lib.containment` (binary containment); the genuinely generic pieces —
`shared_splits`, `runio`, `representations`, `plotstyle` — are imported
from there, not duplicated.
"""

from .bags import NodeBagSet, build_node_bags, egonet_rep_to_node_frame, khop_reach
from .baselines import (
    KC_METHODS,
    center_structural_features,
    degree_only,
    kc_embed,
    to_networkx,
)
from .data import (
    MIN_CLASS_COUNT,
    build_node_table,
    filter_rare_classes,
    sample_centers,
)
from .evaluate import evaluate_node_representation, majority_floor, permutation_floor, summarize_node_metrics

__all__ = [
    "NodeBagSet",
    "build_node_bags",
    "egonet_rep_to_node_frame",
    "khop_reach",
    "KC_METHODS",
    "kc_embed",
    "to_networkx",
    "center_structural_features",
    "degree_only",
    "MIN_CLASS_COUNT",
    "build_node_table",
    "filter_rare_classes",
    "sample_centers",
    "evaluate_node_representation",
    "majority_floor",
    "permutation_floor",
    "summarize_node_metrics",
]
