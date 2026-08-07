"""Experiment 4 (new research line): egonet signal test, Amazon Ratings.

Roman-mechanism twin at the density midpoint (mean degree 7.6, between
Roman's 2.9 and Chameleon's 27.6): heterophily-benchmark co-purchase graph,
rating-class labels. Prediction on record (2026-08-06): node baseline
weak-to-moderate, local k1/k2 beat it, monotone dilution decay, global
below local but less degenerate than Roman, normal walk locality ordering.
Full protocol: khop k1-3 + walk trio, both scopes, floors + node baseline;
3,000 stratified centers.
"""

CONFIG = {
    "dataset": "AMAZON_RATINGS",
    "label_column": "rating_class",
    "graph_type": "igraph",
    "min_class_count": 10,
    # centers: stratified sample (24.5k nodes; same sampling as Roman)
    "n_centers": 3000,
    "center_seed": 13,
    # egonet construction
    "k_hops": [1, 2, 3, 4, 5],
    "egonet_seed": 13,
    # features: computed WITHIN each egonet (local scope), own-node value only
    "feature_list": ["all"],
    "feature_vector_length": 1,
    "feature_tag": "fall-vl1",
    # embedding (dimension follows the rule dim = n feature columns)
    "wasserstein_dim": 11,
    "embed_seed": 42,
    # evaluation
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
}

# Walk constructions kept for later phases of this session (unused in the
# first pass; run_experiment --family walk picks them up).
WALK_CONSTRUCTIONS = {
    "walk_tight": {"walk_length": 5, "restart_prob": 0.30, "min_visits": 3},
    "walk_default": {"walk_length": 10, "restart_prob": 0.15, "min_visits": 3},
    "walk_wide": {"walk_length": 20, "restart_prob": 0.05, "min_visits": 2},
}
N_WALKS = 100
