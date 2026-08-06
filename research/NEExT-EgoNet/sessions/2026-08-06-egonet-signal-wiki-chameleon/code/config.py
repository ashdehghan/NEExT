"""Experiment 3 (new research line): egonet signal test, Wikipedia Chameleon.

Airports-mechanism twin in a different domain: dense hyperlink graph (mean
degree 27.6, like airports' 30) with traffic-quintile labels — popularity,
expected to be node-encoded. Prediction on record (2026-08-06): node
features win, local k=1 best egonet, saturation decay at k>=2, global scope
robust. Full protocol: khop k1-3 + walk trio, both scopes, floors +
node-features baseline; all 2,277 nodes as centers.
"""

CONFIG = {
    "dataset": "WIKIPEDIA_ARTICLES_CHAMELEON",
    "label_column": "traffic_quintile",
    "graph_type": "igraph",
    "min_class_count": 10,
    # centers: all nodes (2,277 — small enough to run full, like airports)
    "n_centers": 1000,
    "center_seed": 13,
    # egonet construction
    "k_hops": [1, 2, 3],
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
