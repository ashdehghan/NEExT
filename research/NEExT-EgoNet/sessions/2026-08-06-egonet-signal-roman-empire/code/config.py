"""Experiment 2 (new research line): egonet signal test, Roman Empire.

Sparse heterophilous contrast to AIRPORTS_EUROPE (mean degree 2.9 vs 30):
k=1 bags are starved (~4 nodes), saturation never arrives. First pass per
Ash (2026-08-06): LOCAL scope, k in {1,2}, permutation floor only. Note the
catalog carries structure + label only — this tests STRUCTURAL context, not
semantic context (the original dataset's word embeddings are not present).
"""

CONFIG = {
    "dataset": "ROMAN_EMPIRE",
    "label_column": "syntactic_role",
    "graph_type": "igraph",
    "min_class_count": 10,
    # centers: stratified sample (22.6k nodes is needlessly slow to run full)
    "n_centers": 3000,
    "center_seed": 13,
    # egonet construction
    "k_hops": [1, 2, 3, 4],
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
