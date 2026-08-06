"""Experiment 1 (new research line): egonet signal test, Air Traffic Europe.

One fixed configuration on purpose — this is a yes/no signal detection run,
not a sweep. Every parameter a run needs lives here so config.json snapshots
are complete.
"""

CONFIG = {
    "dataset": "AIRPORTS_EUROPE",
    "label_column": "activity_quartile",
    "graph_type": "igraph",
    "min_class_count": 10,
    # egonet construction
    "k_hops": [1, 2, 3],
    "egonet_seed": 13,
    # features: computed WITHIN each egonet, own-node value only.
    # "all" = the full 11-feature structural pool; feature_tag names the
    # feature configuration in run ids so runs with different feature sets
    # coexist in outputs/ and results.csv.
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

# Random-walk bag variants (Ash-approved 2026-08-06): the tuned trio from the
# walk-bags characterization session. min_visits=3 floor per its findings
# (stricter floors starve high-degree centers); wide relaxes to 2 to let long
# low-restart walks keep their reach. n_walks=100 and visit-weighting held
# fixed — weights are the point of walk bags.
WALK_CONSTRUCTIONS = {
    "walk_tight": {"walk_length": 5, "restart_prob": 0.30, "min_visits": 3},
    "walk_default": {"walk_length": 10, "restart_prob": 0.15, "min_visits": 3},
    "walk_wide": {"walk_length": 20, "restart_prob": 0.05, "min_visits": 2},
}
N_WALKS = 100

# x-axis order + display labels for the results figures. Majority is still
# computed and recorded in outputs/ but deliberately kept OFF the plot
# (Ash 2026-08-06: report the permutation floor as the baseline; macro-F1
# and AUC stay in the artifacts, the figures show accuracy only).
PLOT_METHODS = {
    "permuted": "Random\n(permuted labels)",
    "egonet_k1_wass": "Egonet $k{=}1$",
    "egonet_k2_wass": "Egonet $k{=}2$",
    "egonet_k3_wass": "Egonet $k{=}3$",
}
