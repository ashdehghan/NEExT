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
    "k_hop": 2,
    "egonet_seed": 13,
    # features: computed WITHIN each egonet, own-node value only
    "feature_list": [
        "degree_centrality",
        "clustering_coefficient",
        "page_rank",
        "closeness_centrality",
    ],
    "feature_vector_length": 1,
    # embedding
    "wasserstein_dim": 4,
    "embed_seed": 42,
    # evaluation
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
}

# x-axis order + display labels for the results figure
METHODS = {
    "permuted": "Random\n(permuted labels)",
    "majority": "Majority\nclass",
    "egonet_k2_wass": "Egonet $k{=}2$\nWasserstein",
}
