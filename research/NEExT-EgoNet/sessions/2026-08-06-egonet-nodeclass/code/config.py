"""Benchmark configuration: dataset registry, constructions, global knobs.

Datasets are chosen by hypothesized LABEL MECHANISM — that grouping is the
experiment's organizing axis (the "when do egonet embeddings win" taxonomy).
Order = execution order: small datasets first so partial results are usable
early.
"""

CONFIG = {
    "n_centers": 3000,
    "center_seed": 13,
    "egonet_seed": 13,
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
    "min_class_count": 10,
    "feature_list": ["all"],
    "feature_vector_length": 3,
    "wasserstein_dim": 16,
    "pool_stats": ("mean", "max", "p90", "min", "p10"),
    "kc_seed": 42,
    # hop_k2 is skipped (status="guarded") when the median k=2 neighborhood
    # over the sampled centers exceeds max(abs_nodes, frac * n). Fixed BEFORE
    # the sweep; applied uniformly.
    "k2_guard": {"abs_nodes": 2000, "frac": 0.10},
}

# label_type is the taxonomy group used in figures and the discussion.
DATASETS = [
    # catalog_id, label_column, label_type, n_nodes(approx, pre-filter)
    ("AIRPORTS_USA", "activity_quartile", "structural-role", 1190),
    ("EMAIL_EU_CORE", "department", "community", 1005),
    ("BOOKS", "is_outlier", "outlier", 1418),
    ("POLBLOGS", "leaning", "community", 1490),
    ("CORA", "subject", "community", 2708),
    ("MINESWEEPER", "is_mine", "local-property", 10000),
    ("REDDIT_PYGOD", "is_outlier", "outlier", 10984),
    ("TOLOKERS", "is_banned", "local-property", 11758),
    ("ROMAN_EMPIRE", "syntactic_role", "structural-role", 22662),
]

CONSTRUCTIONS = {
    "hop_k1": {"method": "k_hop", "k_hop": 1},
    "hop_k2": {"method": "k_hop", "k_hop": 2},
    "walk_default": {"method": "random_walk", "walk_length": 10, "restart_prob": 0.15, "min_visits": 3},
    # min_visits=3 (not 5): a stricter floor starves high-degree centers down to
    # single-node egonets (verified on EMAIL_EU_CORE; would be epidemic on
    # TOLOKERS). Same floor as walk_default isolates the walk-shape effect.
    "walk_tight": {"method": "random_walk", "walk_length": 5, "restart_prob": 0.30, "min_visits": 3},
    "walk_wide": {
        "method": "random_walk",
        "walk_length": 20,
        "restart_prob": 0.05,
        "min_visits": 2,
        "max_egonet_size": 300,
    },
}
# every walk variant: n_walks=100, weight_by_visits=True (lib defaults)

KC_BASELINES = ["deepwalk", "node2vec", "node2vec_bfs", "walklets", "graphwave", "role2vec"]
