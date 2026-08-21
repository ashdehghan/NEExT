"""Embedding-baselines session: external node embeddings on the paper's four
datasets (+ the two synthetic calibration anchors), evaluated on the SAME
centers, splits, classifier and floors as the egonet-bag experiments.

Framing: proximity embeddings (DeepWalk / node2vec) encode position -> they
should be strong under shared splits on homophilous graphs and collapse under
region holdout; structural embeddings (struc2vec / Role2Vec / GraphWave) are
position-free -> they should survive, like the egonet fragment code on Roman
Empire. Pre-registered predictions: notes/predictions.md.
"""

# dataset short name -> how to rebuild the canonical center set of its signal session
DATASETS = {
    "airports": {
        "catalog_id": "AIRPORTS_EUROPE",
        "label_column": "activity_quartile",
        "n_centers": None,  # every node is a center
        "signal_session": "2026-08-06-egonet-signal-airports-eu",
        "expected_best_bag": ("egonet_k1_wass", "fall-vl1"),
    },
    "chameleon": {
        "catalog_id": "WIKIPEDIA_ARTICLES_CHAMELEON",
        "label_column": "traffic_quintile",
        "n_centers": 1000,
        "signal_session": "2026-08-06-egonet-signal-wiki-chameleon",
        "expected_best_bag": ("egonet_k2_wass", "fall-vl1-glob"),
    },
    "roman": {
        "catalog_id": "ROMAN_EMPIRE",
        "label_column": "syntactic_role",
        "n_centers": 3000,
        "signal_session": "2026-08-06-egonet-signal-roman-empire",
        "expected_best_bag": ("egonet_k2_wass", "fall-vl1"),
    },
    "amazon": {
        "catalog_id": "AMAZON_RATINGS",
        "label_column": "rating_class",
        "n_centers": 3000,
        "signal_session": "2026-08-06-egonet-signal-amazon-ratings",
        "expected_best_bag": ("egonet_k4_wass", "fall-vl1-glob"),
    },
    "synth_b0": {
        "catalog_id": "SYNTH_HR_L2_B0",
        "label_column": "field_class",
        "n_centers": 2000,
        "signal_session": "2026-08-12-synthetic-homophily-reach",
        "expected_best_bag": None,
    },
    "synth_b1": {
        "catalog_id": "SYNTH_HR_L2_B1",
        "label_column": "field_class",
        "n_centers": 2000,
        "signal_session": "2026-08-12-synthetic-homophily-reach",
        "expected_best_bag": None,
    },
}

CONFIG = {
    "graph_type": "igraph",
    "min_class_count": 10,
    "center_seed": 13,
    # evaluation protocol — identical to the signal sessions
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
    # embedding policy
    "dim": 16,  # every learned bottleneck; >= the bags' D = 11
    "embed_seeds": [42],
    "workers": 1,
}

# method id -> family (see lib/nodeclass/embed_registry.py for parameters + references)
METHODS_BY_FAMILY = {
    "proximity": ["deepwalk", "node2vec_bfs", "node2vec_dfs"],
    "structural": ["struc2vec", "role2vec", "graphwave"],
}
OPTIONAL_METHODS = ["walklets", "netmf", "role2vec_kc", "graphwave_kc"]  # karateclub cross-checks (stretch)
CORE_METHODS = [m for fam in METHODS_BY_FAMILY.values() for m in fam]
