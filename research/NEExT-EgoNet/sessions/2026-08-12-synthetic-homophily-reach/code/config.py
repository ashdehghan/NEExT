"""Synthetic experiment 1: homophily-reach mechanism (planted).

Grid: wavelength L1/L2/L3 x coupling B0/B1 (see datasets.py). Pre-registered
predictions live in notes/predictions.md, written BEFORE any pipeline run.
Runner takes --dataset SYNTH_HR_<L>_<B>.
"""

CONFIG = {
    "dataset": "SYNTH_HR_L2_B1",  # default cell; runner overrides via --dataset
    "label_column": "field_class",
    "graph_type": "igraph",
    "min_class_count": 10,
    "n_centers": 2000,
    "center_seed": 13,
    "k_hops": [1, 2, 3, 4, 5, 6],
    "egonet_seed": 13,
    "feature_list": ["all"],
    "feature_vector_length": 1,
    "feature_tag": "fall-vl1",
    "wasserstein_dim": 11,
    "embed_seed": 42,
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
}

WALK_CONSTRUCTIONS = {}
N_WALKS = 100
