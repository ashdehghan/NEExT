"""Containment-search experiment library (NEExT-EgoNet research).

Shared by every containment session: bag construction, per-bag
representations, bag-level evaluation, run IO discipline, and the phase-1
planted-anomaly generator.
"""

from .bags import BagSet, build_bags
from .evaluate import evaluate_node_oracle, evaluate_representation, shared_splits, summarize_metrics
from .representations import pooled_features, size_only, wasserstein_embedding
from .runio import aggregate, git_sha, neext_version, run_complete, write_run
from .synthetic import ANOMALY_TYPES, FAMILIES, make_synthetic

__all__ = [
    "BagSet",
    "build_bags",
    "evaluate_representation",
    "evaluate_node_oracle",
    "shared_splits",
    "summarize_metrics",
    "wasserstein_embedding",
    "pooled_features",
    "size_only",
    "write_run",
    "run_complete",
    "aggregate",
    "git_sha",
    "neext_version",
    "make_synthetic",
    "FAMILIES",
    "ANOMALY_TYPES",
]
