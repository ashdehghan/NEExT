"""Patch tiling of graphs (one subgraph per community) and pairwise quantities between patches."""

from NEExT.patches.leiden_patches import compute_patch_label, merge_small_communities, relabel_contiguous
from NEExT.patches.pair_priors import coarse_patch_graph, compute_pair_priors

__all__ = ["compute_patch_label", "merge_small_communities", "relabel_contiguous", "coarse_patch_graph", "compute_pair_priors"]
