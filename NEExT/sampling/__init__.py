"""
Neighborhood sampling strategies for NEExT framework.

This module provides various sampling techniques for egonet generation,
allowing for efficient computation on large graphs while preserving
structural properties.
"""

from .base import SamplingStrategy, NodeSampler
from .k_hop_sampler import KHopSampler
from .random_walk_sampler import RandomWalkSampler

__all__ = [
    'SamplingStrategy',
    'NodeSampler', 
    'KHopSampler',
    'RandomWalkSampler'
]