"""Synthetic graph generation module for NEExT.

Provides two layers of graph generation:
- **Layer 1** (``GraphGenerator``): Registry-based generator with 17 built-in
  graph types and support for custom generators via any library.
- **Layer 2** (``GraphBuilder``): Fluent compositional API for building custom
  graphs programmatically.

All generators produce ``List[nx.Graph]`` with ``G.graph['label']`` set,
feeding directly into ``nxt.load_from_networkx()``.
"""

from ._adapters import GraphAdapter
from ._config import AnomalyConfig, AttributeConfig, CollectionSpec, GeneratorSpec
from .anomalies import AnomalyInjector
from .attributes import AttributeGenerator
from .builder import GraphBuilder
from .generator import GraphGenerator
from .presets import SyntheticPresets

__all__ = [
    "GraphGenerator",
    "GraphBuilder",
    "SyntheticPresets",
    "AttributeGenerator",
    "AnomalyInjector",
    "GraphAdapter",
    "GeneratorSpec",
    "CollectionSpec",
    "AttributeConfig",
    "AnomalyConfig",
]
