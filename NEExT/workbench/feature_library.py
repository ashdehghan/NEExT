"""Curated built-in feature catalog for NEExT Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import FeatureCatalogEntry

NODE_FEATURE_OPERATION_ID = "neext.compute_node_features"
NODE_FEATURE_OPERATION_VERSION = "1"


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    operation_version: str
    label: str


OPERATION_REGISTRY: dict[tuple[str, str], OperationDefinition] = {
    (NODE_FEATURE_OPERATION_ID, NODE_FEATURE_OPERATION_VERSION): OperationDefinition(
        operation_id=NODE_FEATURE_OPERATION_ID,
        operation_version=NODE_FEATURE_OPERATION_VERSION,
        label="Compute NEExT structural node features",
    )
}


@dataclass(frozen=True)
class FeatureCatalogItem:
    id: str
    name: str
    description: str
    output: str = "Node feature vector columns"
    operation_id: str = NODE_FEATURE_OPERATION_ID
    operation_version: str = NODE_FEATURE_OPERATION_VERSION

    def to_public_entry(self) -> FeatureCatalogEntry:
        return FeatureCatalogEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            output=self.output,
            operation_id=self.operation_id,
            operation_version=self.operation_version,
        )


FEATURE_CATALOG: tuple[FeatureCatalogItem, ...] = (
    FeatureCatalogItem(
        id="page_rank",
        name="PageRank",
        description="PageRank scores with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="degree_centrality",
        name="Degree Centrality",
        description="Normalized node degree centrality with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="closeness_centrality",
        name="Closeness Centrality",
        description="Closeness centrality values with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="betweenness_centrality",
        name="Betweenness Centrality",
        description="Shortest-path betweenness centrality with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="eigenvector_centrality",
        name="Eigenvector Centrality",
        description="Eigenvector centrality values with NEExT convergence fallback behavior.",
    ),
    FeatureCatalogItem(
        id="clustering_coefficient",
        name="Clustering Coefficient",
        description="Local clustering coefficient with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="local_efficiency",
        name="Local Efficiency",
        description="Local efficiency scores with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="lsme",
        name="LSME",
        description="Local Spectral Method Embedding feature values.",
    ),
    FeatureCatalogItem(
        id="load_centrality",
        name="Load Centrality",
        description="Load centrality values with neighborhood aggregation.",
    ),
    FeatureCatalogItem(
        id="basic_expansion",
        name="Basic Expansion",
        description="Neighborhood expansion ratios across configured hops.",
    ),
    FeatureCatalogItem(
        id="betastar",
        name="Betastar",
        description="Community-aware Betastar node feature values.",
    ),
)


def list_feature_catalog_entries() -> list[FeatureCatalogEntry]:
    return [feature.to_public_entry() for feature in FEATURE_CATALOG]


def get_feature_catalog_item(feature_id: str) -> Optional[FeatureCatalogItem]:
    normalized = feature_id.strip().casefold()
    for feature in FEATURE_CATALOG:
        if feature.id.casefold() == normalized:
            return feature
    return None


def get_operation_definition(operation_id: str, operation_version: str) -> Optional[OperationDefinition]:
    return OPERATION_REGISTRY.get((operation_id, operation_version))
