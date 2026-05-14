"""Curated built-in embedding catalog for NEExT Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import EmbeddingCatalogEntry

GRAPH_EMBEDDING_OPERATION_ID = "neext.compute_graph_embeddings"
GRAPH_EMBEDDING_OPERATION_VERSION = "1"


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    operation_version: str
    label: str


OPERATION_REGISTRY: dict[tuple[str, str], OperationDefinition] = {
    (GRAPH_EMBEDDING_OPERATION_ID, GRAPH_EMBEDDING_OPERATION_VERSION): OperationDefinition(
        operation_id=GRAPH_EMBEDDING_OPERATION_ID,
        operation_version=GRAPH_EMBEDDING_OPERATION_VERSION,
        label="Compute NEExT graph embeddings",
    )
}


@dataclass(frozen=True)
class EmbeddingCatalogItem:
    id: str
    name: str
    description: str
    output: str = "Graph embedding vector columns"
    operation_id: str = GRAPH_EMBEDDING_OPERATION_ID
    operation_version: str = GRAPH_EMBEDDING_OPERATION_VERSION

    def to_public_entry(self) -> EmbeddingCatalogEntry:
        return EmbeddingCatalogEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            output=self.output,
            operation_id=self.operation_id,
            operation_version=self.operation_version,
        )


EMBEDDING_CATALOG: tuple[EmbeddingCatalogItem, ...] = (
    EmbeddingCatalogItem(
        id="approx_wasserstein",
        name="Approx Wasserstein",
        description="Approximate Wasserstein graph embeddings computed from configured node feature artifacts.",
    ),
    EmbeddingCatalogItem(
        id="wasserstein",
        name="Wasserstein",
        description="Wasserstein graph embeddings computed from configured node feature artifacts.",
    ),
    EmbeddingCatalogItem(
        id="sinkhornvectorizer",
        name="Sinkhorn Vectorizer",
        description="Sinkhorn vectorizer graph embeddings computed from configured node feature artifacts.",
    ),
)


def list_embedding_catalog_entries() -> list[EmbeddingCatalogEntry]:
    return [embedding.to_public_entry() for embedding in EMBEDDING_CATALOG]


def get_embedding_catalog_item(embedding_id: str) -> Optional[EmbeddingCatalogItem]:
    normalized = embedding_id.strip().casefold()
    for embedding in EMBEDDING_CATALOG:
        if embedding.id.casefold() == normalized:
            return embedding
    return None


def get_operation_definition(operation_id: str, operation_version: str) -> Optional[OperationDefinition]:
    return OPERATION_REGISTRY.get((operation_id, operation_version))
