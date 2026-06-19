"""Curated built-in model catalog for NEExT Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import ModelCatalogEntry

MODEL_TRAIN_OPERATION_ID = "neext.train_graph_model"
MODEL_TRAIN_OPERATION_VERSION = "1"


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    operation_version: str
    label: str


OPERATION_REGISTRY: dict[tuple[str, str], OperationDefinition] = {
    (MODEL_TRAIN_OPERATION_ID, MODEL_TRAIN_OPERATION_VERSION): OperationDefinition(
        operation_id=MODEL_TRAIN_OPERATION_ID,
        operation_version=MODEL_TRAIN_OPERATION_VERSION,
        label="Train NEExT supervised graph model",
    )
}


@dataclass(frozen=True)
class ModelCatalogItem:
    id: str
    name: str
    description: str
    output: str = "Graph-level supervised model metrics and trained model file"
    operation_id: str = MODEL_TRAIN_OPERATION_ID
    operation_version: str = MODEL_TRAIN_OPERATION_VERSION

    def to_public_entry(self) -> ModelCatalogEntry:
        return ModelCatalogEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            output=self.output,
            operation_id=self.operation_id,
            operation_version=self.operation_version,
        )


MODEL_CATALOG: tuple[ModelCatalogItem, ...] = (
    ModelCatalogItem(
        id="xgboost",
        name="XGBoost",
        description="Gradient boosted tree model for graph-level supervised learning.",
    ),
    ModelCatalogItem(
        id="random_forest",
        name="Random Forest",
        description="Random forest model for graph-level supervised learning.",
    ),
)


def list_model_catalog_entries() -> list[ModelCatalogEntry]:
    return [model.to_public_entry() for model in MODEL_CATALOG]


def get_model_catalog_item(model_id: str) -> Optional[ModelCatalogItem]:
    normalized = model_id.strip().casefold()
    for model in MODEL_CATALOG:
        if model.id.casefold() == normalized:
            return model
    return None


def get_operation_definition(operation_id: str, operation_version: str) -> Optional[OperationDefinition]:
    return OPERATION_REGISTRY.get((operation_id, operation_version))
