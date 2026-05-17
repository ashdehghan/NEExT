"""Curated built-in dataset catalog for NEExT Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schemas import DatasetCatalogEntry

CATALOG_SOURCE = "AnomalyPoint/NEExT_datasets"
CSV_BUNDLE_BASE_URL = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format"
WORKBENCH_CATALOG_SOURCE = "NEExT Workbench curated examples"
CATALOG_DATA_DIR = Path(__file__).with_name("catalog_data")


@dataclass(frozen=True)
class CatalogDataset:
    id: str
    name: str
    description: str
    domain: str
    files: dict[str, str]
    graph_count: int
    node_count: int
    edge_count: int
    source: str = CATALOG_SOURCE
    source_type: str = "neext_csv_bundle"
    source_graph_shape: str = "graph_collection"
    graph_shape: str = "graph_collection"
    node_attribute_columns: tuple[str, ...] = ()

    def to_public_entry(self) -> DatasetCatalogEntry:
        return DatasetCatalogEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            source=self.source,
            domain=self.domain,
            source_type=self.source_type,
            source_graph_shape=self.source_graph_shape,
            graph_shape=self.graph_shape,
            graph_count=self.graph_count,
            node_count=self.node_count,
            edge_count=self.edge_count,
            has_graph_labels="graph_labels" in self.files,
            has_node_features="node_features" in self.files or "nodes" in self.files and bool(self.node_attribute_columns),
            has_edge_features="edge_features" in self.files,
            node_attribute_columns=list(self.node_attribute_columns),
        )


def _bundle_files(dataset: str, *, node_features: bool = False, edge_features: bool = False) -> dict[str, str]:
    base_url = f"{CSV_BUNDLE_BASE_URL}/{dataset}"
    files = {
        "edges": f"{base_url}/edges.csv",
        "node_graph_mapping": f"{base_url}/node_graph_mapping.csv",
        "graph_labels": f"{base_url}/graph_labels.csv",
    }
    if node_features:
        files["node_features"] = f"{base_url}/node_features.csv"
    if edge_features:
        files["edge_features"] = f"{base_url}/edge_features.csv"
    return files


def _single_graph_files(dataset: str) -> dict[str, str]:
    dataset_dir = CATALOG_DATA_DIR / dataset
    return {
        "nodes": str(dataset_dir / "nodes.csv"),
        "edges": str(dataset_dir / "edges.csv"),
    }


DATASET_CATALOG: tuple[CatalogDataset, ...] = (
    CatalogDataset(
        id="MUTAG",
        name="MUTAG",
        description="Molecule graph collection for graph classification workflows.",
        domain="Molecules",
        files=_bundle_files("MUTAG"),
        graph_count=188,
        node_count=3371,
        edge_count=7442,
    ),
    CatalogDataset(
        id="NCI1",
        name="NCI1",
        description="Chemical compound graph collection with graph-level labels.",
        domain="Molecules",
        files=_bundle_files("NCI1"),
        graph_count=4110,
        node_count=122747,
        edge_count=265506,
    ),
    CatalogDataset(
        id="BZR",
        name="BZR",
        description="Benzodiazepine receptor molecule graph collection with node features.",
        domain="Molecules",
        files=_bundle_files("BZR", node_features=True),
        graph_count=405,
        node_count=14479,
        edge_count=31070,
    ),
    CatalogDataset(
        id="PROTEINS",
        name="PROTEINS",
        description="Protein graph collection with graph-level labels and node features.",
        domain="Bioinformatics",
        files=_bundle_files("PROTEINS", node_features=True),
        graph_count=1113,
        node_count=43471,
        edge_count=162088,
    ),
    CatalogDataset(
        id="IMDB",
        name="IMDB",
        description="Movie collaboration graph collection with graph-level labels.",
        domain="Social networks",
        files=_bundle_files("IMDB"),
        graph_count=1000,
        node_count=19773,
        edge_count=386124,
    ),
    CatalogDataset(
        id="KARATE_CLUB",
        name="Zachary Karate Club",
        description="Single social network with node-level club labels for egonet graph classification workflows.",
        domain="Social networks",
        files=_single_graph_files("karate_club"),
        graph_count=1,
        node_count=34,
        edge_count=78,
        source=WORKBENCH_CATALOG_SOURCE,
        source_type="neext_single_graph_csv",
        source_graph_shape="single_graph",
        node_attribute_columns=("club",),
    ),
)


def list_catalog_entries() -> list[DatasetCatalogEntry]:
    return [dataset.to_public_entry() for dataset in DATASET_CATALOG]


def get_catalog_dataset(catalog_id: str) -> Optional[CatalogDataset]:
    normalized = catalog_id.strip().casefold()
    for dataset in DATASET_CATALOG:
        if dataset.id.casefold() == normalized:
            return dataset
    return None
