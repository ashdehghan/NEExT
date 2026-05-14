"""API and persistence schemas for NEExT Workbench."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class WorkspaceInfo(BaseModel):
    path: str
    version: str
    projects: int


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class ProjectManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["project"] = "project"
    id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: str


class ProjectDeletionSummary(BaseModel):
    id: str
    name: str
    trashed_path: str


class DatasetStats(BaseModel):
    graph_count: int
    node_count: int
    edge_count: int
    has_graph_labels: bool
    has_node_features: bool
    has_edge_features: bool


class DatasetDataFiles(BaseModel):
    nodes: str
    edges: str
    graph_labels: Optional[str] = None
    node_features: Optional[str] = None
    edge_features: Optional[str] = None


class DatasetMappingFiles(BaseModel):
    node_mapping: str
    graph_mapping: Optional[str] = None


class ArtifactError(BaseModel):
    message: str
    job_id: Optional[str] = None


class ArtifactInputRef(BaseModel):
    role: str
    artifact_kind: str
    artifact_id: str


class OperationSpec(BaseModel):
    operation_id: str
    operation_version: str
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["dataset"] = "dataset"
    id: str
    project_id: str
    name: str
    description: str = ""
    status: Literal["planned", "running", "completed", "failed"] = "planned"
    created_at: str
    updated_at: str
    source_type: Literal["neext_csv_bundle"] = "neext_csv_bundle"
    source_catalog_id: str
    source_name: str = ""
    source: str = ""
    source_domain: str = ""
    storage_format: Literal["neext-parquet-v1"] = "neext-parquet-v1"
    graph_shape: Literal["graph_collection"] = "graph_collection"
    inputs: list[ArtifactInputRef] = Field(default_factory=list)
    operation: OperationSpec
    source_stats: DatasetStats
    prepared_stats: Optional[DatasetStats] = None
    raw_data_files: Optional[DatasetDataFiles] = None
    prepared_data_files: Optional[DatasetDataFiles] = None
    mapping_files: Optional[DatasetMappingFiles] = None
    error: Optional[ArtifactError] = None
    data_files: Optional[DatasetDataFiles] = None
    stats: Optional[DatasetStats] = None


class DatasetCatalogEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    source: str
    domain: str
    source_type: Literal["neext_csv_bundle"] = "neext_csv_bundle"
    graph_shape: Literal["graph_collection"] = "graph_collection"
    graph_count: int
    node_count: int
    edge_count: int
    has_graph_labels: bool
    has_node_features: bool
    has_edge_features: bool


class DatasetPrepareParams(BaseModel):
    graph_type: Literal["networkx", "igraph"] = "networkx"
    filter_largest_component: bool = True


class DatasetCreateRequest(BaseModel):
    catalog_id: str = Field(min_length=1)
    params: DatasetPrepareParams = Field(default_factory=DatasetPrepareParams)


class FeatureExpectedOutput(BaseModel):
    artifact_kind: Literal["feature"] = "feature"
    storage_format: Literal["neext-feature-parquet-v1"] = "neext-feature-parquet-v1"
    columns: list[str]


class FeatureOutputFiles(BaseModel):
    features: str


class FeatureOutputStats(BaseModel):
    row_count: int
    column_count: int


class FeatureManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["feature"] = "feature"
    id: str
    project_id: str
    name: str
    description: str = ""
    status: Literal["planned", "running", "completed", "failed"] = "planned"
    created_at: str
    updated_at: str
    inputs: list[ArtifactInputRef]
    source_type: Literal["neext_structural_node_feature"] = "neext_structural_node_feature"
    source_feature_id: str
    operation: OperationSpec
    expected_output: FeatureExpectedOutput
    output_files: Optional[FeatureOutputFiles] = None
    output_stats: Optional[FeatureOutputStats] = None
    error: Optional[ArtifactError] = None


class FeatureCreateParams(BaseModel):
    feature_vector_length: int = Field(default=3, ge=1, le=10)
    normalize_features: bool = True
    n_jobs: int = Field(default=1, ge=1, le=32)
    parallel_backend: Literal["loky", "threading"] = "loky"


class FeatureCreateRequest(BaseModel):
    source_dataset_id: str = Field(min_length=1)
    source_feature_id: str = Field(min_length=1)
    params: FeatureCreateParams = Field(default_factory=FeatureCreateParams)


class FeatureRunBatchRequest(BaseModel):
    feature_ids: list[str] = Field(min_length=1)


class EmbeddingExpectedOutput(BaseModel):
    artifact_kind: Literal["embedding"] = "embedding"
    storage_format: Literal["neext-embedding-parquet-v1"] = "neext-embedding-parquet-v1"
    columns: list[str]


class EmbeddingOutputFiles(BaseModel):
    embeddings: str


class EmbeddingOutputStats(BaseModel):
    row_count: int
    column_count: int


class EmbeddingManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["embedding"] = "embedding"
    id: str
    project_id: str
    name: str
    description: str = ""
    status: Literal["planned", "running", "completed", "failed"] = "planned"
    created_at: str
    updated_at: str
    inputs: list[ArtifactInputRef]
    source_type: Literal["neext_graph_embedding"] = "neext_graph_embedding"
    source_embedding_id: str
    operation: OperationSpec
    expected_output: EmbeddingExpectedOutput
    output_files: Optional[EmbeddingOutputFiles] = None
    output_stats: Optional[EmbeddingOutputStats] = None
    error: Optional[ArtifactError] = None


class EmbeddingCreateParams(BaseModel):
    embedding_dimension: int = Field(default=3, ge=1, le=128)


class EmbeddingCreateRequest(BaseModel):
    source_embedding_id: str = Field(min_length=1)
    source_feature_ids: list[str] = Field(min_length=1)
    params: EmbeddingCreateParams = Field(default_factory=EmbeddingCreateParams)


class EmbeddingRunBatchRequest(BaseModel):
    embedding_ids: list[str] = Field(min_length=1)


class JobArtifactRef(BaseModel):
    artifact_kind: str
    artifact_id: str


class JobEvent(BaseModel):
    timestamp: str
    level: Literal["info", "error"] = "info"
    message: str


class JobManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["job"] = "job"
    id: str
    project_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    operation: OperationSpec
    target_artifacts: list[JobArtifactRef] = Field(default_factory=list)
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    events: list[JobEvent] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class TabularPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    offset: int
    limit: int
    total_rows: int


class FeatureCatalogEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    type: Literal["structural_node_feature"] = "structural_node_feature"
    source_type: Literal["neext_structural_node_feature"] = "neext_structural_node_feature"
    output: str
    operation_id: str
    operation_version: str


class EmbeddingCatalogEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    output: str
    operation_id: str
    operation_version: str
