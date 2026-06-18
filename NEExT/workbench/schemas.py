"""API and persistence schemas for NEExT Workbench."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class WorkspaceInfo(BaseModel):
    path: str
    version: str
    projects: int


class WorkspaceResetSummary(BaseModel):
    archived_path: str
    projects_archived: int
    trash_archived: bool


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


class LifecycleArtifactRef(BaseModel):
    artifact_kind: Literal["dataset", "feature", "embedding", "model"]
    artifact_id: str
    name: str
    status: str


class LifecycleJobRef(BaseModel):
    id: str
    status: Literal["queued", "running"]
    operation_id: str
    target_artifacts: list[LifecycleArtifactRef] = Field(default_factory=list)


class ArtifactDeletionPlan(BaseModel):
    project_id: str
    root_artifact: LifecycleArtifactRef
    artifacts: list[LifecycleArtifactRef]
    downstream_artifacts: list[LifecycleArtifactRef] = Field(default_factory=list)
    active_jobs: list[LifecycleJobRef] = Field(default_factory=list)
    requires_cascade: bool
    can_delete: bool


class ArtifactDeletionSummary(BaseModel):
    bundle_id: str
    project_id: str
    root_artifact: LifecycleArtifactRef
    artifacts: list[LifecycleArtifactRef]
    trashed_path: str


class TrashArtifactRef(LifecycleArtifactRef):
    original_path: str
    trashed_path: str


class ArtifactDeletionBundleManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["artifact_deletion_bundle"] = "artifact_deletion_bundle"
    id: str
    project_id: str
    root_artifact: LifecycleArtifactRef
    artifacts: list[TrashArtifactRef]
    created_at: str
    updated_at: str
    delete_mode: Literal["single", "cascade"]


class TrashedProjectEntry(BaseModel):
    trash_id: str
    project_id: str
    name: str
    description: str = ""
    trashed_path: str


class TrashListing(BaseModel):
    projects: list[TrashedProjectEntry] = Field(default_factory=list)
    artifact_deletions: list[ArtifactDeletionBundleManifest] = Field(default_factory=list)


class RestoreSummary(BaseModel):
    restored_kind: Literal["project", "artifact_deletion"]
    id: str
    name: str
    restored_path: str


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
    source_type: Literal["neext_csv_bundle", "neext_single_graph_csv", "uploaded_neext_tables"] = "neext_csv_bundle"
    source_catalog_id: str
    source_name: str = ""
    source: str = ""
    source_domain: str = ""
    source_graph_shape: Literal["graph_collection", "single_graph"] = "graph_collection"
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


class DatasetGraphSummary(BaseModel):
    graph_id: str
    node_count: int
    edge_count: int
    graph_label: Optional[Any] = None
    source_node_id: Optional[str] = None


class DatasetVisualNode(BaseModel):
    id: str
    label: str
    degree: int
    source_node_id: Optional[str] = None
    is_center: Optional[bool] = None


class DatasetVisualEdge(BaseModel):
    source: str
    target: str


class DatasetGraphVisual(BaseModel):
    graph_id: str
    node_count: int
    edge_count: int
    sampled: bool
    nodes: list[DatasetVisualNode]
    edges: list[DatasetVisualEdge]
    sample_reason: Optional[str] = None


class DatasetEgonetMetadata(BaseModel):
    source_graph_shape: Literal["single_graph"]
    operation_id: str
    operation_version: str
    k_hop: int
    node_selection: str
    sample_fraction: float
    random_seed: int
    target_node_attribute: Optional[str] = None


class DatasetAnalysis(BaseModel):
    dataset_id: str
    dataset_name: str
    dataset_status: Literal["completed"]
    egonet_metadata: Optional[DatasetEgonetMetadata] = None
    source_stats: DatasetStats
    prepared_stats: DatasetStats
    dropped_node_count: int
    graph_label_distribution: dict[str, int]
    node_feature_columns: list[str]
    edge_feature_columns: list[str]
    graph_summaries: list[DatasetGraphSummary]
    selected_graph_id: str
    visual: DatasetGraphVisual


class DatasetGraphSearchResult(BaseModel):
    kind: Literal["graph", "node"]
    graph_id: str
    node_id: Optional[str] = None
    graph_label: Optional[Any] = None
    node_count: int
    edge_count: int


class DatasetGraphSearchResponse(BaseModel):
    query: str
    limit: int
    total_matches: int
    results: list[DatasetGraphSearchResult]


class DatasetNodeDetail(BaseModel):
    graph_id: str
    node_id: str
    degree: int
    graph_label: Optional[Any] = None
    source_graph_id: Optional[str] = None
    source_node_id: Optional[str] = None
    feature_values: dict[str, Any] = Field(default_factory=dict)


class DatasetCatalogEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    source: str
    domain: str
    source_type: Literal["neext_csv_bundle", "neext_single_graph_csv"] = "neext_csv_bundle"
    source_graph_shape: Literal["graph_collection", "single_graph"] = "graph_collection"
    graph_shape: Literal["graph_collection"] = "graph_collection"
    graph_count: int
    node_count: int
    edge_count: int
    has_graph_labels: bool
    has_node_features: bool
    has_edge_features: bool
    node_attribute_columns: list[str] = Field(default_factory=list)


class DatasetPrepareParams(BaseModel):
    graph_type: Literal["networkx", "igraph"] = "networkx"
    filter_largest_component: bool = True
    k_hop: int = Field(default=1, ge=0, le=10)
    node_selection: Literal["all_nodes", "sample_fraction", "specific_node_ids"] = "all_nodes"
    sample_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    random_seed: int = Field(default=13, ge=0)
    source_node_ids: list[str] = Field(default_factory=list)
    target_node_attribute: Optional[str] = None


class DatasetCreateRequest(BaseModel):
    catalog_id: str = Field(min_length=1)
    params: DatasetPrepareParams = Field(default_factory=DatasetPrepareParams)


class DatasetIntakeTablePayload(BaseModel):
    format: Literal["records", "csv"] = "records"
    records: list[dict[str, Any]] = Field(default_factory=list)
    csv: str = ""


class DatasetIntakeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    tables: dict[str, DatasetIntakeTablePayload] = Field(default_factory=dict)
    params: DatasetPrepareParams = Field(default_factory=DatasetPrepareParams)


class DatasetIntakeSessionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    params: DatasetPrepareParams = Field(default_factory=DatasetPrepareParams)


class DatasetIntakeSessionTableRequest(BaseModel):
    table: DatasetIntakeTablePayload
    replace: bool = False


class DatasetIntakeValidationError(BaseModel):
    table: str
    message: str
    column: Optional[str] = None
    row: Optional[int] = None
    sample: Optional[list[Any]] = None


class DatasetIntakeValidationResponse(BaseModel):
    valid: bool
    errors: list[DatasetIntakeValidationError] = Field(default_factory=list)
    stats: Optional[DatasetStats] = None
    columns: dict[str, list[str]] = Field(default_factory=dict)


class DatasetIntakeSessionResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: str
    expires_at: str
    tables: list[str] = Field(default_factory=list)
    validation: Optional[DatasetIntakeValidationResponse] = None


class FeatureExpectedOutput(BaseModel):
    artifact_kind: Literal["feature"] = "feature"
    storage_format: Literal["neext-feature-parquet-v1"] = "neext-feature-parquet-v1"
    columns: list[str]


class FeatureOutputFiles(BaseModel):
    features: str


class FeatureOutputStats(BaseModel):
    row_count: int
    column_count: int


class FeatureColumnSummary(BaseModel):
    column: str
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    null_count: int


class FeatureCoverage(BaseModel):
    covered: int
    total: int


class FeatureAnalysisDataset(BaseModel):
    id: str
    name: str
    status: Literal["planned", "running", "completed", "failed"]
    prepared_stats: Optional[DatasetStats] = None


class FeatureAnalysisMethod(BaseModel):
    id: str
    name: str


class FeaturePcaPoint(BaseModel):
    graph_id: str
    x: float
    y: float
    graph_label: Optional[Any] = None
    color_value: str
    node_count: int


class FeaturePcaPayload(BaseModel):
    available: bool
    reason: Optional[str] = None
    plot_level: Literal["graph"] = "graph"
    aggregation_method: Literal["mean"] = "mean"
    projection_method: Literal["pca", "raw"] = "pca"
    x_axis_label: str = "PC1"
    y_axis_label: str = "PC2"
    color_by: Literal["graph_label", "graph_id"]
    numeric_columns: list[str]
    source_row_count: int
    total_graphs: int
    total_rows: int
    fit_row_count: int
    point_count: int
    max_fit_rows: int
    max_points: int
    fit_sampled: bool
    points_sampled: bool
    sampled: bool
    sample_reason: Optional[str] = None
    explained_variance_ratio: list[float] = Field(default_factory=list)
    points: list[FeaturePcaPoint] = Field(default_factory=list)


class FeatureAnalysis(BaseModel):
    feature_id: str
    feature_name: str
    feature_status: Literal["completed"]
    source_dataset: FeatureAnalysisDataset
    method: FeatureAnalysisMethod
    output_stats: FeatureOutputStats
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    column_summaries: list[FeatureColumnSummary]
    graph_coverage: FeatureCoverage
    node_coverage: FeatureCoverage
    graph_label_distribution: dict[str, int]
    pca: FeaturePcaPayload


class FeatureGraphSearchResult(BaseModel):
    kind: Literal["graph"] = "graph"
    graph_id: str
    graph_label: Optional[Any] = None
    in_pca_sample: bool
    node_count: int


class FeatureGraphSearchResponse(BaseModel):
    query: str
    limit: int
    total_matches: int
    results: list[FeatureGraphSearchResult]


class FeatureGraphDetail(BaseModel):
    graph_id: str
    graph_label: Optional[Any] = None
    node_count: int
    aggregation_method: Literal["mean"] = "mean"
    feature_values: dict[str, Any] = Field(default_factory=dict)


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
    source_type: Literal["neext_structural_node_feature", "custom_python_node_feature"] = "neext_structural_node_feature"
    source_feature_id: str
    source_code_file: Optional[str] = None
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


class CustomFeatureCreateParams(BaseModel):
    normalize_features: bool = True
    n_jobs: int = Field(default=1, ge=1, le=32)
    parallel_backend: Literal["loky", "threading"] = "threading"


class CustomFeatureCreateRequest(BaseModel):
    source_dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    code: str = Field(min_length=1)
    params: CustomFeatureCreateParams = Field(default_factory=CustomFeatureCreateParams)


class CustomFeatureValidateRequest(BaseModel):
    source_dataset_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    params: CustomFeatureCreateParams = Field(default_factory=CustomFeatureCreateParams)


class CustomFeatureValidationResponse(BaseModel):
    valid: Literal[True] = True
    columns: list[str]


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


class EmbeddingAnalysisFeature(BaseModel):
    id: str
    name: str
    status: Literal["planned", "running", "completed", "failed"]
    method: FeatureAnalysisMethod


class EmbeddingAnalysisAlgorithm(BaseModel):
    id: str
    name: str


class EmbeddingPcaPoint(BaseModel):
    graph_id: str
    x: float
    y: float
    graph_label: Optional[Any] = None
    color_value: str
    cluster: Optional[int] = None


class EmbeddingClusterSummary(BaseModel):
    cluster: int
    size: int
    dominant_label: Optional[Any] = None
    dominant_label_fraction: Optional[float] = None


class EmbeddingPcaPayload(BaseModel):
    available: bool
    reason: Optional[str] = None
    plot_level: Literal["graph"] = "graph"
    projection_method: Literal["pca", "raw", "tsne", "umap"] = "pca"
    x_axis_label: str = "PC1"
    y_axis_label: str = "PC2"
    color_by: Literal["graph_label", "graph_id"]
    numeric_columns: list[str]
    source_row_count: int
    total_graphs: int
    total_rows: int
    fit_row_count: int
    point_count: int
    max_fit_rows: int
    max_points: int
    fit_sampled: bool
    points_sampled: bool
    sampled: bool
    sample_reason: Optional[str] = None
    explained_variance_ratio: list[float] = Field(default_factory=list)
    points: list[EmbeddingPcaPoint] = Field(default_factory=list)
    # Clustering (populated only when an embedding clustering request is made)
    cluster_k: Optional[int] = None
    cluster_algorithm: Optional[Literal["kmeans"]] = None
    cluster_silhouette: Optional[float] = None
    cluster_label_ari: Optional[float] = None
    cluster_purity: Optional[float] = None
    clusters: list[EmbeddingClusterSummary] = Field(default_factory=list)


class EmbeddingAnalysis(BaseModel):
    embedding_id: str
    embedding_name: str
    embedding_status: Literal["completed"]
    source_dataset: FeatureAnalysisDataset
    source_features: list[EmbeddingAnalysisFeature]
    algorithm: EmbeddingAnalysisAlgorithm
    output_stats: EmbeddingOutputStats
    embedding_columns: list[str]
    numeric_embedding_columns: list[str]
    column_summaries: list[FeatureColumnSummary]
    graph_label_distribution: dict[str, int]
    pca: EmbeddingPcaPayload


class EmbeddingGraphSearchResult(BaseModel):
    kind: Literal["graph"] = "graph"
    graph_id: str
    graph_label: Optional[Any] = None
    in_pca_sample: bool


class EmbeddingGraphSearchResponse(BaseModel):
    query: str
    limit: int
    total_matches: int
    results: list[EmbeddingGraphSearchResult]


class EmbeddingGraphDetail(BaseModel):
    graph_id: str
    graph_label: Optional[Any] = None
    in_pca_sample: bool
    embedding_values: dict[str, Any] = Field(default_factory=dict)


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


class ModelCatalogEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    output: str
    operation_id: str
    operation_version: str


class ModelCreateParams(BaseModel):
    task_type: Literal["classifier", "regressor"]
    sample_size: int = Field(default=5, ge=1, le=100)
    test_size: float = Field(default=0.3, ge=0.05, le=0.95)
    balance_dataset: bool = False
    n_jobs: int = Field(default=1, ge=1, le=32)
    parallel_backend: Literal["thread", "process"] = "thread"


class ModelCreateRequest(BaseModel):
    source_model_id: str
    source_embedding_ids: list[str] = Field(min_length=1)
    params: ModelCreateParams


class ModelRunBatchRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1)


class ModelExpectedOutput(BaseModel):
    artifact_kind: Literal["model"] = "model"
    storage_format: Literal["neext-model-results-v1"] = "neext-model-results-v1"
    metrics: list[str]


class ModelOutputFiles(BaseModel):
    metrics: str
    model: str


class ModelOutputStats(BaseModel):
    metric_count: int
    sample_size: int
    feature_count: int
    graph_count: int


class ModelAnalysisAlgorithm(BaseModel):
    id: str
    name: str


class ModelAnalysisEmbedding(BaseModel):
    id: str
    name: str
    status: Literal["planned", "running", "completed", "failed"]
    algorithm: ModelAnalysisAlgorithm


class ModelMetricPoint(BaseModel):
    iteration: int
    value: Optional[float] = None


class ModelMetricSeries(BaseModel):
    metric: str
    points: list[ModelMetricPoint]


class ModelManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["model"] = "model"
    id: str
    project_id: str
    name: str
    description: str = ""
    status: Literal["planned", "running", "completed", "failed"] = "planned"
    created_at: str
    updated_at: str
    inputs: list[ArtifactInputRef]
    source_type: Literal["neext_supervised_graph_model"] = "neext_supervised_graph_model"
    source_model_id: str
    operation: OperationSpec
    expected_output: ModelExpectedOutput
    output_files: Optional[ModelOutputFiles] = None
    output_stats: Optional[ModelOutputStats] = None
    error: Optional[ArtifactError] = None
    feature_importance_status: Optional[Literal["running", "completed", "failed"]] = None
    feature_importance_file: Optional[str] = None
    feature_importance_error: Optional[ArtifactError] = None


class ModelFeatureImportanceRunRequest(BaseModel):
    algorithm: Literal["supervised_fast", "supervised_greedy"] = "supervised_fast"
    n_iterations: int = Field(default=3, ge=1, le=25)


class ModelFeatureImportanceItem(BaseModel):
    rank: int
    feature_name: str
    score: float


class ModelFeatureImportancePayload(BaseModel):
    status: Literal["running", "completed", "failed"]
    algorithm: str
    embedding_algorithm: str
    score_label: str = "importance"
    ranking: list[ModelFeatureImportanceItem] = Field(default_factory=list)
    computed_at: Optional[str] = None
    error: Optional[str] = None


class ModelPreview(BaseModel):
    summary: dict[str, Union[float, int, str, list[str]]]
    metrics: list[dict[str, Any]]
    feature_columns: list[str]
    classes: Optional[list[str]] = None


class ModelAnalysis(BaseModel):
    model_id: str
    model_name: str
    model_status: Literal["completed"]
    source_dataset: FeatureAnalysisDataset
    source_embeddings: list[ModelAnalysisEmbedding]
    source_features: list[EmbeddingAnalysisFeature]
    algorithm: ModelAnalysisAlgorithm
    task_type: Literal["classifier", "regressor"]
    expected_metrics: list[str]
    output_stats: ModelOutputStats
    sample_size: int
    test_size: float
    random_state: Optional[int] = None
    classes: Optional[list[str]] = None
    feature_columns: list[str]
    summary: dict[str, Union[float, int, str, list[str], None]]
    metrics: list[dict[str, Any]]
    metric_series: list[ModelMetricSeries]
    feature_importance: Optional[ModelFeatureImportancePayload] = None


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


class McpSettingsManifest(BaseModel):
    schema_version: str = "1"
    manifest_type: Literal["mcp_settings"] = "mcp_settings"
    enabled: bool = False
    token_hash: Optional[str] = None
    token_preview: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class McpClientConfigSnippet(BaseModel):
    client: str
    label: str
    target: str
    content: str


class McpStdioReadiness(BaseModel):
    """Result of validating the local stdio launch path before showing config.

    The stdio transport is the only way Claude Desktop can reach a local Workbench
    server, so the command must be runnable from the exact interpreter that runs
    Workbench. ``issues``/``remediation`` are populated when the environment cannot
    host a working stdio server (e.g. a macOS protected folder).
    """

    status: Literal["ready", "blocked"] = "ready"
    ok: bool = True
    interpreter: str = ""
    command_preview: str = ""
    issues: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)


class McpSettingsResponse(BaseModel):
    enabled: bool
    token_preview: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    one_time_token: Optional[str] = None
    endpoint_url: Optional[str] = None
    transport: str = "streamable-http"
    protocol_version: str = "2025-11-25"
    sdk_transport_available: bool = False
    scopes: list[str] = Field(default_factory=list)
    capabilities: list[dict[str, Any]] = Field(default_factory=list)
    client_configs: list[McpClientConfigSnippet] = Field(default_factory=list)
    stdio_readiness: Optional[McpStdioReadiness] = None


class McpActivityEntry(BaseModel):
    id: str
    created_at: str
    event_type: str
    status: Literal["pending", "completed", "failed", "denied"] = "completed"
    message: str
    tool_name: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class McpActivityListing(BaseModel):
    entries: list[McpActivityEntry] = Field(default_factory=list)


class McpUiState(BaseModel):
    id: str
    created_at: str
    route: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class McpApprovalRequest(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: Literal["pending", "approved", "denied", "failed"] = "pending"
    operation: Literal["delete_project", "delete_artifact"]
    summary: str
    project_id: Optional[str] = None
    artifact_kind: Optional[Literal["dataset", "feature", "embedding", "model"]] = None
    artifact_id: Optional[str] = None
    cascade: bool = False
    plan: Optional[ArtifactDeletionPlan] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class McpApprovalListing(BaseModel):
    approvals: list[McpApprovalRequest] = Field(default_factory=list)


class McpApprovalDecision(BaseModel):
    reason: str = ""
