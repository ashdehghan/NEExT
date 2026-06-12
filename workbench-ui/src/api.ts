export interface WorkspaceInfo {
  path: string;
  version: string;
  projects: number;
}

export interface ProjectManifest {
  schema_version: string;
  manifest_type: "project";
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectDeletionSummary {
  id: string;
  name: string;
  trashed_path: string;
}

export type ArtifactKind = "dataset" | "feature" | "embedding" | "model";

export interface LifecycleArtifactRef {
  artifact_kind: ArtifactKind;
  artifact_id: string;
  name: string;
  status: string;
}

export interface LifecycleJobRef {
  id: string;
  status: "queued" | "running";
  operation_id: string;
  target_artifacts: LifecycleArtifactRef[];
}

export interface ArtifactDeletionPlan {
  project_id: string;
  root_artifact: LifecycleArtifactRef;
  artifacts: LifecycleArtifactRef[];
  downstream_artifacts: LifecycleArtifactRef[];
  active_jobs: LifecycleJobRef[];
  requires_cascade: boolean;
  can_delete: boolean;
}

export interface ArtifactDeletionSummary {
  bundle_id: string;
  project_id: string;
  root_artifact: LifecycleArtifactRef;
  artifacts: LifecycleArtifactRef[];
  trashed_path: string;
}

export interface TrashArtifactRef extends LifecycleArtifactRef {
  original_path: string;
  trashed_path: string;
}

export interface ArtifactDeletionBundleManifest {
  schema_version: string;
  manifest_type: "artifact_deletion_bundle";
  id: string;
  project_id: string;
  root_artifact: LifecycleArtifactRef;
  artifacts: TrashArtifactRef[];
  created_at: string;
  updated_at: string;
  delete_mode: "single" | "cascade";
}

export interface TrashedProjectEntry {
  trash_id: string;
  project_id: string;
  name: string;
  description: string;
  trashed_path: string;
}

export interface TrashListing {
  projects: TrashedProjectEntry[];
  artifact_deletions: ArtifactDeletionBundleManifest[];
}

export interface RestoreSummary {
  restored_kind: "project" | "artifact_deletion";
  id: string;
  name: string;
  restored_path: string;
}

export interface DatasetStats {
  graph_count: number;
  node_count: number;
  edge_count: number;
  has_graph_labels: boolean;
  has_node_features: boolean;
  has_edge_features: boolean;
}

export interface DatasetDataFiles {
  nodes: string;
  edges: string;
  graph_labels?: string | null;
  node_features?: string | null;
  edge_features?: string | null;
}

export interface DatasetMappingFiles {
  node_mapping: string;
  graph_mapping?: string | null;
}

export interface ArtifactError {
  message: string;
  job_id?: string | null;
}

export interface ArtifactInputRef {
  role: string;
  artifact_kind: string;
  artifact_id: string;
}

export interface DatasetManifest {
  schema_version: string;
  manifest_type: "dataset";
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: "planned" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  source_type: "neext_csv_bundle" | "neext_single_graph_csv";
  source_catalog_id: string;
  source_name: string;
  source: string;
  source_domain: string;
  source_graph_shape: "graph_collection" | "single_graph";
  storage_format: "neext-parquet-v1";
  graph_shape: "graph_collection";
  inputs: ArtifactInputRef[];
  operation: OperationSpec;
  source_stats: DatasetStats;
  prepared_stats?: DatasetStats | null;
  raw_data_files?: DatasetDataFiles | null;
  prepared_data_files?: DatasetDataFiles | null;
  mapping_files?: DatasetMappingFiles | null;
  error?: ArtifactError | null;
  data_files?: DatasetDataFiles | null;
  stats?: DatasetStats | null;
}

export interface DatasetGraphSummary {
  graph_id: string;
  node_count: number;
  edge_count: number;
  graph_label?: unknown;
  source_node_id?: string | null;
}

export interface DatasetVisualNode {
  id: string;
  label: string;
  degree: number;
  source_node_id?: string | null;
  is_center?: boolean | null;
}

export interface DatasetVisualEdge {
  source: string;
  target: string;
}

export interface DatasetGraphVisual {
  graph_id: string;
  node_count: number;
  edge_count: number;
  sampled: boolean;
  nodes: DatasetVisualNode[];
  edges: DatasetVisualEdge[];
  sample_reason?: string | null;
}

export interface DatasetEgonetMetadata {
  source_graph_shape: "single_graph";
  operation_id: string;
  operation_version: string;
  k_hop: number;
  node_selection: string;
  sample_fraction: number;
  random_seed: number;
  target_node_attribute?: string | null;
}

export interface DatasetAnalysis {
  dataset_id: string;
  dataset_name: string;
  dataset_status: "completed";
  egonet_metadata?: DatasetEgonetMetadata | null;
  source_stats: DatasetStats;
  prepared_stats: DatasetStats;
  dropped_node_count: number;
  graph_label_distribution: Record<string, number>;
  node_feature_columns: string[];
  edge_feature_columns: string[];
  graph_summaries: DatasetGraphSummary[];
  selected_graph_id: string;
  visual: DatasetGraphVisual;
}

export interface DatasetGraphSearchResult {
  kind: "graph" | "node";
  graph_id: string;
  node_id?: string | null;
  graph_label?: unknown;
  node_count: number;
  edge_count: number;
}

export interface DatasetGraphSearchResponse {
  query: string;
  limit: number;
  total_matches: number;
  results: DatasetGraphSearchResult[];
}

export interface DatasetNodeDetail {
  graph_id: string;
  node_id: string;
  degree: number;
  graph_label?: unknown;
  source_graph_id?: string | null;
  source_node_id?: string | null;
  feature_values: Record<string, unknown>;
}

export interface DatasetCatalogEntry {
  id: string;
  name: string;
  description: string;
  source: string;
  domain: string;
  source_type: "neext_csv_bundle" | "neext_single_graph_csv";
  source_graph_shape: "graph_collection" | "single_graph";
  graph_shape: "graph_collection";
  graph_count: number;
  node_count: number;
  edge_count: number;
  has_graph_labels: boolean;
  has_node_features: boolean;
  has_edge_features: boolean;
  node_attribute_columns: string[];
}

export interface OperationSpec {
  operation_id: string;
  operation_version: string;
  params: Record<string, any>;
}

export interface FeatureExpectedOutput {
  artifact_kind: "feature";
  storage_format: "neext-feature-parquet-v1";
  columns: string[];
}

export interface FeatureManifest {
  schema_version: string;
  manifest_type: "feature";
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: "planned" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  inputs: ArtifactInputRef[];
  source_type: "neext_structural_node_feature";
  source_feature_id: string;
  operation: OperationSpec;
  expected_output: FeatureExpectedOutput;
  output_files?: { features: string } | null;
  output_stats?: { row_count: number; column_count: number } | null;
  error?: ArtifactError | null;
}

export interface FeatureColumnSummary {
  column: string;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  null_count: number;
}

export interface FeatureCoverage {
  covered: number;
  total: number;
}

export interface FeatureAnalysisDataset {
  id: string;
  name: string;
  status: "planned" | "running" | "completed" | "failed";
  prepared_stats?: DatasetStats | null;
}

export interface FeatureAnalysisMethod {
  id: string;
  name: string;
}

export interface FeaturePcaPoint {
  graph_id: string;
  x: number;
  y: number;
  graph_label?: unknown;
  color_value: string;
  node_count: number;
}

export interface FeaturePcaPayload {
  available: boolean;
  reason?: string | null;
  plot_level: "graph";
  aggregation_method: "mean";
  projection_method: "pca" | "raw";
  x_axis_label: string;
  y_axis_label: string;
  color_by: "graph_label" | "graph_id";
  numeric_columns: string[];
  source_row_count: number;
  total_graphs: number;
  total_rows: number;
  fit_row_count: number;
  point_count: number;
  max_fit_rows: number;
  max_points: number;
  fit_sampled: boolean;
  points_sampled: boolean;
  sampled: boolean;
  sample_reason?: string | null;
  explained_variance_ratio: number[];
  points: FeaturePcaPoint[];
}

export interface FeatureAnalysis {
  feature_id: string;
  feature_name: string;
  feature_status: "completed";
  source_dataset: FeatureAnalysisDataset;
  method: FeatureAnalysisMethod;
  output_stats: { row_count: number; column_count: number };
  feature_columns: string[];
  numeric_feature_columns: string[];
  column_summaries: FeatureColumnSummary[];
  graph_coverage: FeatureCoverage;
  node_coverage: FeatureCoverage;
  graph_label_distribution: Record<string, number>;
  pca: FeaturePcaPayload;
}

export interface FeatureGraphSearchResult {
  kind: "graph";
  graph_id: string;
  graph_label?: unknown;
  in_pca_sample: boolean;
  node_count: number;
}

export interface FeatureGraphSearchResponse {
  query: string;
  limit: number;
  total_matches: number;
  results: FeatureGraphSearchResult[];
}

export interface FeatureGraphDetail {
  graph_id: string;
  graph_label?: unknown;
  node_count: number;
  aggregation_method: "mean";
  feature_values: Record<string, unknown>;
}

export interface EmbeddingExpectedOutput {
  artifact_kind: "embedding";
  storage_format: "neext-embedding-parquet-v1";
  columns: string[];
}

export interface EmbeddingOutputFiles {
  embeddings: string;
}

export interface EmbeddingOutputStats {
  row_count: number;
  column_count: number;
}

export interface EmbeddingAnalysisFeature {
  id: string;
  name: string;
  status: "planned" | "running" | "completed" | "failed";
  method: FeatureAnalysisMethod;
}

export interface EmbeddingAnalysisAlgorithm {
  id: string;
  name: string;
}

export interface EmbeddingPcaPoint {
  graph_id: string;
  x: number;
  y: number;
  graph_label?: unknown;
  color_value: string;
}

export interface EmbeddingPcaPayload {
  available: boolean;
  reason?: string | null;
  plot_level: "graph";
  projection_method: "pca" | "raw";
  x_axis_label: string;
  y_axis_label: string;
  color_by: "graph_label" | "graph_id";
  numeric_columns: string[];
  source_row_count: number;
  total_graphs: number;
  total_rows: number;
  fit_row_count: number;
  point_count: number;
  max_fit_rows: number;
  max_points: number;
  fit_sampled: boolean;
  points_sampled: boolean;
  sampled: boolean;
  sample_reason?: string | null;
  explained_variance_ratio: number[];
  points: EmbeddingPcaPoint[];
}

export interface EmbeddingAnalysis {
  embedding_id: string;
  embedding_name: string;
  embedding_status: "completed";
  source_dataset: FeatureAnalysisDataset;
  source_features: EmbeddingAnalysisFeature[];
  algorithm: EmbeddingAnalysisAlgorithm;
  output_stats: EmbeddingOutputStats;
  embedding_columns: string[];
  numeric_embedding_columns: string[];
  column_summaries: FeatureColumnSummary[];
  graph_label_distribution: Record<string, number>;
  pca: EmbeddingPcaPayload;
}

export interface EmbeddingGraphSearchResult {
  kind: "graph";
  graph_id: string;
  graph_label?: unknown;
  in_pca_sample: boolean;
}

export interface EmbeddingGraphSearchResponse {
  query: string;
  limit: number;
  total_matches: number;
  results: EmbeddingGraphSearchResult[];
}

export interface EmbeddingGraphDetail {
  graph_id: string;
  graph_label?: unknown;
  in_pca_sample: boolean;
  embedding_values: Record<string, unknown>;
}

export interface EmbeddingManifest {
  schema_version: string;
  manifest_type: "embedding";
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: "planned" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  inputs: ArtifactInputRef[];
  source_type: "neext_graph_embedding";
  source_embedding_id: string;
  operation: OperationSpec;
  expected_output: EmbeddingExpectedOutput;
  output_files?: EmbeddingOutputFiles | null;
  output_stats?: EmbeddingOutputStats | null;
  error?: ArtifactError | null;
}

export interface ModelExpectedOutput {
  artifact_kind: "model";
  storage_format: "neext-model-results-v1";
  metrics: string[];
}

export interface ModelOutputFiles {
  metrics: string;
  model: string;
}

export interface ModelOutputStats {
  metric_count: number;
  sample_size: number;
  feature_count: number;
  graph_count: number;
}

export interface ModelAnalysisAlgorithm {
  id: string;
  name: string;
}

export interface ModelAnalysisEmbedding {
  id: string;
  name: string;
  status: "planned" | "running" | "completed" | "failed";
  algorithm: ModelAnalysisAlgorithm;
}

export interface ModelMetricPoint {
  iteration: number;
  value?: number | null;
}

export interface ModelMetricSeries {
  metric: string;
  points: ModelMetricPoint[];
}

export interface ModelManifest {
  schema_version: string;
  manifest_type: "model";
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: "planned" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  inputs: ArtifactInputRef[];
  source_type: "neext_supervised_graph_model";
  source_model_id: string;
  operation: OperationSpec;
  expected_output: ModelExpectedOutput;
  output_files?: ModelOutputFiles | null;
  output_stats?: ModelOutputStats | null;
  error?: ArtifactError | null;
}

export interface JobManifest {
  schema_version: string;
  manifest_type: "job";
  id: string;
  project_id: string;
  status: "queued" | "running" | "completed" | "failed";
  operation: OperationSpec;
  target_artifacts: { artifact_kind: string; artifact_id: string }[];
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  events: { timestamp: string; level: "info" | "error"; message: string }[];
  log: string[];
  error?: string | null;
}

export interface TabularPreview {
  columns: string[];
  rows: Record<string, unknown>[];
  offset: number;
  limit: number;
  total_rows: number;
}

export interface McpClientConfigSnippet {
  client: string;
  label: string;
  target: string;
  content: string;
}

export interface McpSettingsResponse {
  enabled: boolean;
  token_preview?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  one_time_token?: string | null;
  client_configs: McpClientConfigSnippet[];
}

export type DatasetPreviewTable =
  | "nodes"
  | "edges"
  | "graph_labels"
  | "node_features"
  | "edge_features"
  | "node_mapping"
  | "graph_mapping"
  | "mapping";

export interface DatasetCreatePayload {
  catalog_id: string;
  params: {
    graph_type: "networkx" | "igraph";
    filter_largest_component: boolean;
    k_hop?: number;
    node_selection?: "all_nodes" | "sample_fraction" | "specific_node_ids";
    sample_fraction?: number;
    random_seed?: number;
    source_node_ids?: string[];
    target_node_attribute?: string | null;
  };
}

export interface FeatureCatalogEntry {
  id: string;
  name: string;
  description: string;
  type: "structural_node_feature";
  source_type: "neext_structural_node_feature";
  output: string;
  operation_id: string;
  operation_version: string;
}

export interface EmbeddingCatalogEntry {
  id: string;
  name: string;
  description: string;
  output: string;
  operation_id: string;
  operation_version: string;
}

export interface ModelCatalogEntry {
  id: string;
  name: string;
  description: string;
  output: string;
  operation_id: string;
  operation_version: string;
}

export interface FeatureCreatePayload {
  source_dataset_id: string;
  source_feature_id: string;
  params: {
    feature_vector_length: number;
    normalize_features: boolean;
    n_jobs: number;
    parallel_backend: "loky" | "threading";
  };
}

export interface EmbeddingCreatePayload {
  source_embedding_id: string;
  source_feature_ids: string[];
  params: {
    embedding_dimension: number;
  };
}

export interface ModelCreatePayload {
  source_model_id: string;
  source_embedding_ids: string[];
  params: {
    task_type: "classifier" | "regressor";
    sample_size: number;
    test_size: number;
    balance_dataset: boolean;
    n_jobs: number;
    parallel_backend: "thread" | "process";
  };
}

export interface ModelPreview {
  summary: Record<string, number | string | string[]>;
  metrics: Record<string, unknown>[];
  feature_columns: string[];
  classes?: string[] | null;
}

export interface ModelAnalysis {
  model_id: string;
  model_name: string;
  model_status: "completed";
  source_dataset: FeatureAnalysisDataset;
  source_embeddings: ModelAnalysisEmbedding[];
  source_features: EmbeddingAnalysisFeature[];
  algorithm: ModelAnalysisAlgorithm;
  task_type: "classifier" | "regressor";
  expected_metrics: string[];
  output_stats: ModelOutputStats;
  sample_size: number;
  test_size: number;
  random_state?: number | null;
  classes?: string[] | null;
  feature_columns: string[];
  summary: Record<string, number | string | string[] | null>;
  metrics: Record<string, unknown>[];
  metric_series: ModelMetricSeries[];
}

export interface DownloadPayload {
  blob: Blob;
  filename: string;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message?: unknown }).message || fallback);
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = payload.detail ?? response.statusText;
    throw new ApiError(errorMessage(detail, response.statusText), response.status, detail);
  }
  return response.json() as Promise<T>;
}

async function requestDownload(path: string): Promise<DownloadPayload> {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = payload.detail ?? response.statusText;
    throw new ApiError(errorMessage(detail, response.statusText), response.status, detail);
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="([^"]+)"/);
  return {
    blob: await response.blob(),
    filename: filenameMatch?.[1] || "download.csv"
  };
}

export const api = {
  workspace: () => request<WorkspaceInfo>("/api/workspace"),
  mcpSettings: () => request<McpSettingsResponse>("/api/mcp-settings"),
  enableMcpSettings: () => request<McpSettingsResponse>("/api/mcp-settings/enable", { method: "POST" }),
  regenerateMcpSettings: () => request<McpSettingsResponse>("/api/mcp-settings/regenerate", { method: "POST" }),
  disableMcpSettings: () => request<McpSettingsResponse>("/api/mcp-settings/disable", { method: "POST" }),
  projects: () => request<ProjectManifest[]>("/api/projects"),
  trash: () => request<TrashListing>("/api/trash"),
  restoreProject: (trashId: string) => request<RestoreSummary>(`/api/trash/projects/${trashId}/restore`, { method: "POST" }),
  restoreArtifactDeletion: (projectId: string, bundleId: string) =>
    request<RestoreSummary>(`/api/projects/${projectId}/trash/artifact-deletions/${bundleId}/restore`, { method: "POST" }),
  project: (projectId: string) => request<ProjectManifest>(`/api/projects/${projectId}`),
  datasetLibrary: () => request<DatasetCatalogEntry[]>("/api/dataset-library"),
  featureLibrary: () => request<FeatureCatalogEntry[]>("/api/feature-library"),
  embeddingLibrary: () => request<EmbeddingCatalogEntry[]>("/api/embedding-library"),
  modelLibrary: () => request<ModelCatalogEntry[]>("/api/model-library"),
  projectDatasets: (projectId: string) => request<DatasetManifest[]>(`/api/projects/${projectId}/datasets`),
  projectDataset: (projectId: string, datasetId: string) =>
    request<DatasetManifest>(`/api/projects/${projectId}/datasets/${datasetId}`),
  projectFeatures: (projectId: string) => request<FeatureManifest[]>(`/api/projects/${projectId}/features`),
  projectFeature: (projectId: string, featureId: string) =>
    request<FeatureManifest>(`/api/projects/${projectId}/features/${featureId}`),
  projectEmbeddings: (projectId: string) => request<EmbeddingManifest[]>(`/api/projects/${projectId}/embeddings`),
  projectEmbedding: (projectId: string, embeddingId: string) =>
    request<EmbeddingManifest>(`/api/projects/${projectId}/embeddings/${embeddingId}`),
  projectModels: (projectId: string) => request<ModelManifest[]>(`/api/projects/${projectId}/models`),
  projectModel: (projectId: string, modelId: string) =>
    request<ModelManifest>(`/api/projects/${projectId}/models/${modelId}`),
  projectJobs: (projectId: string) => request<JobManifest[]>(`/api/projects/${projectId}/jobs`),
  projectJob: (projectId: string, jobId: string) => request<JobManifest>(`/api/projects/${projectId}/jobs/${jobId}`),
  createDataset: (projectId: string, payload: DatasetCreatePayload) =>
    request<DatasetManifest>(`/api/projects/${projectId}/datasets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  runDataset: (projectId: string, datasetId: string) =>
    request<JobManifest>(`/api/projects/${projectId}/datasets/${datasetId}/run`, { method: "POST" }),
  datasetAnalysis: (
    projectId: string,
    datasetId: string,
    params: { graph_id?: string; max_nodes?: number; max_edges?: number } = {}
  ) => {
    const search = new URLSearchParams();
    if (params.graph_id) search.set("graph_id", params.graph_id);
    if (params.max_nodes != null) search.set("max_nodes", String(params.max_nodes));
    if (params.max_edges != null) search.set("max_edges", String(params.max_edges));
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<DatasetAnalysis>(`/api/projects/${projectId}/datasets/${datasetId}/analysis${suffix}`);
  },
  datasetGraphSearch: (projectId: string, datasetId: string, query: string, limit = 25) => {
    const search = new URLSearchParams();
    search.set("query", query);
    search.set("limit", String(limit));
    return request<DatasetGraphSearchResponse>(`/api/projects/${projectId}/datasets/${datasetId}/analysis/search?${search.toString()}`);
  },
  datasetNodeDetail: (projectId: string, datasetId: string, graphId: string, nodeId: string) => {
    const search = new URLSearchParams();
    search.set("graph_id", graphId);
    search.set("node_id", nodeId);
    return request<DatasetNodeDetail>(`/api/projects/${projectId}/datasets/${datasetId}/analysis/node?${search.toString()}`);
  },
  datasetPreview: (projectId: string, datasetId: string, table: DatasetPreviewTable, limit = 20, offset = 0) =>
    request<TabularPreview>(`/api/projects/${projectId}/datasets/${datasetId}/preview/${table}?limit=${limit}&offset=${offset}`),
  datasetExport: (projectId: string, datasetId: string, table: DatasetPreviewTable) =>
    requestDownload(`/api/projects/${projectId}/datasets/${datasetId}/export/${table}`),
  artifactDeletionPlan: (projectId: string, artifactKind: ArtifactKind, artifactId: string) =>
    request<ArtifactDeletionPlan>(`/api/projects/${projectId}/${artifactKind}s/${artifactId}/delete-plan`),
  deleteArtifact: (projectId: string, artifactKind: ArtifactKind, artifactId: string, cascade: boolean) => {
    const suffix = cascade ? "?cascade=true" : "";
    return request<ArtifactDeletionSummary>(`/api/projects/${projectId}/${artifactKind}s/${artifactId}${suffix}`, { method: "DELETE" });
  },
  createFeature: (projectId: string, payload: FeatureCreatePayload) =>
    request<FeatureManifest>(`/api/projects/${projectId}/features`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  runFeature: (projectId: string, featureId: string) =>
    request<JobManifest>(`/api/projects/${projectId}/features/${featureId}/run`, { method: "POST" }),
  runFeatureBatch: (projectId: string, featureIds: string[]) =>
    request<JobManifest>(`/api/projects/${projectId}/features/run-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature_ids: featureIds })
    }),
  featurePreview: (projectId: string, featureId: string, limit = 20, offset = 0) =>
    request<TabularPreview>(`/api/projects/${projectId}/features/${featureId}/preview?limit=${limit}&offset=${offset}`),
  featureAnalysis: (projectId: string, featureId: string, params: { max_fit_rows?: number; max_points?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.max_fit_rows != null) search.set("max_fit_rows", String(params.max_fit_rows));
    if (params.max_points != null) search.set("max_points", String(params.max_points));
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<FeatureAnalysis>(`/api/projects/${projectId}/features/${featureId}/analysis${suffix}`);
  },
  featureGraphSearch: (projectId: string, featureId: string, query: string, limit = 25) => {
    const search = new URLSearchParams();
    search.set("query", query);
    search.set("limit", String(limit));
    return request<FeatureGraphSearchResponse>(`/api/projects/${projectId}/features/${featureId}/analysis/search?${search.toString()}`);
  },
  featureGraphDetail: (projectId: string, featureId: string, graphId: string) => {
    const search = new URLSearchParams();
    search.set("graph_id", graphId);
    return request<FeatureGraphDetail>(`/api/projects/${projectId}/features/${featureId}/analysis/graph?${search.toString()}`);
  },
  createEmbedding: (projectId: string, payload: EmbeddingCreatePayload) =>
    request<EmbeddingManifest>(`/api/projects/${projectId}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  runEmbedding: (projectId: string, embeddingId: string) =>
    request<JobManifest>(`/api/projects/${projectId}/embeddings/${embeddingId}/run`, { method: "POST" }),
  runEmbeddingBatch: (projectId: string, embeddingIds: string[]) =>
    request<JobManifest>(`/api/projects/${projectId}/embeddings/run-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding_ids: embeddingIds })
    }),
  embeddingPreview: (projectId: string, embeddingId: string, limit = 20, offset = 0) =>
    request<TabularPreview>(`/api/projects/${projectId}/embeddings/${embeddingId}/preview?limit=${limit}&offset=${offset}`),
  embeddingAnalysis: (projectId: string, embeddingId: string, params: { max_fit_rows?: number; max_points?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.max_fit_rows != null) search.set("max_fit_rows", String(params.max_fit_rows));
    if (params.max_points != null) search.set("max_points", String(params.max_points));
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<EmbeddingAnalysis>(`/api/projects/${projectId}/embeddings/${embeddingId}/analysis${suffix}`);
  },
  embeddingGraphSearch: (projectId: string, embeddingId: string, query: string, limit = 25) => {
    const search = new URLSearchParams();
    search.set("query", query);
    search.set("limit", String(limit));
    return request<EmbeddingGraphSearchResponse>(`/api/projects/${projectId}/embeddings/${embeddingId}/analysis/search?${search.toString()}`);
  },
  embeddingGraphDetail: (projectId: string, embeddingId: string, graphId: string) => {
    const search = new URLSearchParams();
    search.set("graph_id", graphId);
    return request<EmbeddingGraphDetail>(`/api/projects/${projectId}/embeddings/${embeddingId}/analysis/graph?${search.toString()}`);
  },
  createModel: (projectId: string, payload: ModelCreatePayload) =>
    request<ModelManifest>(`/api/projects/${projectId}/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  runModel: (projectId: string, modelId: string) =>
    request<JobManifest>(`/api/projects/${projectId}/models/${modelId}/run`, { method: "POST" }),
  runModelBatch: (projectId: string, modelIds: string[]) =>
    request<JobManifest>(`/api/projects/${projectId}/models/run-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_ids: modelIds })
    }),
  modelPreview: (projectId: string, modelId: string) =>
    request<ModelPreview>(`/api/projects/${projectId}/models/${modelId}/preview`),
  modelAnalysis: (projectId: string, modelId: string) =>
    request<ModelAnalysis>(`/api/projects/${projectId}/models/${modelId}/analysis`),
  createProject: (payload: { name: string; description: string }) =>
    request<ProjectManifest>("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  deleteProject: (projectId: string) =>
    request<ProjectDeletionSummary>(`/api/projects/${projectId}`, {
      method: "DELETE"
    })
};
