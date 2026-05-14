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
  source_type: "neext_csv_bundle";
  source_catalog_id: string;
  source_name: string;
  source: string;
  source_domain: string;
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

export interface DatasetCatalogEntry {
  id: string;
  name: string;
  description: string;
  source: string;
  domain: string;
  source_type: "neext_csv_bundle";
  graph_shape: "graph_collection";
  graph_count: number;
  node_count: number;
  edge_count: number;
  has_graph_labels: boolean;
  has_node_features: boolean;
  has_edge_features: boolean;
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

export interface DatasetCreatePayload {
  catalog_id: string;
  params: {
    graph_type: "networkx" | "igraph";
    filter_largest_component: boolean;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(String(detail.detail || response.statusText));
  }
  return response.json() as Promise<T>;
}

export const api = {
  workspace: () => request<WorkspaceInfo>("/api/workspace"),
  projects: () => request<ProjectManifest[]>("/api/projects"),
  project: (projectId: string) => request<ProjectManifest>(`/api/projects/${projectId}`),
  datasetLibrary: () => request<DatasetCatalogEntry[]>("/api/dataset-library"),
  featureLibrary: () => request<FeatureCatalogEntry[]>("/api/feature-library"),
  embeddingLibrary: () => request<EmbeddingCatalogEntry[]>("/api/embedding-library"),
  projectDatasets: (projectId: string) => request<DatasetManifest[]>(`/api/projects/${projectId}/datasets`),
  projectDataset: (projectId: string, datasetId: string) =>
    request<DatasetManifest>(`/api/projects/${projectId}/datasets/${datasetId}`),
  projectFeatures: (projectId: string) => request<FeatureManifest[]>(`/api/projects/${projectId}/features`),
  projectFeature: (projectId: string, featureId: string) =>
    request<FeatureManifest>(`/api/projects/${projectId}/features/${featureId}`),
  projectEmbeddings: (projectId: string) => request<EmbeddingManifest[]>(`/api/projects/${projectId}/embeddings`),
  projectEmbedding: (projectId: string, embeddingId: string) =>
    request<EmbeddingManifest>(`/api/projects/${projectId}/embeddings/${embeddingId}`),
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
  datasetPreview: (projectId: string, datasetId: string, table: "nodes" | "edges" | "mapping", limit = 20, offset = 0) =>
    request<TabularPreview>(`/api/projects/${projectId}/datasets/${datasetId}/preview/${table}?limit=${limit}&offset=${offset}`),
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
