import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";

export function useWorkspace() {
  return useQuery({ queryKey: ["workspace"], queryFn: api.workspace });
}

export function useMcpSettings() {
  return useQuery({ queryKey: ["mcp-settings"], queryFn: api.mcpSettings });
}

export function useMcpActivity() {
  return useQuery({ queryKey: ["mcp-activity"], queryFn: () => api.mcpActivity(50), refetchInterval: 2_000 });
}

export function useMcpApprovals() {
  return useQuery({ queryKey: ["mcp-approvals"], queryFn: api.mcpApprovals, refetchInterval: 2_000 });
}

export function useMcpUiState() {
  return useQuery({ queryKey: ["mcp-ui-state"], queryFn: api.mcpUiState, refetchInterval: 1_500 });
}

export function useProjects() {
  return useQuery({ queryKey: ["projects"], queryFn: api.projects, refetchInterval: 60_000 });
}

export function useDatasetLibrary() {
  return useQuery({ queryKey: ["dataset-library"], queryFn: api.datasetLibrary });
}

export function useFeatureLibrary() {
  return useQuery({ queryKey: ["feature-library"], queryFn: api.featureLibrary });
}

export function useEmbeddingLibrary() {
  return useQuery({ queryKey: ["embedding-library"], queryFn: api.embeddingLibrary });
}

export function useModelLibrary() {
  return useQuery({ queryKey: ["model-library"], queryFn: api.modelLibrary });
}

export function useProjectDatasets(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "datasets"],
    queryFn: () => api.projectDatasets(projectId),
    enabled: Boolean(projectId),
    refetchInterval: 5_000
  });
}

export function useProjectFeatures(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "features"],
    queryFn: () => api.projectFeatures(projectId),
    enabled: Boolean(projectId),
    refetchInterval: 5_000
  });
}

export function useProjectEmbeddings(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "embeddings"],
    queryFn: () => api.projectEmbeddings(projectId),
    enabled: Boolean(projectId),
    refetchInterval: 5_000
  });
}

export function useProjectModels(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "models"],
    queryFn: () => api.projectModels(projectId),
    enabled: Boolean(projectId),
    refetchInterval: 5_000
  });
}

export function useProjectJobs(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "jobs"],
    queryFn: () => api.projectJobs(projectId),
    enabled: Boolean(projectId),
    refetchInterval: 2_000
  });
}

export function useInvalidateWorkspace() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["workspace"] });
    client.invalidateQueries({ queryKey: ["mcp-settings"] });
    client.invalidateQueries({ queryKey: ["projects"] });
    client.invalidateQueries({ queryKey: ["dataset-library"] });
    client.invalidateQueries({ queryKey: ["feature-library"] });
    client.invalidateQueries({ queryKey: ["embedding-library"] });
    client.invalidateQueries({ queryKey: ["model-library"] });
  };
}
