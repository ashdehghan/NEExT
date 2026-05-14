import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";

export function useWorkspace() {
  return useQuery({ queryKey: ["workspace"], queryFn: api.workspace });
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
    client.invalidateQueries({ queryKey: ["projects"] });
    client.invalidateQueries({ queryKey: ["dataset-library"] });
    client.invalidateQueries({ queryKey: ["feature-library"] });
  };
}
