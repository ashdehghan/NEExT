import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useDatasetLibrary,
  useEmbeddingLibrary,
  useFeatureLibrary,
  useProjectDatasets,
  useProjectEmbeddings,
  useProjectFeatures,
  useProjectJobs,
  useWorkspace,
  useProjects
} from "./hooks/useWorkspace";
import type { MainTab } from "./types";
import { titleCase } from "./types";

import { DesktopShell } from "./components/shell/DesktopShell";
import { TopTabs } from "./components/shell/TopTabs";
import { Ribbon, type RibbonCommand } from "./components/shell/Ribbon";
import { StatusBar } from "./components/shell/StatusBar";

import { SelectionPanel } from "./components/panels/SelectionPanel";
import { Inspector } from "./components/panels/Inspector";
import { JobsPanel } from "./components/panels/JobsPanel";
import { CommandWindow } from "./components/panels/CommandWindow";
import { CreateProjectView, ProjectsView } from "./pages/home/ProjectsPage";
import { ConfigureDatasetView, DatasetLibraryView, DatasetPreviewView, ProjectDatasetsView } from "./pages/datasets/DatasetsPage";
import { ConfigureFeatureView, FeatureLibraryView, FeaturePreviewView, ProjectFeaturesView } from "./pages/features/FeaturesPage";
import { ConfigureEmbeddingView, EmbeddingLibraryView, EmbeddingPreviewView, ProjectEmbeddingsView } from "./pages/embeddings/EmbeddingsPage";

interface Route {
  topTab: MainTab;
  command: RibbonCommand;
}

const DEFAULT_COMMANDS: Record<MainTab, RibbonCommand> = {
  home: "projects",
  datasets: "datasets",
  features: "features",
  embeddings: "embeddings",
  models: "models"
};

const HOME_TITLES: Record<string, string> = {
  import: "Import",
  create: "Create",
  projects: "Projects",
  settings: "Settings",
  help: "Help"
};

function viewTitle(route: Route): string {
  if (route.topTab === "home") return HOME_TITLES[String(route.command)] || titleCase(String(route.command));
  return titleCase(String(route.command));
}

function TitleOnlyView({ title }: { title: string }) {
  return <h1 className="title-only">{title}</h1>;
}

export default function App() {
  const queryClient = useQueryClient();
  const [route, setRoute] = useState<Route>({ topTab: "home", command: DEFAULT_COMMANDS.home });
  const [activeProjectId, setActiveProjectId] = useState("");
  const [hasAutoSelectedProject, setHasAutoSelectedProject] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedCatalogId, setSelectedCatalogId] = useState("");
  const [selectedFeatureId, setSelectedFeatureId] = useState("");
  const [selectedFeatureCatalogId, setSelectedFeatureCatalogId] = useState("");
  const [selectedEmbeddingId, setSelectedEmbeddingId] = useState("");
  const [selectedEmbeddingCatalogId, setSelectedEmbeddingCatalogId] = useState("");
  const [configureDatasetCatalogId, setConfigureDatasetCatalogId] = useState("");
  const [configureFeatureCatalogId, setConfigureFeatureCatalogId] = useState("");
  const [configureEmbeddingCatalogId, setConfigureEmbeddingCatalogId] = useState("");
  const [previewDatasetId, setPreviewDatasetId] = useState("");
  const [previewFeatureId, setPreviewFeatureId] = useState("");
  const [previewEmbeddingId, setPreviewEmbeddingId] = useState("");

  const workspaceQuery = useWorkspace();
  const projectsQuery = useProjects();
  const datasetLibraryQuery = useDatasetLibrary();
  const featureLibraryQuery = useFeatureLibrary();
  const embeddingLibraryQuery = useEmbeddingLibrary();
  const projectDatasetsQuery = useProjectDatasets(activeProjectId);
  const projectFeaturesQuery = useProjectFeatures(activeProjectId);
  const projectEmbeddingsQuery = useProjectEmbeddings(activeProjectId);
  const projectJobsQuery = useProjectJobs(activeProjectId);

  useEffect(() => {
    if (!hasAutoSelectedProject && !activeProjectId && projectsQuery.data?.length) {
      setActiveProjectId(projectsQuery.data[0].id);
      setHasAutoSelectedProject(true);
    }
  }, [activeProjectId, hasAutoSelectedProject, projectsQuery.data]);

  const project = useMemo(
    () => projectsQuery.data?.find((item) => item.id === activeProjectId),
    [activeProjectId, projectsQuery.data]
  );
  const datasets = projectDatasetsQuery.data || [];
  const features = projectFeaturesQuery.data || [];
  const embeddings = projectEmbeddingsQuery.data || [];
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId]
  );
  const selectedFeature = useMemo(
    () => features.find((feature) => feature.id === selectedFeatureId),
    [features, selectedFeatureId]
  );
  const selectedEmbedding = useMemo(
    () => embeddings.find((embedding) => embedding.id === selectedEmbeddingId),
    [embeddings, selectedEmbeddingId]
  );
  const selectedCatalogEntry = useMemo(
    () => datasetLibraryQuery.data?.find((entry) => entry.id === selectedCatalogId),
    [datasetLibraryQuery.data, selectedCatalogId]
  );
  const selectedFeatureCatalogEntry = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === selectedFeatureCatalogId),
    [featureLibraryQuery.data, selectedFeatureCatalogId]
  );
  const selectedEmbeddingCatalogEntry = useMemo(
    () => embeddingLibraryQuery.data?.find((entry) => entry.id === selectedEmbeddingCatalogId),
    [embeddingLibraryQuery.data, selectedEmbeddingCatalogId]
  );
  const configuredFeatureCatalogEntry = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === configureFeatureCatalogId),
    [configureFeatureCatalogId, featureLibraryQuery.data]
  );
  const configuredEmbeddingCatalogEntry = useMemo(
    () => embeddingLibraryQuery.data?.find((entry) => entry.id === configureEmbeddingCatalogId),
    [configureEmbeddingCatalogId, embeddingLibraryQuery.data]
  );
  const configuredDatasetCatalogEntry = useMemo(
    () => datasetLibraryQuery.data?.find((entry) => entry.id === configureDatasetCatalogId),
    [configureDatasetCatalogId, datasetLibraryQuery.data]
  );
  const previewDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === previewDatasetId),
    [datasets, previewDatasetId]
  );
  const previewFeature = useMemo(
    () => features.find((feature) => feature.id === previewFeatureId),
    [features, previewFeatureId]
  );
  const previewEmbedding = useMemo(
    () => embeddings.find((embedding) => embedding.id === previewEmbeddingId),
    [embeddings, previewEmbeddingId]
  );
  const selectedFeatureMethod = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === selectedFeature?.source_feature_id),
    [featureLibraryQuery.data, selectedFeature]
  );
  const selectedFeatureDataset = useMemo(() => {
    const datasetInput = selectedFeature?.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset");
    return datasetInput ? datasets.find((dataset) => dataset.id === datasetInput.artifact_id) : undefined;
  }, [datasets, selectedFeature]);
  const selectedEmbeddingAlgorithm = useMemo(
    () => embeddingLibraryQuery.data?.find((entry) => entry.id === selectedEmbedding?.source_embedding_id),
    [embeddingLibraryQuery.data, selectedEmbedding]
  );
  const selectedEmbeddingFeatures = useMemo(() => {
    if (!selectedEmbedding) return [];
    const featureIds = selectedEmbedding.inputs
      .filter((input) => input.role === "source_feature" && input.artifact_kind === "feature")
      .map((input) => input.artifact_id);
    return featureIds.map((featureId) => features.find((feature) => feature.id === featureId)).filter(Boolean) as typeof features;
  }, [features, selectedEmbedding]);
  const selectedEmbeddingDataset = useMemo(() => {
    const firstFeature = selectedEmbeddingFeatures[0];
    const datasetInput = firstFeature?.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset");
    return datasetInput ? datasets.find((dataset) => dataset.id === datasetInput.artifact_id) : undefined;
  }, [datasets, selectedEmbeddingFeatures]);
  const importedCatalogIds = useMemo(() => new Set(datasets.map((dataset) => dataset.source_catalog_id)), [datasets]);
  const jobs = projectJobsQuery.data || [];

  useEffect(() => {
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, [activeProjectId]);

  useEffect(() => {
    if (selectedDatasetId && datasets.length && !selectedDataset) {
      setSelectedDatasetId("");
    }
  }, [datasets, selectedDataset, selectedDatasetId]);

  useEffect(() => {
    if (selectedCatalogId && datasetLibraryQuery.data?.length && !selectedCatalogEntry) {
      setSelectedCatalogId("");
    }
  }, [datasetLibraryQuery.data, selectedCatalogEntry, selectedCatalogId]);

  useEffect(() => {
    if (selectedFeatureId && features.length && !selectedFeature) {
      setSelectedFeatureId("");
    }
  }, [features, selectedFeature, selectedFeatureId]);

  useEffect(() => {
    if (selectedFeatureCatalogId && featureLibraryQuery.data?.length && !selectedFeatureCatalogEntry) {
      setSelectedFeatureCatalogId("");
    }
  }, [featureLibraryQuery.data, selectedFeatureCatalogEntry, selectedFeatureCatalogId]);

  useEffect(() => {
    if (selectedEmbeddingId && embeddings.length && !selectedEmbedding) {
      setSelectedEmbeddingId("");
    }
  }, [embeddings, selectedEmbedding, selectedEmbeddingId]);

  useEffect(() => {
    if (selectedEmbeddingCatalogId && embeddingLibraryQuery.data?.length && !selectedEmbeddingCatalogEntry) {
      setSelectedEmbeddingCatalogId("");
    }
  }, [embeddingLibraryQuery.data, selectedEmbeddingCatalogEntry, selectedEmbeddingCatalogId]);

  const setActiveTab = useCallback((tab: MainTab) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: tab, command: DEFAULT_COMMANDS[tab] });
  }, []);

  const setActiveCommand = useCallback((command: RibbonCommand) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute((current) => ({ ...current, command }));
  }, []);

  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["workspace"] });
    queryClient.invalidateQueries({ queryKey: ["projects"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-library"] });
    queryClient.invalidateQueries({ queryKey: ["feature-library"] });
    queryClient.invalidateQueries({ queryKey: ["embedding-library"] });
    if (activeProjectId) {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  }, [activeProjectId, queryClient]);

  const handleProjectCreated = useCallback((projectId: string) => {
    setActiveProjectId(projectId);
    setHasAutoSelectedProject(true);
    setRoute({ topTab: "home", command: "projects" });
  }, []);

  const handleSelectCatalog = useCallback((catalogId: string) => {
    setSelectedCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleConfigureDatasetCatalog = useCallback((catalogId: string) => {
    setSelectedCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId(catalogId);
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "datasets", command: "library" });
  }, []);

  const handleSelectDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleSelectFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleConfigureFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId(catalogId);
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "features", command: "library" });
  }, []);

  const handleSelectFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleDatasetCreated = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handleFeatureCreated = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "features", command: "features" });
  }, []);

  const handlePreviewDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setPreviewDatasetId(datasetId);
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handlePreviewFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setPreviewFeatureId(featureId);
    setPreviewDatasetId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "features", command: "features" });
  }, []);

  const handleSelectEmbeddingCatalog = useCallback((catalogId: string) => {
    setSelectedEmbeddingCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleConfigureEmbeddingCatalog = useCallback((catalogId: string) => {
    setSelectedEmbeddingCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId(catalogId);
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "embeddings", command: "library" });
  }, []);

  const handleSelectEmbedding = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
  }, []);

  const handleEmbeddingCreated = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setPreviewEmbeddingId("");
    setRoute({ topTab: "embeddings", command: "embeddings" });
  }, []);

  const handlePreviewEmbedding = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setPreviewEmbeddingId(embeddingId);
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: "embeddings", command: "embeddings" });
  }, []);

  function renderCenterView() {
    const title = viewTitle(route);
    if (route.topTab === "home" && route.command === "create") {
      return <CreateProjectView onCreated={handleProjectCreated} />;
    }
    if (route.topTab === "home" && route.command === "projects") {
      return (
        <ProjectsView
          workspacePath={workspaceQuery.data?.path || ""}
          projects={projectsQuery.data || []}
          activeProjectId={activeProjectId}
          onSelectProject={setActiveProjectId}
        />
      );
    }
    if (route.topTab === "datasets" && route.command === "library" && configureDatasetCatalogId) {
      return (
        <ConfigureDatasetView
          activeProjectId={activeProjectId}
          entry={configuredDatasetCatalogEntry}
          onCreated={handleDatasetCreated}
        />
      );
    }
    if (route.topTab === "datasets" && route.command === "library") {
      return (
        <DatasetLibraryView
          activeProjectId={activeProjectId}
          catalog={datasetLibraryQuery.data || []}
          datasets={datasets}
          loading={datasetLibraryQuery.isLoading || projectDatasetsQuery.isLoading}
          selectedCatalogId={selectedCatalogId}
          onSelectCatalog={handleSelectCatalog}
          onConfigure={handleConfigureDatasetCatalog}
        />
      );
    }
    if (route.topTab === "datasets" && route.command === "datasets" && previewDatasetId) {
      return <DatasetPreviewView activeProjectId={activeProjectId} dataset={previewDataset} />;
    }
    if (route.topTab === "datasets" && route.command === "datasets") {
      return (
        <ProjectDatasetsView
          activeProjectId={activeProjectId}
          datasets={datasets}
          loading={projectDatasetsQuery.isLoading}
          selectedDatasetId={selectedDatasetId}
          onSelectDataset={handleSelectDataset}
          onPreviewDataset={handlePreviewDataset}
        />
      );
    }
    if (route.topTab === "features" && route.command === "library" && configureFeatureCatalogId) {
      return (
        <ConfigureFeatureView
          activeProjectId={activeProjectId}
          feature={configuredFeatureCatalogEntry}
          datasets={datasets}
          loading={projectDatasetsQuery.isLoading}
          onCreated={handleFeatureCreated}
        />
      );
    }
    if (route.topTab === "features" && route.command === "library") {
      return (
        <FeatureLibraryView
          catalog={featureLibraryQuery.data || []}
          loading={featureLibraryQuery.isLoading}
          selectedCatalogId={selectedFeatureCatalogId}
          onSelectCatalog={handleSelectFeatureCatalog}
          onConfigure={handleConfigureFeatureCatalog}
        />
      );
    }
    if (route.topTab === "features" && route.command === "features" && previewFeatureId) {
      return <FeaturePreviewView activeProjectId={activeProjectId} feature={previewFeature} />;
    }
    if (route.topTab === "features" && route.command === "features") {
      return (
        <ProjectFeaturesView
          activeProjectId={activeProjectId}
          features={features}
          datasets={datasets}
          catalog={featureLibraryQuery.data || []}
          loading={projectFeaturesQuery.isLoading}
          selectedFeatureId={selectedFeatureId}
          onSelectFeature={handleSelectFeature}
          onPreviewFeature={handlePreviewFeature}
        />
      );
    }
    if (route.topTab === "embeddings" && route.command === "library" && configureEmbeddingCatalogId) {
      return (
        <ConfigureEmbeddingView
          activeProjectId={activeProjectId}
          embedding={configuredEmbeddingCatalogEntry}
          features={features}
          datasets={datasets}
          loading={projectFeaturesQuery.isLoading || projectDatasetsQuery.isLoading}
          onCreated={handleEmbeddingCreated}
        />
      );
    }
    if (route.topTab === "embeddings" && route.command === "library") {
      return (
        <EmbeddingLibraryView
          activeProjectId={activeProjectId}
          catalog={embeddingLibraryQuery.data || []}
          loading={embeddingLibraryQuery.isLoading}
          selectedCatalogId={selectedEmbeddingCatalogId}
          onSelectCatalog={handleSelectEmbeddingCatalog}
          onConfigure={handleConfigureEmbeddingCatalog}
        />
      );
    }
    if (route.topTab === "embeddings" && route.command === "embeddings" && previewEmbeddingId) {
      return <EmbeddingPreviewView activeProjectId={activeProjectId} embedding={previewEmbedding} />;
    }
    if (route.topTab === "embeddings" && route.command === "embeddings") {
      return (
        <ProjectEmbeddingsView
          activeProjectId={activeProjectId}
          embeddings={embeddings}
          features={features}
          datasets={datasets}
          catalog={embeddingLibraryQuery.data || []}
          loading={projectEmbeddingsQuery.isLoading}
          selectedEmbeddingId={selectedEmbeddingId}
          onSelectEmbedding={handleSelectEmbedding}
          onPreviewEmbedding={handlePreviewEmbedding}
        />
      );
    }
    return <TitleOnlyView title={title} />;
  }

  return (
    <DesktopShell
      topTabs={
        <TopTabs activeTab={route.topTab} onSelect={setActiveTab} onRefresh={refreshAll} />
      }
      ribbon={
        <Ribbon
          activeTab={route.topTab}
          activeCommand={String(route.command)}
          onCommand={setActiveCommand}
        />
      }
      left={
        <SelectionPanel
          project={project}
          datasets={datasets}
          features={features}
          embeddings={embeddings}
          selectedDatasetId={selectedDatasetId}
          selectedFeatureId={selectedFeatureId}
          selectedEmbeddingId={selectedEmbeddingId}
          onSelectDataset={handleSelectDataset}
          onSelectFeature={handleSelectFeature}
          onSelectEmbedding={handleSelectEmbedding}
        />
      }
      center={
        <>
          <section className="document">
            <div className="doc-body">{renderCenterView()}</div>
          </section>
          <CommandWindow jobs={jobs} />
        </>
      }
      right={
        <>
          <Inspector
            project={project}
            dataset={selectedDataset}
            catalogEntry={selectedCatalogEntry}
            catalogImportStatus={
              selectedCatalogEntry
                ? activeProjectId
                  ? importedCatalogIds.has(selectedCatalogEntry.id)
                    ? "Configured"
                    : "Not configured"
                  : "No active project"
                : undefined
            }
            feature={selectedFeature}
            featureDataset={selectedFeatureDataset}
            featureCatalogEntry={selectedFeatureMethod}
            selectedFeatureCatalogEntry={selectedFeatureCatalogEntry}
            embedding={selectedEmbedding}
            embeddingDataset={selectedEmbeddingDataset}
            embeddingFeatures={selectedEmbeddingFeatures}
            embeddingCatalogEntry={selectedEmbeddingAlgorithm}
            selectedEmbeddingCatalogEntry={selectedEmbeddingCatalogEntry}
          />
          <JobsPanel jobs={jobs} />
        </>
      }
      statusBar={
        <StatusBar
          workspacePath={workspaceQuery.data?.path}
          projectName={project?.name}
          version={workspaceQuery.data?.version}
        />
      }
    />
  );
}
