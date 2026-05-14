import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useDatasetLibrary,
  useFeatureLibrary,
  useProjectDatasets,
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
  const [configureDatasetCatalogId, setConfigureDatasetCatalogId] = useState("");
  const [configureFeatureCatalogId, setConfigureFeatureCatalogId] = useState("");
  const [previewDatasetId, setPreviewDatasetId] = useState("");
  const [previewFeatureId, setPreviewFeatureId] = useState("");

  const workspaceQuery = useWorkspace();
  const projectsQuery = useProjects();
  const datasetLibraryQuery = useDatasetLibrary();
  const featureLibraryQuery = useFeatureLibrary();
  const projectDatasetsQuery = useProjectDatasets(activeProjectId);
  const projectFeaturesQuery = useProjectFeatures(activeProjectId);
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
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId]
  );
  const selectedFeature = useMemo(
    () => features.find((feature) => feature.id === selectedFeatureId),
    [features, selectedFeatureId]
  );
  const selectedCatalogEntry = useMemo(
    () => datasetLibraryQuery.data?.find((entry) => entry.id === selectedCatalogId),
    [datasetLibraryQuery.data, selectedCatalogId]
  );
  const selectedFeatureCatalogEntry = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === selectedFeatureCatalogId),
    [featureLibraryQuery.data, selectedFeatureCatalogId]
  );
  const configuredFeatureCatalogEntry = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === configureFeatureCatalogId),
    [configureFeatureCatalogId, featureLibraryQuery.data]
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
  const selectedFeatureMethod = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === selectedFeature?.source_feature_id),
    [featureLibraryQuery.data, selectedFeature]
  );
  const selectedFeatureDataset = useMemo(() => {
    const datasetInput = selectedFeature?.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset");
    return datasetInput ? datasets.find((dataset) => dataset.id === datasetInput.artifact_id) : undefined;
  }, [datasets, selectedFeature]);
  const importedCatalogIds = useMemo(() => new Set(datasets.map((dataset) => dataset.source_catalog_id)), [datasets]);
  const jobs = projectJobsQuery.data || [];

  useEffect(() => {
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
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

  const setActiveTab = useCallback((tab: MainTab) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: tab, command: DEFAULT_COMMANDS[tab] });
  }, []);

  const setActiveCommand = useCallback((command: RibbonCommand) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute((current) => ({ ...current, command }));
  }, []);

  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["workspace"] });
    queryClient.invalidateQueries({ queryKey: ["projects"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-library"] });
    queryClient.invalidateQueries({ queryKey: ["feature-library"] });
    if (activeProjectId) {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
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
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
  }, []);

  const handleConfigureDatasetCatalog = useCallback((catalogId: string) => {
    setSelectedCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId(catalogId);
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: "datasets", command: "library" });
  }, []);

  const handleSelectDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
  }, []);

  const handleSelectFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
  }, []);

  const handleConfigureFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId(catalogId);
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: "features", command: "library" });
  }, []);

  const handleSelectFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
  }, []);

  const handleDatasetCreated = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handleFeatureCreated = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setPreviewDatasetId("");
    setPreviewFeatureId("");
    setRoute({ topTab: "features", command: "features" });
  }, []);

  const handlePreviewDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setPreviewDatasetId(datasetId);
    setPreviewFeatureId("");
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handlePreviewFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setPreviewFeatureId(featureId);
    setPreviewDatasetId("");
    setRoute({ topTab: "features", command: "features" });
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
          selectedDatasetId={selectedDatasetId}
          selectedFeatureId={selectedFeatureId}
          onSelectDataset={handleSelectDataset}
          onSelectFeature={handleSelectFeature}
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
