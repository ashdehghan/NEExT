import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useDatasetLibrary,
  useEmbeddingLibrary,
  useFeatureLibrary,
  useModelLibrary,
  useProjectDatasets,
  useProjectEmbeddings,
  useProjectFeatures,
  useProjectModels,
  useProjectJobs,
  useWorkspace,
  useProjects
} from "./hooks/useWorkspace";
import type { DatasetGraphSummary, DatasetManifest, EmbeddingManifest, FeatureManifest, ModelManifest } from "./api";
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
import { SettingsView } from "./pages/home/SettingsPage";
import { ConfigureDatasetView, DatasetExploreView, DatasetLibraryView, ProjectDatasetsView } from "./pages/datasets/DatasetsPage";
import { ConfigureFeatureView, FeatureExploreView, FeatureLibraryView, ProjectFeaturesView } from "./pages/features/FeaturesPage";
import { ConfigureEmbeddingView, EmbeddingExploreView, EmbeddingLibraryView, ProjectEmbeddingsView } from "./pages/embeddings/EmbeddingsPage";
import { ConfigureModelView, ModelExploreView, ModelLibraryView, ProjectModelsView } from "./pages/models/ModelsPage";

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

function featureDatasetId(feature: FeatureManifest): string {
  return feature.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset")?.artifact_id || "";
}

function embeddingFeatureIds(embedding: EmbeddingManifest): string[] {
  return embedding.inputs
    .filter((input) => input.role === "source_feature" && input.artifact_kind === "feature")
    .map((input) => input.artifact_id);
}

function modelEmbeddingIds(model: ModelManifest): string[] {
  return model.inputs
    .filter((input) => input.role === "source_embedding" && input.artifact_kind === "embedding")
    .map((input) => input.artifact_id);
}

interface SelectionLineage {
  contextLabel: string;
  activeDatasetId: string;
  datasets: DatasetManifest[];
  features: FeatureManifest[];
  embeddings: EmbeddingManifest[];
  models: ModelManifest[];
  relatedEmbeddingIds: string[];
  relatedModelIds: string[];
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
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedModelCatalogId, setSelectedModelCatalogId] = useState("");
  const [configureDatasetCatalogId, setConfigureDatasetCatalogId] = useState("");
  const [configureFeatureCatalogId, setConfigureFeatureCatalogId] = useState("");
  const [configureEmbeddingCatalogId, setConfigureEmbeddingCatalogId] = useState("");
  const [configureModelCatalogId, setConfigureModelCatalogId] = useState("");
  const [exploreDatasetId, setExploreDatasetId] = useState("");
  const [exploreGraphId, setExploreGraphId] = useState("");
  const [exploreGraphSummary, setExploreGraphSummary] = useState<DatasetGraphSummary | null>(null);
  const [exploreNodeId, setExploreNodeId] = useState("");
  const [exploreNodeVisible, setExploreNodeVisible] = useState<boolean | null>(null);
  const [exploreFeatureId, setExploreFeatureId] = useState("");
  const [exploreFeatureGraphId, setExploreFeatureGraphId] = useState("");
  const [exploreFeatureGraphVisible, setExploreFeatureGraphVisible] = useState<boolean | null>(null);
  const [exploreEmbeddingId, setExploreEmbeddingId] = useState("");
  const [exploreEmbeddingGraphId, setExploreEmbeddingGraphId] = useState("");
  const [exploreEmbeddingGraphVisible, setExploreEmbeddingGraphVisible] = useState<boolean | null>(null);
  const [exploreModelId, setExploreModelId] = useState("");
  const [exploreModelIteration, setExploreModelIteration] = useState<number | null>(null);

  const workspaceQuery = useWorkspace();
  const projectsQuery = useProjects();
  const datasetLibraryQuery = useDatasetLibrary();
  const featureLibraryQuery = useFeatureLibrary();
  const embeddingLibraryQuery = useEmbeddingLibrary();
  const modelLibraryQuery = useModelLibrary();
  const projectDatasetsQuery = useProjectDatasets(activeProjectId);
  const projectFeaturesQuery = useProjectFeatures(activeProjectId);
  const projectEmbeddingsQuery = useProjectEmbeddings(activeProjectId);
  const projectModelsQuery = useProjectModels(activeProjectId);
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
  const models = projectModelsQuery.data || [];
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
  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId]
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
  const selectedModelCatalogEntry = useMemo(
    () => modelLibraryQuery.data?.find((entry) => entry.id === selectedModelCatalogId),
    [modelLibraryQuery.data, selectedModelCatalogId]
  );
  const configuredFeatureCatalogEntry = useMemo(
    () => featureLibraryQuery.data?.find((entry) => entry.id === configureFeatureCatalogId),
    [configureFeatureCatalogId, featureLibraryQuery.data]
  );
  const configuredEmbeddingCatalogEntry = useMemo(
    () => embeddingLibraryQuery.data?.find((entry) => entry.id === configureEmbeddingCatalogId),
    [configureEmbeddingCatalogId, embeddingLibraryQuery.data]
  );
  const configuredModelCatalogEntry = useMemo(
    () => modelLibraryQuery.data?.find((entry) => entry.id === configureModelCatalogId),
    [configureModelCatalogId, modelLibraryQuery.data]
  );
  const configuredDatasetCatalogEntry = useMemo(
    () => datasetLibraryQuery.data?.find((entry) => entry.id === configureDatasetCatalogId),
    [configureDatasetCatalogId, datasetLibraryQuery.data]
  );
  const exploreModel = useMemo(
    () => models.find((model) => model.id === exploreModelId),
    [exploreModelId, models]
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
  const selectedModelAlgorithm = useMemo(
    () => modelLibraryQuery.data?.find((entry) => entry.id === selectedModel?.source_model_id),
    [modelLibraryQuery.data, selectedModel]
  );
  const selectedModelEmbeddings = useMemo(() => {
    if (!selectedModel) return [];
    const embeddingIds = selectedModel.inputs
      .filter((input) => input.role === "source_embedding" && input.artifact_kind === "embedding")
      .map((input) => input.artifact_id);
    return embeddingIds.map((embeddingId) => embeddings.find((embedding) => embedding.id === embeddingId)).filter(Boolean) as typeof embeddings;
  }, [embeddings, selectedModel]);
  const selectedModelDataset = useMemo(() => {
    const firstEmbedding = selectedModelEmbeddings[0];
    if (!firstEmbedding) return undefined;
    const featureIds = firstEmbedding.inputs
      .filter((input) => input.role === "source_feature" && input.artifact_kind === "feature")
      .map((input) => input.artifact_id);
    const firstFeature = featureIds.map((featureId) => features.find((feature) => feature.id === featureId)).find(Boolean);
    const datasetInput = firstFeature?.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset");
    return datasetInput ? datasets.find((dataset) => dataset.id === datasetInput.artifact_id) : undefined;
  }, [datasets, features, selectedModelEmbeddings]);
  const selectionLineage = useMemo<SelectionLineage>(() => {
    const projectContext = {
      contextLabel: project ? `Context: Project ${project.name}` : "Context: No project",
      activeDatasetId: "",
      datasets,
      features: [],
      embeddings: [],
      models: [],
      relatedEmbeddingIds: [],
      relatedModelIds: []
    };
    const featuresById = new Map(features.map((feature) => [feature.id, feature]));
    const embeddingsById = new Map(embeddings.map((embedding) => [embedding.id, embedding]));
    const embeddingDatasetId = (embedding: EmbeddingManifest) => {
      const sourceFeatureIds = embeddingFeatureIds(embedding);
      if (sourceFeatureIds.length === 0) return "";
      const datasetIds = new Set<string>();
      for (const featureId of sourceFeatureIds) {
        const feature = featuresById.get(featureId);
        const datasetId = feature ? featureDatasetId(feature) : "";
        if (!datasetId) return "";
        datasetIds.add(datasetId);
      }
      return datasetIds.size === 1 ? Array.from(datasetIds)[0] : "";
    };
    const modelDatasetId = (model: ModelManifest) => {
      const sourceEmbeddingIds = modelEmbeddingIds(model);
      if (sourceEmbeddingIds.length === 0) return "";
      const datasetIds = new Set<string>();
      for (const embeddingId of sourceEmbeddingIds) {
        const embedding = embeddingsById.get(embeddingId);
        const datasetId = embedding ? embeddingDatasetId(embedding) : "";
        if (!datasetId) return "";
        datasetIds.add(datasetId);
      }
      return datasetIds.size === 1 ? Array.from(datasetIds)[0] : "";
    };
    const datasetContext = (
      dataset: DatasetManifest,
      contextLabel: string,
      relatedEmbeddingIds: string[] = [],
      relatedModelIds: string[] = []
    ): SelectionLineage => {
      const datasetId = dataset.id;
      const lineageFeatures = features.filter((feature) => featureDatasetId(feature) === datasetId);
      const lineageEmbeddings = embeddings.filter((embedding) => embeddingDatasetId(embedding) === datasetId);
      const lineageModels = models.filter((model) => modelDatasetId(model) === datasetId);
      return {
        contextLabel,
        activeDatasetId: dataset.id,
        datasets,
        features: lineageFeatures,
        embeddings: lineageEmbeddings,
        models: lineageModels,
        relatedEmbeddingIds,
        relatedModelIds
      };
    };

    if (selectedDataset) {
      return datasetContext(selectedDataset, `Context: Dataset ${selectedDataset.name}`);
    }

    if (selectedFeature) {
      const datasetId = featureDatasetId(selectedFeature);
      const lineageDataset = datasets.find((dataset) => dataset.id === datasetId);
      const relatedEmbeddings = lineageDataset
        ? embeddings.filter((embedding) => embeddingDatasetId(embedding) === datasetId && embeddingFeatureIds(embedding).includes(selectedFeature.id))
        : [];
      const relatedEmbeddingIds = relatedEmbeddings.map((embedding) => embedding.id);
      const relatedEmbeddingIdSet = new Set(relatedEmbeddingIds);
      const relatedModelIds = lineageDataset
        ? models
            .filter(
              (model) =>
                modelDatasetId(model) === datasetId && modelEmbeddingIds(model).some((embeddingId) => relatedEmbeddingIdSet.has(embeddingId))
            )
            .map((model) => model.id)
        : [];
      if (lineageDataset) {
        return datasetContext(lineageDataset, `Context: Feature ${selectedFeature.name}`, relatedEmbeddingIds, relatedModelIds);
      }
      return {
        contextLabel: `Context: Feature ${selectedFeature.name}`,
        activeDatasetId: "",
        datasets: [],
        features: [selectedFeature],
        embeddings: [],
        models: [],
        relatedEmbeddingIds: [],
        relatedModelIds: []
      };
    }

    if (selectedEmbedding) {
      const datasetId = embeddingDatasetId(selectedEmbedding);
      const lineageDataset = datasets.find((dataset) => dataset.id === datasetId);
      const relatedModelIds = lineageDataset
        ? models
            .filter((model) => modelDatasetId(model) === datasetId && modelEmbeddingIds(model).includes(selectedEmbedding.id))
            .map((model) => model.id)
        : [];
      if (lineageDataset) {
        return datasetContext(lineageDataset, `Context: Embedding ${selectedEmbedding.name}`, [], relatedModelIds);
      }
      return {
        contextLabel: `Context: Embedding ${selectedEmbedding.name}`,
        activeDatasetId: "",
        datasets: [],
        features: [],
        embeddings: [selectedEmbedding],
        models: [],
        relatedEmbeddingIds: [],
        relatedModelIds: []
      };
    }

    if (selectedModel) {
      const datasetId = modelDatasetId(selectedModel);
      const lineageDataset = datasets.find((dataset) => dataset.id === datasetId);
      if (lineageDataset) {
        return datasetContext(lineageDataset, `Context: Model ${selectedModel.name}`);
      }
      return {
        contextLabel: `Context: Model ${selectedModel.name}`,
        activeDatasetId: "",
        datasets: [],
        features: [],
        embeddings: [],
        models: [selectedModel],
        relatedEmbeddingIds: [],
        relatedModelIds: []
      };
    }

    return projectContext;
  }, [datasets, embeddings, features, models, project, selectedDataset, selectedEmbedding, selectedFeature, selectedModel]);
  const importedCatalogIds = useMemo(() => new Set(datasets.map((dataset) => dataset.source_catalog_id)), [datasets]);
  const jobs = projectJobsQuery.data || [];
  const isProjectSelected = Boolean(
    project &&
      !selectedDatasetId &&
      !selectedCatalogId &&
      !selectedFeatureId &&
      !selectedFeatureCatalogId &&
      !selectedEmbeddingId &&
      !selectedEmbeddingCatalogId &&
      !selectedModelId &&
      !selectedModelCatalogId &&
      !configureDatasetCatalogId &&
      !configureFeatureCatalogId &&
      !configureEmbeddingCatalogId &&
      !configureModelCatalogId &&
      !exploreDatasetId &&
      !exploreGraphId &&
      !exploreNodeId &&
      !exploreFeatureId &&
      !exploreFeatureGraphId &&
      !exploreEmbeddingId &&
      !exploreEmbeddingGraphId &&
      !exploreModelId
  );

  useEffect(() => {
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
  }, [activeProjectId]);

  useEffect(() => {
    if (selectedDatasetId && datasets.length && !selectedDataset) {
      setSelectedDatasetId("");
    }
  }, [datasets, selectedDataset, selectedDatasetId]);

  useEffect(() => {
    if (exploreDatasetId && datasets.length && !datasets.some((dataset) => dataset.id === exploreDatasetId)) {
      setExploreDatasetId("");
      setExploreGraphId("");
      setExploreGraphSummary(null);
      setExploreNodeId("");
      setExploreNodeVisible(null);
    }
  }, [datasets, exploreDatasetId]);

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
    if (exploreFeatureId && features.length && !features.some((feature) => feature.id === exploreFeatureId)) {
      setExploreFeatureId("");
      setExploreFeatureGraphId("");
      setExploreFeatureGraphVisible(null);
    }
  }, [exploreFeatureId, features]);

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
    if (exploreEmbeddingId && embeddings.length && !embeddings.some((embedding) => embedding.id === exploreEmbeddingId)) {
      setExploreEmbeddingId("");
      setExploreEmbeddingGraphId("");
      setExploreEmbeddingGraphVisible(null);
    }
  }, [embeddings, exploreEmbeddingId]);

  useEffect(() => {
    if (selectedEmbeddingCatalogId && embeddingLibraryQuery.data?.length && !selectedEmbeddingCatalogEntry) {
      setSelectedEmbeddingCatalogId("");
    }
  }, [embeddingLibraryQuery.data, selectedEmbeddingCatalogEntry, selectedEmbeddingCatalogId]);

  useEffect(() => {
    if (selectedModelId && models.length && !selectedModel) {
      setSelectedModelId("");
    }
  }, [models, selectedModel, selectedModelId]);

  useEffect(() => {
    if (exploreModelId && models.length && !exploreModel) {
      setExploreModelId("");
      setExploreModelIteration(null);
    }
  }, [exploreModel, exploreModelId, models]);

  useEffect(() => {
    if (selectedModelCatalogId && modelLibraryQuery.data?.length && !selectedModelCatalogEntry) {
      setSelectedModelCatalogId("");
    }
  }, [modelLibraryQuery.data, selectedModelCatalogEntry, selectedModelCatalogId]);

  const setActiveTab = useCallback((tab: MainTab) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: tab, command: DEFAULT_COMMANDS[tab] });
  }, []);

  const setActiveCommand = useCallback((command: RibbonCommand) => {
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId(command === "explore" && route.topTab === "datasets" ? selectedDatasetId : "");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId(command === "explore" && route.topTab === "features" ? selectedFeatureId : "");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId(command === "explore" && route.topTab === "embeddings" ? selectedEmbeddingId : "");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId(command === "explore" && route.topTab === "models" ? selectedModelId : "");
    setExploreModelIteration(null);
    setRoute((current) => ({ ...current, command }));
  }, [route.topTab, selectedDatasetId, selectedEmbeddingId, selectedFeatureId, selectedModelId]);

  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["workspace"] });
    queryClient.invalidateQueries({ queryKey: ["mcp-settings"] });
    queryClient.invalidateQueries({ queryKey: ["projects"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-library"] });
    queryClient.invalidateQueries({ queryKey: ["feature-library"] });
    queryClient.invalidateQueries({ queryKey: ["embedding-library"] });
    queryClient.invalidateQueries({ queryKey: ["model-library"] });
    if (activeProjectId) {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "models"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  }, [activeProjectId, queryClient]);

  const handleProjectCreated = useCallback((projectId: string) => {
    setActiveProjectId(projectId);
    setHasAutoSelectedProject(true);
    setRoute({ topTab: "home", command: "projects" });
  }, []);

  const handleSelectProject = useCallback((projectId: string) => {
    setActiveProjectId(projectId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "home", command: "projects" });
  }, []);

  const handleSelectActiveProject = useCallback(() => {
    handleSelectProject(activeProjectId);
  }, [activeProjectId, handleSelectProject]);

  const handleSelectCatalog = useCallback((catalogId: string) => {
    setSelectedCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
  }, []);

  const handleConfigureDatasetCatalog = useCallback((catalogId: string) => {
    setSelectedCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId(catalogId);
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "datasets", command: "library" });
  }, []);

  const handleBackToDatasetLibrary = useCallback(() => {
    setConfigureDatasetCatalogId("");
    setRoute({ topTab: "datasets", command: "library" });
  }, []);

  const handleSelectDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handleSelectFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
  }, []);

  const handleConfigureFeatureCatalog = useCallback((catalogId: string) => {
    setSelectedFeatureCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId(catalogId);
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "features", command: "library" });
  }, []);

  const handleSelectFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "features", command: "features" });
  }, []);

  const handleDatasetCreated = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "datasets", command: "datasets" });
  }, []);

  const handleFeatureCreated = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "features", command: "features" });
  }, []);

  const handleExploreDataset = useCallback((datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId(datasetId);
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "datasets", command: "explore" });
  }, []);

  const handleExploreGraphChange = useCallback(
    (graphId: string, summary: DatasetGraphSummary | null, options: { clearNode?: boolean } = {}) => {
      setExploreGraphId(graphId);
      setExploreGraphSummary(summary);
      if (options.clearNode !== false) {
        setExploreNodeId("");
        setExploreNodeVisible(null);
      }
    },
    []
  );

  const handleExploreNodeChange = useCallback((nodeId: string) => {
    setExploreNodeId(nodeId);
    setExploreNodeVisible(null);
  }, []);

  const handleExploreFeature = useCallback((featureId: string) => {
    setSelectedFeatureId(featureId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setExploreFeatureId(featureId);
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "features", command: "explore" });
  }, []);

  const handleClearExploreFeature = useCallback(() => {
    setSelectedFeatureId("");
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setRoute({ topTab: "features", command: "explore" });
  }, []);

  const handleExploreFeatureGraphChange = useCallback((graphId: string, visible: boolean | null) => {
    setExploreFeatureGraphId(graphId);
    setExploreFeatureGraphVisible(visible);
  }, []);

  const handleSelectEmbeddingCatalog = useCallback((catalogId: string) => {
    setSelectedEmbeddingCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
  }, []);

  const handleConfigureEmbeddingCatalog = useCallback((catalogId: string) => {
    setSelectedEmbeddingCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId(catalogId);
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "embeddings", command: "library" });
  }, []);

  const handleSelectEmbedding = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "embeddings", command: "embeddings" });
  }, []);

  const handleEmbeddingCreated = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "embeddings", command: "embeddings" });
  }, []);

  const handleExploreEmbedding = useCallback((embeddingId: string) => {
    setSelectedEmbeddingId(embeddingId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreEmbeddingId(embeddingId);
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "embeddings", command: "explore" });
  }, []);

  const handleClearExploreEmbedding = useCallback(() => {
    setSelectedEmbeddingId("");
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setRoute({ topTab: "embeddings", command: "explore" });
  }, []);

  const handleExploreEmbeddingGraphChange = useCallback((graphId: string, visible: boolean | null) => {
    setExploreEmbeddingGraphId(graphId);
    setExploreEmbeddingGraphVisible(visible);
  }, []);

  const handleSelectModelCatalog = useCallback((catalogId: string) => {
    setSelectedModelCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
  }, []);

  const handleConfigureModelCatalog = useCallback((catalogId: string) => {
    setSelectedModelCatalogId(catalogId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId(catalogId);
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "models", command: "library" });
  }, []);

  const handleSelectModel = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "models", command: "models" });
  }, []);

  const handleModelCreated = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelCatalogId("");
    setConfigureDatasetCatalogId("");
    setConfigureFeatureCatalogId("");
    setConfigureEmbeddingCatalogId("");
    setConfigureModelCatalogId("");
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "models", command: "models" });
  }, []);

  const handleExploreModel = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    setSelectedDatasetId("");
    setSelectedCatalogId("");
    setSelectedFeatureId("");
    setSelectedFeatureCatalogId("");
    setSelectedEmbeddingId("");
    setSelectedEmbeddingCatalogId("");
    setSelectedModelCatalogId("");
    setExploreModelId(modelId);
    setExploreModelIteration(null);
    setExploreDatasetId("");
    setExploreGraphId("");
    setExploreGraphSummary(null);
    setExploreNodeId("");
    setExploreNodeVisible(null);
    setExploreFeatureId("");
    setExploreFeatureGraphId("");
    setExploreFeatureGraphVisible(null);
    setExploreEmbeddingId("");
    setExploreEmbeddingGraphId("");
    setExploreEmbeddingGraphVisible(null);
    setRoute({ topTab: "models", command: "explore" });
  }, []);

  const handleClearExploreModel = useCallback(() => {
    setSelectedModelId("");
    setExploreModelId("");
    setExploreModelIteration(null);
    setRoute({ topTab: "models", command: "explore" });
  }, []);

  const handleExploreModelIterationChange = useCallback((iteration: number | null) => {
    setExploreModelIteration(iteration);
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
          onSelectProject={handleSelectProject}
        />
      );
    }
    if (route.topTab === "home" && route.command === "settings") {
      return <SettingsView />;
    }
    if (route.topTab === "datasets" && route.command === "library" && configureDatasetCatalogId) {
      return (
        <ConfigureDatasetView
          activeProjectId={activeProjectId}
          entry={configuredDatasetCatalogEntry}
          onBack={handleBackToDatasetLibrary}
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
    if (route.topTab === "datasets" && route.command === "explore") {
      return (
        <DatasetExploreView
          activeProjectId={activeProjectId}
          datasets={datasets}
          loading={projectDatasetsQuery.isLoading}
          selectedDatasetId={selectedDatasetId}
          exploreDatasetId={exploreDatasetId}
          exploreGraphId={exploreGraphId}
          exploreNodeId={exploreNodeId}
          onExploreDataset={handleExploreDataset}
          onExploreGraphChange={handleExploreGraphChange}
          onExploreNodeChange={handleExploreNodeChange}
          onExploreNodeVisualStateChange={setExploreNodeVisible}
        />
      );
    }
    if (route.topTab === "datasets" && route.command === "datasets") {
      return (
        <ProjectDatasetsView
          activeProjectId={activeProjectId}
          datasets={datasets}
          loading={projectDatasetsQuery.isLoading}
          selectedDatasetId={selectedDatasetId}
          onSelectDataset={handleSelectDataset}
          onPreviewDataset={handleExploreDataset}
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
    if (route.topTab === "features" && route.command === "explore") {
      return (
        <FeatureExploreView
          activeProjectId={activeProjectId}
          features={features}
          datasets={datasets}
          catalog={featureLibraryQuery.data || []}
          loading={projectFeaturesQuery.isLoading}
          selectedFeatureId={selectedFeatureId}
          exploreFeatureId={exploreFeatureId}
          selectedGraphId={exploreFeatureGraphId}
          onExploreFeature={handleExploreFeature}
          onClearExploreFeature={handleClearExploreFeature}
          onSelectGraph={handleExploreFeatureGraphChange}
          onSelectedGraphVisibilityChange={setExploreFeatureGraphVisible}
        />
      );
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
          onPreviewFeature={handleExploreFeature}
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
    if (route.topTab === "embeddings" && route.command === "explore") {
      return (
        <EmbeddingExploreView
          activeProjectId={activeProjectId}
          embeddings={embeddings}
          features={features}
          datasets={datasets}
          catalog={embeddingLibraryQuery.data || []}
          loading={projectEmbeddingsQuery.isLoading}
          selectedEmbeddingId={selectedEmbeddingId}
          exploreEmbeddingId={exploreEmbeddingId}
          selectedGraphId={exploreEmbeddingGraphId}
          onExploreEmbedding={handleExploreEmbedding}
          onClearExploreEmbedding={handleClearExploreEmbedding}
          onSelectGraph={handleExploreEmbeddingGraphChange}
          onSelectedGraphVisibilityChange={setExploreEmbeddingGraphVisible}
        />
      );
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
          onPreviewEmbedding={handleExploreEmbedding}
        />
      );
    }
    if (route.topTab === "models" && route.command === "library" && configureModelCatalogId) {
      return (
        <ConfigureModelView
          activeProjectId={activeProjectId}
          model={configuredModelCatalogEntry}
          embeddings={embeddings}
          features={features}
          datasets={datasets}
          loading={projectEmbeddingsQuery.isLoading || projectFeaturesQuery.isLoading || projectDatasetsQuery.isLoading}
          onCreated={handleModelCreated}
        />
      );
    }
    if (route.topTab === "models" && route.command === "library") {
      return (
        <ModelLibraryView
          activeProjectId={activeProjectId}
          catalog={modelLibraryQuery.data || []}
          loading={modelLibraryQuery.isLoading}
          selectedCatalogId={selectedModelCatalogId}
          onSelectCatalog={handleSelectModelCatalog}
          onConfigure={handleConfigureModelCatalog}
        />
      );
    }
    if (route.topTab === "models" && route.command === "explore") {
      return (
        <ModelExploreView
          activeProjectId={activeProjectId}
          models={models}
          embeddings={embeddings}
          features={features}
          datasets={datasets}
          catalog={modelLibraryQuery.data || []}
          loading={projectModelsQuery.isLoading}
          selectedModelId={selectedModelId}
          exploreModelId={exploreModelId}
          selectedIteration={exploreModelIteration}
          onExploreModel={handleExploreModel}
          onClearExploreModel={handleClearExploreModel}
          onSelectIteration={handleExploreModelIterationChange}
        />
      );
    }
    if (route.topTab === "models" && route.command === "models") {
      return (
        <ProjectModelsView
          activeProjectId={activeProjectId}
          models={models}
          embeddings={embeddings}
          features={features}
          datasets={datasets}
          catalog={modelLibraryQuery.data || []}
          loading={projectModelsQuery.isLoading}
          selectedModelId={selectedModelId}
          onSelectModel={handleSelectModel}
          onPreviewModel={handleExploreModel}
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
          contextLabel={selectionLineage.contextLabel}
          datasets={selectionLineage.datasets}
          features={selectionLineage.features}
          embeddings={selectionLineage.embeddings}
          models={selectionLineage.models}
          relatedEmbeddingIds={selectionLineage.relatedEmbeddingIds}
          relatedModelIds={selectionLineage.relatedModelIds}
          isProjectSelected={isProjectSelected}
          selectedDatasetId={selectionLineage.activeDatasetId}
          selectedFeatureId={selectedFeatureId}
          selectedEmbeddingId={selectedEmbeddingId}
          selectedModelId={selectedModelId}
          onSelectProject={handleSelectActiveProject}
          onSelectDataset={handleSelectDataset}
          onSelectFeature={handleSelectFeature}
          onSelectEmbedding={handleSelectEmbedding}
          onSelectModel={handleSelectModel}
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
            activeProjectId={activeProjectId}
            project={project}
            dataset={selectedDataset}
            exploreDataset={route.topTab === "datasets" && route.command === "explore" ? selectedDataset : undefined}
            exploreGraphSummary={route.topTab === "datasets" && route.command === "explore" ? exploreGraphSummary : null}
            exploreNodeId={route.topTab === "datasets" && route.command === "explore" ? exploreNodeId : ""}
            exploreNodeVisible={route.topTab === "datasets" && route.command === "explore" ? exploreNodeVisible : null}
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
            exploreFeature={route.topTab === "features" && route.command === "explore" ? selectedFeature : undefined}
            exploreFeatureDataset={route.topTab === "features" && route.command === "explore" ? selectedFeatureDataset : undefined}
            exploreFeatureGraphId={route.topTab === "features" && route.command === "explore" ? exploreFeatureGraphId : ""}
            exploreFeatureGraphVisible={route.topTab === "features" && route.command === "explore" ? exploreFeatureGraphVisible : null}
            embedding={selectedEmbedding}
            embeddingDataset={selectedEmbeddingDataset}
            embeddingFeatures={selectedEmbeddingFeatures}
            embeddingCatalogEntry={selectedEmbeddingAlgorithm}
            selectedEmbeddingCatalogEntry={selectedEmbeddingCatalogEntry}
            exploreEmbedding={route.topTab === "embeddings" && route.command === "explore" ? selectedEmbedding : undefined}
            exploreEmbeddingDataset={route.topTab === "embeddings" && route.command === "explore" ? selectedEmbeddingDataset : undefined}
            exploreEmbeddingFeatures={route.topTab === "embeddings" && route.command === "explore" ? selectedEmbeddingFeatures : []}
            exploreEmbeddingGraphId={route.topTab === "embeddings" && route.command === "explore" ? exploreEmbeddingGraphId : ""}
            exploreEmbeddingGraphVisible={route.topTab === "embeddings" && route.command === "explore" ? exploreEmbeddingGraphVisible : null}
            model={selectedModel}
            modelDataset={selectedModelDataset}
            modelEmbeddings={selectedModelEmbeddings}
            modelCatalogEntry={selectedModelAlgorithm}
            selectedModelCatalogEntry={selectedModelCatalogEntry}
            exploreModel={route.topTab === "models" && route.command === "explore" ? selectedModel : undefined}
            exploreModelDataset={route.topTab === "models" && route.command === "explore" ? selectedModelDataset : undefined}
            exploreModelEmbeddings={route.topTab === "models" && route.command === "explore" ? selectedModelEmbeddings : []}
            exploreModelIteration={route.topTab === "models" && route.command === "explore" ? exploreModelIteration : null}
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
