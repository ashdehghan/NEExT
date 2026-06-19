import { type MutableRefObject, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { ArrowLeft, BarChart3, Download, Eye, Play, Plus, RotateCcw, Trash2 } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type EmbeddingManifest,
  type FeatureManifest,
  type ModelAnalysis,
  type ModelCatalogEntry,
  type ModelCreatePayload,
  type ModelFeatureImportanceItem,
  type ModelManifest,
  type ModelMetricSeries
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { RunningState } from "../../components/primitives/RunningState";
import { ChartCard } from "../../components/viz/ChartCard";

interface ModelLibraryViewProps {
  activeProjectId: string;
  catalog: ModelCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  selectedDataset?: DatasetManifest;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureModelViewProps {
  activeProjectId: string;
  model?: ModelCatalogEntry;
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  dataset?: DatasetManifest;
  loading: boolean;
  initialSelectedEmbeddingId?: string;
  draft?: Record<string, unknown>;
  onCreated: (modelId: string) => void;
}

interface ProjectModelsViewProps {
  activeProjectId: string;
  models: ModelManifest[];
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: ModelCatalogEntry[];
  loading: boolean;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  onPreviewModel: (modelId: string) => void;
  onDeleteArtifact: (artifactKind: "model", artifactId: string) => void;
}

interface ModelExploreViewProps {
  activeProjectId: string;
  models: ModelManifest[];
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: ModelCatalogEntry[];
  loading: boolean;
  selectedModelId: string;
  exploreModelId: string;
  selectedIteration: number | null;
  onExploreModel: (modelId: string) => void;
  onClearExploreModel: () => void;
  onSelectIteration: (iteration: number | null) => void;
}

export function embeddingFeatureIds(embedding: EmbeddingManifest): string[] {
  return embedding.inputs
    .filter((input) => input.role === "source_feature" && input.artifact_kind === "feature")
    .map((input) => input.artifact_id);
}

export function featureDatasetId(feature: FeatureManifest): string {
  return feature.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset")?.artifact_id || "";
}

export function embeddingDatasetId(embedding: EmbeddingManifest, features: FeatureManifest[]): string {
  const featuresById = new Map(features.map((feature) => [feature.id, feature]));
  const datasetIds = new Set(
    embeddingFeatureIds(embedding)
      .map((featureId) => featuresById.get(featureId))
      .filter(Boolean)
      .map((feature) => featureDatasetId(feature as FeatureManifest))
      .filter(Boolean)
  );
  return datasetIds.size === 1 ? Array.from(datasetIds)[0] : "";
}

export function modelEmbeddingIds(model: ModelManifest): string[] {
  return model.inputs
    .filter((input) => input.role === "source_embedding" && input.artifact_kind === "embedding")
    .map((input) => input.artifact_id);
}

export function taskLabel(taskType?: string): string {
  if (taskType === "classifier") return "Classifier";
  if (taskType === "regressor") return "Regressor";
  return "";
}

export function algorithmLabel(catalogEntry?: ModelCatalogEntry, fallback?: string): string {
  return catalogEntry?.name || fallback || "";
}

function artifactStatusLabel(status: string): string {
  if (status === "planned") return "Draft";
  if (status === "completed") return "Ready";
  if (status === "running") return "Running";
  if (status === "failed") return "Failed";
  return status;
}

function artifactStatusClass(status: string): string {
  if (status === "completed") return "is-completed";
  if (status === "running") return "is-running";
  if (status === "failed") return "is-failed";
  if (status === "planned" || status === "queued") return "is-queued";
  return "is-idle";
}

function draftString(draft: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = draft?.[key];
  return typeof value === "string" ? value : undefined;
}

function draftNumber(draft: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = draft?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function draftBoolean(draft: Record<string, unknown> | undefined, key: string): boolean | undefined {
  const value = draft?.[key];
  return typeof value === "boolean" ? value : undefined;
}

function draftStringArray(draft: Record<string, unknown> | undefined, key: string): string[] | undefined {
  const value = draft?.[key];
  return Array.isArray(value) ? value.map(String).filter(Boolean) : undefined;
}

export function ModelLibraryView({
  activeProjectId,
  catalog,
  loading,
  selectedCatalogId,
  selectedDataset,
  onSelectCatalog,
  onConfigure
}: ModelLibraryViewProps) {
  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="library" size={16} />
            Model Library · {catalog.length} {catalog.length === 1 ? "algorithm" : "algorithms"}
          </span>
          <span className="muted">{selectedDataset ? `Dataset: ${selectedDataset.name}` : activeProjectId ? "Select a dataset first" : "No active project"}</span>
        </header>
        {loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading model library.</EmptyState>
          </div>
        ) : catalog.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No model catalog entries.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Algorithm</th>
                  <th>Output</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {catalog.map((entry) => (
                  <tr
                    key={entry.id}
                    className={entry.id === selectedCatalogId ? "is-selected" : ""}
                    onClick={() => onSelectCatalog(entry.id)}
                  >
                    <td>
                      <strong>{entry.name}</strong>
                    </td>
                    <td className="mono">{entry.id}</td>
                    <td>{entry.output}</td>
                    <td className="actions-cell actions-cell-wide">
                      <button
                        type="button"
                        className="btn"
                        disabled={!activeProjectId}
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectCatalog(entry.id);
                          onConfigure(entry.id);
                        }}
                      >
                        <Plus />
                        Add to Project
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export function ConfigureModelView({
  activeProjectId,
  model,
  embeddings,
  features,
  loading,
  dataset,
  initialSelectedEmbeddingId = "",
  draft,
  onCreated
}: ConfigureModelViewProps) {
  const queryClient = useQueryClient();
  const [selectedEmbeddingIds, setSelectedEmbeddingIds] = useState<string[]>([]);
  const [taskType, setTaskType] = useState<"classifier" | "regressor">("classifier");
  const [sampleSize, setSampleSize] = useState(5);
  const [testSize, setTestSize] = useState(0.3);
  const [balanceDataset, setBalanceDataset] = useState(false);
  const [nJobs, setNJobs] = useState(1);
  const [parallelBackend, setParallelBackend] = useState<"thread" | "process">("thread");
  const initialEmbeddingId = initialSelectedEmbeddingId && embeddings.some((embedding) => embedding.id === initialSelectedEmbeddingId)
    ? initialSelectedEmbeddingId
    : "";

  useEffect(() => {
    setSelectedEmbeddingIds(initialEmbeddingId ? [initialEmbeddingId] : []);
    setTaskType("classifier");
    setSampleSize(5);
    setTestSize(0.3);
    setBalanceDataset(false);
    setNJobs(1);
    setParallelBackend("thread");
  }, [activeProjectId, model?.id, dataset?.id, initialEmbeddingId]);

  useEffect(() => {
    if (!draft) return;
    const requestedEmbeddingIds = draftStringArray(draft, "source_embedding_ids") || draftStringArray(draft, "embedding_ids");
    if (requestedEmbeddingIds) {
      const validEmbeddingIds = requestedEmbeddingIds.filter((embeddingId) => embeddings.some((embedding) => embedding.id === embeddingId));
      setSelectedEmbeddingIds(validEmbeddingIds);
    }
    const nextTaskType = draftString(draft, "task_type");
    if (nextTaskType === "classifier" || nextTaskType === "regressor") setTaskType(nextTaskType);
    const nextSampleSize = draftNumber(draft, "sample_size");
    if (nextSampleSize !== undefined) setSampleSize(nextSampleSize);
    const nextTestSize = draftNumber(draft, "test_size");
    if (nextTestSize !== undefined) setTestSize(nextTestSize);
    const nextBalance = draftBoolean(draft, "balance_dataset");
    if (nextBalance !== undefined) setBalanceDataset(nextBalance);
    const nextNJobs = draftNumber(draft, "n_jobs");
    if (nextNJobs !== undefined) setNJobs(nextNJobs);
    const nextParallelBackend = draftString(draft, "parallel_backend");
    if (nextParallelBackend === "thread" || nextParallelBackend === "process") setParallelBackend(nextParallelBackend);
  }, [activeProjectId, model?.id, dataset?.id, embeddings, draft]);

  useEffect(() => {
    setSelectedEmbeddingIds((current) => current.filter((embeddingId) => embeddings.some((embedding) => embedding.id === embeddingId)));
  }, [embeddings]);

  function setEmbeddingSelected(embeddingId: string, selected: boolean) {
    setSelectedEmbeddingIds((current) => {
      const isSelected = current.includes(embeddingId);
      if (selected) return isSelected ? current : [...current, embeddingId];
      return isSelected ? current.filter((currentEmbeddingId) => currentEmbeddingId !== embeddingId) : current;
    });
  }

  const selectedEmbeddings = useMemo(
    () => selectedEmbeddingIds.map((embeddingId) => embeddings.find((embedding) => embedding.id === embeddingId)).filter(Boolean) as EmbeddingManifest[],
    [embeddings, selectedEmbeddingIds]
  );
  const selectedEmbeddingsInDataset = selectedEmbeddings.every((embedding) => dataset?.id && embeddingDatasetId(embedding, features) === dataset.id);
  const paramsValid =
    Number.isInteger(sampleSize) &&
    sampleSize >= 1 &&
    sampleSize <= 100 &&
    Number.isFinite(testSize) &&
    testSize >= 0.05 &&
    testSize <= 0.95 &&
    Number.isInteger(nJobs) &&
    nJobs >= 1 &&
    nJobs <= 32;
  const canSave = Boolean(activeProjectId && model && dataset?.id && selectedEmbeddingIds.length > 0 && selectedEmbeddingsInDataset && paramsValid);
  const saveBlockedMessage = !activeProjectId
    ? "An active project is required."
    : !dataset
      ? "Select a dataset before configuring models."
      : !loading && embeddings.length === 0
        ? "An embedding artifact is required before saving."
        : selectedEmbeddingIds.length === 0
          ? "Select at least one embedding."
          : !selectedEmbeddingsInDataset
            ? "Selected embeddings must belong to the active dataset."
            : !paramsValid
              ? "Sample size must be 1-100, test size 0.05-0.95, and parallel jobs 1-32."
              : "";

  const createModel = useMutation({
    mutationFn: (payload: ModelCreatePayload) => api.createModel(activeProjectId, payload),
    onSuccess: (created) => {
      queryClient.setQueryData<ModelManifest[]>(["projects", activeProjectId, "models"], (current = []) => [
        created,
        ...current.filter((modelItem) => modelItem.id !== created.id)
      ]);
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "models"] });
      onCreated(created.id);
    }
  });

  if (!model) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a model library entry.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <form
      className="card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSave || createModel.isPending) return;
        createModel.mutate({
          source_model_id: model.id,
          source_embedding_ids: selectedEmbeddingIds,
          params: {
            task_type: taskType,
            sample_size: sampleSize,
            test_size: testSize,
            balance_dataset: taskType === "classifier" ? balanceDataset : false,
            n_jobs: nJobs,
            parallel_backend: parallelBackend
          }
        });
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="models" size={32} />
        </span>
        <div>
          <h3>Add {model.name} to Project</h3>
          <p className="form-subtitle">{model.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createModel.error ? <p className="error-text">{createModel.error.message}</p> : null}
        {saveBlockedMessage ? <p className="muted form-note">{saveBlockedMessage}</p> : null}
        {dataset ? <p className="muted form-note">Dataset: {dataset.name}</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Algorithm</span>
            <span className="readonly-value mono">{model.id}</span>
          </label>
          <label className="field">
            <span>Task Type</span>
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as "classifier" | "regressor")}>
              <option value="classifier">classifier</option>
              <option value="regressor">regressor</option>
            </select>
          </label>
          <label className="field">
            <span>Sample Size</span>
            <input type="number" min={1} max={100} value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} />
          </label>
          <label className="field">
            <span>Test Size</span>
            <input
              type="number"
              min={0.05}
              max={0.95}
              step={0.05}
              value={testSize}
              onChange={(event) => setTestSize(Number(event.target.value))}
            />
          </label>
          <label className="field">
            <span>Parallel Jobs</span>
            <input type="number" min={1} max={32} value={nJobs} onChange={(event) => setNJobs(Number(event.target.value))} />
          </label>
          <label className="field">
            <span>Parallel Backend</span>
            <select value={parallelBackend} onChange={(event) => setParallelBackend(event.target.value as "thread" | "process")}>
              <option value="thread">thread</option>
              <option value="process">process</option>
            </select>
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={balanceDataset}
              disabled={taskType !== "classifier"}
              onChange={(event) => setBalanceDataset(event.target.checked)}
            />
            <span>Balance Dataset</span>
          </label>
        </div>
        <div className="feature-picker" aria-label="Embedding artifacts">
          <table className="tbl">
            <thead>
              <tr>
                <th />
                <th>Name</th>
                <th>Dataset</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={4}>Loading embeddings.</td>
                </tr>
              ) : embeddings.length === 0 ? (
                <tr>
                  <td colSpan={4}>No embeddings.</td>
                </tr>
              ) : (
                embeddings.map((embedding) => {
                  const isChecked = selectedEmbeddingIds.includes(embedding.id);
                  return (
                    <tr
                      key={embedding.id}
                      className={isChecked ? "is-selected" : ""}
                      onClick={() => setEmbeddingSelected(embedding.id, !isChecked)}
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(event) => {
                            event.stopPropagation();
                            setEmbeddingSelected(embedding.id, event.target.checked);
                          }}
                          onClick={(event) => event.stopPropagation()}
                        />
                      </td>
                      <td>
                        <strong>{embedding.name}</strong>
                      </td>
                      <td>{dataset?.name || "Unknown dataset"}</td>
                      <td>
                        <span className={`status-pill ${artifactStatusClass(embedding.status)}`}>{artifactStatusLabel(embedding.status)}</span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
      <footer className="card-foot">
        <button type="submit" className="btn btn-primary" disabled={!canSave || createModel.isPending}>
          <Plus />
          {createModel.isPending ? "Creating" : "Create Model"}
        </button>
      </footer>
    </form>
  );
}

export function ProjectModelsView({
  activeProjectId,
  models,
  embeddings,
  features,
  datasets,
  catalog,
  loading,
  selectedModelId,
  onSelectModel,
  onPreviewModel,
  onDeleteArtifact
}: ProjectModelsViewProps) {
  const queryClient = useQueryClient();
  const embeddingsById = useMemo(() => new Map(embeddings.map((embedding) => [embedding.id, embedding])), [embeddings]);
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const [checkedModelIds, setCheckedModelIds] = useState<string[]>([]);

  const invalidateAfterRun = () => {
    queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "models"] });
    queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
    queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
    queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
    queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
  };

  const runModel = useMutation({
    mutationFn: (modelId: string) => api.runModel(activeProjectId, modelId),
    onSuccess: invalidateAfterRun
  });
  const runBatch = useMutation({
    mutationFn: (modelIds: string[]) => api.runModelBatch(activeProjectId, modelIds),
    onSuccess: invalidateAfterRun
  });

  const runnableCheckedModelIds = checkedModelIds.filter((modelId) => {
    const model = models.find((item) => item.id === modelId);
    return model?.status === "planned" || model?.status === "failed";
  });

  useEffect(() => {
    setCheckedModelIds((current) => current.filter((modelId) => models.some((model) => model.id === modelId)));
  }, [models]);

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="models" size={16} />
            Models · {models.length} {models.length === 1 ? "model" : "models"}
          </span>
          <span className="muted">{activeProjectId ? "Active project" : "No active project"}</span>
        </header>
        {runModel.error ? <p className="table-error">{runModel.error.message}</p> : null}
        {runBatch.error ? <p className="table-error">{runBatch.error.message}</p> : null}
        {!activeProjectId ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No active project.</EmptyState>
          </div>
        ) : loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading models.</EmptyState>
          </div>
        ) : models.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No models.</EmptyState>
          </div>
        ) : (
          <>
            <div className="table-toolbar">
              <button
                type="button"
                className="btn"
                disabled={runnableCheckedModelIds.length === 0 || runBatch.isPending}
                onClick={() => runBatch.mutate(runnableCheckedModelIds)}
              >
                <Play />
                Train Selected
              </button>
            </div>
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th />
                    <th>Name</th>
                    <th>Dataset</th>
                    <th>Algorithm</th>
                    <th>Task</th>
                    <th>Embeddings</th>
                    <th>Status</th>
                    <th>Updated</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => {
                    const sourceEmbeddingIds = modelEmbeddingIds(model);
                    const sourceEmbeddings = sourceEmbeddingIds.map((embeddingId) => embeddingsById.get(embeddingId)).filter(Boolean) as EmbeddingManifest[];
                    const datasetId = sourceEmbeddings[0] ? embeddingDatasetId(sourceEmbeddings[0], features) : "";
                    const datasetName = datasetsById.get(datasetId)?.name || "Unknown dataset";
                    const embeddingNames = sourceEmbeddings.map((embedding) => embedding.name).join(", ") || `${sourceEmbeddingIds.length} embeddings`;
                    const algorithm = algorithmLabel(catalogById.get(model.source_model_id), model.source_model_id);
                    const isRunnable = model.status === "planned" || model.status === "failed";
                    const isRunning = model.status === "running" || (runModel.isPending && runModel.variables === model.id);
                    const isChecked = checkedModelIds.includes(model.id);
                    return (
                      <tr
                        key={model.id}
                        className={model.id === selectedModelId ? "is-selected" : ""}
                        onClick={() => onSelectModel(model.id)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            checked={isChecked}
                            disabled={!isRunnable}
                            onChange={(event) => {
                              event.stopPropagation();
                              setCheckedModelIds((current) =>
                                event.target.checked
                                  ? current.includes(model.id)
                                    ? current
                                    : [...current, model.id]
                                  : current.filter((modelId) => modelId !== model.id)
                              );
                            }}
                            onClick={(event) => event.stopPropagation()}
                          />
                        </td>
                        <td>
                          <span className="table-name-with-icon">
                            <BarChart3 />
                            <strong>{model.name}</strong>
                          </span>
                        </td>
                        <td>{datasetName}</td>
                        <td>{algorithm}</td>
                        <td>{taskLabel(String(model.operation.params.task_type))}</td>
                        <td>{embeddingNames}</td>
                        <td>
                          <span className={`status-pill ${artifactStatusClass(model.status)}`}>{artifactStatusLabel(model.status)}</span>
                        </td>
                        <td className="muted mono">{model.updated_at}</td>
                        <td className="actions-cell actions-cell-wide">
                          {isRunnable ? (
                            <button
                              type="button"
                              className="btn"
                              disabled={isRunning}
                              onClick={(event) => {
                                event.stopPropagation();
                                runModel.mutate(model.id);
                              }}
                            >
                              {model.status === "failed" ? <RotateCcw /> : <Play />}
                              {model.status === "failed" ? "Retry Train" : isRunning ? "Training" : "Train"}
                            </button>
                          ) : null}
                          {model.status === "completed" ? (
                            <button
                              type="button"
                              className="btn"
                              onClick={(event) => {
                                event.stopPropagation();
                                onSelectModel(model.id);
                                onPreviewModel(model.id);
                              }}
                            >
                              <Eye />
                              Preview
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="icon-btn icon-btn-danger"
                            aria-label={`Delete ${model.name}`}
                            title={`Delete ${model.name}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteArtifact("model", model.id);
                            }}
                          >
                            <Trash2 />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function metricLabel(metric: string): string {
  if (metric === "f1_score") return "F1 Score";
  if (metric === "auc") return "AUC";
  if (metric === "rmse") return "RMSE";
  if (metric === "mae") return "MAE";
  return metric.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatMetric(value: unknown): string {
  return typeof value === "number" ? value.toFixed(4) : value == null ? "" : String(value);
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function modelDatasetId(model: ModelManifest, embeddings: EmbeddingManifest[], features: FeatureManifest[]): string {
  const embeddingsById = new Map(embeddings.map((embedding) => [embedding.id, embedding]));
  const sourceEmbedding = modelEmbeddingIds(model).map((embeddingId) => embeddingsById.get(embeddingId)).find(Boolean);
  return sourceEmbedding ? embeddingDatasetId(sourceEmbedding, features) : "";
}

function modelFeatureIds(model: ModelManifest, embeddings: EmbeddingManifest[]): string[] {
  const embeddingsById = new Map(embeddings.map((embedding) => [embedding.id, embedding]));
  const featureIds: string[] = [];
  for (const embeddingId of modelEmbeddingIds(model)) {
    const embedding = embeddingsById.get(embeddingId);
    if (!embedding) continue;
    for (const featureId of embeddingFeatureIds(embedding)) {
      if (!featureIds.includes(featureId)) featureIds.push(featureId);
    }
  }
  return featureIds;
}

type ModelMetricsChartPoint = {
  value: [number, number | null];
  iteration: number;
  metric: string;
  symbolSize: number;
};

type ModelMetricsChartElement = HTMLDivElement & {
  __modelMetricsChart?: ReturnType<typeof echarts.init>;
};

// Shared academic chart styling tokens (kept in sync with viz/AnalysisCommandCenter).
const CHART_FONT = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const AXIS_NAME_COLOR = "#3a444e";
const AXIS_LABEL_COLOR = "#5b6671";
const AXIS_LINE_COLOR = "#c8d0d6";
const SPLIT_LINE_COLOR = "#eef1f4";

function ModelMetricsChart({
  analysis,
  selectedIteration,
  onSelectIteration,
  chartRef
}: {
  analysis: ModelAnalysis;
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
  chartRef: MutableRefObject<ReturnType<typeof echarts.init> | null>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const onSelectIterationRef = useRef(onSelectIteration);

  useEffect(() => {
    onSelectIterationRef.current = onSelectIteration;
  }, [onSelectIteration]);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current);
    chartRef.current = chart;
    (containerRef.current as ModelMetricsChartElement).__modelMetricsChart = chart;
    const handleClick = (params: { data?: unknown }) => {
      const data = params.data as ModelMetricsChartPoint | undefined;
      if (typeof data?.iteration !== "number") return;
      onSelectIterationRef.current(data.iteration);
    };
    chart.on("click", handleClick);
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(containerRef.current);
    return () => {
      chart.off("click", handleClick);
      resizeObserver.disconnect();
      chart.dispose();
      if (containerRef.current) delete (containerRef.current as ModelMetricsChartElement).__modelMetricsChart;
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const palette = ["#176ea9", "#d86c1f", "#2d8754", "#8d5db8", "#a4513d", "#4f758b"];
    const series = analysis.metric_series.map((metricSeries: ModelMetricSeries, seriesIndex) => ({
      name: metricLabel(metricSeries.metric),
      type: "line" as const,
      showSymbol: true,
      symbolSize: 8,
      data: metricSeries.points.map((point) => ({
        value: [point.iteration, point.value ?? null] as [number, number | null],
        iteration: point.iteration,
        metric: metricSeries.metric,
        symbolSize: selectedIteration === point.iteration ? 14 : 8,
        itemStyle: {
          color: palette[seriesIndex % palette.length],
          borderColor: selectedIteration === point.iteration ? "#111820" : palette[seriesIndex % palette.length],
          borderWidth: selectedIteration === point.iteration ? 2 : 0
        }
      }))
    }));

    const option: EChartsOption = {
      animation: false,
      color: palette,
      textStyle: { fontFamily: CHART_FONT },
      grid: { left: 54, right: 24, top: 34, bottom: 48 },
      legend: { top: 4, type: "scroll", icon: "circle", itemWidth: 8, itemHeight: 8, textStyle: { fontSize: 11, color: AXIS_NAME_COLOR, fontFamily: CHART_FONT } },
      xAxis: {
        type: "value",
        name: "Iteration",
        nameLocation: "middle",
        nameGap: 30,
        minInterval: 1,
        nameTextStyle: { color: AXIS_NAME_COLOR, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: AXIS_LABEL_COLOR, fontSize: 11 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } },
        splitLine: { lineStyle: { color: SPLIT_LINE_COLOR } }
      },
      yAxis: {
        type: "value",
        name: "Metric",
        nameLocation: "middle",
        nameGap: 38,
        scale: true,
        nameTextStyle: { color: AXIS_NAME_COLOR, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: AXIS_LABEL_COLOR, fontSize: 11 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } },
        splitLine: { lineStyle: { color: SPLIT_LINE_COLOR } }
      },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = Array.isArray(params) ? params[0] : params;
          const dataPoint = (item as { data?: ModelMetricsChartPoint }).data;
          if (!dataPoint) return "";
          return [
            `Iteration ${dataPoint.iteration}`,
            `${metricLabel(dataPoint.metric)} ${formatMetric(dataPoint.value[1])}`
          ].join("<br/>");
        }
      },
      series
    };

    chart.setOption(option, { notMerge: true });
  }, [analysis, selectedIteration]);

  return (
    <div
      ref={containerRef}
      className="feature-pca-chart chart-card-canvas model-metrics-chart"
      role="img"
      aria-label={`${analysis.model_name} metric results`}
      tabIndex={0}
    />
  );
}

function ModelFeatureImportanceChart({
  ranking,
  scoreLabel,
  chartRef
}: {
  ranking: ModelFeatureImportanceItem[];
  scoreLabel: string;
  chartRef: MutableRefObject<ReturnType<typeof echarts.init> | null>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current);
    chartRef.current = chart;
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(containerRef.current);
    return () => {
      resizeObserver.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const ordered = [...ranking].sort((a, b) => a.rank - b.rank);
    // echarts category axis plots bottom-up, so reverse to show rank 1 at the top
    const categories = ordered.map((item) => item.feature_name).reverse();
    const values = ordered.map((item) => item.score).reverse();
    const option: EChartsOption = {
      animation: false,
      textStyle: { fontFamily: CHART_FONT },
      grid: { left: 160, right: 28, top: 14, bottom: 44 },
      xAxis: {
        type: "value",
        name: scoreLabel,
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { color: AXIS_NAME_COLOR, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: AXIS_LABEL_COLOR, fontSize: 11 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } },
        splitLine: { lineStyle: { color: SPLIT_LINE_COLOR } }
      },
      yAxis: {
        type: "category",
        data: categories,
        axisLabel: { fontSize: 11, color: AXIS_LABEL_COLOR },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } }
      },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = Array.isArray(params) ? params[0] : params;
          const point = item as { name?: string; value?: number };
          return `${point.name}<br/>${scoreLabel} ${typeof point.value === "number" ? point.value.toFixed(4) : point.value}`;
        }
      },
      series: [{ type: "bar", data: values, itemStyle: { color: "#176ea9" }, barMaxWidth: 18 }]
    };
    chart.setOption(option, { notMerge: true });
  }, [ranking, scoreLabel]);

  return (
    <div
      ref={containerRef}
      className="feature-pca-chart chart-card-canvas model-metrics-chart"
      role="img"
      aria-label="Feature importance ranking"
      tabIndex={0}
    />
  );
}

function ModelFeatureImportanceTab({
  activeProjectId,
  model,
  analysis
}: {
  activeProjectId: string;
  model: ModelManifest;
  analysis: ModelAnalysis;
}) {
  const queryClient = useQueryClient();
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const fi = analysis.feature_importance ?? null;
  const compute = useMutation({
    mutationFn: () => api.runModelFeatureImportance(activeProjectId, model.id, { algorithm: "supervised_fast", n_iterations: 3 }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "models", model.id, "analysis"] });
    }
  });

  const running = fi?.status === "running" || compute.isPending;

  if (!fi && !running) {
    return (
      <div className="dataset-tab-panel">
        <div className="model-importance-intro">
          <p className="muted">
            Rank which input features drive this model&apos;s predictions. NEExT trains models over feature subsets, so this
            runs as a background job and can take a little while.
          </p>
          <button type="button" className="btn btn-primary" onClick={() => compute.mutate()} disabled={compute.isPending}>
            <BarChart3 />
            {compute.isPending ? "Starting." : "Compute Feature Importance"}
          </button>
          {compute.error ? <p className="table-error">{(compute.error as Error).message}</p> : null}
        </div>
      </div>
    );
  }

  if (running) {
    return (
      <div className="dataset-tab-panel model-importance-running">
        <RunningState
          label="Computing feature importance…"
          detail="Training models over feature subsets (background job)"
        />
      </div>
    );
  }

  if (fi?.status === "failed") {
    return (
      <div className="dataset-tab-panel">
        <p className="table-error">{fi.error || "Feature importance failed."}</p>
        <button type="button" className="btn" onClick={() => compute.mutate()} disabled={compute.isPending}>
          <RotateCcw />
          Retry
        </button>
      </div>
    );
  }

  const ranking = fi?.ranking ?? [];
  const scoreLabel = fi?.score_label || "score";
  const orderedRanking = [...ranking].sort((a, b) => a.rank - b.rank);
  return (
    <div className="dataset-tab-panel">
      <ChartCard
        className="is-fill"
        title="Feature importance"
        chartRef={chartRef}
        exportName={`${model.name} feature importance`}
        subtitle={`${fi?.algorithm} · embedding ${fi?.embedding_algorithm} · ${ranking.length} features ranked`}
        controls={
          <button type="button" className="btn" onClick={() => compute.mutate()} disabled={compute.isPending}>
            <RotateCcw />
            {compute.isPending ? "Starting…" : "Recompute"}
          </button>
        }
      >
        <ModelFeatureImportanceChart ranking={ranking} scoreLabel={scoreLabel} chartRef={chartRef} />
      </ChartCard>
      <div className="artifact-table-scroll model-fi-table">
        <table className="tbl">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Feature</th>
              <th>{scoreLabel}</th>
            </tr>
          </thead>
          <tbody>
            {orderedRanking.map((item) => (
              <tr key={item.feature_name}>
                <td>{item.rank + 1}</td>
                <td>
                  <strong>{item.feature_name}</strong>
                </td>
                <td>{item.score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModelOverviewTab({ analysis }: { analysis: ModelAnalysis }) {
  return (
    <div className="dataset-tab-panel">
      <div className="stat-grid">
        <div className="stat-tile">
          <span>Graphs</span>
          <strong>{formatCount(analysis.output_stats.graph_count)}</strong>
          <small>{analysis.source_dataset.name}</small>
        </div>
        <div className="stat-tile">
          <span>Features</span>
          <strong>{formatCount(analysis.output_stats.feature_count)}</strong>
          <small>{formatCount(analysis.feature_columns.length)} columns</small>
        </div>
        <div className="stat-tile">
          <span>Iterations</span>
          <strong>{formatCount(analysis.output_stats.sample_size)}</strong>
          <small>test size {formatValue(analysis.test_size)}</small>
        </div>
        <div className="stat-tile">
          <span>Task</span>
          <strong>{taskLabel(analysis.task_type)}</strong>
          <small>{analysis.expected_metrics.map(metricLabel).join(", ")}</small>
        </div>
        <div className="stat-tile">
          <span>Algorithm</span>
          <strong>{analysis.algorithm.name}</strong>
          <small>{analysis.algorithm.id}</small>
        </div>
        <div className="stat-tile">
          <span>Embeddings</span>
          <strong>{formatCount(analysis.source_embeddings.length)}</strong>
          <small>{analysis.source_embeddings.map((embedding) => embedding.name).join(", ")}</small>
        </div>
      </div>
      <div className="dataset-detail-grid">
        <section>
          <h3>Metric Summary</h3>
          <table className="tbl compact-tbl">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
              </tr>
            </thead>
            <tbody>
              {analysis.expected_metrics.map((metricName) => (
                <tr key={metricName}>
                  <td>{metricLabel(metricName)}</td>
                  <td>{formatMetric(analysis.summary[`${metricName}_mean`])}</td>
                  <td>{formatMetric(analysis.summary[`${metricName}_std`])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section>
          <h3>Sources</h3>
          <table className="tbl compact-tbl">
            <thead>
              <tr>
                <th>Kind</th>
                <th>Name</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Dataset</td>
                <td>{analysis.source_dataset.name}</td>
                <td>{analysis.source_dataset.status}</td>
              </tr>
              {analysis.source_embeddings.map((embedding) => (
                <tr key={embedding.id}>
                  <td>Embedding</td>
                  <td>{embedding.name}</td>
                  <td>{embedding.status}</td>
                </tr>
              ))}
              {analysis.source_features.map((feature) => (
                <tr key={feature.id}>
                  <td>Feature</td>
                  <td>{feature.name}</td>
                  <td>{feature.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section>
          <h3>Classes</h3>
          {analysis.classes?.length ? <p>{analysis.classes.join(", ")}</p> : <p className="muted">No classes for this task.</p>}
        </section>
      </div>
    </div>
  );
}

function ModelMetricsTab({
  analysis,
  selectedIteration,
  onSelectIteration
}: {
  analysis: ModelAnalysis;
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
}) {
  if (analysis.metric_series.every((series) => series.points.length === 0)) {
    return (
      <div className="dataset-tab-panel">
        <div className="artifact-table-empty">
          <EmptyState compact>No metric rows are available.</EmptyState>
        </div>
      </div>
    );
  }

  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  return (
    <div className="dataset-tab-panel graph-tab-panel">
      <ChartCard
        className="is-fill"
        title="Metric results across iterations"
        chartRef={chartRef}
        exportName={`${analysis.model_name} metrics`}
        subtitle={
          <div className="chart-card-chips">
            {analysis.expected_metrics.map((metricName) => (
              <span className="metric-chip" key={metricName}>
                {metricLabel(metricName)} {formatMetric(analysis.summary[`${metricName}_mean`])} ±{" "}
                {formatMetric(analysis.summary[`${metricName}_std`])}
              </span>
            ))}
          </div>
        }
        controls={
          <span className="muted dataset-page-count">
            {formatCount(analysis.metrics.length)} iterations
            {selectedIteration == null ? "" : ` · iteration ${selectedIteration} selected`}
          </span>
        }
      >
        <ModelMetricsChart
          analysis={analysis}
          selectedIteration={selectedIteration}
          onSelectIteration={onSelectIteration}
          chartRef={chartRef}
        />
      </ChartCard>
    </div>
  );
}

function ModelDataTab({
  activeProjectId,
  model,
  analysis,
  selectedIteration,
  onSelectIteration
}: {
  activeProjectId: string;
  model: ModelManifest;
  analysis: ModelAnalysis;
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
}) {
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");

  const downloadMetrics = async () => {
    setExportError("");
    setExporting(true);
    try {
      const download = await api.modelMetricsExport(activeProjectId, model.id);
      const objectUrl = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = download.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="dataset-tab-panel model-data-panel">
      <div className="model-data-toolbar">
        <span className="muted dataset-page-count">
          {formatCount(analysis.metrics.length)} iterations · {formatCount(analysis.expected_metrics.length)} metrics
        </span>
        <button type="button" className="btn" onClick={downloadMetrics} disabled={exporting}>
          <Download />
          {exporting ? "Exporting…" : "Download CSV"}
        </button>
      </div>
      {exportError ? <p className="table-error">{exportError}</p> : null}

      <section className="model-data-section">
        <h3>Metrics by iteration</h3>
        <div className="model-data-table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Iteration</th>
                {analysis.expected_metrics.map((metricName) => (
                  <th key={metricName}>{metricLabel(metricName)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {analysis.metrics.map((row, rowIndex) => {
                const iteration = Number(row.iteration ?? rowIndex);
                return (
                  <tr
                    key={rowIndex}
                    className={selectedIteration === iteration ? "is-selected" : ""}
                    onClick={() => onSelectIteration(iteration)}
                  >
                    <td>{String(row.iteration ?? rowIndex)}</td>
                    {analysis.expected_metrics.map((metricName) => (
                      <td key={metricName}>{formatMetric(row[metricName])}</td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="model-data-section">
        <h3>Feature columns</h3>
        {analysis.feature_columns.length ? (
          <div className="model-data-table-wrap">
            <table className="tbl compact-tbl">
              <thead>
                <tr>
                  <th>Column</th>
                </tr>
              </thead>
              <tbody>
                {analysis.feature_columns.map((column) => (
                  <tr key={column}>
                    <td>{column}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No feature columns.</p>
        )}
      </section>

      <section className="model-data-section">
        <h3>Classes</h3>
        {analysis.classes?.length ? <p>{analysis.classes.join(", ")}</p> : <p className="muted">No classes for this task.</p>}
      </section>
    </div>
  );
}

export function ModelExploreView({
  activeProjectId,
  models,
  embeddings,
  features,
  datasets,
  catalog,
  loading,
  selectedModelId,
  exploreModelId,
  selectedIteration,
  onExploreModel,
  onClearExploreModel,
  onSelectIteration
}: ModelExploreViewProps) {
  const [tab, setTab] = useState<"overview" | "metrics" | "importance" | "data">("overview");
  const embeddingsById = useMemo(() => new Map(embeddings.map((embedding) => [embedding.id, embedding])), [embeddings]);
  const featuresById = useMemo(() => new Map(features.map((feature) => [feature.id, feature])), [features]);
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const model = useMemo(
    () => models.find((item) => item.id === exploreModelId) || models.find((item) => item.id === selectedModelId),
    [exploreModelId, models, selectedModelId]
  );

  useEffect(() => {
    setTab("overview");
    onSelectIteration(null);
  }, [model?.id, onSelectIteration]);

  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "models", model?.id, "analysis"],
    queryFn: () => api.modelAnalysis(activeProjectId, model!.id),
    enabled: Boolean(activeProjectId && model?.id && model.status === "completed"),
    refetchInterval: (query) => (query.state.data?.feature_importance?.status === "running" ? 1500 : false)
  });

  if (!activeProjectId) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>No active project.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              Model Explore
            </span>
            <span className="muted">{loading ? "Loading" : `${models.length} models`}</span>
          </header>
          {loading ? (
            <div className="artifact-table-empty">
              <EmptyState compact>Loading models.</EmptyState>
            </div>
          ) : models.length === 0 ? (
            <div className="artifact-table-empty">
              <EmptyState compact>No models.</EmptyState>
            </div>
          ) : (
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Dataset</th>
                    <th>Algorithm</th>
                    <th>Task</th>
                    <th>Embeddings</th>
                    <th>Status</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((item) => {
                    const sourceEmbeddingIds = modelEmbeddingIds(item);
                    const sourceEmbeddings = sourceEmbeddingIds.map((embeddingId) => embeddingsById.get(embeddingId)).filter(Boolean) as EmbeddingManifest[];
                    const datasetName = datasetsById.get(modelDatasetId(item, embeddings, features))?.name || "Unknown dataset";
                    const embeddingNames = sourceEmbeddings.map((embedding) => embedding.name).join(", ") || `${sourceEmbeddingIds.length} embeddings`;
                    const algorithm = algorithmLabel(catalogById.get(item.source_model_id), item.source_model_id);
                    return (
                      <tr key={item.id} onClick={() => onExploreModel(item.id)}>
                        <td>
                          <span className="table-name-with-icon">
                            <BarChart3 />
                            <strong>{item.name}</strong>
                          </span>
                        </td>
                        <td>{datasetName}</td>
                        <td>{algorithm}</td>
                        <td>{taskLabel(String(item.operation.params.task_type))}</td>
                        <td>{embeddingNames}</td>
                        <td>
                          <span className={`status-pill ${artifactStatusClass(item.status)}`}>{artifactStatusLabel(item.status)}</span>
                        </td>
                        <td className="actions-cell actions-cell-wide">
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              onExploreModel(item.id);
                            }}
                          >
                            <Eye />
                            {item.status === "completed" ? "Explore" : "Train First"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    );
  }

  if (model.status !== "completed") {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <button type="button" className="btn" onClick={onClearExploreModel}>
                <ArrowLeft />
                Choose Model
              </button>
            </span>
            <span className="explore-title">{model.name}</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Train this model before exploring it.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="workflow workflow-fill">
      <section className="artifact-table dataset-explore model-explore">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <button type="button" className="btn" onClick={onClearExploreModel}>
              <ArrowLeft />
              Choose Model
            </button>
          </span>
          <span className="explore-title">{model.name}</span>
        </header>
        <div className="tab-strip">
          {(["overview", "metrics", "importance", "data"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={`tab-btn ${tab === item ? "is-active" : ""}`}
              onClick={() => setTab(item)}
            >
              {item === "overview"
                ? "Overview"
                : item === "metrics"
                  ? "Metrics"
                  : item === "importance"
                    ? "Feature Importance"
                    : "Data"}
            </button>
          ))}
        </div>
        {analysis.error ? <p className="table-error">{analysis.error.message}</p> : null}
        {analysis.isLoading || !analysis.data ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading analysis.</EmptyState>
          </div>
        ) : null}
        {tab === "overview" && analysis.data ? <ModelOverviewTab analysis={analysis.data} /> : null}
        {tab === "metrics" && analysis.data ? (
          <ModelMetricsTab analysis={analysis.data} selectedIteration={selectedIteration} onSelectIteration={onSelectIteration} />
        ) : null}
        {tab === "importance" && analysis.data ? (
          <ModelFeatureImportanceTab activeProjectId={activeProjectId} model={model} analysis={analysis.data} />
        ) : null}
        {tab === "data" && analysis.data ? (
          <ModelDataTab
            activeProjectId={activeProjectId}
            model={model}
            analysis={analysis.data}
            selectedIteration={selectedIteration}
            onSelectIteration={onSelectIteration}
          />
        ) : null}
      </section>
    </div>
  );
}
