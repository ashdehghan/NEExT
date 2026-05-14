import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BarChart3, Eye, Play, RotateCcw, Save, Settings2 } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type EmbeddingManifest,
  type FeatureManifest,
  type ModelCatalogEntry,
  type ModelCreatePayload,
  type ModelManifest
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface ModelLibraryViewProps {
  activeProjectId: string;
  catalog: ModelCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureModelViewProps {
  activeProjectId: string;
  model?: ModelCatalogEntry;
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  loading: boolean;
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
}

interface ModelPreviewViewProps {
  activeProjectId: string;
  model?: ModelManifest;
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

export function ModelLibraryView({
  activeProjectId,
  catalog,
  loading,
  selectedCatalogId,
  onSelectCatalog,
  onConfigure
}: ModelLibraryViewProps) {
  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="library" size={16} />
            Model Library - {catalog.length} {catalog.length === 1 ? "algorithm" : "algorithms"}
          </span>
          <span className="muted">{activeProjectId ? "Project target active" : "No active project"}</span>
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
                        <Settings2 />
                        Configure
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
  datasets,
  loading,
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
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);

  useEffect(() => {
    setSelectedEmbeddingIds([]);
    setTaskType("classifier");
    setSampleSize(5);
    setTestSize(0.3);
    setBalanceDataset(false);
    setNJobs(1);
    setParallelBackend("thread");
  }, [activeProjectId, model?.id]);

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
  const selectedDatasetIds = useMemo(
    () => new Set(selectedEmbeddings.map((embedding) => embeddingDatasetId(embedding, features)).filter(Boolean)),
    [features, selectedEmbeddings]
  );
  const selectedDataset = selectedDatasetIds.size === 1 ? datasetsById.get(Array.from(selectedDatasetIds)[0]) : undefined;
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
  const canSave = Boolean(activeProjectId && model && selectedEmbeddingIds.length > 0 && selectedDataset && paramsValid);
  const saveBlockedMessage = !activeProjectId
    ? "An active project is required."
    : !loading && embeddings.length === 0
      ? "An embedding artifact is required before saving."
      : selectedEmbeddingIds.length === 0
        ? "Select at least one embedding."
        : !selectedDataset
          ? "Selected embeddings must share one dataset."
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
          <h3>Configure {model.name}</h3>
          <p className="form-subtitle">{model.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createModel.error ? <p className="error-text">{createModel.error.message}</p> : null}
        {saveBlockedMessage ? <p className="muted form-note">{saveBlockedMessage}</p> : null}
        {selectedDataset ? <p className="muted form-note">Dataset: {selectedDataset.name}</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Algorithm</span>
            <input value={model.id} readOnly />
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
                  const datasetName = datasetsById.get(embeddingDatasetId(embedding, features))?.name || "Unknown dataset";
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
                      <td>{datasetName}</td>
                      <td>
                        <span className={`status-pill ${embedding.status === "completed" ? "is-ready" : "is-idle"}`}>{embedding.status}</span>
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
          <Save />
          {createModel.isPending ? "Saving" : "Save"}
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
  onPreviewModel
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
            Models - {models.length} {models.length === 1 ? "model" : "models"}
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
                Run Selected
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
                          <span className={`status-pill ${model.status === "completed" ? "is-ready" : "is-idle"}`}>{model.status}</span>
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
                              {model.status === "failed" ? "Retry" : isRunning ? "Running" : "Run"}
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
  if (metric === "rmse") return "RMSE";
  if (metric === "mae") return "MAE";
  return metric.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatMetric(value: unknown): string {
  return typeof value === "number" ? value.toFixed(4) : value == null ? "" : String(value);
}

export function ModelPreviewView({ activeProjectId, model }: ModelPreviewViewProps) {
  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "models", model?.id, "preview"],
    queryFn: () => api.modelPreview(activeProjectId, model!.id),
    enabled: Boolean(activeProjectId && model?.id && model.status === "completed")
  });
  const metricNames = model?.expected_output.metrics || [];

  if (!model) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a model.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="models" size={16} />
            {model.name} Results
          </span>
          <span className="muted">{model.output_stats ? `${model.output_stats.sample_size} iterations` : model.status}</span>
        </header>
        {preview.error ? <p className="table-error">{preview.error.message}</p> : null}
        {preview.isLoading || !preview.data ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading model results.</EmptyState>
          </div>
        ) : (
          <>
            <div className="table-toolbar">
              {metricNames.map((metricName) => (
                <span className="status-pill is-ready" key={metricName}>
                  {metricLabel(metricName)} {formatMetric(preview.data.summary[`${metricName}_mean`])} +/-{" "}
                  {formatMetric(preview.data.summary[`${metricName}_std`])}
                </span>
              ))}
              {model.output_stats ? <span className="muted">{model.output_stats.feature_count} features</span> : null}
              {model.output_stats ? <span className="muted">{model.output_stats.graph_count} graphs</span> : null}
            </div>
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Iteration</th>
                    {metricNames.map((metricName) => (
                      <th key={metricName}>{metricLabel(metricName)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.data.metrics.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      <td>{String(row.iteration ?? rowIndex)}</td>
                      {metricNames.map((metricName) => (
                        <td key={metricName}>{formatMetric(row[metricName])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-toolbar">
              <span className="muted">{preview.data.feature_columns.length} feature columns</span>
              {preview.data.classes?.length ? <span className="muted">Classes: {preview.data.classes.join(", ")}</span> : null}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
