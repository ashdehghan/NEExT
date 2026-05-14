import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, Eye, Play, RotateCcw, Save, Settings2 } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type EmbeddingCatalogEntry,
  type EmbeddingCreatePayload,
  type EmbeddingManifest,
  type FeatureManifest
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface EmbeddingLibraryViewProps {
  activeProjectId: string;
  catalog: EmbeddingCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureEmbeddingViewProps {
  activeProjectId: string;
  embedding?: EmbeddingCatalogEntry;
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  loading: boolean;
  onCreated: (embeddingId: string) => void;
}

interface ProjectEmbeddingsViewProps {
  activeProjectId: string;
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: EmbeddingCatalogEntry[];
  loading: boolean;
  selectedEmbeddingId: string;
  onSelectEmbedding: (embeddingId: string) => void;
  onPreviewEmbedding: (embeddingId: string) => void;
}

interface EmbeddingPreviewViewProps {
  activeProjectId: string;
  embedding?: EmbeddingManifest;
}

function featureDatasetId(feature: FeatureManifest): string {
  return feature.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset")?.artifact_id || "";
}

function embeddingFeatureIds(embedding: EmbeddingManifest): string[] {
  return embedding.inputs
    .filter((input) => input.role === "source_feature" && input.artifact_kind === "feature")
    .map((input) => input.artifact_id);
}

function algorithmLabel(entry?: EmbeddingCatalogEntry, fallback?: string): string {
  return entry?.name || fallback || "";
}

export function EmbeddingLibraryView({
  activeProjectId,
  catalog,
  loading,
  selectedCatalogId,
  onSelectCatalog,
  onConfigure
}: EmbeddingLibraryViewProps) {
  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="library" size={16} />
            Embedding Library · {catalog.length} {catalog.length === 1 ? "algorithm" : "algorithms"}
          </span>
          <span className="muted">{activeProjectId ? "Project target active" : "No active project"}</span>
        </header>
        {loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading embedding library.</EmptyState>
          </div>
        ) : catalog.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No embedding catalog entries.</EmptyState>
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

export function ConfigureEmbeddingView({
  activeProjectId,
  embedding,
  features,
  datasets,
  loading,
  onCreated
}: ConfigureEmbeddingViewProps) {
  const queryClient = useQueryClient();
  const [selectedFeatureIds, setSelectedFeatureIds] = useState<string[]>([]);
  const [embeddingDimension, setEmbeddingDimension] = useState(3);
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);

  useEffect(() => {
    setSelectedFeatureIds([]);
    setEmbeddingDimension(3);
  }, [activeProjectId, embedding?.id]);

  useEffect(() => {
    setSelectedFeatureIds((current) => current.filter((featureId) => features.some((feature) => feature.id === featureId)));
  }, [features]);

  function setFeatureSelected(featureId: string, selected: boolean) {
    setSelectedFeatureIds((current) => {
      const isSelected = current.includes(featureId);
      if (selected) return isSelected ? current : [...current, featureId];
      return isSelected ? current.filter((currentFeatureId) => currentFeatureId !== featureId) : current;
    });
  }

  const selectedFeatures = useMemo(
    () => selectedFeatureIds.map((featureId) => features.find((feature) => feature.id === featureId)).filter(Boolean) as FeatureManifest[],
    [features, selectedFeatureIds]
  );
  const selectedDatasetIds = useMemo(() => new Set(selectedFeatures.map(featureDatasetId).filter(Boolean)), [selectedFeatures]);
  const selectedDataset =
    selectedDatasetIds.size === 1 ? datasetsById.get(Array.from(selectedDatasetIds)[0]) : undefined;
  const paramsValid = Number.isInteger(embeddingDimension) && embeddingDimension >= 1 && embeddingDimension <= 128;
  const canSave = Boolean(activeProjectId && embedding && selectedFeatureIds.length > 0 && selectedDataset && paramsValid);
  const saveBlockedMessage = !activeProjectId
    ? "An active project is required."
    : !loading && features.length === 0
      ? "A feature artifact is required before saving."
      : selectedFeatureIds.length === 0
        ? "Select at least one feature."
        : !selectedDataset
          ? "Selected features must share one dataset."
          : !paramsValid
            ? "Embedding dimension must be 1-128."
            : "";

  const createEmbedding = useMutation({
    mutationFn: (payload: EmbeddingCreatePayload) => api.createEmbedding(activeProjectId, payload),
    onSuccess: (created) => {
      queryClient.setQueryData<EmbeddingManifest[]>(["projects", activeProjectId, "embeddings"], (current = []) => [
        created,
        ...current.filter((embeddingItem) => embeddingItem.id !== created.id)
      ]);
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
      onCreated(created.id);
    }
  });

  if (!embedding) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select an embedding library entry.</EmptyState>
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
        if (!canSave || createEmbedding.isPending) return;
        createEmbedding.mutate({
          source_embedding_id: embedding.id,
          source_feature_ids: selectedFeatureIds,
          params: {
            embedding_dimension: embeddingDimension
          }
        });
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="embeddings" size={32} />
        </span>
        <div>
          <h3>Configure {embedding.name}</h3>
          <p className="form-subtitle">{embedding.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createEmbedding.error ? <p className="error-text">{createEmbedding.error.message}</p> : null}
        {saveBlockedMessage ? <p className="muted form-note">{saveBlockedMessage}</p> : null}
        {selectedDataset ? <p className="muted form-note">Dataset: {selectedDataset.name}</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Algorithm</span>
            <input value={embedding.id} readOnly />
          </label>
          <label className="field">
            <span>Embedding Dimension</span>
            <input
              type="number"
              min={1}
              max={128}
              value={embeddingDimension}
              onChange={(event) => setEmbeddingDimension(Number(event.target.value))}
            />
          </label>
        </div>
        <div className="feature-picker" aria-label="Feature artifacts">
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
                  <td colSpan={4}>Loading features.</td>
                </tr>
              ) : features.length === 0 ? (
                <tr>
                  <td colSpan={4}>No features.</td>
                </tr>
              ) : (
                features.map((feature) => {
                  const isChecked = selectedFeatureIds.includes(feature.id);
                  return (
                    <tr
                      key={feature.id}
                      className={isChecked ? "is-selected" : ""}
                      onClick={() => setFeatureSelected(feature.id, !isChecked)}
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(event) => {
                            event.stopPropagation();
                            setFeatureSelected(feature.id, event.target.checked);
                          }}
                          onClick={(event) => event.stopPropagation()}
                        />
                      </td>
                      <td>
                        <strong>{feature.name}</strong>
                      </td>
                      <td>{datasetsById.get(featureDatasetId(feature))?.name || "Unknown dataset"}</td>
                      <td>
                        <span className={`status-pill ${feature.status === "completed" ? "is-ready" : "is-idle"}`}>{feature.status}</span>
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
        <button type="submit" className="btn btn-primary" disabled={!canSave || createEmbedding.isPending}>
          <Save />
          {createEmbedding.isPending ? "Saving" : "Save"}
        </button>
      </footer>
    </form>
  );
}

export function ProjectEmbeddingsView({
  activeProjectId,
  embeddings,
  features,
  datasets,
  catalog,
  loading,
  selectedEmbeddingId,
  onSelectEmbedding,
  onPreviewEmbedding
}: ProjectEmbeddingsViewProps) {
  const queryClient = useQueryClient();
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const featuresById = useMemo(() => new Map(features.map((feature) => [feature.id, feature])), [features]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const [checkedEmbeddingIds, setCheckedEmbeddingIds] = useState<string[]>([]);

  const runEmbedding = useMutation({
    mutationFn: (embeddingId: string) => api.runEmbedding(activeProjectId, embeddingId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  });
  const runBatch = useMutation({
    mutationFn: (embeddingIds: string[]) => api.runEmbeddingBatch(activeProjectId, embeddingIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "embeddings"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  });

  const runnableCheckedEmbeddingIds = checkedEmbeddingIds.filter((embeddingId) => {
    const embedding = embeddings.find((item) => item.id === embeddingId);
    return embedding?.status === "planned" || embedding?.status === "failed";
  });

  useEffect(() => {
    setCheckedEmbeddingIds((current) => current.filter((embeddingId) => embeddings.some((embedding) => embedding.id === embeddingId)));
  }, [embeddings]);

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="embeddings" size={16} />
            Embeddings · {embeddings.length} {embeddings.length === 1 ? "embedding" : "embeddings"}
          </span>
          <span className="muted">{activeProjectId ? "Active project" : "No active project"}</span>
        </header>
        {runEmbedding.error ? <p className="table-error">{runEmbedding.error.message}</p> : null}
        {runBatch.error ? <p className="table-error">{runBatch.error.message}</p> : null}
        {!activeProjectId ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No active project.</EmptyState>
          </div>
        ) : loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading embeddings.</EmptyState>
          </div>
        ) : embeddings.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No embeddings.</EmptyState>
          </div>
        ) : (
          <>
            <div className="table-toolbar">
              <button
                type="button"
                className="btn"
                disabled={runnableCheckedEmbeddingIds.length === 0 || runBatch.isPending}
                onClick={() => runBatch.mutate(runnableCheckedEmbeddingIds)}
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
                    <th>Features</th>
                    <th>Dimension</th>
                    <th>Status</th>
                    <th>Updated</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {embeddings.map((embedding) => {
                    const sourceFeatureIds = embeddingFeatureIds(embedding);
                    const sourceFeatures = sourceFeatureIds.map((featureId) => featuresById.get(featureId)).filter(Boolean) as FeatureManifest[];
                    const datasetId = sourceFeatures[0] ? featureDatasetId(sourceFeatures[0]) : "";
                    const datasetName = datasetsById.get(datasetId)?.name || "Unknown dataset";
                    const featureNames = sourceFeatures.map((feature) => feature.name).join(", ") || `${sourceFeatureIds.length} features`;
                    const algorithm = algorithmLabel(catalogById.get(embedding.source_embedding_id), embedding.source_embedding_id);
                    const isRunnable = embedding.status === "planned" || embedding.status === "failed";
                    const isRunning = embedding.status === "running" || (runEmbedding.isPending && runEmbedding.variables === embedding.id);
                    const isChecked = checkedEmbeddingIds.includes(embedding.id);
                    return (
                      <tr
                        key={embedding.id}
                        className={embedding.id === selectedEmbeddingId ? "is-selected" : ""}
                        onClick={() => onSelectEmbedding(embedding.id)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            checked={isChecked}
                            disabled={!isRunnable}
                            onChange={(event) => {
                              event.stopPropagation();
                              setCheckedEmbeddingIds((current) =>
                                event.target.checked
                                  ? current.includes(embedding.id)
                                    ? current
                                    : [...current, embedding.id]
                                  : current.filter((embeddingId) => embeddingId !== embedding.id)
                              );
                            }}
                            onClick={(event) => event.stopPropagation()}
                          />
                        </td>
                        <td>
                          <span className="table-name-with-icon">
                            <Box />
                            <strong>{embedding.name}</strong>
                          </span>
                        </td>
                        <td>{datasetName}</td>
                        <td>{algorithm}</td>
                        <td>{featureNames}</td>
                        <td>{String(embedding.operation.params.embedding_dimension)}</td>
                        <td>
                          <span className={`status-pill ${embedding.status === "completed" ? "is-ready" : "is-idle"}`}>{embedding.status}</span>
                        </td>
                        <td className="muted mono">{embedding.updated_at}</td>
                        <td className="actions-cell actions-cell-wide">
                          {isRunnable ? (
                            <button
                              type="button"
                              className="btn"
                              disabled={isRunning}
                              onClick={(event) => {
                                event.stopPropagation();
                                runEmbedding.mutate(embedding.id);
                              }}
                            >
                              {embedding.status === "failed" ? <RotateCcw /> : <Play />}
                              {embedding.status === "failed" ? "Retry" : isRunning ? "Running" : "Run"}
                            </button>
                          ) : null}
                          {embedding.status === "completed" ? (
                            <button
                              type="button"
                              className="btn"
                              onClick={(event) => {
                                event.stopPropagation();
                                onSelectEmbedding(embedding.id);
                                onPreviewEmbedding(embedding.id);
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

export function EmbeddingPreviewView({ activeProjectId, embedding }: EmbeddingPreviewViewProps) {
  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding?.id, "preview"],
    queryFn: () => api.embeddingPreview(activeProjectId, embedding!.id),
    enabled: Boolean(activeProjectId && embedding?.id && embedding.status === "completed")
  });

  if (!embedding) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select an embedding.</EmptyState>
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
            <FcIcon name="embeddings" size={16} />
            {embedding.name} Preview
          </span>
          <span className="muted">{embedding.output_stats ? `${embedding.output_stats.row_count} rows` : embedding.status}</span>
        </header>
        {preview.error ? <p className="table-error">{preview.error.message}</p> : null}
        {preview.isLoading || !preview.data ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading preview.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  {preview.data.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.data.rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {preview.data.columns.map((column) => (
                      <td key={column}>{row[column] == null ? "" : String(row[column])}</td>
                    ))}
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
