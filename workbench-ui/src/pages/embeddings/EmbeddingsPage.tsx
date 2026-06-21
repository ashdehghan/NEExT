import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Box, Eye, Play, Plus, RotateCcw, Trash2 } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type EmbeddingAnalysis,
  type EmbeddingCatalogEntry,
  type EmbeddingCreatePayload,
  type EmbeddingManifest,
  type FeatureManifest,
  type GnnArchitecture
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { AnalysisCommandCenter } from "../../components/viz/AnalysisCommandCenter";

const GNN_ARCHITECTURES: GnnArchitecture[] = ["GCN", "GraphSAGE", "GIN"];

interface EmbeddingLibraryViewProps {
  activeProjectId: string;
  catalog: EmbeddingCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  selectedDataset?: DatasetManifest;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureEmbeddingViewProps {
  activeProjectId: string;
  embedding?: EmbeddingCatalogEntry;
  features: FeatureManifest[];
  dataset?: DatasetManifest;
  loading: boolean;
  initialSelectedFeatureId?: string;
  draft?: Record<string, unknown>;
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
  onDeleteArtifact: (artifactKind: "embedding", artifactId: string) => void;
}

interface EmbeddingExploreViewProps {
  activeProjectId: string;
  embeddings: EmbeddingManifest[];
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: EmbeddingCatalogEntry[];
  loading: boolean;
  selectedEmbeddingId: string;
  exploreEmbeddingId: string;
  selectedGraphId: string;
  onExploreEmbedding: (embeddingId: string) => void;
  onClearExploreEmbedding: () => void;
  onSelectGraph: (graphId: string, visible: boolean | null) => void;
  onSelectedGraphVisibilityChange: (visible: boolean | null) => void;
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

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
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

function draftNumber(draft: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = draft?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function draftStringArray(draft: Record<string, unknown> | undefined, key: string): string[] | undefined {
  const value = draft?.[key];
  return Array.isArray(value) ? value.map(String).filter(Boolean) : undefined;
}

export function EmbeddingLibraryView({
  activeProjectId,
  catalog,
  loading,
  selectedCatalogId,
  selectedDataset,
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
          <span className="muted">{selectedDataset ? `Dataset: ${selectedDataset.name}` : activeProjectId ? "Select a dataset first" : "No active project"}</span>
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

export function ConfigureEmbeddingView({
  activeProjectId,
  embedding,
  features,
  loading,
  dataset,
  initialSelectedFeatureId = "",
  draft,
  onCreated
}: ConfigureEmbeddingViewProps) {
  const queryClient = useQueryClient();
  const [selectedFeatureIds, setSelectedFeatureIds] = useState<string[]>([]);
  const [embeddingDimension, setEmbeddingDimension] = useState(3);
  const [architecture, setArchitecture] = useState<GnnArchitecture>("GCN");
  const isGnn = embedding?.id === "gnn";
  const initialFeatureId = initialSelectedFeatureId && features.some((feature) => feature.id === initialSelectedFeatureId)
    ? initialSelectedFeatureId
    : "";

  useEffect(() => {
    setSelectedFeatureIds(initialFeatureId ? [initialFeatureId] : []);
    setEmbeddingDimension(3);
    setArchitecture("GCN");
  }, [activeProjectId, embedding?.id, dataset?.id, initialFeatureId]);

  useEffect(() => {
    if (!draft) return;
    const requestedFeatureIds = draftStringArray(draft, "source_feature_ids") || draftStringArray(draft, "feature_ids");
    if (requestedFeatureIds) {
      const validFeatureIds = requestedFeatureIds.filter((featureId) => features.some((feature) => feature.id === featureId));
      setSelectedFeatureIds(validFeatureIds);
    }
    const nextDimension = draftNumber(draft, "embedding_dimension");
    if (nextDimension !== undefined) setEmbeddingDimension(nextDimension);
    const nextArchitecture = draft["architecture"];
    if (GNN_ARCHITECTURES.includes(nextArchitecture as GnnArchitecture)) {
      setArchitecture(nextArchitecture as GnnArchitecture);
    }
  }, [activeProjectId, embedding?.id, dataset?.id, features, draft]);

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
  const paramsValid = Number.isInteger(embeddingDimension) && embeddingDimension >= 1 && embeddingDimension <= 128;
  const selectedFeaturesInDataset = selectedFeatures.every((feature) => dataset?.id && featureDatasetId(feature) === dataset.id);
  const canSave = Boolean(activeProjectId && embedding && dataset?.id && selectedFeatureIds.length > 0 && selectedFeaturesInDataset && paramsValid);
  const saveBlockedMessage = !activeProjectId
    ? "An active project is required."
    : !dataset
      ? "Select a dataset before configuring embeddings."
      : !loading && features.length === 0
        ? "A feature artifact is required before saving."
        : selectedFeatureIds.length === 0
          ? "Select at least one feature."
          : !selectedFeaturesInDataset
            ? "Selected features must belong to the active dataset."
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
            embedding_dimension: embeddingDimension,
            ...(isGnn ? { architecture } : {})
          }
        });
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="embeddings" size={32} />
        </span>
        <div>
          <h3>Add {embedding.name} to Project</h3>
          <p className="form-subtitle">{embedding.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createEmbedding.error ? <p className="error-text">{createEmbedding.error.message}</p> : null}
        {saveBlockedMessage ? <p className="muted form-note">{saveBlockedMessage}</p> : null}
        {dataset ? <p className="muted form-note">Dataset: {dataset.name}</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Algorithm</span>
            <span className="readonly-value mono">{embedding.id}</span>
          </label>
          {isGnn ? (
            <label className="field">
              <span>Architecture</span>
              <select value={architecture} onChange={(event) => setArchitecture(event.target.value as GnnArchitecture)}>
                {GNN_ARCHITECTURES.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
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
                      <td>{dataset?.name || "Unknown dataset"}</td>
                      <td>
                        <span className={`status-pill ${artifactStatusClass(feature.status)}`}>{artifactStatusLabel(feature.status)}</span>
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
          <Plus />
          {createEmbedding.isPending ? "Creating" : "Create Embedding"}
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
  onPreviewEmbedding,
  onDeleteArtifact
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
                Compute Selected
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
                          <span className={`status-pill ${artifactStatusClass(embedding.status)}`}>{artifactStatusLabel(embedding.status)}</span>
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
                              {embedding.status === "failed" ? "Retry Compute" : isRunning ? "Computing" : "Compute"}
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
                          <button
                            type="button"
                            className="icon-btn icon-btn-danger"
                            aria-label={`Delete ${embedding.name}`}
                            title={`Delete ${embedding.name}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteArtifact("embedding", embedding.id);
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

function EmbeddingPreviewTable({ preview }: { preview: { columns: string[]; rows: Record<string, unknown>[] } }) {
  return (
    <div className="artifact-table-scroll dataset-data-scroll">
      <table className="tbl">
        <thead>
          <tr>
            {preview.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {preview.columns.map((column) => (
                <td key={column}>{row[column] == null ? "" : String(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EmbeddingDataTab({ activeProjectId, embedding }: { activeProjectId: string; embedding: EmbeddingManifest }) {
  const [offset, setOffset] = useState(0);
  const pageSize = 50;

  useEffect(() => {
    setOffset(0);
  }, [embedding.id]);

  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding.id, "preview", pageSize, offset],
    queryFn: () => api.embeddingPreview(activeProjectId, embedding.id, pageSize, offset),
    enabled: Boolean(activeProjectId && embedding.id && embedding.status === "completed")
  });

  const totalRows = preview.data?.total_rows || 0;
  const pageStart = totalRows === 0 ? 0 : offset + 1;
  const pageEnd = preview.data ? Math.min(offset + preview.data.rows.length, totalRows) : 0;

  return (
    <div className="dataset-tab-panel">
      <div className="table-toolbar dataset-table-toolbar">
        <span className="muted dataset-page-count">
          {pageStart}-{pageEnd} of {formatCount(totalRows)}
        </span>
        <span className="toolbar-spacer" />
        <button type="button" className="btn" onClick={() => setOffset(Math.max(0, offset - pageSize))} disabled={offset === 0}>
          Previous
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => setOffset(offset + pageSize)}
          disabled={!preview.data || offset + pageSize >= preview.data.total_rows}
        >
          Next
        </button>
      </div>
      {preview.error ? <p className="table-error">{preview.error.message}</p> : null}
      {preview.isLoading || !preview.data ? (
        <div className="artifact-table-empty">
          <EmptyState compact>Loading table.</EmptyState>
        </div>
      ) : (
        <EmbeddingPreviewTable preview={preview.data} />
      )}
    </div>
  );
}

function EmbeddingStatisticsTab({ analysis }: { analysis: EmbeddingAnalysis }) {
  return (
    <div className="dataset-tab-panel">
      <div className="stat-grid">
        <div className="stat-tile">
          <span>Graphs</span>
          <strong>{formatCount(analysis.output_stats.row_count)}</strong>
          <small>{formatCount(analysis.output_stats.column_count)} output columns</small>
        </div>
        <div className="stat-tile">
          <span>Dimensions</span>
          <strong>{formatCount(analysis.embedding_columns.length)}</strong>
          <small>{formatCount(analysis.numeric_embedding_columns.length)} numeric</small>
        </div>
        <div className="stat-tile">
          <span>Source Dataset</span>
          <strong>{analysis.source_dataset.name}</strong>
          <small>{analysis.source_dataset.status}</small>
        </div>
        <div className="stat-tile">
          <span>Source Features</span>
          <strong>{formatCount(analysis.source_features.length)}</strong>
          <small>{analysis.source_features.map((feature) => feature.name).join(", ")}</small>
        </div>
        <div className="stat-tile">
          <span>Algorithm</span>
          <strong>{analysis.algorithm.name}</strong>
          <small>{analysis.algorithm.id}</small>
        </div>
        <div className="stat-tile">
          <span>Columns</span>
          <strong>{formatCount(analysis.embedding_columns.length)}</strong>
          <small>{analysis.embedding_columns.join(", ")}</small>
        </div>
      </div>
      <div className="dataset-detail-grid">
        <section>
          <h3>Graph Labels</h3>
          {Object.keys(analysis.graph_label_distribution).length ? (
            <table className="tbl compact-tbl">
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Graphs</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(analysis.graph_label_distribution).map(([label, count]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{formatCount(count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No graph labels.</p>
          )}
        </section>
        <section>
          <h3>Numeric Summaries</h3>
          <table className="tbl compact-tbl">
            <thead>
              <tr>
                <th>Column</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Nulls</th>
              </tr>
            </thead>
            <tbody>
              {analysis.column_summaries.map((summary) => (
                <tr key={summary.column}>
                  <td>{summary.column}</td>
                  <td>{formatValue(summary.min)}</td>
                  <td>{formatValue(summary.max)}</td>
                  <td>{formatValue(summary.mean)}</td>
                  <td>{formatValue(summary.std)}</td>
                  <td>{formatCount(summary.null_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}

function EmbeddingAnalysisTab({
  activeProjectId,
  embedding,
  hasLabels
}: {
  activeProjectId: string;
  embedding: EmbeddingManifest;
  hasLabels: boolean;
}) {
  return (
    <AnalysisCommandCenter
      analyze={(params) => api.embeddingAnalysis(activeProjectId, embedding.id, params)}
      queryKeyBase={["projects", activeProjectId, "embeddings", embedding.id, "analysis"]}
      exportName={embedding.name}
      hasLabels={hasLabels}
    />
  );
}

export function EmbeddingExploreView({
  activeProjectId,
  embeddings,
  features,
  datasets,
  catalog,
  loading,
  selectedEmbeddingId,
  exploreEmbeddingId,
  selectedGraphId,
  onExploreEmbedding,
  onClearExploreEmbedding,
  onSelectGraph,
  onSelectedGraphVisibilityChange
}: EmbeddingExploreViewProps) {
  const [tab, setTab] = useState<"statistics" | "analysis" | "data">("statistics");
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const featuresById = useMemo(() => new Map(features.map((feature) => [feature.id, feature])), [features]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const embedding = useMemo(
    () => embeddings.find((item) => item.id === exploreEmbeddingId) || embeddings.find((item) => item.id === selectedEmbeddingId),
    [embeddings, exploreEmbeddingId, selectedEmbeddingId]
  );

  useEffect(() => {
    setTab("statistics");
    onSelectGraph("", null);
    onSelectedGraphVisibilityChange(null);
  }, [embedding?.id, onSelectGraph, onSelectedGraphVisibilityChange]);

  // Base analysis (default PCA, no clustering) feeds Statistics/Data and tells us whether labels
  // exist; the Analysis tab's cards each fetch their own projection/clustering.
  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding?.id, "analysis", "base"],
    queryFn: () => api.embeddingAnalysis(activeProjectId, embedding!.id),
    enabled: Boolean(activeProjectId && embedding?.id && embedding.status === "completed")
  });

  const hasLabels = Object.keys(analysis.data?.graph_label_distribution ?? {}).length > 0;

  useEffect(() => {
    if (tab === "analysis" && analysis.data && !analysis.data.pca.available) {
      setTab("statistics");
    }
  }, [analysis.data, tab]);

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

  if (!embedding) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              Embedding Explore
            </span>
            <span className="muted">{loading ? "Loading" : `${embeddings.length} embeddings`}</span>
          </header>
          {loading ? (
            <div className="artifact-table-empty">
              <EmptyState compact>Loading embeddings.</EmptyState>
            </div>
          ) : embeddings.length === 0 ? (
            <div className="artifact-table-empty">
              <EmptyState compact>No embeddings.</EmptyState>
            </div>
          ) : (
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Dataset</th>
                    <th>Algorithm</th>
                    <th>Features</th>
                    <th>Status</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {embeddings.map((item) => {
                    const sourceFeatureIds = embeddingFeatureIds(item);
                    const sourceFeatures = sourceFeatureIds.map((featureId) => featuresById.get(featureId)).filter(Boolean) as FeatureManifest[];
                    const datasetId = sourceFeatures[0] ? featureDatasetId(sourceFeatures[0]) : "";
                    const datasetName = datasetsById.get(datasetId)?.name || "Unknown dataset";
                    const featureNames = sourceFeatures.map((feature) => feature.name).join(", ") || `${sourceFeatureIds.length} features`;
                    const algorithm = algorithmLabel(catalogById.get(item.source_embedding_id), item.source_embedding_id);
                    return (
                      <tr key={item.id} onClick={() => onExploreEmbedding(item.id)}>
                        <td>
                          <span className="table-name-with-icon">
                            <Box />
                            <strong>{item.name}</strong>
                          </span>
                        </td>
                        <td>{datasetName}</td>
                        <td>{algorithm}</td>
                        <td>{featureNames}</td>
                        <td>
                          <span className={`status-pill ${artifactStatusClass(item.status)}`}>{artifactStatusLabel(item.status)}</span>
                        </td>
                        <td className="actions-cell actions-cell-wide">
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              onExploreEmbedding(item.id);
                            }}
                          >
                            <Eye />
                            {item.status === "completed" ? "Explore" : "Compute First"}
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

  if (embedding.status !== "completed") {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <button type="button" className="btn" onClick={onClearExploreEmbedding}>
                <ArrowLeft />
                Choose Embedding
              </button>
            </span>
            <span className="explore-title">{embedding.name}</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Compute this embedding before exploring it.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="workflow workflow-fill">
      <section className="artifact-table dataset-explore embedding-explore">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <button type="button" className="btn" onClick={onClearExploreEmbedding}>
              <ArrowLeft />
              Choose Embedding
            </button>
          </span>
          <span className="explore-title">{embedding.name}</span>
        </header>
        <div className="tab-strip">
          {(["statistics", "analysis", "data"] as const).map((item) => {
            const analysisDisabled = item === "analysis" && Boolean(analysis.data && !analysis.data.pca.available);
            return (
              <button
                key={item}
                type="button"
                className={`tab-btn ${tab === item ? "is-active" : ""}`}
                onClick={() => setTab(item)}
                disabled={analysisDisabled}
                title={analysisDisabled ? analysis.data?.pca.reason || "Analysis is unavailable for this embedding." : undefined}
              >
                {item === "statistics" ? "Statistics" : item === "analysis" ? "Analysis" : "Data"}
              </button>
            );
          })}
        </div>
        {analysis.error ? <p className="table-error">{analysis.error.message}</p> : null}
        {tab !== "data" && (analysis.isLoading || !analysis.data) ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading analysis.</EmptyState>
          </div>
        ) : null}
        {tab === "statistics" && analysis.data ? <EmbeddingStatisticsTab analysis={analysis.data} /> : null}
        {tab === "analysis" && analysis.data ? (
          <EmbeddingAnalysisTab activeProjectId={activeProjectId} embedding={embedding} hasLabels={hasLabels} />
        ) : null}
        {tab === "data" ? <EmbeddingDataTab activeProjectId={activeProjectId} embedding={embedding} /> : null}
      </section>
    </div>
  );
}
