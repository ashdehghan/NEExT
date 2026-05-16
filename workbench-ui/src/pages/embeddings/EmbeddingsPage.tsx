import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { ArrowLeft, Box, ChevronLeft, ChevronRight, Eye, Play, RotateCcw, Save, Search, Settings2 } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type EmbeddingAnalysis,
  type EmbeddingCatalogEntry,
  type EmbeddingCreatePayload,
  type EmbeddingGraphSearchResult,
  type EmbeddingManifest,
  type EmbeddingPcaPoint,
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

type EmbeddingPcaChartDatum = EmbeddingPcaPoint & {
  value: [number, number];
  itemStyle: { color: string };
};
type EmbeddingPcaChartElement = HTMLDivElement & {
  __embeddingPcaChart?: ReturnType<typeof echarts.init>;
};

function EmbeddingPcaChart({
  analysis,
  selectedGraphId,
  onSelectGraph
}: {
  analysis: EmbeddingAnalysis;
  selectedGraphId: string;
  onSelectGraph: (graphId: string, visible: boolean | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const onSelectGraphRef = useRef(onSelectGraph);
  const previousSelectedIndexRef = useRef<number | null>(null);

  useEffect(() => {
    onSelectGraphRef.current = onSelectGraph;
  }, [onSelectGraph]);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current);
    chartRef.current = chart;
    (containerRef.current as EmbeddingPcaChartElement).__embeddingPcaChart = chart;
    const handleClick = (params: { data?: unknown }) => {
      const data = params.data as EmbeddingPcaChartDatum | undefined;
      if (!data?.graph_id) return;
      onSelectGraphRef.current(String(data.graph_id), true);
    };
    chart.on("click", handleClick);
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(containerRef.current);
    return () => {
      chart.off("click", handleClick);
      resizeObserver.disconnect();
      chart.dispose();
      if (containerRef.current) delete (containerRef.current as EmbeddingPcaChartElement).__embeddingPcaChart;
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const palette = ["#176ea9", "#d86c1f", "#2d8754", "#8d5db8", "#a4513d", "#4f758b", "#6b7f2a", "#9a5b91"];
    const colorValues = Array.from(new Set(analysis.pca.points.map((point) => point.color_value)));
    const colorByValue = new Map(colorValues.map((value, index) => [value, palette[index % palette.length]]));
    const data: EmbeddingPcaChartDatum[] = analysis.pca.points.map((point) => ({
      ...point,
      value: [point.x, point.y],
      itemStyle: { color: colorByValue.get(point.color_value) || palette[0] }
    }));

    const option: EChartsOption = {
      animation: false,
      grid: { left: 44, right: 18, top: 22, bottom: 38 },
      xAxis: {
        type: "value",
        name: analysis.pca.x_axis_label,
        nameLocation: "middle",
        nameGap: 24,
        splitLine: { lineStyle: { color: "#dfe5e9" } }
      },
      yAxis: {
        type: "value",
        name: analysis.pca.y_axis_label,
        nameLocation: "middle",
        nameGap: 30,
        splitLine: { lineStyle: { color: "#dfe5e9" } }
      },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = Array.isArray(params) ? params[0] : params;
          const dataPoint = (item as { data?: EmbeddingPcaChartDatum }).data;
          if (!dataPoint) return "";
          return [
            `Graph ${dataPoint.graph_id}`,
            `Label ${formatValue(dataPoint.graph_label)}`,
            `${analysis.pca.x_axis_label} ${dataPoint.x.toFixed(4)}`,
            `${analysis.pca.y_axis_label} ${dataPoint.y.toFixed(4)}`
          ].join("<br/>");
        }
      },
      series: [
        {
          type: "scatter",
          data,
          symbolSize: 16,
          emphasis: {
            itemStyle: {
              borderColor: "#111820",
              borderWidth: 2
            }
          }
        }
      ]
    };

    chart.setOption(option, { notMerge: true });
    previousSelectedIndexRef.current = null;
  }, [analysis]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    if (previousSelectedIndexRef.current != null) {
      chart.dispatchAction({ type: "downplay", seriesIndex: 0, dataIndex: previousSelectedIndexRef.current });
      previousSelectedIndexRef.current = null;
    }
    if (!selectedGraphId) return;
    const selectedIndex = analysis.pca.points.findIndex((point) => point.graph_id === selectedGraphId);
    if (selectedIndex < 0) return;
    chart.dispatchAction({ type: "highlight", seriesIndex: 0, dataIndex: selectedIndex });
    previousSelectedIndexRef.current = selectedIndex;
  }, [analysis, selectedGraphId]);

  return (
    <div
      ref={containerRef}
      className="feature-pca-chart embedding-pca-chart"
      role="img"
      aria-label={`${analysis.embedding_name} ${analysis.pca.projection_method === "raw" ? "2D embedding plot" : "PCA"}`}
      tabIndex={0}
    />
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

function EmbeddingPcaTab({
  activeProjectId,
  embedding,
  analysis,
  selectedGraphId,
  onSelectGraph
}: {
  activeProjectId: string;
  embedding: EmbeddingManifest;
  analysis: EmbeddingAnalysis;
  selectedGraphId: string;
  onSelectGraph: (graphId: string, visible: boolean | null) => void;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const trimmedSearch = searchQuery.trim();
  const graphSearch = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding.id, "analysis", "search", trimmedSearch],
    queryFn: () => api.embeddingGraphSearch(activeProjectId, embedding.id, trimmedSearch, 25),
    enabled: Boolean(activeProjectId && embedding.id && trimmedSearch)
  });

  const selectSearchResult = (result: EmbeddingGraphSearchResult) => {
    onSelectGraph(result.graph_id, result.in_pca_sample);
  };

  const selectedResultIndex = graphSearch.data?.results.findIndex((result) => result.graph_id === selectedGraphId) ?? -1;
  const searchResultCount = graphSearch.data?.results.length || 0;

  const selectResultByIndex = (index: number) => {
    const result = graphSearch.data?.results[index];
    if (result) selectSearchResult(result);
  };

  if (!analysis.pca.available) {
    return (
      <div className="dataset-tab-panel">
        <div className="artifact-table-empty">
          <EmptyState compact>{analysis.pca.reason || "PCA is unavailable for this embedding."}</EmptyState>
        </div>
      </div>
    );
  }

  const selectedGraphOutsideSample = Boolean(selectedGraphId && !analysis.pca.points.some((point) => point.graph_id === selectedGraphId));
  const projectionLabel = analysis.pca.projection_method === "raw" ? "Direct 2D" : "PCA";
  const colorLabel = analysis.pca.color_by === "graph_label" ? "graph label" : "graph ID";
  const searchStatus = trimmedSearch
    ? graphSearch.data
      ? `${formatCount(graphSearch.data.total_matches)} ${graphSearch.data.total_matches === 1 ? "match" : "matches"}`
      : graphSearch.isLoading
        ? "Searching"
        : "Search results"
    : "Graph ID or label";

  return (
    <div className="dataset-tab-panel graph-tab-panel">
      <div className="feature-pca-control-band">
        <div className="feature-pca-nav-group" aria-label="Embedding search navigation">
          <button
            type="button"
            className="icon-btn graph-nav-btn"
            aria-label="Previous result"
            title="Previous result"
            onClick={() => selectResultByIndex(selectedResultIndex - 1)}
            disabled={searchResultCount === 0 || selectedResultIndex <= 0}
          >
            <ChevronLeft />
          </button>
          <button
            type="button"
            className="icon-btn graph-nav-btn"
            aria-label="Next result"
            title="Next result"
            onClick={() => selectResultByIndex(selectedResultIndex + 1)}
            disabled={searchResultCount === 0 || selectedResultIndex < 0 || selectedResultIndex >= searchResultCount - 1}
          >
            <ChevronRight />
          </button>
        </div>
        <label className="field graph-search-field feature-pca-search-field">
          <span>Search</span>
          <div className="graph-search-input">
            <Search />
            <input
              aria-label="Search embedding graph IDs and labels"
              value={searchQuery}
              placeholder="Graph ID or label"
              onChange={(event) => setSearchQuery(event.target.value)}
            />
          </div>
        </label>
        <div className="feature-pca-status-group">
          <span className="status-pill is-ready">{projectionLabel}</span>
          {analysis.pca.sampled ? <span className="status-pill is-idle">sampled</span> : null}
          <span className="muted dataset-page-count">{searchStatus}</span>
        </div>
        <div className="feature-pca-meta-group">
          <span className="muted dataset-page-count">
            {formatCount(analysis.pca.point_count)} plotted of {formatCount(analysis.pca.total_graphs)} graphs
          </span>
          <span className="muted dataset-page-count">color by {colorLabel}</span>
        </div>
      </div>
      {trimmedSearch ? (
        <div className="graph-search-results" role="listbox" aria-label="Embedding graph search results">
          {graphSearch.isLoading ? <span className="muted">Searching.</span> : null}
          {graphSearch.error ? <span className="table-error inline-error">{graphSearch.error.message}</span> : null}
          {graphSearch.data ? (
            <>
              <span className="muted">
                {formatCount(graphSearch.data.total_matches)} {graphSearch.data.total_matches === 1 ? "match" : "matches"}
              </span>
              {graphSearch.data.results.length ? (
                graphSearch.data.results.map((result) => (
                  <button
                    type="button"
                    key={`${result.kind}-${result.graph_id}`}
                    className={`graph-search-result ${result.graph_id === selectedGraphId ? "is-selected" : ""}`}
                    onClick={() => selectSearchResult(result)}
                  >
                    <span className="status-pill is-idle">{result.kind}</span>
                    <strong>{result.graph_id}</strong>
                    <span className="muted">
                      label {formatValue(result.graph_label)} · {result.in_pca_sample ? "plotted" : "not plotted"}
                    </span>
                  </button>
                ))
              ) : (
                <span className="muted">No graph matches.</span>
              )}
            </>
          ) : null}
        </div>
      ) : null}
      {analysis.pca.sampled ? (
        <p className="table-note">
          Showing {formatCount(analysis.pca.point_count)} plotted graphs
          {analysis.pca.projection_method === "pca" ? ` and fitting on ${formatCount(analysis.pca.fit_row_count)} graphs` : ""} (
          {analysis.pca.sample_reason}).
        </p>
      ) : null}
      {selectedGraphOutsideSample ? (
        <p className="table-note">
          Selected graph {selectedGraphId} is outside the plotted sample. Inspector details are shown in the Right Panel.
        </p>
      ) : null}
      <EmbeddingPcaChart analysis={analysis} selectedGraphId={selectedGraphId} onSelectGraph={onSelectGraph} />
    </div>
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
  const [tab, setTab] = useState<"statistics" | "pca" | "data">("statistics");
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

  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding?.id, "analysis"],
    queryFn: () => api.embeddingAnalysis(activeProjectId, embedding!.id),
    enabled: Boolean(activeProjectId && embedding?.id && embedding.status === "completed")
  });

  useEffect(() => {
    if (!selectedGraphId || !analysis.data) {
      onSelectedGraphVisibilityChange(null);
      return;
    }
    onSelectedGraphVisibilityChange(analysis.data.pca.points.some((point) => point.graph_id === selectedGraphId));
  }, [analysis.data, onSelectedGraphVisibilityChange, selectedGraphId]);

  useEffect(() => {
    if (tab === "pca" && analysis.data && !analysis.data.pca.available) {
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
                          <span className={`status-pill ${item.status === "completed" ? "is-ready" : "is-idle"}`}>{item.status}</span>
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
                            {item.status === "completed" ? "Explore" : "Run First"}
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
              <FcIcon name="explore" size={16} />
              {embedding.name} Explore
            </span>
            <div className="artifact-table-head-actions">
              <span className="muted">{embedding.status}</span>
              <button type="button" className="btn" onClick={onClearExploreEmbedding}>
                <ArrowLeft />
                Choose Embedding
              </button>
            </div>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Run embedding computation before exploring this embedding.</EmptyState>
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
            <FcIcon name="explore" size={16} />
            {embedding.name} Explore
          </span>
          <div className="artifact-table-head-actions">
            <span className="muted">{embedding.output_stats ? `${formatCount(embedding.output_stats.row_count)} graphs` : embedding.status}</span>
            <button type="button" className="btn" onClick={onClearExploreEmbedding}>
              <ArrowLeft />
              Choose Embedding
            </button>
          </div>
        </header>
        <div className="tab-strip">
          {(["statistics", "pca", "data"] as const).map((item) => {
            const pcaDisabled = item === "pca" && Boolean(analysis.data && !analysis.data.pca.available);
            return (
              <button
                key={item}
                type="button"
                className={`tab-btn ${tab === item ? "is-active" : ""}`}
                onClick={() => setTab(item)}
                disabled={pcaDisabled}
                title={pcaDisabled ? analysis.data?.pca.reason || "PCA is unavailable for this embedding." : undefined}
              >
                {item === "statistics" ? "Statistics" : item === "pca" ? "PCA" : "Data"}
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
        {tab === "pca" && analysis.data ? (
          <EmbeddingPcaTab
            activeProjectId={activeProjectId}
            embedding={embedding}
            analysis={analysis.data}
            selectedGraphId={selectedGraphId}
            onSelectGraph={onSelectGraph}
          />
        ) : null}
        {tab === "data" ? <EmbeddingDataTab activeProjectId={activeProjectId} embedding={embedding} /> : null}
      </section>
    </div>
  );
}
