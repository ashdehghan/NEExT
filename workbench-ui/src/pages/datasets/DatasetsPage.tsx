import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { ChevronLeft, ChevronRight, Database, Eye, Play, RotateCcw, Save, Search, Settings2 } from "lucide-react";
import {
  api,
  type DatasetCatalogEntry,
  type DatasetCreatePayload,
  type DatasetGraphSearchResult,
  type DatasetGraphSummary,
  type DatasetGraphVisual,
  type DatasetManifest,
  type DatasetPreviewTable,
  type TabularPreview
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface DatasetLibraryViewProps {
  activeProjectId: string;
  catalog: DatasetCatalogEntry[];
  datasets: DatasetManifest[];
  loading: boolean;
  selectedCatalogId: string;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureDatasetViewProps {
  activeProjectId: string;
  entry?: DatasetCatalogEntry;
  onCreated: (datasetId: string) => void;
}

interface ProjectDatasetsViewProps {
  activeProjectId: string;
  datasets: DatasetManifest[];
  loading: boolean;
  selectedDatasetId: string;
  onSelectDataset: (datasetId: string) => void;
  onPreviewDataset: (datasetId: string) => void;
}

interface DatasetExploreViewProps {
  activeProjectId: string;
  datasets: DatasetManifest[];
  loading: boolean;
  selectedDatasetId: string;
  exploreDatasetId: string;
  exploreGraphId: string;
  exploreNodeId: string;
  onExploreDataset: (datasetId: string) => void;
  onExploreGraphChange: (graphId: string, summary: DatasetGraphSummary | null, options?: { clearNode?: boolean }) => void;
  onExploreNodeChange: (nodeId: string) => void;
  onExploreNodeVisualStateChange: (visible: boolean | null) => void;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function catalogSize(entry: DatasetCatalogEntry): string {
  return `${formatCount(entry.graph_count)} graphs, ${formatCount(entry.node_count)} nodes, ${formatCount(entry.edge_count)} edges`;
}

function formatValue(value: unknown): string {
  if (value == null) return "None";
  return String(value);
}

function formatAverage(total: number, count: number): string {
  if (!count) return "0";
  return (total / count).toFixed(1);
}

const DATASET_TABLE_LABELS: Record<DatasetPreviewTable, string> = {
  nodes: "Nodes",
  edges: "Edges",
  graph_labels: "Graph Labels",
  node_features: "Node Features",
  edge_features: "Edge Features",
  node_mapping: "Node Mapping",
  graph_mapping: "Graph Mapping",
  mapping: "Node Mapping"
};

function availableDatasetTables(dataset: DatasetManifest): { id: DatasetPreviewTable; label: string }[] {
  const tables: { id: DatasetPreviewTable; label: string }[] = [
    { id: "nodes", label: DATASET_TABLE_LABELS.nodes },
    { id: "edges", label: DATASET_TABLE_LABELS.edges }
  ];
  if (dataset.prepared_data_files?.graph_labels) tables.push({ id: "graph_labels", label: DATASET_TABLE_LABELS.graph_labels });
  if (dataset.prepared_data_files?.node_features) tables.push({ id: "node_features", label: DATASET_TABLE_LABELS.node_features });
  if (dataset.prepared_data_files?.edge_features) tables.push({ id: "edge_features", label: DATASET_TABLE_LABELS.edge_features });
  if (dataset.mapping_files?.node_mapping) tables.push({ id: "node_mapping", label: DATASET_TABLE_LABELS.node_mapping });
  if (dataset.mapping_files?.graph_mapping) tables.push({ id: "graph_mapping", label: DATASET_TABLE_LABELS.graph_mapping });
  return tables;
}

export function DatasetLibraryView({
  activeProjectId,
  catalog,
  datasets,
  loading,
  selectedCatalogId,
  onSelectCatalog,
  onConfigure
}: DatasetLibraryViewProps) {
  const configuredCatalogIds = useMemo(() => new Set(datasets.map((dataset) => dataset.source_catalog_id)), [datasets]);

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="library" size={16} />
            Dataset Library · {catalog.length} {catalog.length === 1 ? "dataset" : "datasets"}
          </span>
          <span className="muted">{activeProjectId ? "Project target active" : "No active project"}</span>
        </header>
        {loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading dataset library.</EmptyState>
          </div>
        ) : catalog.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No catalog datasets.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Source</th>
                  <th>Size</th>
                  <th>Status</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {catalog.map((entry) => {
                  const isConfigured = configuredCatalogIds.has(entry.id);
                  return (
                    <tr
                      key={entry.id}
                      className={entry.id === selectedCatalogId ? "is-selected" : ""}
                      onClick={() => onSelectCatalog(entry.id)}
                    >
                      <td>
                        <strong>{entry.name}</strong>
                      </td>
                      <td className="muted">{entry.source}</td>
                      <td>{catalogSize(entry)}</td>
                      <td>
                        <span className={`status-pill ${isConfigured ? "is-ready" : "is-idle"}`}>
                          {isConfigured ? "configured" : "available"}
                        </span>
                      </td>
                      <td className="actions-cell actions-cell-wide">
                        <button
                          type="button"
                          className="btn"
                          onClick={(event) => {
                            event.stopPropagation();
                            onSelectCatalog(entry.id);
                            onConfigure(entry.id);
                          }}
                          disabled={!activeProjectId}
                        >
                          <Settings2 />
                          Configure
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

export function ConfigureDatasetView({ activeProjectId, entry, onCreated }: ConfigureDatasetViewProps) {
  const queryClient = useQueryClient();
  const [graphType, setGraphType] = useState<"networkx" | "igraph">("networkx");
  const [filterLargestComponent, setFilterLargestComponent] = useState(true);

  const createDataset = useMutation({
    mutationFn: (payload: DatasetCreatePayload) => api.createDataset(activeProjectId, payload),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      onCreated(created.id);
    }
  });

  if (!entry) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a dataset library entry.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  const canSave = Boolean(activeProjectId && !createDataset.isPending);

  return (
    <form
      className="card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSave) return;
        createDataset.mutate({
          catalog_id: entry.id,
          params: {
            graph_type: graphType,
            filter_largest_component: filterLargestComponent
          }
        });
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="datasets" size={32} />
        </span>
        <div>
          <h3>Configure {entry.name}</h3>
          <p className="form-subtitle">{entry.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createDataset.error ? <p className="error-text">{createDataset.error.message}</p> : null}
        {!activeProjectId ? <p className="muted form-note">An active project is required.</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Graph Backend</span>
            <select value={graphType} onChange={(event) => setGraphType(event.target.value as "networkx" | "igraph")}>
              <option value="networkx">networkx</option>
              <option value="igraph">igraph</option>
            </select>
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={filterLargestComponent}
              onChange={(event) => setFilterLargestComponent(event.target.checked)}
            />
            <span>Filter Largest Component</span>
          </label>
          <label className="checkbox-field">
            <input type="checkbox" checked readOnly />
            <span>Reindex Nodes</span>
          </label>
        </div>
      </div>
      <footer className="card-foot">
        <button type="submit" className="btn btn-primary" disabled={!canSave}>
          <Save />
          {createDataset.isPending ? "Saving" : "Save"}
        </button>
      </footer>
    </form>
  );
}

export function ProjectDatasetsView({
  activeProjectId,
  datasets,
  loading,
  selectedDatasetId,
  onSelectDataset,
  onPreviewDataset
}: ProjectDatasetsViewProps) {
  const queryClient = useQueryClient();
  const runDataset = useMutation({
    mutationFn: (datasetId: string) => api.runDataset(activeProjectId, datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  });

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="datasets" size={16} />
            Datasets · {datasets.length} {datasets.length === 1 ? "dataset" : "datasets"}
          </span>
          <span className="muted">{activeProjectId ? "Active project" : "No active project"}</span>
        </header>
        {!activeProjectId ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No active project.</EmptyState>
          </div>
        ) : loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading datasets.</EmptyState>
          </div>
        ) : datasets.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No datasets.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Source</th>
                  <th>Backend</th>
                  <th>Filter</th>
                  <th>Status</th>
                  <th>Updated</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => {
                  const isRunnable = dataset.status === "planned" || dataset.status === "failed";
                  const isRunning = dataset.status === "running" || (runDataset.isPending && runDataset.variables === dataset.id);
                  return (
                    <tr
                      key={dataset.id}
                      className={dataset.id === selectedDatasetId ? "is-selected" : ""}
                      onClick={() => onSelectDataset(dataset.id)}
                    >
                      <td>
                        <span className="table-name-with-icon">
                          <Database />
                          <strong>{dataset.name}</strong>
                        </span>
                      </td>
                      <td className="muted">{dataset.source_catalog_id}</td>
                      <td>{String(dataset.operation.params.graph_type)}</td>
                      <td>{dataset.operation.params.filter_largest_component ? "Yes" : "No"}</td>
                      <td>
                        <span className={`status-pill ${dataset.status === "completed" ? "is-ready" : "is-idle"}`}>{dataset.status}</span>
                      </td>
                      <td className="muted mono">{dataset.updated_at}</td>
                      <td className="actions-cell actions-cell-wide">
                        {isRunnable ? (
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              runDataset.mutate(dataset.id);
                            }}
                            disabled={isRunning}
                          >
                            {dataset.status === "failed" ? <RotateCcw /> : <Play />}
                            {dataset.status === "failed" ? "Retry" : isRunning ? "Running" : "Run"}
                          </button>
                        ) : null}
                        {dataset.status === "completed" ? (
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              onSelectDataset(dataset.id);
                              onPreviewDataset(dataset.id);
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
        )}
      </section>
    </div>
  );
}

function DatasetGraphChart({
  visual,
  selectedNodeId,
  onSelectNode
}: {
  visual: DatasetGraphVisual;
  selectedNodeId: string;
  onSelectNode: (nodeId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const onSelectNodeRef = useRef(onSelectNode);
  const previousSelectedNodeRef = useRef("");

  useEffect(() => {
    onSelectNodeRef.current = onSelectNode;
  }, [onSelectNode]);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = echarts.init(containerRef.current);
    chartRef.current = chart;
    const handleClick = (params: { dataType?: string; data?: unknown; name?: string }) => {
      if (params.dataType !== "node") return;
      const data = params.data as { id?: string } | undefined;
      onSelectNodeRef.current(String(data?.id || params.name || ""));
    };
    chart.on("click", handleClick);
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(containerRef.current);

    return () => {
      chart.off("click", handleClick);
      resizeObserver.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const maxDegree = Math.max(1, ...visual.nodes.map((node) => node.degree));
    const option: EChartsOption = {
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const item = Array.isArray(params) ? params[0] : params;
          const data = item.data as { name?: string; value?: number };
          return data.name ? `${data.name}<br/>Degree: ${data.value ?? 0}` : "";
        }
      },
      series: [
        {
          type: "graph",
          layout: "force",
          roam: true,
          animation: false,
          label: {
            show: visual.nodes.length <= 40,
            position: "right",
            color: "#34424c",
            fontSize: 10
          },
          data: visual.nodes.map((node) => ({
            id: node.id,
            name: node.label,
            value: node.degree,
            symbolSize: 9 + (node.degree / maxDegree) * 18,
            itemStyle: { color: "#176ea9" }
          })),
          links: visual.edges.map((edge) => ({ source: edge.source, target: edge.target })),
          lineStyle: {
            color: "#8a98a6",
            opacity: 0.72,
            width: 1.2
          },
          emphasis: {
            focus: "adjacency",
            itemStyle: { color: "#d86c1f" },
            lineStyle: { width: 2 }
          },
          force: {
            repulsion: 80,
            edgeLength: [28, 90]
          }
        }
      ]
    };

    chart.setOption(option, { notMerge: true });
    previousSelectedNodeRef.current = "";
  }, [visual]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const previousSelectedNodeId = previousSelectedNodeRef.current;
    if (previousSelectedNodeId) {
      const previousIndex = visual.nodes.findIndex((node) => node.id === previousSelectedNodeId);
      if (previousIndex >= 0) {
        chart.dispatchAction({ type: "downplay", seriesIndex: 0, dataIndex: previousIndex });
      }
    }
    previousSelectedNodeRef.current = "";

    if (!selectedNodeId) return;
    const selectedIndex = visual.nodes.findIndex((node) => node.id === selectedNodeId);
    if (selectedIndex < 0) return;

    chart.dispatchAction({ type: "highlight", seriesIndex: 0, dataIndex: selectedIndex });
    previousSelectedNodeRef.current = selectedNodeId;
  }, [selectedNodeId, visual]);

  return <div ref={containerRef} className="dataset-graph-chart" role="img" aria-label={`Graph ${visual.graph_id}`} tabIndex={0} />;
}

function DatasetPreviewTable({ preview }: { preview: TabularPreview }) {
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

function DatasetDataTab({ activeProjectId, dataset }: { activeProjectId: string; dataset: DatasetManifest }) {
  const tables = useMemo(() => availableDatasetTables(dataset), [dataset]);
  const tableIds = useMemo(() => tables.map((table) => table.id).join(","), [tables]);
  const [table, setTable] = useState<DatasetPreviewTable>(tables[0]?.id || "nodes");
  const [offset, setOffset] = useState(0);
  const pageSize = 50;

  useEffect(() => {
    if (!tables.some((item) => item.id === table)) {
      setTable(tables[0]?.id || "nodes");
    }
    setOffset(0);
  }, [dataset.id, table, tableIds, tables]);

  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", dataset.id, "preview", table, pageSize, offset],
    queryFn: () => api.datasetPreview(activeProjectId, dataset.id, table, pageSize, offset),
    enabled: Boolean(activeProjectId && dataset.id && tables.length)
  });

  const totalRows = preview.data?.total_rows || 0;
  const pageStart = totalRows === 0 ? 0 : offset + 1;
  const pageEnd = preview.data ? Math.min(offset + preview.data.rows.length, totalRows) : 0;

  return (
    <div className="dataset-tab-panel">
      <div className="table-toolbar dataset-table-toolbar">
        <label className="field compact-field">
          <span>Table</span>
          <select
            aria-label="Dataset table"
            value={table}
            onChange={(event) => {
              setTable(event.target.value as DatasetPreviewTable);
              setOffset(0);
            }}
          >
            {tables.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
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
        <DatasetPreviewTable preview={preview.data} />
      )}
    </div>
  );
}

export function DatasetExploreView({
  activeProjectId,
  datasets,
  loading,
  selectedDatasetId,
  exploreDatasetId,
  exploreGraphId,
  exploreNodeId,
  onExploreDataset,
  onExploreGraphChange,
  onExploreNodeChange,
  onExploreNodeVisualStateChange
}: DatasetExploreViewProps) {
  const [tab, setTab] = useState<"statistics" | "graph" | "data">("statistics");
  const [graphSearchQuery, setGraphSearchQuery] = useState("");
  const dataset = useMemo(
    () => datasets.find((item) => item.id === exploreDatasetId) || datasets.find((item) => item.id === selectedDatasetId),
    [datasets, exploreDatasetId, selectedDatasetId]
  );

  useEffect(() => {
    setTab("statistics");
    setGraphSearchQuery("");
    onExploreGraphChange("", null);
    onExploreNodeVisualStateChange(null);
  }, [dataset?.id, onExploreGraphChange, onExploreNodeVisualStateChange]);

  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", dataset?.id, "analysis", exploreGraphId],
    queryFn: () =>
      api.datasetAnalysis(activeProjectId, dataset!.id, {
        graph_id: exploreGraphId || undefined,
        max_nodes: 150,
        max_edges: 300
      }),
    enabled: Boolean(activeProjectId && dataset?.id && dataset.status === "completed")
  });
  const trimmedGraphSearch = graphSearchQuery.trim();
  const graphSearch = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", dataset?.id, "analysis", "search", trimmedGraphSearch],
    queryFn: () => api.datasetGraphSearch(activeProjectId, dataset!.id, trimmedGraphSearch, 25),
    enabled: Boolean(activeProjectId && dataset?.id && dataset.status === "completed" && tab === "graph" && trimmedGraphSearch)
  });
  const graphSummaries = analysis.data?.graph_summaries || [];
  const graphCounts = graphSummaries.map((summary) => summary.node_count);
  const edgeCounts = graphSummaries.map((summary) => summary.edge_count);
  const selectedSummary = graphSummaries.find((summary) => summary.graph_id === analysis.data?.selected_graph_id) || null;
  const selectedGraphIndex = graphSummaries.findIndex((summary) => summary.graph_id === analysis.data?.selected_graph_id);
  const selectedNodeVisible = Boolean(exploreNodeId && analysis.data?.visual.nodes.some((node) => node.id === exploreNodeId));
  const selectedNodeOutsideSample = Boolean(exploreNodeId && analysis.data?.visual.sampled && !selectedNodeVisible);

  useEffect(() => {
    if (!analysis.data) return;
    onExploreGraphChange(analysis.data.selected_graph_id, selectedSummary, { clearNode: false });
  }, [analysis.data?.selected_graph_id, onExploreGraphChange, selectedSummary]);

  useEffect(() => {
    if (!exploreNodeId || !analysis.data) {
      onExploreNodeVisualStateChange(null);
      return;
    }
    onExploreNodeVisualStateChange(selectedNodeVisible);
  }, [analysis.data, exploreNodeId, onExploreNodeVisualStateChange, selectedNodeVisible]);

  const selectGraph = (summary: DatasetGraphSummary) => {
    onExploreGraphChange(summary.graph_id, summary);
  };

  const selectGraphByIndex = (index: number) => {
    const summary = graphSummaries[index];
    if (summary) selectGraph(summary);
  };

  const selectSearchResult = (result: DatasetGraphSearchResult) => {
    const summary = graphSummaries.find((item) => item.graph_id === result.graph_id) || null;
    onExploreGraphChange(result.graph_id, summary, { clearNode: result.kind === "graph" });
    if (result.kind === "node" && result.node_id) {
      onExploreNodeChange(result.node_id);
    }
  };

  const handleGraphKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
    if (event.key === "ArrowLeft" && selectedGraphIndex > 0) {
      event.preventDefault();
      selectGraphByIndex(selectedGraphIndex - 1);
    }
    if (event.key === "ArrowRight" && selectedGraphIndex >= 0 && selectedGraphIndex < graphSummaries.length - 1) {
      event.preventDefault();
      selectGraphByIndex(selectedGraphIndex + 1);
    }
  };

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

  if (!dataset) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              Dataset Explore
            </span>
            <span className="muted">{loading ? "Loading" : `${datasets.length} datasets`}</span>
          </header>
          {loading ? (
            <div className="artifact-table-empty">
              <EmptyState compact>Loading datasets.</EmptyState>
            </div>
          ) : datasets.length === 0 ? (
            <div className="artifact-table-empty">
              <EmptyState compact>No datasets.</EmptyState>
            </div>
          ) : (
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Source</th>
                    <th>Status</th>
                    <th>Updated</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((item) => (
                    <tr key={item.id} onClick={() => onExploreDataset(item.id)}>
                      <td>
                        <span className="table-name-with-icon">
                          <Database />
                          <strong>{item.name}</strong>
                        </span>
                      </td>
                      <td className="muted">{item.source_catalog_id}</td>
                      <td>
                        <span className={`status-pill ${item.status === "completed" ? "is-ready" : "is-idle"}`}>{item.status}</span>
                      </td>
                      <td className="muted mono">{item.updated_at}</td>
                      <td className="actions-cell actions-cell-wide">
                        <button
                          type="button"
                          className="btn"
                          onClick={(event) => {
                            event.stopPropagation();
                            onExploreDataset(item.id);
                          }}
                        >
                          <Eye />
                          Explore
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

  if (dataset.status !== "completed") {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              {dataset.name} Explore
            </span>
            <span className="muted">{dataset.status}</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Run dataset preparation before exploring this dataset.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="workflow workflow-fill">
      <section className="artifact-table dataset-explore">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="explore" size={16} />
            {dataset.name} Explore
          </span>
          <span className="muted">{dataset.prepared_stats ? `${formatCount(dataset.prepared_stats.node_count)} nodes` : dataset.status}</span>
        </header>
        <div className="tab-strip">
          {(["statistics", "graph", "data"] as const).map((item) => (
            <button key={item} type="button" className={`tab-btn ${tab === item ? "is-active" : ""}`} onClick={() => setTab(item)}>
              {item === "statistics" ? "Statistics" : item === "graph" ? "Graph" : "Data"}
            </button>
          ))}
        </div>
        {analysis.error ? <p className="table-error">{analysis.error.message}</p> : null}
        {tab !== "data" && (analysis.isLoading || !analysis.data) ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading analysis.</EmptyState>
          </div>
        ) : null}
        {tab === "statistics" && analysis.data ? (
          <div className="dataset-tab-panel">
            <div className="stat-grid">
              <div className="stat-tile">
                <span>Graphs</span>
                <strong>{formatCount(analysis.data.prepared_stats.graph_count)}</strong>
                <small>Source {formatCount(analysis.data.source_stats.graph_count)}</small>
              </div>
              <div className="stat-tile">
                <span>Nodes</span>
                <strong>{formatCount(analysis.data.prepared_stats.node_count)}</strong>
                <small>Source {formatCount(analysis.data.source_stats.node_count)}</small>
              </div>
              <div className="stat-tile">
                <span>Edges</span>
                <strong>{formatCount(analysis.data.prepared_stats.edge_count)}</strong>
                <small>Source {formatCount(analysis.data.source_stats.edge_count)}</small>
              </div>
              <div className="stat-tile">
                <span>Dropped Nodes</span>
                <strong>{formatCount(analysis.data.dropped_node_count)}</strong>
                <small>Preparation filter</small>
              </div>
              <div className="stat-tile">
                <span>Nodes / Graph</span>
                <strong>{formatAverage(analysis.data.prepared_stats.node_count, analysis.data.prepared_stats.graph_count)}</strong>
                <small>
                  {formatCount(Math.min(...graphCounts))} min / {formatCount(Math.max(...graphCounts))} max
                </small>
              </div>
              <div className="stat-tile">
                <span>Edges / Graph</span>
                <strong>{formatAverage(analysis.data.prepared_stats.edge_count, analysis.data.prepared_stats.graph_count)}</strong>
                <small>
                  {formatCount(Math.min(...edgeCounts))} min / {formatCount(Math.max(...edgeCounts))} max
                </small>
              </div>
            </div>
            <div className="dataset-detail-grid">
              <section>
                <h3>Graph Labels</h3>
                {Object.keys(analysis.data.graph_label_distribution).length ? (
                  <table className="tbl compact-tbl">
                    <thead>
                      <tr>
                        <th>Label</th>
                        <th>Graphs</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(analysis.data.graph_label_distribution).map(([label, count]) => (
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
                <h3>Feature Columns</h3>
                <p className="muted">Node: {analysis.data.node_feature_columns.length ? analysis.data.node_feature_columns.join(", ") : "None"}</p>
                <p className="muted">Edge: {analysis.data.edge_feature_columns.length ? analysis.data.edge_feature_columns.join(", ") : "None"}</p>
              </section>
            </div>
          </div>
        ) : null}
        {tab === "graph" && analysis.data ? (
          <div className="dataset-tab-panel graph-tab-panel" tabIndex={0} onKeyDown={handleGraphKeyDown} aria-label="Dataset graph view">
            <div className="table-toolbar dataset-table-toolbar dataset-graph-toolbar">
              <button
                type="button"
                className="icon-btn graph-nav-btn"
                aria-label="Previous graph"
                title="Previous graph"
                onClick={() => selectGraphByIndex(selectedGraphIndex - 1)}
                disabled={selectedGraphIndex <= 0}
              >
                <ChevronLeft />
              </button>
              <button
                type="button"
                className="icon-btn graph-nav-btn"
                aria-label="Next graph"
                title="Next graph"
                onClick={() => selectGraphByIndex(selectedGraphIndex + 1)}
                disabled={selectedGraphIndex < 0 || selectedGraphIndex >= graphSummaries.length - 1}
              >
                <ChevronRight />
              </button>
              <label className="field graph-search-field">
                <span>Search</span>
                <div className="graph-search-input">
                  <Search />
                  <input
                    aria-label="Search graphs and nodes"
                    value={graphSearchQuery}
                    placeholder="Graph or node ID"
                    onChange={(event) => setGraphSearchQuery(event.target.value)}
                  />
                </div>
              </label>
              {analysis.data.visual.sampled ? <span className="status-pill is-idle">sampled</span> : null}
              <span className="muted dataset-page-count">
                {selectedSummary
                  ? `Graph ${selectedGraphIndex + 1} of ${formatCount(graphSummaries.length)} · ${
                      selectedSummary.graph_id
                    } · ${formatCount(selectedSummary.node_count)} nodes · ${formatCount(selectedSummary.edge_count)} edges · label ${formatValue(
                      selectedSummary.graph_label
                    )}`
                  : null}
              </span>
            </div>
            {trimmedGraphSearch ? (
              <div className="graph-search-results" role="listbox" aria-label="Graph search results">
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
                          key={`${result.kind}-${result.graph_id}-${result.node_id || ""}`}
                          className={`graph-search-result ${
                            result.graph_id === analysis.data?.selected_graph_id && (!result.node_id || result.node_id === exploreNodeId)
                              ? "is-selected"
                              : ""
                          }`}
                          onClick={() => selectSearchResult(result)}
                        >
                          <span className="status-pill is-idle">{result.kind}</span>
                          <strong>{result.kind === "node" ? result.node_id : result.graph_id}</strong>
                          <span className="muted">
                            {result.kind === "node" ? `graph ${result.graph_id} · ` : ""}
                            {formatCount(result.node_count)} nodes · {formatCount(result.edge_count)} edges · label{" "}
                            {formatValue(result.graph_label)}
                          </span>
                        </button>
                      ))
                    ) : (
                      <span className="muted">No graph or node matches.</span>
                    )}
                  </>
                ) : null}
              </div>
            ) : null}
            {analysis.data.visual.sampled ? (
              <p className="table-note">
                Showing {formatCount(analysis.data.visual.nodes.length)} nodes and {formatCount(analysis.data.visual.edges.length)} edges (
                {analysis.data.visual.sample_reason}).
              </p>
            ) : null}
            {selectedNodeOutsideSample ? (
              <p className="table-note">
                Selected node {exploreNodeId} is outside the sampled visual. Inspector details are shown in the Right Panel.
              </p>
            ) : null}
            <DatasetGraphChart visual={analysis.data.visual} selectedNodeId={exploreNodeId} onSelectNode={onExploreNodeChange} />
          </div>
        ) : null}
        {tab === "data" ? <DatasetDataTab activeProjectId={activeProjectId} dataset={dataset} /> : null}
      </section>
    </div>
  );
}
