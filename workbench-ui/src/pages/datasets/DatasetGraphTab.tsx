import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import {
  api,
  type DatasetAnalysis,
  type DatasetGraphSearchResult,
  type DatasetGraphSummary,
  type DatasetGraphTile,
  type DatasetVisualEdge,
  type DatasetVisualNode
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";

const GRID_PAGE_SIZE = 12;
const OVERVIEW_FULL_MAX_NODES = 2000;
const OVERVIEW_FULL_MAX_EDGES = 6000;

const GRAPH_LABEL_COLORS = [
  { background: "#e9f5ff", borderColor: "#87bde8", color: "#155f8e" },
  { background: "#eaf7ef", borderColor: "#82c89a", color: "#236f3b" },
  { background: "#fff3d6", borderColor: "#e3b85f", color: "#7a5510" },
  { background: "#f0e9ff", borderColor: "#ad93e5", color: "#5c3ea0" },
  { background: "#ffe9ef", borderColor: "#e99aae", color: "#8f2d49" },
  { background: "#eaf7f6", borderColor: "#7fc7c1", color: "#1e6d67" }
];

export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatValue(value: unknown): string {
  if (value == null) return "None";
  return String(value);
}

export function graphLabelStyle(label: unknown): CSSProperties {
  const text = formatValue(label);
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return GRAPH_LABEL_COLORS[hash % GRAPH_LABEL_COLORS.length];
}

type GraphViewMode = "overview" | "grid" | "single";
type GraphChartVariant = "single" | "tile" | "overview";

interface GraphVisualData {
  nodes: DatasetVisualNode[];
  edges: DatasetVisualEdge[];
}

export function DatasetGraphChart({
  visual,
  variant = "single",
  ariaLabel,
  selectedNodeId = "",
  onSelectNode
}: {
  visual: GraphVisualData;
  variant?: GraphChartVariant;
  ariaLabel: string;
  selectedNodeId?: string;
  onSelectNode?: (nodeId: string) => void;
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
      onSelectNodeRef.current?.(String(data?.id || params.name || ""));
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

    const isTile = variant === "tile";
    const isOverview = variant === "overview";
    const maxDegree = Math.max(1, ...visual.nodes.map((node) => node.degree));
    const option: EChartsOption = {
      tooltip: isTile
        ? { show: false }
        : {
            trigger: "item",
            formatter: (params) => {
              const item = Array.isArray(params) ? params[0] : params;
              const data = item.data as { name?: string; value?: number; sourceNodeId?: string | null; isCenter?: boolean | null };
              if (!data.name) return "";
              if (isOverview) {
                const centroidLine = data.isCenter ? "<br/>Egonet centroid — click to open its egonet" : "";
                return `${data.name}<br/>Degree: ${data.value ?? 0}${centroidLine}`;
              }
              const sourceLine = data.sourceNodeId && !data.isCenter ? `<br/>Source node: ${data.sourceNodeId}` : "";
              const centerLine = data.isCenter ? `<br/>Center source node: ${data.sourceNodeId || data.name}` : "";
              return `${data.name}<br/>Degree: ${data.value ?? 0}${sourceLine}${centerLine}`;
            }
          },
      series: [
        {
          type: "graph",
          layout: "force",
          roam: !isTile,
          animation: false,
          silent: isTile,
          label: {
            show: variant === "single" && visual.nodes.length <= 40,
            position: "right",
            color: "#34424c",
            fontSize: 10
          },
          data: visual.nodes.map((node) => {
            const isCenter = Boolean(node.is_center);
            return {
              id: node.id,
              name: node.label,
              value: node.degree,
              sourceNodeId: node.source_node_id,
              isCenter,
              symbolSize: isTile
                ? 5 + (node.degree / maxDegree) * 7 + (isCenter ? 4 : 0)
                : 9 + (node.degree / maxDegree) * 18 + (isCenter ? 8 : 0),
              itemStyle: {
                color: isCenter ? (isOverview ? "#c0541d" : "#f28c28") : "#176ea9",
                borderColor: isCenter ? (isOverview ? "#7a3300" : "#8f3e00") : "#ffffff",
                borderWidth: isCenter ? (isTile ? 1.5 : 2.5) : isTile ? 0.5 : 1
              }
            };
          }),
          links: visual.edges.map((edge) => ({ source: edge.source, target: edge.target })),
          lineStyle: {
            color: "#8a98a6",
            opacity: 0.72,
            width: isTile ? 0.8 : 1.2
          },
          emphasis: isTile
            ? undefined
            : {
                focus: "adjacency",
                itemStyle: { color: "#f28c28", borderColor: "#7a3300", borderWidth: 3 },
                lineStyle: { width: 2 }
              },
          force: isTile
            ? { repulsion: 30, edgeLength: [10, 30] }
            : isOverview
              ? { repulsion: 42, edgeLength: [10, 34] }
              : { repulsion: 80, edgeLength: [28, 90] }
        }
      ]
    };

    chart.setOption(option, { notMerge: true });
    previousSelectedNodeRef.current = "";
  }, [visual, variant]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || variant !== "single") return;

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
  }, [selectedNodeId, visual, variant]);

  return (
    <div
      ref={containerRef}
      className={`dataset-graph-chart ${variant === "tile" ? "graph-tile-chart" : ""}`}
      role="img"
      aria-label={ariaLabel}
      tabIndex={variant === "tile" ? -1 : 0}
    />
  );
}

interface DatasetGraphTabProps {
  activeProjectId: string;
  datasetId: string;
  analysis: DatasetAnalysis;
  graphSummaries: DatasetGraphSummary[];
  selectedGraphIndex: number;
  selectedSummary: DatasetGraphSummary | null;
  exploreNodeId: string;
  selectedNodeOutsideSample: boolean;
  graphSearchQuery: string;
  onGraphSearchQueryChange: (value: string) => void;
  onSelectGraphByIndex: (index: number) => void;
  onSelectSearchResult: (result: DatasetGraphSearchResult) => void;
  onExploreGraphChange: (graphId: string, summary: DatasetGraphSummary | null, options?: { clearNode?: boolean }) => void;
  onExploreNodeChange: (nodeId: string) => void;
}

export function DatasetGraphTab({
  activeProjectId,
  datasetId,
  analysis,
  graphSummaries,
  selectedGraphIndex,
  selectedSummary,
  exploreNodeId,
  selectedNodeOutsideSample,
  graphSearchQuery,
  onGraphSearchQueryChange,
  onSelectGraphByIndex,
  onSelectSearchResult,
  onExploreGraphChange,
  onExploreNodeChange
}: DatasetGraphTabProps) {
  const [viewMode, setViewMode] = useState<GraphViewMode>("single");
  const [gridOffset, setGridOffset] = useState(0);
  const [overviewFull, setOverviewFull] = useState(false);
  const hasOverview = Boolean(analysis.egonet_metadata);
  const trimmedGraphSearch = graphSearchQuery.trim();

  useEffect(() => {
    setGridOffset(0);
  }, [trimmedGraphSearch]);

  const gridQuery = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", datasetId, "analysis", "grid", gridOffset, trimmedGraphSearch],
    queryFn: () =>
      api.datasetGraphGrid(activeProjectId, datasetId, {
        offset: gridOffset,
        limit: GRID_PAGE_SIZE,
        query: trimmedGraphSearch || undefined
      }),
    enabled: viewMode === "grid"
  });
  const overviewQuery = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", datasetId, "analysis", "source-graph", overviewFull],
    queryFn: () =>
      api.datasetSourceGraph(
        activeProjectId,
        datasetId,
        overviewFull ? { max_nodes: OVERVIEW_FULL_MAX_NODES, max_edges: OVERVIEW_FULL_MAX_EDGES } : {}
      ),
    enabled: viewMode === "overview" && hasOverview
  });
  const graphSearch = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", datasetId, "analysis", "search", trimmedGraphSearch],
    queryFn: () => api.datasetGraphSearch(activeProjectId, datasetId, trimmedGraphSearch, 25),
    enabled: Boolean(trimmedGraphSearch) && viewMode !== "grid"
  });

  const gridTotal = gridQuery.data?.total_graphs ?? 0;
  const gridPageCount = Math.max(1, Math.ceil(gridTotal / GRID_PAGE_SIZE));
  const gridPage = Math.floor(gridOffset / GRID_PAGE_SIZE) + 1;
  const canGridPrev = gridOffset > 0;
  const canGridNext = gridOffset + GRID_PAGE_SIZE < gridTotal;
  const overview = overviewQuery.data;

  const openTile = (tile: DatasetGraphTile) => {
    const summary = graphSummaries.find((item) => item.graph_id === tile.graph_id) || null;
    onExploreGraphChange(tile.graph_id, summary);
    setViewMode("single");
  };

  const openEgonetForCentroid = (sourceNodeId: string) => {
    const summary = graphSummaries.find((item) => item.source_node_id === sourceNodeId) || null;
    if (!summary) return;
    onExploreGraphChange(summary.graph_id, summary);
    setViewMode("single");
  };

  const selectSearchResult = (result: DatasetGraphSearchResult) => {
    onSelectSearchResult(result);
    if (viewMode === "overview") setViewMode("single");
  };

  const navPrev = () => {
    if (viewMode === "single") onSelectGraphByIndex(selectedGraphIndex - 1);
    if (viewMode === "grid" && canGridPrev) setGridOffset(Math.max(0, gridOffset - GRID_PAGE_SIZE));
  };
  const navNext = () => {
    if (viewMode === "single") onSelectGraphByIndex(selectedGraphIndex + 1);
    if (viewMode === "grid" && canGridNext) setGridOffset(gridOffset + GRID_PAGE_SIZE);
  };
  const navPrevDisabled =
    viewMode === "overview" || (viewMode === "single" ? selectedGraphIndex <= 0 : !canGridPrev);
  const navNextDisabled =
    viewMode === "overview" ||
    (viewMode === "single" ? selectedGraphIndex < 0 || selectedGraphIndex >= graphSummaries.length - 1 : !canGridNext);
  const navLabel = viewMode === "grid" ? "page" : "graph";
  const positionText =
    viewMode === "single"
      ? selectedGraphIndex >= 0
        ? `${selectedGraphIndex + 1} / ${formatCount(graphSummaries.length)}`
        : ""
      : viewMode === "grid" && gridQuery.data
        ? `${gridPage} / ${formatCount(gridPageCount)}`
        : "";

  const handleGraphKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
    if (event.key === "ArrowLeft" && !navPrevDisabled) {
      event.preventDefault();
      navPrev();
    }
    if (event.key === "ArrowRight" && !navNextDisabled) {
      event.preventDefault();
      navNext();
    }
  };

  const gridRangeStart = gridTotal === 0 ? 0 : gridOffset + 1;
  const gridRangeEnd = Math.min(gridOffset + GRID_PAGE_SIZE, gridTotal);

  return (
    <div className="dataset-tab-panel graph-tab-panel" tabIndex={0} onKeyDown={handleGraphKeyDown} aria-label="Dataset graph view">
      <div className="dataset-graph-header">
        <div className="graph-toolbar-row">
          <div className="graph-nav-group">
            <button
              type="button"
              className="icon-btn graph-nav-btn"
              aria-label={`Previous ${navLabel}`}
              title={`Previous ${navLabel}`}
              onClick={navPrev}
              disabled={navPrevDisabled}
            >
              <ChevronLeft />
            </button>
            <button
              type="button"
              className="icon-btn graph-nav-btn"
              aria-label={`Next ${navLabel}`}
              title={`Next ${navLabel}`}
              onClick={navNext}
              disabled={navNextDisabled}
            >
              <ChevronRight />
            </button>
            <span className="graph-position">{positionText}</span>
          </div>
          <div className="seg-control" role="tablist" aria-label="Graph view mode">
            {hasOverview ? (
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "overview"}
                className={`seg-btn ${viewMode === "overview" ? "is-active" : ""}`}
                onClick={() => setViewMode("overview")}
              >
                Overview
              </button>
            ) : null}
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === "grid"}
              className={`seg-btn ${viewMode === "grid" ? "is-active" : ""}`}
              onClick={() => setViewMode("grid")}
            >
              Grid
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === "single"}
              className={`seg-btn ${viewMode === "single" ? "is-active" : ""}`}
              onClick={() => setViewMode("single")}
            >
              Single
            </button>
          </div>
          <label className="field graph-search-field">
            <span>Search</span>
            <div className="graph-search-input">
              <Search />
              <input
                aria-label={viewMode === "grid" ? "Filter graphs" : "Search graphs and nodes"}
                value={graphSearchQuery}
                placeholder={viewMode === "grid" ? "Filter by graph ID" : "Graph or node ID"}
                onChange={(event) => onGraphSearchQueryChange(event.target.value)}
              />
            </div>
          </label>
        </div>
        <div className="graph-meta-strip">
          {viewMode === "single" && selectedSummary ? (
            <>
              <span className="graph-id-badge mono" title={`Graph ${selectedSummary.graph_id}`}>
                Graph {selectedSummary.graph_id}
              </span>
              <span className="graph-meta-badge">{formatCount(selectedSummary.node_count)} nodes</span>
              <span className="graph-meta-badge">{formatCount(selectedSummary.edge_count)} edges</span>
              {selectedSummary.graph_label != null ? (
                <span
                  className="graph-label-badge"
                  style={graphLabelStyle(selectedSummary.graph_label)}
                  title={`Label ${formatValue(selectedSummary.graph_label)}`}
                >
                  Label {formatValue(selectedSummary.graph_label)}
                </span>
              ) : null}
              {selectedSummary.source_node_id ? (
                <span className="graph-center-badge" title={`Center source node ${selectedSummary.source_node_id}`}>
                  Center {selectedSummary.source_node_id}
                </span>
              ) : null}
              {analysis.egonet_metadata ? (
                <span className="graph-meta-badge">{formatCount(analysis.egonet_metadata.k_hop)}-hop egonet</span>
              ) : null}
              {analysis.egonet_metadata?.target_node_attribute ? (
                <span className="graph-meta-badge" title={`Target ${analysis.egonet_metadata.target_node_attribute}`}>
                  Target {analysis.egonet_metadata.target_node_attribute}
                </span>
              ) : null}
              {analysis.visual.sampled ? (
                <span
                  className="status-pill is-idle"
                  title={`Showing ${formatCount(analysis.visual.nodes.length)} nodes and ${formatCount(
                    analysis.visual.edges.length
                  )} edges (${analysis.visual.sample_reason})`}
                >
                  sampled
                </span>
              ) : null}
            </>
          ) : null}
          {viewMode === "overview" ? (
            overview ? (
              <>
                <span className="graph-id-badge">Source graph</span>
                <span className="graph-meta-badge">{formatCount(overview.node_count)} nodes</span>
                <span className="graph-meta-badge">{formatCount(overview.edge_count)} edges</span>
                <span className="graph-center-badge">{formatCount(overview.centroid_count)} centroids</span>
                {overview.sampled ? (
                  <span className="graph-meta-note" title={overview.sample_reason || undefined}>
                    showing {formatCount(overview.nodes.length)} of {formatCount(overview.node_count)} nodes
                    {overviewFull ? " (cap)" : ""}
                  </span>
                ) : null}
                {overview.sampled && !overviewFull ? (
                  <button type="button" className="graph-meta-action" onClick={() => setOverviewFull(true)}>
                    Load full graph
                  </button>
                ) : null}
              </>
            ) : (
              <span className="graph-meta-note">Loading source graph.</span>
            )
          ) : null}
          {viewMode === "grid" ? (
            gridQuery.data ? (
              <>
                <span className="graph-meta-badge">
                  Graphs {formatCount(gridRangeStart)}–{formatCount(gridRangeEnd)} of {formatCount(gridTotal)}
                </span>
                {gridQuery.data.query ? (
                  <span className="graph-meta-note" title={`Filtered by "${gridQuery.data.query}"`}>
                    filtered by &quot;{gridQuery.data.query}&quot;
                  </span>
                ) : null}
              </>
            ) : (
              <span className="graph-meta-note">Loading graphs.</span>
            )
          ) : null}
        </div>
      </div>
      {trimmedGraphSearch && viewMode !== "grid" ? (
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
                      result.graph_id === analysis.selected_graph_id && (!result.node_id || result.node_id === exploreNodeId)
                        ? "is-selected"
                        : ""
                    }`}
                    onClick={() => selectSearchResult(result)}
                  >
                    <span className="status-pill is-idle">{result.kind}</span>
                    <strong>{result.kind === "node" ? result.node_id : result.graph_id}</strong>
                    <span className="muted">
                      {result.kind === "node" ? `graph ${result.graph_id} · ` : ""}
                      {formatCount(result.node_count)} nodes · {formatCount(result.edge_count)} edges
                      {result.graph_label != null ? (
                        <span className="graph-label-badge inline-label-badge" style={graphLabelStyle(result.graph_label)}>
                          Label {formatValue(result.graph_label)}
                        </span>
                      ) : null}
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
      {viewMode === "single" ? (
        <>
          {selectedNodeOutsideSample ? (
            <p className="table-note">
              Selected node {exploreNodeId} is outside the sampled visual. Inspector details are shown in the Right Panel.
            </p>
          ) : null}
          <DatasetGraphChart
            visual={analysis.visual}
            variant="single"
            ariaLabel={`Graph ${analysis.visual.graph_id}`}
            selectedNodeId={exploreNodeId}
            onSelectNode={onExploreNodeChange}
          />
        </>
      ) : null}
      {viewMode === "overview" ? (
        overviewQuery.error ? (
          <p className="table-error">{overviewQuery.error.message}</p>
        ) : overviewQuery.isLoading || !overview ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading source graph.</EmptyState>
          </div>
        ) : (
          <DatasetGraphChart
            visual={overview}
            variant="overview"
            ariaLabel="Source graph overview"
            onSelectNode={openEgonetForCentroid}
          />
        )
      ) : null}
      {viewMode === "grid" ? (
        gridQuery.error ? (
          <p className="table-error">{gridQuery.error.message}</p>
        ) : gridQuery.isLoading || !gridQuery.data ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading graphs.</EmptyState>
          </div>
        ) : gridQuery.data.tiles.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No graphs match the current filter.</EmptyState>
          </div>
        ) : (
          <div className="graph-tile-grid">
            {gridQuery.data.tiles.map((tile) => (
              <button
                type="button"
                key={tile.graph_id}
                className={`graph-tile ${tile.graph_id === analysis.selected_graph_id ? "is-selected" : ""}`}
                onClick={() => openTile(tile)}
                title={`Open graph ${tile.graph_id}`}
              >
                <DatasetGraphChart visual={tile.visual} variant="tile" ariaLabel={`Graph ${tile.graph_id} preview`} />
                <span className="graph-tile-caption">
                  <span className="mono">Graph {tile.graph_id}</span>
                  <span className="muted">
                    {formatCount(tile.node_count)} nodes · {formatCount(tile.edge_count)} edges
                  </span>
                  {tile.graph_label != null ? (
                    <span className="graph-label-badge inline-label-badge" style={graphLabelStyle(tile.graph_label)}>
                      Label {formatValue(tile.graph_label)}
                    </span>
                  ) : null}
                </span>
              </button>
            ))}
          </div>
        )
      ) : null}
    </div>
  );
}
