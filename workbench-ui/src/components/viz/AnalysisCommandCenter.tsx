/**
 * AnalysisCommandCenter — a grid of auto-computing analysis cards (PCA / t-SNE / UMAP / Clustering)
 * shared by the Embedding and Feature Explore pages. Each card fetches its own projection/clustering
 * via the supplied `analyze` function, exposes knobs + a re-run button + a color-by-label toggle, and
 * renders a publication-quality scatter inside a ChartCard (with PNG/SVG export).
 *
 * The page supplies `analyze` (e.g. params => api.embeddingAnalysis(pid, id, params)), a stable
 * `queryKeyBase`, an `exportName`, and whether graph labels exist.
 */
import { type MutableRefObject, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { RotateCcw } from "lucide-react";
import type { AnalysisAnalyzeParams, AnalysisPcaPayload, AnalysisPcaPoint, AnalysisResult } from "../../api";
import { ChartCard } from "./ChartCard";

type AnalyzeFn = (params: AnalysisAnalyzeParams) => Promise<AnalysisResult>;
type ProjectionMethod = "pca" | "tsne" | "umap";

const ANALYSIS_PALETTE = [
  "#176ea9", "#d86c1f", "#2d8754", "#8d5db8", "#a4513d", "#4f758b", "#6b7f2a", "#9a5b91", "#b0883a", "#3f7d7a"
];

// Shared academic chart styling tokens.
const CHART_FONT = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const AXIS_NAME_COLOR = "#3a444e";
const AXIS_LABEL_COLOR = "#5b6671";
const AXIS_LINE_COLOR = "#c8d0d6";
const SPLIT_LINE_COLOR = "#eef1f4";

type AnalysisScatterDatum = {
  value: [number, number];
  graph_id: string;
  graph_label: unknown;
  cluster: number | null;
  node_count: number | null;
};

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const value = Math.round(Number(raw));
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function clampFloat(raw: string, min: number, max: number, fallback: number): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function labelText(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

const PROJECTION_META: Record<ProjectionMethod, { title: string; description: string }> = {
  pca: { title: "PCA", description: "Linear projection onto the top two principal components." },
  tsne: { title: "t-SNE", description: "Nonlinear neighborhood embedding (stochastic)." },
  umap: { title: "UMAP", description: "Uniform Manifold Approximation and Projection." }
};

/** Publication-quality scatter: one series per category (label / cluster) for a clean legend. */
function AnalysisScatter({
  payload,
  colorMode,
  chartRef
}: {
  payload: AnalysisPcaPayload;
  colorMode: "label" | "cluster" | "none";
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
  }, [chartRef]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const points = payload.points;

    const categoryOf = (point: AnalysisPcaPoint): string => {
      if (colorMode === "cluster") return point.cluster != null ? `Cluster ${point.cluster}` : "Unclustered";
      if (colorMode === "label") return point.graph_label != null ? String(point.graph_label) : "Unlabeled";
      return "Graphs";
    };

    const categories = Array.from(new Set(points.map(categoryOf)));
    const showLegend = colorMode !== "none" && categories.length > 1;

    const series = categories.map((category, index) => ({
      name: category,
      type: "scatter" as const,
      symbolSize: 11,
      itemStyle: {
        color: colorMode === "none" ? "#176ea9" : ANALYSIS_PALETTE[index % ANALYSIS_PALETTE.length],
        opacity: 0.82,
        borderColor: "rgba(255,255,255,.7)",
        borderWidth: 0.5
      },
      data: points
        .filter((point) => categoryOf(point) === category)
        .map((point) => ({
          value: [point.x, point.y],
          graph_id: point.graph_id,
          graph_label: point.graph_label,
          cluster: point.cluster ?? null,
          node_count: point.node_count ?? null
        }))
    }));

    const option: EChartsOption = {
      animation: false,
      textStyle: { fontFamily: CHART_FONT },
      legend: showLegend
        ? {
            type: "scroll",
            top: 2,
            icon: "circle",
            itemWidth: 8,
            itemHeight: 8,
            textStyle: { fontSize: 11, color: AXIS_NAME_COLOR, fontFamily: CHART_FONT }
          }
        : undefined,
      grid: { left: 58, right: 22, top: showLegend ? 30 : 16, bottom: 50 },
      xAxis: {
        type: "value",
        name: payload.x_axis_label,
        nameLocation: "middle",
        nameGap: 28,
        scale: true,
        nameTextStyle: { color: AXIS_NAME_COLOR, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: AXIS_LABEL_COLOR, fontSize: 11, hideOverlap: true },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } },
        splitLine: { lineStyle: { color: SPLIT_LINE_COLOR } }
      },
      yAxis: {
        type: "value",
        name: payload.y_axis_label,
        nameLocation: "middle",
        nameGap: 40,
        scale: true,
        nameTextStyle: { color: AXIS_NAME_COLOR, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: AXIS_LABEL_COLOR, fontSize: 11, hideOverlap: true },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: AXIS_LINE_COLOR } },
        splitLine: { lineStyle: { color: SPLIT_LINE_COLOR } }
      },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = Array.isArray(params) ? params[0] : params;
          const dataPoint = (item as { data?: AnalysisScatterDatum }).data;
          if (!dataPoint) return "";
          return [
            `Graph ${dataPoint.graph_id}`,
            dataPoint.graph_label != null ? `Label ${labelText(dataPoint.graph_label)}` : null,
            dataPoint.node_count != null ? `${dataPoint.node_count} nodes` : null,
            dataPoint.cluster != null ? `Cluster ${dataPoint.cluster}` : null,
            `${payload.x_axis_label} ${dataPoint.value[0].toFixed(4)}`,
            `${payload.y_axis_label} ${dataPoint.value[1].toFixed(4)}`
          ]
            .filter(Boolean)
            .join("<br/>");
        }
      },
      series
    };

    chart.setOption(option, { notMerge: true });
  }, [payload, colorMode, chartRef]);

  return <div ref={containerRef} className="chart-card-canvas analysis-scatter" role="img" aria-label="Analysis projection scatter" />;
}

function ProjectionCard({
  analyze,
  queryKeyBase,
  exportName,
  method,
  hasLabels
}: {
  analyze: AnalyzeFn;
  queryKeyBase: unknown[];
  exportName: string;
  method: ProjectionMethod;
  hasLabels: boolean;
}) {
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const [colorByLabel, setColorByLabel] = useState(hasLabels);
  const [perplexity, setPerplexity] = useState(20);
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);

  const params = useMemo<AnalysisAnalyzeParams>(() => {
    const next: AnalysisAnalyzeParams = { projection_method: method };
    if (method === "tsne") next.perplexity = perplexity;
    if (method === "umap") {
      next.n_neighbors = nNeighbors;
      next.min_dist = minDist;
    }
    return next;
  }, [method, perplexity, nNeighbors, minDist]);

  const query = useQuery({
    queryKey: [...queryKeyBase, "projection", params],
    queryFn: () => analyze(params),
    enabled: queryKeyBase.every(Boolean),
    placeholderData: keepPreviousData
  });

  const payload = query.data?.pca;
  const meta = PROJECTION_META[method];

  let subtitle = meta.description;
  if (method === "pca" && payload?.explained_variance_ratio?.length) {
    const [pc1, pc2] = payload.explained_variance_ratio;
    subtitle = `Explained variance — PC1 ${((pc1 ?? 0) * 100).toFixed(1)}%, PC2 ${((pc2 ?? 0) * 100).toFixed(1)}%`;
  } else if (method === "tsne") {
    subtitle = `Perplexity ${perplexity}`;
  } else if (method === "umap") {
    subtitle = `n_neighbors ${nNeighbors} · min_dist ${minDist}`;
  }

  const controls = (
    <>
      {method === "tsne" ? (
        <label className="card-knob">
          <span>Perplexity</span>
          <input
            type="number"
            min={2}
            max={100}
            value={perplexity}
            aria-label="t-SNE perplexity"
            onChange={(event) => setPerplexity(clampInt(event.target.value, 2, 100, 20))}
          />
        </label>
      ) : null}
      {method === "umap" ? (
        <>
          <label className="card-knob">
            <span>Neighbors</span>
            <input
              type="number"
              min={2}
              max={200}
              value={nNeighbors}
              aria-label="UMAP neighbors"
              onChange={(event) => setNNeighbors(clampInt(event.target.value, 2, 200, 15))}
            />
          </label>
          <label className="card-knob">
            <span>Min dist</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minDist}
              aria-label="UMAP min dist"
              onChange={(event) => setMinDist(clampFloat(event.target.value, 0, 1, 0.1))}
            />
          </label>
        </>
      ) : null}
      {hasLabels ? (
        <label className="card-toggle">
          <input type="checkbox" checked={colorByLabel} onChange={(event) => setColorByLabel(event.target.checked)} />
          <span>Color by label</span>
        </label>
      ) : null}
      <button
        type="button"
        className="icon-btn"
        aria-label={`Re-run ${meta.title}`}
        title="Re-run"
        onClick={() => query.refetch()}
        disabled={query.isFetching}
      >
        <RotateCcw />
      </button>
    </>
  );

  const error = query.error
    ? (query.error as Error).message
    : payload && !payload.available
      ? payload.reason || "Projection unavailable."
      : undefined;

  return (
    <ChartCard
      title={meta.title}
      subtitle={subtitle}
      controls={controls}
      chartRef={chartRef}
      exportName={`${exportName} ${method}`}
      running={query.isFetching}
      runningLabel={`Computing ${meta.title}…`}
      error={error}
      className="chart-card-square"
    >
      {payload && payload.available ? (
        <AnalysisScatter payload={payload} colorMode={hasLabels && colorByLabel ? "label" : "none"} chartRef={chartRef} />
      ) : (
        <div className="chart-card-canvas" />
      )}
    </ChartCard>
  );
}

function ClusteringCard({
  analyze,
  queryKeyBase,
  exportName,
  hasLabels
}: {
  analyze: AnalyzeFn;
  queryKeyBase: unknown[];
  exportName: string;
  hasLabels: boolean;
}) {
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const [clusterK, setClusterK] = useState(2);

  const query = useQuery({
    queryKey: [...queryKeyBase, "clustering", clusterK],
    queryFn: () => analyze({ projection_method: "pca", cluster_k: clusterK }),
    enabled: queryKeyBase.every(Boolean),
    placeholderData: keepPreviousData
  });

  const payload = query.data?.pca;
  const metrics: string[] = [];
  if (payload?.cluster_silhouette != null) metrics.push(`silhouette ${payload.cluster_silhouette.toFixed(3)}`);
  if (hasLabels && payload?.cluster_label_ari != null) metrics.push(`ARI ${payload.cluster_label_ari.toFixed(3)}`);
  if (hasLabels && payload?.cluster_purity != null) metrics.push(`purity ${(payload.cluster_purity * 100).toFixed(0)}%`);
  const subtitle = metrics.length ? metrics.join(" · ") : "KMeans clustering shown on the PCA projection.";

  const controls = (
    <>
      <label className="card-knob">
        <span>Clusters (k)</span>
        <input
          type="number"
          min={2}
          max={20}
          value={clusterK}
          aria-label="Number of KMeans clusters"
          onChange={(event) => setClusterK(clampInt(event.target.value, 2, 20, 2))}
        />
      </label>
      <button
        type="button"
        className="icon-btn"
        aria-label="Re-run clustering"
        title="Re-run"
        onClick={() => query.refetch()}
        disabled={query.isFetching}
      >
        <RotateCcw />
      </button>
    </>
  );

  const error = query.error
    ? (query.error as Error).message
    : payload && !payload.available
      ? payload.reason || "Clustering unavailable."
      : undefined;

  return (
    <ChartCard
      title="Clustering (KMeans)"
      subtitle={subtitle}
      controls={controls}
      chartRef={chartRef}
      exportName={`${exportName} clusters k${clusterK}`}
      running={query.isFetching}
      runningLabel="Clustering…"
      error={error}
      className="chart-card-square"
    >
      {payload && payload.available ? (
        <AnalysisScatter payload={payload} colorMode="cluster" chartRef={chartRef} />
      ) : (
        <div className="chart-card-canvas" />
      )}
    </ChartCard>
  );
}

export function AnalysisCommandCenter({
  analyze,
  queryKeyBase,
  exportName,
  hasLabels
}: {
  analyze: AnalyzeFn;
  queryKeyBase: unknown[];
  exportName: string;
  hasLabels: boolean;
}) {
  return (
    <div className="dataset-tab-panel analysis-panel">
      <div className="analysis-grid">
        <ProjectionCard analyze={analyze} queryKeyBase={queryKeyBase} exportName={exportName} method="pca" hasLabels={hasLabels} />
        <ProjectionCard analyze={analyze} queryKeyBase={queryKeyBase} exportName={exportName} method="tsne" hasLabels={hasLabels} />
        <ProjectionCard analyze={analyze} queryKeyBase={queryKeyBase} exportName={exportName} method="umap" hasLabels={hasLabels} />
        <ClusteringCard analyze={analyze} queryKeyBase={queryKeyBase} exportName={exportName} hasLabels={hasLabels} />
      </div>
    </div>
  );
}
