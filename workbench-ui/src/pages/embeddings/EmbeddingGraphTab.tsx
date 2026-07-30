import { useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { Search } from "lucide-react";
import { api, type EmbeddingManifest, type EmbeddingSimilarityGraph, type EmbeddingSimilarityGraphPayload, type EmbeddingSimilarityNode } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";

const SIMILARITY_RAMP = ["#ffffff", "#f28c28", "#c0541d"];
const BELOW_THRESHOLD_COLOR = "#dfe3e7";
const UNCOLORED_COLOR = "#9fb6c6";
const BASE_BORDER = "#d4d9df";
const MARK_BORDER = "#1f2428";
const DEFAULT_MAX_NODES = 500;
const DEFAULT_MAX_EDGES = 1500;
const FULL_MAX_NODES = 2000;
const FULL_MAX_EDGES = 6000;
const HISTOGRAM_BINS = 24;
const CHIP_DISPLAY_LIMIT = 12;

type ReferenceMode = "node" | "selection" | "label";
type ScalingMode = "linear" | "percentile";

function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function similarityRampColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (SIMILARITY_RAMP.length - 1);
  const index = Math.min(SIMILARITY_RAMP.length - 2, Math.floor(scaled));
  const frac = scaled - index;
  const from = hexToRgb(SIMILARITY_RAMP[index]);
  const to = hexToRgb(SIMILARITY_RAMP[index + 1]);
  const mix = from.map((channel, channelIndex) => Math.round(channel + (to[channelIndex] - channel) * frac));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

function stableIdCompare(a: string, b: string): number {
  const aNumeric = /^\d+$/.test(a);
  const bNumeric = /^\d+$/.test(b);
  if (aNumeric && bNumeric) return Number(a) - Number(b);
  if (aNumeric) return -1;
  if (bNumeric) return 1;
  return a.localeCompare(b);
}

function displayValues(nodes: EmbeddingSimilarityNode[], scaling: ScalingMode): Map<string, number> {
  const values = new Map<string, number>();
  const embedded = nodes.filter((node) => node.similarity != null);
  if (scaling === "linear") {
    for (const node of embedded) values.set(node.id, node.similarity as number);
    return values;
  }
  const sorted = [...embedded].sort((a, b) => (a.similarity as number) - (b.similarity as number));
  sorted.forEach((node, rank) => values.set(node.id, sorted.length > 1 ? rank / (sorted.length - 1) : 1));
  return values;
}

function SimilarityGraphChart({
  payload,
  scaling,
  threshold,
  truthLabel,
  ariaLabel,
  onSelectNode
}: {
  payload: EmbeddingSimilarityGraph;
  scaling: ScalingMode;
  threshold: number;
  truthLabel: string | null;
  ariaLabel: string;
  onSelectNode?: (nodeId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const onSelectNodeRef = useRef(onSelectNode);
  const structureKeyRef = useRef("");

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

    const hasReference = payload.reference_node_count > 0;
    const values = displayValues(payload.nodes, scaling);
    const maxDegree = Math.max(1, ...payload.nodes.map((node) => node.degree));
    const largeGraph = payload.nodes.length > 300;
    const chartNodes = payload.nodes.map((node) => {
      const isReference = Boolean(node.is_reference);
      const isTruth = Boolean(truthLabel && !isReference && String(node.graph_label) === truthLabel);
      let color: string;
      let opacity = 1;
      if (!hasReference || node.similarity == null) {
        color = UNCOLORED_COLOR;
      } else if ((node.similarity as number) < threshold) {
        color = BELOW_THRESHOLD_COLOR;
        opacity = 0.45;
      } else {
        color = similarityRampColor(values.get(node.id) ?? 0);
      }
      return {
        id: node.id,
        name: node.id,
        value: node.similarity,
        similarity: node.similarity,
        cosine: node.cosine,
        internalGraphId: node.internal_graph_id,
        graphLabel: node.graph_label,
        isReference,
        isTruth,
        symbol: isReference ? "diamond" : "circle",
        symbolSize: (largeGraph ? 6 : 9) + (node.degree / maxDegree) * (largeGraph ? 12 : 18) + (isReference ? 6 : 0),
        itemStyle: {
          color,
          opacity,
          borderColor: isReference || isTruth ? MARK_BORDER : BASE_BORDER,
          borderWidth: isReference ? 2 : isTruth ? 1.5 : 1
        }
      };
    });

    const option: EChartsOption = {
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const item = Array.isArray(params) ? params[0] : params;
          const data = item.data as {
            name?: string;
            value?: number | null;
            similarity?: number | null;
            cosine?: number | null;
            internalGraphId?: string | null;
            graphLabel?: unknown;
            isReference?: boolean;
            isTruth?: boolean;
          };
          if (!data.name) return "";
          const node = payload.nodes.find((candidate) => candidate.id === data.name);
          const lines = [`<strong>Node ${data.name}</strong>`, `Degree: ${node?.degree ?? 0}`];
          if (data.similarity != null) {
            lines.push(`Similarity: ${(data.similarity as number).toFixed(3)}`);
            if (data.cosine != null) lines.push(`Cosine: ${(data.cosine as number).toFixed(3)}`);
          }
          if (data.internalGraphId != null) lines.push(`Egonet: ${data.internalGraphId}`);
          if (data.graphLabel != null) lines.push(`Label: ${String(data.graphLabel)}`);
          if (data.isReference) lines.push("Reference node");
          if (data.isTruth) lines.push("Ground-truth match");
          return lines.join("<br/>");
        }
      },
      series: [
        {
          type: "graph",
          layout: "force",
          roam: true,
          animation: false,
          label: {
            show: payload.nodes.length <= 40,
            position: "right",
            color: "#34424c",
            fontSize: 10
          },
          data: chartNodes,
          links: payload.edges.map((edge) => ({ source: edge.source, target: edge.target })),
          lineStyle: { color: "#8a98a6", opacity: 0.5, width: largeGraph ? 0.8 : 1 },
          emphasis: { focus: "none", itemStyle: { borderColor: MARK_BORDER, borderWidth: 2 } },
          force: largeGraph ? { repulsion: 42, edgeLength: [10, 34], gravity: 0.14 } : { repulsion: 70, edgeLength: [24, 70], gravity: 0.12 }
        }
      ]
    };

    // Full rebuild only when the underlying graph changes; style-only updates merge so the
    // force layout warm-starts from current positions instead of re-randomizing.
    const structureKey = `${payload.embedding_id}:${payload.nodes.length}:${payload.edges.length}:${payload.nodes[0]?.id ?? ""}`;
    const rebuild = structureKey !== structureKeyRef.current;
    structureKeyRef.current = structureKey;
    chart.setOption(option, { notMerge: rebuild });
  }, [payload, scaling, threshold, truthLabel]);

  return <div ref={containerRef} className="similarity-graph-chart" role="img" aria-label={ariaLabel} tabIndex={0} />;
}

export function EmbeddingGraphTab({
  activeProjectId,
  embedding,
  labelDistribution
}: {
  activeProjectId: string;
  embedding: EmbeddingManifest;
  labelDistribution: Record<string, number>;
}) {
  const labelOptions = useMemo(() => Object.keys(labelDistribution).sort(stableIdCompare), [labelDistribution]);
  const [referenceMode, setReferenceMode] = useState<ReferenceMode>("node");
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [labelValue, setLabelValue] = useState("");
  const [fractionText, setFractionText] = useState("0.1");
  const [seedText, setSeedText] = useState("42");
  const [appliedLabelSample, setAppliedLabelSample] = useState<{ labelValue: string; sampleFraction: number; randomSeed: number } | null>(null);
  const [scaling, setScaling] = useState<ScalingMode>("linear");
  const [threshold, setThreshold] = useState(0);
  const [showTruth, setShowTruth] = useState(false);
  const [loadFull, setLoadFull] = useState(false);
  const [searchText, setSearchText] = useState("");

  useEffect(() => {
    if (!labelValue && labelOptions.length > 0) setLabelValue(labelOptions[0]);
  }, [labelOptions, labelValue]);

  const requestPayload: EmbeddingSimilarityGraphPayload = useMemo(() => {
    const caps = {
      max_nodes: loadFull ? FULL_MAX_NODES : DEFAULT_MAX_NODES,
      max_edges: loadFull ? FULL_MAX_EDGES : DEFAULT_MAX_EDGES
    };
    if (referenceMode === "label" && appliedLabelSample) {
      return {
        reference_mode: "label_sample",
        label_value: appliedLabelSample.labelValue,
        sample_fraction: appliedLabelSample.sampleFraction,
        random_seed: appliedLabelSample.randomSeed,
        ...caps
      };
    }
    return {
      reference_mode: "nodes",
      reference_source_node_ids: referenceMode === "label" ? [] : selectedNodeIds,
      ...caps
    };
  }, [referenceMode, selectedNodeIds, appliedLabelSample, loadFull]);

  const similarity = useQuery({
    queryKey: ["projects", activeProjectId, "embeddings", embedding.id, "analysis", "similarity-graph", requestPayload],
    queryFn: () => api.embeddingSimilarityGraph(activeProjectId, embedding.id, requestPayload),
    enabled: Boolean(activeProjectId && embedding.id),
    placeholderData: keepPreviousData
  });
  const payload = similarity.data;
  const hasReference = (payload?.reference_node_count ?? 0) > 0;
  const truthLabel = showTruth && referenceMode === "label" && appliedLabelSample ? appliedLabelSample.labelValue : null;

  const handleSelectNode = (nodeId: string) => {
    if (!nodeId || referenceMode === "label") return;
    if (referenceMode === "node") {
      setSelectedNodeIds([nodeId]);
      return;
    }
    setSelectedNodeIds((current) => (current.includes(nodeId) ? current.filter((id) => id !== nodeId) : [...current, nodeId].sort(stableIdCompare)));
  };

  const handleSearchSubmit = () => {
    const nodeId = searchText.trim();
    if (!nodeId || referenceMode === "label") return;
    setSearchText("");
    if (referenceMode === "node") setSelectedNodeIds([nodeId]);
    else setSelectedNodeIds((current) => (current.includes(nodeId) ? current : [...current, nodeId].sort(stableIdCompare)));
  };

  const handleModeChange = (mode: ReferenceMode) => {
    setReferenceMode(mode);
    setSelectedNodeIds([]);
    setAppliedLabelSample(null);
    setShowTruth(false);
  };

  const handleApplyLabelSample = () => {
    const fraction = Number(fractionText);
    const seed = Number(seedText);
    if (!labelValue || !Number.isFinite(fraction) || fraction <= 0 || fraction > 1 || !Number.isInteger(seed)) return;
    setAppliedLabelSample({ labelValue, sampleFraction: fraction, randomSeed: seed });
  };

  const histogram = useMemo(() => {
    const counts = new Array(HISTOGRAM_BINS).fill(0) as number[];
    if (payload && hasReference) {
      for (const node of payload.nodes) {
        if (node.similarity == null) continue;
        counts[Math.min(HISTOGRAM_BINS - 1, Math.floor((node.similarity as number) * HISTOGRAM_BINS))] += 1;
      }
    }
    const maxCount = Math.max(1, ...counts);
    return counts.map((count, index) => ({ height: (count / maxCount) * 100, center: (index + 0.5) / HISTOGRAM_BINS }));
  }, [payload, hasReference]);

  const referenceChips = payload?.reference_source_node_ids ?? selectedNodeIds;
  const visibleChips = referenceChips.slice(0, CHIP_DISPLAY_LIMIT);
  const hiddenChipCount = referenceChips.length - visibleChips.length;
  const labelModeDisabled = labelOptions.length === 0;

  return (
    <div className="dataset-tab-panel graph-tab-panel similarity-graph-panel" aria-label="Embedding similarity graph view">
      <div className="similarity-card">
        <section className="similarity-section">
          <h3>Reference</h3>
          <div className="seg-control similarity-seg" role="tablist" aria-label="Reference mode">
            {(
              [
                { mode: "node" as const, label: "Node" },
                { mode: "selection" as const, label: "Selection" },
                { mode: "label" as const, label: "Label" }
              ]
            ).map(({ mode, label }) => (
              <button
                key={mode}
                type="button"
                role="tab"
                aria-selected={referenceMode === mode}
                className={`seg-btn ${referenceMode === mode ? "is-active" : ""}`}
                onClick={() => handleModeChange(mode)}
                disabled={mode === "label" && labelModeDisabled}
                title={mode === "label" && labelModeDisabled ? "The source dataset has no node labels." : undefined}
              >
                {label}
              </button>
            ))}
          </div>
          {referenceMode !== "label" ? (
            <div className="similarity-field-row">
              <div className="graph-search-input similarity-search">
                <Search />
                <input
                  aria-label="Reference node id"
                  placeholder="node id"
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") handleSearchSubmit();
                  }}
                />
              </div>
              <button type="button" className="btn" onClick={handleSearchSubmit}>
                {referenceMode === "node" ? "Set" : "Add"}
              </button>
            </div>
          ) : (
            <>
              <label className="similarity-field-row">
                <span>Label</span>
                <select value={labelValue} onChange={(event) => setLabelValue(event.target.value)} aria-label="Label value">
                  {labelOptions.map((option) => (
                    <option key={option} value={option}>
                      {option} ({labelDistribution[option]})
                    </option>
                  ))}
                </select>
              </label>
              <label className="similarity-field-row">
                <span>Fraction</span>
                <input
                  type="number"
                  min={0.01}
                  max={1}
                  step={0.05}
                  value={fractionText}
                  onChange={(event) => setFractionText(event.target.value)}
                  aria-label="Sample fraction"
                />
              </label>
              <div className="similarity-field-row">
                <label className="similarity-inline-label">
                  <span>Seed</span>
                  <input type="number" value={seedText} onChange={(event) => setSeedText(event.target.value)} aria-label="Random seed" />
                </label>
                <button type="button" className="btn" onClick={handleApplyLabelSample}>
                  Apply
                </button>
              </div>
              <label className="similarity-field-row similarity-toggle">
                <span>Show ground truth</span>
                <input
                  type="checkbox"
                  checked={showTruth}
                  onChange={(event) => setShowTruth(event.target.checked)}
                  disabled={!appliedLabelSample}
                  aria-label="Show ground truth"
                />
              </label>
            </>
          )}
          <div className="similarity-chip-list" aria-label="Reference nodes">
            {referenceChips.length === 0 ? (
              <span className="similarity-chip-empty">No reference selected</span>
            ) : (
              <>
                {visibleChips.map((nodeId) => (
                  <span key={nodeId} className="similarity-chip">
                    {nodeId}
                    {referenceMode !== "label" ? (
                      <button
                        type="button"
                        aria-label={`Remove reference node ${nodeId}`}
                        onClick={() => setSelectedNodeIds((current) => current.filter((id) => id !== nodeId))}
                      >
                        ×
                      </button>
                    ) : null}
                  </span>
                ))}
                {hiddenChipCount > 0 ? <span className="similarity-chip-more">+{hiddenChipCount} more</span> : null}
              </>
            )}
          </div>
          <button
            type="button"
            className="btn similarity-clear-btn"
            onClick={() => {
              setSelectedNodeIds([]);
              setAppliedLabelSample(null);
              setShowTruth(false);
            }}
            disabled={referenceChips.length === 0}
          >
            Clear reference
          </button>
        </section>
        <section className="similarity-section">
          <h3>Coloring</h3>
          <div className="seg-control similarity-seg" role="tablist" aria-label="Color scaling">
            {(
              [
                { mode: "linear" as const, label: "Linear" },
                { mode: "percentile" as const, label: "Percentile" }
              ]
            ).map(({ mode, label }) => (
              <button
                key={mode}
                type="button"
                role="tab"
                aria-selected={scaling === mode}
                className={`seg-btn ${scaling === mode ? "is-active" : ""}`}
                onClick={() => setScaling(mode)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="similarity-field-row similarity-threshold">
            <span>Threshold</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
              aria-label="Similarity threshold"
            />
            <span className="similarity-threshold-value">{threshold.toFixed(2)}</span>
          </label>
          <div className="similarity-legend">
            <div className="similarity-histogram" aria-hidden="true">
              {histogram.map((bin, index) => (
                <div
                  key={index}
                  className="similarity-histogram-bin"
                  style={{
                    height: `${bin.height}%`,
                    background: hasReference && bin.center >= threshold ? similarityRampColor(bin.center) : "var(--chrome-300)"
                  }}
                />
              ))}
              {threshold > 0 ? <div className="similarity-histogram-threshold" style={{ left: `${threshold * 100}%` }} /> : null}
            </div>
            <div className="similarity-legend-gradient" style={{ background: `linear-gradient(to right, ${SIMILARITY_RAMP.join(", ")})` }} />
            <div className="similarity-legend-scale">
              <span>0.0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>
          <div className="similarity-swatch-rows">
            <div>
              <span className="similarity-dot is-reference" /> Reference / seed node
            </div>
            <div>
              <span className="similarity-dot is-truth" /> Ground-truth match
            </div>
          </div>
        </section>
        <section className="similarity-section">
          <h3>Graph</h3>
          {payload ? (
            <div className="similarity-stats" aria-label="Similarity graph counts">
              <span>
                <strong>{payload.node_count}</strong> nodes
              </span>
              <span>
                <strong>{payload.edge_count}</strong> edges
              </span>
              <span>
                <strong>{payload.embedded_node_count}</strong> embedded
              </span>
              <span>
                <strong>{payload.dropped_node_count}</strong> dropped
              </span>
              <span>
                <strong>{payload.reference_node_count}</strong> reference
              </span>
              <span>{payload.sampled ? "Sampled" : "Not sampled"}</span>
            </div>
          ) : null}
          {payload?.sampled && payload.sample_reason ? <p className="similarity-sample-note">{payload.sample_reason}</p> : null}
          {payload?.sampled && !loadFull ? (
            <button type="button" className="btn similarity-load-full" onClick={() => setLoadFull(true)}>
              Load full graph
            </button>
          ) : null}
        </section>
      </div>
      <div className="similarity-chart-wrap">
        {similarity.error ? <p className="table-error">{similarity.error.message}</p> : null}
        {!payload && similarity.isLoading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading similarity graph.</EmptyState>
          </div>
        ) : null}
        {payload ? (
          <>
            <SimilarityGraphChart
              payload={payload}
              scaling={scaling}
              threshold={threshold}
              truthLabel={truthLabel}
              ariaLabel={`Source graph of ${payload.node_count} nodes colored by embedding similarity`}
              onSelectNode={handleSelectNode}
            />
            {!hasReference ? (
              <div className="similarity-hint">
                {referenceMode === "label" ? "Choose a label and press Apply to seed a reference sample" : "Click a node (or search by id) to choose a reference"}
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
