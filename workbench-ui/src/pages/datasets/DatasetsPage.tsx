import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import JSZip from "jszip";
import { ChevronLeft, ChevronRight, Database, Download, Eye, Play, Plus, RotateCcw, Search, Trash2, Upload } from "lucide-react";
import {
  api,
  type DatasetCatalogEntry,
  type DatasetCreatePayload,
  type DatasetGraphSearchResult,
  type DatasetGraphSummary,
  type DatasetGraphVisual,
  type DatasetIntakePayload,
  type DatasetIntakeValidationResponse,
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
  draft?: Record<string, unknown>;
  onBack: () => void;
  onCreated: (datasetId: string) => void;
}

interface DatasetImportViewProps {
  activeProjectId: string;
  onCreated: (datasetId: string) => void;
}

interface ProjectDatasetsViewProps {
  activeProjectId: string;
  datasets: DatasetManifest[];
  loading: boolean;
  selectedDatasetId: string;
  onSelectDataset: (datasetId: string) => void;
  onPreviewDataset: (datasetId: string) => void;
  onDeleteArtifact: (artifactKind: "dataset", artifactId: string) => void;
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
  onBackToDatasets: () => void;
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

function sourceTypeLabel(entry: DatasetCatalogEntry): string {
  return entry.source_graph_shape === "single_graph" ? "Single Graph" : "Collection";
}

function sourceTypeClass(entry: DatasetCatalogEntry): string {
  return entry.source_graph_shape === "single_graph" ? "is-single-graph" : "is-collection";
}

function parseSourceNodeIds(value: string): string[] {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatValue(value: unknown): string {
  if (value == null) return "None";
  return String(value);
}

function formatAverage(total: number, count: number): string {
  if (!count) return "0";
  return (total / count).toFixed(1);
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

const DATASET_INTAKE_TABLES = ["edges", "node_graph_mapping", "graph_labels", "node_features", "edge_features"] as const;
const REQUIRED_DATASET_INTAKE_TABLES = new Set<string>(["edges", "node_graph_mapping"]);
type DatasetIntakeTableName = (typeof DATASET_INTAKE_TABLES)[number];

const DATASET_INTAKE_LABELS: Record<DatasetIntakeTableName, string> = {
  edges: "edges.csv",
  node_graph_mapping: "node_graph_mapping.csv",
  graph_labels: "graph_labels.csv",
  node_features: "node_features.csv",
  edge_features: "edge_features.csv"
};

const GRAPH_LABEL_COLORS = [
  { background: "#e9f5ff", borderColor: "#87bde8", color: "#155f8e" },
  { background: "#eaf7ef", borderColor: "#82c89a", color: "#236f3b" },
  { background: "#fff3d6", borderColor: "#e3b85f", color: "#7a5510" },
  { background: "#f0e9ff", borderColor: "#ad93e5", color: "#5c3ea0" },
  { background: "#ffe9ef", borderColor: "#e99aae", color: "#8f2d49" },
  { background: "#eaf7f6", borderColor: "#7fc7c1", color: "#1e6d67" }
];

function intakeTableNameFromPath(path: string): DatasetIntakeTableName | null {
  const fileName = path.split(/[\\/]/).pop()?.trim().toLowerCase() || "";
  if (!fileName.endsWith(".csv")) return null;
  const stem = fileName.replace(/\.csv$/, "");
  return DATASET_INTAKE_TABLES.includes(stem as DatasetIntakeTableName) ? (stem as DatasetIntakeTableName) : null;
}

function graphLabelStyle(label: unknown): CSSProperties {
  const text = formatValue(label);
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return GRAPH_LABEL_COLORS[hash % GRAPH_LABEL_COLORS.length];
}

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

export function DatasetImportView({ activeProjectId, onCreated }: DatasetImportViewProps) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [graphType, setGraphType] = useState<"networkx" | "igraph">("networkx");
  const [filterLargestComponent, setFilterLargestComponent] = useState(true);
  const [tables, setTables] = useState<Partial<Record<DatasetIntakeTableName, string>>>({});
  const [fileError, setFileError] = useState("");
  const [validation, setValidation] = useState<DatasetIntakeValidationResponse | null>(null);

  const missingRequiredTables = DATASET_INTAKE_TABLES.filter((table) => REQUIRED_DATASET_INTAKE_TABLES.has(table) && !tables[table]);
  const hasRequiredTables = missingRequiredTables.length === 0;

  const buildPayload = (): DatasetIntakePayload => ({
    name: name.trim(),
    description,
    tables: Object.fromEntries(
      Object.entries(tables).map(([table, csv]) => [
        table,
        {
          format: "csv" as const,
          csv
        }
      ])
    ) as DatasetIntakePayload["tables"],
    params: {
      graph_type: graphType,
      filter_largest_component: filterLargestComponent
    }
  });

  const validateImport = useMutation({
    mutationFn: () => api.validateDatasetIntake(activeProjectId, buildPayload()),
    onSuccess: (result) => setValidation(result)
  });
  const createDataset = useMutation({
    mutationFn: () => api.createDatasetFromIntake(activeProjectId, buildPayload()),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      onCreated(created.id);
    }
  });

  const resetValidation = () => {
    setValidation(null);
    validateImport.reset();
    createDataset.reset();
  };

  const parseFiles = async (files: FileList | null) => {
    resetValidation();
    setFileError("");
    if (!files?.length) {
      setTables({});
      return;
    }

    const nextTables: Partial<Record<DatasetIntakeTableName, string>> = {};
    const rejected: string[] = [];
    for (const file of Array.from(files)) {
      if (file.name.toLowerCase().endsWith(".zip")) {
        const zip = await JSZip.loadAsync(file);
        for (const [path, entry] of Object.entries(zip.files)) {
          if (entry.dir) continue;
          const tableName = intakeTableNameFromPath(path);
          if (!tableName) {
            if (path.toLowerCase().endsWith(".csv")) rejected.push(path);
            continue;
          }
          nextTables[tableName] = await entry.async("string");
        }
        continue;
      }

      const tableName = intakeTableNameFromPath(file.name);
      if (!tableName) {
        rejected.push(file.name);
        continue;
      }
      nextTables[tableName] = await file.text();
    }

    setTables(nextTables);
    if (Object.keys(nextTables).length === 0) {
      setFileError("No NEExT table CSV files were found.");
    } else if (rejected.length) {
      setFileError(`Ignored unsupported CSV file names: ${rejected.slice(0, 4).join(", ")}${rejected.length > 4 ? ", ..." : ""}`);
    }
  };

  const canValidate = Boolean(activeProjectId && name.trim() && hasRequiredTables && !validateImport.isPending && !createDataset.isPending);
  const canCreate = Boolean(canValidate && validation?.valid && !createDataset.isPending);

  return (
    <form
      className="card dataset-import-card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canCreate) return;
        createDataset.mutate();
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="import" size={32} />
        </span>
        <div>
          <h3>Import Dataset</h3>
          <p className="form-subtitle">Create a Draft Dataset from NEExT table CSV files.</p>
        </div>
      </header>
      <div className="card-body">
        {!activeProjectId ? <p className="muted form-note">An active project is required.</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Name</span>
            <input
              value={name}
              onChange={(event) => {
                setName(event.target.value);
                resetValidation();
              }}
              placeholder="Dataset name"
            />
          </label>
          <label className="field">
            <span>Graph Backend</span>
            <select
              value={graphType}
              onChange={(event) => {
                setGraphType(event.target.value as "networkx" | "igraph");
                resetValidation();
              }}
            >
              <option value="networkx">networkx</option>
              <option value="igraph">igraph</option>
            </select>
          </label>
          <label className="field field-wide">
            <span>Description</span>
            <textarea
              value={description}
              rows={3}
              onChange={(event) => {
                setDescription(event.target.value);
                resetValidation();
              }}
              placeholder="Dataset description"
            />
          </label>
          <label className="field field-wide">
            <span>CSV Files or Zip Bundle</span>
            <input
              type="file"
              accept=".csv,.zip,text/csv,application/zip"
              multiple
              onChange={(event) => {
                parseFiles(event.target.files).catch((error) => {
                  setTables({});
                  setFileError(error instanceof Error ? error.message : String(error));
                });
              }}
            />
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={filterLargestComponent}
              onChange={(event) => {
                setFilterLargestComponent(event.target.checked);
                resetValidation();
              }}
            />
            <span>Filter Largest Component</span>
          </label>
          <label className="checkbox-field">
            <input type="checkbox" checked readOnly />
            <span>Reindex Nodes</span>
          </label>
        </div>

        <section className="dataset-intake-contract">
          <header>
            <strong>NEExT Tables</strong>
            <span className="muted">Node IDs must be integer-compatible; graph labels use graph_label.</span>
          </header>
          <div className="dataset-intake-table-list">
            {DATASET_INTAKE_TABLES.map((table) => {
              const loaded = Boolean(tables[table]);
              const required = REQUIRED_DATASET_INTAKE_TABLES.has(table);
              return (
                <div className="dataset-intake-table-row" key={table}>
                  <span className="mono">{DATASET_INTAKE_LABELS[table]}</span>
                  <span className={`status-pill ${loaded ? "is-ready" : "is-idle"}`}>{loaded ? "loaded" : required ? "required" : "optional"}</span>
                </div>
              );
            })}
          </div>
        </section>

        {fileError ? <p className="table-error">{fileError}</p> : null}
        {missingRequiredTables.length ? (
          <p className="table-note">Missing required tables: {missingRequiredTables.map((table) => DATASET_INTAKE_LABELS[table]).join(", ")}.</p>
        ) : null}
        {validateImport.error ? <p className="table-error">{validateImport.error.message}</p> : null}
        {createDataset.error ? <p className="table-error">{createDataset.error.message}</p> : null}
        {validation ? (
          <section className={`dataset-intake-validation ${validation.valid ? "is-valid" : "is-invalid"}`}>
            <strong>{validation.valid ? "Validation passed" : "Validation failed"}</strong>
            {validation.stats ? (
              <span className="muted">
                {formatCount(validation.stats.graph_count)} graphs · {formatCount(validation.stats.node_count)} nodes ·{" "}
                {formatCount(validation.stats.edge_count)} edges
              </span>
            ) : null}
            {validation.errors.length ? (
              <ul>
                {validation.errors.slice(0, 6).map((error, index) => (
                  <li key={`${error.table}-${error.column || ""}-${index}`}>
                    <span className="mono">{error.table}</span> {error.message}
                  </li>
                ))}
              </ul>
            ) : null}
          </section>
        ) : null}
      </div>
      <footer className="card-foot">
        <button type="button" className="btn" onClick={() => validateImport.mutate()} disabled={!canValidate}>
          <Upload />
          {validateImport.isPending ? "Validating" : "Validate"}
        </button>
        <button type="submit" className="btn btn-primary" disabled={!canCreate}>
          <Plus />
          {createDataset.isPending ? "Creating" : "Create Dataset"}
        </button>
      </footer>
    </form>
  );
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
          <span className="muted">{activeProjectId ? "Templates for new project Dataset artifacts" : "No active project"}</span>
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
                  <th>Type</th>
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
                      <td>
                        <span className={`source-type-pill ${sourceTypeClass(entry)}`}>{sourceTypeLabel(entry)}</span>
                      </td>
                      <td className="muted">{entry.source}</td>
                      <td>{catalogSize(entry)}</td>
                      <td>
                        <span className={`status-pill ${isConfigured ? "is-ready" : "is-idle"}`}>
                          {isConfigured ? "Added" : "Available"}
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
                          <Plus />
                          Add to Project
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

export function ConfigureDatasetView({ activeProjectId, entry, draft, onBack, onCreated }: ConfigureDatasetViewProps) {
  const queryClient = useQueryClient();
  const [graphType, setGraphType] = useState<"networkx" | "igraph">("networkx");
  const [filterLargestComponent, setFilterLargestComponent] = useState(true);
  const [kHop, setKHop] = useState(1);
  const [nodeSelection, setNodeSelection] = useState<"all_nodes" | "sample_fraction" | "specific_node_ids">("all_nodes");
  const [sampleFraction, setSampleFraction] = useState(1);
  const [randomSeed, setRandomSeed] = useState(13);
  const [sourceNodeIdsText, setSourceNodeIdsText] = useState("");
  const [targetNodeAttribute, setTargetNodeAttribute] = useState("");

  useEffect(() => {
    if (!draft) return;
    const nextGraphType = draftString(draft, "graph_type");
    if (nextGraphType === "networkx" || nextGraphType === "igraph") setGraphType(nextGraphType);
    const nextFilter = draftBoolean(draft, "filter_largest_component");
    if (nextFilter !== undefined) setFilterLargestComponent(nextFilter);
    const nextKHop = draftNumber(draft, "k_hop");
    if (nextKHop !== undefined) setKHop(nextKHop);
    const nextNodeSelection = draftString(draft, "node_selection");
    if (nextNodeSelection === "all_nodes" || nextNodeSelection === "sample_fraction" || nextNodeSelection === "specific_node_ids") {
      setNodeSelection(nextNodeSelection);
    }
    const nextSampleFraction = draftNumber(draft, "sample_fraction");
    if (nextSampleFraction !== undefined) setSampleFraction(nextSampleFraction);
    const nextRandomSeed = draftNumber(draft, "random_seed");
    if (nextRandomSeed !== undefined) setRandomSeed(nextRandomSeed);
    const sourceNodeIds = draft?.source_node_ids;
    if (Array.isArray(sourceNodeIds)) setSourceNodeIdsText(sourceNodeIds.map(String).join("\n"));
    const nextTargetAttribute = draftString(draft, "target_node_attribute");
    if (nextTargetAttribute !== undefined) setTargetNodeAttribute(nextTargetAttribute);
  }, [activeProjectId, entry?.id, draft]);

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

  const sourceNodeIds = parseSourceNodeIds(sourceNodeIdsText);
  const isSingleGraph = entry.source_graph_shape === "single_graph";
  const canSave = Boolean(
    activeProjectId &&
      !createDataset.isPending &&
      (!isSingleGraph || nodeSelection !== "specific_node_ids" || sourceNodeIds.length > 0)
  );

  return (
    <form
      className="card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSave) return;
        if (isSingleGraph) {
          createDataset.mutate({
            catalog_id: entry.id,
            params: {
              graph_type: "networkx",
              filter_largest_component: false,
              k_hop: kHop,
              node_selection: nodeSelection,
              sample_fraction: nodeSelection === "sample_fraction" ? sampleFraction : 1,
              random_seed: randomSeed,
              source_node_ids: nodeSelection === "specific_node_ids" ? sourceNodeIds : [],
              target_node_attribute: targetNodeAttribute || null
            }
          });
        } else {
          createDataset.mutate({
            catalog_id: entry.id,
            params: {
              graph_type: graphType,
              filter_largest_component: filterLargestComponent
            }
          });
        }
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="datasets" size={32} />
        </span>
        <div>
          <h3>Add {entry.name} to Project</h3>
          <p className="form-subtitle">{entry.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createDataset.error ? <p className="error-text">{createDataset.error.message}</p> : null}
        {!activeProjectId ? <p className="muted form-note">An active project is required.</p> : null}
        {isSingleGraph ? (
          <>
            <div className="stat-grid compact-stat-grid">
              <div className="stat-tile">
                <span>Type</span>
                <strong>Single Graph</strong>
                <small>{entry.domain}</small>
              </div>
              <div className="stat-tile">
                <span>Nodes</span>
                <strong>{formatCount(entry.node_count)}</strong>
                <small>{formatCount(entry.edge_count)} edges</small>
              </div>
              <div className="stat-tile">
                <span>Node Attributes</span>
                <strong>{formatCount(entry.node_attribute_columns.length)}</strong>
                <small>{entry.node_attribute_columns.length ? entry.node_attribute_columns.join(", ") : "None"}</small>
              </div>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>K-Hop</span>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={kHop}
                  onChange={(event) => setKHop(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Node Selection</span>
                <select
                  value={nodeSelection}
                  onChange={(event) => setNodeSelection(event.target.value as "all_nodes" | "sample_fraction" | "specific_node_ids")}
                >
                  <option value="all_nodes">All nodes</option>
                  <option value="sample_fraction">Sample fraction</option>
                  <option value="specific_node_ids">Specific node IDs</option>
                </select>
              </label>
              <label className="field">
                <span>Target Attribute</span>
                <select value={targetNodeAttribute} onChange={(event) => setTargetNodeAttribute(event.target.value)}>
                  <option value="">None</option>
                  {entry.node_attribute_columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
              {nodeSelection === "sample_fraction" ? (
                <>
                  <label className="field">
                    <span>Sample Fraction</span>
                    <input
                      type="number"
                      min={0.01}
                      max={1}
                      step={0.01}
                      value={sampleFraction}
                      onChange={(event) => setSampleFraction(Number(event.target.value))}
                    />
                  </label>
                  <label className="field">
                    <span>Random Seed</span>
                    <input
                      type="number"
                      min={0}
                      value={randomSeed}
                      onChange={(event) => setRandomSeed(Number(event.target.value))}
                    />
                  </label>
                </>
              ) : null}
              {nodeSelection === "specific_node_ids" ? (
                <label className="field field-wide">
                  <span>Source Node IDs</span>
                  <textarea
                    value={sourceNodeIdsText}
                    onChange={(event) => setSourceNodeIdsText(event.target.value)}
                    rows={4}
                  />
                </label>
              ) : null}
            </div>
          </>
        ) : (
          <div className="field-grid dataset-config-grid">
            <label className="field">
              <span>Graph Backend</span>
              <select value={graphType} onChange={(event) => setGraphType(event.target.value as "networkx" | "igraph")}>
                <option value="networkx">networkx</option>
                <option value="igraph">igraph</option>
              </select>
            </label>
            <div className="checkbox-stack">
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
        )}
      </div>
      <footer className="card-foot">
        <button type="button" className="btn" onClick={onBack}>
          <ChevronLeft />
          Back
        </button>
        <button type="submit" className="btn btn-primary" disabled={!canSave}>
          <Plus />
          {createDataset.isPending ? "Creating" : "Create Dataset"}
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
  onPreviewDataset,
  onDeleteArtifact
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
                      <td className="muted">{dataset.source_name || dataset.source_catalog_id}</td>
                      <td>{String(dataset.operation.params.graph_type)}</td>
                      <td>{dataset.operation.params.filter_largest_component ? "Yes" : "No"}</td>
                      <td>
                        <span className={`status-pill ${artifactStatusClass(dataset.status)}`}>{artifactStatusLabel(dataset.status)}</span>
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
                            {dataset.status === "failed" ? "Retry Prepare" : isRunning ? "Preparing" : "Prepare"}
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
                        <button
                          type="button"
                          className="icon-btn icon-btn-danger"
                          aria-label={`Delete ${dataset.name}`}
                          title={`Delete ${dataset.name}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            onDeleteArtifact("dataset", dataset.id);
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
          const data = item.data as { name?: string; value?: number; sourceNodeId?: string | null; isCenter?: boolean | null };
          if (!data.name) return "";
          const sourceLine = data.sourceNodeId && !data.isCenter ? `<br/>Source node: ${data.sourceNodeId}` : "";
          const centerLine = data.isCenter ? `<br/>Center source node: ${data.sourceNodeId || data.name}` : "";
          return `${data.name}<br/>Degree: ${data.value ?? 0}${sourceLine}${centerLine}`;
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
          data: visual.nodes.map((node) => {
            const isCenter = Boolean(node.is_center);
            return {
              id: node.id,
              name: node.label,
              value: node.degree,
              sourceNodeId: node.source_node_id,
              isCenter,
              symbolSize: 9 + (node.degree / maxDegree) * 18 + (isCenter ? 8 : 0),
              itemStyle: {
                color: isCenter ? "#f28c28" : "#176ea9",
                borderColor: isCenter ? "#8f3e00" : "#ffffff",
                borderWidth: isCenter ? 2.5 : 1
              }
            };
          }),
          links: visual.edges.map((edge) => ({ source: edge.source, target: edge.target })),
          lineStyle: {
            color: "#8a98a6",
            opacity: 0.72,
            width: 1.2
          },
          emphasis: {
            focus: "adjacency",
            itemStyle: { color: "#f28c28", borderColor: "#7a3300", borderWidth: 3 },
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
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState("");
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

  const exportTable = async () => {
    setExportError("");
    setIsExporting(true);
    try {
      const download = await api.datasetExport(activeProjectId, dataset.id, table);
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
      setIsExporting(false);
    }
  };

  return (
    <div className="dataset-tab-panel">
      <div className="table-toolbar dataset-table-toolbar">
        <label className="field compact-field dataset-table-select">
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
        <button type="button" className="btn" onClick={exportTable} disabled={isExporting || !tables.length}>
          <Download />
          {isExporting ? "Exporting" : "Export CSV"}
        </button>
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
      {exportError ? <p className="table-error">{exportError}</p> : null}
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
  onBackToDatasets,
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
                        <span className={`status-pill ${artifactStatusClass(item.status)}`}>{artifactStatusLabel(item.status)}</span>
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
              <button type="button" className="btn" onClick={onBackToDatasets}>
                <ChevronLeft />
                Back to Datasets
              </button>
            </span>
            <span className="explore-title">{dataset.name}</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Prepare this dataset before exploring it.</EmptyState>
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
            <button type="button" className="btn" onClick={onBackToDatasets}>
              <ChevronLeft />
              Back to Datasets
            </button>
          </span>
          <span className="explore-title">{dataset.name}</span>
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
          <div className="dataset-tab-panel dataset-stat-panel">
            <section className="dataset-stats-section">
              <h3>Description</h3>
              <p className="dataset-description-text">{dataset.description || "No description."}</p>
            </section>
            {analysis.data.egonet_metadata ? (
              <>
                <section className="dataset-stats-section">
                  <h3>Source Graph</h3>
                  <div className="stat-grid">
                    <div className="stat-tile">
                      <span>Graphs</span>
                      <strong>{formatCount(analysis.data.source_stats.graph_count)}</strong>
                      <small>Single graph source</small>
                    </div>
                    <div className="stat-tile">
                      <span>Nodes</span>
                      <strong>{formatCount(analysis.data.source_stats.node_count)}</strong>
                      <small>Original source nodes</small>
                    </div>
                    <div className="stat-tile">
                      <span>Edges</span>
                      <strong>{formatCount(analysis.data.source_stats.edge_count)}</strong>
                      <small>Original source edges</small>
                    </div>
                  </div>
                </section>
                <section className="dataset-stats-section">
                  <h3>Prepared Egonet Collection</h3>
                  <div className="stat-grid">
                    <div className="stat-tile">
                      <span>Egonets</span>
                      <strong>{formatCount(analysis.data.prepared_stats.graph_count)}</strong>
                      <small>Downstream graph collection</small>
                    </div>
                    <div className="stat-tile">
                      <span>Node Memberships</span>
                      <strong>{formatCount(analysis.data.prepared_stats.node_count)}</strong>
                      <small>Nodes can repeat across egonets</small>
                    </div>
                    <div className="stat-tile">
                      <span>Edge Memberships</span>
                      <strong>{formatCount(analysis.data.prepared_stats.edge_count)}</strong>
                      <small>Prepared egonet edges</small>
                    </div>
                    <div className="stat-tile">
                      <span>Dropped Source Nodes</span>
                      <strong>{formatCount(analysis.data.dropped_node_count)}</strong>
                      <small>Selection or preparation filter</small>
                    </div>
                    <div className="stat-tile">
                      <span>Nodes / Egonet</span>
                      <strong>{formatAverage(analysis.data.prepared_stats.node_count, analysis.data.prepared_stats.graph_count)}</strong>
                      <small>
                        {formatCount(Math.min(...graphCounts))} min / {formatCount(Math.max(...graphCounts))} max
                      </small>
                    </div>
                    <div className="stat-tile">
                      <span>Edges / Egonet</span>
                      <strong>{formatAverage(analysis.data.prepared_stats.edge_count, analysis.data.prepared_stats.graph_count)}</strong>
                      <small>
                        {formatCount(Math.min(...edgeCounts))} min / {formatCount(Math.max(...edgeCounts))} max
                      </small>
                    </div>
                  </div>
                  <p className="table-note">
                    Prepared node counts are egonet memberships; the same source node can appear in multiple prepared egonets.
                  </p>
                </section>
                <div className="dataset-detail-grid dataset-stat-detail-grid">
                  <section>
                    <h3>Egonet Generation</h3>
                    <table className="tbl compact-tbl">
                      <tbody>
                        <tr>
                          <th>Operation</th>
                          <td>{analysis.data.egonet_metadata.operation_id}</td>
                        </tr>
                        <tr>
                          <th>Version</th>
                          <td>{analysis.data.egonet_metadata.operation_version}</td>
                        </tr>
                        <tr>
                          <th>K-Hop</th>
                          <td>{formatCount(analysis.data.egonet_metadata.k_hop)}</td>
                        </tr>
                        <tr>
                          <th>Node Selection</th>
                          <td>{analysis.data.egonet_metadata.node_selection.replace(/_/g, " ")}</td>
                        </tr>
                      </tbody>
                    </table>
                  </section>
                  <section>
                    <h3>Selection Parameters</h3>
                    <table className="tbl compact-tbl">
                      <tbody>
                        <tr>
                          <th>Sample Fraction</th>
                          <td>{analysis.data.egonet_metadata.sample_fraction}</td>
                        </tr>
                        <tr>
                          <th>Random Seed</th>
                          <td>{analysis.data.egonet_metadata.random_seed}</td>
                        </tr>
                        <tr>
                          <th>Target Attribute</th>
                          <td>{formatValue(analysis.data.egonet_metadata.target_node_attribute)}</td>
                        </tr>
                        <tr>
                          <th>Source Shape</th>
                          <td>single graph</td>
                        </tr>
                      </tbody>
                    </table>
                  </section>
                </div>
              </>
            ) : (
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
            )}
            <div className="dataset-detail-grid dataset-stat-detail-grid">
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
                <h3>Data Columns</h3>
                <table className="tbl compact-tbl">
                  <tbody>
                    <tr>
                      <th>Node Features</th>
                      <td>{analysis.data.node_feature_columns.length ? analysis.data.node_feature_columns.join(", ") : "None"}</td>
                    </tr>
                    <tr>
                      <th>Edge Features</th>
                      <td>{analysis.data.edge_feature_columns.length ? analysis.data.edge_feature_columns.join(", ") : "None"}</td>
                    </tr>
                  </tbody>
                </table>
              </section>
            </div>
          </div>
        ) : null}
        {tab === "graph" && analysis.data ? (
          <div className="dataset-tab-panel graph-tab-panel" tabIndex={0} onKeyDown={handleGraphKeyDown} aria-label="Dataset graph view">
            <div className="dataset-graph-header">
              <div className="graph-nav-group">
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
                <span className="graph-position">
                  {selectedGraphIndex >= 0 ? `${selectedGraphIndex + 1} / ${formatCount(graphSummaries.length)}` : ""}
                </span>
              </div>
              <div className="graph-summary-band">
                {selectedSummary ? (
                  <>
                    <span className="graph-id-badge mono">Graph {selectedSummary.graph_id}</span>
                    <span className="graph-meta-badge">{formatCount(selectedSummary.node_count)} nodes</span>
                    <span className="graph-meta-badge">{formatCount(selectedSummary.edge_count)} edges</span>
                    {selectedSummary.graph_label != null ? (
                      <span className="graph-label-badge" style={graphLabelStyle(selectedSummary.graph_label)}>
                        Label {formatValue(selectedSummary.graph_label)}
                      </span>
                    ) : null}
                    {selectedSummary.source_node_id ? (
                      <span className="graph-center-badge">Center Source Node {selectedSummary.source_node_id}</span>
                    ) : null}
                    {analysis.data.egonet_metadata ? (
                      <span className="graph-meta-badge">{formatCount(analysis.data.egonet_metadata.k_hop)}-hop egonet</span>
                    ) : null}
                    {analysis.data.egonet_metadata?.target_node_attribute ? (
                      <span className="graph-meta-badge">Target {analysis.data.egonet_metadata.target_node_attribute}</span>
                    ) : null}
                  </>
                ) : null}
                {analysis.data.visual.sampled ? <span className="status-pill is-idle">sampled</span> : null}
              </div>
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
