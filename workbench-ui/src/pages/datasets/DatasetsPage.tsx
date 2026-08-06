import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import JSZip from "jszip";
import { ChevronLeft, Database, Download, Eye, Play, Plus, RotateCcw, Trash2, Upload } from "lucide-react";
import {
  api,
  type DatasetCatalogEntry,
  type DatasetCreatePayload,
  type DatasetGraphSearchResult,
  type DatasetGraphSummary,
  type DatasetIntakePayload,
  type DatasetIntakeValidationResponse,
  type DatasetManifest,
  type DatasetPreviewTable,
  type TabularPreview
} from "../../api";
import {
  DEFAULT_EGONET_PARAMS,
  EgonetParamsFields,
  egonetParamsPayload,
  type EgonetParamsState
} from "./EgonetParamsFields";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { DatasetGraphTab, formatCount, formatValue } from "./DatasetGraphTab";

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
const DATASET_SINGLE_GRAPH_INTAKE_TABLES = ["edges", "nodes"] as const;
const REQUIRED_SINGLE_GRAPH_INTAKE_TABLES = new Set<string>(["edges"]);
type DatasetIntakeTableName = (typeof DATASET_INTAKE_TABLES)[number];
type DatasetImportTableName = DatasetIntakeTableName | "nodes";
type DatasetImportMode = "graph_collection" | "single_graph";

const DATASET_INTAKE_LABELS: Record<DatasetImportTableName, string> = {
  edges: "edges.csv",
  node_graph_mapping: "node_graph_mapping.csv",
  graph_labels: "graph_labels.csv",
  node_features: "node_features.csv",
  edge_features: "edge_features.csv",
  nodes: "nodes.csv"
};

function intakeTableNameFromPath(path: string, allowedTables: readonly DatasetImportTableName[]): DatasetImportTableName | null {
  const fileName = path.split(/[\\/]/).pop()?.trim().toLowerCase() || "";
  if (!fileName.endsWith(".csv")) return null;
  const stem = fileName.replace(/\.csv$/, "");
  return allowedTables.includes(stem as DatasetImportTableName) ? (stem as DatasetImportTableName) : null;
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
  const [importMode, setImportMode] = useState<DatasetImportMode>("graph_collection");
  const [graphType, setGraphType] = useState<"networkx" | "igraph">("networkx");
  const [filterLargestComponent, setFilterLargestComponent] = useState(true);
  const [egonetParams, setEgonetParams] = useState<EgonetParamsState>(DEFAULT_EGONET_PARAMS);
  const [tables, setTables] = useState<Partial<Record<DatasetImportTableName, string>>>({});
  const [fileError, setFileError] = useState("");
  const [validation, setValidation] = useState<DatasetIntakeValidationResponse | null>(null);

  const isSingleGraph = importMode === "single_graph";
  const activeTables = isSingleGraph ? DATASET_SINGLE_GRAPH_INTAKE_TABLES : DATASET_INTAKE_TABLES;
  const requiredTables = isSingleGraph ? REQUIRED_SINGLE_GRAPH_INTAKE_TABLES : REQUIRED_DATASET_INTAKE_TABLES;
  const missingRequiredTables = activeTables.filter((table) => requiredTables.has(table) && !tables[table]);
  const hasRequiredTables = missingRequiredTables.length === 0;
  const sourceNodeIds = parseSourceNodeIds(egonetParams.sourceNodeIdsText);

  const patchEgonetParams = (patch: Partial<EgonetParamsState>) => {
    setEgonetParams((current) => ({ ...current, ...patch }));
    resetValidation();
  };

  const nodeAttributeColumns = useMemo(() => {
    const csv = tables.nodes;
    if (!csv) return [] as string[];
    const firstLine = csv.split(/\r?\n/, 1)[0] || "";
    return firstLine
      .split(",")
      .map((column) => column.trim().replace(/^"|"$/g, ""))
      .filter((column) => column && column !== "node_id");
  }, [tables]);

  const buildPayload = (): DatasetIntakePayload => ({
    name: name.trim(),
    description,
    source_graph_shape: importMode,
    tables: Object.fromEntries(
      Object.entries(tables).map(([table, csv]) => [
        table,
        {
          format: "csv" as const,
          csv
        }
      ])
    ) as DatasetIntakePayload["tables"],
    params: isSingleGraph
      ? egonetParamsPayload(egonetParams, filterLargestComponent, sourceNodeIds)
      : {
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
    setEgonetParams((current) => ({ ...current, targetNodeAttribute: "" }));
    if (!files?.length) {
      setTables({});
      return;
    }

    const nextTables: Partial<Record<DatasetImportTableName, string>> = {};
    const rejected: string[] = [];
    for (const file of Array.from(files)) {
      if (file.name.toLowerCase().endsWith(".zip")) {
        const zip = await JSZip.loadAsync(file);
        for (const [path, entry] of Object.entries(zip.files)) {
          if (entry.dir) continue;
          const tableName = intakeTableNameFromPath(path, activeTables);
          if (!tableName) {
            if (path.toLowerCase().endsWith(".csv")) rejected.push(path);
            continue;
          }
          nextTables[tableName] = await entry.async("string");
        }
        continue;
      }

      const tableName = intakeTableNameFromPath(file.name, activeTables);
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

  const canValidate = Boolean(
    activeProjectId &&
      name.trim() &&
      hasRequiredTables &&
      (!isSingleGraph || egonetParams.nodeSelection !== "specific_node_ids" || sourceNodeIds.length > 0) &&
      !validateImport.isPending &&
      !createDataset.isPending
  );
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
            <span>Import Mode</span>
            <select
              value={importMode}
              onChange={(event) => {
                setImportMode(event.target.value as DatasetImportMode);
                setTables({});
                setFileError("");
                setEgonetParams((current) => ({ ...current, targetNodeAttribute: "" }));
                resetValidation();
              }}
            >
              <option value="graph_collection">Graph collection</option>
              <option value="single_graph">Single graph</option>
            </select>
          </label>
          {!isSingleGraph ? (
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
          ) : null}
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
          {isSingleGraph ? (
            <>
              <EgonetParamsFields state={egonetParams} onChange={patchEgonetParams} attributeOptions={nodeAttributeColumns} />
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
            </>
          ) : (
            <>
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
            </>
          )}
        </div>

        <section className="dataset-intake-contract">
          <header>
            <strong>NEExT Tables</strong>
            <span className="muted">
              {isSingleGraph
                ? "All nodes belong to one graph; extra nodes.csv columns become node attributes; isolated nodes require nodes.csv."
                : "Node IDs must be integer-compatible; graph labels use graph_label."}
            </span>
          </header>
          <div className="dataset-intake-table-list">
            {activeTables.map((table) => {
              const loaded = Boolean(tables[table]);
              const required = requiredTables.has(table);
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
  const [egonetParams, setEgonetParams] = useState<EgonetParamsState>(DEFAULT_EGONET_PARAMS);
  const patchEgonetParams = (patch: Partial<EgonetParamsState>) => setEgonetParams((current) => ({ ...current, ...patch }));

  useEffect(() => {
    if (!draft) return;
    const nextGraphType = draftString(draft, "graph_type");
    if (nextGraphType === "networkx" || nextGraphType === "igraph") setGraphType(nextGraphType);
    const nextFilter = draftBoolean(draft, "filter_largest_component");
    if (nextFilter !== undefined) setFilterLargestComponent(nextFilter);
    const patch: Partial<EgonetParamsState> = {};
    const nextMethod = draftString(draft, "egonet_method");
    if (nextMethod === "k_hop" || nextMethod === "random_walk") patch.egonetMethod = nextMethod;
    const nextKHop = draftNumber(draft, "k_hop");
    if (nextKHop !== undefined) patch.kHop = nextKHop;
    const nextWalkLength = draftNumber(draft, "walk_length");
    if (nextWalkLength !== undefined) patch.walkLength = nextWalkLength;
    const nextNWalks = draftNumber(draft, "n_walks");
    if (nextNWalks !== undefined) patch.nWalks = nextNWalks;
    const nextRestartProb = draftNumber(draft, "restart_prob");
    if (nextRestartProb !== undefined) patch.restartProb = nextRestartProb;
    const nextNodeSelection = draftString(draft, "node_selection");
    if (nextNodeSelection === "all_nodes" || nextNodeSelection === "sample_fraction" || nextNodeSelection === "specific_node_ids") {
      patch.nodeSelection = nextNodeSelection;
    }
    const nextSampleFraction = draftNumber(draft, "sample_fraction");
    if (nextSampleFraction !== undefined) patch.sampleFraction = nextSampleFraction;
    const nextRandomSeed = draftNumber(draft, "random_seed");
    if (nextRandomSeed !== undefined) patch.randomSeed = nextRandomSeed;
    const sourceNodeIds = draft?.source_node_ids;
    if (Array.isArray(sourceNodeIds)) patch.sourceNodeIdsText = sourceNodeIds.map(String).join("\n");
    const nextTargetAttribute = draftString(draft, "target_node_attribute");
    if (nextTargetAttribute !== undefined) patch.targetNodeAttribute = nextTargetAttribute;
    setEgonetParams((current) => ({ ...current, ...patch }));
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

  const sourceNodeIds = parseSourceNodeIds(egonetParams.sourceNodeIdsText);
  const isSingleGraph = entry.source_graph_shape === "single_graph";
  const canSave = Boolean(
    activeProjectId &&
      !createDataset.isPending &&
      (!isSingleGraph || egonetParams.nodeSelection !== "specific_node_ids" || sourceNodeIds.length > 0)
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
            params: egonetParamsPayload(egonetParams, filterLargestComponent, sourceNodeIds)
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
              <EgonetParamsFields
                state={egonetParams}
                onChange={patchEgonetParams}
                attributeOptions={entry.node_attribute_columns}
              />
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={filterLargestComponent}
                  onChange={(event) => setFilterLargestComponent(event.target.checked)}
                />
                <span>Filter Largest Component</span>
              </label>
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
                          <th>Method</th>
                          <td>{analysis.data.egonet_metadata.egonet_method === "random_walk" ? "random walk" : "k-hop"}</td>
                        </tr>
                        {analysis.data.egonet_metadata.egonet_method === "random_walk" ? (
                          <>
                            <tr>
                              <th>Walks x Length</th>
                              <td>
                                {formatCount(analysis.data.egonet_metadata.n_walks ?? 0)} x{" "}
                                {formatCount(analysis.data.egonet_metadata.walk_length ?? 0)}
                              </td>
                            </tr>
                            <tr>
                              <th>Restart Probability</th>
                              <td>{analysis.data.egonet_metadata.restart_prob}</td>
                            </tr>
                          </>
                        ) : (
                          <tr>
                            <th>K-Hop</th>
                            <td>{formatCount(analysis.data.egonet_metadata.k_hop ?? 0)}</td>
                          </tr>
                        )}
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
          <DatasetGraphTab
            activeProjectId={activeProjectId}
            datasetId={dataset.id}
            analysis={analysis.data}
            graphSummaries={graphSummaries}
            selectedGraphIndex={selectedGraphIndex}
            selectedSummary={selectedSummary}
            exploreNodeId={exploreNodeId}
            selectedNodeOutsideSample={selectedNodeOutsideSample}
            graphSearchQuery={graphSearchQuery}
            onGraphSearchQueryChange={setGraphSearchQuery}
            onSelectGraphByIndex={selectGraphByIndex}
            onSelectSearchResult={selectSearchResult}
            onExploreGraphChange={onExploreGraphChange}
            onExploreNodeChange={onExploreNodeChange}
          />
        ) : null}
        {tab === "data" ? <DatasetDataTab activeProjectId={activeProjectId} dataset={dataset} /> : null}
      </section>
    </div>
  );
}
