import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Eye, Play, RotateCcw, Save, Settings2 } from "lucide-react";
import { api, type DatasetCatalogEntry, type DatasetCreatePayload, type DatasetManifest } from "../../api";
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

interface DatasetPreviewViewProps {
  activeProjectId: string;
  dataset?: DatasetManifest;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function catalogSize(entry: DatasetCatalogEntry): string {
  return `${formatCount(entry.graph_count)} graphs, ${formatCount(entry.node_count)} nodes, ${formatCount(entry.edge_count)} edges`;
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

export function DatasetPreviewView({ activeProjectId, dataset }: DatasetPreviewViewProps) {
  const [tab, setTab] = useState<"nodes" | "edges" | "mapping">("nodes");
  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", dataset?.id, "preview", tab],
    queryFn: () => api.datasetPreview(activeProjectId, dataset!.id, tab),
    enabled: Boolean(activeProjectId && dataset?.id && dataset.status === "completed")
  });

  if (!dataset) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a dataset.</EmptyState>
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
            <FcIcon name="datasets" size={16} />
            {dataset.name} Preview
          </span>
          <span className="muted">{dataset.prepared_stats ? `${dataset.prepared_stats.node_count} nodes` : dataset.status}</span>
        </header>
        <div className="tab-strip">
          {(["nodes", "edges", "mapping"] as const).map((item) => (
            <button key={item} type="button" className={`tab-btn ${tab === item ? "is-active" : ""}`} onClick={() => setTab(item)}>
              {item === "mapping" ? "Node Mapping" : item}
            </button>
          ))}
        </div>
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
