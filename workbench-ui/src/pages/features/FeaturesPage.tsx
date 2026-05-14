import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, Play, RotateCcw, Save, Settings2, Sigma } from "lucide-react";
import {
  api,
  type DatasetManifest,
  type FeatureCatalogEntry,
  type FeatureCreatePayload,
  type FeatureManifest
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface FeatureLibraryViewProps {
  catalog: FeatureCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureFeatureViewProps {
  activeProjectId: string;
  feature?: FeatureCatalogEntry;
  datasets: DatasetManifest[];
  loading: boolean;
  onCreated: (featureId: string) => void;
}

interface ProjectFeaturesViewProps {
  activeProjectId: string;
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: FeatureCatalogEntry[];
  loading: boolean;
  selectedFeatureId: string;
  onSelectFeature: (featureId: string) => void;
  onPreviewFeature: (featureId: string) => void;
}

interface FeaturePreviewViewProps {
  activeProjectId: string;
  feature?: FeatureManifest;
}

function featureTypeLabel(entry: FeatureCatalogEntry): string {
  return entry.type === "structural_node_feature" ? "Structural node feature" : entry.type;
}

function datasetInputId(feature: FeatureManifest): string {
  return feature.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset")?.artifact_id || "";
}

export function FeatureLibraryView({
  catalog,
  loading,
  selectedCatalogId,
  onSelectCatalog,
  onConfigure
}: FeatureLibraryViewProps) {
  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="library" size={16} />
            Feature Library · {catalog.length} {catalog.length === 1 ? "feature" : "features"}
          </span>
          <span className="muted">Define-only structural features</span>
        </header>
        {loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading feature library.</EmptyState>
          </div>
        ) : catalog.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No feature catalog entries.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Type</th>
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
                    <td className="muted">{featureTypeLabel(entry)}</td>
                    <td>{entry.output}</td>
                    <td className="actions-cell actions-cell-wide">
                      <button
                        type="button"
                        className="btn"
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

export function ConfigureFeatureView({ activeProjectId, feature, datasets, loading, onCreated }: ConfigureFeatureViewProps) {
  const queryClient = useQueryClient();
  const [datasetId, setDatasetId] = useState("");
  const [featureVectorLength, setFeatureVectorLength] = useState(3);
  const [normalizeFeatures, setNormalizeFeatures] = useState(true);
  const [nJobs, setNJobs] = useState(1);
  const [parallelBackend, setParallelBackend] = useState<"loky" | "threading">("loky");
  const datasetIdsKey = useMemo(() => datasets.map((dataset) => dataset.id).join("|"), [datasets]);

  useEffect(() => {
    setDatasetId(datasets.length === 1 ? datasets[0].id : "");
  }, [activeProjectId, datasetIdsKey, datasets, feature?.id]);

  const createFeature = useMutation({
    mutationFn: (payload: FeatureCreatePayload) => api.createFeature(activeProjectId, payload),
    onSuccess: (created) => {
      queryClient.setQueryData<FeatureManifest[]>(["projects", activeProjectId, "features"], (current = []) => [
        created,
        ...current.filter((featureItem) => featureItem.id !== created.id)
      ]);
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      onCreated(created.id);
    }
  });

  if (!feature) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a feature library entry.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  const noDatasets = !loading && datasets.length === 0;
  const paramsValid =
    Number.isInteger(featureVectorLength) && featureVectorLength >= 1 && featureVectorLength <= 10 && Number.isInteger(nJobs) && nJobs >= 1 && nJobs <= 32;
  const canSave = Boolean(activeProjectId && datasetId && paramsValid && !createFeature.isPending);
  const saveMessage = !activeProjectId
    ? "An active project is required."
    : noDatasets
      ? "A dataset is required before saving."
      : datasets.length > 1 && !datasetId
        ? "Choose a dataset."
        : !paramsValid
          ? "Feature vector length must be 1-10 and parallel jobs must be 1-32."
        : "";

  return (
    <form
      className="card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSave) return;
        createFeature.mutate({
          source_dataset_id: datasetId,
          source_feature_id: feature.id,
          params: {
            feature_vector_length: featureVectorLength,
            normalize_features: normalizeFeatures,
            n_jobs: nJobs,
            parallel_backend: parallelBackend
          }
        });
      }}
    >
      <header className="card-head">
        <span className="card-head-fc">
          <FcIcon name="features" size={32} />
        </span>
        <div>
          <h3>Configure {feature.name}</h3>
          <p className="form-subtitle">{feature.description}</p>
        </div>
      </header>
      <div className="card-body">
        {createFeature.error ? <p className="error-text">{createFeature.error.message}</p> : null}
        {saveMessage ? <p className="muted form-note">{saveMessage}</p> : null}
        <div className="field-grid">
          <label className="field field-wide">
            <span>Dataset</span>
            <select
              value={datasetId}
              onChange={(event) => setDatasetId(event.target.value)}
              disabled={!activeProjectId || loading || noDatasets}
            >
              {datasets.length === 1 ? null : <option value="">Choose dataset</option>}
              {noDatasets ? <option value="">No datasets available</option> : null}
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Feature Vector Length</span>
            <input
              type="number"
              min={1}
              max={10}
              value={featureVectorLength}
              onChange={(event) => setFeatureVectorLength(Number(event.target.value))}
            />
          </label>
          <label className="field">
            <span>Parallel Jobs</span>
            <input type="number" min={1} max={32} value={nJobs} onChange={(event) => setNJobs(Number(event.target.value))} />
          </label>
          <label className="field">
            <span>Parallel Backend</span>
            <select value={parallelBackend} onChange={(event) => setParallelBackend(event.target.value as "loky" | "threading")}>
              <option value="loky">loky</option>
              <option value="threading">threading</option>
            </select>
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={normalizeFeatures}
              onChange={(event) => setNormalizeFeatures(event.target.checked)}
            />
            <span>Normalize Features</span>
          </label>
        </div>
      </div>
      <footer className="card-foot">
        <button type="submit" className="btn btn-primary" disabled={!canSave}>
          <Save />
          {createFeature.isPending ? "Saving" : "Save"}
        </button>
      </footer>
    </form>
  );
}

export function ProjectFeaturesView({
  activeProjectId,
  features,
  datasets,
  catalog,
  loading,
  selectedFeatureId,
  onSelectFeature,
  onPreviewFeature
}: ProjectFeaturesViewProps) {
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const [checkedFeatureIds, setCheckedFeatureIds] = useState<string[]>([]);
  const queryClient = useQueryClient();
  const runFeature = useMutation({
    mutationFn: (featureId: string) => api.runFeature(activeProjectId, featureId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  });
  const runBatch = useMutation({
    mutationFn: (featureIds: string[]) => api.runFeatureBatch(activeProjectId, featureIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "datasets"] });
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "jobs"] });
    }
  });

  useEffect(() => {
    setCheckedFeatureIds((current) => current.filter((featureId) => features.some((feature) => feature.id === featureId)));
  }, [features]);

  const runnableCheckedFeatureIds = checkedFeatureIds.filter((featureId) => {
    const feature = features.find((item) => item.id === featureId);
    return feature?.status === "planned" || feature?.status === "failed";
  });

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="features" size={16} />
            Features · {features.length} {features.length === 1 ? "feature" : "features"}
          </span>
          <span className="muted">{activeProjectId ? "Active project" : "No active project"}</span>
        </header>
        {runFeature.error ? <p className="table-error">{runFeature.error.message}</p> : null}
        {runBatch.error ? <p className="table-error">{runBatch.error.message}</p> : null}
        {!activeProjectId ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No active project.</EmptyState>
          </div>
        ) : loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading features.</EmptyState>
          </div>
        ) : features.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No features.</EmptyState>
          </div>
        ) : (
          <>
          <div className="table-toolbar">
            <button
              type="button"
              className="btn"
              disabled={runnableCheckedFeatureIds.length === 0 || runBatch.isPending}
              onClick={() => runBatch.mutate(runnableCheckedFeatureIds)}
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
                  <th>Method</th>
                  <th>Status</th>
                  <th>Updated</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {features.map((feature) => {
                  const datasetName = datasetsById.get(datasetInputId(feature))?.name || "Unknown dataset";
                  const methodName = catalogById.get(feature.source_feature_id)?.name || feature.source_feature_id;
                  const isRunnable = feature.status === "planned" || feature.status === "failed";
                  const isRunning = feature.status === "running" || (runFeature.isPending && runFeature.variables === feature.id);
                  const isChecked = checkedFeatureIds.includes(feature.id);
                  return (
                    <tr
                      key={feature.id}
                      className={feature.id === selectedFeatureId ? "is-selected" : ""}
                      onClick={() => onSelectFeature(feature.id)}
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          disabled={!isRunnable}
                          onChange={(event) => {
                            event.stopPropagation();
                            setCheckedFeatureIds((current) =>
                              event.target.checked ? [...current, feature.id] : current.filter((featureId) => featureId !== feature.id)
                            );
                          }}
                          onClick={(event) => event.stopPropagation()}
                        />
                      </td>
                      <td>
                        <span className="table-name-with-icon">
                          <Sigma />
                          <strong>{feature.name}</strong>
                        </span>
                      </td>
                      <td>{datasetName}</td>
                      <td>{methodName}</td>
                      <td>
                        <span className={`status-pill ${feature.status === "completed" ? "is-ready" : "is-idle"}`}>{feature.status}</span>
                      </td>
                      <td className="muted mono">{feature.updated_at}</td>
                      <td className="actions-cell actions-cell-wide">
                        {isRunnable ? (
                          <button
                            type="button"
                            className="btn"
                            disabled={isRunning}
                            onClick={(event) => {
                              event.stopPropagation();
                              runFeature.mutate(feature.id);
                            }}
                          >
                            {feature.status === "failed" ? <RotateCcw /> : <Play />}
                            {feature.status === "failed" ? "Retry" : isRunning ? "Running" : "Run"}
                          </button>
                        ) : null}
                        {feature.status === "completed" ? (
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              onSelectFeature(feature.id);
                              onPreviewFeature(feature.id);
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

export function FeaturePreviewView({ activeProjectId, feature }: FeaturePreviewViewProps) {
  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "features", feature?.id, "preview"],
    queryFn: () => api.featurePreview(activeProjectId, feature!.id),
    enabled: Boolean(activeProjectId && feature?.id && feature.status === "completed")
  });

  if (!feature) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <div className="artifact-table-empty">
            <EmptyState compact>Select a feature.</EmptyState>
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
            <FcIcon name="features" size={16} />
            {feature.name} Preview
          </span>
          <span className="muted">{feature.output_stats ? `${feature.output_stats.row_count} rows` : feature.status}</span>
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
