import { useEffect, useMemo, useRef, useState, type ReactNode, type UIEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import {
  ArrowLeft,
  CheckCircle2,
  CircleHelp,
  Code2,
  ChevronLeft,
  ChevronRight,
  Eye,
  Play,
  RotateCcw,
  Save,
  Search,
  Settings2,
  Sigma,
  Trash2
} from "lucide-react";
import {
  api,
  type CustomFeatureCreatePayload,
  type CustomFeatureValidatePayload,
  type DatasetManifest,
  type FeatureAnalysis,
  type FeatureCatalogEntry,
  type FeatureCreatePayload,
  type FeatureGraphSearchResult,
  type FeatureManifest,
  type FeaturePcaPoint,
  type TabularPreview
} from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface FeatureLibraryViewProps {
  catalog: FeatureCatalogEntry[];
  loading: boolean;
  selectedCatalogId: string;
  selectedDataset?: DatasetManifest;
  onSelectCatalog: (catalogId: string) => void;
  onConfigure: (catalogId: string) => void;
}

interface ConfigureFeatureViewProps {
  activeProjectId: string;
  feature?: FeatureCatalogEntry;
  dataset?: DatasetManifest;
  onCreated: (featureId: string) => void;
}

interface CreateFeatureViewProps {
  activeProjectId: string;
  dataset?: DatasetManifest;
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
  onDeleteArtifact: (artifactKind: "feature", artifactId: string) => void;
}

interface FeatureExploreViewProps {
  activeProjectId: string;
  features: FeatureManifest[];
  datasets: DatasetManifest[];
  catalog: FeatureCatalogEntry[];
  loading: boolean;
  selectedFeatureId: string;
  exploreFeatureId: string;
  selectedGraphId: string;
  onExploreFeature: (featureId: string) => void;
  onClearExploreFeature: () => void;
  onSelectGraph: (graphId: string, visible: boolean | null) => void;
  onSelectedGraphVisibilityChange: (visible: boolean | null) => void;
}

function featureTypeLabel(entry: FeatureCatalogEntry): string {
  return entry.type === "structural_node_feature" ? "Structural node feature" : entry.type;
}

function featureMethodLabel(feature: FeatureManifest, catalogById: Map<string, FeatureCatalogEntry>): string {
  if (feature.source_type === "custom_python_node_feature") return "Custom Python";
  return catalogById.get(feature.source_feature_id)?.name || feature.source_feature_id;
}

function datasetInputId(feature: FeatureManifest): string {
  return feature.inputs.find((input) => input.role === "source_dataset" && input.artifact_kind === "dataset")?.artifact_id || "";
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

function formatAverage(covered: number, total: number): string {
  if (!total) return "0%";
  return `${((covered / total) * 100).toFixed(1)}%`;
}

export function FeatureLibraryView({
  catalog,
  loading,
  selectedCatalogId,
  selectedDataset,
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
          <span className="muted">{selectedDataset ? `Dataset: ${selectedDataset.name}` : "Select a dataset first"}</span>
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

export function ConfigureFeatureView({ activeProjectId, feature, dataset, onCreated }: ConfigureFeatureViewProps) {
  const queryClient = useQueryClient();
  const [featureVectorLength, setFeatureVectorLength] = useState(3);
  const [normalizeFeatures, setNormalizeFeatures] = useState(true);
  const [nJobs, setNJobs] = useState(1);
  const [parallelBackend, setParallelBackend] = useState<"loky" | "threading">("loky");

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

  const paramsValid =
    Number.isInteger(featureVectorLength) && featureVectorLength >= 1 && featureVectorLength <= 10 && Number.isInteger(nJobs) && nJobs >= 1 && nJobs <= 32;
  const canSave = Boolean(activeProjectId && dataset?.id && paramsValid && !createFeature.isPending);
  const saveMessage = !activeProjectId
    ? "An active project is required."
    : !dataset
      ? "Select a dataset before configuring features."
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
          source_dataset_id: dataset!.id,
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
        {dataset ? <p className="muted form-note">Dataset: {dataset.name}</p> : null}
        <div className="field-grid">
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

const CUSTOM_FEATURE_TEMPLATE = `import pandas as pd

def compute_feature(graph):
    nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
    try:
        values = [float(graph.G.degree(node)) for node in nodes]
    except TypeError:
        values = [float(value) for value in graph.G.degree(nodes)]
    df = pd.DataFrame({
        "node_id": nodes,
        "graph_id": graph.graph_id,
        "custom_degree": values,
    })
    return df[["node_id", "graph_id", "custom_degree"]]
`;

const PYTHON_KEYWORDS = new Set([
  "and",
  "as",
  "assert",
  "break",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "False",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "None",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "True",
  "try",
  "while",
  "with",
  "yield"
]);

const PYTHON_BUILTINS = new Set(["dict", "float", "int", "len", "list", "max", "min", "range", "set", "str", "sum", "tuple"]);

function renderPythonTokens(source: string): ReactNode[] {
  const tokens: ReactNode[] = [];
  let index = 0;

  const tokenPatterns: Array<[string, RegExp]> = [
    ["syntax-comment", /^#[^\n]*/],
    ["syntax-string", /^(?:[rRuUbBfF]{0,2})(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/],
    ["syntax-number", /^\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/i],
    ["syntax-decorator", /^@[A-Za-z_][A-Za-z0-9_.]*/],
    ["syntax-operator", /^[{}()[\].,:+\-*/%=<>!]+/]
  ];

  while (index < source.length) {
    const remaining = source.slice(index);
    const whitespace = remaining.match(/^\s+/);
    if (whitespace) {
      tokens.push(whitespace[0]);
      index += whitespace[0].length;
      continue;
    }

    const identifier = remaining.match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (identifier) {
      const value = identifier[0];
      const className = PYTHON_KEYWORDS.has(value) ? "syntax-keyword" : PYTHON_BUILTINS.has(value) ? "syntax-builtin" : "";
      tokens.push(
        className ? (
          <span key={index} className={className}>
            {value}
          </span>
        ) : (
          value
        )
      );
      index += value.length;
      continue;
    }

    const matched = tokenPatterns.find(([, pattern]) => pattern.test(remaining));
    if (matched) {
      const [className, pattern] = matched;
      const value = remaining.match(pattern)?.[0] || "";
      tokens.push(
        <span key={index} className={className}>
          {value}
        </span>
      );
      index += value.length;
      continue;
    }

    tokens.push(source[index]);
    index += 1;
  }

  return tokens;
}

export function CreateFeatureView({ activeProjectId, dataset, onCreated }: CreateFeatureViewProps) {
  const queryClient = useQueryClient();
  const codeHighlightRef = useRef<HTMLPreElement | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [code, setCode] = useState(CUSTOM_FEATURE_TEMPLATE);
  const [normalizeFeatures, setNormalizeFeatures] = useState(true);
  const [showGuide, setShowGuide] = useState(false);

  useEffect(() => {
    setName("");
    setDescription("");
    setCode(CUSTOM_FEATURE_TEMPLATE);
    setNormalizeFeatures(true);
  }, [activeProjectId]);

  const customFeatureParams = useMemo(
    () => ({
      normalize_features: normalizeFeatures,
      n_jobs: 1,
      parallel_backend: "threading" as const
    }),
    [normalizeFeatures]
  );

  const validateCustomFeature = useMutation({
    mutationFn: (payload: CustomFeatureValidatePayload) => api.validateCustomFeature(activeProjectId, payload)
  });

  const createCustomFeature = useMutation({
    mutationFn: (payload: CustomFeatureCreatePayload) => api.createCustomFeature(activeProjectId, payload),
    onSuccess: (created) => {
      queryClient.setQueryData<FeatureManifest[]>(["projects", activeProjectId, "features"], (current = []) => [
        created,
        ...current.filter((featureItem) => featureItem.id !== created.id)
      ]);
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      onCreated(created.id);
    }
  });

  useEffect(() => {
    validateCustomFeature.reset();
  }, [activeProjectId, code, dataset?.id, normalizeFeatures]);

  const datasetComplete = dataset?.status === "completed";
  const canValidate = Boolean(activeProjectId && dataset?.id && datasetComplete && code.trim() && !validateCustomFeature.isPending);
  const canSave = Boolean(activeProjectId && dataset?.id && datasetComplete && name.trim() && code.trim() && !createCustomFeature.isPending);
  const saveMessage = !activeProjectId
    ? "An active project is required."
    : !dataset
      ? "Select a dataset before creating a custom feature."
      : !datasetComplete
        ? `Dataset ${dataset.name} must be completed before creating custom features.`
        : !name.trim()
          ? "Name is required."
          : !code.trim()
            ? "Python code is required."
            : "";

  const syncCodeHighlightScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    if (!codeHighlightRef.current) return;
    codeHighlightRef.current.scrollTop = event.currentTarget.scrollTop;
    codeHighlightRef.current.scrollLeft = event.currentTarget.scrollLeft;
  };

  const header = (
    <header className="card-head feature-create-head">
      <span className="card-head-fc">
        <Code2 />
      </span>
      <div>
        <h3>{showGuide ? "Custom Feature Guide" : "Create Custom Feature"}</h3>
        <p className="form-subtitle">{showGuide ? "Feature function contract" : "Trusted local Python"}</p>
      </div>
      <button
        type="button"
        className="btn feature-guide-toggle"
        onClick={() => setShowGuide((current) => !current)}
        title={showGuide ? "Back to custom feature form" : "Custom feature guide"}
        aria-label={showGuide ? "Back to custom feature form" : "Show custom feature guide"}
      >
        {showGuide ? <ArrowLeft /> : <CircleHelp />}
        {showGuide ? "Back" : "Guide"}
      </button>
    </header>
  );

  if (showGuide) {
    return (
      <section className="card feature-create-card feature-guide-card">
        {header}
        <div className="card-body feature-guide-body">
          <section className="feature-guide-section">
            <h4>Function Contract</h4>
            <ul>
              <li>Define one callable named <span className="mono">compute_feature(graph)</span>.</li>
              <li>Use <span className="mono">graph.nodes</span>, <span className="mono">graph.sampled_nodes</span>, <span className="mono">graph.graph_id</span>, and <span className="mono">graph.G</span> to read the prepared graph.</li>
              <li>Return a <span className="mono">pandas.DataFrame</span> with columns in this order: <span className="mono">node_id</span>, <span className="mono">graph_id</span>, then one or more numeric feature columns.</li>
              <li>Return exactly one row for every node in the validation graph. Node IDs and graph IDs must match the graph being evaluated.</li>
            </ul>
          </section>

          <section className="feature-guide-section">
            <h4>Validate and Save</h4>
            <ul>
              <li>Validate runs the code against the first prepared graph in the active completed Dataset.</li>
              <li>Save repeats backend validation before creating the planned Feature artifact.</li>
              <li>The code is trusted local Python, not sandboxed. Missing Python packages are reported clearly, but Workbench does not install packages.</li>
              <li>Workbench uses fixed execution defaults for custom features: <span className="mono">n_jobs=1</span> and <span className="mono">parallel_backend="threading"</span>. The Normalize checkbox controls <span className="mono">normalize_features</span>.</li>
            </ul>
          </section>

          <section className="feature-guide-section">
            <h4>Working Example</h4>
            <pre className="settings-code feature-guide-code">
              <code>{renderPythonTokens(CUSTOM_FEATURE_TEMPLATE.trim())}</code>
            </pre>
          </section>
        </div>
      </section>
    );
  }

  return (
    <form
      className="card feature-create-card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSave) return;
        createCustomFeature.mutate({
          source_dataset_id: dataset!.id,
          name: name.trim(),
          description,
          code,
          params: customFeatureParams
        });
      }}
    >
      {header}
      <div className="card-body">
        {createCustomFeature.error ? <p className="error-text">{createCustomFeature.error.message}</p> : null}
        {saveMessage ? <p className="muted form-note">{saveMessage}</p> : null}
        {dataset ? <p className="muted form-note">Dataset: {dataset.name}</p> : null}
        <div className="field-grid">
          <label className="field">
            <span>Name</span>
            <input type="text" value={name} onChange={(event) => setName(event.target.value)} maxLength={120} />
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={normalizeFeatures}
              onChange={(event) => setNormalizeFeatures(event.target.checked)}
            />
            <span>Normalize Features</span>
          </label>
          <label className="field field-wide">
            <span>Description</span>
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={2} />
          </label>
          <label className="field field-wide">
            <span>Python Code</span>
            <div className="code-editor-shell">
              <pre ref={codeHighlightRef} className="code-highlight" aria-hidden="true">
                <code>
                  {renderPythonTokens(code)}
                  {code.endsWith("\n") ? " " : null}
                </code>
              </pre>
              <textarea
                className="code-editor"
                value={code}
                onChange={(event) => setCode(event.target.value)}
                onScroll={syncCodeHighlightScroll}
                rows={18}
                spellCheck={false}
              />
            </div>
          </label>
        </div>
      </div>
      <footer className="card-foot">
        {validateCustomFeature.error ? (
          <p className="validation-feedback is-error">{validateCustomFeature.error.message}</p>
        ) : validateCustomFeature.data ? (
          <p className="validation-feedback is-success">
            Valid feature output: {validateCustomFeature.data.columns.join(", ")}
          </p>
        ) : null}
        <button
          type="button"
          className="btn"
          disabled={!canValidate}
          onClick={() => {
            if (!canValidate) return;
            validateCustomFeature.mutate({
              source_dataset_id: dataset!.id,
              code,
              params: customFeatureParams
            });
          }}
        >
          <CheckCircle2 />
          {validateCustomFeature.isPending ? "Validating" : "Validate"}
        </button>
        <button type="submit" className="btn btn-primary" disabled={!canSave}>
          <Save />
          {createCustomFeature.isPending ? "Saving" : "Save"}
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
  onPreviewFeature,
  onDeleteArtifact
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
                  const methodName = featureMethodLabel(feature, catalogById);
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
                        <button
                          type="button"
                          className="icon-btn icon-btn-danger"
                          aria-label={`Delete ${feature.name}`}
                          title={`Delete ${feature.name}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            onDeleteArtifact("feature", feature.id);
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

function FeaturePreviewTable({ preview }: { preview: TabularPreview }) {
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

function FeatureDataTab({ activeProjectId, feature }: { activeProjectId: string; feature: FeatureManifest }) {
  const [offset, setOffset] = useState(0);
  const pageSize = 50;

  useEffect(() => {
    setOffset(0);
  }, [feature.id]);

  const preview = useQuery({
    queryKey: ["projects", activeProjectId, "features", feature.id, "preview", pageSize, offset],
    queryFn: () => api.featurePreview(activeProjectId, feature.id, pageSize, offset),
    enabled: Boolean(activeProjectId && feature.id && feature.status === "completed")
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
        <FeaturePreviewTable preview={preview.data} />
      )}
    </div>
  );
}

type FeaturePcaChartDatum = FeaturePcaPoint & {
  value: [number, number];
  itemStyle: { color: string };
};
type FeaturePcaChartElement = HTMLDivElement & {
  __featurePcaChart?: ReturnType<typeof echarts.init>;
};

function FeaturePcaChart({
  analysis,
  selectedGraphId,
  onSelectGraph
}: {
  analysis: FeatureAnalysis;
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
    (containerRef.current as FeaturePcaChartElement).__featurePcaChart = chart;
    const handleClick = (params: { data?: unknown }) => {
      const data = params.data as FeaturePcaChartDatum | undefined;
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
      if (containerRef.current) delete (containerRef.current as FeaturePcaChartElement).__featurePcaChart;
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const palette = ["#176ea9", "#d86c1f", "#2d8754", "#8d5db8", "#a4513d", "#4f758b", "#6b7f2a", "#9a5b91"];
    const colorValues = Array.from(new Set(analysis.pca.points.map((point) => point.color_value)));
    const colorByValue = new Map(colorValues.map((value, index) => [value, palette[index % palette.length]]));
    const data: FeaturePcaChartDatum[] = analysis.pca.points.map((point) => ({
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
          const dataPoint = (item as { data?: FeaturePcaChartDatum }).data;
          if (!dataPoint) return "";
          return [
            `Graph ${dataPoint.graph_id}`,
            `${formatCount(dataPoint.node_count)} nodes`,
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
      className="feature-pca-chart"
      role="img"
      aria-label={`${analysis.feature_name} ${analysis.pca.projection_method === "raw" ? "2D feature plot" : "PCA"}`}
      tabIndex={0}
    />
  );
}

function FeatureStatisticsTab({ analysis }: { analysis: FeatureAnalysis }) {
  return (
    <div className="dataset-tab-panel">
      <div className="stat-grid">
        <div className="stat-tile">
          <span>Rows</span>
          <strong>{formatCount(analysis.output_stats.row_count)}</strong>
          <small>{formatAverage(analysis.node_coverage.covered, analysis.node_coverage.total)} node coverage</small>
        </div>
        <div className="stat-tile">
          <span>Feature Columns</span>
          <strong>{formatCount(analysis.feature_columns.length)}</strong>
          <small>{formatCount(analysis.numeric_feature_columns.length)} numeric</small>
        </div>
        <div className="stat-tile">
          <span>Source Dataset</span>
          <strong>{analysis.source_dataset.name}</strong>
          <small>{analysis.source_dataset.status}</small>
        </div>
        <div className="stat-tile">
          <span>Method</span>
          <strong>{analysis.method.name}</strong>
          <small>{analysis.method.id}</small>
        </div>
        <div className="stat-tile">
          <span>Vector Length</span>
          <strong>{formatCount(Number(analysis.feature_columns.length))}</strong>
          <small>{analysis.feature_columns.join(", ")}</small>
        </div>
        <div className="stat-tile">
          <span>Graph Coverage</span>
          <strong>
            {formatCount(analysis.graph_coverage.covered)} / {formatCount(analysis.graph_coverage.total)}
          </strong>
          <small>{formatAverage(analysis.graph_coverage.covered, analysis.graph_coverage.total)}</small>
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

function FeaturePcaTab({
  activeProjectId,
  feature,
  analysis,
  selectedGraphId,
  onSelectGraph
}: {
  activeProjectId: string;
  feature: FeatureManifest;
  analysis: FeatureAnalysis;
  selectedGraphId: string;
  onSelectGraph: (graphId: string, visible: boolean | null) => void;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const trimmedSearch = searchQuery.trim();
  const graphSearch = useQuery({
    queryKey: ["projects", activeProjectId, "features", feature.id, "analysis", "search", trimmedSearch],
    queryFn: () => api.featureGraphSearch(activeProjectId, feature.id, trimmedSearch, 25),
    enabled: Boolean(activeProjectId && feature.id && trimmedSearch)
  });

  const selectSearchResult = (result: FeatureGraphSearchResult) => {
    onSelectGraph(result.graph_id, result.in_pca_sample);
  };

  const selectedResultIndex =
    graphSearch.data?.results.findIndex((result) => result.graph_id === selectedGraphId) ?? -1;
  const searchResultCount = graphSearch.data?.results.length || 0;

  const selectResultByIndex = (index: number) => {
    const result = graphSearch.data?.results[index];
    if (result) selectSearchResult(result);
  };

  if (!analysis.pca.available) {
    return (
      <div className="dataset-tab-panel">
        <div className="artifact-table-empty">
          <EmptyState compact>{analysis.pca.reason || "PCA is unavailable for this feature."}</EmptyState>
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
        <div className="feature-pca-nav-group" aria-label="Feature search navigation">
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
              aria-label="Search feature graph IDs and labels"
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
        <div className="graph-search-results" role="listbox" aria-label="Feature graph search results">
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
                      label {formatValue(result.graph_label)} · {formatCount(result.node_count)} nodes ·{" "}
                      {result.in_pca_sample ? "plotted" : "not plotted"}
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
      <FeaturePcaChart
        analysis={analysis}
        selectedGraphId={selectedGraphId}
        onSelectGraph={onSelectGraph}
      />
    </div>
  );
}

export function FeatureExploreView({
  activeProjectId,
  features,
  datasets,
  catalog,
  loading,
  selectedFeatureId,
  exploreFeatureId,
  selectedGraphId,
  onExploreFeature,
  onClearExploreFeature,
  onSelectGraph,
  onSelectedGraphVisibilityChange
}: FeatureExploreViewProps) {
  const [tab, setTab] = useState<"statistics" | "pca" | "data">("statistics");
  const datasetsById = useMemo(() => new Map(datasets.map((dataset) => [dataset.id, dataset])), [datasets]);
  const catalogById = useMemo(() => new Map(catalog.map((entry) => [entry.id, entry])), [catalog]);
  const feature = useMemo(
    () => features.find((item) => item.id === exploreFeatureId) || features.find((item) => item.id === selectedFeatureId),
    [exploreFeatureId, features, selectedFeatureId]
  );

  useEffect(() => {
    setTab("statistics");
    onSelectGraph("", null);
    onSelectedGraphVisibilityChange(null);
  }, [feature?.id, onSelectGraph, onSelectedGraphVisibilityChange]);

  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "features", feature?.id, "analysis"],
    queryFn: () => api.featureAnalysis(activeProjectId, feature!.id),
    enabled: Boolean(activeProjectId && feature?.id && feature.status === "completed")
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

  if (!feature) {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              Feature Explore
            </span>
            <span className="muted">{loading ? "Loading" : `${features.length} features`}</span>
          </header>
          {loading ? (
            <div className="artifact-table-empty">
              <EmptyState compact>Loading features.</EmptyState>
            </div>
          ) : features.length === 0 ? (
            <div className="artifact-table-empty">
              <EmptyState compact>No features.</EmptyState>
            </div>
          ) : (
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Dataset</th>
                    <th>Method</th>
                    <th>Status</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {features.map((item) => {
                    const datasetName = datasetsById.get(datasetInputId(item))?.name || "Unknown dataset";
                    const methodName = catalogById.get(item.source_feature_id)?.name || item.source_feature_id;
                    return (
                      <tr key={item.id} onClick={() => onExploreFeature(item.id)}>
                        <td>
                          <span className="table-name-with-icon">
                            <Sigma />
                            <strong>{item.name}</strong>
                          </span>
                        </td>
                        <td>{datasetName}</td>
                        <td>{methodName}</td>
                        <td>
                          <span className={`status-pill ${item.status === "completed" ? "is-ready" : "is-idle"}`}>{item.status}</span>
                        </td>
                        <td className="actions-cell actions-cell-wide">
                          <button
                            type="button"
                            className="btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              onExploreFeature(item.id);
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

  if (feature.status !== "completed") {
    return (
      <div className="workflow">
        <section className="artifact-table">
          <header className="artifact-table-head">
            <span className="artifact-table-title">
              <FcIcon name="explore" size={16} />
              {feature.name} Explore
            </span>
            <div className="artifact-table-head-actions">
              <span className="muted">{feature.status}</span>
              <button type="button" className="btn" onClick={onClearExploreFeature}>
                <ArrowLeft />
                Choose Feature
              </button>
            </div>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Run feature computation before exploring this feature.</EmptyState>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="workflow workflow-fill">
      <section className="artifact-table dataset-explore feature-explore">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="explore" size={16} />
            {feature.name} Explore
          </span>
          <div className="artifact-table-head-actions">
            <span className="muted">{feature.output_stats ? `${formatCount(feature.output_stats.row_count)} rows` : feature.status}</span>
            <button type="button" className="btn" onClick={onClearExploreFeature}>
              <ArrowLeft />
              Choose Feature
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
                title={pcaDisabled ? analysis.data?.pca.reason || "PCA is unavailable for this feature." : undefined}
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
        {tab === "statistics" && analysis.data ? <FeatureStatisticsTab analysis={analysis.data} /> : null}
        {tab === "pca" && analysis.data ? (
          <FeaturePcaTab
            activeProjectId={activeProjectId}
            feature={feature}
            analysis={analysis.data}
            selectedGraphId={selectedGraphId}
            onSelectGraph={onSelectGraph}
          />
        ) : null}
        {tab === "data" ? <FeatureDataTab activeProjectId={activeProjectId} feature={feature} /> : null}
      </section>
    </div>
  );
}
