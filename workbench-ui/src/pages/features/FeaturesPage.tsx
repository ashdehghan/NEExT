import { useEffect, useMemo, useRef, useState, type ReactNode, type UIEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  CheckCircle2,
  CircleHelp,
  Code2,
  Eye,
  Play,
  Plus,
  RotateCcw,
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
  type FeatureManifest,
  type TabularPreview
} from "../../api";
import { AnalysisCommandCenter } from "../../components/viz/AnalysisCommandCenter";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface FeatureLibraryViewProps {
  activeProjectId: string;
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
  draft?: Record<string, unknown>;
  onCreated: (featureId: string) => void;
}

interface CreateFeatureViewProps {
  activeProjectId: string;
  dataset?: DatasetManifest;
  draft?: Record<string, unknown>;
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

export function FeatureLibraryView({
  activeProjectId,
  catalog,
  loading,
  selectedCatalogId,
  selectedDataset,
  onSelectCatalog,
  onConfigure
}: FeatureLibraryViewProps) {
  const queryClient = useQueryClient();
  const [checkedCatalogIds, setCheckedCatalogIds] = useState<string[]>([]);
  const headerCheckboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setCheckedCatalogIds((current) => current.filter((catalogId) => catalog.some((entry) => entry.id === catalogId)));
  }, [catalog]);

  const allChecked = catalog.length > 0 && checkedCatalogIds.length === catalog.length;
  const someChecked = checkedCatalogIds.length > 0 && !allChecked;

  useEffect(() => {
    if (headerCheckboxRef.current) headerCheckboxRef.current.indeterminate = someChecked;
  }, [someChecked]);

  const addSelected = useMutation({
    mutationFn: async (catalogIds: string[]) => {
      const created: FeatureManifest[] = [];
      const failures: string[] = [];
      for (const catalogId of catalogIds) {
        try {
          const result = await api.createFeature(activeProjectId, {
            source_dataset_id: selectedDataset!.id,
            source_feature_id: catalogId,
            params: { feature_vector_length: 3, normalize_features: true, n_jobs: 1, parallel_backend: "loky" }
          });
          created.push(result);
        } catch (error) {
          const name = catalog.find((entry) => entry.id === catalogId)?.name || catalogId;
          failures.push(`${name}: ${(error as Error).message}`);
        }
      }
      if (failures.length) throw new Error(`Failed to add ${failures.length} feature(s): ${failures.join("; ")}`);
      return created;
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["projects", activeProjectId, "features"] });
      setCheckedCatalogIds([]);
    }
  });

  const canAddSelected = checkedCatalogIds.length > 0 && Boolean(selectedDataset) && !addSelected.isPending;

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
        {addSelected.error ? <p className="table-error">{addSelected.error.message}</p> : null}
        {loading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading feature library.</EmptyState>
          </div>
        ) : catalog.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No feature catalog entries.</EmptyState>
          </div>
        ) : (
          <>
            <div className="table-toolbar">
              <button
                type="button"
                className="btn"
                disabled={!canAddSelected}
                onClick={() => addSelected.mutate(checkedCatalogIds)}
              >
                <Plus />
                {addSelected.isPending ? "Adding…" : "Add Selected to Project"}
              </button>
            </div>
            <div className="artifact-table-scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>
                      <input
                        ref={headerCheckboxRef}
                        type="checkbox"
                        aria-label="Select all features"
                        checked={allChecked}
                        onChange={(event) =>
                          setCheckedCatalogIds(event.target.checked ? catalog.map((entry) => entry.id) : [])
                        }
                      />
                    </th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Output</th>
                    <th className="actions-col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {catalog.map((entry) => {
                    const isChecked = checkedCatalogIds.includes(entry.id);
                    return (
                      <tr
                        key={entry.id}
                        className={entry.id === selectedCatalogId ? "is-selected" : ""}
                        onClick={() => onSelectCatalog(entry.id)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            aria-label={`Select ${entry.name}`}
                            checked={isChecked}
                            onChange={(event) => {
                              event.stopPropagation();
                              setCheckedCatalogIds((current) =>
                                event.target.checked ? [...current, entry.id] : current.filter((catalogId) => catalogId !== entry.id)
                              );
                            }}
                            onClick={(event) => event.stopPropagation()}
                          />
                        </td>
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
          </>
        )}
      </section>
    </div>
  );
}

export function ConfigureFeatureView({ activeProjectId, feature, dataset, draft, onCreated }: ConfigureFeatureViewProps) {
  const queryClient = useQueryClient();
  const [featureVectorLength, setFeatureVectorLength] = useState(3);
  const [normalizeFeatures, setNormalizeFeatures] = useState(true);
  const [nJobs, setNJobs] = useState(1);
  const [parallelBackend, setParallelBackend] = useState<"loky" | "threading">("loky");

  useEffect(() => {
    if (!draft) return;
    const nextFeatureVectorLength = draftNumber(draft, "feature_vector_length");
    if (nextFeatureVectorLength !== undefined) setFeatureVectorLength(nextFeatureVectorLength);
    const nextNormalize = draftBoolean(draft, "normalize_features");
    if (nextNormalize !== undefined) setNormalizeFeatures(nextNormalize);
    const nextNJobs = draftNumber(draft, "n_jobs");
    if (nextNJobs !== undefined) setNJobs(nextNJobs);
    const nextParallelBackend = draftString(draft, "parallel_backend");
    if (nextParallelBackend === "loky" || nextParallelBackend === "threading") setParallelBackend(nextParallelBackend);
  }, [activeProjectId, feature?.id, dataset?.id, draft]);

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
          <h3>Add {feature.name} to Project</h3>
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
          <Plus />
          {createFeature.isPending ? "Creating" : "Create Feature"}
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

export function CreateFeatureView({ activeProjectId, dataset, draft, onCreated }: CreateFeatureViewProps) {
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

  useEffect(() => {
    if (!draft) return;
    const nextName = draftString(draft, "name");
    if (nextName !== undefined) setName(nextName);
    const nextDescription = draftString(draft, "description");
    if (nextDescription !== undefined) setDescription(nextDescription);
    const nextCode = draftString(draft, "code");
    if (nextCode !== undefined) setCode(nextCode);
    const nextNormalize = draftBoolean(draft, "normalize_features");
    if (nextNormalize !== undefined) setNormalizeFeatures(nextNormalize);
  }, [activeProjectId, dataset?.id, draft]);

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
  const nodeAttributeQuery = useQuery({
    queryKey: ["projects", activeProjectId, "datasets", dataset?.id, "analysis", "node-attributes"],
    queryFn: () => api.datasetAnalysis(activeProjectId, dataset!.id, { max_nodes: 1, max_edges: 1 }),
    enabled: Boolean(activeProjectId && dataset?.id && datasetComplete)
  });
  const nodeAttributeColumns = nodeAttributeQuery.data?.node_feature_columns ?? [];
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
              <li>Read imported node attributes via <span className="mono">graph.node_attributes[node_id]</span> (a dict per node) to build features from uploaded <span className="mono">node_features</span> columns.</li>
              <li>Return a <span className="mono">pandas.DataFrame</span> with columns in this order: <span className="mono">node_id</span>, <span className="mono">graph_id</span>, then one or more numeric feature columns.</li>
              <li>Return exactly one row for every node in the validation graph. Node IDs and graph IDs must match the graph being evaluated.</li>
            </ul>
          </section>

          <section className="feature-guide-section">
            <h4>Validate and Create</h4>
            <ul>
              <li>Validate runs the code against the first prepared graph in the active completed Dataset.</li>
              <li>Create repeats backend validation before creating the Draft Feature artifact.</li>
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
        {datasetComplete && nodeAttributeColumns.length > 0 ? (
          <p className="muted form-note">
            Available node attributes (read via <span className="mono">graph.node_attributes[node_id]</span>):{" "}
            {nodeAttributeColumns.map((column, index) => (
              <span key={column} className="mono">
                {index > 0 ? ", " : ""}
                {column}
              </span>
            ))}
          </p>
        ) : null}
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
          <Plus />
          {createCustomFeature.isPending ? "Creating" : "Create Feature"}
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

  const runnableFeatureIds = features
    .filter((feature) => feature.status === "planned" || feature.status === "failed")
    .map((feature) => feature.id);
  const allRunnableChecked = runnableFeatureIds.length > 0 && runnableFeatureIds.every((id) => checkedFeatureIds.includes(id));
  const someRunnableChecked = runnableCheckedFeatureIds.length > 0 && !allRunnableChecked;
  const headerCheckboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (headerCheckboxRef.current) headerCheckboxRef.current.indeterminate = someRunnableChecked;
  }, [someRunnableChecked]);

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
              Compute Selected
            </button>
          </div>
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>
                    <input
                      ref={headerCheckboxRef}
                      type="checkbox"
                      aria-label="Select all runnable features"
                      checked={allRunnableChecked}
                      disabled={runnableFeatureIds.length === 0}
                      onChange={(event) => setCheckedFeatureIds(event.target.checked ? runnableFeatureIds : [])}
                    />
                  </th>
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
                        <span className={`status-pill ${artifactStatusClass(feature.status)}`}>{artifactStatusLabel(feature.status)}</span>
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
                            {feature.status === "failed" ? "Retry Compute" : isRunning ? "Computing" : "Compute"}
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
  const [tab, setTab] = useState<"statistics" | "analysis" | "data">("statistics");
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

  // Base analysis (default PCA, no clustering) feeds Statistics/Data and tells us whether labels
  // exist; the Analysis tab's cards each fetch their own projection/clustering.
  const analysis = useQuery({
    queryKey: ["projects", activeProjectId, "features", feature?.id, "analysis", "base"],
    queryFn: () => api.featureAnalysis(activeProjectId, feature!.id),
    enabled: Boolean(activeProjectId && feature?.id && feature.status === "completed")
  });

  const hasLabels = Object.keys(analysis.data?.graph_label_distribution ?? {}).length > 0;

  useEffect(() => {
    if (tab === "analysis" && analysis.data && !analysis.data.pca.available) {
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
                          <span className={`status-pill ${artifactStatusClass(item.status)}`}>{artifactStatusLabel(item.status)}</span>
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
                            {item.status === "completed" ? "Explore" : "Compute First"}
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
              <button type="button" className="btn" onClick={onClearExploreFeature}>
                <ArrowLeft />
                Choose Feature
              </button>
            </span>
            <span className="explore-title">{feature.name}</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Compute this feature before exploring it.</EmptyState>
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
            <button type="button" className="btn" onClick={onClearExploreFeature}>
              <ArrowLeft />
              Choose Feature
            </button>
          </span>
          <span className="explore-title">{feature.name}</span>
        </header>
        <div className="tab-strip">
          {(["statistics", "analysis", "data"] as const).map((item) => {
            const analysisDisabled = item === "analysis" && Boolean(analysis.data && !analysis.data.pca.available);
            return (
              <button
                key={item}
                type="button"
                className={`tab-btn ${tab === item ? "is-active" : ""}`}
                onClick={() => setTab(item)}
                disabled={analysisDisabled}
                title={analysisDisabled ? analysis.data?.pca.reason || "Analysis is unavailable for this feature." : undefined}
              >
                {item === "statistics" ? "Statistics" : item === "analysis" ? "Analysis" : "Data"}
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
        {tab === "analysis" && analysis.data ? (
          <AnalysisCommandCenter
            analyze={(params) => api.featureAnalysis(activeProjectId, feature.id, params)}
            queryKeyBase={["projects", activeProjectId, "features", feature.id, "analysis"]}
            exportName={feature.name}
            hasLabels={hasLabels}
          />
        ) : null}
        {tab === "data" ? <FeatureDataTab activeProjectId={activeProjectId} feature={feature} /> : null}
      </section>
    </div>
  );
}
