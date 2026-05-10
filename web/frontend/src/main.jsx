import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const mainTabs = ["HOME", "PLOTS", "APPS", "FEATURES", "EMBEDDINGS", "MODELS"];

const documents = [
  { id: "data", label: "workflow.mlx", tab: "HOME" },
  { id: "explore", label: "Figure 1: Graph", tab: "PLOTS" },
  { id: "features", label: "features_all", tab: "FEATURES" },
  { id: "embeddings", label: "embeddings_3d", tab: "EMBEDDINGS" },
  { id: "models", label: "model_run_02", tab: "MODELS" },
  { id: "exports", label: "export.py", tab: "APPS" },
];

const docByTab = {
  HOME: "data",
  PLOTS: "explore",
  APPS: "exports",
  FEATURES: "features",
  EMBEDDINGS: "embeddings",
  MODELS: "models",
};

const presetOptions = ["er_vs_ba", "multiclass_topology", "community_gradient", "scalability", "variable_size", "anomaly_detection"];
const embeddingAlgorithms = ["approx_wasserstein", "wasserstein", "sinkhornvectorizer"];

function App() {
  const [activeTab, setActiveTab] = useState("HOME");
  const [activeDocument, setActiveDocument] = useState("data");
  const [project, setProject] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedArtifactId, setSelectedArtifactId] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [log, setLog] = useState([]);
  const [collectionSummary, setCollectionSummary] = useState(null);
  const [graphElements, setGraphElements] = useState(null);
  const [featureTable, setFeatureTable] = useState(null);
  const [embeddingTable, setEmbeddingTable] = useState(null);

  const [presetForm, setPresetForm] = useState({ name: "ER vs BA demo", preset: "er_vs_ba", seed: 42 });
  const [csvForm, setCsvForm] = useState({
    name: "Imported graph dataset",
    edges_path: "",
    node_graph_mapping_path: "",
    graph_label_path: "",
    node_features_path: "",
  });
  const [featureForm, setFeatureForm] = useState({ name: "All structural features", dataset_id: "", feature_list: "all", feature_vector_length: 3, n_jobs: 1 });
  const [embeddingForm, setEmbeddingForm] = useState({
    name: "Approx Wasserstein embeddings",
    dataset_id: "",
    features_id: "",
    embedding_algorithm: "approx_wasserstein",
    embedding_dimension: 3,
  });
  const [modelForm, setModelForm] = useState({ name: "Classifier evaluation", dataset_id: "", embeddings_id: "", model_type: "classifier", sample_size: 5 });
  const [exportForm, setExportForm] = useState({ name: "Reproduce selected artifact", artifact_id: "" });
  const [tableSelections, setTableSelections] = useState({ feature: "", embedding: "" });

  const artifactGroups = useMemo(
    () => ({
      dataset: artifacts.filter((artifact) => artifact.type === "dataset"),
      features: artifacts.filter((artifact) => artifact.type === "features"),
      embeddings: artifacts.filter((artifact) => artifact.type === "embeddings"),
      model: artifacts.filter((artifact) => artifact.type === "model_run"),
      export: artifacts.filter((artifact) => artifact.type === "export"),
    }),
    [artifacts],
  );

  const currentDatasetId = selectedDatasetId || artifactGroups.dataset[0]?.id || "";
  const statusText = project?.status === "ok" ? "online" : "offline";
  const runningJobs = jobs.filter((job) => job.status === "queued" || job.status === "running");

  function pushLog(message) {
    setLog((items) => [`[${new Date().toLocaleTimeString()}] ${message}`, ...items].slice(0, 100));
  }

  async function refreshProject() {
    try {
      const [health, artifactPayload, jobPayload] = await Promise.all([api("/api/health"), api("/api/artifacts"), api("/api/jobs")]);
      const nextArtifacts = artifactPayload.artifacts || [];
      setProject(health);
      setArtifacts(nextArtifacts);
      setJobs(jobPayload.jobs || []);
      setSelectedDatasetId((current) => current || nextArtifacts.find((artifact) => artifact.type === "dataset")?.id || "");
      return nextArtifacts;
    } catch (error) {
      setProject((current) => ({ ...(current || {}), status: "offline" }));
      pushLog(`Refresh failed: ${error.message}`);
      throw error;
    }
  }

  useEffect(() => {
    refreshProject().catch(() => {});
  }, []);

  function activateTab(tab) {
    setActiveTab(tab);
    setActiveDocument(docByTab[tab]);
  }

  function activateDocument(documentId) {
    setActiveDocument(documentId);
    setActiveTab(documents.find((item) => item.id === documentId)?.tab || "HOME");
  }

  function setPresetValue(key, value) {
    setPresetForm((form) => ({ ...form, [key]: value }));
  }

  function setCsvValue(key, value) {
    setCsvForm((form) => ({ ...form, [key]: value }));
  }

  function setFeatureValue(key, value) {
    setFeatureForm((form) => ({ ...form, [key]: value }));
  }

  function setEmbeddingValue(key, value) {
    setEmbeddingForm((form) => ({ ...form, [key]: value }));
  }

  function setModelValue(key, value) {
    setModelForm((form) => ({ ...form, [key]: value }));
  }

  function setExportValue(key, value) {
    setExportForm((form) => ({ ...form, [key]: value }));
  }

  function chooseArtifact(artifact) {
    setSelectedArtifactId(artifact.id);
    if (artifact.type === "dataset") {
      setSelectedDatasetId(artifact.id);
      setFeatureValue("dataset_id", artifact.id);
      setEmbeddingValue("dataset_id", artifact.id);
      setModelValue("dataset_id", artifact.id);
    }
  }

  async function generateDataset(event) {
    event?.preventDefault();
    activateDocument("data");
    try {
      const result = await api("/api/datasets/generate", {
        method: "POST",
        body: JSON.stringify({
          name: presetForm.name,
          preset: presetForm.preset,
          seed: Number(presetForm.seed),
          params: {},
        }),
      });
      setSelectedArtifactId(result.artifact.id);
      setSelectedDatasetId(result.artifact.id);
      pushLog(`Generated dataset ${result.artifact.id}`);
      await refreshProject();
    } catch (error) {
      pushLog(`Dataset generation failed: ${error.message}`);
    }
  }

  async function importDataset(event) {
    event?.preventDefault();
    activateDocument("data");
    if (!csvForm.edges_path || !csvForm.node_graph_mapping_path) {
      pushLog("CSV import needs edges and node graph mapping paths.");
      return;
    }
    const payload = cleanPayload(csvForm);
    try {
      const result = await api("/api/datasets/import", { method: "POST", body: JSON.stringify(payload) });
      setSelectedArtifactId(result.artifact.id);
      setSelectedDatasetId(result.artifact.id);
      pushLog(`Imported dataset ${result.artifact.id}`);
      await refreshProject();
    } catch (error) {
      pushLog(`Dataset import failed: ${error.message}`);
    }
  }

  async function loadExplore() {
    activateDocument("explore");
    const datasetId = featureForm.dataset_id || embeddingForm.dataset_id || modelForm.dataset_id || currentDatasetId;
    if (!datasetId) {
      pushLog("Select or create a dataset artifact first.");
      return;
    }
    try {
      const summary = await api(`/api/graphs/${datasetId}`);
      setCollectionSummary(summary);
      setSelectedDatasetId(datasetId);
      const firstGraph = summary.graphs?.[0];
      if (firstGraph) {
        await drawGraph(datasetId, firstGraph.graph_id);
      }
      pushLog(`Loaded graph collection ${datasetId}`);
    } catch (error) {
      pushLog(`Explore load failed: ${error.message}`);
    }
  }

  async function drawGraph(datasetId, graphId) {
    const payload = await api(`/api/graphs/${datasetId}/${graphId}/elements?max_nodes=120`);
    setGraphElements({ graphId, ...payload.elements });
  }

  async function submitFeatureJob(event) {
    event?.preventDefault();
    activateDocument("features");
    const datasetId = featureForm.dataset_id || currentDatasetId;
    if (!datasetId) {
      pushLog("Feature jobs need a dataset artifact.");
      return;
    }
    await submitJob("/api/jobs/features", {
      name: featureForm.name,
      dataset_id: datasetId,
      feature_list: String(featureForm.feature_list || "all")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      feature_vector_length: Number(featureForm.feature_vector_length),
      normalize_features: true,
      n_jobs: Number(featureForm.n_jobs),
      parallel_backend: "loky",
    });
  }

  async function submitEmbeddingJob(event) {
    event?.preventDefault();
    activateDocument("embeddings");
    const datasetId = embeddingForm.dataset_id || currentDatasetId;
    const featuresId = embeddingForm.features_id || artifactGroups.features[0]?.id || "";
    if (!datasetId || !featuresId) {
      pushLog("Embedding jobs need dataset and feature artifacts.");
      return;
    }
    await submitJob("/api/jobs/embeddings", {
      name: embeddingForm.name,
      dataset_id: datasetId,
      features_id: featuresId,
      embedding_algorithm: embeddingForm.embedding_algorithm,
      embedding_dimension: Number(embeddingForm.embedding_dimension),
    });
  }

  async function submitModelJob(event) {
    event?.preventDefault();
    activateDocument("models");
    const datasetId = modelForm.dataset_id || currentDatasetId;
    const embeddingsId = modelForm.embeddings_id || artifactGroups.embeddings[0]?.id || "";
    if (!datasetId || !embeddingsId) {
      pushLog("Model jobs need dataset and embedding artifacts.");
      return;
    }
    await submitJob("/api/jobs/models", {
      name: modelForm.name,
      dataset_id: datasetId,
      embeddings_id: embeddingsId,
      model_type: modelForm.model_type,
      sample_size: Number(modelForm.sample_size),
    });
  }

  async function submitJob(path, payload) {
    try {
      const result = await api(path, { method: "POST", body: JSON.stringify(payload) });
      const job = result.job;
      pushLog(`Submitted ${job.type} job ${job.id}`);
      await refreshProject();
      pollJob(job.id);
    } catch (error) {
      pushLog(`Job submission failed: ${error.message}`);
    }
  }

  async function pollJob(jobId) {
    try {
      const result = await api(`/api/jobs/${jobId}`);
      const job = result.job;
      pushLog(`${job.name}: ${job.status} (${Math.round((job.progress || 0) * 100)}%)`);
      await refreshProject();
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(() => pollJob(jobId), 1200);
      }
    } catch (error) {
      pushLog(`Job poll failed: ${error.message}`);
    }
  }

  async function loadFeatureTable() {
    activateDocument("features");
    const artifactId = tableSelections.feature || artifactGroups.features[0]?.id || "";
    if (!artifactId) {
      pushLog("Select a feature artifact first.");
      return;
    }
    try {
      const payload = await api(`/api/artifacts/${artifactId}/table?limit=120`);
      setFeatureTable(payload);
      setTableSelections((current) => ({ ...current, feature: artifactId }));
      pushLog(`Loaded table for ${artifactId}`);
    } catch (error) {
      pushLog(`Feature table load failed: ${error.message}`);
    }
  }

  async function loadEmbeddingPlot() {
    activateDocument("embeddings");
    const artifactId = tableSelections.embedding || artifactGroups.embeddings[0]?.id || "";
    if (!artifactId) {
      pushLog("Select an embedding artifact first.");
      return;
    }
    try {
      const payload = await api(`/api/artifacts/${artifactId}/table?limit=1000`);
      setEmbeddingTable(payload);
      setTableSelections((current) => ({ ...current, embedding: artifactId }));
      pushLog(`Loaded embedding plot for ${artifactId}`);
    } catch (error) {
      pushLog(`Embedding plot load failed: ${error.message}`);
    }
  }

  async function exportPython(event) {
    event?.preventDefault();
    activateDocument("exports");
    const artifactId = exportForm.artifact_id || selectedArtifactId || artifacts[0]?.id || "";
    if (!artifactId) {
      pushLog("Select an artifact to export.");
      return;
    }
    try {
      const result = await api("/api/exports/python", {
        method: "POST",
        body: JSON.stringify({ name: exportForm.name, artifact_id: artifactId }),
      });
      setSelectedArtifactId(result.artifact.id);
      pushLog(`Created export ${result.artifact.id}`);
      await refreshProject();
    } catch (error) {
      pushLog(`Export failed: ${error.message}`);
    }
  }

  function runPrimaryAction() {
    const actions = {
      data: generateDataset,
      explore: loadExplore,
      features: submitFeatureJob,
      embeddings: submitEmbeddingJob,
      models: submitModelJob,
      exports: exportPython,
    };
    actions[activeDocument]?.();
  }

  function stopCurrentJob() {
    pushLog("Stop requested; this local workbench does not expose job cancellation yet.");
  }

  const ribbon = ribbonForTab(activeTab, {
    generateDataset,
    importDataset,
    loadExplore,
    loadFeatureTable,
    loadEmbeddingPlot,
    submitFeatureJob,
    submitEmbeddingJob,
    submitModelJob,
    exportPython,
    runPrimaryAction,
    stopCurrentJob,
    refreshProject: () => refreshProject().then(() => pushLog("Project refreshed")).catch(() => {}),
    activateDocument,
  });

  return (
    <div className="desktop">
      <header className="windowbar">
        <div className="quick-access">
          <span className="qa-button undo" aria-hidden="true" />
          <span className="qa-button redo" aria-hidden="true" />
          <span className="qa-button save" aria-hidden="true" />
          <span className="window-title">NEExT Workbench - {project?.project_dir || "Connecting..."}</span>
        </div>
        <div className="quick-access">
          <input className="search-box" value="Search Documentation" readOnly />
          <span className={`health-badge ${statusText === "online" ? "ok" : ""}`}>{statusText}</span>
          <span className="qa-button help" aria-hidden="true" />
          <button className="qa-button options" aria-label="Refresh project" onClick={() => refreshProject().catch(() => {})} />
        </div>
      </header>

      <nav className="tabstrip" aria-label="Workbench tabs">
        {mainTabs.map((tab) => (
          <button key={tab} className={tab === activeTab ? "main-tab active" : "main-tab"} onClick={() => activateTab(tab)}>
            {tab}
          </button>
        ))}
      </nav>

      <section className="toolstrip" aria-label="Ribbon commands">
        {ribbon.map((group) => (
          <div className="toolgroup" key={group.label}>
            <div className="tools">
              {group.tools.map((tool) => (
                <ToolButton key={tool.label} {...tool} />
              ))}
            </div>
            <div className="group-label">{group.label}</div>
          </div>
        ))}
      </section>

      <main className="workspace-layout">
        <aside className="dock-stack left">
          <section className="panel">
            <div className="panel-title">
              <span>Current Folder</span>
              <span className="panel-actions">v [] x</span>
            </div>
            <div className="panel-body">
              <div className="folderbar">
                <span className="path-glyph" aria-hidden="true" />
                <input className="path-input" value={project?.project_dir || ""} readOnly />
              </div>
              <ProjectTree artifacts={artifacts} selectedArtifactId={selectedArtifactId} onSelect={chooseArtifact} />
            </div>
          </section>

          <section className="panel">
            <div className="panel-title">
              <span>Workspace</span>
              <span className="panel-actions">v [] x</span>
            </div>
            <div className="panel-body workspace-table">
              <WorkspaceTable groups={artifactGroups} jobs={jobs} />
            </div>
          </section>
        </aside>

        <section className="document-area">
          <div className="document-tabs">
            {documents.map((document) => (
              <button key={document.id} className={document.id === activeDocument ? "doc-tab active" : "doc-tab"} onClick={() => activateDocument(document.id)}>
                {document.label}
              </button>
            ))}
          </div>

          <div className="doc-vertical">
            <div className="doc-split">
              <section className="editor">
                <ActiveDocument
                  activeDocument={activeDocument}
                  artifacts={artifacts}
                  artifactGroups={artifactGroups}
                  currentDatasetId={currentDatasetId}
                  presetForm={presetForm}
                  csvForm={csvForm}
                  featureForm={featureForm}
                  embeddingForm={embeddingForm}
                  modelForm={modelForm}
                  exportForm={exportForm}
                  tableSelections={tableSelections}
                  collectionSummary={collectionSummary}
                  featureTable={featureTable}
                  setPresetValue={setPresetValue}
                  setCsvValue={setCsvValue}
                  setFeatureValue={setFeatureValue}
                  setEmbeddingValue={setEmbeddingValue}
                  setModelValue={setModelValue}
                  setExportValue={setExportValue}
                  setTableSelections={setTableSelections}
                  generateDataset={generateDataset}
                  importDataset={importDataset}
                  loadExplore={loadExplore}
                  submitFeatureJob={submitFeatureJob}
                  submitEmbeddingJob={submitEmbeddingJob}
                  submitModelJob={submitModelJob}
                  loadFeatureTable={loadFeatureTable}
                  loadEmbeddingPlot={loadEmbeddingPlot}
                  exportPython={exportPython}
                />
              </section>
              <FigurePane activeDocument={activeDocument} graphElements={graphElements} collectionSummary={collectionSummary} embeddingTable={embeddingTable} jobs={jobs} />
            </div>

            <section className="command-window">
              <div className="panel-title">
                <span>Command Window</span>
                <span className="panel-actions">v [] x</span>
              </div>
              <div className="command-grid">
                <pre className="console">
                  {log.length ? log.join("\n") : ">> Ready\n>> Select or create a dataset artifact to begin"}
                </pre>
                <div className="job-strip">
                  {jobs.length === 0 ? <div className="empty">No jobs yet.</div> : jobs.slice().reverse().map((job) => <JobItem key={job.id} job={job} />)}
                </div>
              </div>
            </section>
          </div>
        </section>
      </main>

      <footer className="statusbar">
        <span>
          <span className="pill">Aero Classic</span>
          <span className="pill">{statusText}</span>
          <span className="pill">{currentDatasetId || "no dataset"}</span>
        </span>
        <span>
          {runningJobs.length ? `${runningJobs.length} running job${runningJobs.length === 1 ? "" : "s"}` : "Ready"} | {artifacts.length} artifacts
        </span>
      </footer>
    </div>
  );
}

function ActiveDocument(props) {
  const {
    activeDocument,
    artifacts,
    artifactGroups,
    currentDatasetId,
    presetForm,
    csvForm,
    featureForm,
    embeddingForm,
    modelForm,
    exportForm,
    tableSelections,
    collectionSummary,
    featureTable,
    setPresetValue,
    setCsvValue,
    setFeatureValue,
    setEmbeddingValue,
    setModelValue,
    setExportValue,
    setTableSelections,
    generateDataset,
    importDataset,
    loadExplore,
    submitFeatureJob,
    submitEmbeddingJob,
    submitModelJob,
    loadFeatureTable,
    loadEmbeddingPlot,
    exportPython,
  } = props;

  if (activeDocument === "explore") {
    return (
      <div className="document-content">
        <div className="document-header">
          <h1>Graph Collection</h1>
          <button onClick={loadExplore}>Load Selected Dataset</button>
        </div>
        <MetricGrid summary={collectionSummary} />
        <DataTable
          payload={
            collectionSummary
              ? {
                  columns: ["graph_id", "graph_label", "num_nodes", "num_edges"],
                  rows: collectionSummary.graphs || [],
                }
              : null
          }
        />
      </div>
    );
  }

  if (activeDocument === "features") {
    return (
      <div className="document-content">
        <div className="form-grid two">
          <form className="work-form" onSubmit={submitFeatureJob}>
            <h1>Feature Job</h1>
            <TextField label="Name" value={featureForm.name} onChange={(value) => setFeatureValue("name", value)} />
            <SelectField label="Dataset artifact" value={featureForm.dataset_id || currentDatasetId} onChange={(value) => setFeatureValue("dataset_id", value)} options={artifactGroups.dataset} />
            <TextField label="Feature list" value={featureForm.feature_list} onChange={(value) => setFeatureValue("feature_list", value)} />
            <TextField label="Vector length" type="number" min="1" value={featureForm.feature_vector_length} onChange={(value) => setFeatureValue("feature_vector_length", value)} />
            <TextField label="Jobs" type="number" value={featureForm.n_jobs} onChange={(value) => setFeatureValue("n_jobs", value)} />
            <button type="submit">Compute Features</button>
          </form>
          <div className="work-form">
            <h1>Feature Table</h1>
            <SelectField
              label="Feature artifact"
              value={tableSelections.feature || artifactGroups.features[0]?.id || ""}
              onChange={(value) => setTableSelections((current) => ({ ...current, feature: value }))}
              options={artifactGroups.features}
            />
            <button onClick={loadFeatureTable}>Load Table</button>
            <DataTable payload={featureTable} compact />
          </div>
        </div>
      </div>
    );
  }

  if (activeDocument === "embeddings") {
    return (
      <div className="document-content">
        <div className="form-grid two">
          <form className="work-form" onSubmit={submitEmbeddingJob}>
            <h1>Embedding Job</h1>
            <TextField label="Name" value={embeddingForm.name} onChange={(value) => setEmbeddingValue("name", value)} />
            <SelectField label="Dataset artifact" value={embeddingForm.dataset_id || currentDatasetId} onChange={(value) => setEmbeddingValue("dataset_id", value)} options={artifactGroups.dataset} />
            <SelectField label="Feature artifact" value={embeddingForm.features_id || artifactGroups.features[0]?.id || ""} onChange={(value) => setEmbeddingValue("features_id", value)} options={artifactGroups.features} />
            <SimpleSelect label="Algorithm" value={embeddingForm.embedding_algorithm} onChange={(value) => setEmbeddingValue("embedding_algorithm", value)} options={embeddingAlgorithms} />
            <TextField label="Dimension" type="number" min="2" value={embeddingForm.embedding_dimension} onChange={(value) => setEmbeddingValue("embedding_dimension", value)} />
            <button type="submit">Compute Embeddings</button>
          </form>
          <div className="work-form">
            <h1>Embedding Plot</h1>
            <SelectField
              label="Embedding artifact"
              value={tableSelections.embedding || artifactGroups.embeddings[0]?.id || ""}
              onChange={(value) => setTableSelections((current) => ({ ...current, embedding: value }))}
              options={artifactGroups.embeddings}
            />
            <button onClick={loadEmbeddingPlot}>Load Plot</button>
          </div>
        </div>
      </div>
    );
  }

  if (activeDocument === "models") {
    return (
      <div className="document-content">
        <form className="work-form narrow" onSubmit={submitModelJob}>
          <h1>Model Job</h1>
          <TextField label="Name" value={modelForm.name} onChange={(value) => setModelValue("name", value)} />
          <SelectField label="Dataset artifact" value={modelForm.dataset_id || currentDatasetId} onChange={(value) => setModelValue("dataset_id", value)} options={artifactGroups.dataset} />
          <SelectField label="Embedding artifact" value={modelForm.embeddings_id || artifactGroups.embeddings[0]?.id || ""} onChange={(value) => setModelValue("embeddings_id", value)} options={artifactGroups.embeddings} />
          <SimpleSelect label="Model type" value={modelForm.model_type} onChange={(value) => setModelValue("model_type", value)} options={["classifier", "regressor"]} />
          <TextField label="Sample size" type="number" min="1" value={modelForm.sample_size} onChange={(value) => setModelValue("sample_size", value)} />
          <button type="submit">Run Model</button>
        </form>
      </div>
    );
  }

  if (activeDocument === "exports") {
    return (
      <div className="document-content">
        <form className="work-form narrow" onSubmit={exportPython}>
          <h1>Python Export</h1>
          <TextField label="Name" value={exportForm.name} onChange={(value) => setExportValue("name", value)} />
          <SelectField label="Artifact" value={exportForm.artifact_id || artifacts[0]?.id || ""} onChange={(value) => setExportValue("artifact_id", value)} options={artifacts} showType />
          <button type="submit">Generate Script</button>
        </form>
      </div>
    );
  }

  return (
    <div className="document-content">
      <div className="form-grid two">
        <form className="work-form" onSubmit={generateDataset}>
          <h1>Synthetic Preset</h1>
          <TextField label="Name" value={presetForm.name} onChange={(value) => setPresetValue("name", value)} />
          <SimpleSelect label="Preset" value={presetForm.preset} onChange={(value) => setPresetValue("preset", value)} options={presetOptions} />
          <TextField label="Seed" type="number" value={presetForm.seed} onChange={(value) => setPresetValue("seed", value)} />
          <button type="submit">Generate Dataset</button>
        </form>
        <form className="work-form" onSubmit={importDataset}>
          <h1>CSV or URL Import</h1>
          <TextField label="Name" value={csvForm.name} onChange={(value) => setCsvValue("name", value)} />
          <TextField label="Edges CSV" value={csvForm.edges_path} onChange={(value) => setCsvValue("edges_path", value)} placeholder="edges.csv or https://..." />
          <TextField label="Node graph mapping CSV" value={csvForm.node_graph_mapping_path} onChange={(value) => setCsvValue("node_graph_mapping_path", value)} placeholder="node_graph_mapping.csv or https://..." />
          <TextField label="Graph labels CSV" value={csvForm.graph_label_path} onChange={(value) => setCsvValue("graph_label_path", value)} placeholder="optional" />
          <TextField label="Node features CSV" value={csvForm.node_features_path} onChange={(value) => setCsvValue("node_features_path", value)} placeholder="optional" />
          <button type="submit">Import Dataset</button>
        </form>
      </div>
    </div>
  );
}

function FigurePane({ activeDocument, graphElements, collectionSummary, embeddingTable, jobs }) {
  const title =
    {
      data: "Figure 1 - Workflow Output",
      explore: "Figure 1 - Selected Graph",
      features: "Feature Table Preview",
      embeddings: "Approximate Wasserstein Embedding",
      models: "Model Run Status",
      exports: "Export Status",
    }[activeDocument] || "Figure";

  return (
    <section className="figure">
      <div className="figure-title">{title}</div>
      {activeDocument === "explore" ? (
        <GraphCanvas graphElements={graphElements} collectionSummary={collectionSummary} />
      ) : activeDocument === "embeddings" ? (
        <EmbeddingPlot payload={embeddingTable} />
      ) : activeDocument === "models" ? (
        <JobStatus jobs={jobs.filter((job) => job.type === "models")} />
      ) : activeDocument === "features" ? (
        <div className="preview-surface">
          <span className="preview-table" />
          <span className="preview-note">feature table</span>
        </div>
      ) : activeDocument === "exports" ? (
        <div className="preview-surface">
          <span className="preview-page" />
          <span className="preview-note">python export</span>
        </div>
      ) : (
        <div className="workflow-figure">
          <span className="workflow-node source">CSV</span>
          <span className="workflow-line l1" />
          <span className="workflow-node features">Features</span>
          <span className="workflow-line l2" />
          <span className="workflow-node embed">Embeddings</span>
          <span className="workflow-line l3" />
          <span className="workflow-node model">Model</span>
        </div>
      )}
    </section>
  );
}

function GraphCanvas({ graphElements, collectionSummary }) {
  if (!graphElements?.nodes?.length) {
    return <div className="empty figure-empty">{collectionSummary ? "No graph elements available." : "Load a dataset to draw a graph."}</div>;
  }
  const width = 760;
  const height = 420;
  const radius = Math.min(width, height) * 0.38;
  const cx = width / 2;
  const cy = height / 2;
  const positions = {};
  graphElements.nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(graphElements.nodes.length, 1);
    positions[node.data.id] = { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
  });
  return (
    <svg className="graph-canvas" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Graph ${graphElements.graphId}`}>
      {graphElements.edges.map((edge, index) => {
        const source = positions[edge.data.source];
        const target = positions[edge.data.target];
        if (!source || !target) return null;
        return <line key={`${edge.data.source}-${edge.data.target}-${index}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} className="graph-edge-line" />;
      })}
      {graphElements.nodes.map((node) => {
        const position = positions[node.data.id];
        return (
          <circle key={node.data.id} cx={position.x} cy={position.y} r="5" className="graph-dot">
            <title>{node.data.label}</title>
          </circle>
        );
      })}
    </svg>
  );
}

function EmbeddingPlot({ payload }) {
  const rows = payload?.rows || [];
  if (!rows.length) {
    return <div className="empty figure-empty">Load an embedding artifact to draw the scatter plot.</div>;
  }
  const width = 760;
  const height = 420;
  const pad = 38;
  const xs = rows.map((row) => Number(row.emb_0 || 0));
  const ys = rows.map((row) => Number(row.emb_1 || 0));
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 1);
  return (
    <svg className="embedding-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Embedding scatter plot">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis-line" />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis-line" />
      {rows.map((row, index) => {
        const x = scale(Number(row.emb_0 || 0), xMin, xMax, pad, width - pad);
        const y = scale(Number(row.emb_1 || 0), yMin, yMax, height - pad, pad);
        return (
          <circle key={`${row.graph_id}-${index}`} cx={x} cy={y} r="5" className={index % 3 === 0 ? "plot-dot orange" : "plot-dot"}>
            <title>{`graph ${row.graph_id} (${xs[index]}, ${ys[index]})`}</title>
          </circle>
        );
      })}
    </svg>
  );
}

function MetricGrid({ summary }) {
  const metrics = summary
    ? [
        ["Graphs", summary.num_graphs],
        ["Nodes", summary.total_nodes],
        ["Edges", summary.total_edges],
        ["Labels", summary.has_labels ? "yes" : "no"],
      ]
    : [
        ["Graphs", "-"],
        ["Nodes", "-"],
        ["Edges", "-"],
        ["Labels", "-"],
      ];
  return (
    <div className="metric-grid">
      {metrics.map(([label, value]) => (
        <div className="metric" key={label}>
          <div className="metric-value">{value}</div>
          <div className="metric-label">{label}</div>
        </div>
      ))}
    </div>
  );
}

function DataTable({ payload, compact = false }) {
  if (!payload?.rows?.length) {
    return <div className={compact ? "empty table-empty compact" : "empty table-empty"}>No rows to display.</div>;
  }
  return (
    <div className={compact ? "table-wrap compact" : "table-wrap"}>
      <table>
        <thead>
          <tr>
            {payload.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {payload.rows.map((row, index) => (
            <tr key={index}>
              {payload.columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProjectTree({ artifacts, selectedArtifactId, onSelect }) {
  if (!artifacts.length) {
    return (
      <div className="tree">
        <div className="tree-row active">
          <span>&gt;</span>
          <span>manifest.json</span>
          <span className="muted">empty</span>
        </div>
        <div className="tree-row">
          <span>v</span>
          <span>artifacts</span>
          <span className="muted">folder</span>
        </div>
      </div>
    );
  }
  return (
    <div className="tree">
      <div className="tree-row">
        <span>&gt;</span>
        <span>manifest.json</span>
        <span className="muted">project</span>
      </div>
      <div className="tree-row">
        <span>v</span>
        <span>artifacts</span>
        <span className="muted">{artifacts.length}</span>
      </div>
      {artifacts.map((artifact) => (
        <button key={artifact.id} className={artifact.id === selectedArtifactId ? "tree-row artifact-row active" : "tree-row artifact-row"} onClick={() => onSelect(artifact)}>
          <span />
          <span>{artifact.name}</span>
          <span className="muted">{artifact.type}</span>
        </button>
      ))}
    </div>
  );
}

function WorkspaceTable({ groups, jobs }) {
  const rows = [
    ["datasets", groups.dataset.length, "GraphCollection"],
    ["features_all", groups.features.length, "Features"],
    ["embeddings_3d", groups.embeddings.length, "Embeddings"],
    ["model_runs", groups.model.length, "Model"],
    ["jobs", jobs.length, "Job"],
  ];
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Value</th>
          <th>Class</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, value, className]) => (
          <tr key={name}>
            <td>{name}</td>
            <td>{value}</td>
            <td>{className}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function JobItem({ job }) {
  return (
    <div className="job">
      <div className="job-name">{job.name}</div>
      <div className="job-meta">
        {job.type} | {job.status}
        <br />
        {job.result_artifact_id || job.id}
      </div>
      <div className="progress">
        <span style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} />
      </div>
    </div>
  );
}

function JobStatus({ jobs }) {
  if (!jobs.length) return <div className="empty figure-empty">No model jobs yet.</div>;
  return (
    <div className="job-status-list">
      {jobs.slice().reverse().map((job) => (
        <JobItem key={job.id} job={job} />
      ))}
    </div>
  );
}

function ToolButton({ icon, label, wide = false, onClick }) {
  return (
    <button className={wide ? "tool wide" : "tool"} onClick={onClick} title={label.replace("\n", " ")}>
      <Icon name={icon} />
      <span className="tool-label">
        {label.split("\n").map((part, index) => (
          <React.Fragment key={part}>
            {index > 0 ? <br /> : null}
            {part}
          </React.Fragment>
        ))}
      </span>
    </button>
  );
}

function Icon({ name }) {
  if (name === "new") {
    return (
      <span className="pic-icon icon-new">
        <span className="page" />
        <span className="page-fold" />
        <span className="plus-badge" />
      </span>
    );
  }
  if (name === "open") {
    return (
      <span className="pic-icon icon-open">
        <span className="folder-sheet" />
        <span className="folder" />
      </span>
    );
  }
  if (name === "import") {
    return (
      <span className="pic-icon icon-import">
        <span className="table-window" />
        <span className="csv-chip">CSV</span>
      </span>
    );
  }
  if (name === "url") {
    return (
      <span className="pic-icon icon-url">
        <span className="globe">
          <span className="lat" />
          <span className="lon" />
        </span>
        <span className="chain" />
      </span>
    );
  }
  if (name === "graph") {
    return (
      <span className="pic-icon icon-graph">
        <span className="graph-edge" />
        <span className="graph-edge edge-2" />
        <span className="graph-node n1" />
        <span className="graph-node n2" />
        <span className="graph-node n3" />
      </span>
    );
  }
  if (name === "run") return <span className="pic-icon icon-run"><span className="play-triangle" /></span>;
  if (name === "stop") return <span className="pic-icon icon-stop"><span className="stop-block" /></span>;
  if (name === "section") return <span className="pic-icon icon-section"><span className="run-lines" /><span className="mini-play" /></span>;
  if (name === "features") return <span className="pic-icon icon-features"><span className="toolbox" /><span className="gear-face" /></span>;
  if (name === "embed") return <span className="pic-icon icon-embed"><span className="axes-mini" /><span className="dot d1" /><span className="dot d2" /><span className="dot d3" /></span>;
  if (name === "variable") return <span className="pic-icon icon-variable"><span className="table-window" /></span>;
  if (name === "layout") return <span className="pic-icon icon-layout"><span className="panel-window" /></span>;
  if (name === "panels") return <span className="pic-icon icon-panels"><span className="stack-pane p1" /><span className="stack-pane p2" /><span className="stack-pane p3" /></span>;
  return <span className="pic-icon" />;
}

function TextField({ label, value, onChange, type = "text", min, placeholder }) {
  return (
    <label>
      {label}
      <input type={type} min={min} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function SelectField({ label, value, onChange, options, showType = false }) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.length ? (
          options.map((artifact) => (
            <option key={artifact.id} value={artifact.id}>
              {artifact.name} ({showType ? artifact.type : artifact.id})
            </option>
          ))
        ) : (
          <option value="">No artifact available</option>
        )}
      </select>
    </label>
  );
}

function SimpleSelect({ label, value, onChange, options }) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function ribbonForTab(tab, actions) {
  const commonEnvironment = {
    label: "Environment",
    tools: [
      { icon: "layout", label: "Layout", onClick: () => actions.activateDocument("explore") },
      { icon: "panels", label: "Panels", onClick: actions.refreshProject },
    ],
  };

  if (tab === "PLOTS") {
    return [
      { label: "Figures", tools: [{ icon: "graph", label: "Load\nGraph", wide: true, onClick: actions.loadExplore }, { icon: "embed", label: "Embedding\nPlot", wide: true, onClick: actions.loadEmbeddingPlot }] },
      { label: "Tables", tools: [{ icon: "variable", label: "Feature\nTable", wide: true, onClick: actions.loadFeatureTable }] },
      commonEnvironment,
    ];
  }

  if (tab === "FEATURES") {
    return [
      { label: "Features", tools: [{ icon: "features", label: "Feature\nSet", wide: true, onClick: actions.submitFeatureJob }, { icon: "variable", label: "Load\nTable", wide: true, onClick: actions.loadFeatureTable }] },
      { label: "Run", tools: [{ icon: "run", label: "Run", onClick: actions.submitFeatureJob }, { icon: "stop", label: "Stop", onClick: actions.stopCurrentJob }] },
      commonEnvironment,
    ];
  }

  if (tab === "EMBEDDINGS") {
    return [
      { label: "Embeddings", tools: [{ icon: "embed", label: "Graph\nEmbed", wide: true, onClick: actions.submitEmbeddingJob }, { icon: "graph", label: "Load\nPlot", wide: true, onClick: actions.loadEmbeddingPlot }] },
      { label: "Run", tools: [{ icon: "run", label: "Run", onClick: actions.submitEmbeddingJob }, { icon: "stop", label: "Stop", onClick: actions.stopCurrentJob }] },
      commonEnvironment,
    ];
  }

  if (tab === "MODELS") {
    return [
      { label: "Model", tools: [{ icon: "run", label: "Run\nModel", wide: true, onClick: actions.submitModelJob }, { icon: "variable", label: "Export\nResult", wide: true, onClick: actions.exportPython }] },
      { label: "Run", tools: [{ icon: "run", label: "Run", onClick: actions.submitModelJob }, { icon: "stop", label: "Stop", onClick: actions.stopCurrentJob }] },
      commonEnvironment,
    ];
  }

  if (tab === "APPS") {
    return [
      { label: "Apps", tools: [{ icon: "variable", label: "Python\nExport", wide: true, onClick: actions.exportPython }, { icon: "open", label: "Refresh", onClick: actions.refreshProject }] },
      { label: "Data Apps", tools: [{ icon: "import", label: "Import\nCSV", wide: true, onClick: () => actions.activateDocument("data") }, { icon: "url", label: "URL\nData", wide: true, onClick: () => actions.activateDocument("data") }] },
      commonEnvironment,
    ];
  }

  return [
    { label: "File", tools: [{ icon: "new", label: "New\nProject", wide: true, onClick: () => actions.activateDocument("data") }, { icon: "open", label: "Open", onClick: actions.refreshProject }] },
    { label: "Data", tools: [{ icon: "import", label: "Import\nCSV", wide: true, onClick: actions.importDataset }, { icon: "url", label: "URL\nData", wide: true, onClick: actions.importDataset }, { icon: "graph", label: "Preset\nGraph", wide: true, onClick: actions.generateDataset }] },
    { label: "Run", tools: [{ icon: "run", label: "Run", onClick: actions.runPrimaryAction }, { icon: "stop", label: "Stop", onClick: actions.stopCurrentJob }, { icon: "section", label: "Run\nSection", wide: true, onClick: actions.runPrimaryAction }] },
    { label: "Analysis", tools: [{ icon: "features", label: "Feature\nSet", wide: true, onClick: () => actions.activateDocument("features") }, { icon: "embed", label: "Graph\nEmbed", wide: true, onClick: () => actions.activateDocument("embeddings") }, { icon: "variable", label: "Variable\nEditor", wide: true, onClick: actions.loadFeatureTable }] },
    commonEnvironment,
  ];
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch (_) {
      // Keep the plain text response.
    }
    throw new Error(detail);
  }
  return response.json();
}

function cleanPayload(payload) {
  return Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, value === "" ? null : value]));
}

function scale(value, fromMin, fromMax, toMin, toMax) {
  if (fromMax === fromMin) return (toMin + toMax) / 2;
  return toMin + ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin);
}

function formatCell(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

createRoot(document.getElementById("root")).render(<App />);
