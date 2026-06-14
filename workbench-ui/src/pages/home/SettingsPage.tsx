import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Power, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { api, type McpClientConfigSnippet, type McpSettingsResponse } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { useMcpSettings } from "../../hooks/useWorkspace";

type SettingsTab = "general" | "agentic" | "docs";

interface SettingsDocSection {
  heading: string;
  body?: string[];
  bullets?: string[];
  code?: string;
}

interface SettingsDocTopic {
  id: string;
  title: string;
  summary: string;
  sections: SettingsDocSection[];
}

const SETTINGS_TABS: { id: SettingsTab; label: string }[] = [
  { id: "general", label: "General" },
  { id: "agentic", label: "Agentic" },
  { id: "docs", label: "Docs" }
];

const SETTINGS_DOC_TOPICS: SettingsDocTopic[] = [
  {
    id: "overview",
    title: "Overview",
    summary: "NEExT Workbench is a local, project-first interface over real NEExT graph machine learning workflows.",
    sections: [
      {
        heading: "What Workbench Manages",
        body: [
          "Workbench organizes work into projects. Each project owns datasets, feature sets, embeddings, models, jobs, and local trash state.",
          "The main workflow is dataset-first: configure or select a Dataset artifact, compute Feature artifacts, compute Embedding artifacts, then train Model artifacts."
        ]
      },
      {
        heading: "Current Boundaries",
        bullets: [
          "Project Create and custom Feature Create are the active Create workflows.",
          "Dataset, Embedding, and Model Create are archived until their workflows are designed.",
          "Broader import/export remains deferred except for current-table CSV export in Dataset Explore.",
          "Home Help is unchanged; operational docs live here in Settings Docs."
        ]
      }
    ]
  },
  {
    id: "workbench-flow",
    title: "Workbench Flow",
    summary: "Use the top Spaces, Ribbon commands, Left Panel context, and Center Views together.",
    sections: [
      {
        heading: "Navigation Model",
        bullets: [
          "Spaces group the main work areas: Home, Datasets, Features, Embeddings, and Models.",
          "Ribbon commands switch the active Center View inside the selected Space.",
          "The Left Panel shows the active project branch and scopes downstream artifact lists.",
          "The Right Panel shows Inspector details and job status; the Command Window shows job output."
        ]
      },
      {
        heading: "Typical Path",
        bullets: [
          "Create or select a project from Home.",
          "Configure a Dataset from the Dataset Library, then run Dataset preparation.",
          "Configure built-in Features or create a custom Feature for the active Dataset, then run Feature jobs.",
          "Configure Embeddings from Feature artifacts and Models from Embedding artifacts.",
          "Use Explore views to inspect statistics, plots, tables, lineage, and previews."
        ]
      }
    ]
  },
  {
    id: "projects-datasets",
    title: "Projects and Datasets",
    summary: "Projects are the workspace boundary. Datasets are explicit roots of the compute graph.",
    sections: [
      {
        heading: "Projects",
        body: [
          "A project stores its manifest, jobs, and typed artifact folders under the local Workbench workspace. Project display names can change without changing artifact IDs.",
          "Project deletion moves the project folder to workspace trash. Restore is available from Home Trash when the original live project folder does not already exist."
        ]
      },
      {
        heading: "Datasets",
        bullets: [
          "Dataset Library rows are templates, not executable project artifacts.",
          "A catalog row must be configured into a project Dataset artifact before Features, Embeddings, or Models can use it.",
          "Dataset preparation writes canonical Parquet outputs, mappings, summaries, and job logs.",
          "Browser previews stay limited and paginated; they do not load complete large files by default."
        ]
      }
    ]
  },
  {
    id: "features-custom",
    title: "Features and Custom Features",
    summary: "Feature artifacts compute node-level values for one prepared Dataset artifact.",
    sections: [
      {
        heading: "Built-in Features",
        body: [
          "Feature Library workflows are dataset-first. Select a Dataset in the Left Panel, choose a feature method from the Library, configure it, save the planned artifact, then run it.",
          "If a Feature run targets a planned Dataset, Workbench prepares the Dataset first before computing the Feature output."
        ]
      },
      {
        heading: "Custom Feature Contract",
        bullets: [
          "Custom Feature Create requires an active completed Dataset artifact.",
          "The Python code must define compute_feature(graph).",
          "The function must return a pandas.DataFrame with columns ordered as node_id, graph_id, then one or more numeric feature columns.",
          "Validate runs against the first prepared graph. Save repeats backend validation before creating the planned Feature artifact.",
          "Custom code is trusted local Python, not sandboxed. Missing packages are reported clearly, but Workbench does not install packages."
        ],
        code: `import pandas as pd

def compute_feature(graph):
    nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
    values = [float(graph.G.degree(node)) for node in nodes]
    df = pd.DataFrame({
        "node_id": nodes,
        "graph_id": graph.graph_id,
        "custom_degree": values,
    })
    return df[["node_id", "graph_id", "custom_degree"]]`
      }
    ]
  },
  {
    id: "embeddings-models",
    title: "Embeddings and Models",
    summary: "Embeddings summarize graph-level structure from Feature artifacts. Models train from Embedding artifacts.",
    sections: [
      {
        heading: "Embeddings",
        bullets: [
          "Embedding Library rows are templates. Configure one into a project Embedding artifact from the active Dataset context.",
          "Embedding Configure lists Feature artifacts from the active Dataset only.",
          "When an active Feature belongs to the active Dataset branch, it is preselected for Embedding Configure.",
          "Embedding execution can auto-run planned or failed upstream Dataset and Feature work."
        ]
      },
      {
        heading: "Models",
        bullets: [
          "Model Library rows are templates. Configure one into a project Model artifact from the active Dataset context.",
          "Model Configure lists Embedding artifacts from the active Dataset only.",
          "When an active Embedding belongs to the active Dataset branch, it is preselected for Model Configure.",
          "Model execution can auto-run planned or failed upstream Embedding, Feature, and Dataset work before training."
        ]
      }
    ]
  },
  {
    id: "artifact-lifecycle",
    title: "Artifact Lifecycle",
    summary: "Artifacts are persisted compute graph nodes with lineage, status, jobs, and trash behavior.",
    sections: [
      {
        heading: "Lineage",
        body: [
          "Dataset artifacts are roots. Feature artifacts reference Dataset inputs, Embedding artifacts reference Feature inputs, and Model artifacts reference Embedding inputs.",
          "The Left Panel and Center Views use that lineage to keep project, dataset, feature, embedding, and model context consistent."
        ]
      },
      {
        heading: "Delete and Restore",
        bullets: [
          "Leaf artifacts can be deleted directly.",
          "Artifacts with downstream dependents require a cascade confirmation.",
          "Cascade delete moves the selected artifact and all downstream dependents into one project-scoped trash bundle.",
          "Bundle restore is available when no live target folder conflicts.",
          "Queued or running jobs targeting any artifact in the delete set block deletion."
        ]
      }
    ]
  },
  {
    id: "library-quickstart",
    title: "NEExT Library Quickstart",
    summary: "Use the Python library directly when you want notebook or script control outside Workbench.",
    sections: [
      {
        heading: "Core Pipeline",
        body: [
          "The library flow mirrors Workbench: load graph data, compute node features, compute graph embeddings, then train or evaluate a model.",
          "Graph data can come from CSV files, DataFrames, or NetworkX graphs. Workbench uses the same underlying NEExT capabilities for its persisted workflows."
        ],
        code: `from NEExT.framework import NEExT

nxt = NEExT()
graphs = nxt.read_from_csv(
    "edges.csv",
    "node_graph_mapping.csv",
    "graph_labels.csv",
)
features = nxt.compute_node_features(graphs, feature_list=["all"])
embeddings = nxt.compute_graph_embeddings(
    graphs,
    features,
    embedding_algorithm="approx_wasserstein",
    embedding_dimension=8,
)
results = nxt.train_ml_model(graphs, embeddings, model_type="classifier")`
      },
      {
        heading: "CSV Inputs",
        bullets: [
          "edges.csv uses src_node_id and dest_node_id.",
          "node_graph_mapping.csv uses node_id and graph_id.",
          "graph_labels.csv uses graph_id and graph_label.",
          "Optional node feature CSVs start with node_id and graph_id, followed by feature columns."
        ]
      }
    ]
  },
  {
    id: "agentic-mcp",
    title: "Agentic and MCP",
    summary: "The Agentic tab exposes local MCP setup for clients that need controlled Workbench access.",
    sections: [
      {
        heading: "MCP Setup",
        bullets: [
          "Enable MCP from Settings Agentic when a supported local client needs Workbench access.",
          "Workbench shows the full token only once. Regenerate it when a fresh token and copy-ready snippets are needed.",
          "Client snippets include the workspace path and NEEXT_WORKBENCH_MCP_TOKEN environment variable.",
          "Disable MCP when clients should no longer connect."
        ]
      },
      {
        heading: "Security Boundary",
        body: [
          "MCP setup is local Workbench configuration. Keep generated tokens out of commits, screenshots, logs, and shared notes.",
          "Custom Feature Create is separate from MCP. Custom feature code is trusted local Python and runs in the local Workbench environment."
        ]
      }
    ]
  }
];

function ConfigSnippet({ snippet }: { snippet: McpClientConfigSnippet }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(snippet.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section className="settings-snippet">
      <header className="settings-snippet-head">
        <div>
          <h4>{snippet.label}</h4>
          <span className="muted mono">{snippet.target}</span>
        </div>
        <button type="button" className="btn" onClick={copy} aria-label={`Copy ${snippet.label} MCP config`}>
          <Copy /> {copied ? "Copied" : "Copy"}
        </button>
      </header>
      <pre className="settings-code">
        <code>{snippet.content}</code>
      </pre>
    </section>
  );
}

function SettingsDocsView() {
  const [activeTopicId, setActiveTopicId] = useState(SETTINGS_DOC_TOPICS[0].id);
  const activeTopic = SETTINGS_DOC_TOPICS.find((topic) => topic.id === activeTopicId) || SETTINGS_DOC_TOPICS[0];

  return (
    <div className="settings-tab-panel settings-docs-panel">
      <nav className="settings-docs-sidebar" aria-label="Docs topics">
        {SETTINGS_DOC_TOPICS.map((topic) => (
          <button
            key={topic.id}
            type="button"
            className={`settings-doc-topic ${topic.id === activeTopic.id ? "is-active" : ""}`}
            onClick={() => setActiveTopicId(topic.id)}
          >
            <span>{topic.title}</span>
            <small>{topic.summary}</small>
          </button>
        ))}
      </nav>

      <article className="settings-doc-content">
        <header className="settings-doc-head">
          <h4>{activeTopic.title}</h4>
          <p>{activeTopic.summary}</p>
        </header>
        {activeTopic.sections.map((section) => (
          <section className="settings-doc-section" key={section.heading}>
            <h5>{section.heading}</h5>
            {section.body?.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
            {section.bullets ? (
              <ul>
                {section.bullets.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : null}
            {section.code ? (
              <pre className="settings-code settings-doc-code">
                <code>{section.code}</code>
              </pre>
            ) : null}
          </section>
        ))}
      </article>
    </div>
  );
}

export function SettingsView() {
  const queryClient = useQueryClient();
  const settingsQuery = useMcpSettings();
  const [latestSetup, setLatestSetup] = useState<McpSettingsResponse | null>(null);
  const [activeSettingsTab, setActiveSettingsTab] = useState<SettingsTab>("general");
  const [activeClient, setActiveClient] = useState("claude_desktop");

  const enableMcp = useMutation({
    mutationFn: api.enableMcpSettings,
    onSuccess: (settings) => {
      setLatestSetup(settings);
      queryClient.setQueryData(["mcp-settings"], settings);
      queryClient.invalidateQueries({ queryKey: ["mcp-settings"] });
    }
  });

  const regenerateMcp = useMutation({
    mutationFn: api.regenerateMcpSettings,
    onSuccess: (settings) => {
      setLatestSetup(settings);
      queryClient.setQueryData(["mcp-settings"], settings);
      queryClient.invalidateQueries({ queryKey: ["mcp-settings"] });
    }
  });

  const disableMcp = useMutation({
    mutationFn: api.disableMcpSettings,
    onSuccess: (settings) => {
      setLatestSetup(null);
      queryClient.setQueryData(["mcp-settings"], settings);
      queryClient.invalidateQueries({ queryKey: ["mcp-settings"] });
    }
  });

  const settings = settingsQuery.data;
  const setupSettings = latestSetup?.one_time_token ? latestSetup : null;
  const busy = enableMcp.isPending || regenerateMcp.isPending || disableMcp.isPending;
  const error = enableMcp.error?.message || regenerateMcp.error?.message || disableMcp.error?.message || settingsQuery.error?.message;
  const statusLabel = settings?.enabled ? "enabled" : "disabled";
  const snippets = useMemo(() => setupSettings?.client_configs || [], [setupSettings]);
  const selectedSnippet = snippets.find((snippet) => snippet.client === activeClient) || snippets[0];

  return (
    <div className="workflow settings-workflow">
      <section className="artifact-table settings-surface">
        <header className="artifact-table-head settings-surface-head">
          <h3 className="artifact-table-title">
            <FcIcon name="settings" size={16} />
            Settings
          </h3>
        </header>
        <div className="tab-strip">
          {SETTINGS_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab-btn ${activeSettingsTab === tab.id ? "is-active" : ""}`}
              onClick={() => setActiveSettingsTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeSettingsTab === "general" ? (
          <div className="settings-tab-panel settings-general-panel">
            <EmptyState compact>No general settings yet.</EmptyState>
          </div>
        ) : activeSettingsTab === "docs" ? (
          <SettingsDocsView />
        ) : (
          <div className="settings-tab-panel settings-agentic-panel">
            <section className="settings-panel settings-mcp-panel">
              <header className="settings-panel-head">
                <span className={`settings-status ${settings?.enabled ? "is-enabled" : "is-disabled"}`}>
                  {settings?.enabled ? <ShieldCheck /> : <ShieldOff />}
                  MCP {statusLabel}
                </span>
                {settings?.token_preview ? <span className="mono muted">{settings.token_preview}</span> : null}
              </header>
              <div className="settings-actions">
                {settings?.enabled ? (
                  <>
                    <button type="button" className="btn" onClick={() => regenerateMcp.mutate()} disabled={busy}>
                      <RefreshCw /> Regenerate Token
                    </button>
                    <button type="button" className="btn btn-danger" onClick={() => disableMcp.mutate()} disabled={busy}>
                      <Power /> Disable MCP
                    </button>
                  </>
                ) : (
                  <button type="button" className="btn btn-primary" onClick={() => enableMcp.mutate()} disabled={busy || settingsQuery.isLoading}>
                    <Power /> Enable MCP
                  </button>
                )}
              </div>
              {error ? <p className="error-text">{error}</p> : null}
              {settings?.enabled && !setupSettings ? (
                <p className="settings-note">The full token is hidden after setup. Regenerate it to display fresh client snippets.</p>
              ) : null}
              {setupSettings?.one_time_token ? (
                <div className="settings-token">
                  <span>One-time token</span>
                  <code>{setupSettings.one_time_token}</code>
                </div>
              ) : null}
            </section>

            <section className="settings-panel settings-install-panel">
              <header className="settings-panel-head">
                <h4>Install</h4>
              </header>
              <pre className="settings-code settings-install-code">
                <code>{'python3 -m pip install --upgrade "NEExT[workbench-mcp]"'}</code>
              </pre>
            </section>

            {settings?.enabled ? (
              <section className="settings-panel settings-configs">
                <header className="settings-panel-head">
                  <h4>Client Config Snippets</h4>
                  <span className="muted">Shown with the current token once</span>
                </header>
                {snippets.length ? (
                  <>
                    <div className="tab-strip settings-client-tabs">
                      {snippets.map((snippet) => (
                        <button
                          key={snippet.client}
                          type="button"
                          className={`tab-btn ${selectedSnippet?.client === snippet.client ? "is-active" : ""}`}
                          onClick={() => setActiveClient(snippet.client)}
                        >
                          {snippet.label}
                        </button>
                      ))}
                    </div>
                    {selectedSnippet ? <ConfigSnippet snippet={selectedSnippet} /> : null}
                  </>
                ) : (
                  <div className="artifact-table-empty settings-config-empty">
                    <EmptyState compact>Regenerate the token to show copy-ready snippets.</EmptyState>
                  </div>
                )}
              </section>
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}
