import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Power, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { api, type McpClientConfigSnippet, type McpSettingsResponse, type ProjectManifest, type WorkspaceInfo } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { useMcpActivity, useMcpApprovals, useMcpSettings } from "../../hooks/useWorkspace";

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

interface SettingsViewProps {
  workspace?: WorkspaceInfo;
  projects: ProjectManifest[];
  activeProject?: ProjectManifest;
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
          "The main workflow is dataset-first: add or select a Dataset artifact, compute Feature artifacts, compute Embedding artifacts, then train Model artifacts."
        ]
      },
      {
        heading: "Current Boundaries",
        bullets: [
          "Project Create and custom Feature Create are the active Create workflows.",
          "Dataset, Embedding, and Model Create are archived until their workflows are designed.",
          "Broader import/export remains deferred except for Dataset Import and current-table CSV export in Dataset Explore.",
          "Operational docs live here in Settings Docs."
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
          "Add a Dataset from the Dataset Library, then prepare it.",
          "Add built-in Features or create a custom Feature for the active Dataset, then compute Feature jobs.",
          "Add Embeddings from Feature artifacts, then train Models from Embedding artifacts.",
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
          "A catalog row must be added to the project as a Dataset artifact before Features, Embeddings, or Models can use it.",
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
          "Feature Library workflows are dataset-first. Select a Dataset in the Left Panel, choose a feature method from the Library, create the Draft artifact, then compute it.",
          "If a Feature computation targets a Draft Dataset, Workbench prepares the Dataset first before computing the Feature output."
        ]
      },
      {
        heading: "Custom Feature Contract",
        bullets: [
          "Custom Feature Create requires an active completed Dataset artifact.",
          "The Python code must define compute_feature(graph).",
          "The function must return a pandas.DataFrame with columns ordered as node_id, graph_id, then one or more numeric feature columns.",
          "Validate runs against the first prepared graph. Create repeats backend validation before creating the Draft Feature artifact.",
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
          "Embedding Library rows are templates. Add one into a project Embedding artifact from the active Dataset context.",
          "Embedding Add-to-Project lists Feature artifacts from the active Dataset only.",
          "When an active Feature belongs to the active Dataset branch, it is preselected.",
          "Embedding execution can auto-run Draft or failed upstream Dataset and Feature work."
        ]
      },
      {
        heading: "Models",
        bullets: [
          "Model Library rows are templates. Add one into a project Model artifact from the active Dataset context.",
          "Model Add-to-Project lists Embedding artifacts from the active Dataset only.",
          "When an active Embedding belongs to the active Dataset branch, it is preselected.",
          "Model execution can auto-run Draft or failed upstream Embedding, Feature, and Dataset work before training."
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
          "Claude Desktop only reaches a local server over stdio: its snippet launches the Python interpreter running Workbench with the Workbench MCP launcher.",
          "MCP Inspector, Cursor, Claude Code, and generic clients can use the local Streamable HTTP endpoint at http://127.0.0.1:8765/mcp.",
          "Remote Claude connectors are separate from claude_desktop_config.json and require a publicly reachable server, not 127.0.0.1.",
          "On macOS, Claude Desktop cannot read servers inside ~/Desktop, ~/Documents, or ~/Downloads. When the Python environment lives there, Workbench replaces the Claude Desktop snippet with remediation steps instead of a config that would fail to launch.",
          "Disable MCP when clients should no longer connect."
        ]
      },
      {
        heading: "Agentic Behavior",
        bullets: [
          "MCP tools can read catalogs and artifacts, add current approved library workflows to projects, run jobs, preview and analyze outputs, request Workbench navigation, and record visible activity.",
          "Tool access is gated by scopes: read, write, run, custom-code, ui-control, export, and lifecycle.",
          "MCP UI navigation can open existing Spaces, Center Views, artifacts, graphs, nodes, and approved add/create form drafts.",
          "MCP delete tools create Workbench approval requests instead of deleting immediately.",
          "Recent MCP activity is visible in Settings Agentic and the Command Window."
        ]
      },
      {
        heading: "Security Boundary",
        body: [
          "MCP setup is local Workbench configuration. Keep generated tokens out of commits, screenshots, logs, and shared notes.",
          "Custom feature code is trusted local Python and runs in the local Workbench environment. Delete requests are enforced through Workbench approval.",
          "Remote OAuth, multi-user MCP hosting, and deferred Workbench workflows are not part of the current MCP surface."
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

export function SettingsView({ workspace, projects, activeProject }: SettingsViewProps) {
  const queryClient = useQueryClient();
  const settingsQuery = useMcpSettings();
  const activityQuery = useMcpActivity();
  const approvalsQuery = useMcpApprovals();
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

  const approveMcpRequest = useMutation({
    mutationFn: api.approveMcpRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp-approvals"] });
      queryClient.invalidateQueries({ queryKey: ["mcp-activity"] });
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["trash"] });
    }
  });

  const denyMcpRequest = useMutation({
    mutationFn: (requestId: string) => api.denyMcpRequest(requestId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp-approvals"] });
      queryClient.invalidateQueries({ queryKey: ["mcp-activity"] });
    }
  });

  const settings = settingsQuery.data;
  const setupSettings = latestSetup?.one_time_token ? latestSetup : null;
  const busy = enableMcp.isPending || regenerateMcp.isPending || disableMcp.isPending;
  const approvalBusy = approveMcpRequest.isPending || denyMcpRequest.isPending;
  const error =
    enableMcp.error?.message ||
    regenerateMcp.error?.message ||
    disableMcp.error?.message ||
    approveMcpRequest.error?.message ||
    denyMcpRequest.error?.message ||
    settingsQuery.error?.message;
  const statusLabel = settings?.enabled ? "enabled" : "disabled";
  const snippets = useMemo(() => setupSettings?.client_configs || [], [setupSettings]);
  const selectedSnippet = snippets.find((snippet) => snippet.client === activeClient) || snippets[0];
  const pendingApprovals = (approvalsQuery.data?.approvals || []).filter((approval) => approval.status === "pending");
  const recentActivity = activityQuery.data?.entries || [];
  const availableCapabilities = settings?.capabilities.filter((capability) => capability.available) || [];
  const writeCapabilityCount = availableCapabilities.filter((capability) => !capability.read_only).length;
  const destructiveCapabilityCount = availableCapabilities.filter((capability) => capability.destructive).length;

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
            <section className="settings-panel">
              <header className="settings-panel-head">
                <h4>Workspace</h4>
              </header>
              <div className="settings-general-grid">
                <div className="settings-general-item">
                  <span>Location</span>
                  <strong className="mono">{workspace?.path || "Unavailable"}</strong>
                </div>
                <div className="settings-general-item">
                  <span>Schema Version</span>
                  <strong>{workspace?.version || "Unavailable"}</strong>
                </div>
                <div className="settings-general-item">
                  <span>Projects</span>
                  <strong>{workspace?.projects ?? projects.length}</strong>
                </div>
                <div className="settings-general-item">
                  <span>Active Project</span>
                  <strong>{activeProject?.name || "None"}</strong>
                </div>
              </div>
            </section>
            <section className="settings-panel">
              <header className="settings-panel-head">
                <h4>About NEExT</h4>
              </header>
              <div className="settings-general-grid">
                <div className="settings-general-item">
                  <span>Application</span>
                  <strong>NEExT Workbench</strong>
                </div>
                <div className="settings-general-item">
                  <span>License</span>
                  <strong>MIT License</strong>
                </div>
                <div className="settings-general-item">
                  <span>Runtime</span>
                  <strong>Local single-user Workbench</strong>
                </div>
                <div className="settings-general-item">
                  <span>Release Notes</span>
                  <strong>No bundled release notes file</strong>
                </div>
              </div>
            </section>
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
              {settings?.endpoint_url ? (
                <div className="settings-token">
                  <span>Local endpoint</span>
                  <code>{settings.endpoint_url}</code>
                </div>
              ) : null}
              {settings?.scopes?.length ? (
                <p className="settings-note">Enabled scopes: {settings.scopes.join(", ")}. Delete requests require Workbench approval.</p>
              ) : null}
              {settings ? (
                <div className="settings-mcp-meta">
                  <span>Transport: <strong>{settings.transport}</strong></span>
                  <span>Protocol: <strong>{settings.protocol_version}</strong></span>
                  <span>SDK transport: <strong>{settings.sdk_transport_available ? "active" : "unavailable"}</strong></span>
                </div>
              ) : null}
            </section>

            <section className="settings-panel settings-install-panel">
              <header className="settings-panel-head">
                <h4>Setup Checklist</h4>
              </header>
              <ol className="settings-steps">
                <li>Start Workbench with the canonical command.</li>
                <li>Enable MCP, or regenerate the token if the full token is hidden.</li>
                <li>Copy the snippet for the target client.</li>
                <li>Restart or reconnect the MCP client.</li>
                <li>Verify the client can list tools and call <span className="mono">neext_workspace_summary</span>.</li>
              </ol>
              <pre className="settings-code settings-install-code">
                <code>{"make neext-workbench\n# endpoint: http://127.0.0.1:8765/mcp\n# auth header: Authorization: Bearer <token>"}</code>
              </pre>
            </section>

            {settings?.enabled ? (
              <section className="settings-panel settings-capabilities">
                <header className="settings-panel-head">
                  <h4>Capability Summary</h4>
                  <span className="muted">{availableCapabilities.length} tools available</span>
                </header>
                <div className="settings-mcp-meta">
                  <span>Write/run/custom tools: <strong>{writeCapabilityCount}</strong></span>
                  <span>Delete approval tools: <strong>{destructiveCapabilityCount}</strong></span>
                  <span>Scopes: <strong>{settings.scopes.length}</strong></span>
                </div>
              </section>
            ) : null}

            {settings?.enabled ? (
              <section className="settings-panel settings-configs">
                <header className="settings-panel-head">
                  <h4>Client Config Snippets</h4>
                  <span className="muted">Shown with the current token once</span>
                </header>
                {settings.stdio_readiness && settings.stdio_readiness.status === "blocked" ? (
                  <div className="settings-readiness is-blocked">
                    <p className="settings-note">
                      Claude Desktop (local stdio) cannot launch in this environment, so its snippet is hidden. The HTTP clients below still work.
                    </p>
                    {settings.stdio_readiness.issues.length ? (
                      <ul className="settings-readiness-list">
                        {settings.stdio_readiness.issues.map((issue) => (
                          <li key={issue}>{issue}</li>
                        ))}
                      </ul>
                    ) : null}
                    {settings.stdio_readiness.remediation.length ? (
                      <>
                        <p className="settings-note">To enable Claude Desktop:</p>
                        <ul className="settings-readiness-list">
                          {settings.stdio_readiness.remediation.map((fix) => (
                            <li key={fix}>{fix}</li>
                          ))}
                        </ul>
                      </>
                    ) : null}
                  </div>
                ) : null}
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

            {settings?.enabled ? (
              <section className="settings-panel settings-approvals">
                <header className="settings-panel-head">
                  <h4>Delete Approvals</h4>
                  <span className="muted">{pendingApprovals.length} pending</span>
                </header>
                {pendingApprovals.length ? (
                  <div className="settings-approval-list">
                    {pendingApprovals.map((approval) => (
                      <div className="settings-approval-row" key={approval.id}>
                        <div>
                          <strong>{approval.summary}</strong>
                          <span className="muted mono">{approval.id}</span>
                        </div>
                        <div className="settings-actions">
                          <button
                            type="button"
                            className="btn btn-danger"
                            disabled={approvalBusy}
                            onClick={() => approveMcpRequest.mutate(approval.id)}
                          >
                            Approve Delete
                          </button>
                          <button type="button" className="btn" disabled={approvalBusy} onClick={() => denyMcpRequest.mutate(approval.id)}>
                            Deny
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="artifact-table-empty settings-config-empty">
                    <EmptyState compact>No pending MCP delete approvals.</EmptyState>
                  </div>
                )}
              </section>
            ) : null}

            {settings?.enabled ? (
              <section className="settings-panel settings-activity">
                <header className="settings-panel-head">
                  <h4>Recent MCP Activity</h4>
                  <span className="muted">Visible in the Command Window too</span>
                </header>
                {recentActivity.length ? (
                  <div className="settings-activity-list">
                    {recentActivity.slice(0, 8).map((entry) => (
                      <div className="settings-activity-row" key={entry.id}>
                        <span className={`status-pill ${entry.status === "completed" ? "is-ready" : entry.status === "pending" ? "is-idle" : "is-failed"}`}>
                          {entry.status}
                        </span>
                        <span>{entry.message}</span>
                        <span className="muted mono">{entry.tool_name || entry.event_type}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="artifact-table-empty settings-config-empty">
                    <EmptyState compact>No MCP activity yet.</EmptyState>
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
