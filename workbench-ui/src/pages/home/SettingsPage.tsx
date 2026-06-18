import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Power, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { api, type McpClientConfigSnippet, type McpSettingsResponse, type ProjectManifest, type WorkspaceInfo } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { useDocs, useMcpActivity, useMcpApprovals, useMcpSettings } from "../../hooks/useWorkspace";

type SettingsTab = "general" | "agentic" | "docs";

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
  const docsQuery = useDocs();
  const topics = useMemo(() => docsQuery.data ?? [], [docsQuery.data]);
  const [activeTopicId, setActiveTopicId] = useState("");
  const activeTopic = topics.find((topic) => topic.id === activeTopicId) || topics[0];

  if (!topics.length) {
    return (
      <div className="settings-tab-panel settings-docs-panel">
        <EmptyState>{docsQuery.isLoading ? "Loading docs…" : "Docs unavailable"}</EmptyState>
      </div>
    );
  }

  return (
    <div className="settings-tab-panel settings-docs-panel">
      <nav className="settings-docs-sidebar" aria-label="Docs topics">
        {topics.map((topic) => (
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
