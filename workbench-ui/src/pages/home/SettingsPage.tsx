import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Power, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { api, type McpClientConfigSnippet, type McpSettingsResponse } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { useMcpSettings } from "../../hooks/useWorkspace";

type SettingsTab = "general" | "agentic";

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
          {(["general", "agentic"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              className={`tab-btn ${activeSettingsTab === tab ? "is-active" : ""}`}
              onClick={() => setActiveSettingsTab(tab)}
            >
              {tab === "general" ? "General" : "Agentic"}
            </button>
          ))}
        </div>

        {activeSettingsTab === "general" ? (
          <div className="settings-tab-panel settings-general-panel">
            <EmptyState compact>No general settings yet.</EmptyState>
          </div>
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
