import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, ExternalLink, Power, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { api, type McpClientConfigSnippet, type McpSettingsResponse } from "../../api";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";
import { useMcpSettings } from "../../hooks/useWorkspace";

const CHATGPT_MCP_DOCS = "https://developers.openai.com/apps-sdk/deploy/connect-chatgpt";

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

  return (
    <div className="workflow settings-workflow">
      <section className="card settings-card">
        <header className="card-head">
          <span className="card-head-fc">
            <FcIcon name="settings" size={32} />
          </span>
          <div>
            <h3>Settings</h3>
            <p className="form-subtitle">Local MCP access for this Workbench workspace.</p>
          </div>
        </header>
        <div className="card-body settings-grid">
          <section className="settings-panel">
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

          <section className="settings-panel">
            <header className="settings-panel-head">
              <h4>ChatGPT</h4>
            </header>
            <p className="settings-note">
              This v1 server is local stdio only, so it is not a usable ChatGPT connector. ChatGPT Apps require an HTTPS MCP endpoint.
            </p>
            <a className="settings-link" href={CHATGPT_MCP_DOCS} target="_blank" rel="noreferrer">
              <ExternalLink /> Open ChatGPT connector docs
            </a>
          </section>
        </div>
      </section>

      {settings?.enabled && snippets.length ? (
        <section className="artifact-table settings-configs">
          <header className="artifact-table-head">
            <span className="artifact-table-title">Client Config Snippets</span>
            <span className="muted">Shown with the current token once</span>
          </header>
          <div className="settings-snippet-list">
            {snippets.map((snippet) => (
              <ConfigSnippet key={snippet.client} snippet={snippet} />
            ))}
          </div>
        </section>
      ) : settings?.enabled ? (
        <section className="artifact-table settings-configs">
          <header className="artifact-table-head">
            <span className="artifact-table-title">Client Config Snippets</span>
          </header>
          <div className="artifact-table-empty">
            <EmptyState compact>Regenerate the token to show copy-ready snippets.</EmptyState>
          </div>
        </section>
      ) : null}
    </div>
  );
}
