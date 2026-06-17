import { Terminal } from "lucide-react";
import type { JobManifest, McpActivityEntry } from "../../api";

interface CommandWindowProps {
  jobs: JobManifest[];
  mcpActivity?: McpActivityEntry[];
}

export function CommandWindow({ jobs, mcpActivity = [] }: CommandWindowProps) {
  const latestJob = jobs[0];
  const latestActivity = mcpActivity.slice(0, 4);

  return (
    <section className="cmd">
      <header className="cmd-header">
        <span>
          <Terminal aria-hidden /> Command Window
        </span>
      </header>
      <div className="cmd-body">
        {latestActivity.map((entry) => (
          <div className="cmd-line" key={entry.id}>
            [mcp:{entry.status}] {entry.message}
          </div>
        ))}
        {latestJob ? (
          latestJob.log.slice(-Math.max(1, 8 - latestActivity.length)).map((line, index) => (
            <div className="cmd-line" key={`${latestJob.id}-${index}`}>
              {line}
            </div>
          ))
        ) : latestActivity.length === 0 ? (
          <div className="cmd-line muted">No job output.</div>
        ) : null}
      </div>
    </section>
  );
}
