import type { PointerEvent as ReactPointerEvent } from "react";
import { ChevronDown, ChevronUp, Terminal } from "lucide-react";
import type { JobManifest, McpActivityEntry } from "../../api";

interface CommandWindowProps {
  jobs: JobManifest[];
  mcpActivity?: McpActivityEntry[];
  collapsed: boolean;
  height: number;
  onToggleCollapsed: () => void;
  onHeightChange: (height: number) => void;
}

export function CommandWindow({ jobs, mcpActivity = [], collapsed, height, onToggleCollapsed, onHeightChange }: CommandWindowProps) {
  const latestJob = jobs[0];
  const latestActivity = mcpActivity.slice(0, 4);
  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (collapsed) return;
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = height;
    const handleMove = (moveEvent: PointerEvent) => {
      const nextHeight = Math.max(96, Math.min(420, startHeight + startY - moveEvent.clientY));
      onHeightChange(nextHeight);
    };
    const stop = () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", stop);
  };

  return (
    <section className={`cmd ${collapsed ? "is-collapsed" : ""}`}>
      <div className="cmd-resize-handle" onPointerDown={startResize} aria-hidden />
      <header className="cmd-header">
        <span>
          <Terminal aria-hidden /> Command Window
        </span>
        <button
          type="button"
          className="cmd-toggle"
          onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expand command window" : "Collapse command window"}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? <ChevronUp /> : <ChevronDown />}
        </button>
      </header>
      {!collapsed ? (
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
      ) : null}
    </section>
  );
}
