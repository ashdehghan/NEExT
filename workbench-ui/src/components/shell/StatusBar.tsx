import { Circle, Folder } from "lucide-react";

interface StatusBarProps {
  workspacePath?: string;
  projectName?: string;
  version?: string;
}

export function StatusBar({ workspacePath, projectName, version }: StatusBarProps) {
  return (
    <footer className="status-bar">
      <span className="status-item">
        <Folder aria-hidden />
        <span>{workspacePath || "Loading workspace"}</span>
      </span>
      {projectName && (
        <>
          <span className="status-sep">·</span>
          <span className="status-item">
            Project: <strong>{projectName}</strong>
          </span>
        </>
      )}
      <span className="status-spacer" />
      {version && (
        <>
          <span>Workbench {version}</span>
          <span className="status-sep">·</span>
        </>
      )}
      <span className="status-ready">
        <Circle aria-hidden />
        Ready
      </span>
    </footer>
  );
}
