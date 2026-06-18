/**
 * Shared status pill used across the Jobs Panel, Left Panel artifacts, and Center views.
 *
 * Maps a backend status (job: queued/running/completed/failed; artifact:
 * planned/running/completed/failed) to a consistent visual treatment. The `running`
 * state is animated (see `.status-pill.is-running` in styles.css).
 */

export type WorkbenchStatus =
  | "queued"
  | "planned"
  | "running"
  | "completed"
  | "failed"
  | "idle"
  | (string & {});

export function statusPillClass(status: string): string {
  switch (status) {
    case "completed":
      return "status-pill is-completed";
    case "running":
      return "status-pill is-running";
    case "failed":
      return "status-pill is-failed";
    case "queued":
    case "planned":
      return "status-pill is-queued";
    default:
      return "status-pill is-idle";
  }
}

interface StatusPillProps {
  status: WorkbenchStatus;
  /** Override the displayed text (defaults to the raw status). */
  label?: string;
  className?: string;
}

export function StatusPill({ status, label, className }: StatusPillProps) {
  return (
    <span className={[statusPillClass(status), className].filter(Boolean).join(" ")}>
      {label ?? status}
    </span>
  );
}
