/**
 * RunningState — a visible in-progress affordance for the Center Panel and analysis cards.
 *
 * Shows an indeterminate animated bar (activity, not a real percentage) plus a label and an
 * optional detail line (e.g. the latest job log line such as "computing embeddings"). Use this
 * wherever a job is running so the user can see work is happening instead of a static view.
 */

interface RunningStateProps {
  /** Headline, e.g. "Training model…". */
  label?: string;
  /** Secondary line, typically the latest job log message. */
  detail?: string;
  /** Tighter layout for inline/card use. */
  compact?: boolean;
}

export function RunningState({ label = "Running…", detail, compact }: RunningStateProps) {
  return (
    <div className={`running-state${compact ? " is-compact" : ""}`} role="status" aria-live="polite">
      <div className="running-bar" aria-hidden>
        <span />
      </div>
      <div className="running-text">
        <span className="running-label">{label}</span>
        {detail ? <span className="running-detail mono">{detail}</span> : null}
      </div>
    </div>
  );
}
