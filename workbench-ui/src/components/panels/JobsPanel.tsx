import type { JobManifest } from "../../api";
import { EmptyState } from "../primitives/EmptyState";
import { StatusPill } from "../primitives/StatusPill";

interface JobsPanelProps {
  jobs: JobManifest[];
}

export function JobsPanel({ jobs }: JobsPanelProps) {
  return (
    <section className="panel jobs-panel">
      <div className="panel-header">
        <span>Jobs</span>
      </div>
      <div className="panel-body">
        {jobs.length === 0 ? (
          <EmptyState compact>No jobs.</EmptyState>
        ) : (
          <div className="job-list">
            {jobs.slice(0, 8).map((job) => {
              const active = job.status === "running" || job.status === "queued";
              const latestLog = job.log && job.log.length ? job.log[job.log.length - 1] : null;
              return (
                <div className={`job-item${active ? " is-active" : ""}`} key={job.id}>
                  <div className="job-item-head">
                    <StatusPill status={job.status} />
                    <span className="job-op">{job.operation.operation_id}</span>
                  </div>
                  {active ? (
                    <div className="job-progress" aria-hidden>
                      <span />
                    </div>
                  ) : null}
                  {active && latestLog ? (
                    <span className="job-log-line mono" title={latestLog}>
                      {latestLog}
                    </span>
                  ) : (
                    <span className="muted mono job-time">{job.updated_at}</span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
