import type { JobManifest } from "../../api";
import { EmptyState } from "../primitives/EmptyState";

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
            {jobs.slice(0, 8).map((job) => (
              <div className="job-item" key={job.id}>
                <span className={`status-pill ${job.status === "completed" ? "is-ready" : "is-idle"}`}>{job.status}</span>
                <span className="job-op">{job.operation.operation_id}</span>
                <span className="muted mono">{job.updated_at}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
