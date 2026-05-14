import { Terminal } from "lucide-react";
import type { JobManifest } from "../../api";

interface CommandWindowProps {
  jobs: JobManifest[];
}

export function CommandWindow({ jobs }: CommandWindowProps) {
  const latestJob = jobs[0];

  return (
    <section className="cmd">
      <header className="cmd-header">
        <span>
          <Terminal aria-hidden /> Command Window
        </span>
      </header>
      <div className="cmd-body">
        {latestJob ? (
          latestJob.log.slice(-8).map((line, index) => (
            <div className="cmd-line" key={`${latestJob.id}-${index}`}>
              {line}
            </div>
          ))
        ) : (
          <div className="cmd-line muted">No job output.</div>
        )}
      </div>
    </section>
  );
}
