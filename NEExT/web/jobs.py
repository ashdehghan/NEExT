"""Small local job manager for the NEExT web workbench."""

from __future__ import annotations

import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from NEExT.web.project import ProjectManager, utc_now


@dataclass
class JobRecord:
    id: str
    type: str
    name: str
    status: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    result_artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JobManager:
    def __init__(self, project: ProjectManager, max_workers: int = 2):
        self.project = project
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.records: dict[str, JobRecord] = {}
        self.futures: dict[str, Future] = {}

    def submit(
        self,
        job_type: str,
        name: str,
        work: Callable[[Callable[[float, str], None]], str],
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id,
            type=job_type,
            name=name,
            status="queued",
            created_at=utc_now(),
            updated_at=utc_now(),
            metadata=dict(metadata or {}),
        )
        self.records[job_id] = record
        self._persist(record)
        self.futures[job_id] = self.executor.submit(self._run, job_id, work)
        return record

    def get(self, job_id: str) -> JobRecord:
        if job_id in self.records:
            return self.records[job_id]
        for job in self.project.list_jobs():
            if job["id"] == job_id:
                record = JobRecord(**job)
                self.records[job_id] = record
                return record
        raise KeyError(f"Unknown job id: {job_id}")

    def _run(self, job_id: str, work: Callable[[Callable[[float, str], None]], str]) -> None:
        record = self.records[job_id]
        record.status = "running"
        record.updated_at = utc_now()
        self._persist(record)

        def update(progress: float, message: str) -> None:
            record.progress = max(0.0, min(1.0, progress))
            record.logs.append(message)
            record.updated_at = utc_now()
            self._persist(record)

        try:
            artifact_id = work(update)
            record.result_artifact_id = artifact_id
            record.progress = 1.0
            record.status = "succeeded"
            record.logs.append("Job completed")
        except Exception as exc:  # pragma: no cover - defensive logging path
            record.status = "failed"
            record.error = f"{exc}\n{traceback.format_exc()}"
            record.logs.append(f"Job failed: {exc}")
        finally:
            record.updated_at = utc_now()
            self._persist(record)

    def _persist(self, record: JobRecord) -> None:
        self.project.upsert_job(asdict(record))
