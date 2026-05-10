"""Local project and artifact management for the NEExT web workbench."""

from __future__ import annotations

import json
import pickle
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from NEExT import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArtifactRecord:
    id: str
    type: str
    name: str
    created_at: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)


class ProjectManager:
    """Manage a local NEExT web project folder.

    The manifest is intentionally transparent JSON. Python object pickles are
    used only as internal cache files for local web sessions.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.artifacts_dir = self.root / "artifacts"
        self.exports_dir = self.root / "exports"
        self.jobs_dir = self.root / "jobs"
        self.manifest_path = self.root / "manifest.json"
        self._object_cache: dict[str, Any] = {}

        for directory in [self.root, self.data_dir, self.artifacts_dir, self.exports_dir, self.jobs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            self._write_manifest(
                {
                    "schema_version": 1,
                    "neext_version": __version__,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "artifacts": [],
                    "jobs": [],
                }
            )

    def manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now()
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        artifacts = self.manifest().get("artifacts", [])
        if artifact_type:
            artifacts = [artifact for artifact in artifacts if artifact.get("type") == artifact_type]
        return artifacts

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        for artifact in self.list_artifacts():
            if artifact["id"] == artifact_id:
                return artifact
        raise KeyError(f"Unknown artifact id: {artifact_id}")

    def register_artifact(
        self,
        artifact_type: str,
        name: str,
        obj: Any | None = None,
        dataframe: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
        parent_ids: Iterable[str] | None = None,
    ) -> ArtifactRecord:
        artifact_id = uuid.uuid4().hex[:12]
        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=False)

        object_path = None
        if obj is not None:
            object_path = artifact_dir / "object.pkl"
            with object_path.open("wb") as handle:
                pickle.dump(obj, handle)
            self._object_cache[artifact_id] = obj

        table_path = None
        if dataframe is not None:
            table_path = artifact_dir / "table.csv"
            dataframe.to_csv(table_path, index=False)

        record_metadata = dict(metadata or {})
        if object_path is not None:
            record_metadata["object_path"] = str(object_path.relative_to(self.root))
        if table_path is not None:
            record_metadata["table_path"] = str(table_path.relative_to(self.root))

        record = ArtifactRecord(
            id=artifact_id,
            type=artifact_type,
            name=name,
            created_at=utc_now(),
            path=str(artifact_dir.relative_to(self.root)),
            metadata=record_metadata,
            parent_ids=list(parent_ids or []),
        )

        manifest = self.manifest()
        manifest.setdefault("artifacts", []).append(asdict(record))
        self._write_manifest(manifest)
        return record

    def load_object(self, artifact_id: str) -> Any:
        if artifact_id in self._object_cache:
            return self._object_cache[artifact_id]

        artifact = self.get_artifact(artifact_id)
        object_path = artifact.get("metadata", {}).get("object_path")
        if not object_path:
            raise ValueError(f"Artifact {artifact_id} does not contain a cached Python object")

        absolute_path = self.root / object_path
        with absolute_path.open("rb") as handle:
            obj = pickle.load(handle)
        self._object_cache[artifact_id] = obj
        return obj

    def load_table(self, artifact_id: str) -> pd.DataFrame:
        artifact = self.get_artifact(artifact_id)
        table_path = artifact.get("metadata", {}).get("table_path")
        if not table_path:
            raise ValueError(f"Artifact {artifact_id} does not contain a table")
        return pd.read_csv(self.root / table_path)

    def upsert_job(self, job: dict[str, Any]) -> None:
        manifest = self.manifest()
        jobs = manifest.setdefault("jobs", [])
        for index, existing in enumerate(jobs):
            if existing["id"] == job["id"]:
                jobs[index] = job
                break
        else:
            jobs.append(job)
        self._write_manifest(manifest)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.manifest().get("jobs", [])
