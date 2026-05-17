"""Filesystem persistence for the local NEExT Workbench."""

from __future__ import annotations

import hashlib
import hmac
import json
import queue
import secrets
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pydantic import ValidationError

from .dataset_library import get_catalog_dataset, list_catalog_entries
from .embedding_library import get_embedding_catalog_item, list_embedding_catalog_entries
from .embedding_library import get_operation_definition as get_embedding_operation_definition
from .feature_library import get_feature_catalog_item, get_operation_definition, list_feature_catalog_entries
from .model_library import get_model_catalog_item, list_model_catalog_entries
from .model_library import get_operation_definition as get_model_operation_definition
from .schemas import (
    ArtifactError,
    ArtifactInputRef,
    DatasetAnalysis,
    DatasetCatalogEntry,
    DatasetCreateRequest,
    DatasetDataFiles,
    DatasetGraphSearchResponse,
    DatasetGraphSearchResult,
    DatasetGraphSummary,
    DatasetGraphVisual,
    DatasetManifest,
    DatasetMappingFiles,
    DatasetNodeDetail,
    DatasetStats,
    DatasetVisualEdge,
    DatasetVisualNode,
    EmbeddingAnalysis,
    EmbeddingAnalysisAlgorithm,
    EmbeddingAnalysisFeature,
    EmbeddingCatalogEntry,
    EmbeddingCreateParams,
    EmbeddingCreateRequest,
    EmbeddingExpectedOutput,
    EmbeddingGraphDetail,
    EmbeddingGraphSearchResponse,
    EmbeddingGraphSearchResult,
    EmbeddingManifest,
    EmbeddingOutputFiles,
    EmbeddingOutputStats,
    EmbeddingPcaPayload,
    EmbeddingPcaPoint,
    EmbeddingRunBatchRequest,
    FeatureAnalysis,
    FeatureAnalysisDataset,
    FeatureAnalysisMethod,
    FeatureCatalogEntry,
    FeatureColumnSummary,
    FeatureCoverage,
    FeatureCreateParams,
    FeatureCreateRequest,
    FeatureExpectedOutput,
    FeatureGraphDetail,
    FeatureGraphSearchResponse,
    FeatureGraphSearchResult,
    FeatureManifest,
    FeatureOutputFiles,
    FeatureOutputStats,
    FeaturePcaPayload,
    FeaturePcaPoint,
    FeatureRunBatchRequest,
    JobArtifactRef,
    JobEvent,
    JobManifest,
    McpSettingsManifest,
    ModelAnalysis,
    ModelAnalysisAlgorithm,
    ModelAnalysisEmbedding,
    ModelCatalogEntry,
    ModelCreateParams,
    ModelCreateRequest,
    ModelExpectedOutput,
    ModelManifest,
    ModelMetricPoint,
    ModelMetricSeries,
    ModelOutputFiles,
    ModelOutputStats,
    ModelPreview,
    ModelRunBatchRequest,
    OperationSpec,
    ProjectCreate,
    ProjectDeletionSummary,
    ProjectManifest,
    TabularPreview,
)

SCHEMA_VERSION = "1"
WORKSPACE_MANIFEST_TYPE = "workspace"
PROJECT_MANIFEST_TYPE = "project"
DATASET_MANIFEST_TYPE = "dataset"
FEATURE_MANIFEST_TYPE = "feature"
EMBEDDING_MANIFEST_TYPE = "embedding"
MODEL_MANIFEST_TYPE = "model"
JOB_MANIFEST_TYPE = "job"
MCP_SETTINGS_MANIFEST_TYPE = "mcp_settings"
ARTIFACT_KINDS = ("datasets", "features", "embeddings", "models")
DATASET_PREP_OPERATION_ID = "neext.prepare_graph_collection"
DATASET_PREP_OPERATION_VERSION = "1"
FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS = 5000
FEATURE_ANALYSIS_DEFAULT_MAX_POINTS = 5000
EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS = 5000
EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS = 5000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkbenchStore:
    """Small filesystem-backed store for Workbench projects."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.projects_path = self.workspace_path / "projects"
        self.trash_projects_path = self.workspace_path / "trash" / "projects"
        self.workspace_file = self.workspace_path / "workspace.json"
        self.mcp_settings_file = self.workspace_path / "mcp.json"
        self._job_queue: queue.Queue[tuple[str, str, Callable[[], None]]] = queue.Queue()
        self._job_lock = threading.Lock()
        self._job_worker = threading.Thread(target=self._job_worker_loop, daemon=True)
        self._job_worker.start()
        self.ensure_workspace()

    def ensure_workspace(self) -> None:
        self.projects_path.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        if self.workspace_file.exists():
            try:
                existing = self._read_json(self.workspace_file)
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}

        workspace_manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": WORKSPACE_MANIFEST_TYPE,
            "created_at": str(existing.get("created_at") or now),
            "updated_at": str(existing.get("updated_at") or now),
        }
        if existing != workspace_manifest:
            self._write_json(self.workspace_file, workspace_manifest)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _new_project_id(self) -> str:
        while True:
            project_id = str(uuid.uuid4())
            if not (self.projects_path / project_id).exists():
                return project_id

    def _new_dataset_id(self, project_id: str) -> str:
        datasets_path = self.project_path(project_id) / "artifacts" / "datasets"
        while True:
            dataset_id = str(uuid.uuid4())
            if not (datasets_path / dataset_id).exists():
                return dataset_id

    def _new_feature_id(self, project_id: str) -> str:
        features_path = self.project_path(project_id) / "artifacts" / "features"
        while True:
            feature_id = str(uuid.uuid4())
            if not (features_path / feature_id).exists():
                return feature_id

    def _new_embedding_id(self, project_id: str) -> str:
        embeddings_path = self.project_path(project_id) / "artifacts" / "embeddings"
        while True:
            embedding_id = str(uuid.uuid4())
            if not (embeddings_path / embedding_id).exists():
                return embedding_id

    def _new_model_id(self, project_id: str) -> str:
        models_path = self.project_path(project_id) / "artifacts" / "models"
        while True:
            model_id = str(uuid.uuid4())
            if not (models_path / model_id).exists():
                return model_id

    def _new_job_id(self, project_id: str) -> str:
        jobs_path = self.project_path(project_id) / "jobs"
        while True:
            job_id = str(uuid.uuid4())
            if not (jobs_path / job_id).exists():
                return job_id

    def _trash_folder_name(self, project_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"{project_id}-{timestamp}"

    def _normalize_project_id(self, project_id: str) -> str:
        return str(uuid.UUID(project_id))

    def _normalize_artifact_id(self, artifact_id: str) -> str:
        parsed = uuid.UUID(artifact_id)
        if parsed.version != 4:
            raise ValueError("Artifact ID must be a UUIDv4 value")
        return str(parsed)

    def project_path(self, project_id: str) -> Path:
        root = self.projects_path.resolve()
        path = (self.projects_path / self._normalize_project_id(project_id)).resolve()
        if root not in path.parents and path != root:
            raise ValueError("Project path escaped the workspace")
        return path

    def dataset_path(self, project_id: str, dataset_id: str) -> Path:
        datasets_root = (self.project_path(project_id) / "artifacts" / "datasets").resolve()
        path = (datasets_root / self._normalize_artifact_id(dataset_id)).resolve()
        if datasets_root not in path.parents and path != datasets_root:
            raise ValueError("Dataset path escaped the project")
        return path

    def feature_path(self, project_id: str, feature_id: str) -> Path:
        features_root = (self.project_path(project_id) / "artifacts" / "features").resolve()
        path = (features_root / self._normalize_artifact_id(feature_id)).resolve()
        if features_root not in path.parents and path != features_root:
            raise ValueError("Feature path escaped the project")
        return path

    def embedding_path(self, project_id: str, embedding_id: str) -> Path:
        embeddings_root = (self.project_path(project_id) / "artifacts" / "embeddings").resolve()
        path = (embeddings_root / self._normalize_artifact_id(embedding_id)).resolve()
        if embeddings_root not in path.parents and path != embeddings_root:
            raise ValueError("Embedding path escaped the project")
        return path

    def model_path(self, project_id: str, model_id: str) -> Path:
        models_root = (self.project_path(project_id) / "artifacts" / "models").resolve()
        path = (models_root / self._normalize_artifact_id(model_id)).resolve()
        if models_root not in path.parents and path != models_root:
            raise ValueError("Model path escaped the project")
        return path

    def job_path(self, project_id: str, job_id: str) -> Path:
        jobs_root = (self.project_path(project_id) / "jobs").resolve()
        path = (jobs_root / self._normalize_artifact_id(job_id)).resolve()
        if jobs_root not in path.parents and path != jobs_root:
            raise ValueError("Job path escaped the project")
        return path

    def _create_project_artifact_dirs(self, project_path: Path) -> None:
        for artifact_kind in ARTIFACT_KINDS:
            (project_path / "artifacts" / artifact_kind).mkdir(parents=True, exist_ok=True)

    def _read_project_manifest(self, project_file: Path) -> ProjectManifest:
        payload = self._read_json(project_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != PROJECT_MANIFEST_TYPE:
            raise ValueError("Unsupported project manifest")
        project = ProjectManifest.model_validate(payload)
        if project.id != project_file.parent.name:
            raise ValueError("Project manifest ID does not match its folder")
        self._normalize_project_id(project.id)
        return project

    def _read_dataset_manifest(self, artifact_file: Path, project_id: str) -> DatasetManifest:
        payload = self._read_json(artifact_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != DATASET_MANIFEST_TYPE:
            raise ValueError("Unsupported dataset manifest")
        dataset = DatasetManifest.model_validate(payload)
        if dataset.id != artifact_file.parent.name:
            raise ValueError("Dataset manifest ID does not match its folder")
        if dataset.project_id != project_id:
            raise ValueError("Dataset manifest project ID does not match its project")
        self._normalize_artifact_id(dataset.id)
        self._validate_dataset_data_files(dataset)
        return dataset

    def _read_feature_manifest(self, artifact_file: Path, project_id: str) -> FeatureManifest:
        payload = self._read_json(artifact_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != FEATURE_MANIFEST_TYPE:
            raise ValueError("Unsupported feature manifest")
        feature = FeatureManifest.model_validate(payload)
        if feature.id != artifact_file.parent.name:
            raise ValueError("Feature manifest ID does not match its folder")
        if feature.project_id != project_id:
            raise ValueError("Feature manifest project ID does not match its project")
        self._normalize_artifact_id(feature.id)
        self._validate_feature_manifest(feature)
        return feature

    def _read_embedding_manifest(self, artifact_file: Path, project_id: str) -> EmbeddingManifest:
        payload = self._read_json(artifact_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != EMBEDDING_MANIFEST_TYPE:
            raise ValueError("Unsupported embedding manifest")
        embedding = EmbeddingManifest.model_validate(payload)
        if embedding.id != artifact_file.parent.name:
            raise ValueError("Embedding manifest ID does not match its folder")
        if embedding.project_id != project_id:
            raise ValueError("Embedding manifest project ID does not match its project")
        self._normalize_artifact_id(embedding.id)
        self._validate_embedding_manifest(embedding)
        return embedding

    def _read_model_manifest(self, artifact_file: Path, project_id: str) -> ModelManifest:
        payload = self._read_json(artifact_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != MODEL_MANIFEST_TYPE:
            raise ValueError("Unsupported model manifest")
        model = ModelManifest.model_validate(payload)
        if model.id != artifact_file.parent.name:
            raise ValueError("Model manifest ID does not match its folder")
        if model.project_id != project_id:
            raise ValueError("Model manifest project ID does not match its project")
        self._normalize_artifact_id(model.id)
        self._validate_model_manifest(model)
        return model

    def _read_job_manifest(self, job_file: Path, project_id: str) -> JobManifest:
        payload = self._read_json(job_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != JOB_MANIFEST_TYPE:
            raise ValueError("Unsupported job manifest")
        job = JobManifest.model_validate(payload)
        if job.id != job_file.parent.name:
            raise ValueError("Job manifest ID does not match its folder")
        if job.project_id != project_id:
            raise ValueError("Job manifest project ID does not match its project")
        self._normalize_artifact_id(job.id)
        return job

    def create_project(self, request: ProjectCreate) -> ProjectManifest:
        now = utc_now()
        project = ProjectManifest(
            id=self._new_project_id(),
            name=request.name.strip(),
            description=request.description.strip(),
            created_at=now,
            updated_at=now,
        )
        project_path = self.project_path(project.id)
        project_path.mkdir(parents=True, exist_ok=False)
        self._create_project_artifact_dirs(project_path)
        self._write_json(project_path / "project.json", project.model_dump())
        return project

    def list_projects(self) -> list[ProjectManifest]:
        projects = []
        for project_file in sorted(self.projects_path.glob("*/project.json")):
            try:
                projects.append(self._read_project_manifest(project_file))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def read_project(self, project_id: str) -> ProjectManifest:
        try:
            path = self.project_path(project_id) / "project.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Project not found: {project_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        try:
            return self._read_project_manifest(path)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Project not found: {project_id}") from exc

    def read_workspace_meta(self) -> dict[str, Any]:
        return self._read_json(self.workspace_file)

    def _default_mcp_settings(self) -> McpSettingsManifest:
        now = utc_now()
        return McpSettingsManifest(
            enabled=False,
            token_hash=None,
            token_preview=None,
            created_at=now,
            updated_at=now,
        )

    def _read_mcp_settings_manifest(self) -> McpSettingsManifest:
        if not self.mcp_settings_file.exists():
            return self._default_mcp_settings()
        payload = self._read_json(self.mcp_settings_file)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != MCP_SETTINGS_MANIFEST_TYPE:
            raise ValueError("Unsupported MCP settings manifest")
        return McpSettingsManifest.model_validate(payload)

    def read_mcp_settings(self) -> McpSettingsManifest:
        return self._read_mcp_settings_manifest()

    def _write_mcp_settings(self, settings: McpSettingsManifest) -> None:
        self._write_json(self.mcp_settings_file, settings.model_dump())

    def _new_mcp_token(self) -> str:
        return f"nxt_mcp_{secrets.token_urlsafe(32)}"

    def _mcp_token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _mcp_token_preview(self, token: str) -> str:
        return f"{token[:12]}...{token[-6:]}"

    def enable_mcp(self) -> tuple[McpSettingsManifest, str]:
        existing = self._read_mcp_settings_manifest()
        token = self._new_mcp_token()
        now = utc_now()
        settings = McpSettingsManifest(
            enabled=True,
            token_hash=self._mcp_token_hash(token),
            token_preview=self._mcp_token_preview(token),
            created_at=existing.created_at,
            updated_at=now,
        )
        self._write_mcp_settings(settings)
        return settings, token

    def regenerate_mcp_token(self) -> tuple[McpSettingsManifest, str]:
        existing = self._read_mcp_settings_manifest()
        if not existing.enabled:
            raise ValueError("MCP is disabled")
        token = self._new_mcp_token()
        settings = McpSettingsManifest(
            enabled=True,
            token_hash=self._mcp_token_hash(token),
            token_preview=self._mcp_token_preview(token),
            created_at=existing.created_at,
            updated_at=utc_now(),
        )
        self._write_mcp_settings(settings)
        return settings, token

    def disable_mcp(self) -> McpSettingsManifest:
        existing = self._read_mcp_settings_manifest()
        settings = McpSettingsManifest(
            enabled=False,
            token_hash=None,
            token_preview=None,
            created_at=existing.created_at,
            updated_at=utc_now(),
        )
        self._write_mcp_settings(settings)
        return settings

    def verify_mcp_token(self, token: str) -> bool:
        if not token:
            return False
        try:
            settings = self._read_mcp_settings_manifest()
        except (ValueError, json.JSONDecodeError, ValidationError):
            return False
        if not settings.enabled or not settings.token_hash:
            return False
        return hmac.compare_digest(self._mcp_token_hash(token), settings.token_hash)

    def list_dataset_catalog(self) -> list[DatasetCatalogEntry]:
        return list_catalog_entries()

    def list_feature_catalog(self) -> list[FeatureCatalogEntry]:
        return list_feature_catalog_entries()

    def list_embedding_catalog(self) -> list[EmbeddingCatalogEntry]:
        return list_embedding_catalog_entries()

    def list_model_catalog(self) -> list[ModelCatalogEntry]:
        return list_model_catalog_entries()

    def list_datasets(self, project_id: str) -> list[DatasetManifest]:
        project = self.read_project(project_id)
        datasets_path = self.project_path(project.id) / "artifacts" / "datasets"
        datasets = []
        for artifact_file in sorted(datasets_path.glob("*/artifact.json")):
            try:
                datasets.append(self._read_dataset_manifest(artifact_file, project.id))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(datasets, key=lambda item: item.updated_at, reverse=True)

    def read_dataset(self, project_id: str, dataset_id: str) -> DatasetManifest:
        project = self.read_project(project_id)
        try:
            path = self.dataset_path(project.id, dataset_id) / "artifact.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Dataset not found: {dataset_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")
        try:
            return self._read_dataset_manifest(path, project.id)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Dataset not found: {dataset_id}") from exc

    def list_features(self, project_id: str) -> list[FeatureManifest]:
        project = self.read_project(project_id)
        features_path = self.project_path(project.id) / "artifacts" / "features"
        features = []
        for artifact_file in sorted(features_path.glob("*/artifact.json")):
            try:
                features.append(self._read_feature_manifest(artifact_file, project.id))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(features, key=lambda item: item.updated_at, reverse=True)

    def read_feature(self, project_id: str, feature_id: str) -> FeatureManifest:
        project = self.read_project(project_id)
        try:
            path = self.feature_path(project.id, feature_id) / "artifact.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Feature not found: {feature_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Feature not found: {feature_id}")
        try:
            return self._read_feature_manifest(path, project.id)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Feature not found: {feature_id}") from exc

    def list_embeddings(self, project_id: str) -> list[EmbeddingManifest]:
        project = self.read_project(project_id)
        embeddings_path = self.project_path(project.id) / "artifacts" / "embeddings"
        embeddings = []
        for artifact_file in sorted(embeddings_path.glob("*/artifact.json")):
            try:
                embeddings.append(self._read_embedding_manifest(artifact_file, project.id))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(embeddings, key=lambda item: item.updated_at, reverse=True)

    def read_embedding(self, project_id: str, embedding_id: str) -> EmbeddingManifest:
        project = self.read_project(project_id)
        try:
            path = self.embedding_path(project.id, embedding_id) / "artifact.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Embedding not found: {embedding_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Embedding not found: {embedding_id}")
        try:
            return self._read_embedding_manifest(path, project.id)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Embedding not found: {embedding_id}") from exc

    def list_models(self, project_id: str) -> list[ModelManifest]:
        project = self.read_project(project_id)
        models_path = self.project_path(project.id) / "artifacts" / "models"
        models = []
        for artifact_file in sorted(models_path.glob("*/artifact.json")):
            try:
                models.append(self._read_model_manifest(artifact_file, project.id))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(models, key=lambda item: item.updated_at, reverse=True)

    def read_model(self, project_id: str, model_id: str) -> ModelManifest:
        project = self.read_project(project_id)
        try:
            path = self.model_path(project.id, model_id) / "artifact.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Model not found: {model_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        try:
            return self._read_model_manifest(path, project.id)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Model not found: {model_id}") from exc

    def list_jobs(self, project_id: str) -> list[JobManifest]:
        project = self.read_project(project_id)
        jobs_path = self.project_path(project.id) / "jobs"
        jobs = []
        for job_file in sorted(jobs_path.glob("*/job.json")):
            try:
                jobs.append(self._read_job_manifest(job_file, project.id))
            except (ValueError, json.JSONDecodeError, ValidationError):
                continue
        return sorted(jobs, key=lambda item: item.updated_at, reverse=True)

    def read_job(self, project_id: str, job_id: str) -> JobManifest:
        project = self.read_project(project_id)
        try:
            path = self.job_path(project.id, job_id) / "job.json"
        except ValueError as exc:
            raise FileNotFoundError(f"Job not found: {job_id}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        try:
            return self._read_job_manifest(path, project.id)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FileNotFoundError(f"Job not found: {job_id}") from exc

    def create_feature(self, project_id: str, request: FeatureCreateRequest) -> FeatureManifest:
        project = self.read_project(project_id)
        dataset = self.read_dataset(project.id, request.source_dataset_id)
        catalog = get_feature_catalog_item(request.source_feature_id)
        if catalog is None:
            raise FileNotFoundError(f"Feature catalog entry not found: {request.source_feature_id}")

        operation = get_operation_definition(catalog.operation_id, catalog.operation_version)
        if operation is None:
            raise ValueError(f"Unsupported feature operation: {catalog.operation_id}@{catalog.operation_version}")

        feature_id = self._new_feature_id(project.id)
        artifact_path = self.feature_path(project.id, feature_id)
        artifact_path.mkdir(parents=True, exist_ok=False)

        try:
            params = request.params.model_dump()
            operation_params = {
                "feature_list": [catalog.id],
                "feature_vector_length": params["feature_vector_length"],
                "normalize_features": params["normalize_features"],
                "n_jobs": params["n_jobs"],
                "parallel_backend": params["parallel_backend"],
            }
            columns = [f"{catalog.id}_{index}" for index in range(params["feature_vector_length"])]
            now = utc_now()
            manifest = FeatureManifest(
                id=feature_id,
                project_id=project.id,
                name=f"{dataset.name} - {catalog.name}",
                description="",
                status="planned",
                created_at=now,
                updated_at=now,
                inputs=[
                    ArtifactInputRef(
                        role="source_dataset",
                        artifact_kind="dataset",
                        artifact_id=dataset.id,
                    )
                ],
                source_feature_id=catalog.id,
                operation=OperationSpec(
                    operation_id=operation.operation_id,
                    operation_version=operation.operation_version,
                    params=operation_params,
                ),
                expected_output=FeatureExpectedOutput(columns=columns),
            )
            self._validate_feature_manifest(manifest)
            self._write_json(artifact_path / "artifact.json", manifest.model_dump())
            return manifest
        except Exception:
            shutil.rmtree(artifact_path, ignore_errors=True)
            raise

    def create_embedding(self, project_id: str, request: EmbeddingCreateRequest) -> EmbeddingManifest:
        project = self.read_project(project_id)
        catalog = get_embedding_catalog_item(request.source_embedding_id)
        if catalog is None:
            raise FileNotFoundError(f"Embedding catalog entry not found: {request.source_embedding_id}")

        operation = get_embedding_operation_definition(catalog.operation_id, catalog.operation_version)
        if operation is None:
            raise ValueError(f"Unsupported embedding operation: {catalog.operation_id}@{catalog.operation_version}")

        feature_ids = [self._normalize_artifact_id(feature_id) for feature_id in request.source_feature_ids]
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Embedding source features must be unique")
        features = [self.read_feature(project.id, feature_id) for feature_id in feature_ids]
        dataset_ids = {self._feature_dataset_id(feature) for feature in features}
        if len(dataset_ids) != 1:
            raise ValueError("Selected features must reference the same dataset")
        dataset = self.read_dataset(project.id, next(iter(dataset_ids)))

        embedding_id = self._new_embedding_id(project.id)
        artifact_path = self.embedding_path(project.id, embedding_id)
        artifact_path.mkdir(parents=True, exist_ok=False)

        try:
            params = request.params.model_dump()
            operation_params = {
                "embedding_algorithm": catalog.id,
                "embedding_dimension": params["embedding_dimension"],
                "random_state": 42,
                "memory_size": "4G",
                "feature_ids": feature_ids,
                "feature_columns": "all",
            }
            now = utc_now()
            manifest = EmbeddingManifest(
                id=embedding_id,
                project_id=project.id,
                name=f"{dataset.name} - {catalog.name} Embedding",
                description="",
                status="planned",
                created_at=now,
                updated_at=now,
                inputs=[
                    ArtifactInputRef(
                        role="source_feature",
                        artifact_kind="feature",
                        artifact_id=feature.id,
                    )
                    for feature in features
                ],
                source_embedding_id=catalog.id,
                operation=OperationSpec(
                    operation_id=operation.operation_id,
                    operation_version=operation.operation_version,
                    params=operation_params,
                ),
                expected_output=EmbeddingExpectedOutput(columns=[f"emb_{index}" for index in range(params["embedding_dimension"])]),
            )
            self._validate_embedding_manifest(manifest)
            self._write_json(artifact_path / "artifact.json", manifest.model_dump())
            return manifest
        except Exception:
            shutil.rmtree(artifact_path, ignore_errors=True)
            raise

    def create_model(self, project_id: str, request: ModelCreateRequest) -> ModelManifest:
        project = self.read_project(project_id)
        catalog = get_model_catalog_item(request.source_model_id)
        if catalog is None:
            raise FileNotFoundError(f"Model catalog entry not found: {request.source_model_id}")

        operation = get_model_operation_definition(catalog.operation_id, catalog.operation_version)
        if operation is None:
            raise ValueError(f"Unsupported model operation: {catalog.operation_id}@{catalog.operation_version}")

        embedding_ids = [self._normalize_artifact_id(embedding_id) for embedding_id in request.source_embedding_ids]
        if len(set(embedding_ids)) != len(embedding_ids):
            raise ValueError("Model source embeddings must be unique")
        embeddings = [self.read_embedding(project.id, embedding_id) for embedding_id in embedding_ids]
        dataset_ids = {self._embedding_dataset_id(project.id, embedding) for embedding in embeddings}
        if len(dataset_ids) != 1:
            raise ValueError("Selected embeddings must reference the same dataset")
        dataset = self.read_dataset(project.id, next(iter(dataset_ids)))

        model_id = self._new_model_id(project.id)
        artifact_path = self.model_path(project.id, model_id)
        artifact_path.mkdir(parents=True, exist_ok=False)

        try:
            params = request.params.model_dump()
            operation_params = {
                "model_algorithm": catalog.id,
                "task_type": params["task_type"],
                "sample_size": params["sample_size"],
                "test_size": params["test_size"],
                "balance_dataset": params["balance_dataset"],
                "random_state": 42,
                "n_jobs": params["n_jobs"],
                "parallel_backend": params["parallel_backend"],
                "embedding_ids": embedding_ids,
                "embedding_columns": "all",
            }
            metric_names = self._model_metric_names(params["task_type"])
            now = utc_now()
            manifest = ModelManifest(
                id=model_id,
                project_id=project.id,
                name=f"{dataset.name} - {catalog.name} {params['task_type'].title()}",
                description="",
                status="planned",
                created_at=now,
                updated_at=now,
                inputs=[
                    ArtifactInputRef(
                        role="source_embedding",
                        artifact_kind="embedding",
                        artifact_id=embedding.id,
                    )
                    for embedding in embeddings
                ],
                source_model_id=catalog.id,
                operation=OperationSpec(
                    operation_id=operation.operation_id,
                    operation_version=operation.operation_version,
                    params=operation_params,
                ),
                expected_output=ModelExpectedOutput(metrics=metric_names),
            )
            self._validate_model_manifest(manifest)
            self._write_json(artifact_path / "artifact.json", manifest.model_dump())
            return manifest
        except Exception:
            shutil.rmtree(artifact_path, ignore_errors=True)
            raise

    def create_dataset_from_library(self, project_id: str, request: DatasetCreateRequest) -> DatasetManifest:
        project = self.read_project(project_id)
        catalog = get_catalog_dataset(request.catalog_id)
        if catalog is None:
            raise FileNotFoundError(f"Dataset catalog entry not found: {request.catalog_id}")

        dataset_id = self._new_dataset_id(project.id)
        artifact_path = self.dataset_path(project.id, dataset_id)
        artifact_path.mkdir(parents=True, exist_ok=False)

        try:
            now = utc_now()
            manifest = DatasetManifest(
                id=dataset_id,
                project_id=project.id,
                name=catalog.name,
                description=catalog.description,
                status="planned",
                created_at=now,
                updated_at=now,
                source_catalog_id=catalog.id,
                source_name=catalog.name,
                source=catalog.source,
                source_domain=catalog.domain,
                operation=OperationSpec(
                    operation_id=DATASET_PREP_OPERATION_ID,
                    operation_version=DATASET_PREP_OPERATION_VERSION,
                    params={
                        "graph_type": request.params.graph_type,
                        "reindex_nodes": True,
                        "filter_largest_component": request.params.filter_largest_component,
                    },
                ),
                source_stats=DatasetStats(
                    graph_count=catalog.graph_count,
                    node_count=catalog.node_count,
                    edge_count=catalog.edge_count,
                    has_graph_labels="graph_labels" in catalog.files,
                    has_node_features="node_features" in catalog.files,
                    has_edge_features="edge_features" in catalog.files,
                ),
            )
            self._write_json(artifact_path / "artifact.json", manifest.model_dump())
            return manifest
        except Exception:
            shutil.rmtree(artifact_path, ignore_errors=True)
            raise

    def run_dataset_preparation(self, project_id: str, dataset_id: str) -> JobManifest:
        project = self.read_project(project_id)
        dataset = self.read_dataset(project.id, dataset_id)
        if dataset.status not in {"planned", "failed"}:
            raise ValueError("Only planned or failed datasets can be run")
        job = self._create_job(
            project.id,
            operation=dataset.operation,
            target_artifacts=[JobArtifactRef(artifact_kind="dataset", artifact_id=dataset.id)],
        )
        self._enqueue_job(project.id, job.id, lambda: self._prepare_dataset_artifact(project.id, dataset.id, job.id))
        return job

    def run_feature(self, project_id: str, feature_id: str) -> JobManifest:
        return self.run_feature_batch(project_id, FeatureRunBatchRequest(feature_ids=[feature_id]))

    def run_feature_batch(self, project_id: str, request: FeatureRunBatchRequest) -> JobManifest:
        project = self.read_project(project_id)
        feature_ids = [self._normalize_artifact_id(feature_id) for feature_id in request.feature_ids]
        features = [self.read_feature(project.id, feature_id) for feature_id in feature_ids]
        if any(feature.status not in {"planned", "failed"} for feature in features):
            raise ValueError("Only planned or failed features can be run")

        dataset_ids = {self._feature_dataset_id(feature) for feature in features}
        if len(dataset_ids) != 1:
            raise ValueError("Selected features must reference the same dataset")

        operation_params = {"feature_ids": feature_ids, "source_dataset_id": next(iter(dataset_ids))}
        job = self._create_job(
            project.id,
            operation=OperationSpec(
                operation_id="neext.compute_node_features",
                operation_version="1",
                params=operation_params,
            ),
            target_artifacts=[JobArtifactRef(artifact_kind="feature", artifact_id=feature.id) for feature in features],
        )
        self._enqueue_job(project.id, job.id, lambda: self._compute_feature_artifacts(project.id, feature_ids, job.id))
        return job

    def run_embedding(self, project_id: str, embedding_id: str) -> JobManifest:
        return self.run_embedding_batch(project_id, EmbeddingRunBatchRequest(embedding_ids=[embedding_id]))

    def run_embedding_batch(self, project_id: str, request: EmbeddingRunBatchRequest) -> JobManifest:
        project = self.read_project(project_id)
        embedding_ids = [self._normalize_artifact_id(embedding_id) for embedding_id in request.embedding_ids]
        if len(set(embedding_ids)) != len(embedding_ids):
            raise ValueError("Selected embeddings must be unique")
        embeddings = [self.read_embedding(project.id, embedding_id) for embedding_id in embedding_ids]
        if any(embedding.status not in {"planned", "failed"} for embedding in embeddings):
            raise ValueError("Only planned or failed embeddings can be run")

        job = self._create_job(
            project.id,
            operation=OperationSpec(
                operation_id="neext.compute_graph_embeddings",
                operation_version="1",
                params={"embedding_ids": embedding_ids},
            ),
            target_artifacts=[JobArtifactRef(artifact_kind="embedding", artifact_id=embedding.id) for embedding in embeddings],
        )
        self._enqueue_job(project.id, job.id, lambda: self._compute_embedding_artifacts(project.id, embedding_ids, job.id))
        return job

    def run_model(self, project_id: str, model_id: str) -> JobManifest:
        return self.run_model_batch(project_id, ModelRunBatchRequest(model_ids=[model_id]))

    def run_model_batch(self, project_id: str, request: ModelRunBatchRequest) -> JobManifest:
        project = self.read_project(project_id)
        model_ids = [self._normalize_artifact_id(model_id) for model_id in request.model_ids]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("Selected models must be unique")
        models = [self.read_model(project.id, model_id) for model_id in model_ids]
        if any(model.status not in {"planned", "failed"} for model in models):
            raise ValueError("Only planned or failed models can be run")

        job = self._create_job(
            project.id,
            operation=OperationSpec(
                operation_id="neext.train_graph_model",
                operation_version="1",
                params={"model_ids": model_ids},
            ),
            target_artifacts=[JobArtifactRef(artifact_kind="model", artifact_id=model.id) for model in models],
        )
        self._enqueue_job(project.id, job.id, lambda: self._compute_model_artifacts(project.id, model_ids, job.id))
        return job

    def _create_job(self, project_id: str, operation: OperationSpec, target_artifacts: list[JobArtifactRef]) -> JobManifest:
        job_id = self._new_job_id(project_id)
        job_path = self.job_path(project_id, job_id)
        job_path.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        job = JobManifest(
            id=job_id,
            project_id=project_id,
            status="queued",
            operation=operation,
            target_artifacts=target_artifacts,
            created_at=now,
            updated_at=now,
            events=[JobEvent(timestamp=now, message="Job queued")],
            log=["Job queued"],
        )
        self._write_json(job_path / "job.json", job.model_dump())
        return job

    def _enqueue_job(self, project_id: str, job_id: str, callback: Callable[[], None]) -> None:
        self._job_queue.put((project_id, job_id, callback))

    def _job_worker_loop(self) -> None:
        while True:
            project_id, job_id, callback = self._job_queue.get()
            try:
                self._mark_job_running(project_id, job_id)
                callback()
                self._mark_job_completed(project_id, job_id)
            except Exception as exc:
                self._mark_job_failed(project_id, job_id, str(exc))
            finally:
                self._job_queue.task_done()

    def _write_job_manifest(self, job: JobManifest) -> None:
        self._write_json(self.job_path(job.project_id, job.id) / "job.json", job.model_dump())

    def _update_job(self, project_id: str, job_id: str, *, status: str | None = None, message: str | None = None, error: str | None = None) -> None:
        with self._job_lock:
            job = self.read_job(project_id, job_id)
            now = utc_now()
            if status is not None:
                job.status = status
                if status == "running" and job.started_at is None:
                    job.started_at = now
                if status in {"completed", "failed"}:
                    job.completed_at = now
            if message is not None:
                level = "error" if status == "failed" else "info"
                job.events.append(JobEvent(timestamp=now, level=level, message=message))
                job.log.append(message)
            if error is not None:
                job.error = error
            job.updated_at = now
            self._write_job_manifest(job)

    def _mark_job_running(self, project_id: str, job_id: str) -> None:
        self._update_job(project_id, job_id, status="running", message="Job started")

    def _mark_job_completed(self, project_id: str, job_id: str) -> None:
        self._update_job(project_id, job_id, status="completed", message="Job completed")

    def _mark_job_failed(self, project_id: str, job_id: str, error: str) -> None:
        self._update_job(project_id, job_id, status="failed", message=f"Job failed: {error}", error=error)

    def _log_job(self, project_id: str, job_id: str, message: str) -> None:
        self._update_job(project_id, job_id, message=message)

    def _prepare_dataset_artifact(self, project_id: str, dataset_id: str, job_id: str) -> DatasetManifest:
        dataset = self.read_dataset(project_id, dataset_id)
        artifact_path = self.dataset_path(project_id, dataset.id)
        tmp_path = artifact_path / "_tmp" / job_id
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=False)
        self._set_dataset_status(project_id, dataset.id, "running", error=None)

        try:
            import pyarrow  # noqa: F401

            catalog = get_catalog_dataset(dataset.source_catalog_id)
            if catalog is None:
                raise FileNotFoundError(f"Dataset catalog entry not found: {dataset.source_catalog_id}")

            self._log_job(project_id, job_id, f"Preparing dataset {dataset.name}")
            frames = self._read_catalog_frames(catalog.files)
            nodes_df, edges_df = self._normalized_nodes_and_edges(frames["node_graph_mapping"], frames["edges"])
            graph_labels_df = frames.get("graph_labels")
            node_features_df = frames.get("node_features")
            edge_features_df = frames.get("edge_features")

            if graph_labels_df is not None:
                self._validate_graph_labels(graph_labels_df, nodes_df)
            if node_features_df is not None:
                self._validate_node_features(node_features_df, nodes_df)
            if edge_features_df is not None:
                self._validate_edge_features(edge_features_df, nodes_df)

            raw_files = self._write_raw_dataset_snapshot(
                tmp_path / "raw",
                nodes_df,
                edges_df,
                graph_labels_df,
                node_features_df,
                edge_features_df,
            )

            collection = self._build_graph_collection_for_dataset(
                nodes_df=nodes_df,
                edges_df=edges_df,
                graph_labels_df=graph_labels_df,
                node_features_df=node_features_df,
                edge_features_df=edge_features_df,
                graph_type=str(dataset.operation.params["graph_type"]),
                filter_largest_component=bool(dataset.operation.params["filter_largest_component"]),
            )
            prepared_files, mapping_files, prepared_stats = self._write_prepared_dataset_outputs(tmp_path, collection)

            self._promote_directory(tmp_path / "raw", artifact_path / "raw")
            self._promote_directory(tmp_path / "prepared", artifact_path / "prepared")
            self._promote_directory(tmp_path / "mappings", artifact_path / "mappings")

            source_stats = DatasetStats(
                graph_count=int(nodes_df["graph_id"].nunique(dropna=False)),
                node_count=int(len(nodes_df)),
                edge_count=int(len(edges_df)),
                has_graph_labels=graph_labels_df is not None,
                has_node_features=node_features_df is not None,
                has_edge_features=edge_features_df is not None,
            )
            dataset = self.read_dataset(project_id, dataset.id)
            dataset.status = "completed"
            dataset.source_stats = source_stats
            dataset.prepared_stats = prepared_stats
            dataset.raw_data_files = raw_files
            dataset.prepared_data_files = prepared_files
            dataset.mapping_files = mapping_files
            dataset.data_files = prepared_files
            dataset.stats = prepared_stats
            dataset.error = None
            dataset.updated_at = utc_now()
            self._validate_dataset_data_files(dataset)
            self._write_json(artifact_path / "artifact.json", dataset.model_dump())
            self._log_job(project_id, job_id, f"Dataset {dataset.name} completed")
            return dataset
        except Exception as exc:
            shutil.rmtree(tmp_path, ignore_errors=True)
            self._set_dataset_status(project_id, dataset.id, "failed", error=ArtifactError(message=str(exc), job_id=job_id))
            self._log_job(project_id, job_id, f"Dataset {dataset.name} failed: {exc}")
            raise
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def _set_dataset_status(self, project_id: str, dataset_id: str, status: str, error: ArtifactError | None) -> None:
        dataset = self.read_dataset(project_id, dataset_id)
        dataset.status = status
        dataset.error = error
        dataset.updated_at = utc_now()
        self._write_json(self.dataset_path(project_id, dataset.id) / "artifact.json", dataset.model_dump())

    def _write_raw_dataset_snapshot(
        self,
        raw_dir: Path,
        nodes_df,
        edges_df,
        graph_labels_df,
        node_features_df,
        edge_features_df,
    ) -> DatasetDataFiles:
        raw_dir.mkdir(parents=True, exist_ok=True)
        nodes_df.to_parquet(raw_dir / "nodes.parquet", index=False)
        edges_df.to_parquet(raw_dir / "edges.parquet", index=False)
        files = DatasetDataFiles(nodes="raw/nodes.parquet", edges="raw/edges.parquet")
        if graph_labels_df is not None:
            graph_labels_df.to_parquet(raw_dir / "graph_labels.parquet", index=False)
            files.graph_labels = "raw/graph_labels.parquet"
        if node_features_df is not None:
            node_features_df.to_parquet(raw_dir / "node_features.parquet", index=False)
            files.node_features = "raw/node_features.parquet"
        if edge_features_df is not None:
            edge_features_df.to_parquet(raw_dir / "edge_features.parquet", index=False)
            files.edge_features = "raw/edge_features.parquet"
        return files

    def _build_graph_collection_for_dataset(
        self,
        *,
        nodes_df,
        edges_df,
        graph_labels_df,
        node_features_df,
        edge_features_df,
        graph_type: str,
        filter_largest_component: bool,
    ):
        from NEExT.io import GraphIO

        io = GraphIO()
        return io.load_from_dfs(
            edges_df=edges_df.loc[:, ["src_node_id", "dest_node_id"]],
            node_graph_df=nodes_df.loc[:, ["node_id", "graph_id"]],
            graph_labels_df=graph_labels_df,
            node_features_df=node_features_df,
            edge_features_df=edge_features_df,
            graph_type=graph_type,
            reindex_nodes=True,
            filter_largest_component=filter_largest_component,
            node_sample_rate=1.0,
        )

    def _write_prepared_dataset_outputs(self, tmp_path: Path, collection) -> tuple[DatasetDataFiles, DatasetMappingFiles, DatasetStats]:
        import pandas as pd

        prepared_dir = tmp_path / "prepared"
        mapping_dir = tmp_path / "mappings"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        mapping_dir.mkdir(parents=True, exist_ok=True)

        nodes_records = []
        edges_records = []
        graph_label_records = []
        node_feature_records = []
        edge_feature_records = []

        for graph in collection.graphs:
            for node_id in graph.nodes:
                nodes_records.append({"graph_id": graph.graph_id, "node_id": node_id})
                node_attrs = graph.node_attributes.get(node_id, {})
                if node_attrs:
                    node_feature_records.append({"graph_id": graph.graph_id, "node_id": node_id, **node_attrs})
            for src_node_id, dest_node_id in graph.edges:
                edges_records.append({"graph_id": graph.graph_id, "src_node_id": src_node_id, "dest_node_id": dest_node_id})
                edge_attrs = graph.edge_attributes.get((src_node_id, dest_node_id), {})
                if edge_attrs:
                    edge_feature_records.append(
                        {
                            "graph_id": graph.graph_id,
                            "src_node_id": src_node_id,
                            "dest_node_id": dest_node_id,
                            **edge_attrs,
                        }
                    )
            if graph.graph_label is not None:
                graph_label_records.append({"graph_id": graph.graph_id, "graph_label": graph.graph_label})

        nodes_out = pd.DataFrame(nodes_records, columns=["graph_id", "node_id"])
        edges_out = pd.DataFrame(edges_records, columns=["graph_id", "src_node_id", "dest_node_id"])
        nodes_out.to_parquet(prepared_dir / "nodes.parquet", index=False)
        edges_out.to_parquet(prepared_dir / "edges.parquet", index=False)
        prepared_files = DatasetDataFiles(nodes="prepared/nodes.parquet", edges="prepared/edges.parquet")

        if graph_label_records:
            pd.DataFrame(graph_label_records).to_parquet(prepared_dir / "graph_labels.parquet", index=False)
            prepared_files.graph_labels = "prepared/graph_labels.parquet"
        if node_feature_records:
            pd.DataFrame(node_feature_records).to_parquet(prepared_dir / "node_features.parquet", index=False)
            prepared_files.node_features = "prepared/node_features.parquet"
        if edge_feature_records:
            pd.DataFrame(edge_feature_records).to_parquet(prepared_dir / "edge_features.parquet", index=False)
            prepared_files.edge_features = "prepared/edge_features.parquet"

        node_mapping = collection.export_node_mapping_records()
        graph_mapping = collection.export_graph_mapping_records()
        node_mapping.to_parquet(mapping_dir / "node_mapping.parquet", index=False)
        graph_mapping.to_parquet(mapping_dir / "graph_mapping.parquet", index=False)
        mapping_files = DatasetMappingFiles(node_mapping="mappings/node_mapping.parquet", graph_mapping="mappings/graph_mapping.parquet")

        stats = DatasetStats(
            graph_count=len(collection.graphs),
            node_count=int(len(nodes_out)),
            edge_count=int(len(edges_out)),
            has_graph_labels=bool(graph_label_records),
            has_node_features=bool(node_feature_records),
            has_edge_features=bool(edge_feature_records),
        )
        return prepared_files, mapping_files, stats

    def _promote_directory(self, source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        source.replace(target)

    def _compute_feature_artifacts(self, project_id: str, feature_ids: list[str], job_id: str) -> None:
        features = [self.read_feature(project_id, feature_id) for feature_id in feature_ids]
        dataset_id = self._feature_dataset_id(features[0])
        dataset = self.read_dataset(project_id, dataset_id)

        try:
            if dataset.status != "completed":
                self._log_job(project_id, job_id, f"Preparing upstream dataset {dataset.name}")
                dataset = self._prepare_dataset_artifact(project_id, dataset.id, job_id)

            for feature in features:
                self._set_feature_status(project_id, feature.id, "running", error=None)

            collection = self._load_prepared_collection(project_id, dataset.id)
            grouped_features: dict[tuple[Any, ...], list[FeatureManifest]] = {}
            for feature in features:
                params = feature.operation.params
                key = (
                    params["feature_vector_length"],
                    params["normalize_features"],
                    params["n_jobs"],
                    params["parallel_backend"],
                )
                grouped_features.setdefault(key, []).append(feature)

            for group in grouped_features.values():
                params = group[0].operation.params
                feature_names = [feature.source_feature_id for feature in group]
                self._log_job(project_id, job_id, f"Computing features: {', '.join(feature_names)}")
                computed_df = self._compute_node_features(collection, feature_names, params).features_df
                for feature in group:
                    expected_columns = list(feature.expected_output.columns)
                    output_df = computed_df.loc[:, ["node_id", "graph_id"] + expected_columns].copy()
                    self._write_feature_output(project_id, feature.id, output_df)
                    feature = self.read_feature(project_id, feature.id)
                    feature.status = "completed"
                    feature.output_files = FeatureOutputFiles(features="output/features.parquet")
                    feature.output_stats = FeatureOutputStats(row_count=int(len(output_df)), column_count=int(len(output_df.columns)))
                    feature.error = None
                    feature.updated_at = utc_now()
                    self._write_json(self.feature_path(project_id, feature.id) / "artifact.json", feature.model_dump())
                    self._log_job(project_id, job_id, f"Feature {feature.name} completed")
        except Exception as exc:
            for feature in features:
                self._clean_feature_output(project_id, feature.id)
                self._set_feature_status(project_id, feature.id, "failed", error=ArtifactError(message=str(exc), job_id=job_id))
            self._log_job(project_id, job_id, f"Feature job failed: {exc}")
            raise

    def _compute_node_features(self, collection, feature_names: list[str], params: dict[str, Any]):
        from NEExT.features import StructuralNodeFeatures

        return StructuralNodeFeatures(
            graph_collection=collection,
            feature_list=feature_names,
            feature_vector_length=params["feature_vector_length"],
            normalize_features=params["normalize_features"],
            show_progress=False,
            n_jobs=params["n_jobs"],
            parallel_backend=params["parallel_backend"],
        ).compute()

    def _compute_embedding_artifacts(self, project_id: str, embedding_ids: list[str], job_id: str) -> None:
        for embedding_id in embedding_ids:
            embedding = self.read_embedding(project_id, embedding_id)
            self._set_embedding_status(project_id, embedding.id, "running", error=None)
            try:
                feature_ids = self._embedding_feature_ids(embedding)
                features = [self.read_feature(project_id, feature_id) for feature_id in feature_ids]
                dataset_ids = {self._feature_dataset_id(feature) for feature in features}
                if len(dataset_ids) != 1:
                    raise ValueError("Selected features must reference the same dataset")

                dataset = self.read_dataset(project_id, next(iter(dataset_ids)))
                if dataset.status in {"planned", "failed"}:
                    self._log_job(project_id, job_id, f"Preparing upstream dataset {dataset.name}")
                    dataset = self._prepare_dataset_artifact(project_id, dataset.id, job_id)
                if dataset.status != "completed":
                    raise ValueError("Source dataset must be completed before embedding computation")

                pending_feature_ids = [feature.id for feature in features if feature.status in {"planned", "failed"}]
                if pending_feature_ids:
                    self._log_job(project_id, job_id, f"Computing upstream features for embedding {embedding.name}")
                    self._compute_feature_artifacts(project_id, pending_feature_ids, job_id)

                features = [self.read_feature(project_id, feature_id) for feature_id in feature_ids]
                incomplete_features = [feature.name for feature in features if feature.status != "completed"]
                if incomplete_features:
                    raise ValueError(f"Source features must be completed before embedding computation: {', '.join(incomplete_features)}")

                collection = self._load_prepared_collection(project_id, dataset.id)
                feature_set = self._load_embedding_features(project_id, features)
                self._log_job(project_id, job_id, f"Computing embedding {embedding.name} with {embedding.source_embedding_id}")
                embeddings = self._compute_graph_embeddings(collection, feature_set, embedding.operation.params)
                output_df = embeddings.embeddings_df.loc[:, ["graph_id"] + list(embedding.expected_output.columns)].copy()
                self._write_embedding_output(project_id, embedding.id, output_df)

                embedding = self.read_embedding(project_id, embedding.id)
                embedding.status = "completed"
                embedding.output_files = EmbeddingOutputFiles(embeddings="output/embeddings.parquet")
                embedding.output_stats = EmbeddingOutputStats(row_count=int(len(output_df)), column_count=int(len(output_df.columns)))
                embedding.error = None
                embedding.updated_at = utc_now()
                self._write_json(self.embedding_path(project_id, embedding.id) / "artifact.json", embedding.model_dump())
                self._log_job(project_id, job_id, f"Embedding {embedding.name} completed")
            except Exception as exc:
                self._clean_embedding_output(project_id, embedding.id)
                self._set_embedding_status(project_id, embedding.id, "failed", error=ArtifactError(message=str(exc), job_id=job_id))
                self._log_job(project_id, job_id, f"Embedding {embedding.name} failed: {exc}")
                raise

    def _compute_model_artifacts(self, project_id: str, model_ids: list[str], job_id: str) -> None:
        for model_id in model_ids:
            model = self.read_model(project_id, model_id)
            self._set_model_status(project_id, model.id, "running", error=None)
            try:
                embedding_ids = self._model_embedding_ids(model)
                embeddings = [self.read_embedding(project_id, embedding_id) for embedding_id in embedding_ids]
                dataset_ids = {self._embedding_dataset_id(project_id, embedding) for embedding in embeddings}
                if len(dataset_ids) != 1:
                    raise ValueError("Selected embeddings must reference the same dataset")
                dataset_id = next(iter(dataset_ids))

                pending_embedding_ids = [embedding.id for embedding in embeddings if embedding.status in {"planned", "failed"}]
                if pending_embedding_ids:
                    self._log_job(project_id, job_id, f"Computing upstream embeddings for model {model.name}")
                    self._compute_embedding_artifacts(project_id, pending_embedding_ids, job_id)

                embeddings = [self.read_embedding(project_id, embedding_id) for embedding_id in embedding_ids]
                incomplete_embeddings = [embedding.name for embedding in embeddings if embedding.status != "completed"]
                if incomplete_embeddings:
                    raise ValueError(f"Source embeddings must be completed before model training: {', '.join(incomplete_embeddings)}")

                collection = self._load_prepared_collection(project_id, dataset_id)
                self._validate_model_labels(collection, str(model.operation.params["task_type"]))
                merged_df, feature_columns = self._load_model_embeddings(project_id, embeddings)
                self._log_job(project_id, job_id, f"Training model {model.name} with {model.source_model_id}")
                results = self._compute_graph_model(collection, merged_df, feature_columns, model)
                metrics_payload, trained_model = self._model_metrics_payload(model, results, feature_columns)
                self._write_model_output(project_id, model.id, metrics_payload, trained_model)

                metric_names = self._model_metric_names(str(model.operation.params["task_type"]))
                model = self.read_model(project_id, model.id)
                model.status = "completed"
                model.output_files = ModelOutputFiles(metrics="output/metrics.json", model="output/model.joblib")
                model.output_stats = ModelOutputStats(
                    metric_count=len(metric_names),
                    sample_size=int(model.operation.params["sample_size"]),
                    feature_count=len(feature_columns),
                    graph_count=int(len(merged_df)),
                )
                model.error = None
                model.updated_at = utc_now()
                self._write_json(self.model_path(project_id, model.id) / "artifact.json", model.model_dump())

                if model.operation.params["task_type"] == "classifier":
                    accuracy_mean = metrics_payload["summary"].get("accuracy_mean", 0)
                    self._log_job(project_id, job_id, f"Model {model.name} completed with accuracy mean {accuracy_mean:.4f}")
                else:
                    rmse_mean = metrics_payload["summary"].get("rmse_mean", 0)
                    self._log_job(project_id, job_id, f"Model {model.name} completed with RMSE mean {rmse_mean:.4f}")
            except Exception as exc:
                self._clean_model_output(project_id, model.id)
                self._set_model_status(project_id, model.id, "failed", error=ArtifactError(message=str(exc), job_id=job_id))
                self._log_job(project_id, job_id, f"Model {model.name} failed: {exc}")
                raise

    def _validate_model_labels(self, collection, task_type: str) -> None:
        labels = [graph.graph_label for graph in collection.graphs]
        if any(label is None for label in labels):
            raise ValueError("Model training requires graph labels for all graphs")
        if task_type == "regressor":
            converted_labels = []
            for label in labels:
                try:
                    converted_labels.append(float(label))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Regressor models require numeric graph labels") from exc
            for graph, converted_label in zip(collection.graphs, converted_labels):
                graph.graph_label = converted_label

    def _load_model_embeddings(self, project_id: str, embeddings: list[EmbeddingManifest]):
        import pandas as pd

        merged_df = None
        feature_columns: list[str] = []
        for embedding in embeddings:
            if embedding.output_files is None:
                raise ValueError(f"Embedding {embedding.name} has no output file")
            embedding_df = pd.read_parquet(self.embedding_path(project_id, embedding.id) / embedding.output_files.embeddings)
            expected_columns = list(embedding.expected_output.columns)
            missing_columns = sorted(set(["graph_id"] + expected_columns) - set(embedding_df.columns))
            if missing_columns:
                raise ValueError(f"Embedding {embedding.name} output is missing columns: {', '.join(missing_columns)}")
            prefix = f"{embedding.id[:8]}__"
            renamed_columns = {column: f"{prefix}{column}" for column in expected_columns}
            prepared_df = embedding_df.loc[:, ["graph_id"] + expected_columns].rename(columns=renamed_columns)
            feature_columns.extend(renamed_columns[column] for column in expected_columns)
            merged_df = prepared_df if merged_df is None else pd.merge(merged_df, prepared_df, on="graph_id", how="outer")
        if merged_df is None:
            raise ValueError("Model requires at least one embedding input")
        return merged_df, feature_columns

    def _compute_graph_model(self, collection, merged_df, feature_columns: list[str], model: ModelManifest) -> dict[str, Any]:
        from NEExT.embeddings import Embeddings
        from NEExT.ml_models import MLModels

        params = model.operation.params
        embeddings_obj = Embeddings(merged_df.loc[:, ["graph_id"] + feature_columns].copy(), model.name or model.source_model_id, feature_columns)
        return MLModels(
            graph_collection=collection,
            embeddings=embeddings_obj,
            model_type=params["task_type"],
            model_name=params["model_algorithm"],
            balance_dataset=params["balance_dataset"] if params["task_type"] == "classifier" else False,
            compute_feature_importance=False,
            sample_size=params["sample_size"],
            test_size=params["test_size"],
            random_state=42,
            n_jobs=params["n_jobs"],
            parallel_backend=params["parallel_backend"],
        ).compute()

    def _model_metrics_payload(
        self,
        model: ModelManifest,
        results: dict[str, Any],
        feature_columns: list[str],
    ) -> tuple[dict[str, Any], Any]:
        task_type = str(model.operation.params["task_type"])
        metric_names = self._model_metric_names(task_type)
        summary = {
            f"{metric_name}_mean": self._json_safe(results[f"{metric_name}_mean"])
            for metric_name in metric_names
        }
        summary.update(
            {
                f"{metric_name}_std": self._json_safe(results[f"{metric_name}_std"])
                for metric_name in metric_names
            }
        )
        metric_rows = []
        for iteration in range(int(model.operation.params["sample_size"])):
            row = {"iteration": iteration}
            for metric_name in metric_names:
                row[metric_name] = self._json_safe(results[metric_name][iteration])
            metric_rows.append(row)

        payload = {
            "model_type": task_type,
            "model_name": model.source_model_id,
            "sample_size": int(model.operation.params["sample_size"]),
            "test_size": float(model.operation.params["test_size"]),
            "random_state": 42,
            "feature_columns": list(feature_columns),
            "classes": [str(item) for item in results.get("classes", [])] if task_type == "classifier" else None,
            "summary": summary,
            "metrics": metric_rows,
        }
        return payload, results["model"]

    def _json_safe(self, value):
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _load_embedding_features(self, project_id: str, features: list[FeatureManifest]):
        import pandas as pd

        from NEExT.features import Features

        merged_features = None
        for feature in features:
            if feature.output_files is None:
                raise ValueError(f"Feature {feature.name} has no output file")
            feature_df = pd.read_parquet(self.feature_path(project_id, feature.id) / feature.output_files.features)
            expected_columns = list(feature.expected_output.columns)
            feature_obj = Features(feature_df.loc[:, ["node_id", "graph_id"] + expected_columns].copy(), expected_columns)
            merged_features = feature_obj if merged_features is None else merged_features + feature_obj
        if merged_features is None:
            raise ValueError("Embedding requires at least one feature input")
        return merged_features

    def _compute_graph_embeddings(self, collection, features, params: dict[str, Any]):
        from NEExT.embeddings import GraphEmbeddings

        return GraphEmbeddings(
            graph_collection=collection,
            features=features,
            embedding_algorithm=params["embedding_algorithm"],
            embedding_dimension=params["embedding_dimension"],
            random_state=params["random_state"],
            memory_size=params["memory_size"],
        ).compute()

    def _load_prepared_collection(self, project_id: str, dataset_id: str):
        import pandas as pd

        from NEExT.io import GraphIO

        dataset = self.read_dataset(project_id, dataset_id)
        if dataset.prepared_data_files is None:
            raise ValueError("Dataset has no prepared graph data")
        artifact_path = self.dataset_path(project_id, dataset.id)
        files = dataset.prepared_data_files

        nodes_df = pd.read_parquet(artifact_path / files.nodes)
        edges_df = pd.read_parquet(artifact_path / files.edges)
        graph_labels_df = pd.read_parquet(artifact_path / files.graph_labels) if files.graph_labels else None
        node_features_df = pd.read_parquet(artifact_path / files.node_features) if files.node_features else None
        edge_features_df = pd.read_parquet(artifact_path / files.edge_features) if files.edge_features else None

        return GraphIO().load_from_dfs(
            edges_df=edges_df.loc[:, ["src_node_id", "dest_node_id"]],
            node_graph_df=nodes_df.loc[:, ["node_id", "graph_id"]],
            graph_labels_df=graph_labels_df,
            node_features_df=node_features_df,
            edge_features_df=edge_features_df,
            graph_type=str(dataset.operation.params["graph_type"]),
            reindex_nodes=False,
            filter_largest_component=False,
            node_sample_rate=1.0,
        )

    def _write_feature_output(self, project_id: str, feature_id: str, output_df) -> None:
        artifact_path = self.feature_path(project_id, feature_id)
        tmp_path = artifact_path / "_tmp" / "output"
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=False)
        output_df.to_parquet(tmp_path / "features.parquet", index=False)
        self._promote_directory(tmp_path, artifact_path / "output")
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)

    def _clean_feature_output(self, project_id: str, feature_id: str) -> None:
        artifact_path = self.feature_path(project_id, feature_id)
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)
        if not self.read_feature(project_id, feature_id).output_files:
            shutil.rmtree(artifact_path / "output", ignore_errors=True)

    def _write_embedding_output(self, project_id: str, embedding_id: str, output_df) -> None:
        artifact_path = self.embedding_path(project_id, embedding_id)
        tmp_path = artifact_path / "_tmp" / "output"
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=False)
        output_df.to_parquet(tmp_path / "embeddings.parquet", index=False)
        self._promote_directory(tmp_path, artifact_path / "output")
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)

    def _clean_embedding_output(self, project_id: str, embedding_id: str) -> None:
        artifact_path = self.embedding_path(project_id, embedding_id)
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)
        if not self.read_embedding(project_id, embedding_id).output_files:
            shutil.rmtree(artifact_path / "output", ignore_errors=True)

    def _write_model_output(self, project_id: str, model_id: str, metrics_payload: dict[str, Any], trained_model) -> None:
        import joblib

        artifact_path = self.model_path(project_id, model_id)
        tmp_path = artifact_path / "_tmp" / "output"
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=False)
        self._write_json(tmp_path / "metrics.json", metrics_payload)
        joblib.dump(trained_model, tmp_path / "model.joblib")
        self._promote_directory(tmp_path, artifact_path / "output")
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)

    def _clean_model_output(self, project_id: str, model_id: str) -> None:
        artifact_path = self.model_path(project_id, model_id)
        shutil.rmtree(artifact_path / "_tmp", ignore_errors=True)
        if not self.read_model(project_id, model_id).output_files:
            shutil.rmtree(artifact_path / "output", ignore_errors=True)

    def _set_feature_status(self, project_id: str, feature_id: str, status: str, error: ArtifactError | None) -> None:
        feature = self.read_feature(project_id, feature_id)
        feature.status = status
        feature.error = error
        feature.updated_at = utc_now()
        self._write_json(self.feature_path(project_id, feature.id) / "artifact.json", feature.model_dump())

    def _set_embedding_status(self, project_id: str, embedding_id: str, status: str, error: ArtifactError | None) -> None:
        embedding = self.read_embedding(project_id, embedding_id)
        embedding.status = status
        embedding.error = error
        embedding.updated_at = utc_now()
        self._write_json(self.embedding_path(project_id, embedding.id) / "artifact.json", embedding.model_dump())

    def _set_model_status(self, project_id: str, model_id: str, status: str, error: ArtifactError | None) -> None:
        model = self.read_model(project_id, model_id)
        model.status = status
        model.error = error
        model.updated_at = utc_now()
        self._write_json(self.model_path(project_id, model.id) / "artifact.json", model.model_dump())

    def _feature_dataset_id(self, feature: FeatureManifest) -> str:
        for input_ref in feature.inputs:
            if input_ref.role == "source_dataset" and input_ref.artifact_kind == "dataset":
                return input_ref.artifact_id
        raise ValueError("Feature has no source dataset input")

    def _embedding_feature_ids(self, embedding: EmbeddingManifest) -> list[str]:
        feature_ids = [
            input_ref.artifact_id
            for input_ref in embedding.inputs
            if input_ref.role == "source_feature" and input_ref.artifact_kind == "feature"
        ]
        if not feature_ids:
            raise ValueError("Embedding has no source feature inputs")
        return feature_ids

    def _model_embedding_ids(self, model: ModelManifest) -> list[str]:
        embedding_ids = [
            input_ref.artifact_id
            for input_ref in model.inputs
            if input_ref.role == "source_embedding" and input_ref.artifact_kind == "embedding"
        ]
        if not embedding_ids:
            raise ValueError("Model has no source embedding inputs")
        return embedding_ids

    def _embedding_dataset_id(self, project_id: str, embedding: EmbeddingManifest) -> str:
        feature_ids = self._embedding_feature_ids(embedding)
        features = [self.read_feature(project_id, feature_id) for feature_id in feature_ids]
        dataset_ids = {self._feature_dataset_id(feature) for feature in features}
        if len(dataset_ids) != 1:
            raise ValueError("Embedding source features must reference the same dataset")
        return next(iter(dataset_ids))

    def _model_metric_names(self, task_type: str) -> list[str]:
        if task_type == "classifier":
            return ["accuracy", "recall", "precision", "f1_score"]
        if task_type == "regressor":
            return ["rmse", "mae"]
        raise ValueError(f"Unsupported model task type: {task_type}")

    def analyze_dataset(
        self,
        project_id: str,
        dataset_id: str,
        *,
        graph_id: str | None = None,
        max_nodes: int = 150,
        max_edges: int = 300,
    ) -> DatasetAnalysis:
        import pandas as pd

        if max_nodes < 1 or max_nodes > 1000:
            raise ValueError("Graph visualization max_nodes must be between 1 and 1000")
        if max_edges < 1 or max_edges > 5000:
            raise ValueError("Graph visualization max_edges must be between 1 and 5000")

        dataset = self.read_dataset(project_id, dataset_id)
        if dataset.status != "completed" or dataset.prepared_data_files is None or dataset.prepared_stats is None:
            raise ValueError("Dataset analysis is available only after preparation completes")

        artifact_path = self.dataset_path(project_id, dataset.id)
        nodes = pd.read_parquet(artifact_path / dataset.prepared_data_files.nodes)
        edges = pd.read_parquet(artifact_path / dataset.prepared_data_files.edges)
        graph_labels = None
        node_features = None
        edge_features = None
        if dataset.prepared_data_files.graph_labels:
            graph_labels = pd.read_parquet(artifact_path / dataset.prepared_data_files.graph_labels)
        if dataset.prepared_data_files.node_features:
            node_features = pd.read_parquet(artifact_path / dataset.prepared_data_files.node_features)
        if dataset.prepared_data_files.edge_features:
            edge_features = pd.read_parquet(artifact_path / dataset.prepared_data_files.edge_features)

        dropped_node_count = max(0, dataset.source_stats.node_count - dataset.prepared_stats.node_count)
        if dataset.mapping_files is not None:
            node_mapping_path = artifact_path / dataset.mapping_files.node_mapping
            if node_mapping_path.exists():
                node_mapping = pd.read_parquet(node_mapping_path)
                if "included" in node_mapping.columns:
                    dropped_node_count = int((~node_mapping["included"].astype(bool)).sum())

        label_by_graph: dict[str, object] = {}
        graph_label_distribution: dict[str, int] = {}
        if graph_labels is not None and not graph_labels.empty:
            for row in graph_labels.to_dict(orient="records"):
                label_by_graph[str(row["graph_id"])] = self._json_scalar(row["graph_label"])
            for label, count in graph_labels["graph_label"].map(self._json_scalar).value_counts(dropna=False).items():
                graph_label_distribution[str(label)] = int(count)

        node_feature_columns = [] if node_features is None else [column for column in node_features.columns if column not in {"graph_id", "node_id"}]
        edge_feature_columns = (
            []
            if edge_features is None
            else [column for column in edge_features.columns if column not in {"graph_id", "src_node_id", "dest_node_id"}]
        )

        node_counts = nodes.assign(_graph_id=nodes["graph_id"].map(str)).groupby("_graph_id").size().to_dict()
        edge_counts = edges.assign(_graph_id=edges["graph_id"].map(str)).groupby("_graph_id").size().to_dict() if not edges.empty else {}
        graph_ids = set(node_counts) | set(edge_counts)
        graph_summaries = [
            DatasetGraphSummary(
                graph_id=str(current_graph_id),
                node_count=int(node_counts.get(current_graph_id, 0)),
                edge_count=int(edge_counts.get(current_graph_id, 0)),
                graph_label=label_by_graph.get(str(current_graph_id)),
            )
            for current_graph_id in graph_ids
        ]
        graph_summaries.sort(key=lambda item: (-item.node_count, -item.edge_count, item.graph_id))
        if not graph_summaries:
            raise ValueError("Dataset has no prepared graph data")

        selected_graph_id = str(graph_id) if graph_id else graph_summaries[0].graph_id
        if selected_graph_id not in {summary.graph_id for summary in graph_summaries}:
            raise ValueError(f"Prepared graph not found: {selected_graph_id}")

        visual = self._dataset_graph_visual(
            nodes=nodes,
            edges=edges,
            graph_id=selected_graph_id,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        return DatasetAnalysis(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            dataset_status="completed",
            source_stats=dataset.source_stats,
            prepared_stats=dataset.prepared_stats,
            dropped_node_count=dropped_node_count,
            graph_label_distribution=graph_label_distribution,
            node_feature_columns=node_feature_columns,
            edge_feature_columns=edge_feature_columns,
            graph_summaries=graph_summaries,
            selected_graph_id=selected_graph_id,
            visual=visual,
        )

    def search_dataset_graphs(self, project_id: str, dataset_id: str, *, query: str = "", limit: int = 25) -> DatasetGraphSearchResponse:
        import pandas as pd

        if limit < 1 or limit > 100:
            raise ValueError("Dataset graph search limit must be between 1 and 100")

        query_text = query.strip()
        if not query_text:
            return DatasetGraphSearchResponse(query=query_text, limit=limit, total_matches=0, results=[])

        dataset = self.read_dataset(project_id, dataset_id)
        if dataset.status != "completed" or dataset.prepared_data_files is None or dataset.prepared_stats is None:
            raise ValueError("Dataset analysis is available only after preparation completes")

        artifact_path = self.dataset_path(project_id, dataset.id)
        nodes = pd.read_parquet(artifact_path / dataset.prepared_data_files.nodes)
        edges = pd.read_parquet(artifact_path / dataset.prepared_data_files.edges)
        graph_labels = None
        if dataset.prepared_data_files.graph_labels:
            graph_labels = pd.read_parquet(artifact_path / dataset.prepared_data_files.graph_labels)

        label_by_graph: dict[str, object] = {}
        if graph_labels is not None and not graph_labels.empty:
            for row in graph_labels.to_dict(orient="records"):
                label_by_graph[str(row["graph_id"])] = self._json_scalar(row["graph_label"])

        node_counts = nodes.assign(_graph_id=nodes["graph_id"].map(str)).groupby("_graph_id").size().to_dict()
        edge_counts = edges.assign(_graph_id=edges["graph_id"].map(str)).groupby("_graph_id").size().to_dict() if not edges.empty else {}
        graph_ids = sorted(set(node_counts) | set(edge_counts), key=lambda graph_id: (-int(node_counts.get(graph_id, 0)), -int(edge_counts.get(graph_id, 0)), graph_id))
        graph_order = {graph_id: index for index, graph_id in enumerate(graph_ids)}

        def match_rank(value: str) -> int:
            value_lower = value.lower()
            query_lower = query_text.lower()
            if value_lower == query_lower:
                return 0
            if value_lower.startswith(query_lower):
                return 1
            return 2

        matches: list[tuple[int, int, str, DatasetGraphSearchResult]] = []
        query_lower = query_text.lower()
        for graph_id in graph_ids:
            if query_lower in graph_id.lower():
                matches.append(
                    (
                        match_rank(graph_id),
                        graph_order[graph_id],
                        graph_id,
                        DatasetGraphSearchResult(
                            kind="graph",
                            graph_id=graph_id,
                            graph_label=label_by_graph.get(graph_id),
                            node_count=int(node_counts.get(graph_id, 0)),
                            edge_count=int(edge_counts.get(graph_id, 0)),
                        ),
                    )
                )

        node_pairs = (
            nodes.assign(_graph_id=nodes["graph_id"].map(str), _node_id=nodes["node_id"].map(str))[["_graph_id", "_node_id"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        for row in node_pairs:
            graph_id = str(row["_graph_id"])
            node_id = str(row["_node_id"])
            if query_lower not in node_id.lower():
                continue
            matches.append(
                (
                    match_rank(node_id),
                    graph_order.get(graph_id, len(graph_order)),
                    node_id,
                    DatasetGraphSearchResult(
                        kind="node",
                        graph_id=graph_id,
                        node_id=node_id,
                        graph_label=label_by_graph.get(graph_id),
                        node_count=int(node_counts.get(graph_id, 0)),
                        edge_count=int(edge_counts.get(graph_id, 0)),
                    ),
                )
            )

        matches.sort(key=lambda item: (item[0], 0 if item[3].kind == "graph" else 1, item[1], item[2]))
        results = [item[3] for item in matches[:limit]]
        return DatasetGraphSearchResponse(query=query_text, limit=limit, total_matches=len(matches), results=results)

    def dataset_node_detail(self, project_id: str, dataset_id: str, *, graph_id: str, node_id: str) -> DatasetNodeDetail:
        import pandas as pd

        dataset = self.read_dataset(project_id, dataset_id)
        if dataset.status != "completed" or dataset.prepared_data_files is None or dataset.prepared_stats is None:
            raise ValueError("Dataset analysis is available only after preparation completes")

        artifact_path = self.dataset_path(project_id, dataset.id)
        nodes = pd.read_parquet(artifact_path / dataset.prepared_data_files.nodes)
        edges = pd.read_parquet(artifact_path / dataset.prepared_data_files.edges)
        graph_id = str(graph_id)
        node_id = str(node_id)

        graph_nodes = nodes[nodes["graph_id"].map(str) == graph_id]
        if graph_nodes.empty:
            raise ValueError(f"Prepared graph not found: {graph_id}")
        if graph_nodes[graph_nodes["node_id"].map(str) == node_id].empty:
            raise ValueError(f"Prepared node not found: {node_id}")

        graph_edges = edges[edges["graph_id"].map(str) == graph_id]
        degree = 0
        for row in graph_edges.to_dict(orient="records"):
            if str(row["src_node_id"]) == node_id:
                degree += 1
            if str(row["dest_node_id"]) == node_id:
                degree += 1

        graph_label = None
        if dataset.prepared_data_files.graph_labels:
            graph_labels = pd.read_parquet(artifact_path / dataset.prepared_data_files.graph_labels)
            label_rows = graph_labels[graph_labels["graph_id"].map(str) == graph_id]
            if not label_rows.empty:
                graph_label = self._json_scalar(label_rows.iloc[0]["graph_label"])

        feature_values: dict[str, object] = {}
        if dataset.prepared_data_files.node_features:
            node_features = pd.read_parquet(artifact_path / dataset.prepared_data_files.node_features)
            feature_rows = node_features[
                (node_features["graph_id"].map(str) == graph_id) & (node_features["node_id"].map(str) == node_id)
            ]
            if not feature_rows.empty:
                feature_row = feature_rows.iloc[0]
                for column in node_features.columns:
                    if column not in {"graph_id", "node_id"}:
                        feature_values[column] = self._json_scalar(feature_row[column])

        source_graph_id = None
        source_node_id = None
        if dataset.mapping_files is not None:
            mapping_path = artifact_path / dataset.mapping_files.node_mapping
            if mapping_path.exists():
                mapping = pd.read_parquet(mapping_path)
                mapping_rows = mapping[
                    (mapping["internal_graph_id"].map(str) == graph_id) & (mapping["internal_node_id"].map(str) == node_id)
                ]
                if "included" in mapping_rows.columns:
                    mapping_rows = mapping_rows[mapping_rows["included"].astype(bool)]
                if not mapping_rows.empty:
                    mapping_row = mapping_rows.iloc[0]
                    source_graph_id = str(self._json_scalar(mapping_row["source_graph_id"]))
                    source_node_id = str(self._json_scalar(mapping_row["source_node_id"]))

        return DatasetNodeDetail(
            graph_id=graph_id,
            node_id=node_id,
            degree=degree,
            graph_label=graph_label,
            source_graph_id=source_graph_id,
            source_node_id=source_node_id,
            feature_values=feature_values,
        )

    def _dataset_graph_visual(self, *, nodes, edges, graph_id: str, max_nodes: int, max_edges: int) -> DatasetGraphVisual:
        graph_nodes = nodes[nodes["graph_id"].map(str) == graph_id]
        graph_edges = edges[edges["graph_id"].map(str) == graph_id]
        node_ids = [str(node_id) for node_id in graph_nodes["node_id"].tolist()]
        node_id_set = set(node_ids)
        degree = dict.fromkeys(node_ids, 0)
        edge_records: list[tuple[str, str]] = []

        for row in graph_edges.to_dict(orient="records"):
            source = str(row["src_node_id"])
            target = str(row["dest_node_id"])
            edge_records.append((source, target))
            if source in degree:
                degree[source] += 1
            if target in degree:
                degree[target] += 1

        sampled = False
        sample_reasons = []
        included_node_ids = node_id_set
        if len(included_node_ids) > max_nodes:
            sampled = True
            sample_reasons.append(f"nodes limited to {max_nodes}")
            ordered_nodes = sorted(included_node_ids, key=lambda node_id: (-degree.get(node_id, 0), node_id))
            included_node_ids = set(ordered_nodes[:max_nodes])

        included_edges = [(source, target) for source, target in edge_records if source in included_node_ids and target in included_node_ids]
        if len(included_edges) > max_edges:
            sampled = True
            sample_reasons.append(f"edges limited to {max_edges}")
            included_edges = sorted(
                included_edges,
                key=lambda edge: (-(degree.get(edge[0], 0) + degree.get(edge[1], 0)), edge[0], edge[1]),
            )[:max_edges]

        ordered_visual_nodes = sorted(included_node_ids, key=lambda node_id: (-degree.get(node_id, 0), node_id))
        return DatasetGraphVisual(
            graph_id=graph_id,
            node_count=int(len(graph_nodes)),
            edge_count=int(len(graph_edges)),
            sampled=sampled,
            sample_reason=", ".join(sample_reasons) if sample_reasons else None,
            nodes=[
                DatasetVisualNode(
                    id=node_id,
                    label=node_id,
                    degree=int(degree.get(node_id, 0)),
                )
                for node_id in ordered_visual_nodes
            ],
            edges=[DatasetVisualEdge(source=source, target=target) for source, target in included_edges],
        )

    def analyze_feature(
        self,
        project_id: str,
        feature_id: str,
        *,
        max_fit_rows: int = FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
        max_points: int = FEATURE_ANALYSIS_DEFAULT_MAX_POINTS,
    ) -> FeatureAnalysis:
        import pandas as pd

        if max_fit_rows < 2 or max_fit_rows > 100000:
            raise ValueError("Feature PCA max_fit_rows must be between 2 and 100000")
        if max_points < 1 or max_points > 100000:
            raise ValueError("Feature PCA max_points must be between 1 and 100000")

        feature = self.read_feature(project_id, feature_id)
        if feature.status != "completed" or feature.output_files is None:
            raise ValueError("Feature analysis is available only after computation completes")

        dataset = self.read_dataset(project_id, self._feature_dataset_id(feature))
        if dataset.status != "completed" or dataset.prepared_stats is None:
            raise ValueError("Feature analysis requires a completed source dataset")

        feature_path = self.feature_path(project_id, feature.id) / feature.output_files.features
        if not feature_path.exists():
            raise ValueError("Feature output is missing")

        features = pd.read_parquet(feature_path)
        feature_columns = [column for column in features.columns if column not in {"graph_id", "node_id"}]
        numeric_feature_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(features[column])]
        label_by_graph, graph_label_distribution = self._dataset_graph_label_maps(project_id, dataset)
        catalog = get_feature_catalog_item(feature.source_feature_id)

        output_stats = feature.output_stats or FeatureOutputStats(row_count=int(len(features)), column_count=int(len(features.columns)))
        graph_coverage = FeatureCoverage(
            covered=int(features["graph_id"].map(str).nunique(dropna=False)) if "graph_id" in features.columns else 0,
            total=dataset.prepared_stats.graph_count,
        )
        node_coverage = FeatureCoverage(
            covered=int(features.loc[:, ["graph_id", "node_id"]].drop_duplicates().shape[0])
            if {"graph_id", "node_id"}.issubset(features.columns)
            else int(len(features)),
            total=dataset.prepared_stats.node_count,
        )

        return FeatureAnalysis(
            feature_id=feature.id,
            feature_name=feature.name,
            feature_status="completed",
            source_dataset=FeatureAnalysisDataset(
                id=dataset.id,
                name=dataset.name,
                status=dataset.status,
                prepared_stats=dataset.prepared_stats,
            ),
            method=FeatureAnalysisMethod(id=feature.source_feature_id, name=catalog.name if catalog else feature.source_feature_id),
            output_stats=output_stats,
            feature_columns=feature_columns,
            numeric_feature_columns=numeric_feature_columns,
            column_summaries=self._feature_column_summaries(features, numeric_feature_columns),
            graph_coverage=graph_coverage,
            node_coverage=node_coverage,
            graph_label_distribution=graph_label_distribution,
            pca=self._feature_pca_payload(
                features,
                numeric_feature_columns,
                label_by_graph,
                max_fit_rows=max_fit_rows,
                max_points=max_points,
            ),
        )

    def search_feature_graphs(self, project_id: str, feature_id: str, *, query: str = "", limit: int = 25) -> FeatureGraphSearchResponse:
        import pandas as pd

        if limit < 1 or limit > 100:
            raise ValueError("Feature graph search limit must be between 1 and 100")

        query_text = query.strip()
        if not query_text:
            return FeatureGraphSearchResponse(query=query_text, limit=limit, total_matches=0, results=[])

        feature = self.read_feature(project_id, feature_id)
        if feature.status != "completed" or feature.output_files is None:
            raise ValueError("Feature analysis is available only after computation completes")

        dataset = self.read_dataset(project_id, self._feature_dataset_id(feature))
        feature_path = self.feature_path(project_id, feature.id) / feature.output_files.features
        if not feature_path.exists():
            raise ValueError("Feature output is missing")

        features = pd.read_parquet(feature_path)
        label_by_graph, _ = self._dataset_graph_label_maps(project_id, dataset)
        feature_columns = [column for column in features.columns if column not in {"graph_id", "node_id"}]
        numeric_feature_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(features[column])]
        graph_vectors = self._feature_graph_vectors(features, numeric_feature_columns, label_by_graph)
        sampled_graphs = set(graph_vectors.head(FEATURE_ANALYSIS_DEFAULT_MAX_POINTS)["graph_id"].tolist())
        graph_order = {str(row["graph_id"]): index for index, row in enumerate(graph_vectors.to_dict(orient="records"))}

        def match_rank(value: str) -> int:
            value_lower = value.lower()
            query_lower = query_text.lower()
            if value_lower == query_lower:
                return 0
            if value_lower.startswith(query_lower):
                return 1
            return 2

        matches_by_graph: dict[str, tuple[int, int, str, FeatureGraphSearchResult]] = {}
        query_lower = query_text.lower()
        for row in graph_vectors.to_dict(orient="records"):
            graph_id = str(row["graph_id"])
            graph_label = row.get("graph_label")
            graph_label_text = "" if graph_label is None else str(graph_label)
            graph_id_match = query_lower in graph_id.lower()
            label_match = bool(graph_label_text) and query_lower in graph_label_text.lower()
            if not graph_id_match and not label_match:
                continue
            matched_value = graph_id if graph_id_match else graph_label_text
            result = FeatureGraphSearchResult(
                graph_id=graph_id,
                graph_label=graph_label,
                in_pca_sample=graph_id in sampled_graphs,
                node_count=int(row["node_count"]),
            )
            matches_by_graph[graph_id] = (
                match_rank(matched_value),
                graph_order.get(graph_id, len(graph_order)),
                graph_id,
                result,
            )

        matches = sorted(matches_by_graph.values(), key=lambda item: (item[0], item[1], item[2]))
        return FeatureGraphSearchResponse(query=query_text, limit=limit, total_matches=len(matches), results=[item[3] for item in matches[:limit]])

    def feature_graph_detail(self, project_id: str, feature_id: str, *, graph_id: str) -> FeatureGraphDetail:
        import pandas as pd

        feature = self.read_feature(project_id, feature_id)
        if feature.status != "completed" or feature.output_files is None:
            raise ValueError("Feature analysis is available only after computation completes")

        dataset = self.read_dataset(project_id, self._feature_dataset_id(feature))
        feature_path = self.feature_path(project_id, feature.id) / feature.output_files.features
        if not feature_path.exists():
            raise ValueError("Feature output is missing")

        graph_id = str(graph_id)
        features = pd.read_parquet(feature_path)
        label_by_graph, _ = self._dataset_graph_label_maps(project_id, dataset)
        feature_columns = [column for column in features.columns if column not in {"graph_id", "node_id"}]
        numeric_feature_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(features[column])]
        graph_vectors = self._feature_graph_vectors(features, numeric_feature_columns, label_by_graph)
        graph_rows = graph_vectors[graph_vectors["graph_id"] == graph_id]
        if graph_rows.empty:
            raise ValueError(f"Feature graph not found: {graph_id}")

        graph_row = graph_rows.iloc[0]
        feature_values = {column: self._json_scalar(graph_row[column]) for column in numeric_feature_columns}
        return FeatureGraphDetail(
            graph_id=graph_id,
            graph_label=label_by_graph.get(graph_id),
            node_count=int(graph_row["node_count"]),
            feature_values=feature_values,
        )

    def analyze_embedding(
        self,
        project_id: str,
        embedding_id: str,
        *,
        max_fit_rows: int = EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
        max_points: int = EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS,
    ) -> EmbeddingAnalysis:
        import pandas as pd

        if max_fit_rows < 2 or max_fit_rows > 100000:
            raise ValueError("Embedding PCA max_fit_rows must be between 2 and 100000")
        if max_points < 1 or max_points > 100000:
            raise ValueError("Embedding PCA max_points must be between 1 and 100000")

        embedding = self.read_embedding(project_id, embedding_id)
        if embedding.status != "completed" or embedding.output_files is None:
            raise ValueError("Embedding analysis is available only after computation completes")

        feature_ids = self._embedding_feature_ids(embedding)
        features = [self.read_feature(project_id, feature_id) for feature_id in feature_ids]
        dataset = self.read_dataset(project_id, self._embedding_dataset_id(project_id, embedding))
        if dataset.status != "completed" or dataset.prepared_stats is None:
            raise ValueError("Embedding analysis requires a completed source dataset")

        embedding_path = self.embedding_path(project_id, embedding.id) / embedding.output_files.embeddings
        if not embedding_path.exists():
            raise ValueError("Embedding output is missing")

        embeddings = pd.read_parquet(embedding_path)
        embedding_columns = [column for column in embeddings.columns if column != "graph_id"]
        numeric_embedding_columns = [column for column in embedding_columns if pd.api.types.is_numeric_dtype(embeddings[column])]
        label_by_graph, graph_label_distribution = self._dataset_graph_label_maps(project_id, dataset)
        catalog = get_embedding_catalog_item(embedding.source_embedding_id)
        output_stats = embedding.output_stats or EmbeddingOutputStats(row_count=int(len(embeddings)), column_count=int(len(embeddings.columns)))

        source_features = []
        for feature in features:
            feature_catalog = get_feature_catalog_item(feature.source_feature_id)
            source_features.append(
                EmbeddingAnalysisFeature(
                    id=feature.id,
                    name=feature.name,
                    status=feature.status,
                    method=FeatureAnalysisMethod(
                        id=feature.source_feature_id,
                        name=feature_catalog.name if feature_catalog else feature.source_feature_id,
                    ),
                )
            )

        return EmbeddingAnalysis(
            embedding_id=embedding.id,
            embedding_name=embedding.name,
            embedding_status="completed",
            source_dataset=FeatureAnalysisDataset(
                id=dataset.id,
                name=dataset.name,
                status=dataset.status,
                prepared_stats=dataset.prepared_stats,
            ),
            source_features=source_features,
            algorithm=EmbeddingAnalysisAlgorithm(
                id=embedding.source_embedding_id,
                name=catalog.name if catalog else embedding.source_embedding_id,
            ),
            output_stats=output_stats,
            embedding_columns=embedding_columns,
            numeric_embedding_columns=numeric_embedding_columns,
            column_summaries=self._feature_column_summaries(embeddings, numeric_embedding_columns),
            graph_label_distribution=graph_label_distribution,
            pca=self._embedding_pca_payload(
                embeddings,
                numeric_embedding_columns,
                label_by_graph,
                max_fit_rows=max_fit_rows,
                max_points=max_points,
            ),
        )

    def search_embedding_graphs(self, project_id: str, embedding_id: str, *, query: str = "", limit: int = 25) -> EmbeddingGraphSearchResponse:
        import pandas as pd

        if limit < 1 or limit > 100:
            raise ValueError("Embedding graph search limit must be between 1 and 100")

        query_text = query.strip()
        if not query_text:
            return EmbeddingGraphSearchResponse(query=query_text, limit=limit, total_matches=0, results=[])

        embedding = self.read_embedding(project_id, embedding_id)
        if embedding.status != "completed" or embedding.output_files is None:
            raise ValueError("Embedding analysis is available only after computation completes")

        dataset = self.read_dataset(project_id, self._embedding_dataset_id(project_id, embedding))
        embedding_path = self.embedding_path(project_id, embedding.id) / embedding.output_files.embeddings
        if not embedding_path.exists():
            raise ValueError("Embedding output is missing")

        embeddings = pd.read_parquet(embedding_path)
        label_by_graph, _ = self._dataset_graph_label_maps(project_id, dataset)
        embedding_columns = [column for column in embeddings.columns if column != "graph_id"]
        numeric_embedding_columns = [column for column in embedding_columns if pd.api.types.is_numeric_dtype(embeddings[column])]
        graph_vectors = self._embedding_graph_vectors(embeddings, numeric_embedding_columns, label_by_graph)
        sampled_graphs = set(graph_vectors.head(EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS)["graph_id"].tolist())
        graph_order = {str(row["graph_id"]): index for index, row in enumerate(graph_vectors.to_dict(orient="records"))}

        def match_rank(value: str) -> int:
            value_lower = value.lower()
            query_lower = query_text.lower()
            if value_lower == query_lower:
                return 0
            if value_lower.startswith(query_lower):
                return 1
            return 2

        matches_by_graph: dict[str, tuple[int, int, str, EmbeddingGraphSearchResult]] = {}
        query_lower = query_text.lower()
        for row in graph_vectors.to_dict(orient="records"):
            graph_id = str(row["graph_id"])
            graph_label = row.get("graph_label")
            graph_label_text = "" if graph_label is None else str(graph_label)
            graph_id_match = query_lower in graph_id.lower()
            label_match = bool(graph_label_text) and query_lower in graph_label_text.lower()
            if not graph_id_match and not label_match:
                continue
            matched_value = graph_id if graph_id_match else graph_label_text
            result = EmbeddingGraphSearchResult(
                graph_id=graph_id,
                graph_label=graph_label,
                in_pca_sample=graph_id in sampled_graphs,
            )
            matches_by_graph[graph_id] = (
                match_rank(matched_value),
                graph_order.get(graph_id, len(graph_order)),
                graph_id,
                result,
            )

        matches = sorted(matches_by_graph.values(), key=lambda item: (item[0], item[1], item[2]))
        return EmbeddingGraphSearchResponse(query=query_text, limit=limit, total_matches=len(matches), results=[item[3] for item in matches[:limit]])

    def embedding_graph_detail(self, project_id: str, embedding_id: str, *, graph_id: str) -> EmbeddingGraphDetail:
        import pandas as pd

        embedding = self.read_embedding(project_id, embedding_id)
        if embedding.status != "completed" or embedding.output_files is None:
            raise ValueError("Embedding analysis is available only after computation completes")

        dataset = self.read_dataset(project_id, self._embedding_dataset_id(project_id, embedding))
        embedding_path = self.embedding_path(project_id, embedding.id) / embedding.output_files.embeddings
        if not embedding_path.exists():
            raise ValueError("Embedding output is missing")

        graph_id = str(graph_id)
        embeddings = pd.read_parquet(embedding_path)
        label_by_graph, _ = self._dataset_graph_label_maps(project_id, dataset)
        embedding_columns = [column for column in embeddings.columns if column != "graph_id"]
        numeric_embedding_columns = [column for column in embedding_columns if pd.api.types.is_numeric_dtype(embeddings[column])]
        graph_vectors = self._embedding_graph_vectors(embeddings, numeric_embedding_columns, label_by_graph)
        graph_rows = graph_vectors[graph_vectors["graph_id"] == graph_id]
        if graph_rows.empty:
            raise ValueError(f"Embedding graph not found: {graph_id}")

        graph_row = graph_rows.iloc[0]
        embedding_values = {column: self._json_scalar(graph_row[column]) for column in numeric_embedding_columns}
        sampled_graphs = set(graph_vectors.head(EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS)["graph_id"].tolist())
        return EmbeddingGraphDetail(
            graph_id=graph_id,
            graph_label=label_by_graph.get(graph_id),
            in_pca_sample=graph_id in sampled_graphs,
            embedding_values=embedding_values,
        )

    def _dataset_graph_label_maps(self, project_id: str, dataset: DatasetManifest) -> tuple[dict[str, object], dict[str, int]]:
        import pandas as pd

        if dataset.prepared_data_files is None or not dataset.prepared_data_files.graph_labels:
            return {}, {}

        labels_path = self.dataset_path(project_id, dataset.id) / dataset.prepared_data_files.graph_labels
        if not labels_path.exists():
            return {}, {}

        graph_labels = pd.read_parquet(labels_path)
        if graph_labels.empty:
            return {}, {}

        label_by_graph: dict[str, object] = {}
        graph_label_distribution: dict[str, int] = {}
        for row in graph_labels.to_dict(orient="records"):
            label_by_graph[str(row["graph_id"])] = self._json_scalar(row["graph_label"])
        for label, count in graph_labels["graph_label"].map(self._json_scalar).value_counts(dropna=False).items():
            graph_label_distribution[str(label)] = int(count)
        return label_by_graph, graph_label_distribution

    def _feature_column_summaries(self, features, numeric_columns: list[str]) -> list[FeatureColumnSummary]:
        import pandas as pd

        summaries = []
        for column in numeric_columns:
            series = pd.to_numeric(features[column], errors="coerce")
            summaries.append(
                FeatureColumnSummary(
                    column=column,
                    min=self._finite_float(series.min(skipna=True)),
                    max=self._finite_float(series.max(skipna=True)),
                    mean=self._finite_float(series.mean(skipna=True)),
                    std=self._finite_float(series.std(skipna=True)),
                    null_count=int(series.isna().sum()),
                )
            )
        return summaries

    def _feature_graph_vectors(self, features, numeric_columns: list[str], label_by_graph: dict[str, object]):
        import pandas as pd

        ordered = self._ordered_feature_rows(features)
        graph_counts = (
            ordered.groupby("_graph_id_str", sort=False)["_node_id_str"]
            .nunique()
            .rename("node_count")
            .to_frame()
        )
        if numeric_columns:
            numeric_values = ordered.loc[:, ["_graph_id_str", *numeric_columns]].copy()
            for column in numeric_columns:
                numeric_values[column] = pd.to_numeric(numeric_values[column], errors="coerce")
            graph_means = numeric_values.groupby("_graph_id_str", sort=False)[numeric_columns].mean()
            graph_vectors = graph_counts.join(graph_means)
        else:
            graph_vectors = graph_counts

        graph_vectors = graph_vectors.reset_index().rename(columns={"_graph_id_str": "graph_id"})
        graph_vectors["graph_label"] = graph_vectors["graph_id"].map(lambda graph_id: label_by_graph.get(str(graph_id)))
        graph_vectors["color_value"] = graph_vectors.apply(
            lambda row: str(row["graph_label"]) if label_by_graph else str(row["graph_id"]),
            axis=1,
        )
        return graph_vectors.sort_values("graph_id", kind="mergesort").reset_index(drop=True)

    def _feature_pca_payload(
        self,
        features,
        numeric_columns: list[str],
        label_by_graph: dict[str, object],
        *,
        max_fit_rows: int,
        max_points: int,
    ) -> FeaturePcaPayload:
        import numpy as np
        import pandas as pd

        graph_vectors = self._feature_graph_vectors(features, numeric_columns, label_by_graph)
        source_row_count = int(len(features))
        total_graphs = int(len(graph_vectors))
        fit_row_count = min(total_graphs, max_fit_rows)
        point_count = min(total_graphs, max_points)
        fit_sampled = total_graphs > max_fit_rows
        points_sampled = total_graphs > max_points
        sample_reasons = []
        if fit_sampled:
            sample_reasons.append(f"PCA fit graphs limited to {max_fit_rows}")
        if points_sampled:
            sample_reasons.append(f"plotted graphs limited to {max_points}")

        base_payload = {
            "color_by": "graph_label" if label_by_graph else "graph_id",
            "numeric_columns": numeric_columns,
            "source_row_count": source_row_count,
            "total_graphs": total_graphs,
            "total_rows": total_graphs,
            "fit_row_count": fit_row_count,
            "point_count": point_count,
            "max_fit_rows": max_fit_rows,
            "max_points": max_points,
            "fit_sampled": fit_sampled,
            "points_sampled": points_sampled,
            "sampled": fit_sampled or points_sampled,
            "sample_reason": ", ".join(sample_reasons) if sample_reasons else None,
        }
        if total_graphs < 1:
            return FeaturePcaPayload(available=False, reason="2D plotting requires at least one graph", **base_payload)
        if len(numeric_columns) < 2:
            return FeaturePcaPayload(available=False, reason="2D plotting requires at least two numeric feature columns", **base_payload)

        fit_rows = graph_vectors.head(max_fit_rows)
        point_rows = graph_vectors.head(max_points)

        def graph_points(coordinates) -> list[FeaturePcaPoint]:
            points = []
            for index, row in enumerate(point_rows.to_dict(orient="records")):
                graph_id = str(row["graph_id"])
                points.append(
                    FeaturePcaPoint(
                        graph_id=graph_id,
                        x=float(coordinates[index, 0]),
                        y=float(coordinates[index, 1]),
                        graph_label=row.get("graph_label"),
                        color_value=str(row["color_value"]),
                        node_count=int(row["node_count"]),
                    )
                )
            return points

        if len(numeric_columns) == 2:
            point_frame = point_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
            coordinates = np.nan_to_num(point_frame.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            raw_payload = {
                **base_payload,
                "fit_row_count": 0,
                "fit_sampled": False,
                "sampled": points_sampled,
                "sample_reason": f"plotted graphs limited to {max_points}" if points_sampled else None,
            }
            return FeaturePcaPayload(
                available=True,
                projection_method="raw",
                x_axis_label=numeric_columns[0],
                y_axis_label=numeric_columns[1],
                explained_variance_ratio=[],
                points=graph_points(coordinates),
                **raw_payload,
            )

        from sklearn.decomposition import PCA

        if total_graphs < 2:
            return FeaturePcaPayload(available=False, reason="PCA requires at least two graphs", **base_payload)

        fit_frame = fit_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        fill_values = fit_frame.mean(axis=0, skipna=True).fillna(0.0)
        fit_matrix = np.nan_to_num(fit_frame.fillna(fill_values).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        point_matrix = np.nan_to_num(
            point_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float).fillna(fill_values).to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if min(fit_matrix.shape[0], fit_matrix.shape[1]) < 2:
            return FeaturePcaPayload(available=False, reason="PCA requires at least two fit graphs and columns", **base_payload)

        pca = PCA(n_components=2, svd_solver="full")
        coordinates = np.nan_to_num(pca.fit(fit_matrix).transform(point_matrix), nan=0.0, posinf=0.0, neginf=0.0)
        explained_variance_ratio = [float(value) for value in np.nan_to_num(pca.explained_variance_ratio_, nan=0.0).tolist()]

        return FeaturePcaPayload(
            available=True,
            projection_method="pca",
            x_axis_label="PC1",
            y_axis_label="PC2",
            explained_variance_ratio=explained_variance_ratio,
            points=graph_points(coordinates),
            **base_payload,
        )

    def _embedding_graph_vectors(self, embeddings, numeric_columns: list[str], label_by_graph: dict[str, object]):
        import pandas as pd

        if "graph_id" not in embeddings.columns:
            raise ValueError("Embedding output is missing graph_id")

        graph_vectors = embeddings.copy()
        graph_vectors["graph_id"] = graph_vectors["graph_id"].map(str)
        for column in numeric_columns:
            graph_vectors[column] = pd.to_numeric(graph_vectors[column], errors="coerce")
        graph_vectors["graph_label"] = graph_vectors["graph_id"].map(lambda graph_id: label_by_graph.get(str(graph_id)))
        graph_vectors["color_value"] = graph_vectors.apply(
            lambda row: str(row["graph_label"]) if label_by_graph else str(row["graph_id"]),
            axis=1,
        )
        return graph_vectors.sort_values("graph_id", kind="mergesort").reset_index(drop=True)

    def _embedding_pca_payload(
        self,
        embeddings,
        numeric_columns: list[str],
        label_by_graph: dict[str, object],
        *,
        max_fit_rows: int,
        max_points: int,
    ) -> EmbeddingPcaPayload:
        import numpy as np
        import pandas as pd

        graph_vectors = self._embedding_graph_vectors(embeddings, numeric_columns, label_by_graph)
        source_row_count = int(len(embeddings))
        total_graphs = int(len(graph_vectors))
        fit_row_count = min(total_graphs, max_fit_rows)
        point_count = min(total_graphs, max_points)
        fit_sampled = total_graphs > max_fit_rows
        points_sampled = total_graphs > max_points
        sample_reasons = []
        if fit_sampled:
            sample_reasons.append(f"PCA fit graphs limited to {max_fit_rows}")
        if points_sampled:
            sample_reasons.append(f"plotted graphs limited to {max_points}")

        base_payload = {
            "color_by": "graph_label" if label_by_graph else "graph_id",
            "numeric_columns": numeric_columns,
            "source_row_count": source_row_count,
            "total_graphs": total_graphs,
            "total_rows": total_graphs,
            "fit_row_count": fit_row_count,
            "point_count": point_count,
            "max_fit_rows": max_fit_rows,
            "max_points": max_points,
            "fit_sampled": fit_sampled,
            "points_sampled": points_sampled,
            "sampled": fit_sampled or points_sampled,
            "sample_reason": ", ".join(sample_reasons) if sample_reasons else None,
        }
        if total_graphs < 1:
            return EmbeddingPcaPayload(available=False, reason="2D plotting requires at least one graph", **base_payload)
        if len(numeric_columns) < 2:
            return EmbeddingPcaPayload(available=False, reason="2D plotting requires at least two numeric embedding columns", **base_payload)

        fit_rows = graph_vectors.head(max_fit_rows)
        point_rows = graph_vectors.head(max_points)

        def graph_points(coordinates) -> list[EmbeddingPcaPoint]:
            points = []
            for index, row in enumerate(point_rows.to_dict(orient="records")):
                graph_id = str(row["graph_id"])
                points.append(
                    EmbeddingPcaPoint(
                        graph_id=graph_id,
                        x=float(coordinates[index, 0]),
                        y=float(coordinates[index, 1]),
                        graph_label=row.get("graph_label"),
                        color_value=str(row["color_value"]),
                    )
                )
            return points

        if len(numeric_columns) == 2:
            point_frame = point_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
            coordinates = np.nan_to_num(point_frame.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            raw_payload = {
                **base_payload,
                "fit_row_count": 0,
                "fit_sampled": False,
                "sampled": points_sampled,
                "sample_reason": f"plotted graphs limited to {max_points}" if points_sampled else None,
            }
            return EmbeddingPcaPayload(
                available=True,
                projection_method="raw",
                x_axis_label=numeric_columns[0],
                y_axis_label=numeric_columns[1],
                explained_variance_ratio=[],
                points=graph_points(coordinates),
                **raw_payload,
            )

        from sklearn.decomposition import PCA

        if total_graphs < 2:
            return EmbeddingPcaPayload(available=False, reason="PCA requires at least two graphs", **base_payload)

        fit_frame = fit_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        fill_values = fit_frame.mean(axis=0, skipna=True).fillna(0.0)
        fit_matrix = np.nan_to_num(fit_frame.fillna(fill_values).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        point_matrix = np.nan_to_num(
            point_rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float).fillna(fill_values).to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if min(fit_matrix.shape[0], fit_matrix.shape[1]) < 2:
            return EmbeddingPcaPayload(available=False, reason="PCA requires at least two fit graphs and columns", **base_payload)

        pca = PCA(n_components=2, svd_solver="full")
        coordinates = np.nan_to_num(pca.fit(fit_matrix).transform(point_matrix), nan=0.0, posinf=0.0, neginf=0.0)
        explained_variance_ratio = [float(value) for value in np.nan_to_num(pca.explained_variance_ratio_, nan=0.0).tolist()]

        return EmbeddingPcaPayload(
            available=True,
            projection_method="pca",
            x_axis_label="PC1",
            y_axis_label="PC2",
            explained_variance_ratio=explained_variance_ratio,
            points=graph_points(coordinates),
            **base_payload,
        )

    def _ordered_feature_rows(self, features):
        ordered = features.copy()
        ordered["_graph_id_str"] = ordered["graph_id"].map(str)
        ordered["_node_id_str"] = ordered["node_id"].map(str)
        return ordered.sort_values(["_graph_id_str", "_node_id_str"], kind="mergesort").reset_index(drop=True)

    def _finite_float(self, value) -> float | None:
        import math

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _json_scalar(self, value):
        import pandas as pd

        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    def _dataset_preview_path(self, project_id: str, dataset: DatasetManifest, table: str) -> Path:
        prepared_files = dataset.prepared_data_files
        mapping_files = dataset.mapping_files
        table_paths = {
            "nodes": prepared_files.nodes if prepared_files is not None else None,
            "edges": prepared_files.edges if prepared_files is not None else None,
            "graph_labels": prepared_files.graph_labels if prepared_files is not None else None,
            "node_features": prepared_files.node_features if prepared_files is not None else None,
            "edge_features": prepared_files.edge_features if prepared_files is not None else None,
            "node_mapping": mapping_files.node_mapping if mapping_files is not None else None,
            "mapping": mapping_files.node_mapping if mapping_files is not None else None,
            "graph_mapping": mapping_files.graph_mapping if mapping_files is not None else None,
        }
        if table not in table_paths:
            raise ValueError("Unsupported dataset preview table")
        data_file = table_paths[table]
        if not data_file:
            raise ValueError(f"Dataset has no {table} data")
        path = self.dataset_path(project_id, dataset.id) / data_file
        if not path.exists():
            raise ValueError(f"Dataset {table} data is missing")
        return path

    def preview_dataset(self, project_id: str, dataset_id: str, table: str, *, limit: int = 20, offset: int = 0) -> TabularPreview:
        dataset = self.read_dataset(project_id, dataset_id)
        if dataset.status != "completed":
            raise ValueError("Dataset preview is available only after preparation completes")
        path = self._dataset_preview_path(project_id, dataset, table)
        return self._preview_parquet(path, limit=limit, offset=offset)

    def preview_feature(self, project_id: str, feature_id: str, *, limit: int = 20, offset: int = 0) -> TabularPreview:
        feature = self.read_feature(project_id, feature_id)
        if feature.status != "completed" or feature.output_files is None:
            raise ValueError("Feature preview is available only after computation completes")
        return self._preview_parquet(self.feature_path(project_id, feature.id) / feature.output_files.features, limit=limit, offset=offset)

    def preview_embedding(self, project_id: str, embedding_id: str, *, limit: int = 20, offset: int = 0) -> TabularPreview:
        embedding = self.read_embedding(project_id, embedding_id)
        if embedding.status != "completed" or embedding.output_files is None:
            raise ValueError("Embedding preview is available only after computation completes")
        return self._preview_parquet(self.embedding_path(project_id, embedding.id) / embedding.output_files.embeddings, limit=limit, offset=offset)

    def analyze_model(self, project_id: str, model_id: str) -> ModelAnalysis:
        model = self.read_model(project_id, model_id)
        if model.status != "completed" or model.output_files is None:
            raise ValueError("Model analysis is available only after training completes")

        metrics_path = self.model_path(project_id, model.id) / model.output_files.metrics
        if not metrics_path.exists():
            raise ValueError("Model metrics output is missing")

        payload = self._read_json(metrics_path)
        embedding_ids = self._model_embedding_ids(model)
        embeddings = [self.read_embedding(project_id, embedding_id) for embedding_id in embedding_ids]
        dataset_ids = {self._embedding_dataset_id(project_id, embedding) for embedding in embeddings}
        if len(dataset_ids) != 1:
            raise ValueError("Model source embeddings must reference the same dataset")
        dataset = self.read_dataset(project_id, next(iter(dataset_ids)))

        source_embeddings = []
        for embedding in embeddings:
            embedding_catalog = get_embedding_catalog_item(embedding.source_embedding_id)
            source_embeddings.append(
                ModelAnalysisEmbedding(
                    id=embedding.id,
                    name=embedding.name,
                    status=embedding.status,
                    algorithm=ModelAnalysisAlgorithm(
                        id=embedding.source_embedding_id,
                        name=embedding_catalog.name if embedding_catalog else embedding.source_embedding_id,
                    ),
                )
            )

        source_features = []
        seen_feature_ids: set[str] = set()
        for embedding in embeddings:
            for feature_id in self._embedding_feature_ids(embedding):
                if feature_id in seen_feature_ids:
                    continue
                seen_feature_ids.add(feature_id)
                feature = self.read_feature(project_id, feature_id)
                feature_catalog = get_feature_catalog_item(feature.source_feature_id)
                source_features.append(
                    EmbeddingAnalysisFeature(
                        id=feature.id,
                        name=feature.name,
                        status=feature.status,
                        method=FeatureAnalysisMethod(
                            id=feature.source_feature_id,
                            name=feature_catalog.name if feature_catalog else feature.source_feature_id,
                        ),
                    )
                )

        metric_rows = payload.get("metrics", [])
        expected_metrics = list(model.expected_output.metrics)
        metric_series = []
        for metric_name in expected_metrics:
            points = []
            for row_index, row in enumerate(metric_rows):
                try:
                    iteration = int(row.get("iteration", row_index))
                except (TypeError, ValueError):
                    iteration = row_index
                points.append(ModelMetricPoint(iteration=iteration, value=self._finite_float(row.get(metric_name))))
            metric_series.append(ModelMetricSeries(metric=metric_name, points=points))

        task_type = str(model.operation.params.get("task_type", payload.get("model_type", "")))
        if task_type not in {"classifier", "regressor"}:
            raise ValueError(f"Unsupported model task type: {task_type}")

        feature_columns = list(payload.get("feature_columns", []))
        output_stats = model.output_stats or ModelOutputStats(
            metric_count=len(expected_metrics),
            sample_size=len(metric_rows),
            feature_count=len(feature_columns),
            graph_count=dataset.prepared_stats.graph_count if dataset.prepared_stats else 0,
        )
        catalog = get_model_catalog_item(model.source_model_id)
        return ModelAnalysis(
            model_id=model.id,
            model_name=model.name,
            model_status="completed",
            source_dataset=FeatureAnalysisDataset(
                id=dataset.id,
                name=dataset.name,
                status=dataset.status,
                prepared_stats=dataset.prepared_stats,
            ),
            source_embeddings=source_embeddings,
            source_features=source_features,
            algorithm=ModelAnalysisAlgorithm(
                id=model.source_model_id,
                name=catalog.name if catalog else model.source_model_id,
            ),
            task_type=task_type,
            expected_metrics=expected_metrics,
            output_stats=output_stats,
            sample_size=int(payload.get("sample_size", model.operation.params.get("sample_size", len(metric_rows)))),
            test_size=float(payload.get("test_size", model.operation.params.get("test_size", 0.0))),
            random_state=payload.get("random_state", model.operation.params.get("random_state")),
            classes=payload.get("classes"),
            feature_columns=feature_columns,
            summary=payload.get("summary", {}),
            metrics=metric_rows,
            metric_series=metric_series,
        )

    def preview_model(self, project_id: str, model_id: str) -> ModelPreview:
        model = self.read_model(project_id, model_id)
        if model.status != "completed" or model.output_files is None:
            raise ValueError("Model preview is available only after training completes")
        metrics_path = self.model_path(project_id, model.id) / model.output_files.metrics
        if not metrics_path.exists():
            raise ValueError("Model metrics output is missing")
        payload = self._read_json(metrics_path)
        return ModelPreview(
            summary=payload.get("summary", {}),
            metrics=payload.get("metrics", []),
            feature_columns=payload.get("feature_columns", []),
            classes=payload.get("classes"),
        )

    def _preview_parquet(self, path: Path, *, limit: int, offset: int) -> TabularPreview:
        import pandas as pd
        import pyarrow.parquet as pq

        if limit < 1 or limit > 200:
            raise ValueError("Preview limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("Preview offset must be non-negative")
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        rows_remaining_to_skip = offset
        rows_remaining_to_take = limit
        chunks = []
        for batch in parquet_file.iter_batches(batch_size=max(limit, 1024)):
            batch_rows = batch.num_rows
            if rows_remaining_to_skip >= batch_rows:
                rows_remaining_to_skip -= batch_rows
                continue
            table = batch.slice(rows_remaining_to_skip, rows_remaining_to_take)
            chunks.append(table.to_pandas())
            rows_remaining_to_take -= table.num_rows
            rows_remaining_to_skip = 0
            if rows_remaining_to_take <= 0:
                break
        if chunks:
            frame = pd.concat(chunks, ignore_index=True)
        else:
            frame = pd.DataFrame(columns=parquet_file.schema.names)
        frame = frame.astype(object).where(pd.notnull(frame), None)
        return TabularPreview(
            columns=list(frame.columns),
            rows=frame.to_dict(orient="records"),
            offset=offset,
            limit=limit,
            total_rows=total_rows,
        )

    def _read_catalog_frames(self, files: dict[str, str]):
        import pandas as pd

        required_files = {"edges", "node_graph_mapping"}
        missing = sorted(required_files - set(files))
        if missing:
            raise ValueError(f"Dataset catalog source is missing required file declarations: {', '.join(missing)}")

        frames = {}
        for key, source_path in files.items():
            try:
                frames[key] = pd.read_csv(source_path)
            except Exception as exc:
                raise ValueError(f"Could not read dataset source file: {key}") from exc

        self._require_columns(frames["edges"], {"src_node_id", "dest_node_id"}, "edges.csv")
        self._require_columns(frames["node_graph_mapping"], {"node_id", "graph_id"}, "node_graph_mapping.csv")
        if "graph_labels" in frames:
            self._require_columns(frames["graph_labels"], {"graph_id", "graph_label"}, "graph_labels.csv")
        if "node_features" in frames:
            self._require_columns(frames["node_features"], {"node_id"}, "node_features.csv")
        if "edge_features" in frames:
            self._require_columns(frames["edge_features"], {"src_node_id", "dest_node_id"}, "edge_features.csv")
        return frames

    def _require_columns(self, frame, columns: set[str], file_name: str) -> None:
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f"{file_name} is missing required columns: {', '.join(missing)}")

    def _normalized_nodes_and_edges(self, node_graph_df, edges_df):
        import pandas as pd

        nodes_df = node_graph_df.loc[:, ["node_id", "graph_id"]].copy()
        if nodes_df.empty:
            raise ValueError("node_graph_mapping.csv must contain at least one node")
        if nodes_df["node_id"].duplicated().any():
            raise ValueError("node_graph_mapping.csv must contain unique node_id values")

        node_graph_by_id = nodes_df.set_index("node_id")["graph_id"]
        src_graph = edges_df["src_node_id"].map(node_graph_by_id)
        dest_graph = edges_df["dest_node_id"].map(node_graph_by_id)
        if src_graph.isna().any() or dest_graph.isna().any():
            raise ValueError("edges.csv contains endpoints that are not present in node_graph_mapping.csv")
        if (src_graph != dest_graph).any():
            raise ValueError("edges.csv contains endpoints from different graphs")

        edges_out = pd.DataFrame(
            {
                "graph_id": src_graph.to_numpy(),
                "src_node_id": edges_df["src_node_id"].to_numpy(),
                "dest_node_id": edges_df["dest_node_id"].to_numpy(),
            }
        )
        return nodes_df, edges_out

    def _validate_graph_labels(self, graph_labels_df, nodes_df) -> None:
        unknown_graphs = set(graph_labels_df["graph_id"]) - set(nodes_df["graph_id"])
        if unknown_graphs:
            raise ValueError("graph_labels.csv contains graph_id values that are not present in node_graph_mapping.csv")

    def _validate_node_features(self, node_features_df, nodes_df) -> None:
        unknown_nodes = set(node_features_df["node_id"]) - set(nodes_df["node_id"])
        if unknown_nodes:
            raise ValueError("node_features.csv contains node_id values that are not present in node_graph_mapping.csv")

    def _validate_edge_features(self, edge_features_df, nodes_df) -> None:
        node_graph_by_id = nodes_df.set_index("node_id")["graph_id"]
        src_graph = edge_features_df["src_node_id"].map(node_graph_by_id)
        dest_graph = edge_features_df["dest_node_id"].map(node_graph_by_id)
        if src_graph.isna().any() or dest_graph.isna().any():
            raise ValueError("edge_features.csv contains endpoints that are not present in node_graph_mapping.csv")
        if (src_graph != dest_graph).any():
            raise ValueError("edge_features.csv contains endpoints from different graphs")

    def _validate_dataset_data_files(self, manifest: DatasetManifest) -> None:
        file_groups = [
            manifest.data_files,
            manifest.raw_data_files,
            manifest.prepared_data_files,
            manifest.mapping_files,
        ]
        for file_group in file_groups:
            if file_group is None:
                continue
            for data_file in file_group.model_dump(exclude_none=True).values():
                self._validate_relative_file_ref(data_file)

    def _validate_relative_file_ref(self, data_file: str) -> None:
        path = PurePosixPath(data_file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest file paths must be relative")

    def _validate_feature_manifest(self, manifest: FeatureManifest) -> None:
        if len(manifest.inputs) != 1:
            raise ValueError("Feature manifests must reference exactly one dataset input")
        input_ref = manifest.inputs[0]
        if input_ref.role != "source_dataset" or input_ref.artifact_kind != "dataset":
            raise ValueError("Feature manifests must reference a source dataset input")
        self._normalize_artifact_id(input_ref.artifact_id)
        if get_feature_catalog_item(manifest.source_feature_id) is None:
            raise ValueError("Feature manifest references an unknown feature catalog entry")
        if get_operation_definition(manifest.operation.operation_id, manifest.operation.operation_version) is None:
            raise ValueError("Feature manifest references an unsupported operation")
        FeatureCreateParams.model_validate(
            {
                "feature_vector_length": manifest.operation.params.get("feature_vector_length"),
                "normalize_features": manifest.operation.params.get("normalize_features"),
                "n_jobs": manifest.operation.params.get("n_jobs"),
                "parallel_backend": manifest.operation.params.get("parallel_backend"),
            }
        )
        if manifest.operation.params.get("feature_list") != [manifest.source_feature_id]:
            raise ValueError("Feature manifest operation must target exactly one source feature")
        if manifest.output_files is not None:
            for data_file in manifest.output_files.model_dump(exclude_none=True).values():
                self._validate_relative_file_ref(data_file)

    def _validate_embedding_manifest(self, manifest: EmbeddingManifest) -> None:
        if len(manifest.inputs) < 1:
            raise ValueError("Embedding manifests must reference at least one feature input")
        feature_ids = []
        for input_ref in manifest.inputs:
            if input_ref.role != "source_feature" or input_ref.artifact_kind != "feature":
                raise ValueError("Embedding manifests must reference source feature inputs")
            self._normalize_artifact_id(input_ref.artifact_id)
            feature_ids.append(input_ref.artifact_id)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Embedding manifests must reference unique source feature inputs")
        catalog = get_embedding_catalog_item(manifest.source_embedding_id)
        if catalog is None:
            raise ValueError("Embedding manifest references an unknown embedding catalog entry")
        if (
            manifest.operation.operation_id != catalog.operation_id
            or manifest.operation.operation_version != catalog.operation_version
        ):
            raise ValueError("Embedding manifest operation must match its catalog entry")
        if get_embedding_operation_definition(manifest.operation.operation_id, manifest.operation.operation_version) is None:
            raise ValueError("Embedding manifest references an unsupported operation")
        EmbeddingCreateParams.model_validate(
            {
                "embedding_dimension": manifest.operation.params.get("embedding_dimension"),
            }
        )
        if manifest.operation.params.get("embedding_algorithm") != manifest.source_embedding_id:
            raise ValueError("Embedding manifest operation must target its source embedding algorithm")
        if manifest.operation.params.get("random_state") != 42:
            raise ValueError("Embedding manifest operation must use the fixed random seed")
        if manifest.operation.params.get("memory_size") != "4G":
            raise ValueError("Embedding manifest operation must use the fixed memory size")
        if manifest.operation.params.get("feature_ids") != feature_ids:
            raise ValueError("Embedding manifest operation feature IDs must match source feature inputs")
        if manifest.operation.params.get("feature_columns") != "all":
            raise ValueError("Embedding manifest operation must use all source feature columns")
        expected_columns = [f"emb_{index}" for index in range(manifest.operation.params["embedding_dimension"])]
        if manifest.expected_output.columns != expected_columns:
            raise ValueError("Embedding manifest expected columns must match the embedding dimension")
        if manifest.output_files is not None:
            for data_file in manifest.output_files.model_dump(exclude_none=True).values():
                self._validate_relative_file_ref(data_file)

    def _validate_model_manifest(self, manifest: ModelManifest) -> None:
        if len(manifest.inputs) < 1:
            raise ValueError("Model manifests must reference at least one embedding input")
        embedding_ids = []
        for input_ref in manifest.inputs:
            if input_ref.role != "source_embedding" or input_ref.artifact_kind != "embedding":
                raise ValueError("Model manifests must reference source embedding inputs")
            self._normalize_artifact_id(input_ref.artifact_id)
            embedding_ids.append(input_ref.artifact_id)
        if len(set(embedding_ids)) != len(embedding_ids):
            raise ValueError("Model manifests must reference unique source embedding inputs")
        catalog = get_model_catalog_item(manifest.source_model_id)
        if catalog is None:
            raise ValueError("Model manifest references an unknown model catalog entry")
        if (
            manifest.operation.operation_id != catalog.operation_id
            or manifest.operation.operation_version != catalog.operation_version
        ):
            raise ValueError("Model manifest operation must match its catalog entry")
        if get_model_operation_definition(manifest.operation.operation_id, manifest.operation.operation_version) is None:
            raise ValueError("Model manifest references an unsupported operation")
        ModelCreateParams.model_validate(
            {
                "task_type": manifest.operation.params.get("task_type"),
                "sample_size": manifest.operation.params.get("sample_size"),
                "test_size": manifest.operation.params.get("test_size"),
                "balance_dataset": manifest.operation.params.get("balance_dataset"),
                "n_jobs": manifest.operation.params.get("n_jobs"),
                "parallel_backend": manifest.operation.params.get("parallel_backend"),
            }
        )
        if manifest.operation.params.get("model_algorithm") != manifest.source_model_id:
            raise ValueError("Model manifest operation must target its source model algorithm")
        if manifest.operation.params.get("random_state") != 42:
            raise ValueError("Model manifest operation must use the fixed random seed")
        if manifest.operation.params.get("embedding_ids") != embedding_ids:
            raise ValueError("Model manifest operation embedding IDs must match source embedding inputs")
        if manifest.operation.params.get("embedding_columns") != "all":
            raise ValueError("Model manifest operation must use all source embedding columns")
        expected_metrics = self._model_metric_names(str(manifest.operation.params["task_type"]))
        if manifest.expected_output.metrics != expected_metrics:
            raise ValueError("Model manifest expected metrics must match the task type")
        if manifest.output_files is not None:
            for data_file in manifest.output_files.model_dump(exclude_none=True).values():
                self._validate_relative_file_ref(data_file)

    def delete_project(self, project_id: str) -> ProjectDeletionSummary:
        project = self.read_project(project_id)
        project_path = self.project_path(project.id)

        self.trash_projects_path.mkdir(parents=True, exist_ok=True)
        trash_path = self.trash_projects_path / project.id
        if trash_path.exists():
            trash_path = self.trash_projects_path / self._trash_folder_name(project.id)

        shutil.move(str(project_path), str(trash_path))
        return ProjectDeletionSummary(
            id=project.id,
            name=project.name,
            trashed_path=trash_path.relative_to(self.workspace_path).as_posix(),
        )
