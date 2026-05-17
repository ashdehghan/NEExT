"""Curated MCP service methods for the local NEExT Workbench workspace."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .schemas import (
    DatasetCreateRequest,
    DatasetPrepareParams,
    EmbeddingCreateParams,
    EmbeddingCreateRequest,
    EmbeddingRunBatchRequest,
    FeatureCreateParams,
    FeatureCreateRequest,
    FeatureRunBatchRequest,
    ModelCreateParams,
    ModelCreateRequest,
    ModelRunBatchRequest,
    ProjectCreate,
)
from .storage import (
    EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
    EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS,
    FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
    FEATURE_ANALYSIS_DEFAULT_MAX_POINTS,
    WorkbenchStore,
)

ARTIFACT_KIND_ALIASES = {
    "dataset": "datasets",
    "datasets": "datasets",
    "feature": "features",
    "features": "features",
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "model": "models",
    "models": "models",
}


class WorkbenchMcpService:
    """Researcher-oriented operations backed by WorkbenchStore."""

    def __init__(self, store: WorkbenchStore):
        self.store = store

    def _dump(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [self._dump(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._dump(item) for key, item in value.items()}
        return value

    def _artifact_kind(self, kind: str) -> str:
        normalized = ARTIFACT_KIND_ALIASES.get(kind.strip().lower())
        if normalized is None:
            raise ValueError("Artifact kind must be one of datasets, features, embeddings, or models")
        return normalized

    def workspace_summary(self) -> dict[str, Any]:
        meta = self.store.read_workspace_meta()
        projects = self.store.list_projects()
        return {
            "schema_version": str(meta.get("schema_version", meta.get("version", "1"))),
            "manifest_type": meta.get("manifest_type", "workspace"),
            "project_count": len(projects),
            "catalog_counts": {
                "datasets": len(self.store.list_dataset_catalog()),
                "features": len(self.store.list_feature_catalog()),
                "embeddings": len(self.store.list_embedding_catalog()),
                "models": len(self.store.list_model_catalog()),
            },
        }

    def list_projects(self) -> list[dict[str, Any]]:
        return self._dump(self.store.list_projects())

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        return self._dump(self.store.create_project(ProjectCreate(name=name, description=description)))

    def list_catalog(self, kind: str) -> list[dict[str, Any]]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            return self._dump(self.store.list_dataset_catalog())
        if artifact_kind == "features":
            return self._dump(self.store.list_feature_catalog())
        if artifact_kind == "embeddings":
            return self._dump(self.store.list_embedding_catalog())
        return self._dump(self.store.list_model_catalog())

    def list_artifacts(self, project_id: str, kind: str) -> list[dict[str, Any]]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            return self._dump(self.store.list_datasets(project_id))
        if artifact_kind == "features":
            return self._dump(self.store.list_features(project_id))
        if artifact_kind == "embeddings":
            return self._dump(self.store.list_embeddings(project_id))
        return self._dump(self.store.list_models(project_id))

    def get_artifact(self, project_id: str, kind: str, artifact_id: str) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            return self._dump(self.store.read_dataset(project_id, artifact_id))
        if artifact_kind == "features":
            return self._dump(self.store.read_feature(project_id, artifact_id))
        if artifact_kind == "embeddings":
            return self._dump(self.store.read_embedding(project_id, artifact_id))
        return self._dump(self.store.read_model(project_id, artifact_id))

    def configure_dataset(self, project_id: str, source_catalog_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = DatasetCreateRequest(
            catalog_id=source_catalog_id,
            params=DatasetPrepareParams.model_validate(params or {}),
        )
        return self._dump(self.store.create_dataset_from_library(project_id, request))

    def configure_feature(
        self,
        project_id: str,
        source_dataset_id: str,
        source_feature_id: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = FeatureCreateRequest(
            source_dataset_id=source_dataset_id,
            source_feature_id=source_feature_id,
            params=FeatureCreateParams.model_validate(params or {}),
        )
        return self._dump(self.store.create_feature(project_id, request))

    def configure_embedding(
        self,
        project_id: str,
        source_embedding_id: str,
        source_feature_ids: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = EmbeddingCreateRequest(
            source_embedding_id=source_embedding_id,
            source_feature_ids=source_feature_ids,
            params=EmbeddingCreateParams.model_validate(params or {}),
        )
        return self._dump(self.store.create_embedding(project_id, request))

    def configure_model(
        self,
        project_id: str,
        source_model_id: str,
        source_embedding_ids: list[str],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request = ModelCreateRequest(
            source_model_id=source_model_id,
            source_embedding_ids=source_embedding_ids,
            params=ModelCreateParams.model_validate(params),
        )
        return self._dump(self.store.create_model(project_id, request))

    def run_artifacts(self, project_id: str, kind: str, ids: list[str]) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        if not ids:
            raise ValueError("At least one artifact ID is required")
        if artifact_kind == "datasets":
            jobs = [self.store.run_dataset_preparation(project_id, artifact_id) for artifact_id in ids]
        elif artifact_kind == "features":
            jobs = [self.store.run_feature_batch(project_id, FeatureRunBatchRequest(feature_ids=ids))]
        elif artifact_kind == "embeddings":
            jobs = [self.store.run_embedding_batch(project_id, EmbeddingRunBatchRequest(embedding_ids=ids))]
        else:
            jobs = [self.store.run_model_batch(project_id, ModelRunBatchRequest(model_ids=ids))]
        return {"jobs": self._dump(jobs)}

    def list_jobs(self, project_id: str) -> list[dict[str, Any]]:
        return self._dump(self.store.list_jobs(project_id))

    def get_job(self, project_id: str, job_id: str) -> dict[str, Any]:
        return self._dump(self.store.read_job(project_id, job_id))

    def preview_artifact(
        self,
        project_id: str,
        kind: str,
        artifact_id: str,
        table: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            return self._dump(self.store.preview_dataset(project_id, artifact_id, table or "nodes", limit=limit, offset=offset))
        if artifact_kind == "features":
            return self._dump(self.store.preview_feature(project_id, artifact_id, limit=limit, offset=offset))
        if artifact_kind == "embeddings":
            return self._dump(self.store.preview_embedding(project_id, artifact_id, limit=limit, offset=offset))
        return self._dump(self.store.preview_model(project_id, artifact_id))

    def analyze_artifact(
        self,
        project_id: str,
        kind: str,
        artifact_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        opts = options or {}
        if artifact_kind == "datasets":
            return self._dump(
                self.store.analyze_dataset(
                    project_id,
                    artifact_id,
                    graph_id=opts.get("graph_id"),
                    max_nodes=int(opts.get("max_nodes", 150)),
                    max_edges=int(opts.get("max_edges", 300)),
                )
            )
        if artifact_kind == "features":
            return self._dump(
                self.store.analyze_feature(
                    project_id,
                    artifact_id,
                    max_fit_rows=int(opts.get("max_fit_rows", FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS)),
                    max_points=int(opts.get("max_points", FEATURE_ANALYSIS_DEFAULT_MAX_POINTS)),
                )
            )
        if artifact_kind == "embeddings":
            return self._dump(
                self.store.analyze_embedding(
                    project_id,
                    artifact_id,
                    max_fit_rows=int(opts.get("max_fit_rows", EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS)),
                    max_points=int(opts.get("max_points", EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS)),
                )
            )
        return self._dump(self.store.analyze_model(project_id, artifact_id))

    def search_graphs(self, project_id: str, kind: str, artifact_id: str, query: str, limit: int = 25) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            return self._dump(self.store.search_dataset_graphs(project_id, artifact_id, query=query, limit=limit))
        if artifact_kind == "features":
            return self._dump(self.store.search_feature_graphs(project_id, artifact_id, query=query, limit=limit))
        if artifact_kind == "embeddings":
            return self._dump(self.store.search_embedding_graphs(project_id, artifact_id, query=query, limit=limit))
        raise ValueError("Graph search is available for datasets, features, and embeddings")

    def get_graph_detail(
        self,
        project_id: str,
        kind: str,
        artifact_id: str,
        graph_id: str,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind)
        if artifact_kind == "datasets":
            if node_id:
                return self._dump(self.store.dataset_node_detail(project_id, artifact_id, graph_id=graph_id, node_id=node_id))
            analysis = self.store.analyze_dataset(project_id, artifact_id, graph_id=graph_id)
            return self._dump(
                {
                    "graph_id": analysis.selected_graph_id,
                    "summary": next((item for item in analysis.graph_summaries if item.graph_id == analysis.selected_graph_id), None),
                    "visual": analysis.visual,
                }
            )
        if artifact_kind == "features":
            return self._dump(self.store.feature_graph_detail(project_id, artifact_id, graph_id=graph_id))
        if artifact_kind == "embeddings":
            return self._dump(self.store.embedding_graph_detail(project_id, artifact_id, graph_id=graph_id))
        raise ValueError("Graph detail is available for datasets, features, and embeddings")

    def all_project_artifacts(self, project_id: str) -> dict[str, Any]:
        self.store.read_project(project_id)
        return {
            "datasets": self.list_artifacts(project_id, "datasets"),
            "features": self.list_artifacts(project_id, "features"),
            "embeddings": self.list_artifacts(project_id, "embeddings"),
            "models": self.list_artifacts(project_id, "models"),
        }
