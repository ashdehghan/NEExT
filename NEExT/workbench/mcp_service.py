"""Curated MCP service methods for the local NEExT Workbench workspace."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .schemas import (
    CustomFeatureCreateParams,
    CustomFeatureCreateRequest,
    CustomFeatureValidateRequest,
    DatasetCreateRequest,
    DatasetIntakeRequest,
    DatasetIntakeSessionCreateRequest,
    DatasetIntakeSessionTableRequest,
    DatasetIntakeTablePayload,
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

MCP_ROUTE_TOP_TABS = {"home", "datasets", "features", "embeddings", "models"}
MCP_ROUTE_COMMANDS = {
    "import",
    "create",
    "projects",
    "trash",
    "settings",
    "library",
    "explore",
    "datasets",
    "features",
    "embeddings",
    "models",
}
MCP_ROUTE_ARTIFACT_KINDS = {"dataset", "feature", "embedding", "model"}
MCP_ROUTE_KEYS = {
    "top_tab",
    "command",
    "project_id",
    "artifact_kind",
    "artifact_id",
    "catalog_kind",
    "catalog_id",
    "graph_id",
    "node_id",
    "draft",
}
MCP_DRAFT_MAX_KEYS = 32


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
        result = self.store.create_project(ProjectCreate(name=name, description=description))
        self.store.record_mcp_activity(
            "tool_call", f'Created project "{result.name}"', tool_name="neext_create_project", details={"project_id": result.id}
        )
        return self._dump(result)

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
        result = self.store.create_dataset_from_library(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Added dataset "{result.name}" to project',
            tool_name="neext_configure_dataset",
            details={"project_id": project_id, "dataset_id": result.id},
        )
        return self._dump(result)

    def _dataset_intake_tables(self, tables: dict[str, Any]) -> dict[str, DatasetIntakeTablePayload]:
        return {str(name): DatasetIntakeTablePayload.model_validate(payload) for name, payload in tables.items()}

    def validate_dataset_intake(
        self,
        project_id: str,
        name: str,
        tables: dict[str, Any],
        description: str = "",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = DatasetIntakeRequest(
            name=name,
            description=description,
            tables=self._dataset_intake_tables(tables),
            params=DatasetPrepareParams.model_validate(params or {}),
        )
        result = self.store.validate_dataset_intake(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            "Validated Dataset Intake tables",
            tool_name="neext_validate_dataset_intake",
            details={"project_id": project_id, "valid": result.valid, "tables": sorted(tables)},
        )
        return self._dump(result)

    def create_dataset_intake_session(
        self,
        project_id: str,
        name: str,
        description: str = "",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = DatasetIntakeSessionCreateRequest(
            name=name,
            description=description,
            params=DatasetPrepareParams.model_validate(params or {}),
        )
        result = self.store.create_dataset_intake_session(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Created Dataset Intake session "{result.name}"',
            tool_name="neext_create_dataset_intake_session",
            details={"project_id": project_id, "session_id": result.id},
        )
        return self._dump(result)

    def append_dataset_intake_table(
        self,
        project_id: str,
        session_id: str,
        table_name: str,
        table: dict[str, Any],
        replace: bool = False,
    ) -> dict[str, Any]:
        request = DatasetIntakeSessionTableRequest(
            table=DatasetIntakeTablePayload.model_validate(table),
            replace=replace,
        )
        result = self.store.append_dataset_intake_session_table(project_id, session_id, table_name, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Appended Dataset Intake table "{table_name}"',
            tool_name="neext_append_dataset_intake_table",
            details={"project_id": project_id, "session_id": session_id, "table": table_name, "replace": replace},
        )
        return self._dump(result)

    def validate_dataset_intake_session(self, project_id: str, session_id: str) -> dict[str, Any]:
        result = self.store.validate_dataset_intake_session(project_id, session_id)
        self.store.record_mcp_activity(
            "tool_call",
            "Validated Dataset Intake session",
            tool_name="neext_validate_dataset_intake_session",
            details={"project_id": project_id, "session_id": session_id, "valid": result.validation.valid if result.validation else False},
        )
        return self._dump(result)

    def create_dataset_from_intake(self, project_id: str, session_id: str) -> dict[str, Any]:
        result = self.store.create_dataset_from_intake_session(project_id, session_id)
        self.store.record_mcp_activity(
            "tool_call",
            f'Created Dataset artifact "{result.name}" from Dataset Intake',
            tool_name="neext_create_dataset_from_intake",
            details={"project_id": project_id, "session_id": session_id, "dataset_id": result.id},
        )
        return self._dump(result)

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
        result = self.store.create_feature(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Added feature "{result.name}" to project',
            tool_name="neext_configure_feature",
            details={"project_id": project_id, "feature_id": result.id},
        )
        return self._dump(result)

    def validate_custom_feature(
        self,
        project_id: str,
        source_dataset_id: str,
        code: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = CustomFeatureValidateRequest(
            source_dataset_id=source_dataset_id,
            code=code,
            params=CustomFeatureCreateParams.model_validate(params or {}),
        )
        result = self.store.validate_custom_feature(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            "Validated custom feature code",
            tool_name="neext_validate_custom_feature",
            details={"project_id": project_id, "source_dataset_id": source_dataset_id, "columns": result.columns},
        )
        return self._dump(result)

    def configure_custom_feature(
        self,
        project_id: str,
        source_dataset_id: str,
        name: str,
        description: str,
        code: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = CustomFeatureCreateRequest(
            source_dataset_id=source_dataset_id,
            name=name,
            description=description,
            code=code,
            params=CustomFeatureCreateParams.model_validate(params or {}),
        )
        result = self.store.create_custom_feature(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Created custom feature "{result.name}"',
            tool_name="neext_configure_custom_feature",
            details={"project_id": project_id, "feature_id": result.id, "source_dataset_id": source_dataset_id},
        )
        return self._dump(result)

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
        result = self.store.create_embedding(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Added embedding "{result.name}" to project',
            tool_name="neext_configure_embedding",
            details={"project_id": project_id, "embedding_id": result.id},
        )
        return self._dump(result)

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
        result = self.store.create_model(project_id, request)
        self.store.record_mcp_activity(
            "tool_call",
            f'Added model "{result.name}" to project',
            tool_name="neext_configure_model",
            details={"project_id": project_id, "model_id": result.id},
        )
        return self._dump(result)

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
        payload = {"jobs": self._dump(jobs)}
        self.store.record_mcp_activity(
            "tool_call",
            f"Started {artifact_kind} run for {len(ids)} artifact(s)",
            tool_name="neext_run_artifacts",
            details={"project_id": project_id, "kind": artifact_kind, "ids": ids, "job_ids": [job.id for job in jobs]},
        )
        return payload

    def compute_model_feature_importance(
        self, project_id: str, model_id: str, algorithm: str = "supervised_fast", n_iterations: int = 3
    ) -> dict[str, Any]:
        job = self.store.run_model_feature_importance(
            project_id, model_id, {"algorithm": algorithm, "n_iterations": n_iterations}
        )
        self.store.record_mcp_activity(
            "tool_call",
            "Started feature importance for model",
            tool_name="neext_compute_feature_importance",
            details={"project_id": project_id, "model_id": model_id, "job_id": job.id},
        )
        return self._dump(job)

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

    def export_dataset_table(self, project_id: str, dataset_id: str, table: str) -> dict[str, Any]:
        filename, content = self.store.export_dataset_csv(project_id, dataset_id, table)
        self.store.record_mcp_activity(
            "tool_call",
            f'Exported dataset table "{table}"',
            tool_name="neext_export_dataset_table",
            details={"project_id": project_id, "dataset_id": dataset_id, "table": table, "filename": filename},
        )
        return {"filename": filename, "content_type": "text/csv", "content": content}

    def export_model_metrics(self, project_id: str, model_id: str) -> dict[str, Any]:
        filename, content = self.store.export_model_metrics_csv(project_id, model_id)
        self.store.record_mcp_activity(
            "tool_call",
            "Exported model metrics",
            tool_name="neext_export_model_metrics",
            details={"project_id": project_id, "model_id": model_id, "filename": filename},
        )
        return {"filename": filename, "content_type": "text/csv", "content": content}

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
            cluster_k = opts.get("cluster_k")
            perplexity = opts.get("perplexity")
            n_neighbors = opts.get("n_neighbors")
            min_dist = opts.get("min_dist")
            return self._dump(
                self.store.analyze_embedding(
                    project_id,
                    artifact_id,
                    max_fit_rows=int(opts.get("max_fit_rows", EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS)),
                    max_points=int(opts.get("max_points", EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS)),
                    cluster_k=int(cluster_k) if cluster_k is not None else None,
                    projection_method=str(opts.get("projection_method", "pca")),
                    perplexity=float(perplexity) if perplexity is not None else None,
                    n_neighbors=int(n_neighbors) if n_neighbors is not None else None,
                    min_dist=float(min_dist) if min_dist is not None else None,
                )
            )
        return self._dump(self.store.analyze_model(project_id, artifact_id))

    def cluster_embedding(self, project_id: str, embedding_id: str, k: int) -> dict[str, Any]:
        """KMeans-cluster a completed embedding; return assignments + label overlap."""
        analysis = self.store.analyze_embedding(project_id, embedding_id, cluster_k=int(k))
        pca = analysis.pca
        assignments = [
            {"graph_id": point.graph_id, "cluster": point.cluster, "graph_label": point.graph_label}
            for point in pca.points
        ]
        return {
            "embedding_id": embedding_id,
            "available": pca.available,
            "cluster_k": pca.cluster_k,
            "cluster_algorithm": pca.cluster_algorithm,
            "cluster_silhouette": pca.cluster_silhouette,
            "cluster_label_ari": pca.cluster_label_ari,
            "cluster_purity": pca.cluster_purity,
            "clusters": [self._dump(summary) for summary in pca.clusters],
            "assignments": assignments,
        }

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

    def list_trash(self) -> dict[str, Any]:
        return self._dump(self.store.list_trash())

    def restore_project(self, trash_id: str) -> dict[str, Any]:
        result = self.store.restore_project(trash_id)
        self.store.record_mcp_activity(
            "tool_call",
            f'Restored project "{result.name}"',
            tool_name="neext_restore_project",
            details={"trash_id": trash_id, "project_id": result.id},
        )
        return self._dump(result)

    def restore_artifact_deletion(self, project_id: str, bundle_id: str) -> dict[str, Any]:
        result = self.store.restore_artifact_deletion(project_id, bundle_id)
        self.store.record_mcp_activity(
            "tool_call",
            f'Restored artifact deletion "{result.name}"',
            tool_name="neext_restore_artifact_deletion",
            details={"project_id": project_id, "bundle_id": bundle_id},
        )
        return self._dump(result)

    def plan_delete_artifact(self, project_id: str, kind: str, artifact_id: str) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind).removesuffix("s")
        return self._dump(self.store.plan_artifact_deletion(project_id, artifact_kind, artifact_id))

    def request_delete_artifact(self, project_id: str, kind: str, artifact_id: str, cascade: bool = False) -> dict[str, Any]:
        artifact_kind = self._artifact_kind(kind).removesuffix("s")
        return self._dump(self.store.request_artifact_delete_approval(project_id, artifact_kind, artifact_id, cascade=cascade))

    def request_delete_project(self, project_id: str) -> dict[str, Any]:
        return self._dump(self.store.request_project_delete_approval(project_id))

    def list_mcp_activity(self, limit: int = 50) -> dict[str, Any]:
        return self._dump(self.store.list_mcp_activity(limit=limit))

    def list_mcp_approvals(self) -> dict[str, Any]:
        return self._dump(self.store.list_mcp_approvals())

    def get_workbench_view(self) -> dict[str, Any]:
        return self._dump(self.store.read_mcp_ui_state())

    def set_workbench_view(self, route: dict[str, Any], message: str = "") -> dict[str, Any]:
        return self._dump(self.store.set_mcp_ui_state(self._validate_workbench_route(route), message))

    def _validate_workbench_route(self, route: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(route, dict):
            raise ValueError("Workbench route must be an object")
        unknown_keys = set(route) - MCP_ROUTE_KEYS
        if unknown_keys:
            raise ValueError(f"Unsupported Workbench route key(s): {', '.join(sorted(unknown_keys))}")

        cleaned: dict[str, Any] = {}
        for key, value in route.items():
            if value is None or value == "":
                continue
            if key == "draft":
                if not isinstance(value, dict):
                    raise ValueError("Workbench route draft must be an object")
                if len(value) > MCP_DRAFT_MAX_KEYS:
                    raise ValueError(f"Workbench route draft can include at most {MCP_DRAFT_MAX_KEYS} keys")
                cleaned[key] = value
                continue
            if not isinstance(value, str):
                raise ValueError(f"Workbench route field {key} must be a string")
            cleaned[key] = value

        top_tab = cleaned.get("top_tab")
        if top_tab and top_tab not in MCP_ROUTE_TOP_TABS:
            raise ValueError(f"Unsupported Workbench Space: {top_tab}")
        command = cleaned.get("command")
        if command and command not in MCP_ROUTE_COMMANDS:
            raise ValueError(f"Unsupported Workbench command: {command}")
        artifact_kind = cleaned.get("artifact_kind")
        if artifact_kind and artifact_kind not in MCP_ROUTE_ARTIFACT_KINDS:
            raise ValueError(f"Unsupported Workbench artifact kind: {artifact_kind}")
        catalog_kind = cleaned.get("catalog_kind")
        if catalog_kind and catalog_kind not in MCP_ROUTE_ARTIFACT_KINDS:
            raise ValueError(f"Unsupported Workbench catalog kind: {catalog_kind}")
        return cleaned
