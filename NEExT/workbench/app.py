"""FastAPI application for the local NEExT Workbench."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from .mcp_server import (
    MCP_PROTOCOL_VERSION,
    MCP_TRANSPORT,
    capability_summary,
    create_mcp_token_verifier,
    create_streamable_http_session_manager,
    evaluate_stdio_readiness,
    require_sdk_streamable_http,
    sdk_streamable_http_available,
    stdio_launch_command,
)
from .mcp_service import WorkbenchMcpService
from .paths import resolve_workspace_path
from .schemas import (
    ArtifactDeletionPlan,
    ArtifactDeletionSummary,
    CustomFeatureCreateRequest,
    CustomFeatureValidateRequest,
    CustomFeatureValidationResponse,
    DatasetAnalysis,
    DatasetCreateRequest,
    DatasetGraphSearchResponse,
    DatasetNodeDetail,
    EmbeddingAnalysis,
    EmbeddingCreateRequest,
    EmbeddingGraphDetail,
    EmbeddingGraphSearchResponse,
    EmbeddingRunBatchRequest,
    FeatureAnalysis,
    FeatureCreateRequest,
    FeatureGraphDetail,
    FeatureGraphSearchResponse,
    FeatureRunBatchRequest,
    McpActivityListing,
    McpApprovalDecision,
    McpApprovalListing,
    McpApprovalRequest,
    McpClientConfigSnippet,
    McpSettingsManifest,
    McpSettingsResponse,
    McpStdioReadiness,
    McpUiState,
    ModelAnalysis,
    ModelCreateRequest,
    ModelRunBatchRequest,
    ProjectCreate,
    RestoreSummary,
    TrashListing,
    WorkspaceInfo,
)
from .storage import (
    EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
    EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS,
    FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
    FEATURE_ANALYSIS_DEFAULT_MAX_POINTS,
    WorkbenchConflictError,
    WorkbenchStore,
)


def create_app(workspace_path: Optional[Union[str, Path]] = None):
    """Create the FastAPI app.

    FastAPI is imported inside this function so core NEExT imports remain free of
    optional workbench dependencies.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    store = WorkbenchStore(resolve_workspace_path(workspace_path))
    mcp_service = WorkbenchMcpService(store)
    sdk_transport_available = sdk_streamable_http_available()
    mcp_session_manager = create_streamable_http_session_manager(mcp_service) if sdk_transport_available else None

    @asynccontextmanager
    async def lifespan(app):
        if mcp_session_manager is None:
            yield
            return
        async with mcp_session_manager.run():
            yield

    app = FastAPI(title="NEExT Workbench", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    mcp_endpoint_url = "http://127.0.0.1:8765/mcp"

    if mcp_session_manager is not None:
        from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
        from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
        from starlette.middleware.authentication import AuthenticationMiddleware
        from starlette.routing import Route

        app.add_middleware(AuthContextMiddleware)
        app.add_middleware(AuthenticationMiddleware, backend=BearerAuthBackend(create_mcp_token_verifier(store)))
        app.router.routes.append(
            Route(
                "/mcp",
                endpoint=RequireAuthMiddleware(mcp_session_manager.handle_request, required_scopes=[]),
                methods=["GET", "POST", "DELETE"],
            )
        )
    else:

        @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
        async def missing_mcp_sdk():
            try:
                require_sdk_streamable_http()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(status_code=503, detail="SDK-backed Workbench MCP is unavailable.")

    def api_exception(exc: Exception) -> HTTPException:
        if isinstance(exc, FileNotFoundError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, WorkbenchConflictError):
            return HTTPException(status_code=409, detail=exc.detail)
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    def mcp_client_config_snippets(
        settings: McpSettingsManifest,
        one_time_token: Optional[str],
        stdio_readiness: McpStdioReadiness,
    ) -> list[McpClientConfigSnippet]:
        if not settings.enabled:
            return []

        token_value = one_time_token or "<regenerate-token-in-workbench-settings>"
        http_server_config = {
            "type": "streamable-http",
            "url": mcp_endpoint_url,
            "headers": {"Authorization": f"Bearer {token_value}"},
        }
        http_config_json = json.dumps({"mcpServers": {"neext-workbench": http_server_config}}, indent=2)

        snippets: list[McpClientConfigSnippet] = []

        # Claude Desktop can only reach a local server over stdio, and the launch
        # command must run from the exact interpreter hosting Workbench. Only emit
        # the snippet when that environment can actually host the server; otherwise
        # the UI surfaces stdio_readiness remediation instead of a broken config.
        if stdio_readiness.ok:
            command = stdio_launch_command(store.workspace_path)
            claude_desktop_config = {
                "command": command[0],
                "args": command[1:],
                "env": {
                    "NEEXT_WORKBENCH_MCP_TOKEN": token_value,
                },
            }
            claude_desktop_json = json.dumps({"mcpServers": {"neext-workbench": claude_desktop_config}}, indent=2)
            snippets.append(
                McpClientConfigSnippet(
                    client="claude_desktop",
                    label="Claude Desktop local",
                    target="claude_desktop_config.json",
                    content=claude_desktop_json,
                )
            )

        snippets.extend(
            [
                McpClientConfigSnippet(
                    client="claude_code",
                    label="Claude Code",
                    target="CLI",
                    content=f"claude mcp add --transport http neext-workbench {mcp_endpoint_url} --header 'Authorization: Bearer {token_value}'",
                ),
                McpClientConfigSnippet(
                    client="cursor",
                    label="Cursor",
                    target=".cursor/mcp.json",
                    content=http_config_json,
                ),
                McpClientConfigSnippet(
                    client="generic",
                    label="Generic Streamable HTTP",
                    target="mcp.json",
                    content=http_config_json,
                ),
            ]
        )
        return snippets

    def mcp_settings_response(settings: McpSettingsManifest, one_time_token: Optional[str] = None) -> McpSettingsResponse:
        created_at = settings.created_at if settings.created_at else None
        updated_at = settings.updated_at if settings.updated_at else None
        stdio_readiness = evaluate_stdio_readiness(store.workspace_path)
        return McpSettingsResponse(
            enabled=settings.enabled,
            token_preview=settings.token_preview,
            created_at=created_at,
            updated_at=updated_at,
            one_time_token=one_time_token,
            endpoint_url=mcp_endpoint_url if settings.enabled else None,
            transport=MCP_TRANSPORT,
            protocol_version=MCP_PROTOCOL_VERSION,
            sdk_transport_available=sdk_transport_available,
            scopes=settings.scopes,
            capabilities=capability_summary(settings.scopes),
            client_configs=mcp_client_config_snippets(settings, one_time_token, stdio_readiness),
            stdio_readiness=stdio_readiness,
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/workspace", response_model=WorkspaceInfo)
    def workspace() -> WorkspaceInfo:
        meta = store.read_workspace_meta()
        return WorkspaceInfo(
            path=str(store.workspace_path),
            version=str(meta.get("schema_version", meta.get("version", "1"))),
            projects=len(store.list_projects()),
        )

    @app.get("/api/mcp-settings", response_model=McpSettingsResponse)
    def get_mcp_settings() -> McpSettingsResponse:
        try:
            return mcp_settings_response(store.read_mcp_settings())
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/mcp-settings/enable", response_model=McpSettingsResponse)
    def enable_mcp_settings() -> McpSettingsResponse:
        try:
            settings, token = store.enable_mcp()
            return mcp_settings_response(settings, one_time_token=token)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/mcp-settings/regenerate", response_model=McpSettingsResponse)
    def regenerate_mcp_settings() -> McpSettingsResponse:
        try:
            settings, token = store.regenerate_mcp_token()
            return mcp_settings_response(settings, one_time_token=token)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/mcp-settings/disable", response_model=McpSettingsResponse)
    def disable_mcp_settings() -> McpSettingsResponse:
        try:
            return mcp_settings_response(store.disable_mcp())
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/mcp-activity", response_model=McpActivityListing)
    def list_mcp_activity(limit: int = 50) -> McpActivityListing:
        try:
            return store.list_mcp_activity(limit=limit)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/mcp-ui-state", response_model=McpUiState)
    def get_mcp_ui_state() -> McpUiState:
        try:
            return store.read_mcp_ui_state()
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/mcp-approvals", response_model=McpApprovalListing)
    def list_mcp_approvals() -> McpApprovalListing:
        try:
            return store.list_mcp_approvals()
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/mcp-approvals/{request_id}/approve", response_model=McpApprovalRequest)
    def approve_mcp_request(request_id: str) -> McpApprovalRequest:
        try:
            return store.approve_mcp_request(request_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/mcp-approvals/{request_id}/deny", response_model=McpApprovalRequest)
    def deny_mcp_request(request_id: str, decision: McpApprovalDecision) -> McpApprovalRequest:
        try:
            return store.deny_mcp_request(request_id, decision.reason)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects")
    def list_projects():
        return store.list_projects()

    @app.get("/api/dataset-library")
    def dataset_library():
        return store.list_dataset_catalog()

    @app.get("/api/feature-library")
    def feature_library():
        return store.list_feature_catalog()

    @app.get("/api/embedding-library")
    def embedding_library():
        return store.list_embedding_catalog()

    @app.get("/api/model-library")
    def model_library():
        return store.list_model_catalog()

    @app.post("/api/projects")
    def create_project(request: ProjectCreate):
        try:
            return store.create_project(request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            return store.read_project(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str):
        try:
            return store.delete_project(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/trash", response_model=TrashListing)
    def list_trash() -> TrashListing:
        try:
            return store.list_trash()
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/trash/projects/{trash_id}/restore", response_model=RestoreSummary)
    def restore_project(trash_id: str) -> RestoreSummary:
        try:
            return store.restore_project(trash_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/trash/artifact-deletions/{bundle_id}/restore", response_model=RestoreSummary)
    def restore_artifact_deletion(project_id: str, bundle_id: str) -> RestoreSummary:
        try:
            return store.restore_artifact_deletion(project_id, bundle_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets")
    def list_project_datasets(project_id: str):
        try:
            return store.list_datasets(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/datasets")
    def create_project_dataset(project_id: str, request: DatasetCreateRequest):
        try:
            return store.create_dataset_from_library(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}")
    def get_project_dataset(project_id: str, dataset_id: str):
        try:
            return store.read_dataset(project_id, dataset_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/delete-plan", response_model=ArtifactDeletionPlan)
    def plan_project_dataset_delete(project_id: str, dataset_id: str) -> ArtifactDeletionPlan:
        try:
            return store.plan_artifact_deletion(project_id, "dataset", dataset_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.delete("/api/projects/{project_id}/datasets/{dataset_id}", response_model=ArtifactDeletionSummary)
    def delete_project_dataset(project_id: str, dataset_id: str, cascade: bool = False) -> ArtifactDeletionSummary:
        try:
            return store.delete_artifact(project_id, "dataset", dataset_id, cascade=cascade)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/datasets/{dataset_id}/run")
    def run_project_dataset(project_id: str, dataset_id: str):
        try:
            return store.run_dataset_preparation(project_id, dataset_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/preview/{table}")
    def preview_project_dataset(project_id: str, dataset_id: str, table: str, limit: int = 20, offset: int = 0):
        try:
            return store.preview_dataset(project_id, dataset_id, table, limit=limit, offset=offset)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/export/{table}")
    def export_project_dataset(project_id: str, dataset_id: str, table: str):
        try:
            filename, content = store.export_dataset_csv(project_id, dataset_id, table)
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/analysis", response_model=DatasetAnalysis, response_model_exclude_none=True)
    def analyze_project_dataset(
        project_id: str,
        dataset_id: str,
        graph_id: Optional[str] = None,
        max_nodes: int = 150,
        max_edges: int = 300,
    ) -> DatasetAnalysis:
        try:
            return store.analyze_dataset(project_id, dataset_id, graph_id=graph_id, max_nodes=max_nodes, max_edges=max_edges)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/analysis/search", response_model=DatasetGraphSearchResponse)
    def search_project_dataset_graphs(project_id: str, dataset_id: str, query: str = "", limit: int = 25) -> DatasetGraphSearchResponse:
        try:
            return store.search_dataset_graphs(project_id, dataset_id, query=query, limit=limit)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/analysis/node", response_model=DatasetNodeDetail)
    def inspect_project_dataset_node(project_id: str, dataset_id: str, graph_id: str, node_id: str) -> DatasetNodeDetail:
        try:
            return store.dataset_node_detail(project_id, dataset_id, graph_id=graph_id, node_id=node_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features")
    def list_project_features(project_id: str):
        try:
            return store.list_features(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/features")
    def create_project_feature(project_id: str, request: FeatureCreateRequest):
        try:
            return store.create_feature(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/features/custom")
    def create_project_custom_feature(project_id: str, request: CustomFeatureCreateRequest):
        try:
            return store.create_custom_feature(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/features/custom/validate", response_model=CustomFeatureValidationResponse)
    def validate_project_custom_feature(project_id: str, request: CustomFeatureValidateRequest):
        try:
            return store.validate_custom_feature(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}")
    def get_project_feature(project_id: str, feature_id: str):
        try:
            return store.read_feature(project_id, feature_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}/delete-plan", response_model=ArtifactDeletionPlan)
    def plan_project_feature_delete(project_id: str, feature_id: str) -> ArtifactDeletionPlan:
        try:
            return store.plan_artifact_deletion(project_id, "feature", feature_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.delete("/api/projects/{project_id}/features/{feature_id}", response_model=ArtifactDeletionSummary)
    def delete_project_feature(project_id: str, feature_id: str, cascade: bool = False) -> ArtifactDeletionSummary:
        try:
            return store.delete_artifact(project_id, "feature", feature_id, cascade=cascade)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/features/{feature_id}/run")
    def run_project_feature(project_id: str, feature_id: str):
        try:
            return store.run_feature(project_id, feature_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/features/run-batch")
    def run_project_features(project_id: str, request: FeatureRunBatchRequest):
        try:
            return store.run_feature_batch(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}/preview")
    def preview_project_feature(project_id: str, feature_id: str, limit: int = 20, offset: int = 0):
        try:
            return store.preview_feature(project_id, feature_id, limit=limit, offset=offset)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}/analysis", response_model=FeatureAnalysis)
    def analyze_project_feature(
        project_id: str,
        feature_id: str,
        max_fit_rows: int = FEATURE_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
        max_points: int = FEATURE_ANALYSIS_DEFAULT_MAX_POINTS,
    ) -> FeatureAnalysis:
        try:
            return store.analyze_feature(project_id, feature_id, max_fit_rows=max_fit_rows, max_points=max_points)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}/analysis/search", response_model=FeatureGraphSearchResponse)
    def search_project_feature_graphs(project_id: str, feature_id: str, query: str = "", limit: int = 25) -> FeatureGraphSearchResponse:
        try:
            return store.search_feature_graphs(project_id, feature_id, query=query, limit=limit)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/features/{feature_id}/analysis/graph", response_model=FeatureGraphDetail)
    def inspect_project_feature_graph(project_id: str, feature_id: str, graph_id: str) -> FeatureGraphDetail:
        try:
            return store.feature_graph_detail(project_id, feature_id, graph_id=graph_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings")
    def list_project_embeddings(project_id: str):
        try:
            return store.list_embeddings(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/embeddings")
    def create_project_embedding(project_id: str, request: EmbeddingCreateRequest):
        try:
            return store.create_embedding(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}")
    def get_project_embedding(project_id: str, embedding_id: str):
        try:
            return store.read_embedding(project_id, embedding_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}/delete-plan", response_model=ArtifactDeletionPlan)
    def plan_project_embedding_delete(project_id: str, embedding_id: str) -> ArtifactDeletionPlan:
        try:
            return store.plan_artifact_deletion(project_id, "embedding", embedding_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.delete("/api/projects/{project_id}/embeddings/{embedding_id}", response_model=ArtifactDeletionSummary)
    def delete_project_embedding(project_id: str, embedding_id: str, cascade: bool = False) -> ArtifactDeletionSummary:
        try:
            return store.delete_artifact(project_id, "embedding", embedding_id, cascade=cascade)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/embeddings/{embedding_id}/run")
    def run_project_embedding(project_id: str, embedding_id: str):
        try:
            return store.run_embedding(project_id, embedding_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/embeddings/run-batch")
    def run_project_embeddings(project_id: str, request: EmbeddingRunBatchRequest):
        try:
            return store.run_embedding_batch(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}/preview")
    def preview_project_embedding(project_id: str, embedding_id: str, limit: int = 20, offset: int = 0):
        try:
            return store.preview_embedding(project_id, embedding_id, limit=limit, offset=offset)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}/analysis", response_model=EmbeddingAnalysis)
    def analyze_project_embedding(
        project_id: str,
        embedding_id: str,
        max_fit_rows: int = EMBEDDING_ANALYSIS_DEFAULT_MAX_FIT_ROWS,
        max_points: int = EMBEDDING_ANALYSIS_DEFAULT_MAX_POINTS,
    ) -> EmbeddingAnalysis:
        try:
            return store.analyze_embedding(project_id, embedding_id, max_fit_rows=max_fit_rows, max_points=max_points)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}/analysis/search", response_model=EmbeddingGraphSearchResponse)
    def search_project_embedding_graphs(project_id: str, embedding_id: str, query: str = "", limit: int = 25) -> EmbeddingGraphSearchResponse:
        try:
            return store.search_embedding_graphs(project_id, embedding_id, query=query, limit=limit)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/embeddings/{embedding_id}/analysis/graph", response_model=EmbeddingGraphDetail)
    def inspect_project_embedding_graph(project_id: str, embedding_id: str, graph_id: str) -> EmbeddingGraphDetail:
        try:
            return store.embedding_graph_detail(project_id, embedding_id, graph_id=graph_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/models")
    def list_project_models(project_id: str):
        try:
            return store.list_models(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/models")
    def create_project_model(project_id: str, request: ModelCreateRequest):
        try:
            return store.create_model(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/models/{model_id}")
    def get_project_model(project_id: str, model_id: str):
        try:
            return store.read_model(project_id, model_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/models/{model_id}/delete-plan", response_model=ArtifactDeletionPlan)
    def plan_project_model_delete(project_id: str, model_id: str) -> ArtifactDeletionPlan:
        try:
            return store.plan_artifact_deletion(project_id, "model", model_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.delete("/api/projects/{project_id}/models/{model_id}", response_model=ArtifactDeletionSummary)
    def delete_project_model(project_id: str, model_id: str, cascade: bool = False) -> ArtifactDeletionSummary:
        try:
            return store.delete_artifact(project_id, "model", model_id, cascade=cascade)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/models/{model_id}/run")
    def run_project_model(project_id: str, model_id: str):
        try:
            return store.run_model(project_id, model_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.post("/api/projects/{project_id}/models/run-batch")
    def run_project_models(project_id: str, request: ModelRunBatchRequest):
        try:
            return store.run_model_batch(project_id, request)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/models/{model_id}/preview")
    def preview_project_model(project_id: str, model_id: str):
        try:
            return store.preview_model(project_id, model_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/models/{model_id}/analysis", response_model=ModelAnalysis)
    def analyze_project_model(project_id: str, model_id: str) -> ModelAnalysis:
        try:
            return store.analyze_model(project_id, model_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/jobs")
    def list_project_jobs(project_id: str):
        try:
            return store.list_jobs(project_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    @app.get("/api/projects/{project_id}/jobs/{job_id}")
    def get_project_job(project_id: str, job_id: str):
        try:
            return store.read_job(project_id, job_id)
        except Exception as exc:
            raise api_exception(exc) from exc

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="workbench-static")
    else:

        @app.get("/")
        def missing_static():
            return JSONResponse(
                {
                    "message": "NEExT Workbench API is running, but frontend assets are not built.",
                    "build": "Run: cd workbench-ui && npm install && npm run build",
                }
            )

    return app
