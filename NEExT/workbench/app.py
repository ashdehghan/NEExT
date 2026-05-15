"""FastAPI application for the local NEExT Workbench."""

from pathlib import Path
from typing import Optional, Union

from .paths import resolve_workspace_path
from .schemas import (
    DatasetAnalysis,
    DatasetCreateRequest,
    DatasetGraphSearchResponse,
    DatasetNodeDetail,
    EmbeddingCreateRequest,
    EmbeddingRunBatchRequest,
    FeatureCreateRequest,
    FeatureRunBatchRequest,
    ModelCreateRequest,
    ModelRunBatchRequest,
    ProjectCreate,
    WorkspaceInfo,
)
from .storage import WorkbenchStore


def create_app(workspace_path: Optional[Union[str, Path]] = None):
    """Create the FastAPI app.

    FastAPI is imported inside this function so core NEExT imports remain free of
    optional workbench dependencies.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    store = WorkbenchStore(resolve_workspace_path(workspace_path))
    app = FastAPI(title="NEExT Workbench", version="0.1.0")
    app.state.store = store

    def api_exception(exc: Exception) -> HTTPException:
        if isinstance(exc, FileNotFoundError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

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

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}/analysis", response_model=DatasetAnalysis)
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

    @app.get("/api/projects/{project_id}/features/{feature_id}")
    def get_project_feature(project_id: str, feature_id: str):
        try:
            return store.read_feature(project_id, feature_id)
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
