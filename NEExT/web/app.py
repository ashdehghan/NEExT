"""FastAPI application for the optional NEExT local web workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from NEExT.web.jobs import JobManager
from NEExT.web.project import ProjectManager
from NEExT.web.serializers import collection_summary, dataframe_payload, graph_elements
from NEExT.web.service import WorkbenchService


class ImportDatasetRequest(BaseModel):
    name: str = "Imported dataset"
    edges_path: str
    node_graph_mapping_path: str
    graph_label_path: str | None = None
    node_features_path: str | None = None
    edge_features_path: str | None = None
    graph_type: str = "networkx"
    reindex_nodes: bool = True
    filter_largest_component: bool = True
    node_sample_rate: float = Field(default=1.0, gt=0.0, le=1.0)


class GenerateDatasetRequest(BaseModel):
    name: str = "Synthetic dataset"
    preset: str = "er_vs_ba"
    seed: int = 42
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureJobRequest(BaseModel):
    dataset_id: str
    name: str = "Node features"
    feature_list: list[str] = Field(default_factory=lambda: ["all"])
    feature_vector_length: int = 3
    normalize_features: bool = True
    n_jobs: int = 1
    parallel_backend: str = "loky"


class EmbeddingJobRequest(BaseModel):
    dataset_id: str
    features_id: str
    name: str = "Graph embeddings"
    embedding_algorithm: str = "approx_wasserstein"
    embedding_dimension: int = 3
    feature_columns: list[str] | None = None
    random_state: int = 42
    memory_size: str = "4G"


class ModelJobRequest(BaseModel):
    dataset_id: str
    embeddings_id: str
    name: str = "Model run"
    model_type: str = "classifier"
    balance_dataset: bool = False
    sample_size: int = 5
    n_jobs: int = -1
    parallel_backend: str = "process"


class ExportRequest(BaseModel):
    artifact_id: str
    name: str = "Python reproduction script"


def create_app(project_dir: Path) -> FastAPI:
    project = ProjectManager(Path(project_dir))
    service = WorkbenchService(project)
    jobs = JobManager(project)

    app = FastAPI(title="NEExT Local Workbench", version="0.1.0")
    app.state.project = project
    app.state.service = service
    app.state.jobs = jobs

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "project_dir": str(project.root)}

    @app.get("/api/project")
    def get_project() -> dict[str, Any]:
        return project.manifest()

    @app.get("/api/artifacts")
    def get_artifacts(artifact_type: str | None = None) -> dict[str, Any]:
        return {"artifacts": project.list_artifacts(artifact_type)}

    @app.post("/api/datasets/import")
    def import_dataset(request: ImportDatasetRequest) -> dict[str, Any]:
        try:
            artifact = service.import_dataset_from_paths(**request.model_dump())
            return {"artifact": artifact.__dict__}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/datasets/generate")
    def generate_dataset(request: GenerateDatasetRequest) -> dict[str, Any]:
        try:
            artifact = service.generate_preset_dataset(**request.model_dump())
            return {"artifact": artifact.__dict__}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/datasets/upload")
    async def upload_dataset_files(
        edges: UploadFile = File(...),
        node_graph_mapping: UploadFile = File(...),
        graph_labels: UploadFile | None = File(None),
        node_features: UploadFile | None = File(None),
        edge_features: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        upload_dir = project.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "edges_path": await _save_upload(upload_dir, edges),
            "node_graph_mapping_path": await _save_upload(upload_dir, node_graph_mapping),
            "graph_label_path": await _save_upload(upload_dir, graph_labels),
            "node_features_path": await _save_upload(upload_dir, node_features),
            "edge_features_path": await _save_upload(upload_dir, edge_features),
        }
        return files

    @app.get("/api/graphs/{dataset_id}")
    def get_graphs(dataset_id: str) -> dict[str, Any]:
        try:
            collection = project.load_object(dataset_id)
            return collection_summary(collection)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/graphs/{dataset_id}/{graph_id}/elements")
    def get_graph_elements(dataset_id: str, graph_id: int, max_nodes: int = 500) -> dict[str, Any]:
        try:
            collection = project.load_object(dataset_id)
            graph = collection.get_graph_by_id(graph_id)
            if graph is None:
                raise KeyError(f"Unknown graph id: {graph_id}")
            return graph_elements(graph, max_nodes=max_nodes)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/features")
    def create_feature_job(request: FeatureJobRequest) -> dict[str, Any]:
        payload = request.model_dump()

        def work(update):
            update(0.1, "Loading dataset artifact")
            update(0.25, "Computing node features")
            artifact = service.compute_features(**payload)
            update(0.9, f"Saved feature artifact {artifact.id}")
            return artifact.id

        job = jobs.submit("features", request.name, work, metadata=payload)
        return {"job": job.__dict__}

    @app.post("/api/jobs/embeddings")
    def create_embedding_job(request: EmbeddingJobRequest) -> dict[str, Any]:
        payload = request.model_dump()

        def work(update):
            update(0.1, "Loading dataset and feature artifacts")
            update(0.35, "Computing graph embeddings")
            artifact = service.compute_embeddings(**payload)
            update(0.9, f"Saved embedding artifact {artifact.id}")
            return artifact.id

        job = jobs.submit("embeddings", request.name, work, metadata=payload)
        return {"job": job.__dict__}

    @app.post("/api/jobs/models")
    def create_model_job(request: ModelJobRequest) -> dict[str, Any]:
        payload = request.model_dump()

        def work(update):
            update(0.1, "Loading dataset and embedding artifacts")
            update(0.35, "Training and evaluating model")
            artifact = service.train_model(**payload)
            update(0.9, f"Saved model run artifact {artifact.id}")
            return artifact.id

        job = jobs.submit("models", request.name, work, metadata=payload)
        return {"job": job.__dict__}

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": project.list_jobs()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": jobs.get(job_id).__dict__}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/artifacts/{artifact_id}/table")
    def get_table(artifact_id: str, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        try:
            return dataframe_payload(project.load_table(artifact_id), limit=limit, offset=offset)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/exports/python")
    def export_python(request: ExportRequest) -> dict[str, Any]:
        try:
            artifact = service.generate_python_export(name=request.name, artifact_id=request.artifact_id)
            return {"artifact": artifact.__dict__}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str) -> FileResponse:
        target = static_dir / path
        if path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(static_dir / "index.html")

    return app


async def _save_upload(upload_dir: Path, upload: UploadFile | None) -> str | None:
    if upload is None:
        return None
    destination = upload_dir / upload.filename
    with destination.open("wb") as handle:
        handle.write(await upload.read())
    return str(destination)
