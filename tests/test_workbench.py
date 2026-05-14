import json
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ARTIFACT_DIRS = [
    Path("artifacts/datasets"),
    Path("artifacts/features"),
    Path("artifacts/embeddings"),
    Path("artifacts/models"),
]


def assert_uuid4(value: str) -> None:
    parsed = uuid.UUID(value)
    assert str(parsed) == value
    assert parsed.version == 4


def write_dataset_source_bundle(source_dir: Path, *, invalid_edge: bool = False, include_isolated: bool = False) -> dict[str, str]:
    source_dir.mkdir(parents=True)
    isolated_node = "5,g1\n" if include_isolated else ""
    (source_dir / "node_graph_mapping.csv").write_text(
        "node_id,graph_id\n"
        "1,g1\n"
        "2,g1\n"
        "3,g2\n"
        "4,g2\n"
        + isolated_node,
        encoding="utf-8",
    )
    edge_dest = "99" if invalid_edge else "2"
    (source_dir / "edges.csv").write_text(
        "src_node_id,dest_node_id\n"
        f"1,{edge_dest}\n"
        "3,4\n",
        encoding="utf-8",
    )
    (source_dir / "graph_labels.csv").write_text(
        "graph_id,graph_label\n"
        "g1,0\n"
        "g2,1\n",
        encoding="utf-8",
    )
    isolated_feature = "5,0.5\n" if include_isolated else ""
    (source_dir / "node_features.csv").write_text(
        "node_id,feature_a\n"
        "1,0.1\n"
        "2,0.2\n"
        "3,0.3\n"
        "4,0.4\n"
        + isolated_feature,
        encoding="utf-8",
    )
    (source_dir / "edge_features.csv").write_text(
        "src_node_id,dest_node_id,edge_weight\n"
        "1,2,1.5\n"
        "3,4,2.5\n",
        encoding="utf-8",
    )
    return {
        "edges": str(source_dir / "edges.csv"),
        "node_graph_mapping": str(source_dir / "node_graph_mapping.csv"),
        "graph_labels": str(source_dir / "graph_labels.csv"),
        "node_features": str(source_dir / "node_features.csv"),
        "edge_features": str(source_dir / "edge_features.csv"),
    }


def wait_for_job(client, project_id: str, job_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/api/projects/{project_id}/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def test_workbench_health_and_workspace_load():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        workspace = client.get("/api/workspace")
        assert workspace.status_code == 200
        assert workspace.json() == {
            "path": str(Path(tmpdir).resolve()),
            "version": "1",
            "projects": 0,
        }

        workspace_manifest = json.loads((Path(tmpdir) / "workspace.json").read_text(encoding="utf-8"))
        assert workspace_manifest["schema_version"] == "1"
        assert workspace_manifest["manifest_type"] == "workspace"
        assert workspace_manifest["created_at"]
        assert workspace_manifest["updated_at"]


def test_existing_workspace_manifest_is_normalized():
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        workspace_file = Path(tmpdir) / "workspace.json"
        workspace_file.write_text(
            json.dumps(
                {
                    "version": "1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "defaults": {"graph_type": "igraph"},
                }
            ),
            encoding="utf-8",
        )

        WorkbenchStore(Path(tmpdir))
        workspace_manifest = json.loads(workspace_file.read_text(encoding="utf-8"))
        assert set(workspace_manifest) == {"schema_version", "manifest_type", "created_at", "updated_at"}
        assert workspace_manifest["schema_version"] == "1"
        assert workspace_manifest["manifest_type"] == "workspace"
        assert workspace_manifest["created_at"] == "2026-01-01T00:00:00+00:00"
        assert workspace_manifest["updated_at"]


def test_workbench_project_create_list_and_read_by_id():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        created = client.post("/api/projects", json={"name": "API Project", "description": "test"})
        assert created.status_code == 200
        project = created.json()
        project_id = project["id"]
        assert_uuid4(project_id)
        assert project["schema_version"] == "1"
        assert project["manifest_type"] == "project"
        assert project["name"] == "API Project"
        assert project["description"] == "test"
        assert project["created_at"]
        assert project["updated_at"]
        assert "path" not in project

        project_path = Path(tmpdir) / "projects" / project_id
        assert project_path.is_dir()
        assert (project_path / "project.json").is_file()
        for artifact_dir in ARTIFACT_DIRS:
            assert (project_path / artifact_dir).is_dir()

        projects = client.get("/api/projects")
        assert projects.status_code == 200
        assert [item["id"] for item in projects.json()] == [project_id]
        assert all("path" not in item for item in projects.json())

        read = client.get(f"/api/projects/{project_id}")
        assert read.status_code == 200
        assert read.json() == project
        assert "path" not in read.json()

        workspace = client.get("/api/workspace")
        assert workspace.status_code == 200
        assert workspace.json()["projects"] == 1


def test_project_creation_writes_project_foundation():
    from NEExT.workbench.schemas import ProjectCreate
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        store = WorkbenchStore(Path(tmpdir))
        project = store.create_project(ProjectCreate(name="Metadata Project", description="foundation"))
        project_path = store.project_path(project.id)

        assert_uuid4(project.id)
        assert project_path == Path(tmpdir).resolve() / "projects" / project.id
        assert sorted(path.name for path in project_path.iterdir()) == ["artifacts", "project.json"]
        for artifact_dir in ARTIFACT_DIRS:
            assert (project_path / artifact_dir).is_dir()

        payload = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
        assert payload == project.model_dump()
        assert payload["schema_version"] == "1"
        assert payload["manifest_type"] == "project"
        assert str(Path(tmpdir).resolve()) not in json.dumps(payload)
        assert store.read_project(project.id) == project


def test_dataset_library_catalog_endpoint_exposes_curated_metadata_without_paths():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        response = client.get("/api/dataset-library")

        assert response.status_code == 200
        catalog = response.json()
        assert {"MUTAG", "NCI1", "BZR", "PROTEINS", "IMDB"}.issubset({entry["id"] for entry in catalog})
        mutag = next(entry for entry in catalog if entry["id"] == "MUTAG")
        assert mutag["graph_count"] == 188
        assert mutag["node_count"] == 3371
        assert mutag["edge_count"] == 7442
        for entry in catalog:
            payload = json.dumps(entry)
            assert "files" not in entry
            assert "path" not in entry
            assert "http://" not in payload
            assert "https://" not in payload
            assert entry["source_type"] == "neext_csv_bundle"
            assert entry["graph_shape"] == "graph_collection"
            assert isinstance(entry["graph_count"], int)
            assert isinstance(entry["node_count"], int)
            assert isinstance(entry["edge_count"], int)
            assert entry["graph_count"] > 0
            assert entry["node_count"] > 0
            assert entry["edge_count"] > 0


def test_configure_and_run_dataset_writes_prepared_outputs_and_mapping(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source", include_isolated=True)
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=5,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Dataset Project"}).json()
        project_id = project["id"]

        configured = client.post(
            f"/api/projects/{project_id}/datasets",
            json={
                "catalog_id": "TINY",
                "params": {
                    "graph_type": "networkx",
                    "filter_largest_component": True,
                },
            },
        )
        assert configured.status_code == 200
        dataset = configured.json()
        dataset_id = dataset["id"]
        assert_uuid4(dataset_id)
        assert dataset["schema_version"] == "1"
        assert dataset["manifest_type"] == "dataset"
        assert dataset["project_id"] == project_id
        assert dataset["name"] == "Tiny Dataset"
        assert dataset["description"] == "local test bundle"
        assert dataset["status"] == "planned"
        assert dataset["source_type"] == "neext_csv_bundle"
        assert dataset["source_catalog_id"] == "TINY"
        assert dataset["storage_format"] == "neext-parquet-v1"
        assert dataset["graph_shape"] == "graph_collection"
        assert dataset["inputs"] == []
        assert dataset["operation"] == {
            "operation_id": "neext.prepare_graph_collection",
            "operation_version": "1",
            "params": {
                "graph_type": "networkx",
                "reindex_nodes": True,
                "filter_largest_component": True,
            },
        }
        assert dataset["raw_data_files"] is None
        assert dataset["prepared_data_files"] is None
        assert dataset["mapping_files"] is None
        assert dataset["prepared_stats"] is None
        assert dataset["source_stats"] == {
            "graph_count": 2,
            "node_count": 5,
            "edge_count": 2,
            "has_graph_labels": True,
            "has_node_features": True,
            "has_edge_features": True,
        }
        assert str(Path(tmpdir).resolve()) not in json.dumps(dataset)

        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / dataset_id
        assert (artifact_path / "artifact.json").is_file()
        assert json.loads((artifact_path / "artifact.json").read_text(encoding="utf-8")) == dataset
        assert not (artifact_path / "raw").exists()
        assert not (artifact_path / "prepared").exists()

        run = client.post(f"/api/projects/{project_id}/datasets/{dataset_id}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert "Preparing dataset Tiny Dataset" in job["log"]

        read = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}")
        assert read.status_code == 200
        dataset = read.json()
        assert dataset["status"] == "completed"
        assert dataset["raw_data_files"] == {
            "nodes": "raw/nodes.parquet",
            "edges": "raw/edges.parquet",
            "graph_labels": "raw/graph_labels.parquet",
            "node_features": "raw/node_features.parquet",
            "edge_features": "raw/edge_features.parquet",
        }
        assert dataset["prepared_data_files"] == {
            "nodes": "prepared/nodes.parquet",
            "edges": "prepared/edges.parquet",
            "graph_labels": "prepared/graph_labels.parquet",
            "node_features": "prepared/node_features.parquet",
            "edge_features": "prepared/edge_features.parquet",
        }
        assert dataset["mapping_files"] == {
            "node_mapping": "mappings/node_mapping.parquet",
            "graph_mapping": "mappings/graph_mapping.parquet",
        }
        assert dataset["prepared_stats"] == {
            "graph_count": 2,
            "node_count": 4,
            "edge_count": 2,
            "has_graph_labels": True,
            "has_node_features": True,
            "has_edge_features": True,
        }

        raw_nodes = pd.read_parquet(artifact_path / "raw" / "nodes.parquet")
        nodes = pd.read_parquet(artifact_path / "prepared" / "nodes.parquet")
        edges = pd.read_parquet(artifact_path / "prepared" / "edges.parquet")
        graph_labels = pd.read_parquet(artifact_path / "prepared" / "graph_labels.parquet")
        node_features = pd.read_parquet(artifact_path / "prepared" / "node_features.parquet")
        edge_features = pd.read_parquet(artifact_path / "prepared" / "edge_features.parquet")
        mapping = pd.read_parquet(artifact_path / "mappings" / "node_mapping.parquet")
        assert list(raw_nodes.columns) == ["node_id", "graph_id"]
        assert list(nodes.columns) == ["graph_id", "node_id"]
        assert list(edges.columns) == ["graph_id", "src_node_id", "dest_node_id"]
        assert list(edges["graph_id"]) == ["g1", "g2"]
        assert list(graph_labels.columns) == ["graph_id", "graph_label"]
        assert list(node_features.columns) == ["graph_id", "node_id", "feature_a"]
        assert list(edge_features.columns) == ["graph_id", "src_node_id", "dest_node_id", "edge_weight"]
        assert list(mapping.columns) == [
            "source_graph_id",
            "source_node_id",
            "internal_graph_id",
            "internal_node_id",
            "included",
            "drop_reason",
        ]
        assert mapping["included"].tolist() == [True, True, False, True, True]
        dropped = mapping[~mapping["included"]].iloc[0]
        assert dropped["source_node_id"] == 5
        assert dropped["drop_reason"] == "not_in_largest_connected_component"

        listed = client.get(f"/api/projects/{project_id}/datasets")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == [dataset_id]

        preview = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/preview/mapping?limit=2")
        assert preview.status_code == 200
        assert preview.json()["total_rows"] == 5
        assert len(preview.json()["rows"]) == 2

        second = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"})
        assert second.status_code == 200
        second_dataset = second.json()
        assert_uuid4(second_dataset["id"])
        assert second_dataset["id"] != dataset_id
        assert second_dataset["source_catalog_id"] == "TINY"
        listed_again = client.get(f"/api/projects/{project_id}/datasets")
        assert listed_again.status_code == 200
        assert len(listed_again.json()) == 2


def test_feature_library_catalog_endpoint_exposes_built_ins_without_paths_or_code():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    expected_feature_ids = {
        "page_rank",
        "degree_centrality",
        "closeness_centrality",
        "betweenness_centrality",
        "eigenvector_centrality",
        "clustering_coefficient",
        "local_efficiency",
        "lsme",
        "load_centrality",
        "basic_expansion",
        "betastar",
    }

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        response = client.get("/api/feature-library")

        assert response.status_code == 200
        catalog = response.json()
        assert {entry["id"] for entry in catalog} == expected_feature_ids
        assert len(catalog) == 11
        page_rank = next(entry for entry in catalog if entry["id"] == "page_rank")
        assert page_rank["name"] == "PageRank"
        assert page_rank["type"] == "structural_node_feature"
        assert page_rank["source_type"] == "neext_structural_node_feature"
        assert page_rank["operation_id"] == "neext.compute_node_features"
        assert page_rank["operation_version"] == "1"
        for entry in catalog:
            payload = json.dumps(entry)
            assert "files" not in entry
            assert "path" not in entry
            assert "callable" not in entry
            assert "function" not in entry
            assert "http://" not in payload
            assert "https://" not in payload


def test_create_feature_artifact_writes_planned_dag_manifest_and_lists_project_local(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        first_project = client.post("/api/projects", json={"name": "Feature Project"}).json()
        second_project = client.post("/api/projects", json={"name": "Other Project"}).json()
        first_project_id = first_project["id"]
        second_project_id = second_project["id"]

        configured = client.post(f"/api/projects/{first_project_id}/datasets", json={"catalog_id": "TINY"})
        assert configured.status_code == 200
        dataset = configured.json()
        dataset_id = dataset["id"]

        created = client.post(
            f"/api/projects/{first_project_id}/features",
            json={
                "source_dataset_id": dataset_id,
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 3,
                    "normalize_features": True,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        )
        assert created.status_code == 200
        feature = created.json()
        feature_id = feature["id"]
        assert_uuid4(feature_id)
        assert feature["schema_version"] == "1"
        assert feature["manifest_type"] == "feature"
        assert feature["project_id"] == first_project_id
        assert feature["name"] == "Tiny Dataset - PageRank"
        assert feature["description"] == ""
        assert feature["status"] == "planned"
        assert feature["inputs"] == [
            {
                "role": "source_dataset",
                "artifact_kind": "dataset",
                "artifact_id": dataset_id,
            }
        ]
        assert feature["source_type"] == "neext_structural_node_feature"
        assert feature["source_feature_id"] == "page_rank"
        assert feature["operation"] == {
            "operation_id": "neext.compute_node_features",
            "operation_version": "1",
            "params": {
                "feature_list": ["page_rank"],
                "feature_vector_length": 3,
                "normalize_features": True,
                "n_jobs": 1,
                "parallel_backend": "loky",
            },
        }
        assert feature["expected_output"] == {
            "artifact_kind": "feature",
            "storage_format": "neext-feature-parquet-v1",
            "columns": ["page_rank_0", "page_rank_1", "page_rank_2"],
        }
        assert str(Path(tmpdir).resolve()) not in json.dumps(feature)

        artifact_path = Path(tmpdir) / "projects" / first_project_id / "artifacts" / "features" / feature_id
        assert (artifact_path / "artifact.json").is_file()
        assert json.loads((artifact_path / "artifact.json").read_text(encoding="utf-8")) == feature
        assert not (artifact_path / "data").exists()

        listed = client.get(f"/api/projects/{first_project_id}/features")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == [feature_id]

        read = client.get(f"/api/projects/{first_project_id}/features/{feature_id}")
        assert read.status_code == 200
        assert read.json() == feature

        other_project_features = client.get(f"/api/projects/{second_project_id}/features")
        assert other_project_features.status_code == 200
        assert other_project_features.json() == []
        assert client.get(f"/api/projects/{second_project_id}/features/{feature_id}").status_code == 404
        assert client.post(
            f"/api/projects/{second_project_id}/features",
            json={
                "source_dataset_id": dataset_id,
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 3,
                    "normalize_features": True,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        ).status_code == 404


def test_feature_create_returns_api_errors_for_invalid_project_dataset_feature_or_params(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )
        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Feature Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        dataset_id = dataset["id"]
        missing_project_id = str(uuid.uuid4())
        missing_dataset_id = str(uuid.uuid4())

        valid_payload = {
            "source_dataset_id": dataset_id,
            "source_feature_id": "page_rank",
            "params": {
                "feature_vector_length": 3,
                "normalize_features": True,
                "n_jobs": 1,
                "parallel_backend": "loky",
            },
        }

        assert client.get(f"/api/projects/{missing_project_id}/features").status_code == 404
        assert client.get(f"/api/projects/{project_id}/features/{uuid.uuid4()}").status_code == 404
        assert client.post(f"/api/projects/{missing_project_id}/features", json=valid_payload).status_code == 404

        missing_dataset_payload = dict(valid_payload, source_dataset_id=missing_dataset_id)
        assert client.post(f"/api/projects/{project_id}/features", json=missing_dataset_payload).status_code == 404

        unknown_feature_payload = dict(valid_payload, source_feature_id="unknown_feature")
        assert client.post(f"/api/projects/{project_id}/features", json=unknown_feature_payload).status_code == 404

        invalid_length_payload = {
            **valid_payload,
            "params": {
                **valid_payload["params"],
                "feature_vector_length": 11,
            },
        }
        assert client.post(f"/api/projects/{project_id}/features", json=invalid_length_payload).status_code == 422

        invalid_jobs_payload = {
            **valid_payload,
            "params": {
                **valid_payload["params"],
                "n_jobs": 0,
            },
        }
        assert client.post(f"/api/projects/{project_id}/features", json=invalid_jobs_payload).status_code == 422

        invalid_backend_payload = {
            **valid_payload,
            "params": {
                **valid_payload["params"],
                "parallel_backend": "processes",
            },
        }
        assert client.post(f"/api/projects/{project_id}/features", json=invalid_backend_payload).status_code == 422


def test_feature_batch_run_auto_prepares_planned_dataset_and_writes_outputs(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Feature Run Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        assert dataset["status"] == "planned"

        feature_ids = []
        for feature_id in ["page_rank", "degree_centrality"]:
            created = client.post(
                f"/api/projects/{project_id}/features",
                json={
                    "source_dataset_id": dataset["id"],
                    "source_feature_id": feature_id,
                    "params": {
                        "feature_vector_length": 2,
                        "normalize_features": False,
                        "n_jobs": 1,
                        "parallel_backend": "loky",
                    },
                },
            )
            assert created.status_code == 200
            feature_ids.append(created.json()["id"])

        run = client.post(f"/api/projects/{project_id}/features/run-batch", json={"feature_ids": feature_ids})
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert job["operation"]["operation_id"] == "neext.compute_node_features"
        assert "Preparing upstream dataset Tiny Dataset" in job["log"]
        assert "Computing features: page_rank, degree_centrality" in job["log"]

        dataset_after = client.get(f"/api/projects/{project_id}/datasets/{dataset['id']}").json()
        assert dataset_after["status"] == "completed"

        for feature_id in feature_ids:
            feature = client.get(f"/api/projects/{project_id}/features/{feature_id}").json()
            assert feature["status"] == "completed"
            assert feature["output_files"] == {"features": "output/features.parquet"}
            output_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "features" / feature_id / "output" / "features.parquet"
            output_df = pd.read_parquet(output_path)
            assert list(output_df.columns[:2]) == ["node_id", "graph_id"]
            assert output_df["node_id"].tolist() == [0, 1, 0, 1]

            preview = client.get(f"/api/projects/{project_id}/features/{feature_id}/preview?limit=2")
            assert preview.status_code == 200
            assert preview.json()["total_rows"] == 4
            assert len(preview.json()["rows"]) == 2

        jobs = client.get(f"/api/projects/{project_id}/jobs")
        assert jobs.status_code == 200
        assert jobs.json()[0]["id"] == job["id"]


def test_embedding_library_catalog_endpoint_exposes_built_ins_without_paths_or_code():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        response = client.get("/api/embedding-library")

        assert response.status_code == 200
        catalog = response.json()
        assert {entry["id"] for entry in catalog} == {"approx_wasserstein", "wasserstein", "sinkhornvectorizer"}
        assert len(catalog) == 3
        approx = next(entry for entry in catalog if entry["id"] == "approx_wasserstein")
        assert approx["name"] == "Approx Wasserstein"
        assert approx["operation_id"] == "neext.compute_graph_embeddings"
        assert approx["operation_version"] == "1"
        for entry in catalog:
            payload = json.dumps(entry)
            assert "files" not in entry
            assert "path" not in entry
            assert "callable" not in entry
            assert "function" not in entry
            assert "http://" not in payload
            assert "https://" not in payload


def test_create_embedding_artifacts_validate_source_features_and_dataset_lineage(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Embedding Project"}).json()
        project_id = project["id"]
        first_dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        second_dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()

        first_feature_ids = []
        for source_feature_id in ["page_rank", "degree_centrality"]:
            created = client.post(
                f"/api/projects/{project_id}/features",
                json={
                    "source_dataset_id": first_dataset["id"],
                    "source_feature_id": source_feature_id,
                    "params": {
                        "feature_vector_length": 2,
                        "normalize_features": False,
                        "n_jobs": 1,
                        "parallel_backend": "threading",
                    },
                },
            )
            assert created.status_code == 200
            first_feature_ids.append(created.json()["id"])

        second_feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": second_dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "threading",
                },
            },
        ).json()

        created = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": first_feature_ids,
                "params": {"embedding_dimension": 2},
            },
        )
        assert created.status_code == 200
        embedding = created.json()
        embedding_id = embedding["id"]
        assert_uuid4(embedding_id)
        assert embedding["schema_version"] == "1"
        assert embedding["manifest_type"] == "embedding"
        assert embedding["project_id"] == project_id
        assert embedding["name"] == "Tiny Dataset - Approx Wasserstein Embedding"
        assert embedding["status"] == "planned"
        assert embedding["source_type"] == "neext_graph_embedding"
        assert embedding["source_embedding_id"] == "approx_wasserstein"
        assert embedding["inputs"] == [
            {"role": "source_feature", "artifact_kind": "feature", "artifact_id": first_feature_ids[0]},
            {"role": "source_feature", "artifact_kind": "feature", "artifact_id": first_feature_ids[1]},
        ]
        assert embedding["operation"] == {
            "operation_id": "neext.compute_graph_embeddings",
            "operation_version": "1",
            "params": {
                "embedding_algorithm": "approx_wasserstein",
                "embedding_dimension": 2,
                "random_state": 42,
                "memory_size": "4G",
                "feature_ids": first_feature_ids,
                "feature_columns": "all",
            },
        }
        assert embedding["expected_output"] == {
            "artifact_kind": "embedding",
            "storage_format": "neext-embedding-parquet-v1",
            "columns": ["emb_0", "emb_1"],
        }
        assert embedding["output_files"] is None
        assert embedding["output_stats"] is None
        assert str(Path(tmpdir).resolve()) not in json.dumps(embedding)

        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "embeddings" / embedding_id
        assert (artifact_path / "artifact.json").is_file()
        assert json.loads((artifact_path / "artifact.json").read_text(encoding="utf-8")) == embedding

        listed = client.get(f"/api/projects/{project_id}/embeddings")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == [embedding_id]

        cross_dataset = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [first_feature_ids[0], second_feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        )
        assert cross_dataset.status_code == 400
        assert "same dataset" in cross_dataset.json()["detail"]

        duplicate_features = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [first_feature_ids[0], first_feature_ids[0]],
                "params": {"embedding_dimension": 2},
            },
        )
        assert duplicate_features.status_code == 400
        assert "unique" in duplicate_features.json()["detail"]


def test_run_embedding_auto_runs_planned_dataset_and_features_and_writes_preview(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")
    pytest.importorskip("vectorizers")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Embedding Run Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        feature_ids = []
        for source_feature_id in ["page_rank", "degree_centrality"]:
            feature = client.post(
                f"/api/projects/{project_id}/features",
                json={
                    "source_dataset_id": dataset["id"],
                    "source_feature_id": source_feature_id,
                    "params": {
                        "feature_vector_length": 2,
                        "normalize_features": False,
                        "n_jobs": 1,
                        "parallel_backend": "threading",
                    },
                },
            ).json()
            feature_ids.append(feature["id"])

        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": feature_ids,
                "params": {"embedding_dimension": 2},
            },
        ).json()

        run = client.post(f"/api/projects/{project_id}/embeddings/{embedding['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert job["operation"]["operation_id"] == "neext.compute_graph_embeddings"
        assert job["target_artifacts"] == [{"artifact_kind": "embedding", "artifact_id": embedding["id"]}]
        assert "Preparing upstream dataset Tiny Dataset" in job["log"]
        assert f"Computing upstream features for embedding {embedding['name']}" in job["log"]
        assert "Computing features: page_rank, degree_centrality" in job["log"]
        assert f"Computing embedding {embedding['name']} with approx_wasserstein" in job["log"]

        dataset_after = client.get(f"/api/projects/{project_id}/datasets/{dataset['id']}").json()
        assert dataset_after["status"] == "completed"
        for feature_id in feature_ids:
            assert client.get(f"/api/projects/{project_id}/features/{feature_id}").json()["status"] == "completed"

        completed = client.get(f"/api/projects/{project_id}/embeddings/{embedding['id']}").json()
        assert completed["status"] == "completed"
        assert completed["output_files"] == {"embeddings": "output/embeddings.parquet"}
        assert completed["output_stats"] == {"row_count": 2, "column_count": 3}
        assert completed["error"] is None

        output_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "embeddings" / embedding["id"] / "output" / "embeddings.parquet"
        output_df = pd.read_parquet(output_path)
        assert list(output_df.columns) == ["graph_id", "emb_0", "emb_1"]

        preview = client.get(f"/api/projects/{project_id}/embeddings/{embedding['id']}/preview?limit=1")
        assert preview.status_code == 200
        assert preview.json()["columns"] == ["graph_id", "emb_0", "emb_1"]
        assert preview.json()["total_rows"] == 2
        assert len(preview.json()["rows"]) == 1


def test_embedding_batch_run_completes_selected_outputs_in_one_job(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")
    pytest.importorskip("vectorizers")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Embedding Batch Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "threading",
                },
            },
        ).json()

        embedding_ids = []
        for dimension in [1, 2]:
            embedding = client.post(
                f"/api/projects/{project_id}/embeddings",
                json={
                    "source_embedding_id": "approx_wasserstein",
                    "source_feature_ids": [feature["id"]],
                    "params": {"embedding_dimension": dimension},
                },
            ).json()
            embedding_ids.append(embedding["id"])

        duplicate_run = client.post(
            f"/api/projects/{project_id}/embeddings/run-batch",
            json={"embedding_ids": [embedding_ids[0], embedding_ids[0]]},
        )
        assert duplicate_run.status_code == 400
        assert "unique" in duplicate_run.json()["detail"]

        run = client.post(f"/api/projects/{project_id}/embeddings/run-batch", json={"embedding_ids": embedding_ids})
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert job["target_artifacts"] == [
            {"artifact_kind": "embedding", "artifact_id": embedding_ids[0]},
            {"artifact_kind": "embedding", "artifact_id": embedding_ids[1]},
        ]

        for embedding_id in embedding_ids:
            embedding = client.get(f"/api/projects/{project_id}/embeddings/{embedding_id}").json()
            assert embedding["status"] == "completed"
            assert embedding["output_files"] == {"embeddings": "output/embeddings.parquet"}


def test_failed_embedding_execution_sets_error_and_cleans_partial_output(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        project = client.post("/api/projects", json={"name": "Embedding Failure Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "threading",
                },
            },
        ).json()
        feature_run = client.post(f"/api/projects/{project_id}/features/{feature['id']}/run")
        assert wait_for_job(client, project_id, feature_run.json()["id"])["status"] == "completed"
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()

        class DummyEmbeddings:
            embeddings_df = pd.DataFrame({"graph_id": ["g1", "g2"], "emb_0": [0.1, 0.2], "emb_1": [0.3, 0.4]})

        monkeypatch.setattr(app.state.store, "_compute_graph_embeddings", lambda collection, features, params: DummyEmbeddings())

        def fail_writer(project_id: str, embedding_id: str, output_df):
            artifact_path = app.state.store.embedding_path(project_id, embedding_id)
            tmp_path = artifact_path / "_tmp" / "output"
            tmp_path.mkdir(parents=True, exist_ok=True)
            (tmp_path / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("forced embedding write failure")

        monkeypatch.setattr(app.state.store, "_write_embedding_output", fail_writer)

        run = client.post(f"/api/projects/{project_id}/embeddings/{embedding['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "failed"
        assert "forced embedding write failure" in job["error"]

        failed = client.get(f"/api/projects/{project_id}/embeddings/{embedding['id']}").json()
        assert failed["status"] == "failed"
        assert "forced embedding write failure" in failed["error"]["message"]
        assert failed["output_files"] is None
        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "embeddings" / embedding["id"]
        assert not (artifact_path / "_tmp").exists()
        assert not (artifact_path / "output").exists()


def test_dataset_endpoints_return_404_for_missing_project_or_catalog(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="TINY",
                    name="Tiny Dataset",
                    description="local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )
        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Dataset Project"}).json()
        missing_project_id = str(uuid.uuid4())

        assert client.get(f"/api/projects/{missing_project_id}/datasets").status_code == 404
        assert client.get(f"/api/projects/{project['id']}/datasets/{uuid.uuid4()}").status_code == 404
        assert client.post(f"/api/projects/{missing_project_id}/datasets", json={"catalog_id": "TINY"}).status_code == 404
        assert client.post(f"/api/projects/{project['id']}/datasets", json={"catalog_id": "UNKNOWN"}).status_code == 404


def test_failed_dataset_run_cleans_outputs_and_records_failure(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_dataset_source_bundle(Path(tmpdir) / "bad-source", invalid_edge=True)
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="BAD",
                    name="Bad Dataset",
                    description="invalid local test bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=2,
                    node_count=4,
                    edge_count=2,
                    source="Test catalog",
                ),
            ),
        )
        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Dataset Project"}).json()

        response = client.post(f"/api/projects/{project['id']}/datasets", json={"catalog_id": "BAD"})
        assert response.status_code == 200
        dataset = response.json()

        run = client.post(f"/api/projects/{project['id']}/datasets/{dataset['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project["id"], run.json()["id"])
        assert job["status"] == "failed"
        assert "endpoints" in job["error"]

        read = client.get(f"/api/projects/{project['id']}/datasets/{dataset['id']}")
        assert read.status_code == 200
        failed_dataset = read.json()
        assert failed_dataset["status"] == "failed"
        assert "endpoints" in failed_dataset["error"]["message"]

        artifact_path = Path(tmpdir) / "projects" / project["id"] / "artifacts" / "datasets" / dataset["id"]
        assert not (artifact_path / "raw").exists()
        assert not (artifact_path / "prepared").exists()
        assert not (artifact_path / "mappings").exists()


def test_project_list_scans_valid_manifests_and_sorts_by_updated_at():
    from NEExT.workbench.schemas import ProjectCreate
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        store = WorkbenchStore(Path(tmpdir))
        older = store.create_project(ProjectCreate(name="Older Project"))
        newer = store.create_project(ProjectCreate(name="Newer Project"))

        older_payload = older.model_dump()
        older_payload["updated_at"] = "2026-01-01T00:00:00+00:00"
        (store.project_path(older.id) / "project.json").write_text(json.dumps(older_payload), encoding="utf-8")

        newer_payload = newer.model_dump()
        newer_payload["updated_at"] = "2026-01-02T00:00:00+00:00"
        (store.project_path(newer.id) / "project.json").write_text(json.dumps(newer_payload), encoding="utf-8")

        legacy_path = Path(tmpdir) / "projects" / "legacy-slug"
        legacy_path.mkdir(parents=True)
        (legacy_path / "project.json").write_text(
            json.dumps(
                {
                    "id": "legacy-slug",
                    "name": "Legacy Slug",
                    "description": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-03T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        assert [project.id for project in store.list_projects()] == [newer.id, older.id]


def test_missing_project_returns_404():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        response = client.get("/api/projects/not-found")
        assert response.status_code == 404


def test_project_delete_moves_project_to_workspace_trash():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        created = client.post("/api/projects", json={"name": "Trash Me", "description": "delete test"})
        assert created.status_code == 200
        project = created.json()
        project_id = project["id"]

        response = client.delete(f"/api/projects/{project_id}")
        assert response.status_code == 200
        assert response.json() == {
            "id": project_id,
            "name": "Trash Me",
            "trashed_path": f"trash/projects/{project_id}",
        }
        assert "path" not in response.json()

        assert not (Path(tmpdir) / "projects" / project_id).exists()
        assert (Path(tmpdir) / "trash" / "projects" / project_id / "project.json").is_file()

        projects = client.get("/api/projects")
        assert projects.status_code == 200
        assert all(item["id"] != project_id for item in projects.json())

        read = client.get(f"/api/projects/{project_id}")
        assert read.status_code == 404

        workspace = client.get("/api/workspace")
        assert workspace.status_code == 200
        assert workspace.json()["projects"] == 0


def test_missing_project_delete_returns_404():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        missing = client.delete(f"/api/projects/{uuid.uuid4()}")
        assert missing.status_code == 404

        invalid = client.delete("/api/projects/not-a-project-id")
        assert invalid.status_code == 404


def test_project_delete_trash_collision_uses_suffixed_folder():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))

        created = client.post("/api/projects", json={"name": "Collision Project"})
        assert created.status_code == 200
        project = created.json()
        project_id = project["id"]

        existing_trash = Path(tmpdir) / "trash" / "projects" / project_id
        existing_trash.mkdir(parents=True)
        sentinel = existing_trash / "sentinel.txt"
        sentinel.write_text("existing trash", encoding="utf-8")

        response = client.delete(f"/api/projects/{project_id}")
        assert response.status_code == 200
        summary = response.json()
        assert summary["id"] == project_id
        assert summary["name"] == "Collision Project"
        assert summary["trashed_path"].startswith(f"trash/projects/{project_id}-")

        moved_path = Path(tmpdir) / summary["trashed_path"]
        assert moved_path.is_dir()
        assert (moved_path / "project.json").is_file()
        assert sentinel.read_text(encoding="utf-8") == "existing trash"
        assert not (Path(tmpdir) / "projects" / project_id).exists()
