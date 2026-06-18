import json
import sys
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
        "node_id,graph_id\n" "1,g1\n" "2,g1\n" "3,g2\n" "4,g2\n" + isolated_node,
        encoding="utf-8",
    )
    edge_dest = "99" if invalid_edge else "2"
    (source_dir / "edges.csv").write_text(
        "src_node_id,dest_node_id\n" f"1,{edge_dest}\n" "3,4\n",
        encoding="utf-8",
    )
    (source_dir / "graph_labels.csv").write_text(
        "graph_id,graph_label\n" "g1,0\n" "g2,1\n",
        encoding="utf-8",
    )
    isolated_feature = "5,0.5\n" if include_isolated else ""
    (source_dir / "node_features.csv").write_text(
        "node_id,feature_a\n" "1,0.1\n" "2,0.2\n" "3,0.3\n" "4,0.4\n" + isolated_feature,
        encoding="utf-8",
    )
    (source_dir / "edge_features.csv").write_text(
        "src_node_id,dest_node_id,edge_weight\n" "1,2,1.5\n" "3,4,2.5\n",
        encoding="utf-8",
    )
    return {
        "edges": str(source_dir / "edges.csv"),
        "node_graph_mapping": str(source_dir / "node_graph_mapping.csv"),
        "graph_labels": str(source_dir / "graph_labels.csv"),
        "node_features": str(source_dir / "node_features.csv"),
        "edge_features": str(source_dir / "edge_features.csv"),
    }


def write_labeled_graph_source_bundle(source_dir: Path, *, graph_count: int = 8, numeric_labels: bool = True) -> dict[str, str]:
    source_dir.mkdir(parents=True)
    node_rows = ["node_id,graph_id\n"]
    edge_rows = ["src_node_id,dest_node_id\n"]
    label_rows = ["graph_id,graph_label\n"]
    node_feature_rows = ["node_id,feature_a\n"]
    for graph_index in range(graph_count):
        graph_id = f"g{graph_index}"
        node_a = graph_index * 3
        node_b = graph_index * 3 + 1
        node_c = graph_index * 3 + 2
        node_rows.extend([f"{node_a},{graph_id}\n", f"{node_b},{graph_id}\n", f"{node_c},{graph_id}\n"])
        edge_rows.extend([f"{node_a},{node_b}\n", f"{node_b},{node_c}\n"])
        label = graph_index % 2 if numeric_labels else ("left" if graph_index % 2 == 0 else "right")
        label_rows.append(f"{graph_id},{label}\n")
        node_feature_rows.extend(
            [
                f"{node_a},{float(graph_index)}\n",
                f"{node_b},{float(graph_index) + 0.5}\n",
                f"{node_c},{float(graph_index) + 1.0}\n",
            ]
        )

    (source_dir / "node_graph_mapping.csv").write_text("".join(node_rows), encoding="utf-8")
    (source_dir / "edges.csv").write_text("".join(edge_rows), encoding="utf-8")
    (source_dir / "graph_labels.csv").write_text("".join(label_rows), encoding="utf-8")
    (source_dir / "node_features.csv").write_text("".join(node_feature_rows), encoding="utf-8")
    return {
        "edges": str(source_dir / "edges.csv"),
        "node_graph_mapping": str(source_dir / "node_graph_mapping.csv"),
        "graph_labels": str(source_dir / "graph_labels.csv"),
        "node_features": str(source_dir / "node_features.csv"),
    }


def write_single_graph_source_bundle(source_dir: Path) -> dict[str, str]:
    source_dir.mkdir(parents=True)
    (source_dir / "nodes.csv").write_text(
        "node_id,role,score\n"
        "101,left,1.0\n"
        "102,right,2.0\n"
        "103,left,3.0\n"
        "104,right,4.0\n"
        "105,left,5.0\n"
        "106,right,6.0\n"
        "107,left,7.0\n"
        "108,right,8.0\n",
        encoding="utf-8",
    )
    (source_dir / "edges.csv").write_text(
        "src_node_id,dest_node_id\n" "101,102\n" "102,103\n" "103,104\n" "104,105\n" "105,106\n" "106,107\n" "107,108\n",
        encoding="utf-8",
    )
    return {
        "nodes": str(source_dir / "nodes.csv"),
        "edges": str(source_dir / "edges.csv"),
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


def create_planned_lifecycle_chain(client, project_id: str) -> dict:
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
    embedding = client.post(
        f"/api/projects/{project_id}/embeddings",
        json={
            "source_embedding_id": "approx_wasserstein",
            "source_feature_ids": [feature["id"]],
            "params": {"embedding_dimension": 2},
        },
    ).json()
    model = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "source_model_id": "random_forest",
            "source_embedding_ids": [embedding["id"]],
            "params": {
                "task_type": "classifier",
                "sample_size": 1,
                "test_size": 0.5,
                "balance_dataset": False,
                "n_jobs": 1,
                "parallel_backend": "thread",
            },
        },
    ).json()
    return {"dataset": dataset, "feature": feature, "embedding": embedding, "model": model}


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


def test_mcp_settings_lifecycle_hashes_tokens_and_hides_full_token(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp.server.streamable_http_manager")

    from fastapi.testclient import TestClient

    from NEExT.workbench import app as workbench_app
    from NEExT.workbench.app import create_app
    from NEExT.workbench.schemas import McpStdioReadiness

    # Force a ready stdio environment so the Claude Desktop snippet is emitted
    # regardless of where the test interpreter lives (a dev venv under ~/Desktop
    # would otherwise be correctly blocked by macOS protected-folder detection).
    monkeypatch.setattr(
        workbench_app,
        "evaluate_stdio_readiness",
        lambda workspace_path: McpStdioReadiness(
            status="ready",
            ok=True,
            interpreter=sys.executable,
            command_preview=f"{sys.executable} -m NEExT.workbench.mcp_cli --workspace {workspace_path}",
        ),
    )

    with TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        client = TestClient(app)

        initial = client.get("/api/mcp-settings")
        assert initial.status_code == 200
        assert initial.json()["enabled"] is False
        assert initial.json()["one_time_token"] is None
        assert "token_hash" not in initial.json()

        enabled = client.post("/api/mcp-settings/enable")
        assert enabled.status_code == 200
        enabled_payload = enabled.json()
        first_token = enabled_payload["one_time_token"]
        assert enabled_payload["enabled"] is True
        assert first_token.startswith("nxt_mcp_")
        assert enabled_payload["token_preview"].startswith(first_token[:12])
        assert enabled_payload["token_preview"].endswith(first_token[-6:])
        assert "token_hash" not in enabled_payload
        assert app.state.store.verify_mcp_token(first_token)
        client_configs = {snippet["client"]: snippet for snippet in enabled_payload["client_configs"]}
        claude_payload = json.loads(client_configs["claude_desktop"]["content"])
        claude_server = claude_payload["mcpServers"]["neext-workbench"]
        # Canonical claude_desktop_config.json stdio entry is bare command/args/env.
        assert "type" not in claude_server
        assert Path(claude_server["command"]).resolve() == Path(sys.executable).resolve()
        assert claude_server["args"][:3] == ["-m", "NEExT.workbench.mcp_cli", "--workspace"]
        assert Path(claude_server["args"][3]).resolve() == Path(tmpdir).resolve()
        assert claude_server["env"]["NEEXT_WORKBENCH_MCP_TOKEN"] == first_token
        assert enabled_payload["stdio_readiness"]["status"] == "ready"
        cursor_payload = json.loads(client_configs["cursor"]["content"])
        cursor_server = cursor_payload["mcpServers"]["neext-workbench"]
        assert cursor_server["type"] == "streamable-http"
        assert cursor_server["url"] == "http://127.0.0.1:8765/mcp"
        assert cursor_server["headers"]["Authorization"] == f"Bearer {first_token}"
        assert enabled_payload["endpoint_url"] == "http://127.0.0.1:8765/mcp"
        assert enabled_payload["transport"] == "streamable-http"
        assert enabled_payload["protocol_version"] == "2025-11-25"
        assert enabled_payload["sdk_transport_available"] is True
        assert set(enabled_payload["scopes"]) == {"read", "write", "run", "custom-code", "ui-control", "export", "lifecycle"}
        capability_names = {capability["name"] for capability in enabled_payload["capabilities"]}
        assert {"neext_workspace_summary", "neext_configure_custom_feature", "neext_request_delete_project"}.issubset(capability_names)

        settings_file = Path(tmpdir) / "mcp.json"
        settings_payload = json.loads(settings_file.read_text(encoding="utf-8"))
        assert settings_payload["schema_version"] == "1"
        assert settings_payload["manifest_type"] == "mcp_settings"
        assert settings_payload["enabled"] is True
        assert settings_payload["token_hash"]
        assert settings_payload["token_hash"] != first_token
        assert first_token not in settings_file.read_text(encoding="utf-8")

        read_after_enable = client.get("/api/mcp-settings")
        assert read_after_enable.status_code == 200
        assert read_after_enable.json()["one_time_token"] is None
        assert first_token not in json.dumps(read_after_enable.json())

        regenerated = client.post("/api/mcp-settings/regenerate")
        assert regenerated.status_code == 200
        second_token = regenerated.json()["one_time_token"]
        assert second_token.startswith("nxt_mcp_")
        assert second_token != first_token
        assert not app.state.store.verify_mcp_token(first_token)
        assert app.state.store.verify_mcp_token(second_token)

        disabled = client.post("/api/mcp-settings/disable")
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert disabled.json()["one_time_token"] is None
        disabled_payload = json.loads(settings_file.read_text(encoding="utf-8"))
        assert disabled_payload["enabled"] is False
        assert disabled_payload["token_hash"] is None
        assert disabled_payload["token_preview"] is None
        assert not app.state.store.verify_mcp_token(second_token)


def test_mcp_claude_desktop_snippet_suppressed_when_stdio_blocked(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp.server.streamable_http_manager")

    from fastapi.testclient import TestClient

    from NEExT.workbench import app as workbench_app
    from NEExT.workbench.app import create_app
    from NEExT.workbench.schemas import McpStdioReadiness

    blocked = McpStdioReadiness(
        status="blocked",
        ok=False,
        interpreter="/some/python",
        command_preview="/some/python -m NEExT.workbench.mcp_cli --workspace /ws",
        issues=["The Python environment lives inside your macOS Desktop folder, which Claude Desktop cannot read."],
        remediation=["Recreate the virtual environment outside ~/Desktop and reinstall NEExT[workbench-mcp]."],
    )
    monkeypatch.setattr(workbench_app, "evaluate_stdio_readiness", lambda workspace_path: blocked)

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        payload = client.post("/api/mcp-settings/enable").json()
        clients = {snippet["client"] for snippet in payload["client_configs"]}
        # Claude Desktop is stdio-only; when stdio can't launch we suppress the snippet.
        assert "claude_desktop" not in clients
        # Local HTTP clients are unaffected.
        assert {"claude_code", "cursor", "generic"}.issubset(clients)
        assert payload["stdio_readiness"]["status"] == "blocked"
        assert payload["stdio_readiness"]["remediation"]


def test_http_mcp_endpoint_tools_ui_state_and_delete_approval():
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp.server.streamable_http_manager")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app, base_url="http://127.0.0.1:8765", follow_redirects=False) as client:
            token = client.post("/api/mcp-settings/enable").json()["one_time_token"]
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}

            unauthorized = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert unauthorized.status_code == 401

            initialized = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}},
                },
            )
            assert initialized.status_code == 200
            assert initialized.json()["result"]["protocolVersion"] == "2025-11-25"

            tools = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tool_payloads = tools.json()["result"]["tools"]
            tool_names = {tool["name"] for tool in tool_payloads}
            assert {
                "neext_create_project",
                "neext_configure_custom_feature",
                "neext_request_delete_project",
                "neext_set_workbench_view",
            }.issubset(tool_names)
            delete_tool = next(tool for tool in tool_payloads if tool["name"] == "neext_request_delete_project")
            assert delete_tool["annotations"]["destructiveHint"] is True
            assert delete_tool["_meta"]["neextScope"] == "lifecycle"

            created_response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "neext_create_project",
                        "arguments": {"name": "MCP HTTP Project", "description": "sdk-http"},
                    },
                },
            )
            created_payload = json.loads(created_response.json()["result"]["content"][0]["text"])
            project_id = created_payload["id"]
            assert_uuid4(project_id)

            view_response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "neext_set_workbench_view",
                        "arguments": {"route": {"top_tab": "home", "command": "projects", "project_id": project_id}, "message": "Show MCP project"},
                    },
                },
            )
            view_payload = json.loads(view_response.json()["result"]["content"][0]["text"])
            assert view_payload["route"]["project_id"] == project_id
            assert client.get("/api/mcp-ui-state").json()["route"]["project_id"] == project_id

            delete_response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "neext_request_delete_project", "arguments": {"project_id": project_id}},
                },
            )
            approval = json.loads(delete_response.json()["result"]["content"][0]["text"])
            assert approval["status"] == "pending"
            assert approval["operation"] == "delete_project"
            assert client.get(f"/api/projects/{project_id}").status_code == 200

            approved = client.post(f"/api/mcp-approvals/{approval['id']}/approve")
            assert approved.status_code == 200
            assert approved.json()["status"] == "approved"
            assert client.get(f"/api/projects/{project_id}").status_code == 404
            trash = client.get("/api/trash").json()
            assert trash["projects"][0]["project_id"] == project_id


def test_http_mcp_endpoint_enforces_sdk_auth_scopes_and_route_validation():
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp.server.streamable_http_manager")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app, base_url="http://127.0.0.1:8765", follow_redirects=False) as client:
            token = client.post("/api/mcp-settings/enable").json()["one_time_token"]
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}

            settings = app.state.store.read_mcp_settings()
            settings.scopes = ["read"]
            app.state.store._write_mcp_settings(settings)

            tools = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tool_names = {tool["name"] for tool in tools.json()["result"]["tools"]}
            assert "neext_list_projects" in tool_names
            assert "neext_create_project" not in tool_names

            blocked_write = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "neext_create_project",
                        "arguments": {"name": "Blocked Project"},
                    },
                },
            )
            assert blocked_write.status_code == 200
            assert blocked_write.json()["result"]["isError"] is True
            assert '"write" scope' in blocked_write.json()["result"]["content"][0]["text"]
            assert client.get("/api/projects").json() == []

            settings.scopes = ["read", "ui-control"]
            app.state.store._write_mcp_settings(settings)
            invalid_route = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "neext_set_workbench_view",
                        "arguments": {"route": {"top_tab": "admin"}},
                    },
                },
            )
            assert invalid_route.status_code == 200
            assert invalid_route.json()["result"]["isError"] is True
            assert "Unsupported Workbench Space" in invalid_route.json()["result"]["content"][0]["text"]


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
        assert {"graph_collection", "single_graph"}.issubset({entry["source_graph_shape"] for entry in catalog})
        mutag = next(entry for entry in catalog if entry["id"] == "MUTAG")
        assert mutag["graph_count"] == 188
        assert mutag["node_count"] == 3371
        assert mutag["edge_count"] == 7442
        karate = next(entry for entry in catalog if entry["id"] == "KARATE_CLUB")
        assert karate["source_graph_shape"] == "single_graph"
        assert karate["graph_shape"] == "graph_collection"
        assert karate["node_attribute_columns"] == ["club"]
        for entry in catalog:
            payload = json.dumps(entry)
            assert "files" not in entry
            assert "path" not in entry
            assert "http://" not in payload
            assert "https://" not in payload
            assert entry["source_type"] in {"neext_csv_bundle", "neext_single_graph_csv"}
            assert entry["source_graph_shape"] in {"graph_collection", "single_graph"}
            assert entry["graph_shape"] == "graph_collection"
            assert isinstance(entry["graph_count"], int)
            assert isinstance(entry["node_count"], int)
            assert isinstance(entry["edge_count"], int)
            assert entry["graph_count"] > 0
            assert entry["node_count"] > 0
            assert entry["edge_count"] > 0


def test_dataset_intake_validates_creates_and_prepares_uploaded_neext_tables():
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    payload = {
        "name": "Imported Tables",
        "description": "NEExT table import",
        "tables": {
            "node_graph_mapping": {"format": "csv", "csv": "node_id,graph_id\n1,g1\n2,g1\n3,g2\n4,g2\n"},
            "edges": {"format": "csv", "csv": "src_node_id,dest_node_id\n1,2\n3,4\n"},
            "graph_labels": {"format": "csv", "csv": "graph_id,graph_label\ng1,0\ng2,1\n"},
            "node_features": {"format": "csv", "csv": "node_id,feature_a\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n"},
        },
        "params": {"graph_type": "networkx", "filter_largest_component": True},
    }

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Intake Project"}).json()
        project_id = project["id"]

        validation = client.post(f"/api/projects/{project_id}/dataset-intake/validate", json=payload)
        assert validation.status_code == 200
        validation_payload = validation.json()
        assert validation_payload["valid"] is True
        assert validation_payload["errors"] == []
        assert validation_payload["stats"] == {
            "graph_count": 2,
            "node_count": 4,
            "edge_count": 2,
            "has_graph_labels": True,
            "has_node_features": True,
            "has_edge_features": False,
        }

        created = client.post(f"/api/projects/{project_id}/dataset-intake/create", json=payload)
        assert created.status_code == 200
        dataset = created.json()
        dataset_id = dataset["id"]
        assert_uuid4(dataset_id)
        assert dataset["status"] == "planned"
        assert dataset["source_type"] == "uploaded_neext_tables"
        assert dataset["source_catalog_id"] == "dataset-intake"
        assert dataset["source_name"] == "Imported Tables"
        assert dataset["raw_data_files"]["nodes"] == "source/nodes.parquet"
        assert dataset["raw_data_files"]["edges"] == "source/edges.parquet"
        assert "path" not in json.dumps(dataset)

        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / dataset_id
        assert (artifact_path / "source" / "nodes.parquet").is_file()
        assert not (artifact_path / "raw").exists()

        run = client.post(f"/api/projects/{project_id}/datasets/{dataset_id}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        prepared = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}").json()
        assert prepared["status"] == "completed"
        assert prepared["prepared_stats"]["graph_count"] == 2
        assert prepared["raw_data_files"]["nodes"] == "raw/nodes.parquet"
        assert (artifact_path / "prepared" / "nodes.parquet").is_file()

        invalid_payload = {
            **payload,
            "tables": {
                **payload["tables"],
                "node_graph_mapping": {"format": "csv", "csv": "node_id,graph_id\nnode-a,g1\n"},
            },
        }
        invalid = client.post(f"/api/projects/{project_id}/dataset-intake/validate", json=invalid_payload)
        assert invalid.status_code == 200
        assert invalid.json()["valid"] is False
        assert "integer-compatible node IDs" in invalid.json()["errors"][0]["message"]


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
        assert dataset["source_graph_shape"] == "graph_collection"
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

        not_ready_analysis = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis")
        assert not_ready_analysis.status_code == 400
        assert "available only after preparation completes" in not_ready_analysis.json()["detail"]
        not_ready_search = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/search?query=g1")
        assert not_ready_search.status_code == 400
        assert "available only after preparation completes" in not_ready_search.json()["detail"]
        not_ready_node = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/node?graph_id=g1&node_id=0")
        assert not_ready_node.status_code == 400
        assert "available only after preparation completes" in not_ready_node.json()["detail"]
        not_ready_export = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/export/nodes")
        assert not_ready_export.status_code == 400
        assert "available only after preparation completes" in not_ready_export.json()["detail"]

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

        preview_tables = {
            "graph_labels": 2,
            "node_features": 4,
            "edge_features": 2,
            "node_mapping": 5,
            "graph_mapping": 2,
            "mapping": 5,
        }
        for table, total_rows in preview_tables.items():
            table_preview = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/preview/{table}?limit=2")
            assert table_preview.status_code == 200
            assert table_preview.json()["total_rows"] == total_rows
            assert str(Path(tmpdir).resolve()) not in json.dumps(table_preview.json())

        node_mapping = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/preview/node_mapping?limit=2")
        assert node_mapping.status_code == 200
        assert node_mapping.json()["columns"] == [
            "source_graph_id",
            "source_node_id",
            "internal_graph_id",
            "internal_node_id",
            "included",
            "drop_reason",
        ]

        exported = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/export/node_mapping")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("text/csv")
        assert exported.headers["content-disposition"] == 'attachment; filename="Tiny_Dataset_node_mapping.csv"'
        assert str(Path(tmpdir).resolve()) not in exported.text
        assert exported.text.splitlines()[0] == "source_graph_id,source_node_id,internal_graph_id,internal_node_id,included,drop_reason"
        assert "g1,1,g1,0,True," in exported.text

        unsupported_export = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/export/not_a_table")
        assert unsupported_export.status_code == 400
        assert "Unsupported dataset preview table" in unsupported_export.json()["detail"]

        analysis = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis")
        assert analysis.status_code == 200
        analysis_payload = analysis.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(analysis_payload)
        assert analysis_payload["dataset_id"] == dataset_id
        assert analysis_payload["dataset_name"] == "Tiny Dataset"
        assert analysis_payload["dataset_status"] == "completed"
        assert "egonet_metadata" not in analysis_payload
        assert analysis_payload["source_stats"]["node_count"] == 5
        assert analysis_payload["prepared_stats"]["node_count"] == 4
        assert analysis_payload["dropped_node_count"] == 1
        assert analysis_payload["graph_label_distribution"] == {"0": 1, "1": 1}
        assert analysis_payload["node_feature_columns"] == ["feature_a"]
        assert analysis_payload["edge_feature_columns"] == ["edge_weight"]
        assert analysis_payload["graph_summaries"] == [
            {"graph_id": "g1", "node_count": 2, "edge_count": 1, "graph_label": 0},
            {"graph_id": "g2", "node_count": 2, "edge_count": 1, "graph_label": 1},
        ]
        assert analysis_payload["selected_graph_id"] == "g1"
        assert analysis_payload["visual"]["graph_id"] == "g1"
        assert analysis_payload["visual"]["node_count"] == 2
        assert analysis_payload["visual"]["edge_count"] == 1
        assert analysis_payload["visual"]["sampled"] is False
        assert {node["id"] for node in analysis_payload["visual"]["nodes"]} == {"0", "1"}
        assert analysis_payload["visual"]["edges"] == [{"source": "0", "target": "1"}]

        sampled = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis?graph_id=g2&max_nodes=1&max_edges=1")
        assert sampled.status_code == 200
        sampled_payload = sampled.json()
        assert sampled_payload["selected_graph_id"] == "g2"
        assert sampled_payload["visual"]["sampled"] is True
        assert sampled_payload["visual"]["node_count"] == 2
        assert len(sampled_payload["visual"]["nodes"]) == 1

        graph_search = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/search?query=g1")
        assert graph_search.status_code == 200
        graph_search_payload = graph_search.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(graph_search_payload)
        assert graph_search_payload == {
            "query": "g1",
            "limit": 25,
            "total_matches": 1,
            "results": [
                {
                    "kind": "graph",
                    "graph_id": "g1",
                    "node_id": None,
                    "graph_label": 0,
                    "node_count": 2,
                    "edge_count": 1,
                }
            ],
        }
        node_search = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/search?query=0&limit=1")
        assert node_search.status_code == 200
        node_search_payload = node_search.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(node_search_payload)
        assert node_search_payload["query"] == "0"
        assert node_search_payload["limit"] == 1
        assert node_search_payload["total_matches"] == 2
        assert len(node_search_payload["results"]) == 1
        assert node_search_payload["results"][0] == {
            "kind": "node",
            "graph_id": "g1",
            "node_id": "0",
            "graph_label": 0,
            "node_count": 2,
            "edge_count": 1,
        }

        node_detail = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/node?graph_id=g1&node_id=0")
        assert node_detail.status_code == 200
        node_detail_payload = node_detail.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(node_detail_payload)
        assert node_detail_payload == {
            "graph_id": "g1",
            "node_id": "0",
            "degree": 1,
            "graph_label": 0,
            "source_graph_id": "g1",
            "source_node_id": "1",
            "feature_values": {"feature_a": 0.1},
        }
        missing_node = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/node?graph_id=g1&node_id=missing")
        assert missing_node.status_code == 400
        assert "Prepared node not found" in missing_node.json()["detail"]

        second = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"})
        assert second.status_code == 200
        second_dataset = second.json()
        assert_uuid4(second_dataset["id"])
        assert second_dataset["id"] != dataset_id
        assert second_dataset["source_catalog_id"] == "TINY"
        listed_again = client.get(f"/api/projects/{project_id}/datasets")
        assert listed_again.status_code == 200
        assert len(listed_again.json()) == 2


def test_single_graph_dataset_prepares_egonet_collection_and_feeds_downstream_artifacts(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_single_graph_source_bundle(Path(tmpdir) / "single-source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="SINGLE",
                    name="Single Source",
                    description="local single graph",
                    domain="Tests",
                    files=source_files,
                    graph_count=1,
                    node_count=8,
                    edge_count=7,
                    source="Test catalog",
                    source_type="neext_single_graph_csv",
                    source_graph_shape="single_graph",
                    node_attribute_columns=("role", "score"),
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Single Graph Project"}).json()
        project_id = project["id"]

        configured = client.post(
            f"/api/projects/{project_id}/datasets",
            json={
                "catalog_id": "SINGLE",
                "params": {
                    "k_hop": 1,
                    "node_selection": "all_nodes",
                    "target_node_attribute": "role",
                },
            },
        )
        assert configured.status_code == 200
        dataset = configured.json()
        dataset_id = dataset["id"]
        assert dataset["source_type"] == "neext_single_graph_csv"
        assert dataset["source_graph_shape"] == "single_graph"
        assert dataset["graph_shape"] == "graph_collection"
        assert dataset["operation"] == {
            "operation_id": "neext.prepare_single_graph_egonets",
            "operation_version": "1",
            "params": {
                "graph_type": "networkx",
                "reindex_nodes": True,
                "filter_largest_component": False,
                "k_hop": 1,
                "node_selection": "all_nodes",
                "sample_fraction": 1.0,
                "random_seed": 13,
                "source_node_ids": [],
                "target_node_attribute": "role",
            },
        }

        run = client.post(f"/api/projects/{project_id}/datasets/{dataset_id}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert "Computing 1-hop egonets for Single Source" in job["log"]

        dataset = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}").json()
        assert dataset["status"] == "completed"
        assert dataset["source_stats"] == {
            "graph_count": 1,
            "node_count": 8,
            "edge_count": 7,
            "has_graph_labels": False,
            "has_node_features": True,
            "has_edge_features": False,
        }
        assert dataset["prepared_stats"]["graph_count"] == 8
        assert dataset["prepared_stats"]["has_graph_labels"] is True
        assert dataset["prepared_stats"]["has_node_features"] is True

        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / dataset_id
        graph_labels = pd.read_parquet(artifact_path / "prepared" / "graph_labels.parquet")
        node_features = pd.read_parquet(artifact_path / "prepared" / "node_features.parquet")
        node_mapping = pd.read_parquet(artifact_path / "mappings" / "node_mapping.parquet")
        graph_mapping = pd.read_parquet(artifact_path / "mappings" / "graph_mapping.parquet")
        assert len(graph_labels) == 8
        assert set(graph_labels["graph_label"]) == {"left", "right"}
        assert "role" not in node_features.columns
        assert "score" in node_features.columns
        assert {"source_graph_id", "source_node_id", "internal_graph_id", "internal_node_id", "center_source_node_id"}.issubset(node_mapping.columns)
        assert graph_mapping["source_node_id"].tolist() == [101, 102, 103, 104, 105, 106, 107, 108]
        assert graph_mapping["internal_graph_id"].tolist() == list(range(8))

        analysis = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis")
        assert analysis.status_code == 200
        analysis_payload = analysis.json()
        assert analysis_payload["egonet_metadata"] == {
            "source_graph_shape": "single_graph",
            "operation_id": "neext.prepare_single_graph_egonets",
            "operation_version": "1",
            "k_hop": 1,
            "node_selection": "all_nodes",
            "sample_fraction": 1.0,
            "random_seed": 13,
            "target_node_attribute": "role",
        }
        assert analysis_payload["source_stats"]["node_count"] == 8
        assert analysis_payload["source_stats"]["edge_count"] == 7
        assert analysis_payload["prepared_stats"]["graph_count"] == 8
        assert analysis_payload["prepared_stats"]["node_count"] == int(node_mapping.groupby("internal_graph_id").size().sum())
        assert analysis_payload["graph_label_distribution"] == {"left": 4, "right": 4}
        graph_zero_summary = next(summary for summary in analysis_payload["graph_summaries"] if summary["graph_id"] == "0")
        assert graph_zero_summary["source_node_id"] == "101"
        assert graph_zero_summary["graph_label"] == "left"

        graph_zero_analysis = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis?graph_id=0")
        assert graph_zero_analysis.status_code == 200
        graph_zero_payload = graph_zero_analysis.json()
        center_nodes = [node for node in graph_zero_payload["visual"]["nodes"] if node.get("is_center")]
        assert len(center_nodes) == 1
        assert center_nodes[0]["id"] == "0"
        assert center_nodes[0]["source_node_id"] == "101"
        assert graph_zero_payload["selected_graph_id"] == "0"
        node_detail = client.get(f"/api/projects/{project_id}/datasets/{dataset_id}/analysis/node?graph_id=0&node_id=0")
        assert node_detail.status_code == 200
        assert node_detail.json()["source_graph_id"] == "SINGLE"
        assert node_detail.json()["source_node_id"] == "101"

        feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset_id,
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        )
        assert feature.status_code == 200
        feature_id = feature.json()["id"]
        feature_run = client.post(f"/api/projects/{project_id}/features/{feature_id}/run")
        assert feature_run.status_code == 200
        assert wait_for_job(client, project_id, feature_run.json()["id"], timeout_s=20.0)["status"] == "completed"

        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature_id],
                "params": {"embedding_dimension": 2},
            },
        )
        assert embedding.status_code == 200
        embedding_id = embedding.json()["id"]
        embedding_run = client.post(f"/api/projects/{project_id}/embeddings/{embedding_id}/run")
        assert embedding_run.status_code == 200
        assert wait_for_job(client, project_id, embedding_run.json()["id"], timeout_s=20.0)["status"] == "completed"

        model = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [embedding_id],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        )
        assert model.status_code == 200
        model_run = client.post(f"/api/projects/{project_id}/models/{model.json()['id']}/run")
        assert model_run.status_code == 200
        assert wait_for_job(client, project_id, model_run.json()["id"], timeout_s=20.0)["status"] == "completed"


def test_single_graph_node_selection_modes_are_deterministic_and_validate_unknown_ids(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_single_graph_source_bundle(Path(tmpdir) / "single-source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="SINGLE",
                    name="Single Source",
                    description="local single graph",
                    domain="Tests",
                    files=source_files,
                    graph_count=1,
                    node_count=8,
                    edge_count=7,
                    source="Test catalog",
                    source_type="neext_single_graph_csv",
                    source_graph_shape="single_graph",
                    node_attribute_columns=("role", "score"),
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Single Selection Project"}).json()
        project_id = project["id"]

        selected_sequences = []
        for _ in range(2):
            configured = client.post(
                f"/api/projects/{project_id}/datasets",
                json={
                    "catalog_id": "SINGLE",
                    "params": {
                        "k_hop": 1,
                        "node_selection": "sample_fraction",
                        "sample_fraction": 0.5,
                        "random_seed": 7,
                    },
                },
            )
            assert configured.status_code == 200
            dataset_id = configured.json()["id"]
            run = client.post(f"/api/projects/{project_id}/datasets/{dataset_id}/run")
            assert run.status_code == 200
            assert wait_for_job(client, project_id, run.json()["id"])["status"] == "completed"
            graph_mapping = pd.read_parquet(
                Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / dataset_id / "mappings" / "graph_mapping.parquet"
            )
            selected_sequences.append(graph_mapping["source_node_id"].tolist())

        assert selected_sequences[0] == selected_sequences[1]
        assert len(selected_sequences[0]) == 4

        specific = client.post(
            f"/api/projects/{project_id}/datasets",
            json={
                "catalog_id": "SINGLE",
                "params": {
                    "k_hop": 1,
                    "node_selection": "specific_node_ids",
                    "source_node_ids": ["101", "108"],
                },
            },
        )
        assert specific.status_code == 200
        specific_dataset_id = specific.json()["id"]
        run_specific = client.post(f"/api/projects/{project_id}/datasets/{specific_dataset_id}/run")
        assert run_specific.status_code == 200
        assert wait_for_job(client, project_id, run_specific.json()["id"])["status"] == "completed"
        specific_mapping = pd.read_parquet(
            Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / specific_dataset_id / "mappings" / "graph_mapping.parquet"
        )
        assert specific_mapping["source_node_id"].tolist() == [101, 108]

        unknown = client.post(
            f"/api/projects/{project_id}/datasets",
            json={
                "catalog_id": "SINGLE",
                "params": {
                    "k_hop": 1,
                    "node_selection": "specific_node_ids",
                    "source_node_ids": ["101", "999"],
                },
            },
        )
        assert unknown.status_code == 200
        unknown_dataset_id = unknown.json()["id"]
        run_unknown = client.post(f"/api/projects/{project_id}/datasets/{unknown_dataset_id}/run")
        assert run_unknown.status_code == 200
        unknown_job = wait_for_job(client, project_id, run_unknown.json()["id"])
        assert unknown_job["status"] == "failed"
        assert "Unknown source node IDs: 999" in unknown_job["error"]
        failed_dataset = client.get(f"/api/projects/{project_id}/datasets/{unknown_dataset_id}").json()
        assert failed_dataset["status"] == "failed"
        assert "Unknown source node IDs: 999" in failed_dataset["error"]["message"]


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
        assert (
            client.post(
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
            ).status_code
            == 404
        )


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


def test_custom_feature_create_validates_saves_runs_and_feeds_embeddings(monkeypatch):
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
        project = client.post("/api/projects", json={"name": "Custom Feature Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        dataset_run = client.post(f"/api/projects/{project_id}/datasets/{dataset['id']}/run")
        assert dataset_run.status_code == 200
        assert wait_for_job(client, project_id, dataset_run.json()["id"])["status"] == "completed"

        code = """
import pandas as pd

def compute_feature(graph):
    nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
    values = [float(graph.G.degree(node) + 1) for node in nodes]
    df = pd.DataFrame({
        "node_id": nodes,
        "graph_id": graph.graph_id,
        "degree_plus_one": values,
    })
    return df[["node_id", "graph_id", "degree_plus_one"]]
"""
        validation = client.post(
            f"/api/projects/{project_id}/features/custom/validate",
            json={
                "source_dataset_id": dataset["id"],
                "code": code,
            },
        )
        assert validation.status_code == 200
        assert validation.json() == {"valid": True, "columns": ["degree_plus_one"]}
        assert client.get(f"/api/projects/{project_id}/features").json() == []

        created = client.post(
            f"/api/projects/{project_id}/features/custom",
            json={
                "source_dataset_id": dataset["id"],
                "name": "Degree Plus One",
                "description": "custom degree feature",
                "code": code,
                "params": {
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "threading",
                },
            },
        )
        assert created.status_code == 200
        feature = created.json()
        feature_id = feature["id"]
        assert_uuid4(feature_id)
        assert feature["name"] == "Degree Plus One"
        assert feature["description"] == "custom degree feature"
        assert feature["source_type"] == "custom_python_node_feature"
        assert feature["source_feature_id"] == feature_id
        assert feature["source_code_file"] == "source/feature.py"
        assert feature["operation"] == {
            "operation_id": "neext.compute_node_features",
            "operation_version": "1",
            "params": {
                "feature_list": [feature_id],
                "feature_vector_length": 1,
                "normalize_features": False,
                "n_jobs": 1,
                "parallel_backend": "threading",
                "custom_feature_function": "compute_feature",
                "custom_feature_code_file": "source/feature.py",
            },
        }
        assert feature["expected_output"] == {
            "artifact_kind": "feature",
            "storage_format": "neext-feature-parquet-v1",
            "columns": ["degree_plus_one"],
        }
        assert str(Path(tmpdir).resolve()) not in json.dumps(feature)
        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "features" / feature_id
        assert (artifact_path / "source" / "feature.py").read_text(encoding="utf-8") == code

        built_in = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "degree_centrality",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "threading",
                },
            },
        )
        assert built_in.status_code == 200
        run = client.post(f"/api/projects/{project_id}/features/run-batch", json={"feature_ids": [feature_id, built_in.json()["id"]]})
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert f"Computing features: {feature_id}" in job["log"]
        assert "Computing features: degree_centrality" in job["log"]

        feature_after = client.get(f"/api/projects/{project_id}/features/{feature_id}").json()
        assert feature_after["status"] == "completed"
        output_path = artifact_path / "output" / "features.parquet"
        output = pd.read_parquet(output_path)
        assert list(output.columns) == ["node_id", "graph_id", "degree_plus_one"]
        assert output["degree_plus_one"].tolist() == [2.0, 2.0, 2.0, 2.0]
        preview = client.get(f"/api/projects/{project_id}/features/{feature_id}/preview")
        assert preview.status_code == 200
        assert preview.json()["columns"] == ["node_id", "graph_id", "degree_plus_one"]

        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature_id],
                "params": {"embedding_dimension": 2},
            },
        )
        assert embedding.status_code == 200
        assert embedding.json()["inputs"][0]["artifact_id"] == feature_id

        defaulted = client.post(
            f"/api/projects/{project_id}/features/custom",
            json={
                "source_dataset_id": dataset["id"],
                "name": "Default Params",
                "description": "",
                "code": code,
            },
        )
        assert defaulted.status_code == 200
        assert defaulted.json()["operation"]["params"]["normalize_features"] is True
        assert defaulted.json()["operation"]["params"]["n_jobs"] == 1
        assert defaulted.json()["operation"]["params"]["parallel_backend"] == "threading"


def test_custom_feature_create_rejects_unprepared_dataset_and_invalid_code(monkeypatch):
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
        project = client.post("/api/projects", json={"name": "Invalid Custom Feature Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()

        base_payload = {
            "source_dataset_id": dataset["id"],
            "name": "Invalid Custom Feature",
            "description": "",
            "code": "def compute_feature(graph):\n    return None\n",
            "params": {
                "normalize_features": False,
                "n_jobs": 1,
                "parallel_backend": "threading",
            },
        }
        unprepared = client.post(f"/api/projects/{project_id}/features/custom", json=base_payload)
        assert unprepared.status_code == 400
        assert "requires a completed dataset" in unprepared.json()["detail"]
        unprepared_validation = client.post(
            f"/api/projects/{project_id}/features/custom/validate",
            json={"source_dataset_id": dataset["id"], "code": base_payload["code"]},
        )
        assert unprepared_validation.status_code == 400
        assert "validation requires a completed dataset" in unprepared_validation.json()["detail"]

        dataset_run = client.post(f"/api/projects/{project_id}/datasets/{dataset['id']}/run")
        assert dataset_run.status_code == 200
        assert wait_for_job(client, project_id, dataset_run.json()["id"])["status"] == "completed"

        invalid_cases = [
            ("missing function", "def not_the_function(graph):\n    return None\n", "compute_feature"),
            (
                "missing columns",
                "import pandas as pd\n\ndef compute_feature(graph):\n    return pd.DataFrame({'node_id': graph.nodes})\n",
                "node_id, graph_id",
            ),
            (
                "non numeric",
                "import pandas as pd\n\ndef compute_feature(graph):\n    nodes = list(graph.nodes)\n    return pd.DataFrame({'node_id': nodes, 'graph_id': graph.graph_id, 'label': ['x' for _ in nodes]})[['node_id', 'graph_id', 'label']]\n",
                "numeric",
            ),
            (
                "wrong nodes",
                "import pandas as pd\n\ndef compute_feature(graph):\n    return pd.DataFrame({'node_id': [999], 'graph_id': graph.graph_id, 'value': [1.0]})[['node_id', 'graph_id', 'value']]\n",
                "one row for each graph node",
            ),
            (
                "missing import",
                "import definitely_missing_workbench_package\n\ndef compute_feature(graph):\n    return None\n",
                "Missing Python package: definitely_missing_workbench_package",
            ),
            (
                "missing runtime import",
                "def compute_feature(graph):\n    import definitely_missing_runtime_package\n    return None\n",
                "Missing Python package: definitely_missing_runtime_package",
            ),
        ]
        for _, code, expected_message in invalid_cases:
            response = client.post(f"/api/projects/{project_id}/features/custom", json={**base_payload, "code": code})
            assert response.status_code == 400
            assert expected_message in response.json()["detail"]
            validation_response = client.post(
                f"/api/projects/{project_id}/features/custom/validate",
                json={"source_dataset_id": dataset["id"], "code": code},
            )
            assert validation_response.status_code == 400
            assert expected_message in validation_response.json()["detail"]

        assert client.get(f"/api/projects/{project_id}/features").json() == []


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


def test_feature_analysis_search_and_graph_detail(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")
    pytest.importorskip("sklearn")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.schemas import FeatureOutputFiles, FeatureOutputStats
    from NEExT.workbench.storage import utc_now

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
        project = client.post("/api/projects", json={"name": "Feature Analysis Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()

        blocked_feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "degree_centrality",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        ).json()
        planned_analysis = client.get(f"/api/projects/{project_id}/features/{blocked_feature['id']}/analysis")
        assert planned_analysis.status_code == 400
        assert "available only after computation completes" in planned_analysis.json()["detail"]

        store = app.state.store
        for status in ["running", "failed"]:
            manifest = store.read_feature(project_id, blocked_feature["id"])
            manifest.status = status
            store._write_json(store.feature_path(project_id, manifest.id) / "artifact.json", manifest.model_dump())
            blocked_analysis = client.get(f"/api/projects/{project_id}/features/{blocked_feature['id']}/analysis")
            assert blocked_analysis.status_code == 400
            assert "available only after computation completes" in blocked_analysis.json()["detail"]

        created = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        )
        assert created.status_code == 200
        feature_id = created.json()["id"]

        run = client.post(f"/api/projects/{project_id}/features/{feature_id}/run")
        assert run.status_code == 200
        assert wait_for_job(client, project_id, run.json()["id"])["status"] == "completed"

        def complete_manual_feature(name_suffix: str, rows: list[dict[str, object]]) -> str:
            created_manual = client.post(
                f"/api/projects/{project_id}/features",
                json={
                    "source_dataset_id": dataset["id"],
                    "source_feature_id": "page_rank",
                    "params": {
                        "feature_vector_length": max(1, len(rows[0]) - 2),
                        "normalize_features": False,
                        "n_jobs": 1,
                        "parallel_backend": "loky",
                    },
                },
            )
            assert created_manual.status_code == 200
            manual_id = created_manual.json()["id"]
            manifest = store.read_feature(project_id, manual_id)
            feature_path = store.feature_path(project_id, manual_id)
            output_dir = feature_path / "output"
            output_dir.mkdir(parents=True, exist_ok=False)
            output = pd.DataFrame(rows)
            output.to_parquet(output_dir / "features.parquet", index=False)
            manifest.name = f"{manifest.name} {name_suffix}"
            manifest.status = "completed"
            manifest.output_files = FeatureOutputFiles(features="output/features.parquet")
            manifest.output_stats = FeatureOutputStats(row_count=int(len(output)), column_count=int(len(output.columns)))
            manifest.updated_at = utc_now()
            store._write_json(feature_path / "artifact.json", manifest.model_dump())
            return manual_id

        one_column_feature_id = complete_manual_feature(
            "One Column",
            [
                {"node_id": "0", "graph_id": "g1", "single_0": 1.0},
                {"node_id": "1", "graph_id": "g1", "single_0": 2.0},
                {"node_id": "0", "graph_id": "g2", "single_0": 3.0},
            ],
        )
        three_column_feature_id = complete_manual_feature(
            "Three Columns",
            [
                {"node_id": "0", "graph_id": "g1", "tri_0": 1.0, "tri_1": 0.0, "tri_2": 0.0},
                {"node_id": "1", "graph_id": "g1", "tri_0": 0.0, "tri_1": 1.0, "tri_2": 0.0},
                {"node_id": "0", "graph_id": "g2", "tri_0": 0.0, "tri_1": 0.0, "tri_2": 1.0},
                {"node_id": "1", "graph_id": "g2", "tri_0": 1.0, "tri_1": 1.0, "tri_2": 1.0},
            ],
        )
        large_feature_id = complete_manual_feature(
            "Large",
            [
                {
                    "node_id": "0",
                    "graph_id": f"big{index:04d}",
                    "large_0": float(index),
                    "large_1": float(index % 7),
                    "large_2": float(index % 11),
                }
                for index in range(5001)
            ],
        )

        analysis = client.get(f"/api/projects/{project_id}/features/{feature_id}/analysis")
        assert analysis.status_code == 200
        payload = analysis.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(payload)
        assert payload["feature_id"] == feature_id
        assert payload["feature_name"] == "Tiny Dataset - PageRank"
        assert payload["feature_status"] == "completed"
        assert payload["source_dataset"]["id"] == dataset["id"]
        assert payload["source_dataset"]["name"] == "Tiny Dataset"
        assert payload["method"] == {"id": "page_rank", "name": "PageRank"}
        assert payload["output_stats"] == {"row_count": 4, "column_count": 4}
        assert payload["feature_columns"] == ["page_rank_0", "page_rank_1"]
        assert payload["numeric_feature_columns"] == ["page_rank_0", "page_rank_1"]
        assert [summary["column"] for summary in payload["column_summaries"]] == ["page_rank_0", "page_rank_1"]
        assert all({"min", "max", "mean", "std", "null_count"}.issubset(summary) for summary in payload["column_summaries"])
        assert payload["graph_coverage"] == {"covered": 2, "total": 2}
        assert payload["node_coverage"] == {"covered": 4, "total": 4}
        assert payload["graph_label_distribution"] == {"0": 1, "1": 1}
        assert payload["pca"]["available"] is True
        assert payload["pca"]["plot_level"] == "graph"
        assert payload["pca"]["aggregation_method"] == "mean"
        assert payload["pca"]["projection_method"] == "raw"
        assert payload["pca"]["x_axis_label"] == "page_rank_0"
        assert payload["pca"]["y_axis_label"] == "page_rank_1"
        assert payload["pca"]["color_by"] == "graph_label"
        assert payload["pca"]["source_row_count"] == 4
        assert payload["pca"]["total_graphs"] == 2
        assert payload["pca"]["total_rows"] == 2
        assert payload["pca"]["fit_row_count"] == 0
        assert payload["pca"]["point_count"] == 2
        assert payload["pca"]["sampled"] is False
        assert payload["pca"]["explained_variance_ratio"] == []
        assert len(payload["pca"]["points"]) == 2
        assert {point["graph_label"] for point in payload["pca"]["points"]} == {0, 1}
        assert {point["color_value"] for point in payload["pca"]["points"]} == {"0", "1"}
        assert {point["node_count"] for point in payload["pca"]["points"]} == {2}
        output_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "features" / feature_id / "output" / "features.parquet"
        output_df = pd.read_parquet(output_path)
        graph_means = (
            output_df.assign(_graph_id_str=output_df["graph_id"].map(str))
            .groupby("_graph_id_str")[["page_rank_0", "page_rank_1"]]
            .mean()
            .sort_index()
        )
        points_by_graph = {point["graph_id"]: point for point in payload["pca"]["points"]}
        assert [points_by_graph[graph_id]["x"] for graph_id in graph_means.index] == pytest.approx(graph_means["page_rank_0"].tolist())
        assert [points_by_graph[graph_id]["y"] for graph_id in graph_means.index] == pytest.approx(graph_means["page_rank_1"].tolist())

        unavailable = client.get(f"/api/projects/{project_id}/features/{one_column_feature_id}/analysis")
        assert unavailable.status_code == 200
        unavailable_payload = unavailable.json()
        assert unavailable_payload["pca"]["available"] is False
        assert "at least two numeric feature columns" in unavailable_payload["pca"]["reason"]

        projected = client.get(f"/api/projects/{project_id}/features/{three_column_feature_id}/analysis")
        assert projected.status_code == 200
        projected_payload = projected.json()
        assert projected_payload["pca"]["available"] is True
        assert projected_payload["pca"]["projection_method"] == "pca"
        assert projected_payload["pca"]["x_axis_label"] == "PC1"
        assert projected_payload["pca"]["y_axis_label"] == "PC2"
        assert len(projected_payload["pca"]["explained_variance_ratio"]) == 2

        sampled = client.get(f"/api/projects/{project_id}/features/{feature_id}/analysis?max_fit_rows=2&max_points=1")
        assert sampled.status_code == 200
        sampled_payload = sampled.json()
        assert sampled_payload["pca"]["projection_method"] == "raw"
        assert sampled_payload["pca"]["fit_sampled"] is False
        assert sampled_payload["pca"]["points_sampled"] is True
        assert sampled_payload["pca"]["sampled"] is True
        assert sampled_payload["pca"]["fit_row_count"] == 0
        assert sampled_payload["pca"]["point_count"] == 1
        assert len(sampled_payload["pca"]["points"]) == 1

        sampled_pca = client.get(f"/api/projects/{project_id}/features/{three_column_feature_id}/analysis?max_fit_rows=2&max_points=1")
        assert sampled_pca.status_code == 200
        sampled_pca_payload = sampled_pca.json()
        assert sampled_pca_payload["pca"]["projection_method"] == "pca"
        assert sampled_pca_payload["pca"]["fit_sampled"] is False
        assert sampled_pca_payload["pca"]["points_sampled"] is True
        assert sampled_pca_payload["pca"]["sampled"] is True
        assert sampled_pca_payload["pca"]["fit_row_count"] == 2
        assert sampled_pca_payload["pca"]["point_count"] == 1

        default_sampled = client.get(f"/api/projects/{project_id}/features/{large_feature_id}/analysis")
        assert default_sampled.status_code == 200
        default_sampled_payload = default_sampled.json()
        assert default_sampled_payload["pca"]["projection_method"] == "pca"
        assert default_sampled_payload["pca"]["max_points"] == 5000
        assert default_sampled_payload["pca"]["max_fit_rows"] == 5000
        assert default_sampled_payload["pca"]["total_graphs"] == 5001
        assert default_sampled_payload["pca"]["fit_sampled"] is True
        assert default_sampled_payload["pca"]["points_sampled"] is True
        assert default_sampled_payload["pca"]["fit_row_count"] == 5000
        assert default_sampled_payload["pca"]["point_count"] == 5000
        assert len(default_sampled_payload["pca"]["points"]) == 5000

        out_of_sample_search = client.get(f"/api/projects/{project_id}/features/{large_feature_id}/analysis/search?query=big5000")
        assert out_of_sample_search.status_code == 200
        out_of_sample_payload = out_of_sample_search.json()
        assert out_of_sample_payload["results"]
        assert out_of_sample_payload["results"][0]["graph_id"] == "big5000"
        assert out_of_sample_payload["results"][0]["kind"] == "graph"
        assert out_of_sample_payload["results"][0]["node_count"] == 1
        assert out_of_sample_payload["results"][0]["in_pca_sample"] is False

        graph_search = client.get(f"/api/projects/{project_id}/features/{feature_id}/analysis/search?query=g1")
        assert graph_search.status_code == 200
        graph_payload = graph_search.json()
        assert graph_payload["total_matches"] >= 1
        assert graph_payload["results"][0]["kind"] == "graph"
        assert graph_payload["results"][0]["graph_id"] == "g1"
        assert graph_payload["results"][0]["graph_label"] == 0
        assert graph_payload["results"][0]["in_pca_sample"] is True

        label_search = client.get(f"/api/projects/{project_id}/features/{feature_id}/analysis/search?query=0")
        assert label_search.status_code == 200
        label_payload = label_search.json()
        assert label_payload["results"]
        assert label_payload["results"][0]["kind"] == "graph"
        assert label_payload["results"][0]["graph_label"] == 0
        assert label_payload["results"][0]["in_pca_sample"] is True

        graph_detail = client.get(f"/api/projects/{project_id}/features/{feature_id}/analysis/graph?graph_id=g1")
        assert graph_detail.status_code == 200
        graph_detail_payload = graph_detail.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(graph_detail_payload)
        assert graph_detail_payload["graph_id"] == "g1"
        assert graph_detail_payload["graph_label"] == 0
        assert graph_detail_payload["node_count"] == 2
        assert graph_detail_payload["aggregation_method"] == "mean"
        assert set(graph_detail_payload["feature_values"]) == {"page_rank_0", "page_rank_1"}
        assert graph_detail_payload["feature_values"]["page_rank_0"] == pytest.approx(graph_means.loc["g1", "page_rank_0"])
        assert graph_detail_payload["feature_values"]["page_rank_1"] == pytest.approx(graph_means.loc["g1", "page_rank_1"])


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


def test_embedding_analysis_search_and_graph_detail(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")
    pytest.importorskip("sklearn")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.schemas import EmbeddingOutputFiles, EmbeddingOutputStats, FeatureOutputFiles, FeatureOutputStats
    from NEExT.workbench.storage import utc_now

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
        project = client.post("/api/projects", json={"name": "Embedding Analysis Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "TINY"}).json()
        dataset_run = client.post(f"/api/projects/{project_id}/datasets/{dataset['id']}/run")
        assert dataset_run.status_code == 200
        assert wait_for_job(client, project_id, dataset_run.json()["id"])["status"] == "completed"

        feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        ).json()
        store = app.state.store
        feature_manifest = store.read_feature(project_id, feature["id"])
        feature_path = store.feature_path(project_id, feature_manifest.id)
        (feature_path / "output").mkdir(parents=True, exist_ok=False)
        feature_output = pd.DataFrame(
            [
                {"node_id": "0", "graph_id": "g1", "page_rank_0": 0.1, "page_rank_1": 0.2},
                {"node_id": "1", "graph_id": "g1", "page_rank_0": 0.2, "page_rank_1": 0.3},
                {"node_id": "0", "graph_id": "g2", "page_rank_0": 0.3, "page_rank_1": 0.4},
                {"node_id": "1", "graph_id": "g2", "page_rank_0": 0.4, "page_rank_1": 0.5},
            ]
        )
        feature_output.to_parquet(feature_path / "output" / "features.parquet", index=False)
        feature_manifest.status = "completed"
        feature_manifest.output_files = FeatureOutputFiles(features="output/features.parquet")
        feature_manifest.output_stats = FeatureOutputStats(row_count=int(len(feature_output)), column_count=int(len(feature_output.columns)))
        feature_manifest.updated_at = utc_now()
        store._write_json(feature_path / "artifact.json", feature_manifest.model_dump())

        blocked_embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature_manifest.id],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        planned_analysis = client.get(f"/api/projects/{project_id}/embeddings/{blocked_embedding['id']}/analysis")
        assert planned_analysis.status_code == 400
        assert "available only after computation completes" in planned_analysis.json()["detail"]

        for status in ["running", "failed"]:
            manifest = store.read_embedding(project_id, blocked_embedding["id"])
            manifest.status = status
            store._write_json(store.embedding_path(project_id, manifest.id) / "artifact.json", manifest.model_dump())
            blocked_analysis = client.get(f"/api/projects/{project_id}/embeddings/{blocked_embedding['id']}/analysis")
            assert blocked_analysis.status_code == 400
            assert "available only after computation completes" in blocked_analysis.json()["detail"]

        def complete_manual_embedding(name_suffix: str, dimension: int, rows: list[dict[str, object]]) -> str:
            created = client.post(
                f"/api/projects/{project_id}/embeddings",
                json={
                    "source_embedding_id": "approx_wasserstein",
                    "source_feature_ids": [feature_manifest.id],
                    "params": {"embedding_dimension": dimension},
                },
            )
            assert created.status_code == 200
            embedding_id = created.json()["id"]
            manifest = store.read_embedding(project_id, embedding_id)
            embedding_path = store.embedding_path(project_id, embedding_id)
            output_dir = embedding_path / "output"
            output_dir.mkdir(parents=True, exist_ok=False)
            output = pd.DataFrame(rows)
            output.to_parquet(output_dir / "embeddings.parquet", index=False)
            manifest.name = f"{manifest.name} {name_suffix}"
            manifest.status = "completed"
            manifest.output_files = EmbeddingOutputFiles(embeddings="output/embeddings.parquet")
            manifest.output_stats = EmbeddingOutputStats(row_count=int(len(output)), column_count=int(len(output.columns)))
            manifest.updated_at = utc_now()
            store._write_json(embedding_path / "artifact.json", manifest.model_dump())
            return embedding_id

        one_dim_embedding_id = complete_manual_embedding(
            "One Dimension",
            1,
            [{"graph_id": "g1", "emb_0": 1.0}, {"graph_id": "g2", "emb_0": 2.0}],
        )
        two_dim_embedding_id = complete_manual_embedding(
            "Two Dimensions",
            2,
            [{"graph_id": "g1", "emb_0": 1.0, "emb_1": 2.0}, {"graph_id": "g2", "emb_0": 3.0, "emb_1": 4.0}],
        )
        three_dim_embedding_id = complete_manual_embedding(
            "Three Dimensions",
            3,
            [
                {"graph_id": "g1", "emb_0": 1.0, "emb_1": 0.0, "emb_2": 0.0},
                {"graph_id": "g2", "emb_0": 0.0, "emb_1": 1.0, "emb_2": 1.0},
            ],
        )
        large_embedding_id = complete_manual_embedding(
            "Large",
            3,
            [{"graph_id": f"big{index:04d}", "emb_0": float(index), "emb_1": float(index % 7), "emb_2": float(index % 11)} for index in range(5001)],
        )

        analysis = client.get(f"/api/projects/{project_id}/embeddings/{two_dim_embedding_id}/analysis")
        assert analysis.status_code == 200
        payload = analysis.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(payload)
        assert payload["embedding_id"] == two_dim_embedding_id
        assert payload["embedding_status"] == "completed"
        assert payload["source_dataset"]["id"] == dataset["id"]
        assert payload["source_dataset"]["name"] == "Tiny Dataset"
        assert payload["source_features"][0]["id"] == feature_manifest.id
        assert payload["source_features"][0]["method"] == {"id": "page_rank", "name": "PageRank"}
        assert payload["algorithm"] == {"id": "approx_wasserstein", "name": "Approx Wasserstein"}
        assert payload["output_stats"] == {"row_count": 2, "column_count": 3}
        assert payload["embedding_columns"] == ["emb_0", "emb_1"]
        assert payload["numeric_embedding_columns"] == ["emb_0", "emb_1"]
        assert [summary["column"] for summary in payload["column_summaries"]] == ["emb_0", "emb_1"]
        assert all({"min", "max", "mean", "std", "null_count"}.issubset(summary) for summary in payload["column_summaries"])
        assert payload["graph_label_distribution"] == {"0": 1, "1": 1}
        assert payload["pca"]["available"] is True
        assert payload["pca"]["projection_method"] == "raw"
        assert payload["pca"]["x_axis_label"] == "emb_0"
        assert payload["pca"]["y_axis_label"] == "emb_1"
        assert payload["pca"]["color_by"] == "graph_label"
        assert payload["pca"]["total_graphs"] == 2
        assert payload["pca"]["total_rows"] == 2
        assert payload["pca"]["fit_row_count"] == 0
        assert payload["pca"]["point_count"] == 2
        assert payload["pca"]["sampled"] is False
        assert payload["pca"]["explained_variance_ratio"] == []
        points_by_graph = {point["graph_id"]: point for point in payload["pca"]["points"]}
        assert points_by_graph["g1"]["x"] == pytest.approx(1.0)
        assert points_by_graph["g1"]["y"] == pytest.approx(2.0)
        assert {point["graph_label"] for point in payload["pca"]["points"]} == {0, 1}
        assert {point["color_value"] for point in payload["pca"]["points"]} == {"0", "1"}

        unavailable = client.get(f"/api/projects/{project_id}/embeddings/{one_dim_embedding_id}/analysis")
        assert unavailable.status_code == 200
        unavailable_payload = unavailable.json()
        assert unavailable_payload["pca"]["available"] is False
        assert "at least two numeric embedding columns" in unavailable_payload["pca"]["reason"]

        projected = client.get(f"/api/projects/{project_id}/embeddings/{three_dim_embedding_id}/analysis")
        assert projected.status_code == 200
        projected_payload = projected.json()
        assert projected_payload["pca"]["available"] is True
        assert projected_payload["pca"]["projection_method"] == "pca"
        assert projected_payload["pca"]["x_axis_label"] == "PC1"
        assert projected_payload["pca"]["y_axis_label"] == "PC2"
        assert len(projected_payload["pca"]["explained_variance_ratio"]) == 2

        sampled = client.get(f"/api/projects/{project_id}/embeddings/{three_dim_embedding_id}/analysis?max_fit_rows=2&max_points=1")
        assert sampled.status_code == 200
        sampled_payload = sampled.json()
        assert sampled_payload["pca"]["projection_method"] == "pca"
        assert sampled_payload["pca"]["fit_sampled"] is False
        assert sampled_payload["pca"]["points_sampled"] is True
        assert sampled_payload["pca"]["sampled"] is True
        assert sampled_payload["pca"]["fit_row_count"] == 2
        assert sampled_payload["pca"]["point_count"] == 1

        default_sampled = client.get(f"/api/projects/{project_id}/embeddings/{large_embedding_id}/analysis")
        assert default_sampled.status_code == 200
        default_sampled_payload = default_sampled.json()
        assert default_sampled_payload["pca"]["projection_method"] == "pca"
        assert default_sampled_payload["pca"]["max_points"] == 5000
        assert default_sampled_payload["pca"]["max_fit_rows"] == 5000
        assert default_sampled_payload["pca"]["total_graphs"] == 5001
        assert default_sampled_payload["pca"]["fit_sampled"] is True
        assert default_sampled_payload["pca"]["points_sampled"] is True
        assert default_sampled_payload["pca"]["fit_row_count"] == 5000
        assert default_sampled_payload["pca"]["point_count"] == 5000
        assert len(default_sampled_payload["pca"]["points"]) == 5000

        out_of_sample_search = client.get(f"/api/projects/{project_id}/embeddings/{large_embedding_id}/analysis/search?query=big5000")
        assert out_of_sample_search.status_code == 200
        out_of_sample_payload = out_of_sample_search.json()
        assert out_of_sample_payload["results"]
        assert out_of_sample_payload["results"][0]["graph_id"] == "big5000"
        assert out_of_sample_payload["results"][0]["kind"] == "graph"
        assert out_of_sample_payload["results"][0]["in_pca_sample"] is False

        graph_search = client.get(f"/api/projects/{project_id}/embeddings/{two_dim_embedding_id}/analysis/search?query=g1")
        assert graph_search.status_code == 200
        graph_payload = graph_search.json()
        assert graph_payload["total_matches"] >= 1
        assert graph_payload["results"][0]["kind"] == "graph"
        assert graph_payload["results"][0]["graph_id"] == "g1"
        assert graph_payload["results"][0]["graph_label"] == 0
        assert graph_payload["results"][0]["in_pca_sample"] is True

        label_search = client.get(f"/api/projects/{project_id}/embeddings/{two_dim_embedding_id}/analysis/search?query=0")
        assert label_search.status_code == 200
        label_payload = label_search.json()
        assert label_payload["results"]
        assert label_payload["results"][0]["kind"] == "graph"
        assert label_payload["results"][0]["graph_label"] == 0
        assert label_payload["results"][0]["in_pca_sample"] is True

        graph_detail = client.get(f"/api/projects/{project_id}/embeddings/{two_dim_embedding_id}/analysis/graph?graph_id=g1")
        assert graph_detail.status_code == 200
        graph_detail_payload = graph_detail.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(graph_detail_payload)
        assert graph_detail_payload["graph_id"] == "g1"
        assert graph_detail_payload["graph_label"] == 0
        assert graph_detail_payload["in_pca_sample"] is True
        assert graph_detail_payload["embedding_values"] == {"emb_0": 1.0, "emb_1": 2.0}


def test_model_library_catalog_endpoint_exposes_built_ins_without_paths_or_code():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        response = client.get("/api/model-library")

        assert response.status_code == 200
        catalog = response.json()
        assert {entry["id"] for entry in catalog} == {"xgboost", "random_forest"}
        assert len(catalog) == 2
        random_forest = next(entry for entry in catalog if entry["id"] == "random_forest")
        assert random_forest["name"] == "Random Forest"
        assert random_forest["operation_id"] == "neext.train_graph_model"
        assert random_forest["operation_version"] == "1"
        for entry in catalog:
            payload = json.dumps(entry)
            assert "files" not in entry
            assert "path" not in entry
            assert "callable" not in entry
            assert "function" not in entry
            assert "http://" not in payload
            assert "https://" not in payload


def test_create_model_artifacts_validate_source_embeddings_and_dataset_lineage(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED",
                    name="Labeled Dataset",
                    description="local labeled bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Model Project"}).json()
        project_id = project["id"]
        first_dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()
        second_dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()

        first_features = []
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
            first_features.append(created.json()["id"])

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

        first_embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [first_features[0]],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        second_embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [first_features[1]],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        cross_dataset_embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [second_feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()

        created = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [first_embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        )
        assert created.status_code == 200
        model = created.json()
        model_id = model["id"]
        assert_uuid4(model_id)
        assert model["schema_version"] == "1"
        assert model["manifest_type"] == "model"
        assert model["project_id"] == project_id
        assert model["name"] == "Labeled Dataset - Random Forest Classifier"
        assert model["status"] == "planned"
        assert model["source_type"] == "neext_supervised_graph_model"
        assert model["source_model_id"] == "random_forest"
        assert model["inputs"] == [{"role": "source_embedding", "artifact_kind": "embedding", "artifact_id": first_embedding["id"]}]
        assert model["operation"] == {
            "operation_id": "neext.train_graph_model",
            "operation_version": "1",
            "params": {
                "model_algorithm": "random_forest",
                "task_type": "classifier",
                "sample_size": 1,
                "test_size": 0.5,
                "balance_dataset": False,
                "random_state": 42,
                "n_jobs": 1,
                "parallel_backend": "thread",
                "embedding_ids": [first_embedding["id"]],
                "embedding_columns": "all",
            },
        }
        assert model["expected_output"] == {
            "artifact_kind": "model",
            "storage_format": "neext-model-results-v1",
            "metrics": ["accuracy", "recall", "precision", "f1_score"],
        }
        assert model["output_files"] is None
        assert model["output_stats"] is None
        assert str(Path(tmpdir).resolve()) not in json.dumps(model)

        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "models" / model_id
        assert (artifact_path / "artifact.json").is_file()
        assert json.loads((artifact_path / "artifact.json").read_text(encoding="utf-8")) == model

        multi = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [first_embedding["id"], second_embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        )
        assert multi.status_code == 200
        assert [item["artifact_id"] for item in multi.json()["inputs"]] == [first_embedding["id"], second_embedding["id"]]

        cross_dataset = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [first_embedding["id"], cross_dataset_embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        )
        assert cross_dataset.status_code == 400
        assert "same dataset" in cross_dataset.json()["detail"]

        duplicate_embeddings = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [first_embedding["id"], first_embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        )
        assert duplicate_embeddings.status_code == 400
        assert "unique" in duplicate_embeddings.json()["detail"]


def test_run_model_auto_runs_planned_upstream_artifacts_and_writes_preview(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED",
                    name="Labeled Dataset",
                    description="local labeled bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        project = client.post("/api/projects", json={"name": "Model Run Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()
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
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        model = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        ).json()

        class DummyEmbeddings:
            def __init__(self, embeddings_df):
                self.embeddings_df = embeddings_df

        def fake_graph_embeddings(collection, features, params):
            records = []
            for graph_index, graph in enumerate(collection.graphs):
                label_offset = float(graph.graph_label) * 10.0
                records.append(
                    {
                        "graph_id": graph.graph_id,
                        "emb_0": label_offset + graph_index,
                        "emb_1": label_offset + graph_index + 0.5,
                    }
                )
            return DummyEmbeddings(pd.DataFrame(records))

        monkeypatch.setattr(app.state.store, "_compute_graph_embeddings", fake_graph_embeddings)

        run = client.post(f"/api/projects/{project_id}/models/{model['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert job["operation"]["operation_id"] == "neext.train_graph_model"
        assert job["target_artifacts"] == [{"artifact_kind": "model", "artifact_id": model["id"]}]
        assert f"Computing upstream embeddings for model {model['name']}" in job["log"]
        assert f"Computing upstream features for embedding {embedding['name']}" in job["log"]
        assert f"Training model {model['name']} with random_forest" in job["log"]

        assert client.get(f"/api/projects/{project_id}/datasets/{dataset['id']}").json()["status"] == "completed"
        assert client.get(f"/api/projects/{project_id}/features/{feature['id']}").json()["status"] == "completed"
        assert client.get(f"/api/projects/{project_id}/embeddings/{embedding['id']}").json()["status"] == "completed"

        completed = client.get(f"/api/projects/{project_id}/models/{model['id']}").json()
        assert completed["status"] == "completed"
        assert completed["output_files"] == {"metrics": "output/metrics.json", "model": "output/model.joblib"}
        assert completed["output_stats"] == {
            "metric_count": 4,
            "sample_size": 1,
            "feature_count": 2,
            "graph_count": 8,
        }
        assert completed["error"] is None
        assert str(Path(tmpdir).resolve()) not in json.dumps(completed)

        output_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "models" / model["id"] / "output"
        metrics = json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["model_type"] == "classifier"
        assert metrics["model_name"] == "random_forest"
        assert metrics["sample_size"] == 1
        assert metrics["test_size"] == 0.5
        assert metrics["random_state"] == 42
        assert metrics["classes"] == ["0", "1"]
        assert len(metrics["feature_columns"]) == 2
        assert set(metrics["summary"]) == {
            "accuracy_mean",
            "recall_mean",
            "precision_mean",
            "f1_score_mean",
            "accuracy_std",
            "recall_std",
            "precision_std",
            "f1_score_std",
        }
        assert len(metrics["metrics"]) == 1
        assert metrics["metrics"][0]["iteration"] == 0
        assert (output_path / "model.joblib").is_file()

        preview = client.get(f"/api/projects/{project_id}/models/{model['id']}/preview")
        assert preview.status_code == 200
        assert preview.json()["summary"] == metrics["summary"]
        assert preview.json()["metrics"] == metrics["metrics"]
        assert preview.json()["feature_columns"] == metrics["feature_columns"]
        assert preview.json()["classes"] == ["0", "1"]

        jobs = client.get(f"/api/projects/{project_id}/jobs").json()
        assert jobs[0]["id"] == job["id"]
        assert "Model Labeled Dataset - Random Forest Classifier completed" in "\n".join(jobs[0]["log"])


def test_model_analysis_exposes_metrics_and_lineage_without_paths(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.schemas import ModelOutputFiles, ModelOutputStats
    from NEExT.workbench.storage import utc_now

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED",
                    name="Labeled Dataset",
                    description="local labeled bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        store = app.state.store
        project = client.post("/api/projects", json={"name": "Model Analysis Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()
        dataset_run = client.post(f"/api/projects/{project_id}/datasets/{dataset['id']}/run")
        assert dataset_run.status_code == 200
        assert wait_for_job(client, project_id, dataset_run.json()["id"])["status"] == "completed"

        feature = client.post(
            f"/api/projects/{project_id}/features",
            json={
                "source_dataset_id": dataset["id"],
                "source_feature_id": "page_rank",
                "params": {
                    "feature_vector_length": 2,
                    "normalize_features": False,
                    "n_jobs": 1,
                    "parallel_backend": "loky",
                },
            },
        ).json()
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()

        blocked_model = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        ).json()
        planned_analysis = client.get(f"/api/projects/{project_id}/models/{blocked_model['id']}/analysis")
        assert planned_analysis.status_code == 400
        assert "available only after training completes" in planned_analysis.json()["detail"]

        for status in ["running", "failed"]:
            manifest = store.read_model(project_id, blocked_model["id"])
            manifest.status = status
            store._write_json(store.model_path(project_id, manifest.id) / "artifact.json", manifest.model_dump())
            blocked_analysis = client.get(f"/api/projects/{project_id}/models/{blocked_model['id']}/analysis")
            assert blocked_analysis.status_code == 400
            assert "available only after training completes" in blocked_analysis.json()["detail"]

        def complete_manual_model(task_type: str, metrics_payload: dict[str, object]) -> str:
            created = client.post(
                f"/api/projects/{project_id}/models",
                json={
                    "source_model_id": "random_forest",
                    "source_embedding_ids": [embedding["id"]],
                    "params": {
                        "task_type": task_type,
                        "sample_size": len(metrics_payload["metrics"]),
                        "test_size": metrics_payload["test_size"],
                        "balance_dataset": False,
                        "n_jobs": 1,
                        "parallel_backend": "thread",
                    },
                },
            )
            assert created.status_code == 200
            model_id = created.json()["id"]
            manifest = store.read_model(project_id, model_id)
            model_path = store.model_path(project_id, model_id)
            output_dir = model_path / "output"
            output_dir.mkdir(parents=True, exist_ok=False)
            store._write_json(output_dir / "metrics.json", metrics_payload)
            (output_dir / "model.joblib").write_bytes(b"placeholder")
            manifest.status = "completed"
            manifest.output_files = ModelOutputFiles(metrics="output/metrics.json", model="output/model.joblib")
            manifest.output_stats = ModelOutputStats(
                metric_count=len(manifest.expected_output.metrics),
                sample_size=len(metrics_payload["metrics"]),
                feature_count=len(metrics_payload["feature_columns"]),
                graph_count=8,
            )
            manifest.updated_at = utc_now()
            store._write_json(model_path / "artifact.json", manifest.model_dump())
            return model_id

        classifier_model_id = complete_manual_model(
            "classifier",
            {
                "model_type": "classifier",
                "model_name": "random_forest",
                "sample_size": 2,
                "test_size": 0.25,
                "random_state": 42,
                "feature_columns": ["emb_0", "emb_1"],
                "classes": ["0", "1"],
                "summary": {
                    "accuracy_mean": 0.75,
                    "accuracy_std": 0.25,
                    "recall_mean": 0.5,
                    "recall_std": 0.5,
                    "precision_mean": 0.6,
                    "precision_std": 0.4,
                    "f1_score_mean": 0.55,
                    "f1_score_std": 0.45,
                },
                "metrics": [
                    {"iteration": 0, "accuracy": 1.0, "recall": 1.0, "precision": 1.0, "f1_score": 1.0},
                    {"iteration": 1, "accuracy": 0.5, "recall": 0.0, "precision": 0.2, "f1_score": 0.1},
                ],
            },
        )
        regressor_model_id = complete_manual_model(
            "regressor",
            {
                "model_type": "regressor",
                "model_name": "random_forest",
                "sample_size": 2,
                "test_size": 0.25,
                "random_state": 42,
                "feature_columns": ["emb_0", "emb_1"],
                "classes": None,
                "summary": {"rmse_mean": 1.5, "rmse_std": 0.5, "mae_mean": 1.0, "mae_std": 0.25},
                "metrics": [
                    {"iteration": 0, "rmse": 1.0, "mae": 0.75},
                    {"iteration": 1, "rmse": 2.0, "mae": 1.25},
                ],
            },
        )

        analysis = client.get(f"/api/projects/{project_id}/models/{classifier_model_id}/analysis")
        assert analysis.status_code == 200
        payload = analysis.json()
        assert str(Path(tmpdir).resolve()) not in json.dumps(payload)
        assert payload["model_id"] == classifier_model_id
        assert payload["model_status"] == "completed"
        assert payload["source_dataset"]["id"] == dataset["id"]
        assert payload["source_dataset"]["name"] == "Labeled Dataset"
        assert payload["source_embeddings"][0]["id"] == embedding["id"]
        assert payload["source_embeddings"][0]["algorithm"] == {"id": "approx_wasserstein", "name": "Approx Wasserstein"}
        assert payload["source_features"][0]["id"] == feature["id"]
        assert payload["source_features"][0]["method"] == {"id": "page_rank", "name": "PageRank"}
        assert payload["algorithm"] == {"id": "random_forest", "name": "Random Forest"}
        assert payload["task_type"] == "classifier"
        assert payload["expected_metrics"] == ["accuracy", "recall", "precision", "f1_score"]
        assert payload["output_stats"] == {"metric_count": 4, "sample_size": 2, "feature_count": 2, "graph_count": 8}
        assert payload["sample_size"] == 2
        assert payload["test_size"] == 0.25
        assert payload["random_state"] == 42
        assert payload["classes"] == ["0", "1"]
        assert payload["feature_columns"] == ["emb_0", "emb_1"]
        assert payload["summary"]["accuracy_mean"] == 0.75
        assert payload["metrics"][1]["iteration"] == 1
        series_by_metric = {series["metric"]: series for series in payload["metric_series"]}
        assert series_by_metric["accuracy"]["points"] == [{"iteration": 0, "value": 1.0}, {"iteration": 1, "value": 0.5}]
        assert series_by_metric["f1_score"]["points"][1] == {"iteration": 1, "value": 0.1}

        regressor_analysis = client.get(f"/api/projects/{project_id}/models/{regressor_model_id}/analysis")
        assert regressor_analysis.status_code == 200
        regressor_payload = regressor_analysis.json()
        assert regressor_payload["task_type"] == "regressor"
        assert regressor_payload["expected_metrics"] == ["rmse", "mae"]
        assert regressor_payload["classes"] is None
        regressor_series = {series["metric"]: series for series in regressor_payload["metric_series"]}
        assert regressor_series["rmse"]["points"] == [{"iteration": 0, "value": 1.0}, {"iteration": 1, "value": 2.0}]


def test_model_batch_run_completes_selected_outputs_in_one_job(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED",
                    name="Labeled Dataset",
                    description="local labeled bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        project = client.post("/api/projects", json={"name": "Model Batch Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()
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
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()

        class DummyEmbeddings:
            def __init__(self, embeddings_df):
                self.embeddings_df = embeddings_df

        def fake_graph_embeddings(collection, features, params):
            return DummyEmbeddings(
                pd.DataFrame(
                    [{"graph_id": graph.graph_id, "emb_0": float(index), "emb_1": float(index % 2)} for index, graph in enumerate(collection.graphs)]
                )
            )

        monkeypatch.setattr(app.state.store, "_compute_graph_embeddings", fake_graph_embeddings)

        model_ids = []
        for _ in range(2):
            model = client.post(
                f"/api/projects/{project_id}/models",
                json={
                    "source_model_id": "random_forest",
                    "source_embedding_ids": [embedding["id"]],
                    "params": {
                        "task_type": "classifier",
                        "sample_size": 1,
                        "test_size": 0.5,
                        "balance_dataset": False,
                        "n_jobs": 1,
                        "parallel_backend": "thread",
                    },
                },
            ).json()
            model_ids.append(model["id"])

        duplicate_run = client.post(f"/api/projects/{project_id}/models/run-batch", json={"model_ids": [model_ids[0], model_ids[0]]})
        assert duplicate_run.status_code == 400
        assert "unique" in duplicate_run.json()["detail"]

        run = client.post(f"/api/projects/{project_id}/models/run-batch", json={"model_ids": model_ids})
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "completed"
        assert job["target_artifacts"] == [
            {"artifact_kind": "model", "artifact_id": model_ids[0]},
            {"artifact_kind": "model", "artifact_id": model_ids[1]},
        ]

        for model_id in model_ids:
            completed = client.get(f"/api/projects/{project_id}/models/{model_id}").json()
            assert completed["status"] == "completed"
            assert completed["output_files"] == {"metrics": "output/metrics.json", "model": "output/model.joblib"}


def test_regressor_model_run_fails_clearly_for_non_numeric_labels(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source", numeric_labels=False)
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="CATEGORICAL",
                    name="Categorical Dataset",
                    description="local categorical bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        project = client.post("/api/projects", json={"name": "Regressor Failure Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "CATEGORICAL"}).json()
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
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        model = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [embedding["id"]],
                "params": {
                    "task_type": "regressor",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        ).json()

        class DummyEmbeddings:
            def __init__(self, embeddings_df):
                self.embeddings_df = embeddings_df

        monkeypatch.setattr(
            app.state.store,
            "_compute_graph_embeddings",
            lambda collection, features, params: DummyEmbeddings(
                pd.DataFrame(
                    [{"graph_id": graph.graph_id, "emb_0": float(index), "emb_1": float(index + 1)} for index, graph in enumerate(collection.graphs)]
                )
            ),
        )

        run = client.post(f"/api/projects/{project_id}/models/{model['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "failed"
        assert "numeric graph labels" in job["error"]

        failed = client.get(f"/api/projects/{project_id}/models/{model['id']}").json()
        assert failed["status"] == "failed"
        assert "numeric graph labels" in failed["error"]["message"]
        assert failed["output_files"] is None


def test_failed_model_execution_sets_error_and_cleans_partial_output(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED",
                    name="Labeled Dataset",
                    description="local labeled bundle",
                    domain="Tests",
                    files=source_files,
                    graph_count=8,
                    node_count=24,
                    edge_count=16,
                    source="Test catalog",
                ),
            ),
        )

        app = create_app(tmpdir)
        client = TestClient(app)
        project = client.post("/api/projects", json={"name": "Model Failure Project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/projects/{project_id}/datasets", json={"catalog_id": "LABELED"}).json()
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
        embedding = client.post(
            f"/api/projects/{project_id}/embeddings",
            json={
                "source_embedding_id": "approx_wasserstein",
                "source_feature_ids": [feature["id"]],
                "params": {"embedding_dimension": 2},
            },
        ).json()
        model = client.post(
            f"/api/projects/{project_id}/models",
            json={
                "source_model_id": "random_forest",
                "source_embedding_ids": [embedding["id"]],
                "params": {
                    "task_type": "classifier",
                    "sample_size": 1,
                    "test_size": 0.5,
                    "balance_dataset": False,
                    "n_jobs": 1,
                    "parallel_backend": "thread",
                },
            },
        ).json()

        class DummyEmbeddings:
            def __init__(self, embeddings_df):
                self.embeddings_df = embeddings_df

        monkeypatch.setattr(
            app.state.store,
            "_compute_graph_embeddings",
            lambda collection, features, params: DummyEmbeddings(
                pd.DataFrame(
                    [{"graph_id": graph.graph_id, "emb_0": float(index), "emb_1": float(index + 1)} for index, graph in enumerate(collection.graphs)]
                )
            ),
        )

        class DummyModel:
            pass

        monkeypatch.setattr(
            app.state.store,
            "_compute_graph_model",
            lambda collection, merged_df, feature_columns, model: {
                "model_type": "classifier",
                "accuracy": [1.0],
                "accuracy_mean": 1.0,
                "accuracy_std": 0.0,
                "recall": [1.0],
                "recall_mean": 1.0,
                "recall_std": 0.0,
                "precision": [1.0],
                "precision_mean": 1.0,
                "precision_std": 0.0,
                "f1_score": [1.0],
                "f1_score_mean": 1.0,
                "f1_score_std": 0.0,
                "model": DummyModel(),
                "classes": [0, 1],
                "feature_columns": feature_columns,
            },
        )

        def fail_writer(project_id: str, model_id: str, metrics_payload, trained_model):
            artifact_path = app.state.store.model_path(project_id, model_id)
            tmp_path = artifact_path / "_tmp" / "output"
            tmp_path.mkdir(parents=True, exist_ok=True)
            (tmp_path / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("forced model write failure")

        monkeypatch.setattr(app.state.store, "_write_model_output", fail_writer)

        run = client.post(f"/api/projects/{project_id}/models/{model['id']}/run")
        assert run.status_code == 200
        job = wait_for_job(client, project_id, run.json()["id"])
        assert job["status"] == "failed"
        assert "forced model write failure" in job["error"]

        failed = client.get(f"/api/projects/{project_id}/models/{model['id']}").json()
        assert failed["status"] == "failed"
        assert "forced model write failure" in failed["error"]["message"]
        assert failed["output_files"] is None
        artifact_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "models" / model["id"]
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


def test_artifact_deletion_plan_cascade_delete_and_restore(monkeypatch):
    pytest.importorskip("fastapi")

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
        project = client.post("/api/projects", json={"name": "Lifecycle Project"}).json()
        project_id = project["id"]
        chain = create_planned_lifecycle_chain(client, project_id)

        plan = client.get(f"/api/projects/{project_id}/datasets/{chain['dataset']['id']}/delete-plan")
        assert plan.status_code == 200
        plan_payload = plan.json()
        assert plan_payload["requires_cascade"] is True
        assert plan_payload["can_delete"] is True
        assert [item["artifact_kind"] for item in plan_payload["downstream_artifacts"]] == ["feature", "embedding", "model"]
        assert [item["artifact_id"] for item in plan_payload["artifacts"]] == [
            chain["dataset"]["id"],
            chain["feature"]["id"],
            chain["embedding"]["id"],
            chain["model"]["id"],
        ]

        blocked = client.delete(f"/api/projects/{project_id}/datasets/{chain['dataset']['id']}")
        assert blocked.status_code == 409
        assert "downstream artifacts" in blocked.json()["detail"]["message"]
        assert blocked.json()["detail"]["deletion_plan"]["root_artifact"]["artifact_id"] == chain["dataset"]["id"]

        deleted = client.delete(f"/api/projects/{project_id}/datasets/{chain['dataset']['id']}?cascade=true")
        assert deleted.status_code == 200
        summary = deleted.json()
        assert_uuid4(summary["bundle_id"])
        assert summary["root_artifact"]["artifact_kind"] == "dataset"
        assert summary["trashed_path"].startswith(f"trash/projects/{project_id}/artifact-deletions/")
        assert str(Path(tmpdir).resolve()) not in json.dumps(summary)
        assert not (Path(tmpdir) / "projects" / project_id / "artifacts" / "datasets" / chain["dataset"]["id"]).exists()
        assert not (Path(tmpdir) / "projects" / project_id / "artifacts" / "features" / chain["feature"]["id"]).exists()
        assert client.get(f"/api/projects/{project_id}/datasets/{chain['dataset']['id']}").status_code == 404
        assert client.get(f"/api/projects/{project_id}/features/{chain['feature']['id']}").status_code == 404
        assert client.get(f"/api/projects/{project_id}/embeddings/{chain['embedding']['id']}").status_code == 404
        assert client.get(f"/api/projects/{project_id}/models/{chain['model']['id']}").status_code == 404

        trash = client.get("/api/trash")
        assert trash.status_code == 200
        assert trash.json()["projects"] == []
        bundles = trash.json()["artifact_deletions"]
        assert [bundle["id"] for bundle in bundles] == [summary["bundle_id"]]
        assert bundles[0]["delete_mode"] == "cascade"
        assert [item["artifact_kind"] for item in bundles[0]["artifacts"]] == ["dataset", "feature", "embedding", "model"]
        assert str(Path(tmpdir).resolve()) not in json.dumps(trash.json())

        restored = client.post(f"/api/projects/{project_id}/trash/artifact-deletions/{summary['bundle_id']}/restore")
        assert restored.status_code == 200
        assert restored.json() == {
            "restored_kind": "artifact_deletion",
            "id": summary["bundle_id"],
            "name": "Tiny Dataset",
            "restored_path": f"projects/{project_id}",
        }
        assert client.get(f"/api/projects/{project_id}/datasets/{chain['dataset']['id']}").status_code == 200
        assert client.get(f"/api/projects/{project_id}/features/{chain['feature']['id']}").status_code == 200
        assert client.get(f"/api/projects/{project_id}/embeddings/{chain['embedding']['id']}").status_code == 200
        assert client.get(f"/api/projects/{project_id}/models/{chain['model']['id']}").status_code == 200
        assert client.get("/api/trash").json()["artifact_deletions"] == []


def test_leaf_artifact_delete_keeps_completed_job_history_and_restore_blocks_on_collision(monkeypatch):
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.schemas import JobArtifactRef, OperationSpec

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
        project = client.post("/api/projects", json={"name": "Leaf Delete Project"}).json()
        project_id = project["id"]
        chain = create_planned_lifecycle_chain(client, project_id)
        store = client.app.state.store
        job = store._create_job(
            project_id,
            OperationSpec(operation_id="neext.train_graph_model", operation_version="1", params={"model_ids": [chain["model"]["id"]]}),
            [JobArtifactRef(artifact_kind="model", artifact_id=chain["model"]["id"])],
        )
        store._mark_job_completed(project_id, job.id)

        deleted = client.delete(f"/api/projects/{project_id}/models/{chain['model']['id']}")
        assert deleted.status_code == 200
        bundle_id = deleted.json()["bundle_id"]
        assert client.get(f"/api/projects/{project_id}/models/{chain['model']['id']}").status_code == 404
        jobs = client.get(f"/api/projects/{project_id}/jobs")
        assert jobs.status_code == 200
        assert jobs.json()[0]["id"] == job.id
        assert jobs.json()[0]["status"] == "completed"

        live_collision_path = Path(tmpdir) / "projects" / project_id / "artifacts" / "models" / chain["model"]["id"]
        live_collision_path.mkdir(parents=True)
        (live_collision_path / "sentinel.txt").write_text("collision", encoding="utf-8")
        blocked_restore = client.post(f"/api/projects/{project_id}/trash/artifact-deletions/{bundle_id}/restore")
        assert blocked_restore.status_code == 409
        assert "live artifact folders already exist" in blocked_restore.json()["detail"]["message"]
        assert blocked_restore.json()["detail"]["conflicts"][0]["artifact_id"] == chain["model"]["id"]


def test_artifact_delete_blocks_queued_or_running_target_jobs(monkeypatch):
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.app import create_app
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.schemas import JobArtifactRef, OperationSpec

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
        project = client.post("/api/projects", json={"name": "Active Job Project"}).json()
        project_id = project["id"]
        chain = create_planned_lifecycle_chain(client, project_id)
        store = client.app.state.store
        job = store._create_job(
            project_id,
            OperationSpec(operation_id="neext.train_graph_model", operation_version="1", params={"model_ids": [chain["model"]["id"]]}),
            [JobArtifactRef(artifact_kind="model", artifact_id=chain["model"]["id"])],
        )

        plan = client.get(f"/api/projects/{project_id}/models/{chain['model']['id']}/delete-plan")
        assert plan.status_code == 200
        assert plan.json()["can_delete"] is False
        assert plan.json()["active_jobs"][0]["id"] == job.id

        blocked = client.delete(f"/api/projects/{project_id}/models/{chain['model']['id']}")
        assert blocked.status_code == 409
        assert "queued or running jobs" in blocked.json()["detail"]["message"]
        assert blocked.json()["detail"]["deletion_plan"]["active_jobs"][0]["id"] == job.id
        assert client.get(f"/api/projects/{project_id}/models/{chain['model']['id']}").status_code == 200


def test_project_restore_and_restore_collision():
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from NEExT.workbench.app import create_app

    with TemporaryDirectory() as tmpdir:
        client = TestClient(create_app(tmpdir))
        project = client.post("/api/projects", json={"name": "Restore Me", "description": "restore test"}).json()
        project_id = project["id"]
        deleted = client.delete(f"/api/projects/{project_id}")
        assert deleted.status_code == 200

        trash = client.get("/api/trash")
        assert trash.status_code == 200
        assert trash.json()["projects"] == [
            {
                "trash_id": project_id,
                "project_id": project_id,
                "name": "Restore Me",
                "description": "restore test",
                "trashed_path": f"trash/projects/{project_id}",
            }
        ]

        restored = client.post(f"/api/trash/projects/{project_id}/restore")
        assert restored.status_code == 200
        assert restored.json() == {
            "restored_kind": "project",
            "id": project_id,
            "name": "Restore Me",
            "restored_path": f"projects/{project_id}",
        }
        assert client.get(f"/api/projects/{project_id}").status_code == 200

        second_delete = client.delete(f"/api/projects/{project_id}")
        assert second_delete.status_code == 200
        collision_path = Path(tmpdir) / "projects" / project_id
        collision_path.mkdir(parents=True)
        blocked = client.post(f"/api/trash/projects/{project_id}/restore")
        assert blocked.status_code == 409
        assert "live project with the same ID already exists" in blocked.json()["detail"]["message"]
