import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REQUIRE_WORKBENCH_MCP_ENV = "NEEXT_REQUIRE_WORKBENCH_MCP"


def require_module(module_name: str):
    if os.environ.get(REQUIRE_WORKBENCH_MCP_ENV) == "1":
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            pytest.fail(f"{module_name} is required when {REQUIRE_WORKBENCH_MCP_ENV}=1: {exc}")
    return pytest.importorskip(module_name)


def require_mcp_server_sdk() -> None:
    require_module("mcp.server.lowlevel")
    require_module("mcp.server.streamable_http_manager")


def require_mcp_stdio_client() -> None:
    require_mcp_server_sdk()
    require_module("mcp.client.stdio")


def require_mcp_streamable_http_client() -> None:
    require_mcp_server_sdk()
    require_module("mcp.client.streamable_http")


def intake_records_tables() -> dict[str, dict]:
    return {
        "node_graph_mapping": {
            "format": "records",
            "records": [
                {"node_id": 1, "graph_id": "g1"},
                {"node_id": 2, "graph_id": "g1"},
                {"node_id": 3, "graph_id": "g2"},
                {"node_id": 4, "graph_id": "g2"},
            ],
        },
        "edges": {
            "format": "records",
            "records": [
                {"src_node_id": 1, "dest_node_id": 2},
                {"src_node_id": 3, "dest_node_id": 4},
            ],
        },
        "graph_labels": {
            "format": "records",
            "records": [
                {"graph_id": "g1", "graph_label": 0},
                {"graph_id": "g2", "graph_label": 1},
            ],
        },
        "node_features": {
            "format": "records",
            "records": [
                {"node_id": 1, "feature_a": 0.1},
                {"node_id": 2, "feature_a": 0.2},
                {"node_id": 3, "feature_a": 0.3},
                {"node_id": 4, "feature_a": 0.4},
            ],
        },
    }


def intake_csv_tables() -> dict[str, dict]:
    return {
        "node_graph_mapping": {"format": "csv", "csv": "node_id,graph_id\n1,g1\n2,g1\n3,g2\n4,g2\n"},
        "edges": {"format": "csv", "csv": "src_node_id,dest_node_id\n1,2\n3,4\n"},
        "graph_labels": {"format": "csv", "csv": "graph_id,graph_label\ng1,0\ng2,1\n"},
        "node_features": {"format": "csv", "csv": "node_id,feature_a\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n"},
    }


def write_labeled_graph_source_bundle(source_dir: Path, *, graph_count: int = 8) -> dict[str, str]:
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
        label_rows.append(f"{graph_id},{graph_index % 2}\n")
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


def wait_for_store_job(store, project_id: str, job_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = store.read_job(project_id, job_id)
        if job.status in {"completed", "failed"}:
            return job.model_dump(mode="json")
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def parse_tool_result(result):
    assert result.isError is False
    assert result.content
    return json.loads(result.content[0].text)


async def call_json_tool(session, name: str, arguments: dict | None = None) -> dict:
    return parse_tool_result(await session.call_tool(name, arguments or {}))


def tool_error_text(result) -> str:
    assert result.isError is True
    assert result.content
    return result.content[0].text


async def wait_for_mcp_job(session, project_id: str, job_id: str, timeout_s: float = 10.0) -> dict:
    import anyio

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = await call_json_tool(session, "neext_get_job", {"project_id": project_id, "job_id": job_id})
        if job["status"] in {"completed", "failed"}:
            return job
        await anyio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for MCP job {job_id}")


async def assert_mcp_dataset_intake_upload_flow(session, workspace_path: str, tables: dict[str, dict], *, project_name: str) -> dict:
    tools = await session.list_tools()
    tool_names = {tool.name for tool in tools.tools}
    assert {
        "neext_create_project",
        "neext_validate_dataset_intake",
        "neext_create_dataset_intake_session",
        "neext_append_dataset_intake_table",
        "neext_validate_dataset_intake_session",
        "neext_create_dataset_from_intake",
        "neext_run_artifacts",
        "neext_get_job",
        "neext_preview_artifact",
        "neext_analyze_artifact",
        "neext_search_graphs",
        "neext_get_graph_detail",
    }.issubset(tool_names)
    assert not any("archive" in tool_name for tool_name in tool_names)

    project = await call_json_tool(session, "neext_create_project", {"name": project_name, "description": "mcp intake"})
    project_id = project["id"]
    assert workspace_path not in json.dumps(project)
    assert "path" not in project

    invalid_validation = await call_json_tool(
        session,
        "neext_validate_dataset_intake",
        {
            "project_id": project_id,
            "name": "Missing Required Table",
            "tables": {"edges": tables["edges"]},
        },
    )
    assert invalid_validation["valid"] is False
    assert any(error["table"] == "node_graph_mapping" for error in invalid_validation["errors"])

    validation = await call_json_tool(
        session,
        "neext_validate_dataset_intake",
        {
            "project_id": project_id,
            "name": "Agent Tables",
            "description": "Synthetic graph data supplied through MCP",
            "tables": tables,
            "params": {"graph_type": "networkx", "filter_largest_component": True},
        },
    )
    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["stats"] == {
        "graph_count": 2,
        "node_count": 4,
        "edge_count": 2,
        "has_graph_labels": True,
        "has_node_features": True,
        "has_edge_features": False,
    }

    intake_session = await call_json_tool(
        session,
        "neext_create_dataset_intake_session",
        {
            "project_id": project_id,
            "name": "Agent Tables",
            "description": "Synthetic graph data supplied through MCP",
            "params": {"graph_type": "networkx", "filter_largest_component": True},
        },
    )
    session_id = intake_session["id"]
    assert intake_session["tables"] == []
    assert workspace_path not in json.dumps(intake_session)

    for table_name, table in tables.items():
        append_result = await call_json_tool(
            session,
            "neext_append_dataset_intake_table",
            {
                "project_id": project_id,
                "session_id": session_id,
                "table_name": table_name,
                "table": table,
                "replace": True,
            },
        )
        assert table_name in append_result["tables"]

    session_validation = await call_json_tool(
        session,
        "neext_validate_dataset_intake_session",
        {"project_id": project_id, "session_id": session_id},
    )
    assert session_validation["validation"]["valid"] is True
    assert session_validation["validation"]["stats"]["graph_count"] == 2

    dataset = await call_json_tool(
        session,
        "neext_create_dataset_from_intake",
        {"project_id": project_id, "session_id": session_id},
    )
    dataset_id = dataset["id"]
    assert dataset["status"] == "planned"
    assert dataset["source_type"] == "uploaded_neext_tables"
    assert dataset["source_catalog_id"] == "dataset-intake"
    assert dataset["inputs"] == []
    assert dataset["raw_data_files"]["nodes"] == "source/nodes.parquet"
    assert workspace_path not in json.dumps(dataset)
    assert "path" not in json.dumps(dataset)

    artifact_listing = await call_json_tool(session, "neext_list_artifacts", {"project_id": project_id, "kind": "datasets"})
    assert [artifact["id"] for artifact in artifact_listing] == [dataset_id]

    run = await call_json_tool(session, "neext_run_artifacts", {"project_id": project_id, "kind": "datasets", "ids": [dataset_id]})
    job_id = run["jobs"][0]["id"]
    completed_job = await wait_for_mcp_job(session, project_id, job_id)
    assert completed_job["status"] == "completed"

    prepared = await call_json_tool(
        session,
        "neext_get_artifact",
        {"project_id": project_id, "kind": "datasets", "artifact_id": dataset_id},
    )
    assert prepared["status"] == "completed"
    assert prepared["prepared_stats"] == {
        "graph_count": 2,
        "node_count": 4,
        "edge_count": 2,
        "has_graph_labels": True,
        "has_node_features": True,
        "has_edge_features": False,
    }
    assert prepared["raw_data_files"]["nodes"] == "raw/nodes.parquet"
    assert workspace_path not in json.dumps(prepared)

    preview = await call_json_tool(
        session,
        "neext_preview_artifact",
        {"project_id": project_id, "kind": "datasets", "artifact_id": dataset_id, "table": "nodes", "limit": 2, "offset": 0},
    )
    assert preview["total_rows"] == 4
    assert preview["limit"] == 2
    assert len(preview["rows"]) == 2

    analysis = await call_json_tool(
        session,
        "neext_analyze_artifact",
        {"project_id": project_id, "kind": "datasets", "artifact_id": dataset_id, "options": {"graph_id": "g1"}},
    )
    assert analysis["selected_graph_id"] == "g1"
    assert analysis["prepared_stats"]["graph_count"] == 2

    search = await call_json_tool(
        session,
        "neext_search_graphs",
        {"project_id": project_id, "kind": "datasets", "artifact_id": dataset_id, "query": "g1", "limit": 5},
    )
    assert search["results"][0]["graph_id"] == "g1"

    graph_detail = await call_json_tool(
        session,
        "neext_get_graph_detail",
        {"project_id": project_id, "kind": "datasets", "artifact_id": dataset_id, "graph_id": "g1"},
    )
    assert graph_detail["graph_id"] == "g1"
    assert workspace_path not in json.dumps(
        {
            "artifact_listing": artifact_listing,
            "run": run,
            "completed_job": completed_job,
            "preview": preview,
            "analysis": analysis,
            "search": search,
            "graph_detail": graph_detail,
        }
    )
    return {"project_id": project_id, "dataset_id": dataset_id, "job_id": job_id}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_workbench_mcp_sdk_modules_are_available_when_required():
    if os.environ.get(REQUIRE_WORKBENCH_MCP_ENV) != "1":
        pytest.skip(f"Set {REQUIRE_WORKBENCH_MCP_ENV}=1 to make MCP SDK imports mandatory")
    require_mcp_stdio_client()
    require_mcp_streamable_http_client()


def test_mcp_cli_rejects_disabled_missing_and_invalid_tokens():
    with TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env.pop("NEEXT_WORKBENCH_MCP_TOKEN", None)
        missing = subprocess.run(
            [sys.executable, "-m", "NEExT.workbench.mcp_cli", "--workspace", tmpdir],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert missing.returncode == 2
        assert "MCP is disabled" in missing.stderr

        from NEExT.workbench.storage import WorkbenchStore

        store = WorkbenchStore(Path(tmpdir))
        settings, token = store.enable_mcp()
        assert settings.enabled

        env["NEEXT_WORKBENCH_MCP_TOKEN"] = token + "-wrong"
        invalid = subprocess.run(
            [sys.executable, "-m", "NEExT.workbench.mcp_cli", "--workspace", tmpdir],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert invalid.returncode == 2
        assert "MCP is disabled" in invalid.stderr

        store.disable_mcp()
        env["NEEXT_WORKBENCH_MCP_TOKEN"] = token
        disabled = subprocess.run(
            [sys.executable, "-m", "NEExT.workbench.mcp_cli", "--workspace", tmpdir],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert disabled.returncode == 2
        assert "MCP is disabled" in disabled.stderr


def test_valid_mcp_stdio_server_exposes_expected_tools_resources_and_prompts():
    require_mcp_stdio_client()

    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def inspect_server(workspace_path: str, token: str):
        env = os.environ.copy()
        env["NEEXT_WORKBENCH_MCP_TOKEN"] = token
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "NEExT.workbench.mcp_cli", "--workspace", workspace_path],
            env=env,
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                assert init_result.instructions
                assert "dataset-first" in init_result.instructions
                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert {
                    "neext_workspace_summary",
                    "neext_create_project",
                    "neext_configure_dataset",
                    "neext_run_artifacts",
                    "neext_analyze_artifact",
                    "neext_request_delete_project",
                    "neext_restore_project",
                    "neext_set_workbench_view",
                }.issubset(tool_names)
                assert not any("archive" in tool_name for tool_name in tool_names)

                prompts = await session.list_prompts()
                prompt_names = {prompt.name for prompt in prompts.prompts}
                assert {
                    "explore_neext_project",
                    "configure_neext_pipeline",
                    "run_neext_pipeline",
                    "compare_neext_models",
                    "investigate_neext_graph",
                    "upload_neext_dataset",
                    "run_end_to_end_pipeline",
                    "inspect_neext_results",
                }.issubset(prompt_names)

                resources = await session.list_resources()
                resource_uris = {str(resource.uri) for resource in resources.resources}
                assert "neext://workspace" in resource_uris
                assert {"neext://docs/dataset-intake", "neext://docs/recipes"}.issubset(resource_uris)

                intake_doc = await session.read_resource("neext://docs/dataset-intake")
                assert intake_doc.contents
                assert "src_node_id" in intake_doc.contents[0].text

                created = parse_tool_result(await session.call_tool("neext_create_project", {"name": "MCP Project", "description": "stdio"}))
                assert created["name"] == "MCP Project"
                assert "path" not in created

                projects = parse_tool_result(await session.call_tool("neext_list_projects", {}))
                assert [project["id"] for project in projects] == [created["id"]]

                workspace = parse_tool_result(await session.call_tool("neext_workspace_summary", {}))
                assert workspace["project_count"] == 1
                assert workspace_path not in json.dumps(workspace)

                project_resource = await session.read_resource(f"neext://projects/{created['id']}")
                assert project_resource.contents
                assert workspace_path not in project_resource.contents[0].text

    with TemporaryDirectory() as tmpdir:
        from NEExT.workbench.storage import WorkbenchStore

        store = WorkbenchStore(Path(tmpdir))
        _, token = store.enable_mcp()
        anyio.run(inspect_server, str(Path(tmpdir).resolve()), token)


def test_mcp_stdio_client_uploads_prepares_and_previews_dataset_intake_tables():
    pytest.importorskip("pyarrow")
    require_mcp_stdio_client()

    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def upload_with_stdio(workspace_path: str, token: str):
        env = os.environ.copy()
        env["NEEXT_WORKBENCH_MCP_TOKEN"] = token
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "NEExT.workbench.mcp_cli", "--workspace", workspace_path],
            env=env,
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await assert_mcp_dataset_intake_upload_flow(
                    session,
                    workspace_path,
                    intake_records_tables(),
                    project_name="MCP Stdio Intake",
                )
                assert result["dataset_id"]

    with TemporaryDirectory() as tmpdir:
        from NEExT.workbench.storage import WorkbenchStore

        store = WorkbenchStore(Path(tmpdir))
        _, token = store.enable_mcp()
        anyio.run(upload_with_stdio, str(Path(tmpdir).resolve()), token)


def test_sdk_streamable_http_client_connects_to_workbench_mcp():
    require_mcp_streamable_http_client()

    import anyio
    import httpx
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from NEExT.workbench.app import create_app

    async def inspect_http_server(workspace_path: str):
        app = create_app(workspace_path)
        _, token = app.state.store.enable_mcp()
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.time() + 10.0
            while not server.started and time.time() < deadline:
                await anyio.sleep(0.05)
            assert server.started

            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(headers=headers) as http_client:
                async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        tool_names = {tool.name for tool in tools.tools}
                        assert "neext_workspace_summary" in tool_names
                        assert "neext_create_project" in tool_names
                        workspace = parse_tool_result(await session.call_tool("neext_workspace_summary", {}))
                        assert workspace["project_count"] == 0
                        assert workspace_path not in json.dumps(workspace)
        finally:
            server.should_exit = True
            thread.join(timeout=10.0)

    with TemporaryDirectory() as tmpdir:
        anyio.run(inspect_http_server, str(Path(tmpdir).resolve()))


def test_mcp_streamable_http_client_uploads_prepares_and_previews_dataset_intake_tables():
    pytest.importorskip("pyarrow")
    require_mcp_streamable_http_client()

    import anyio
    import httpx
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from NEExT.workbench.app import create_app
    from NEExT.workbench.storage import MCP_DEFAULT_SCOPES

    async def upload_with_http(workspace_path: str):
        app = create_app(workspace_path)
        _, token = app.state.store.enable_mcp()
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.time() + 10.0
            while not server.started and time.time() < deadline:
                await anyio.sleep(0.05)
            assert server.started

            settings = app.state.store.read_mcp_settings()
            settings.scopes = ["read"]
            app.state.store._write_mcp_settings(settings)

            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(headers=headers) as http_client:
                async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        tool_names = {tool.name for tool in tools.tools}
                        assert "neext_workspace_summary" in tool_names
                        assert "neext_create_project" not in tool_names
                        assert "neext_create_dataset_intake_session" not in tool_names

            settings.scopes = MCP_DEFAULT_SCOPES
            app.state.store._write_mcp_settings(settings)

            async with httpx.AsyncClient(headers=headers) as http_client:
                async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await assert_mcp_dataset_intake_upload_flow(
                            session,
                            workspace_path,
                            intake_csv_tables(),
                            project_name="MCP HTTP Intake",
                        )
                        assert result["dataset_id"]
        finally:
            server.should_exit = True
            thread.join(timeout=10.0)

    with TemporaryDirectory() as tmpdir:
        anyio.run(upload_with_http, str(Path(tmpdir).resolve()))


def test_mcp_service_creates_dataset_from_intake_session():
    pytest.importorskip("pyarrow")

    from NEExT.workbench.mcp_service import WorkbenchMcpService
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        store = WorkbenchStore(Path(tmpdir))
        service = WorkbenchMcpService(store)
        project = service.create_project("Agent Intake", "mcp")
        project_id = project["id"]

        session = service.create_dataset_intake_session(project_id, "Agent Tables", "Synthetic graph data")
        session_id = session["id"]
        service.append_dataset_intake_table(
            project_id,
            session_id,
            "node_graph_mapping",
            {"format": "records", "records": [{"node_id": 1, "graph_id": "g1"}, {"node_id": 2, "graph_id": "g1"}]},
            replace=True,
        )
        service.append_dataset_intake_table(
            project_id,
            session_id,
            "edges",
            {"format": "records", "records": [{"src_node_id": 1, "dest_node_id": 2}]},
            replace=True,
        )
        service.append_dataset_intake_table(
            project_id,
            session_id,
            "graph_labels",
            {"format": "records", "records": [{"graph_id": "g1", "graph_label": 1}]},
            replace=True,
        )

        validation = service.validate_dataset_intake_session(project_id, session_id)
        assert validation["validation"]["valid"] is True
        created = service.create_dataset_from_intake(project_id, session_id)
        assert created["source_type"] == "uploaded_neext_tables"
        assert created["status"] == "planned"
        assert created["raw_data_files"]["nodes"] == "source/nodes.parquet"

        job = store.run_dataset_preparation(project_id, created["id"])
        completed = wait_for_store_job(store, project_id, job.id)
        assert completed["status"] == "completed"
        prepared = store.read_dataset(project_id, created["id"])
        assert prepared.status == "completed"
        assert prepared.prepared_stats.graph_count == 1


def test_mcp_service_dataset_intake_validation_errors_and_append_semantics():
    from NEExT.workbench.mcp_service import WorkbenchMcpService
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        service = WorkbenchMcpService(WorkbenchStore(Path(tmpdir)))
        project = service.create_project("Agent Intake Validation", "mcp")
        project_id = project["id"]

        missing_required = service.validate_dataset_intake(
            project_id,
            "Missing Nodes",
            {"edges": intake_records_tables()["edges"]},
        )
        assert missing_required["valid"] is False
        assert any(error["table"] == "node_graph_mapping" and "Required table" in error["message"] for error in missing_required["errors"])

        wrong_columns = service.validate_dataset_intake(
            project_id,
            "Wrong Columns",
            {
                "node_graph_mapping": {"format": "records", "records": [{"node": 1, "graph_id": "g1"}]},
                "edges": intake_records_tables()["edges"],
            },
        )
        assert wrong_columns["valid"] is False
        assert any(error["table"] == "node_graph_mapping" and error["column"] == "node_id" for error in wrong_columns["errors"])

        invalid_node_ids = service.validate_dataset_intake(
            project_id,
            "Invalid Node IDs",
            {
                **intake_records_tables(),
                "node_graph_mapping": {"format": "records", "records": [{"node_id": "node-a", "graph_id": "g1"}]},
            },
        )
        assert invalid_node_ids["valid"] is False
        assert "integer-compatible node IDs" in invalid_node_ids["errors"][0]["message"]

        session = service.create_dataset_intake_session(project_id, "Append Semantics", "mcp")
        session_id = session["id"]
        with pytest.raises(ValueError, match="Unsupported dataset intake table"):
            service.append_dataset_intake_table(
                project_id,
                session_id,
                "unknown_table",
                {"format": "records", "records": [{"value": 1}]},
                replace=True,
            )

        service.append_dataset_intake_table(
            project_id,
            session_id,
            "node_graph_mapping",
            {"format": "records", "records": [{"node_id": 1, "graph_id": "g1"}, {"node_id": 2, "graph_id": "g1"}]},
            replace=True,
        )
        service.append_dataset_intake_table(
            project_id,
            session_id,
            "node_graph_mapping",
            {"format": "records", "records": [{"node_id": 3, "graph_id": "g2"}, {"node_id": 4, "graph_id": "g2"}]},
            replace=False,
        )
        service.append_dataset_intake_table(project_id, session_id, "edges", intake_records_tables()["edges"], replace=True)
        validation = service.validate_dataset_intake_session(project_id, session_id)
        assert validation["validation"]["valid"] is True
        assert validation["validation"]["stats"]["node_count"] == 4
        assert validation["validation"]["stats"]["graph_count"] == 2


def test_mcp_service_configures_runs_previews_and_analyzes_pipeline(monkeypatch):
    pytest.importorskip("pyarrow")

    import pandas as pd

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.mcp_service import WorkbenchMcpService
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED_MCP",
                    name="Labeled MCP Dataset",
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

        store = WorkbenchStore(Path(tmpdir))
        service = WorkbenchMcpService(store)
        project = service.create_project("MCP Pipeline", "tool coverage")
        project_id = project["id"]
        dataset = service.configure_dataset(project_id, "LABELED_MCP", {"graph_type": "networkx", "filter_largest_component": False})
        feature = service.configure_feature(
            project_id,
            dataset["id"],
            "page_rank",
            {"feature_vector_length": 2, "normalize_features": False, "n_jobs": 1, "parallel_backend": "threading"},
        )
        embedding = service.configure_embedding(project_id, "approx_wasserstein", [feature["id"]], {"embedding_dimension": 2})
        model = service.configure_model(
            project_id,
            "random_forest",
            [embedding["id"]],
            {
                "task_type": "classifier",
                "sample_size": 1,
                "test_size": 0.5,
                "balance_dataset": False,
                "n_jobs": 1,
                "parallel_backend": "thread",
            },
        )

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

        monkeypatch.setattr(store, "_compute_graph_embeddings", fake_graph_embeddings)

        run = service.run_artifacts(project_id, "models", [model["id"]])
        job_id = run["jobs"][0]["id"]
        job = wait_for_store_job(store, project_id, job_id)
        assert job["status"] == "completed"

        jobs = service.list_jobs(project_id)
        assert jobs[0]["id"] == job_id
        assert service.get_artifact(project_id, "datasets", dataset["id"])["status"] == "completed"
        assert service.get_artifact(project_id, "features", feature["id"])["status"] == "completed"
        assert service.get_artifact(project_id, "embeddings", embedding["id"])["status"] == "completed"
        assert service.get_artifact(project_id, "models", model["id"])["status"] == "completed"

        dataset_preview = service.preview_artifact(project_id, "datasets", dataset["id"], table="nodes", limit=2, offset=0)
        assert dataset_preview["limit"] == 2
        assert dataset_preview["total_rows"] == 24

        model_preview = service.preview_artifact(project_id, "models", model["id"])
        assert "accuracy_mean" in model_preview["summary"]

        dataset_analysis = service.analyze_artifact(project_id, "datasets", dataset["id"], {"graph_id": "g0"})
        assert dataset_analysis["selected_graph_id"] == "g0"
        dataset_search = service.search_graphs(project_id, "datasets", dataset["id"], "g0", limit=5)
        assert dataset_search["results"][0]["graph_id"] == "g0"
        graph_detail = service.get_graph_detail(project_id, "datasets", dataset["id"], "g0")
        assert graph_detail["graph_id"] == "g0"

        model_analysis = service.analyze_artifact(project_id, "models", model["id"])
        assert model_analysis["model_status"] == "completed"

        payload = json.dumps(
            {
                "project": project,
                "dataset": dataset,
                "feature": feature,
                "embedding": embedding,
                "model": model,
                "jobs": jobs,
                "dataset_preview": dataset_preview,
                "model_preview": model_preview,
                "dataset_analysis": dataset_analysis,
                "model_analysis": model_analysis,
            }
        )
        assert str(Path(tmpdir).resolve()) not in payload


def test_mcp_service_exposes_gnn_catalog_and_configures_and_runs_it(monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")

    import NEExT.workbench.dataset_library as dataset_library
    from NEExT.workbench.dataset_library import CatalogDataset
    from NEExT.workbench.mcp_service import WorkbenchMcpService
    from NEExT.workbench.storage import WorkbenchStore

    with TemporaryDirectory() as tmpdir:
        source_files = write_labeled_graph_source_bundle(Path(tmpdir) / "source")
        monkeypatch.setattr(
            dataset_library,
            "DATASET_CATALOG",
            (
                CatalogDataset(
                    id="LABELED_MCP",
                    name="Labeled MCP Dataset",
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

        store = WorkbenchStore(Path(tmpdir))
        service = WorkbenchMcpService(store)

        catalog_ids = {entry["id"] for entry in service.list_catalog("embeddings")}
        assert "gnn" in catalog_ids

        project_id = service.create_project("MCP GNN", "gnn coverage")["id"]
        dataset = service.configure_dataset(
            project_id, "LABELED_MCP", {"graph_type": "networkx", "filter_largest_component": False}
        )
        feature = service.configure_feature(
            project_id,
            dataset["id"],
            "page_rank",
            {"feature_vector_length": 2, "normalize_features": False, "n_jobs": 1, "parallel_backend": "threading"},
        )
        embedding = service.configure_embedding(
            project_id,
            "gnn",
            [feature["id"]],
            {"embedding_dimension": 3, "architecture": "GraphSAGE"},
        )
        assert embedding["operation"]["params"]["embedding_algorithm"] == "gnn"
        assert embedding["operation"]["params"]["architecture"] == "GraphSAGE"

        run = service.run_artifacts(project_id, "embeddings", [embedding["id"]])
        job = wait_for_store_job(store, project_id, run["jobs"][0]["id"])
        assert job["status"] == "completed"
        assert service.get_artifact(project_id, "embeddings", embedding["id"])["status"] == "completed"


def test_stdio_launch_command_uses_running_interpreter(tmp_path):
    from NEExT.workbench import mcp_server

    workspace = tmp_path / "ws"
    command = mcp_server.stdio_launch_command(workspace)
    assert command[0] == sys.executable
    assert command[1:] == ["-m", "NEExT.workbench.mcp_cli", "--workspace", str(workspace)]


def test_path_under_protected_dir_is_macos_only(monkeypatch, tmp_path):
    from NEExT.workbench import mcp_server

    home = tmp_path / "home"
    (home / "Desktop" / "proj" / ".venv").mkdir(parents=True)
    (home / "code" / ".venv").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
    assert mcp_server.path_under_protected_dir(home / "Desktop" / "proj" / ".venv") == "Desktop"
    assert mcp_server.path_under_protected_dir(home / "code" / ".venv") is None

    # Windows/Linux have no TCC sandbox and must never be flagged.
    monkeypatch.setattr(mcp_server.sys, "platform", "linux")
    assert mcp_server.path_under_protected_dir(home / "Desktop" / "proj") is None
    monkeypatch.setattr(mcp_server.sys, "platform", "win32")
    assert mcp_server.path_under_protected_dir(home / "Documents" / "proj") is None


def test_evaluate_stdio_readiness_ready_when_unprotected(monkeypatch, tmp_path):
    from NEExT.workbench import mcp_server

    monkeypatch.setattr(mcp_server.sys, "platform", "linux")
    monkeypatch.setattr(mcp_server, "sdk_streamable_http_available", lambda: True)
    readiness = mcp_server.evaluate_stdio_readiness(tmp_path / "ws")
    assert readiness.ok is True
    assert readiness.status == "ready"
    assert readiness.issues == []
    assert readiness.command_preview.endswith(str(tmp_path / "ws"))


def test_evaluate_stdio_readiness_blocks_protected_workspace(monkeypatch, tmp_path):
    from NEExT.workbench import mcp_server

    home = tmp_path / "home"
    workspace = home / "Documents" / "ws"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
    monkeypatch.setattr(mcp_server, "sdk_streamable_http_available", lambda: True)

    readiness = mcp_server.evaluate_stdio_readiness(workspace)
    assert readiness.ok is False
    assert readiness.status == "blocked"
    assert any("Documents" in issue for issue in readiness.issues)
    assert readiness.remediation
