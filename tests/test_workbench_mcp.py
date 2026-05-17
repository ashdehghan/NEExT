import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytest.importorskip("mcp")


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
                await session.initialize()
                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert {
                    "neext_workspace_summary",
                    "neext_create_project",
                    "neext_configure_dataset",
                    "neext_run_artifacts",
                    "neext_analyze_artifact",
                }.issubset(tool_names)
                assert not any("delete" in tool_name or "archive" in tool_name or "restore" in tool_name for tool_name in tool_names)

                prompts = await session.list_prompts()
                prompt_names = {prompt.name for prompt in prompts.prompts}
                assert {
                    "explore_neext_project",
                    "configure_neext_pipeline",
                    "run_neext_pipeline",
                    "compare_neext_models",
                    "investigate_neext_graph",
                } == prompt_names

                resources = await session.list_resources()
                assert "neext://workspace" in {str(resource.uri) for resource in resources.resources}

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
