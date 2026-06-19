#!/usr/bin/env python3
"""End-to-end smoke test / reference client for the NEExT Workbench MCP server.

This connects to a *running* Workbench over the same Streamable HTTP transport
Claude Desktop and other MCP clients use, then walks a full workflow: it reads
the server onboarding (instructions, doc resources, recipe prompts), uploads a
small dataset via the Dataset Intake session flow, prepares it, previews and
analyzes the result, and requests a UI navigation.

It doubles as a copy-and-adapt reference for building your own MCP client.

Usage:
    # 1. Start the Workbench and enable MCP in Settings -> Agentic, copy the token.
    make neext-workbench

    # 2. Run this client against the live server.
    python examples/mcp_smoke_client.py --token nxt_mcp_...
    # or
    NEEXT_WORKBENCH_MCP_TOKEN=nxt_mcp_... python examples/mcp_smoke_client.py

Requires the Workbench MCP extra (``pip install -e ".[workbench-mcp]"``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_URL = "http://127.0.0.1:8765/mcp"

# A tiny two-graph dataset that satisfies the NEExT intake contract.
SAMPLE_TABLES = {
    "edges": {
        "format": "records",
        "records": [
            {"src_node_id": 0, "dest_node_id": 1},
            {"src_node_id": 1, "dest_node_id": 2},
            {"src_node_id": 3, "dest_node_id": 4},
            {"src_node_id": 4, "dest_node_id": 5},
        ],
    },
    "node_graph_mapping": {
        "format": "records",
        "records": [
            {"node_id": 0, "graph_id": 0},
            {"node_id": 1, "graph_id": 0},
            {"node_id": 2, "graph_id": 0},
            {"node_id": 3, "graph_id": 1},
            {"node_id": 4, "graph_id": 1},
            {"node_id": 5, "graph_id": 1},
        ],
    },
    "graph_labels": {
        "format": "records",
        "records": [
            {"graph_id": 0, "graph_label": 0},
            {"graph_id": 1, "graph_label": 1},
        ],
    },
}


def _ok(message: str) -> None:
    print(f"  ✓ {message}")


def _parse_tool_result(result):
    if result.isError:
        text = result.content[0].text if result.content else "<no content>"
        raise RuntimeError(f"tool error: {text}")
    if not result.content:
        return None
    return json.loads(result.content[0].text)


async def _call(session, name: str, arguments: dict | None = None) -> dict:
    return _parse_tool_result(await session.call_tool(name, arguments or {}))


async def _wait_for_job(session, project_id: str, job_id: str, timeout_s: float = 60.0) -> dict:
    import anyio

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = await _call(session, "neext_get_job", {"project_id": project_id, "job_id": job_id})
        if job["status"] in {"completed", "failed"}:
            return job
        await anyio.sleep(0.2)
    raise RuntimeError(f"timed out waiting for job {job_id}")


async def run(url: str, token: str) -> None:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # --- Onboarding ---------------------------------------------------
                print("Onboarding:")
                init = await session.initialize()
                if not init.instructions:
                    raise RuntimeError("server returned empty instructions field")
                _ok(f"instructions present ({len(init.instructions)} chars)")

                tools = {t.name for t in (await session.list_tools()).tools}
                _ok(f"{len(tools)} tools listed")

                prompt_names = {p.name for p in (await session.list_prompts()).prompts}
                for recipe in ("upload_neext_dataset", "run_end_to_end_pipeline", "inspect_neext_results"):
                    if recipe not in prompt_names:
                        raise RuntimeError(f"missing recipe prompt: {recipe}")
                _ok(f"{len(prompt_names)} prompts listed (incl. recipes)")

                resource_uris = {str(r.uri) for r in (await session.list_resources()).resources}
                for uri in ("neext://docs/dataset-intake", "neext://docs/recipes"):
                    if uri not in resource_uris:
                        raise RuntimeError(f"missing doc resource: {uri}")
                intake_doc = await session.read_resource("neext://docs/dataset-intake")
                if "src_node_id" not in intake_doc.contents[0].text:
                    raise RuntimeError("dataset-intake doc missing contract text")
                _ok("dataset-intake + recipes doc resources readable")

                # --- Create a project --------------------------------------------
                print("Pipeline:")
                project = await _call(
                    session,
                    "neext_create_project",
                    {"name": "MCP Smoke Test", "description": "examples/mcp_smoke_client.py"},
                )
                project_id = project["id"]
                _ok(f"project created ({project_id})")

                # --- Upload a dataset via the intake session flow ----------------
                created_session = await _call(
                    session,
                    "neext_create_dataset_intake_session",
                    {"project_id": project_id, "name": "Smoke Dataset"},
                )
                session_id = created_session["id"]
                for table_name, table in SAMPLE_TABLES.items():
                    await _call(
                        session,
                        "neext_append_dataset_intake_table",
                        {"project_id": project_id, "session_id": session_id, "table_name": table_name, "table": table},
                    )
                validation = await _call(
                    session,
                    "neext_validate_dataset_intake_session",
                    {"project_id": project_id, "session_id": session_id},
                )
                if not (validation.get("validation") or {}).get("valid", False):
                    raise RuntimeError(f"intake validation failed: {validation}")
                _ok("intake session validated")

                dataset = await _call(
                    session,
                    "neext_create_dataset_from_intake",
                    {"project_id": project_id, "session_id": session_id},
                )
                dataset_id = dataset["id"]
                _ok(f"draft dataset created ({dataset_id})")

                # --- Prepare the dataset and poll the job ------------------------
                jobs = await _call(
                    session,
                    "neext_run_artifacts",
                    {"project_id": project_id, "kind": "dataset", "ids": [dataset_id]},
                )
                job_id = jobs["jobs"][0]["id"]
                job = await _wait_for_job(session, project_id, job_id)
                if job["status"] != "completed":
                    raise RuntimeError(f"dataset preparation did not complete: {job}")
                _ok("dataset prepared (job completed)")

                # --- Inspect results ---------------------------------------------
                print("Inspection:")
                preview = await _call(
                    session,
                    "neext_preview_artifact",
                    {"project_id": project_id, "kind": "dataset", "artifact_id": dataset_id, "table": "nodes", "limit": 5},
                )
                _ok(f"preview returned {len(preview.get('rows', []))} node rows")

                analysis = await _call(
                    session,
                    "neext_analyze_artifact",
                    {"project_id": project_id, "kind": "dataset", "artifact_id": dataset_id},
                )
                _ok("analysis returned" if analysis else "analysis empty")

                # --- Drive the UI ------------------------------------------------
                await _call(
                    session,
                    "neext_set_workbench_view",
                    {
                        "route": {
                            "top_tab": "datasets",
                            "command": "datasets",
                            "project_id": project_id,
                            "artifact_kind": "dataset",
                            "artifact_id": dataset_id,
                        },
                        "message": "Opened by mcp_smoke_client.py",
                    },
                )
                _ok("requested UI navigation to the new dataset")

    print("\nPASS: full MCP onboarding + pipeline smoke test succeeded.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("NEEXT_WORKBENCH_MCP_URL", DEFAULT_URL), help=f"MCP endpoint URL (default: {DEFAULT_URL})")
    parser.add_argument(
        "--token", default=os.environ.get("NEEXT_WORKBENCH_MCP_TOKEN", ""), help="MCP bearer token (or set NEEXT_WORKBENCH_MCP_TOKEN)"
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("an MCP token is required: pass --token or set NEEXT_WORKBENCH_MCP_TOKEN (enable MCP in Settings -> Agentic)")

    try:
        import anyio
    except ImportError:
        print('error: the Workbench MCP extra is required: pip install -e ".[workbench-mcp]"', file=sys.stderr)
        return 2

    print(f"Connecting to {args.url} ...\n")
    try:
        anyio.run(run, args.url, args.token)
    except Exception as exc:  # noqa: BLE001 - surface a readable failure to the operator
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
