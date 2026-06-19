"""Command-line launcher for the local NEExT Workbench MCP server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from NEExT.workbench.mcp_server import create_mcp_server, require_sdk_streamable_http
    from NEExT.workbench.mcp_service import WorkbenchMcpService
    from NEExT.workbench.paths import resolve_workspace_path
    from NEExT.workbench.storage import WorkbenchStore
else:
    from .mcp_server import create_mcp_server, require_sdk_streamable_http
    from .mcp_service import WorkbenchMcpService
    from .paths import resolve_workspace_path
    from .storage import WorkbenchStore


async def _run_stdio_server(store: WorkbenchStore) -> None:
    try:
        require_sdk_streamable_http()
        import anyio  # noqa: F401
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise RuntimeError("Install NEExT[workbench-mcp] to run the Workbench MCP server") from exc

    server = create_mcp_server(WorkbenchMcpService(store))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Start the local NEExT Workbench MCP server over stdio.")
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace folder for Workbench projects.")
    args = parser.parse_args(argv)

    if sys.version_info < (3, 10):
        print("NEExT Workbench MCP requires Python 3.10 or newer.", file=sys.stderr)
        raise SystemExit(2)

    # The stdio MCP process is an MCP client, not the job executor. It persists jobs to
    # the shared workspace; the running Workbench server is the sole durable executor and
    # picks them up from disk. Running a worker here would create a second, ephemeral
    # queue the UI never observes (jobs would appear stuck "queued").
    store = WorkbenchStore(resolve_workspace_path(args.workspace), run_worker=False)
    token = os.environ.get("NEEXT_WORKBENCH_MCP_TOKEN", "")
    if not store.verify_mcp_token(token):
        print("NEExT Workbench MCP is disabled or NEEXT_WORKBENCH_MCP_TOKEN is invalid.", file=sys.stderr)
        raise SystemExit(2)

    try:
        import anyio

        anyio.run(_run_stdio_server, store)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"NEExT Workbench MCP failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
