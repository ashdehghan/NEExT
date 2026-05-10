"""Command line entry points for NEExT."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neext", description="NEExT command line tools")
    subparsers = parser.add_subparsers(dest="command")

    web_parser = subparsers.add_parser("web", help="Run the optional local NEExT web workbench")
    web_parser.add_argument(
        "project_dir",
        nargs="?",
        default=None,
        help="Project directory for local web artifacts. Defaults to .neext-web-project in the current directory.",
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    web_parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    web_parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab automatically")
    web_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload. Intended for development only.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "web":
        from NEExT.web.launcher import WebDependencyError, run_web

        project_dir = Path(args.project_dir) if args.project_dir else None
        try:
            run_web(
                project_dir=project_dir,
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
                reload=args.reload,
            )
        except WebDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
