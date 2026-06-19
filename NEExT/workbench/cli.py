"""Command-line launcher for NEExT Workbench."""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn

from .app import create_app
from .paths import default_workspace_path


def find_open_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"Could not find an open port starting at {preferred_port}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Start the local NEExT Workbench web app.")
    parser.add_argument("--workspace", type=Path, default=default_workspace_path(), help="Workspace folder for Workbench projects.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Keep localhost for trusted local execution.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    args = parser.parse_args(argv)

    port = find_open_port(args.host, args.port)
    app = create_app(args.workspace)
    url = f"http://{args.host}:{port}"
    print(f"NEExT Workbench workspace: {args.workspace}")
    print(f"NEExT Workbench URL: {url}")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)), daemon=True).start()

    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
