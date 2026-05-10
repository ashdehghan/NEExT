"""Launcher for the optional NEExT web workbench."""

from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path


class WebDependencyError(RuntimeError):
    """Raised when optional web dependencies are not installed."""


def _dependency_error(exc: ImportError) -> WebDependencyError:
    return WebDependencyError(
        "The NEExT web UI dependencies are not installed.\n" 'Install them with: pip install "NEExT[web]"\n' f"Original import error: {exc}"
    )


def default_project_dir() -> Path:
    return Path.cwd() / ".neext-web-project"


def run_web(
    project_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """Run the local NEExT web workbench."""
    try:
        import uvicorn

        from NEExT.web.app import create_app
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    resolved_project_dir = Path(project_dir) if project_dir else default_project_dir()
    resolved_project_dir.mkdir(parents=True, exist_ok=True)

    try:
        app = create_app(resolved_project_dir)
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            raise _dependency_error(ImportError(str(exc))) from exc
        raise
    url = f"http://{host}:{port}"
    print(f"Starting NEExT web workbench at {url}")
    print(f"Project directory: {resolved_project_dir}")

    if open_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    uvicorn.run(app, host=host, port=port, reload=reload)


def _open_browser_later(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url)
