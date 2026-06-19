"""Path helpers for the local NEExT Workbench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

WORKBENCH_ENV_VAR = "NEEXT_WORKBENCH_HOME"
DEFAULT_WORKSPACE_NAME = "NEExT-Workbench"


def default_workspace_path() -> Path:
    """Return the default cross-platform workspace path."""
    configured = os.environ.get(WORKBENCH_ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / DEFAULT_WORKSPACE_NAME).resolve()


def resolve_workspace_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve an explicit or default workspace path."""
    if path is None:
        return default_workspace_path()
    return Path(path).expanduser().resolve()
