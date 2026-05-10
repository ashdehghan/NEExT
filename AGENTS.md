# AGENTS.md

Guidance for Codex CLI and other coding agents working in this repository.

## Project Shape

NEExT is a Python graph machine learning framework. The main public interface is `NEExT.framework.NEExT`, with package-level exports in `NEExT/__init__.py`.

Core subsystems:

- `NEExT/io.py`: CSV, DataFrame, URL, and NetworkX loading into graph collections.
- `NEExT/graphs/` and `NEExT/collections/`: graph wrappers, graph collections, egonets, and k-hop decomposition.
- `NEExT/features/`: structural node feature computation and custom feature registration.
- `NEExT/embeddings/` and `NEExT/builders/`: graph embedding computation, including Wasserstein-style embeddings.
- `NEExT/ml_models/` and `NEExT/datasets/`: sklearn-compatible datasets, model training, and feature importance.
- `NEExT/generators/`: synthetic graph generation, attributes, anomalies, adapters, presets, and fluent graph building.
- `NEExT/outliers/`: outlier detection utilities and benchmark helpers.
- `NEExT/web/`: optional local FastAPI web workbench backend, project/artifact persistence, job management, serializers, and service layer.
- `web/frontend/`: React/Vite source for the web workbench UI. Treat this as the frontend source of truth.
- `NEExT/web/static/`: packaged static web assets served by FastAPI. These are generated from `web/frontend` with `npm run build:package`.
- `sandbox/ui-mockups/`: static HTML/CSS review mockups. `v01-aero-classic.html` is the selected visual direction for the real web workbench.

The central data flow is:

```text
Input data -> GraphIO -> GraphCollection -> node features -> graph embeddings -> ML/outlier workflows
                         |
                         +-> EgonetCollection -> node-centered graph workflows
```

## Setup and Commands

Prefer the repo's `pyproject.toml` configuration over guessed commands.

Recommended setup when `uv` is available:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Fallback setup with standard Python tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Optional local web workbench setup:

```bash
python3 -m pip install -e ".[web,dev]"
neext web
```

Frontend setup and packaging:

```bash
cd web/frontend
npm install
npm run build
npm run build:package
```

`npm run build:package` writes the distributable assets to `NEExT/web/static/`, which is included in wheels by `pyproject.toml`.

Validation commands:

```bash
python3 -m pytest
python3 -m pytest tests/test_web_api.py tests/test_web_workbench.py
python3 -m pytest tests/test_generators.py
python3 test.py
python3 test_networkx.py
python3 -m ruff check .
python3 -m black --check .
python3 -m isort --check-only .
python3 -m mypy .
```

Formatting commands, only when formatting is part of the task:

```bash
python3 -m black .
python3 -m isort .
```

Documentation:

```bash
make docs
# equivalent:
cd docs && make html
```

Do not assume `uv`, `pytest`, `ruff`, or other development tools are installed in the active shell. Always run the actual commands before claiming checks passed; if tooling is unavailable, report that clearly.

## Editing Rules

- Keep changes focused on the user's request. Do not perform broad cleanup unless asked.
- Preserve public APIs and import paths unless the task explicitly requires an API change.
- Prefer existing patterns in the repo over new abstractions.
- Use NetworkX/iGraph-aware code when touching graph operations; do not assume one backend unless the surrounding code already does.
- Treat numerical and graph algorithm behavior as high risk. Add or run targeted tests when changing feature computation, egonet extraction, embeddings, sampling, generators, or outlier logic.
- For web UI changes, edit `web/frontend` first and rebuild `NEExT/web/static` with `npm run build:package`. Do not hand-edit generated static assets as the source of truth.
- The current web UI is intentionally desktop-first and MATLAB/Windows 7 Aero-inspired. Keep the shell/ribbon/docked-panel visual language aligned with `sandbox/ui-mockups/v01-aero-classic.html` unless the user selects a different direction.
- Avoid silently changing release metadata during unrelated work.
- Do not publish packages, create release tags, push to remotes, or run `make deploy` unless explicitly requested.

## Testing Guidance

- For web backend, CLI, artifact, or frontend packaging changes, run `python3 -m pytest tests/test_web_api.py tests/test_web_workbench.py`.
- For web frontend changes, run `npm run build` in `web/frontend`, then `npm run build:package` before testing the FastAPI-served app.
- For generator changes, start with `python3 -m pytest tests/test_generators.py`.
- For graph IO, collection, feature, or embedding changes, run the most relevant targeted pytest tests first, then `python3 -m pytest` when dependencies are available.
- For end-to-end behavior, use `python3 test.py` and `python3 test_networkx.py` as smoke checks.
- For docs-only changes, build docs only when the changed content affects Sphinx output.
- If dependencies are missing, do not install packages globally. Use a virtual environment or report the blocker.

## Repo-Specific Cautions

- Package version is sourced from `NEExT/__init__.py` via Hatch dynamic versioning. Update `__version__` there for release work; do not add a static `version = ...` field back to `pyproject.toml`.
- `CLAUDE.md` is useful context, but inspect the current modules directly for web, generator, and outlier work.
- `pyproject.toml` defines `web`, `dev`, `docs`, `experiments`, `advanced`, `dgl`, `all`, and `all-dl` extras. Do not assume a separate `test` extra exists.
- The root `Makefile` includes documentation and direct PyPI release targets. Quality commands are configured in `pyproject.toml`, not as Makefile targets.
- Local web project folders such as `.neext-web*/`, frontend `node_modules/`, and `dist/` output are ignored. Do not commit `.env`, local project data, or dependency directories.
- PyPI publication is direct/local only. Use `make release-check` plus `make deploy` for the full push/tag/publish flow, or `make publish-only` after `main` has already been pushed. Do not publish packages, create release tags, or push to remotes unless explicitly requested.

## Custom Feature Contract

Custom node feature functions registered through `my_feature_methods` should accept one graph object and return a `pandas.DataFrame` with columns in this order:

```text
node_id, graph_id, <feature columns...>
```

Keep feature names stable and deterministic. For notebook-defined functions, remember the repo relies on `joblib`/cloudpickle-compatible behavior; avoid closing over unpicklable objects.
