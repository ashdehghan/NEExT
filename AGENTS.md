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

Validation commands:

```bash
python3 -m pytest
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
- Avoid silently changing release metadata during unrelated work.
- Do not publish packages, create release tags, push to remotes, or run `make deploy` unless explicitly requested.

## Testing Guidance

- For generator changes, start with `python3 -m pytest tests/test_generators.py`.
- For graph IO, collection, feature, or embedding changes, run the most relevant targeted pytest tests first, then `python3 -m pytest` when dependencies are available.
- For end-to-end behavior, use `python3 test.py` and `python3 test_networkx.py` as smoke checks.
- For docs-only changes, build docs only when the changed content affects Sphinx output.
- If dependencies are missing, do not install packages globally. Use a virtual environment or report the blocker.

## Repo-Specific Cautions

- `pyproject.toml` currently declares project version `0.2.11`, while `NEExT/__init__.py` exports `__version__ = "0.2.10"`. Flag this before release-related work; do not silently modify it in unrelated tasks.
- `CLAUDE.md` is useful context, but it underemphasizes the current `generators` and `outliers` packages. Inspect those modules directly for related work.
- `pyproject.toml` defines `dev`, `docs`, `experiments`, `advanced`, `dgl`, `all`, and `all-dl` extras. Do not assume a separate `test` extra exists.
- The root `Makefile` includes documentation and deploy targets only. Quality commands are configured in `pyproject.toml`, not as Makefile targets.
- The GitHub workflow `.github/workflows/python-publish.yml` publishes on GitHub release publication; local `make deploy` has its own release path. Treat release automation carefully and only on explicit request.

## Custom Feature Contract

Custom node feature functions registered through `my_feature_methods` should accept one graph object and return a `pandas.DataFrame` with columns in this order:

```text
node_id, graph_id, <feature columns...>
```

Keep feature names stable and deterministic. For notebook-defined functions, remember the repo relies on `joblib`/cloudpickle-compatible behavior; avoid closing over unpicklable objects.
