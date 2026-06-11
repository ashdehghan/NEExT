# AGENTS.md

Guidance for Codex CLI and other coding agents working in this repository.

## Operating Rules (read first, enforce every turn)

These rules override every default behavior. They exist because guessing, inventing, and private testing have repeatedly wasted the user's time. Re-read this section before every non-trivial action.

1. **Never guess. Ask.** If a decision is not explicitly answered by user instructions or existing code, or if the user's request conflicts with the current Workbench pattern, stop and ask one narrow clarifying question. No silent defaults, no "reasonable" assumptions.

2. **Do not invent.** Do not add features, buttons, ribbon items, pages, modals, hooks, contexts, providers, helper utilities, Makefile targets, or files the user did not explicitly request. If you think something is needed, ask first. "While I'm here" additions are banned.

3. **One thing, finished.** Complete the literal scope of the current task before opening any other scope. No bundling of unrelated improvements.

4. **Read before writing.** Before creating any new component or behavior, read the relevant existing code and follow the patterns already in place. Reuse existing utilities; do not parallel-invent.

5. **Test the way the user does.** End-to-end verification means running `make neext-workbench` and walking the same click path the user would walk in a browser at `http://127.0.0.1:8765`. Unit tests and Playwright specs are necessary but not sufficient. State explicitly what was tested and what was not — never claim "done" on partial verification.

6. **Pause on every central decision.** Any architectural choice (new top-level pattern, new state model, new file category, new navigation behavior, new abstraction) requires asking the user first, AND asking whether `AGENTS.md` / `CLAUDE.md` should be updated. Update them only with explicit user approval.

7. **Use the Workbench UI pattern.** Primary workflow content belongs in Center Views in the Center Panel. Do not introduce modals, alternative navigation, routes, state models, providers, or file categories unless explicitly approved. Allowed transient UI defaults are `ConfirmDialog` and `Toast`; any other modal/dialog category requires approval.

8. **Canonical run command is `make neext-workbench`.** Do not invent dev/run targets. Test at `http://127.0.0.1:8765` (single server, IPv4 loopback). Do not test at `localhost:5173` unless explicitly asked.

9. **Keep the repo clean. Use `sandbox/`.** This is an open-source project. Never drop screenshots, logs, scratch scripts, dumps, debug output, intermediate artifacts, or any other ephemeral garbage at the repo root or anywhere tracked by git. All such files go in `sandbox/`, which is gitignored (except `sandbox/workbench-mockups/`). If a tool defaults to writing to the working directory (Playwright screenshots, log files, etc.), redirect its output into `sandbox/`. Before declaring a task done, run `git status` and confirm no garbage is staged or untracked at the root.

10. **Keep `AGENTS.md` and `CLAUDE.md` synchronized.** Whenever either file is updated, review the other file and update it in the same change when the guidance applies to both. Before any `git commit`, check whether either file needs an update so repository instructions do not drift.

11. **Never push to git.** Coding agents must never run `git push`, create or update remote branches, push tags, or otherwise publish repository changes to any remote. Only the user may push. If a push seems necessary, stop after committing locally and tell the user what command they can run themselves.

12. **Use voice once.** When using voice output, call the voice tool only once for a given message; if playback seems slow, wait rather than retrying.

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

## Development Lifecycle Dashboard

The committed lifecycle planning artifact is `development-lifecycle/index.html`. It is a self-contained HTML dashboard for current project state, epics, sprints, tasks, decisions, and agent handoff context.

- Update this dashboard when a sprint is created, scoped, completed, or materially changed.
- Use the dashboard to record approved decisions and known deferred areas; do not use it to invent new product scope or bypass the "Never guess. Ask." rule.
- Local drafts, exports, notes, and scratch lifecycle material inside `development-lifecycle/` are ignored by git unless explicitly unignored.
- If lifecycle guidance changes in `AGENTS.md` or `CLAUDE.md`, keep both files synchronized in the same change.

## Workbench Shape

NEExT Workbench is a local, single-user FastAPI + React desktop-style UI for research and scientific NEExT workflows. NEExT is the core lightweight Python library; Workbench is an optional interface over real NEExT capabilities.

- Backend package: `NEExT/workbench/`
- React source: `workbench-ui/`
- Built release assets: `NEExT/workbench/static/`
- Canonical launcher: `make neext-workbench`
- Canonical browser URL: `http://127.0.0.1:8765`
- Core `pip install NEExT` must stay lightweight. Workbench dependencies belong behind the Workbench extra or repository setup.
- Core graph/network functionality belongs in NEExT. Workbench should orchestrate, expose, and enhance real NEExT workflows instead of duplicating algorithms.
- Do not build fake, stub, placeholder, or half-wired Workbench features. Every Workbench UI feature must have real behavior, real data, and user-path verification.
- Phase one Workbench is project-first. The committed frontend may contain the shell architecture for Spaces, Ribbon Groups, Ribbon Commands, Left Panel, Center Panel, Right Panel, Command Window, and status bar, but speculative workflow details are not allowed until designed.
- The current Workbench backend exposes health, workspace, project metadata, Dataset Library/project dataset APIs, Feature Library/project feature APIs, Embedding Library/project embedding APIs, Model Library/project model APIs, local serialized jobs, Dataset preparation, Feature execution, Embedding execution, Model execution, and preview APIs. Arbitrary import/export, prediction workflows, cancellation, artifact lifecycle, concurrency beyond the single worker, and broader execution behavior must still be designed and added layer by layer before code is introduced.
- Do not keep dead Workbench code for later. Remove obsolete workflow code, dialogs, hooks, API clients, schemas, and storage helpers when the corresponding behavior is intentionally deferred.

### Workbench UI Vocabulary

Use this vocabulary when asking questions, planning, implementing, and reporting Workbench changes:

- **Space**: top-row main area, such as `Home`, `Datasets`, `Features`, `Embeddings`, or `Models`.
- **Ribbon Group**: labeled group inside the ribbon, separated visually from other groups.
- **Ribbon Command**: clickable item inside a Ribbon Group.
- **Left Panel**: artifact and selection context.
- **Center Panel**: main dynamic work area.
- **Center View**: specific page/view shown in the Center Panel.
- **Right Panel**: system information area.
- **Inspector Panel**: selected-item details inside the Right Panel.
- **Jobs Panel**: job queue/status inside the Right Panel.
- **Command Window**: bottom logs, errors, and command/job output.
- **Artifact Store**: table/list of artifacts of one kind.
- **Artifact**: saved project hierarchy unit, such as project, dataset, feature set, embedding, or model.
- **Workflow Form**: functional form that creates, imports, computes, or trains something.

The UI follows this pattern:

```text
Spaces (HOME, DATASETS, FEATURES, EMBEDDINGS, MODELS)
  -> Ribbon Groups
  -> Ribbon Commands
  -> Center Views in the Center Panel
```

- The Top Row contains Spaces that logically group artifacts and actions.
- The Ribbon contains commands/subsections for the active Space.
- The Left Panel shows current artifacts and selected elements.
- The Center Panel is where primary dynamic workflow UI goes.
- The Right Panel contains system-level information such as the Inspector Panel and Jobs Panel.
- The Command Window shows logs, errors, and command/job output.
- Primary workflows must use Center Views, not random modals. New modal/dialog categories require explicit approval.

Initial Ribbon guidance:

- `Home`
  - `Project Management`: `Import`, `Create`, `Projects`
  - `App Management`: `Settings`, `Help`
- `Datasets`
  - `Dataset Management`: `Import`, `Library`, `Create`, `Datasets`
- `Features`
  - `Feature Management`: `Import`, `Library`, `Create`, `Features`
- `Embeddings`
  - `Embedding Management`: `Import`, `Library`, `Create`, `Embeddings`
- `Models`
  - `Model Management`: `Import`, `Library`, `Create`, `Models`

Approved Workbench project foundation:

```text
~/NEExT-Workbench/
  workspace.json
  projects/
    <project_uuid>/
      project.json
      jobs/
        <job_uuid>/
          job.json
      artifacts/
        datasets/
        features/
        embeddings/
        models/
```

- Project and future artifact IDs are UUIDv4 values. Folder names are IDs only; display names live in manifests and may change without moving folders.
- `workspace.json` and `project.json` use `schema_version`, `manifest_type`, `created_at`, and `updated_at`. Project manifests also contain `id`, `name`, and `description`.
- Manifests must not store absolute paths. Store relative paths only when files are referenced. Project API responses must not expose on-disk project paths.
- In the current Dataset preparation, Feature execution, Embedding execution, and Model execution phase, projects, configured Dataset artifacts, Feature artifacts, Embedding artifacts, Model artifacts, and local job records are real and persisted. Project creation creates the typed artifact directories above. Jobs are created under `jobs/<job_uuid>/job.json` when execution is requested.
- Project deletion moves `projects/<project_uuid>/` into workspace-relative `trash/projects/<folder-name>/`; when the direct trash folder exists, use a suffixed folder instead of overwriting it.
- Delete APIs/UI must not expose absolute paths. Trash paths are workspace-relative.
- Deletion is project-only until the artifact lifecycle is explicitly designed; do not add artifact deletion, restore, archive, or lifecycle behavior without approval.
- Dataset Library entries are source templates, not executable project artifacts. A catalog row must be configured into a project Dataset artifact before it can participate in the compute graph.
- Dataset artifacts are real project artifacts under `artifacts/datasets/<dataset_uuid>/artifact.json`, with UUIDv4 artifact IDs, typed file references, source metadata, preparation operation specs, and status.
- Workbench compute graph starts at Dataset artifacts. Dataset artifacts are explicit DAG roots and their manifests have `inputs: []`.
- Future non-dataset artifact folders should use `artifacts/<kind>/<artifact_uuid>/artifact.json`, with UUIDv4 artifact IDs and typed file references.
- Artifact lineage is a general DAG represented by typed input references in artifact manifests. This must support one dataset to many feature sets, multiple feature sets to one embedding, one artifact to many downstream artifacts, and future multi-dataset workflows.
- Configured Dataset artifacts are planned compute graph nodes until Dataset preparation runs. Dataset preparation uses NEExT graph construction and normalization semantics, writes a raw Parquet snapshot, prepared NEExT-ready graph Parquet files, complete source-to-internal mapping Parquet files, summary stats, and status/error metadata.
- Feature artifacts define compute graph nodes and may execute once their configured Dataset input is prepared. One Feature artifact targets one Dataset artifact and one built-in structural node feature method, records one dataset input, stores an operation spec with stable operation ID and version, and writes feature output Parquet only on successful execution.
- Embedding Library entries are built-in graph embedding algorithm templates, not executable project artifacts. A catalog row must be configured into a project Embedding artifact before it can participate in the compute graph.
- Embedding artifacts define compute graph nodes downstream of one or more Feature artifacts. All selected Feature inputs must reference the same Dataset. Embedding execution can auto-run planned or failed upstream Dataset preparation and Feature computation before computing graph-level embeddings.
- Workbench persists Embedding manifests, graph embedding Parquet outputs, jobs, readable job logs, preview metadata, and output file metadata.
- Model Library entries are built-in supervised graph model algorithm templates, not executable project artifacts. A catalog row must be configured into a project Model artifact before it can participate in the compute graph.
- Model artifacts define planned DAG nodes downstream of one or more Embedding artifacts. All selected Embedding inputs must trace to the same Dataset. Model execution can auto-run planned or failed upstream Embedding, Feature, and Dataset work before training.
- Workbench persists Model manifests, trained model files, metrics JSON, jobs, readable job logs, and metrics previews.
- Future artifacts are immutable once saved. Edits create new artifact IDs.
- Workbench canonical dataset storage is Parquet plus `artifact.json`; CSV is an import/source format, not the canonical Workbench dataset format.
- Dataset Library v1 uses curated NEExT CSV bundles as graph-collection source templates and may also include curated single-graph CSV source templates. Dataset preparation downloads/loads graph collections directly and prepares single graphs into downstream graph collections through k-hop egonets.
- Dataset manifests and APIs must expose only artifact/workspace/project-relative paths. Do not expose on-disk absolute project paths.
- Feature execution depends on configured Dataset artifacts, never directly on Dataset Library catalog entries or source-shaped imported data. If a Feature run targets a planned Dataset, Workbench prepares the Dataset first.
- Workbench persists raw snapshots, prepared graph data, mappings, jobs, readable job logs, preview metadata, and output files. Browser previews must stay limited/paginated and must not load complete large mapping or output files by default.
- Do not add arbitrary URL imports, PyG/DGL/OGB providers, dataset deletion, feature deletion, embedding deletion, model deletion, feature editing, embedding editing, model editing, feature duplication, embedding duplication, model duplication, model import/export, prediction workflows, feature-importance views, restore, archive, cancellation, additional status transitions, or artifact lifecycle behavior without explicit design approval.
- Future project archives should contain `project.json` and `artifacts/` at zip root. If an imported project UUID already exists, import it as a copy with a new UUID and preserve the original UUID in metadata.
- Do not keep dead Workbench code for later. Add import/export, prediction workflows, feature-importance views, cancellation, concurrency beyond the single local worker, additional storage categories, and broader execution behavior only after each layer is explicitly designed.

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
python3 -m pytest tests/test_workbench.py
python3 test.py
python3 test_networkx.py
python3 -m ruff check .
python3 -m black --check .
python3 -m isort --check-only .
python3 -m mypy .
```

Workbench-specific checks:

```bash
python3 -m pytest tests/test_workbench.py
cd workbench-ui && npm exec tsc
cd workbench-ui && npm run test:e2e
make neext-workbench
```

After `make neext-workbench`, verify behavior with Playwright MCP browser testing at `http://127.0.0.1:8765` by walking the same path a user would walk. `npm run build` writes `NEExT/workbench/static/`; those built assets are intentional release artifacts so installed users can launch the Workbench without Node.

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

## Workbench Feature Workflow

1. Read `AGENTS.md`, `CLAUDE.md`, and the relevant Workbench files before changing behavior.
2. Confirm the requested scope maps to the shared vocabulary: Space, Ribbon Group, Ribbon Command, Left Panel, Center View, Right Panel, Inspector Panel, Jobs Panel, Command Window, Artifact Store, Artifact, or Workflow Form.
3. Ask before every central decision, including new navigation behavior, state model, API shape, file category, provider/context, modal, or reusable abstraction.
4. If the user's wording is ambiguous or conflicts with the Workbench pattern, clarify before planning or editing.
5. Implement only the approved scope. Do not add fake, stub, placeholder, or half-wired features.
6. Run the narrowest relevant automated checks first, then the Workbench-specific checks above when the UI or API is affected.
7. For UI work, perform Playwright MCP browser testing through the user path at `http://127.0.0.1:8765`.
8. Run `git status` before reporting completion and account for every modified or untracked path.

Temporary artifacts from Workbench development, including screenshots, traces, logs, local e2e workspaces, dumps, and scratch scripts, belong under `sandbox/`. The only intended shareable sandbox exception is `sandbox/workbench-mockups/`.

## Security and Secrets

- Never commit API keys, tokens, `.env` files, credentials, private datasets, or local workspace contents.
- Do not paste secret values into issue reports, docs, tests, logs, screenshots, or agent replies.
- If a secret appears in any local agent config or shell history, move it to an environment-specific mechanism and rotate it if the value may have been exposed.

## Testing Guidance

- For generator changes, start with `python3 -m pytest tests/test_generators.py`.
- For graph IO, collection, feature, or embedding changes, run the most relevant targeted pytest tests first, then `python3 -m pytest` when dependencies are available.
- For end-to-end behavior, use `python3 test.py` and `python3 test_networkx.py` as smoke checks.
- For docs-only changes, build docs only when the changed content affects Sphinx output.
- If dependencies are missing, do not install packages globally. Use a virtual environment or report the blocker.

## Repo-Specific Cautions

- Package version is sourced from `NEExT/__init__.py` via Hatch dynamic versioning. Update `__version__` there for release work; do not add a static `version = ...` field back to `pyproject.toml`.
- `CLAUDE.md` is useful context, but it underemphasizes the current `generators` and `outliers` packages. Inspect those modules directly for related work.
- `pyproject.toml` defines `dev`, `docs`, `experiments`, `advanced`, `dgl`, `all`, and `all-dl` extras. Do not assume a separate `test` extra exists.
- The root `Makefile` includes documentation and direct PyPI release targets. Quality commands are configured in `pyproject.toml`, not as Makefile targets.
- PyPI publication is direct/local only. Use `make release-check` plus `make deploy` for the full push/tag/publish flow, or `make publish-only` after `main` has already been pushed. Do not publish packages, create release tags, or push to remotes unless explicitly requested.

## Custom Feature Contract

Custom node feature functions registered through `my_feature_methods` should accept one graph object and return a `pandas.DataFrame` with columns in this order:

```text
node_id, graph_id, <feature columns...>
```

Keep feature names stable and deterministic. For notebook-defined functions, remember the repo relies on `joblib`/cloudpickle-compatible behavior; avoid closing over unpicklable objects.
