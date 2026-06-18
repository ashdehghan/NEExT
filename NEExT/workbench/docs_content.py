"""Canonical NEExT Workbench operational documentation.

Single source of truth shared by:
  * the React Settings -> Docs tab (served via ``GET /api/docs``), and
  * the MCP server (doc resources, server ``instructions``, and recipe prompts).

Keeping both consumers on this one module prevents the UI docs and the
agent-facing docs from drifting apart. Topic content mirrors the shape the
React renderer expects: ``{id, title, summary, sections: [{heading, body?,
bullets?, code?}]}``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Operational doc topics (rendered by the UI Settings Docs tab and exposed to
# MCP clients). Ordering here is the display order.
# ---------------------------------------------------------------------------

DOC_TOPICS: list[dict[str, Any]] = [
    {
        "id": "overview",
        "title": "Overview",
        "summary": "NEExT Workbench is a local, project-first interface over real NEExT graph machine learning workflows.",
        "sections": [
            {
                "heading": "What Workbench Manages",
                "body": [
                    "Workbench organizes work into projects. Each project owns datasets, feature sets, embeddings, models, jobs, and local trash state.",
                    "The main workflow is dataset-first: add or select a Dataset artifact, compute Feature artifacts, compute Embedding artifacts, then train Model artifacts.",
                ],
            },
            {
                "heading": "Current Boundaries",
                "bullets": [
                    "Project Create and custom Feature Create are the active Create workflows.",
                    "Dataset, Embedding, and Model Create are archived until their workflows are designed.",
                    "Broader import/export remains deferred except for Dataset Import and current-table CSV export in Dataset Explore.",
                    "Operational docs live here in Settings Docs.",
                ],
            },
        ],
    },
    {
        "id": "workbench-flow",
        "title": "Workbench Flow",
        "summary": "Use the top Spaces, Ribbon commands, Left Panel context, and Center Views together.",
        "sections": [
            {
                "heading": "Navigation Model",
                "bullets": [
                    "Spaces group the main work areas: Home, Datasets, Features, Embeddings, and Models.",
                    "Ribbon commands switch the active Center View inside the selected Space.",
                    "The Left Panel shows the active project branch and scopes downstream artifact lists.",
                    "The Right Panel shows Inspector details and job status; the Command Window shows job output.",
                ],
            },
            {
                "heading": "Typical Path",
                "bullets": [
                    "Create or select a project from Home.",
                    "Add a Dataset from the Dataset Library, then prepare it.",
                    "Add built-in Features or create a custom Feature for the active Dataset, then compute Feature jobs.",
                    "Add Embeddings from Feature artifacts, then train Models from Embedding artifacts.",
                    "Use Explore views to inspect statistics, plots, tables, lineage, and previews.",
                ],
            },
        ],
    },
    {
        "id": "projects-datasets",
        "title": "Projects and Datasets",
        "summary": "Projects are the workspace boundary. Datasets are explicit roots of the compute graph.",
        "sections": [
            {
                "heading": "Projects",
                "body": [
                    "A project stores its manifest, jobs, and typed artifact folders under the local Workbench workspace. Project display names can change without changing artifact IDs.",
                    "Project deletion moves the project folder to workspace trash. Restore is available from Home Trash when the original live project folder does not already exist.",
                ],
            },
            {
                "heading": "Datasets",
                "bullets": [
                    "Dataset Library rows are templates, not executable project artifacts.",
                    "A catalog row must be added to the project as a Dataset artifact before Features, Embeddings, or Models can use it.",
                    "Dataset preparation writes canonical Parquet outputs, mappings, summaries, and job logs.",
                    "Browser previews stay limited and paginated; they do not load complete large files by default.",
                ],
            },
        ],
    },
    {
        "id": "dataset-intake",
        "title": "Dataset Import Contract",
        "summary": "Dataset Import creates a Draft Dataset artifact from a set of NEExT tables supplied as CSV files (browser) or records/CSV text (MCP).",
        "sections": [
            {
                "heading": "Required and Optional Tables",
                "bullets": [
                    "Required: 'edges' with columns src_node_id, dest_node_id.",
                    "Required: 'node_graph_mapping' with columns node_id, graph_id.",
                    "Optional: 'graph_labels' with columns graph_id, graph_label.",
                    "Optional: 'node_features' beginning with node_id, then one or more feature columns.",
                    "Optional: 'edge_features' beginning with src_node_id, dest_node_id, then one or more feature columns.",
                    "Unknown table names are rejected. Valid names are edges, node_graph_mapping, graph_labels, node_features, edge_features.",
                ],
            },
            {
                "heading": "Node ID Rule",
                "body": [
                    "Node IDs must be integer-compatible. node_id, src_node_id, and dest_node_id are coerced to integers, and non-integer values are rejected because current NEExT graph objects require integer node IDs.",
                ],
            },
            {
                "heading": "Browser Import",
                "bullets": [
                    "Datasets Space -> Import opens the Dataset Import Center View.",
                    "Upload the matching CSV files, or a single zip containing those CSV files.",
                    "Preview the parsed tables, then create the Draft Dataset artifact.",
                ],
            },
            {
                "heading": "MCP Intake",
                "bullets": [
                    "Single-shot: neext_validate_dataset_intake checks tables supplied as records or csv without creating an artifact.",
                    "Session flow: neext_create_dataset_intake_session, then neext_append_dataset_intake_table for each table, then neext_validate_dataset_intake_session, then neext_create_dataset_from_intake.",
                    'Each table payload is {"format": "records"|"csv", "records": [...]} or {"format": "csv", "csv": "..."}. The default format is records.',
                    "Always validate before create. Create returns a Draft Dataset artifact with source_type uploaded_neext_tables; run it with neext_run_artifacts to prepare it.",
                ],
                "code": (
                    'neext_create_dataset_intake_session(project_id, name="My Graphs")\n'
                    'neext_append_dataset_intake_table(project_id, session_id, table_name="edges", table={\n'
                    '    "format": "records",\n'
                    '    "records": [{"src_node_id": 0, "dest_node_id": 1}, {"src_node_id": 1, "dest_node_id": 2}],\n'
                    "})\n"
                    'neext_append_dataset_intake_table(project_id, session_id, table_name="node_graph_mapping", table={\n'
                    '    "format": "records",\n'
                    '    "records": [{"node_id": 0, "graph_id": 0}, {"node_id": 1, "graph_id": 0}, {"node_id": 2, "graph_id": 0}],\n'
                    "})\n"
                    "neext_validate_dataset_intake_session(project_id, session_id)\n"
                    "neext_create_dataset_from_intake(project_id, session_id)"
                ),
            },
        ],
    },
    {
        "id": "features-custom",
        "title": "Features and Custom Features",
        "summary": "Feature artifacts compute node-level values for one prepared Dataset artifact.",
        "sections": [
            {
                "heading": "Built-in Features",
                "body": [
                    "Feature Library workflows are dataset-first. Select a Dataset in the Left Panel, choose a feature method from the Library, create the Draft artifact, then compute it.",
                    "If a Feature computation targets a Draft Dataset, Workbench prepares the Dataset first before computing the Feature output.",
                ],
            },
            {
                "heading": "Custom Feature Contract",
                "bullets": [
                    "Custom Feature Create requires an active completed Dataset artifact.",
                    "The Python code must define compute_feature(graph).",
                    "The function must return a pandas.DataFrame with columns ordered as node_id, graph_id, then one or more numeric feature columns.",
                    "Validate runs against the first prepared graph. Create repeats backend validation before creating the Draft Feature artifact.",
                    "Custom code is trusted local Python, not sandboxed. Missing packages are reported clearly, but Workbench does not install packages.",
                ],
                "code": (
                    "import pandas as pd\n\n"
                    "def compute_feature(graph):\n"
                    "    nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)\n"
                    "    values = [float(graph.G.degree(node)) for node in nodes]\n"
                    "    df = pd.DataFrame({\n"
                    '        "node_id": nodes,\n'
                    '        "graph_id": graph.graph_id,\n'
                    '        "custom_degree": values,\n'
                    "    })\n"
                    '    return df[["node_id", "graph_id", "custom_degree"]]'
                ),
            },
        ],
    },
    {
        "id": "embeddings-models",
        "title": "Embeddings and Models",
        "summary": "Embeddings summarize graph-level structure from Feature artifacts. Models train from Embedding artifacts.",
        "sections": [
            {
                "heading": "Embeddings",
                "bullets": [
                    "Embedding Library rows are templates. Add one into a project Embedding artifact from the active Dataset context.",
                    "Embedding Add-to-Project lists Feature artifacts from the active Dataset only.",
                    "When an active Feature belongs to the active Dataset branch, it is preselected.",
                    "Embedding execution can auto-run Draft or failed upstream Dataset and Feature work.",
                ],
            },
            {
                "heading": "Models",
                "bullets": [
                    "Model Library rows are templates. Add one into a project Model artifact from the active Dataset context.",
                    "Model Add-to-Project lists Embedding artifacts from the active Dataset only.",
                    "When an active Embedding belongs to the active Dataset branch, it is preselected.",
                    "Model execution can auto-run Draft or failed upstream Embedding, Feature, and Dataset work before training.",
                ],
            },
        ],
    },
    {
        "id": "artifact-lifecycle",
        "title": "Artifact Lifecycle",
        "summary": "Artifacts are persisted compute graph nodes with lineage, status, jobs, and trash behavior.",
        "sections": [
            {
                "heading": "Lineage",
                "body": [
                    "Dataset artifacts are roots. Feature artifacts reference Dataset inputs, Embedding artifacts reference Feature inputs, and Model artifacts reference Embedding inputs.",
                    "The Left Panel and Center Views use that lineage to keep project, dataset, feature, embedding, and model context consistent.",
                ],
            },
            {
                "heading": "Delete and Restore",
                "bullets": [
                    "Leaf artifacts can be deleted directly.",
                    "Artifacts with downstream dependents require a cascade confirmation.",
                    "Cascade delete moves the selected artifact and all downstream dependents into one project-scoped trash bundle.",
                    "Bundle restore is available when no live target folder conflicts.",
                    "Queued or running jobs targeting any artifact in the delete set block deletion.",
                ],
            },
        ],
    },
    {
        "id": "library-quickstart",
        "title": "NEExT Library Quickstart",
        "summary": "Use the Python library directly when you want notebook or script control outside Workbench.",
        "sections": [
            {
                "heading": "Core Pipeline",
                "body": [
                    "The library flow mirrors Workbench: load graph data, compute node features, compute graph embeddings, then train or evaluate a model.",
                    "Graph data can come from CSV files, DataFrames, or NetworkX graphs. Workbench uses the same underlying NEExT capabilities for its persisted workflows.",
                ],
                "code": (
                    "from NEExT.framework import NEExT\n\n"
                    "nxt = NEExT()\n"
                    "graphs = nxt.read_from_csv(\n"
                    '    "edges.csv",\n'
                    '    "node_graph_mapping.csv",\n'
                    '    "graph_labels.csv",\n'
                    ")\n"
                    'features = nxt.compute_node_features(graphs, feature_list=["all"])\n'
                    "embeddings = nxt.compute_graph_embeddings(\n"
                    "    graphs,\n"
                    "    features,\n"
                    '    embedding_algorithm="approx_wasserstein",\n'
                    "    embedding_dimension=8,\n"
                    ")\n"
                    'results = nxt.train_ml_model(graphs, embeddings, model_type="classifier")'
                ),
            },
            {
                "heading": "CSV Inputs",
                "bullets": [
                    "edges.csv uses src_node_id and dest_node_id.",
                    "node_graph_mapping.csv uses node_id and graph_id.",
                    "graph_labels.csv uses graph_id and graph_label.",
                    "Optional node feature CSVs start with node_id and graph_id, followed by feature columns.",
                ],
            },
        ],
    },
    {
        "id": "agentic-mcp",
        "title": "Agentic and MCP",
        "summary": "The Agentic tab exposes local MCP setup for clients that need controlled Workbench access.",
        "sections": [
            {
                "heading": "MCP Setup",
                "bullets": [
                    "Enable MCP from Settings Agentic when a supported local client needs Workbench access.",
                    "Workbench shows the full token only once. Regenerate it when a fresh token and copy-ready snippets are needed.",
                    "Claude Desktop only reaches a local server over stdio: its snippet launches the Python interpreter running Workbench with the Workbench MCP launcher.",
                    "MCP Inspector, Cursor, Claude Code, and generic clients can use the local Streamable HTTP endpoint at http://127.0.0.1:8765/mcp.",
                    "Remote Claude connectors are separate from claude_desktop_config.json and require a publicly reachable server, not 127.0.0.1.",
                    "On macOS, Claude Desktop cannot read servers inside ~/Desktop, ~/Documents, or ~/Downloads. When the Python environment lives there, Workbench replaces the Claude Desktop snippet with remediation steps instead of a config that would fail to launch.",
                    "Disable MCP when clients should no longer connect.",
                ],
            },
            {
                "heading": "Agentic Behavior",
                "bullets": [
                    "MCP tools can read catalogs and artifacts, add current approved library workflows to projects, run jobs, preview and analyze outputs, request Workbench navigation, and record visible activity.",
                    "Tool access is gated by scopes: read, write, run, custom-code, ui-control, export, and lifecycle.",
                    "MCP UI navigation can open existing Spaces, Center Views, artifacts, graphs, nodes, and approved add/create form drafts.",
                    "MCP delete tools create Workbench approval requests instead of deleting immediately.",
                    "Recent MCP activity is visible in Settings Agentic and the Command Window.",
                ],
            },
            {
                "heading": "Security Boundary",
                "body": [
                    "MCP setup is local Workbench configuration. Keep generated tokens out of commits, screenshots, logs, and shared notes.",
                    "Custom feature code is trusted local Python and runs in the local Workbench environment. Delete requests are enforced through Workbench approval.",
                    "Remote OAuth, multi-user MCP hosting, and deferred Workbench workflows are not part of the current MCP surface.",
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Recipes: ordered, multi-step workflows an agent cannot infer from individual
# tool descriptions. Surfaced as MCP prompts and as a doc resource.
# ---------------------------------------------------------------------------

RECIPES: list[dict[str, Any]] = [
    {
        "id": "upload_neext_dataset",
        "title": "Upload a dataset from agent-supplied tables",
        "summary": "Create a Draft Dataset artifact from NEExT tables you provide, then prepare it.",
        "arguments": [
            {"name": "project_id", "description": "Target project ID (create one first with neext_create_project if needed).", "required": True},
        ],
        "steps": [
            "Confirm or create the target project with neext_list_projects / neext_create_project.",
            "Create an intake session with neext_create_dataset_intake_session(project_id, name).",
            "Append the required 'edges' table (columns src_node_id, dest_node_id) with neext_append_dataset_intake_table.",
            "Append the required 'node_graph_mapping' table (columns node_id, graph_id). Node IDs must be integer-compatible.",
            "Optionally append 'graph_labels', 'node_features', and/or 'edge_features'.",
            "Validate with neext_validate_dataset_intake_session and fix any reported errors.",
            "Create the Draft Dataset with neext_create_dataset_from_intake.",
            "Prepare it with neext_run_artifacts(kind='dataset', ids=[dataset_id]) and poll neext_get_job until completed.",
        ],
    },
    {
        "id": "run_end_to_end_pipeline",
        "title": "Run an end-to-end pipeline",
        "summary": "From a prepared Dataset, configure Feature -> Embedding -> Model and run each, polling jobs.",
        "arguments": [
            {"name": "project_id", "description": "Project containing a prepared Dataset artifact.", "required": True},
        ],
        "steps": [
            "Identify a completed Dataset with neext_list_artifacts(kind='dataset'); prepare it with neext_run_artifacts if still Draft.",
            "Add a Feature with neext_configure_feature (built-in) or neext_configure_custom_feature (custom Python), referencing the Dataset.",
            "Add an Embedding with neext_configure_embedding from one or more Feature artifacts on the same Dataset.",
            "Add a Model with neext_configure_model from one or more Embedding artifacts on the same Dataset.",
            "Run each artifact with neext_run_artifacts; execution can auto-run Draft or failed upstream work.",
            "Poll neext_list_jobs / neext_get_job until jobs complete, then summarize outputs and any errors.",
        ],
    },
    {
        "id": "inspect_neext_results",
        "title": "Inspect results",
        "summary": "Preview and analyze completed artifacts, and drill into specific graphs or nodes.",
        "arguments": [
            {"name": "project_id", "description": "Project to inspect.", "required": True},
        ],
        "steps": [
            "List artifacts per kind with neext_list_artifacts and pick completed ones.",
            "Preview tabular outputs with neext_preview_artifact (paginated; pass a table for datasets).",
            "Run neext_analyze_artifact for summary stats and plot data with bounded sampling.",
            "Use neext_search_graphs to locate graphs or nodes instead of listing everything.",
            "Drill into one graph or node with neext_get_graph_detail.",
            "Optionally call neext_set_workbench_view to open the matching Explore view in the live UI.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Server instructions: concise orientation injected into the MCP client's
# context on connect. Intentionally brief -- it points to richer resources
# rather than duplicating them.
# ---------------------------------------------------------------------------

SERVER_INSTRUCTIONS = (
    "NEExT Workbench is a local, single-user graph machine learning workbench. Work flows through a "
    "dataset-first compute DAG: Dataset (root) -> Feature (node-level) -> Embedding (graph-level) -> Model "
    "(supervised). Artifacts live inside projects.\n\n"
    "Core loop: list/create a project, add or import a Dataset, then configure and run Feature, Embedding, and "
    "Model artifacts. Configure tools (neext_configure_*) create Draft artifacts; nothing computes until you call "
    "neext_run_artifacts, which returns jobs immediately -- poll neext_get_job / neext_list_jobs until they finish. "
    "Running an artifact can auto-run Draft or failed upstream artifacts.\n\n"
    "Dataset import: supply NEExT tables ('edges' and 'node_graph_mapping' required; 'graph_labels', 'node_features', "
    "'edge_features' optional). Node IDs must be integer-compatible. Always validate before create. See the "
    "neext://docs/dataset-intake resource for the exact column contract and an example.\n\n"
    "Inspecting results: prefer neext_search_graphs and neext_get_graph_detail over listing everything; use "
    "neext_preview_artifact (paginated) and neext_analyze_artifact (bounded sampling) for outputs.\n\n"
    "Driving the UI: neext_set_workbench_view requests the open Workbench to navigate to a Space, Center View, "
    "artifact, graph, node, or pre-filled form draft.\n\n"
    "Scopes gate tools: read, write, run, custom-code, ui-control, export, lifecycle. Deletes are never immediate -- "
    "neext_request_delete_* create Workbench approval requests a human must approve. Custom Feature code is trusted "
    "local Python, not sandboxed.\n\n"
    "Start with the neext://docs/workbench and neext://docs/recipes resources, and the explore_neext_project, "
    "configure_neext_pipeline, and upload_neext_dataset prompts."
)


# ---------------------------------------------------------------------------
# Rendering helpers (Markdown) for MCP doc resources.
# ---------------------------------------------------------------------------


def find_topic(topic_id: str) -> dict[str, Any] | None:
    for topic in DOC_TOPICS:
        if topic["id"] == topic_id:
            return topic
    return None


def _section_markdown(section: dict[str, Any]) -> str:
    lines: list[str] = [f"## {section['heading']}", ""]
    for paragraph in section.get("body", []) or []:
        lines.append(paragraph)
        lines.append("")
    for bullet in section.get("bullets", []) or []:
        lines.append(f"- {bullet}")
    if section.get("bullets"):
        lines.append("")
    code = section.get("code")
    if code:
        lines.append("```")
        lines.append(code)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def topic_markdown(topic: dict[str, Any]) -> str:
    parts: list[str] = [f"# {topic['title']}", "", topic["summary"], ""]
    for section in topic["sections"]:
        parts.append(_section_markdown(section))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def recipes_markdown() -> str:
    parts: list[str] = ["# NEExT Workbench Recipes", ""]
    for recipe in RECIPES:
        parts.append(f"## {recipe['title']}")
        parts.append("")
        parts.append(recipe["summary"])
        parts.append("")
        for index, step in enumerate(recipe["steps"], start=1):
            parts.append(f"{index}. {step}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
