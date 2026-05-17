"""MCP server wiring for NEExT Workbench."""

from __future__ import annotations

import json
from typing import Any, Callable

from .mcp_service import WorkbenchMcpService


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "neext_workspace_summary": _object_schema({}),
    "neext_list_projects": _object_schema({}),
    "neext_create_project": _object_schema(
        {
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string", "default": ""},
        },
        ["name"],
    ),
    "neext_list_catalog": _object_schema({"kind": {"type": "string"}}, ["kind"]),
    "neext_list_artifacts": _object_schema({"project_id": {"type": "string"}, "kind": {"type": "string"}}, ["project_id", "kind"]),
    "neext_get_artifact": _object_schema(
        {"project_id": {"type": "string"}, "kind": {"type": "string"}, "artifact_id": {"type": "string"}},
        ["project_id", "kind", "artifact_id"],
    ),
    "neext_configure_dataset": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_catalog_id": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "source_catalog_id"],
    ),
    "neext_configure_feature": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_dataset_id": {"type": "string"},
            "source_feature_id": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "source_dataset_id", "source_feature_id"],
    ),
    "neext_configure_embedding": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_embedding_id": {"type": "string"},
            "source_feature_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "source_embedding_id", "source_feature_ids"],
    ),
    "neext_configure_model": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_model_id": {"type": "string"},
            "source_embedding_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "params": {"type": "object", "additionalProperties": True},
        },
        ["project_id", "source_model_id", "source_embedding_ids", "params"],
    ),
    "neext_run_artifacts": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        ["project_id", "kind", "ids"],
    ),
    "neext_list_jobs": _object_schema({"project_id": {"type": "string"}}, ["project_id"]),
    "neext_get_job": _object_schema({"project_id": {"type": "string"}, "job_id": {"type": "string"}}, ["project_id", "job_id"]),
    "neext_preview_artifact": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "artifact_id": {"type": "string"},
            "table": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
        },
        ["project_id", "kind", "artifact_id"],
    ),
    "neext_analyze_artifact": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "artifact_id": {"type": "string"},
            "options": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "kind", "artifact_id"],
    ),
    "neext_search_graphs": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "artifact_id": {"type": "string"},
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
        },
        ["project_id", "kind", "artifact_id", "query"],
    ),
    "neext_get_graph_detail": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "artifact_id": {"type": "string"},
            "graph_id": {"type": "string"},
            "node_id": {"type": "string"},
        },
        ["project_id", "kind", "artifact_id", "graph_id"],
    ),
}


TOOL_DESCRIPTIONS = {
    "neext_workspace_summary": "Summarize the current NEExT Workbench workspace without exposing project paths.",
    "neext_list_projects": "List Workbench projects available in the current workspace.",
    "neext_create_project": "Create a new Workbench project in the current workspace.",
    "neext_list_catalog": "List source catalog entries for datasets, features, embeddings, or models.",
    "neext_list_artifacts": "List saved project artifacts of one kind.",
    "neext_get_artifact": "Read one saved project artifact manifest.",
    "neext_configure_dataset": "Create a planned Dataset artifact from a Dataset Library catalog entry.",
    "neext_configure_feature": "Create a planned Feature artifact from a Dataset artifact and Feature Library entry.",
    "neext_configure_embedding": "Create a planned Embedding artifact from one or more Feature artifacts.",
    "neext_configure_model": "Create a planned Model artifact from one or more Embedding artifacts.",
    "neext_run_artifacts": "Start Workbench jobs for planned or failed artifacts and return their job manifests immediately.",
    "neext_list_jobs": "List local Workbench jobs for a project.",
    "neext_get_job": "Read one local Workbench job manifest.",
    "neext_preview_artifact": "Preview artifact outputs with pagination; never loads full output files by default.",
    "neext_analyze_artifact": "Run the existing Workbench analysis view for a completed artifact using bounded sampling options.",
    "neext_search_graphs": "Search graphs or nodes inside completed Dataset, Feature, or Embedding analysis data.",
    "neext_get_graph_detail": "Inspect one graph or dataset node from completed Dataset, Feature, or Embedding analysis data.",
}


PROMPTS = {
    "explore_neext_project": "Explore project {project_id}: list artifacts, inspect completed outputs, summarize lineage, and identify useful next analysis steps.",
    "configure_neext_pipeline": "Configure a NEExT pipeline in project {project_id}: choose catalog entries, create planned artifacts, and stop before running jobs unless asked.",
    "run_neext_pipeline": "Run planned or failed artifacts in project {project_id}, poll jobs until completion, and summarize outputs and errors.",
    "compare_neext_models": "Compare completed model artifacts in project {project_id}; use metrics previews and explain differences in model performance.",
    "investigate_neext_graph": "Investigate graph {graph_id} in project {project_id}; search related graph/node details and summarize structural observations.",
}


def _json_content(payload: Any):
    from mcp import types

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2, sort_keys=True))]


def _tool_handlers(service: WorkbenchMcpService) -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "neext_workspace_summary": lambda args: service.workspace_summary(),
        "neext_list_projects": lambda args: service.list_projects(),
        "neext_create_project": lambda args: service.create_project(args["name"], args.get("description", "")),
        "neext_list_catalog": lambda args: service.list_catalog(args["kind"]),
        "neext_list_artifacts": lambda args: service.list_artifacts(args["project_id"], args["kind"]),
        "neext_get_artifact": lambda args: service.get_artifact(args["project_id"], args["kind"], args["artifact_id"]),
        "neext_configure_dataset": lambda args: service.configure_dataset(
            args["project_id"],
            args["source_catalog_id"],
            args.get("params"),
        ),
        "neext_configure_feature": lambda args: service.configure_feature(
            args["project_id"],
            args["source_dataset_id"],
            args["source_feature_id"],
            args.get("params"),
        ),
        "neext_configure_embedding": lambda args: service.configure_embedding(
            args["project_id"],
            args["source_embedding_id"],
            args["source_feature_ids"],
            args.get("params"),
        ),
        "neext_configure_model": lambda args: service.configure_model(
            args["project_id"],
            args["source_model_id"],
            args["source_embedding_ids"],
            args["params"],
        ),
        "neext_run_artifacts": lambda args: service.run_artifacts(args["project_id"], args["kind"], args["ids"]),
        "neext_list_jobs": lambda args: service.list_jobs(args["project_id"]),
        "neext_get_job": lambda args: service.get_job(args["project_id"], args["job_id"]),
        "neext_preview_artifact": lambda args: service.preview_artifact(
            args["project_id"],
            args["kind"],
            args["artifact_id"],
            table=args.get("table"),
            limit=int(args.get("limit", 20)),
            offset=int(args.get("offset", 0)),
        ),
        "neext_analyze_artifact": lambda args: service.analyze_artifact(
            args["project_id"],
            args["kind"],
            args["artifact_id"],
            options=args.get("options"),
        ),
        "neext_search_graphs": lambda args: service.search_graphs(
            args["project_id"],
            args["kind"],
            args["artifact_id"],
            args["query"],
            limit=int(args.get("limit", 25)),
        ),
        "neext_get_graph_detail": lambda args: service.get_graph_detail(
            args["project_id"],
            args["kind"],
            args["artifact_id"],
            args["graph_id"],
            node_id=args.get("node_id"),
        ),
    }


def create_mcp_server(service: WorkbenchMcpService):
    """Create an MCP Server instance backed by a WorkbenchMcpService."""

    from mcp import types
    from mcp.server import Server

    server = Server("neext-workbench")
    handlers = _tool_handlers(service)

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                inputSchema=TOOL_SCHEMAS[name],
            )
            for name in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        if name not in handlers:
            raise ValueError(f"Unknown tool: {name}")
        return _json_content(handlers[name](arguments or {}))

    @server.list_resources()
    async def list_resources():
        resources = [
            types.Resource(
                uri="neext://workspace",
                name="Workspace",
                description="Current NEExT Workbench workspace summary.",
                mimeType="application/json",
            )
        ]
        for project in service.store.list_projects():
            resources.append(
                types.Resource(
                    uri=f"neext://projects/{project.id}",
                    name=f"Project: {project.name}",
                    description="NEExT Workbench project manifest.",
                    mimeType="application/json",
                )
            )
            resources.append(
                types.Resource(
                    uri=f"neext://projects/{project.id}/artifacts",
                    name=f"Artifacts: {project.name}",
                    description="All project artifacts grouped by kind.",
                    mimeType="application/json",
                )
            )
            for job in service.store.list_jobs(project.id):
                resources.append(
                    types.Resource(
                        uri=f"neext://projects/{project.id}/jobs/{job.id}",
                        name=f"Job: {job.id}",
                        description="NEExT Workbench job manifest.",
                        mimeType="application/json",
                    )
                )
        return resources

    @server.read_resource()
    async def read_resource(uri):
        uri_text = str(uri)
        if uri_text == "neext://workspace":
            return json.dumps(service.workspace_summary(), indent=2, sort_keys=True)
        parts = uri_text.removeprefix("neext://").split("/")
        if len(parts) == 2 and parts[0] == "projects":
            return json.dumps(service._dump(service.store.read_project(parts[1])), indent=2, sort_keys=True)
        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "artifacts":
            return json.dumps(service.all_project_artifacts(parts[1]), indent=2, sort_keys=True)
        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "jobs":
            return json.dumps(service.get_job(parts[1], parts[3]), indent=2, sort_keys=True)
        raise ValueError(f"Unknown resource URI: {uri_text}")

    @server.list_prompts()
    async def list_prompts():
        return [
            types.Prompt(
                name=name,
                description=template,
                arguments=[
                    types.PromptArgument(name="project_id", description="Workbench project UUID.", required="project_id" in template),
                    types.PromptArgument(name="graph_id", description="Prepared graph ID.", required="graph_id" in template),
                ],
            )
            for name, template in PROMPTS.items()
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None):
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt: {name}")
        args = arguments or {}
        prompt_text = PROMPTS[name].format(
            project_id=args.get("project_id", "<project_id>"),
            graph_id=args.get("graph_id", "<graph_id>"),
        )
        return types.GetPromptResult(
            description=TOOL_DESCRIPTIONS.get(name, name),
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_text),
                )
            ],
        )

    return server
