"""MCP server wiring for NEExT Workbench."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .mcp_service import WorkbenchMcpService
from .schemas import McpStdioReadiness
from .storage import WorkbenchStore

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_TRANSPORT = "streamable-http"
MCP_SERVER_NAME = "neext-workbench"
MCP_SERVER_VERSION = "0.1.0"

MCP_SCOPE_READ = "read"
MCP_SCOPE_WRITE = "write"
MCP_SCOPE_RUN = "run"
MCP_SCOPE_CUSTOM_CODE = "custom-code"
MCP_SCOPE_UI_CONTROL = "ui-control"
MCP_SCOPE_EXPORT = "export"
MCP_SCOPE_LIFECYCLE = "lifecycle"


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


DATASET_INTAKE_TABLE_PAYLOAD_SCHEMA = _object_schema(
    {
        "format": {"type": "string", "enum": ["records", "csv"], "default": "records"},
        "records": {"type": "array", "items": {"type": "object", "additionalProperties": True}, "default": []},
        "csv": {"type": "string", "default": ""},
    }
)


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
    "neext_validate_dataset_intake": _object_schema(
        {
            "project_id": {"type": "string"},
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string", "default": ""},
            "tables": {"type": "object", "additionalProperties": DATASET_INTAKE_TABLE_PAYLOAD_SCHEMA},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "name", "tables"],
    ),
    "neext_create_dataset_intake_session": _object_schema(
        {
            "project_id": {"type": "string"},
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string", "default": ""},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "name"],
    ),
    "neext_append_dataset_intake_table": _object_schema(
        {
            "project_id": {"type": "string"},
            "session_id": {"type": "string"},
            "table_name": {"type": "string"},
            "table": DATASET_INTAKE_TABLE_PAYLOAD_SCHEMA,
            "replace": {"type": "boolean", "default": False},
        },
        ["project_id", "session_id", "table_name", "table"],
    ),
    "neext_validate_dataset_intake_session": _object_schema(
        {
            "project_id": {"type": "string"},
            "session_id": {"type": "string"},
        },
        ["project_id", "session_id"],
    ),
    "neext_create_dataset_from_intake": _object_schema(
        {
            "project_id": {"type": "string"},
            "session_id": {"type": "string"},
        },
        ["project_id", "session_id"],
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
    "neext_validate_custom_feature": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_dataset_id": {"type": "string"},
            "code": {"type": "string", "minLength": 1},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "source_dataset_id", "code"],
    ),
    "neext_configure_custom_feature": _object_schema(
        {
            "project_id": {"type": "string"},
            "source_dataset_id": {"type": "string"},
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string", "default": ""},
            "code": {"type": "string", "minLength": 1},
            "params": {"type": "object", "additionalProperties": True, "default": {}},
        },
        ["project_id", "source_dataset_id", "name", "code"],
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
    "neext_export_dataset_table": _object_schema(
        {
            "project_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "table": {"type": "string"},
        },
        ["project_id", "dataset_id", "table"],
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
    "neext_list_trash": _object_schema({}),
    "neext_restore_project": _object_schema({"trash_id": {"type": "string"}}, ["trash_id"]),
    "neext_restore_artifact_deletion": _object_schema(
        {"project_id": {"type": "string"}, "bundle_id": {"type": "string"}},
        ["project_id", "bundle_id"],
    ),
    "neext_plan_delete_artifact": _object_schema(
        {"project_id": {"type": "string"}, "kind": {"type": "string"}, "artifact_id": {"type": "string"}},
        ["project_id", "kind", "artifact_id"],
    ),
    "neext_request_delete_artifact": _object_schema(
        {
            "project_id": {"type": "string"},
            "kind": {"type": "string"},
            "artifact_id": {"type": "string"},
            "cascade": {"type": "boolean", "default": False},
        },
        ["project_id", "kind", "artifact_id"],
    ),
    "neext_request_delete_project": _object_schema({"project_id": {"type": "string"}}, ["project_id"]),
    "neext_list_mcp_activity": _object_schema({"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}}),
    "neext_list_mcp_approvals": _object_schema({}),
    "neext_get_workbench_view": _object_schema({}),
    "neext_set_workbench_view": _object_schema(
        {
            "route": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "top_tab": {"type": "string"},
                    "command": {"type": "string"},
                    "project_id": {"type": "string"},
                    "artifact_kind": {"type": "string"},
                    "artifact_id": {"type": "string"},
                    "catalog_kind": {"type": "string"},
                    "catalog_id": {"type": "string"},
                    "graph_id": {"type": "string"},
                    "node_id": {"type": "string"},
                    "draft": {"type": "object", "additionalProperties": True},
                },
            },
            "message": {"type": "string", "default": ""},
        },
        ["route"],
    ),
}


@dataclass(frozen=True)
class ToolCapability:
    scope: str
    read_only: bool
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False

    def annotations(self) -> dict[str, bool]:
        return {
            "readOnlyHint": self.read_only,
            "destructiveHint": self.destructive,
            "idempotentHint": self.idempotent,
            "openWorldHint": self.open_world,
        }


TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    "neext_workspace_summary": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_list_projects": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_create_project": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_list_catalog": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_list_artifacts": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_get_artifact": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_configure_dataset": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_validate_dataset_intake": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_create_dataset_intake_session": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_append_dataset_intake_table": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_validate_dataset_intake_session": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_create_dataset_from_intake": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_configure_feature": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_validate_custom_feature": ToolCapability(MCP_SCOPE_CUSTOM_CODE, read_only=False),
    "neext_configure_custom_feature": ToolCapability(MCP_SCOPE_CUSTOM_CODE, read_only=False),
    "neext_configure_embedding": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_configure_model": ToolCapability(MCP_SCOPE_WRITE, read_only=False),
    "neext_run_artifacts": ToolCapability(MCP_SCOPE_RUN, read_only=False),
    "neext_list_jobs": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_get_job": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_preview_artifact": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_export_dataset_table": ToolCapability(MCP_SCOPE_EXPORT, read_only=True, idempotent=True),
    "neext_analyze_artifact": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_search_graphs": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_get_graph_detail": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_list_trash": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=True, idempotent=True),
    "neext_restore_project": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=False),
    "neext_restore_artifact_deletion": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=False),
    "neext_plan_delete_artifact": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=True, idempotent=True),
    "neext_request_delete_artifact": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=False, destructive=True),
    "neext_request_delete_project": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=False, destructive=True),
    "neext_list_mcp_activity": ToolCapability(MCP_SCOPE_READ, read_only=True, idempotent=True),
    "neext_list_mcp_approvals": ToolCapability(MCP_SCOPE_LIFECYCLE, read_only=True, idempotent=True),
    "neext_get_workbench_view": ToolCapability(MCP_SCOPE_UI_CONTROL, read_only=True, idempotent=True),
    "neext_set_workbench_view": ToolCapability(MCP_SCOPE_UI_CONTROL, read_only=False, idempotent=True),
}


TOOL_DESCRIPTIONS = {
    "neext_workspace_summary": "Summarize the current NEExT Workbench workspace without exposing project paths.",
    "neext_list_projects": "List Workbench projects available in the current workspace.",
    "neext_create_project": "Create a new Workbench project in the current workspace.",
    "neext_list_catalog": "List source catalog entries for datasets, features, embeddings, or models.",
    "neext_list_artifacts": "List saved project artifacts of one kind.",
    "neext_get_artifact": "Read one saved project artifact manifest.",
    "neext_configure_dataset": "Add a Dataset Library catalog entry to the project as a Draft Dataset artifact.",
    "neext_validate_dataset_intake": "Validate NEExT Dataset Intake tables supplied as records or CSV text without creating an artifact.",
    "neext_create_dataset_intake_session": "Create a temporary Dataset Intake session for agent-supplied NEExT table data.",
    "neext_append_dataset_intake_table": "Append or replace one NEExT table in a Dataset Intake session using records or CSV text.",
    "neext_validate_dataset_intake_session": "Validate the current tables in a Dataset Intake session without creating an artifact.",
    "neext_create_dataset_from_intake": "Create a Draft Dataset artifact from a valid Dataset Intake session.",
    "neext_configure_feature": "Add a Feature Library entry to the project as a Draft Feature artifact for one Dataset artifact.",
    "neext_validate_custom_feature": "Validate trusted local Python custom feature code against a completed Dataset artifact.",
    "neext_configure_custom_feature": "Create a Draft custom Python Feature artifact after backend validation.",
    "neext_configure_embedding": "Add an Embedding Library entry to the project as a Draft Embedding artifact from one or more Feature artifacts.",
    "neext_configure_model": "Add a Model Library entry to the project as a Draft Model artifact from one or more Embedding artifacts.",
    "neext_run_artifacts": "Start Workbench jobs for Draft or failed artifacts and return their job manifests immediately.",
    "neext_list_jobs": "List local Workbench jobs for a project.",
    "neext_get_job": "Read one local Workbench job manifest.",
    "neext_preview_artifact": "Preview artifact outputs with pagination; never loads full output files by default.",
    "neext_export_dataset_table": "Export one approved Dataset table as CSV content using the existing Workbench export behavior.",
    "neext_analyze_artifact": "Run the existing Workbench analysis view for a completed artifact using bounded sampling options.",
    "neext_search_graphs": "Search graphs or nodes inside completed Dataset, Feature, or Embedding analysis data.",
    "neext_get_graph_detail": "Inspect one graph or dataset node from completed Dataset, Feature, or Embedding analysis data.",
    "neext_list_trash": "List Workbench trash entries and artifact deletion bundles.",
    "neext_restore_project": "Restore a trashed Workbench project when no live folder conflict exists.",
    "neext_restore_artifact_deletion": "Restore an artifact deletion bundle when no live artifact folder conflict exists.",
    "neext_plan_delete_artifact": "Plan artifact deletion and downstream cascade impact without deleting anything.",
    "neext_request_delete_artifact": "Request Workbench approval for an artifact delete or cascade delete; does not delete immediately.",
    "neext_request_delete_project": "Request Workbench approval for project deletion; does not delete immediately.",
    "neext_list_mcp_activity": "List recent MCP-originated Workbench activity.",
    "neext_list_mcp_approvals": "List Workbench-enforced MCP approval requests.",
    "neext_get_workbench_view": "Read the latest MCP-requested Workbench UI navigation state.",
    "neext_set_workbench_view": "Ask the open Workbench UI to navigate to a Space, Center View, artifact, graph, node, or workflow draft.",
}


PROMPTS = {
    "explore_neext_project": "Explore project {project_id}: list artifacts, inspect completed outputs, summarize lineage, and identify useful next analysis steps.",
    "configure_neext_pipeline": "Configure a NEExT pipeline in project {project_id}: choose catalog entries, create Draft artifacts, and stop before running jobs unless asked.",
    "run_neext_pipeline": "Run Draft or failed artifacts in project {project_id}, poll jobs until completion, and summarize outputs and errors.",
    "compare_neext_models": "Compare completed model artifacts in project {project_id}; use metrics previews and explain differences in model performance.",
    "investigate_neext_graph": "Investigate graph {graph_id} in project {project_id}; search related graph/node details and summarize structural observations.",
}


DOC_RESOURCES = {
    "neext://docs/workbench": {
        "name": "Workbench Operating Model",
        "description": "Current Workbench vocabulary, workflow boundaries, and MCP parity rules.",
        "text": (
            "NEExT Workbench is local, project-first, and dataset-first. Primary work belongs in Center Views. "
            "Current approved workflows include project create/select/delete/restore, curated Dataset Library configuration, "
            "Dataset preparation, built-in Feature configuration, trusted custom Python Feature creation, Embedding and Model "
            "configuration/execution, previews, analysis, graph search/detail, trash bundle restore, Settings Docs, and MCP setup. "
            "Deferred workflows must not be invented through MCP."
        ),
    },
    "neext://docs/network-science": {
        "name": "Network Science Workflow Guide",
        "description": "Concise network-science guidance for agents using NEExT.",
        "text": (
            "Start with graph provenance and dataset shape before interpreting metrics. Node features describe local or structural "
            "properties such as centrality; graph embeddings summarize graph-level structure for downstream supervised models. "
            "Use prepared Dataset artifacts as DAG roots, Feature artifacts as node-level measurements, Embedding artifacts as "
            "graph-level representations, and Model artifacts for supervised evaluation."
        ),
    },
}


def _normalized_scopes(scopes: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if scopes is None:
        return None
    return {str(scope).strip() for scope in scopes if str(scope).strip()}


def _require_scope(scopes: list[str] | tuple[str, ...] | set[str] | None, scope: str) -> None:
    allowed = _normalized_scopes(scopes)
    if allowed is not None and scope not in allowed:
        raise PermissionError(f'MCP tool requires "{scope}" scope')


def _tool_allowed(name: str, scopes: list[str] | tuple[str, ...] | set[str] | None) -> bool:
    capability = TOOL_CAPABILITIES[name]
    allowed = _normalized_scopes(scopes)
    return allowed is None or capability.scope in allowed


def list_tool_payloads(scopes: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "inputSchema": TOOL_SCHEMAS[name],
            "annotations": TOOL_CAPABILITIES[name].annotations(),
            "_meta": {"neextScope": TOOL_CAPABILITIES[name].scope},
        }
        for name in TOOL_SCHEMAS
        if _tool_allowed(name, scopes)
    ]


def call_tool_payload(
    service: WorkbenchMcpService,
    name: str,
    arguments: dict[str, Any] | None,
    scopes: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Any:
    handlers = _tool_handlers(service)
    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")
    _require_scope(scopes, TOOL_CAPABILITIES[name].scope)
    return handlers[name](arguments or {})


def capability_summary(scopes: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "scope": capability.scope,
            "read_only": capability.read_only,
            "destructive": capability.destructive,
            "idempotent": capability.idempotent,
            "available": _tool_allowed(name, scopes),
        }
        for name, capability in TOOL_CAPABILITIES.items()
    ]


def list_resource_payloads(service: WorkbenchMcpService) -> list[dict[str, Any]]:
    resources = [
        {
            "uri": "neext://workspace",
            "name": "Workspace",
            "description": "Current NEExT Workbench workspace summary.",
            "mimeType": "application/json",
        },
        {
            "uri": "neext://mcp/activity",
            "name": "MCP Activity",
            "description": "Recent MCP-originated Workbench activity.",
            "mimeType": "application/json",
        },
        {
            "uri": "neext://mcp/approvals",
            "name": "MCP Approvals",
            "description": "Workbench-enforced MCP approval requests.",
            "mimeType": "application/json",
        },
    ]
    for uri, doc in DOC_RESOURCES.items():
        resources.append(
            {
                "uri": uri,
                "name": doc["name"],
                "description": doc["description"],
                "mimeType": "text/plain",
            }
        )
    for project in service.store.list_projects():
        resources.append(
            {
                "uri": f"neext://projects/{project.id}",
                "name": f"Project: {project.name}",
                "description": "NEExT Workbench project manifest.",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"neext://projects/{project.id}/artifacts",
                "name": f"Artifacts: {project.name}",
                "description": "All project artifacts grouped by kind.",
                "mimeType": "application/json",
            }
        )
        for job in service.store.list_jobs(project.id):
            resources.append(
                {
                    "uri": f"neext://projects/{project.id}/jobs/{job.id}",
                    "name": f"Job: {job.id}",
                    "description": "NEExT Workbench job manifest.",
                    "mimeType": "application/json",
                }
            )
    return resources


def read_resource_text(service: WorkbenchMcpService, uri_text: str) -> tuple[str, str]:
    if uri_text == "neext://workspace":
        return "application/json", json.dumps(service.workspace_summary(), indent=2, sort_keys=True)
    if uri_text == "neext://mcp/activity":
        return "application/json", json.dumps(service.list_mcp_activity(), indent=2, sort_keys=True)
    if uri_text == "neext://mcp/approvals":
        return "application/json", json.dumps(service.list_mcp_approvals(), indent=2, sort_keys=True)
    if uri_text in DOC_RESOURCES:
        return "text/plain", DOC_RESOURCES[uri_text]["text"]
    parts = uri_text.removeprefix("neext://").split("/")
    if len(parts) == 2 and parts[0] == "projects":
        return "application/json", json.dumps(service._dump(service.store.read_project(parts[1])), indent=2, sort_keys=True)
    if len(parts) == 3 and parts[0] == "projects" and parts[2] == "artifacts":
        return "application/json", json.dumps(service.all_project_artifacts(parts[1]), indent=2, sort_keys=True)
    if len(parts) == 4 and parts[0] == "projects" and parts[2] == "jobs":
        return "application/json", json.dumps(service.get_job(parts[1], parts[3]), indent=2, sort_keys=True)
    raise ValueError(f"Unknown resource URI: {uri_text}")


def list_prompt_payloads() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": template,
            "arguments": [
                {"name": "project_id", "description": "Workbench project UUID.", "required": "project_id" in template},
                {"name": "graph_id", "description": "Prepared graph ID.", "required": "graph_id" in template},
            ],
        }
        for name, template in PROMPTS.items()
    ]


def get_prompt_payload(name: str, arguments: dict[str, str] | None) -> dict[str, Any]:
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    args = arguments or {}
    prompt_text = PROMPTS[name].format(
        project_id=args.get("project_id", "<project_id>"),
        graph_id=args.get("graph_id", "<graph_id>"),
    )
    return {
        "description": name,
        "messages": [
            {
                "role": "user",
                "content": {"type": "text", "text": prompt_text},
            }
        ],
    }


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
        "neext_validate_dataset_intake": lambda args: service.validate_dataset_intake(
            args["project_id"],
            args["name"],
            args["tables"],
            args.get("description", ""),
            args.get("params"),
        ),
        "neext_create_dataset_intake_session": lambda args: service.create_dataset_intake_session(
            args["project_id"],
            args["name"],
            args.get("description", ""),
            args.get("params"),
        ),
        "neext_append_dataset_intake_table": lambda args: service.append_dataset_intake_table(
            args["project_id"],
            args["session_id"],
            args["table_name"],
            args["table"],
            bool(args.get("replace", False)),
        ),
        "neext_validate_dataset_intake_session": lambda args: service.validate_dataset_intake_session(
            args["project_id"],
            args["session_id"],
        ),
        "neext_create_dataset_from_intake": lambda args: service.create_dataset_from_intake(
            args["project_id"],
            args["session_id"],
        ),
        "neext_configure_feature": lambda args: service.configure_feature(
            args["project_id"],
            args["source_dataset_id"],
            args["source_feature_id"],
            args.get("params"),
        ),
        "neext_validate_custom_feature": lambda args: service.validate_custom_feature(
            args["project_id"],
            args["source_dataset_id"],
            args["code"],
            args.get("params"),
        ),
        "neext_configure_custom_feature": lambda args: service.configure_custom_feature(
            args["project_id"],
            args["source_dataset_id"],
            args["name"],
            args.get("description", ""),
            args["code"],
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
        "neext_export_dataset_table": lambda args: service.export_dataset_table(args["project_id"], args["dataset_id"], args["table"]),
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
        "neext_list_trash": lambda args: service.list_trash(),
        "neext_restore_project": lambda args: service.restore_project(args["trash_id"]),
        "neext_restore_artifact_deletion": lambda args: service.restore_artifact_deletion(args["project_id"], args["bundle_id"]),
        "neext_plan_delete_artifact": lambda args: service.plan_delete_artifact(args["project_id"], args["kind"], args["artifact_id"]),
        "neext_request_delete_artifact": lambda args: service.request_delete_artifact(
            args["project_id"],
            args["kind"],
            args["artifact_id"],
            bool(args.get("cascade", False)),
        ),
        "neext_request_delete_project": lambda args: service.request_delete_project(args["project_id"]),
        "neext_list_mcp_activity": lambda args: service.list_mcp_activity(limit=int(args.get("limit", 50))),
        "neext_list_mcp_approvals": lambda args: service.list_mcp_approvals(),
        "neext_get_workbench_view": lambda args: service.get_workbench_view(),
        "neext_set_workbench_view": lambda args: service.set_workbench_view(args["route"], args.get("message", "")),
    }


def create_mcp_server(service: WorkbenchMcpService):
    """Create an MCP Server instance backed by a WorkbenchMcpService."""

    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server(MCP_SERVER_NAME)
    fixed_scopes = service.store.read_mcp_settings().scopes

    def active_scopes() -> list[str]:
        token = _sdk_access_token()
        if token is not None:
            return token.scopes
        return fixed_scopes

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
                annotations=types.ToolAnnotations(**tool["annotations"]),
                _meta=tool["_meta"],
            )
            for tool in list_tool_payloads(active_scopes())
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        payload = call_tool_payload(service, name, arguments, active_scopes())
        text = json.dumps(payload, indent=2, sort_keys=True)
        return [types.TextContent(type="text", text=text)]

    @server.list_resources()
    async def list_resources():
        _require_scope(active_scopes(), MCP_SCOPE_READ)
        return [
            types.Resource(
                uri=resource["uri"],
                name=resource["name"],
                description=resource["description"],
                mimeType=resource["mimeType"],
            )
            for resource in list_resource_payloads(service)
        ]

    @server.read_resource()
    async def read_resource(uri):
        from mcp.server.lowlevel.server import ReadResourceContents

        _require_scope(active_scopes(), MCP_SCOPE_READ)
        mime_type, text = read_resource_text(service, str(uri))
        return [ReadResourceContents(content=text, mime_type=mime_type)]

    @server.list_prompts()
    async def list_prompts():
        _require_scope(active_scopes(), MCP_SCOPE_READ)
        return [
            types.Prompt(
                name=prompt["name"],
                description=prompt["description"],
                arguments=[
                    types.PromptArgument(name=argument["name"], description=argument["description"], required=argument["required"])
                    for argument in prompt["arguments"]
                ],
            )
            for prompt in list_prompt_payloads()
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None):
        _require_scope(active_scopes(), MCP_SCOPE_READ)
        payload = get_prompt_payload(name, arguments)
        return types.GetPromptResult(
            description=payload["description"],
            messages=[
                types.PromptMessage(
                    role=message["role"],
                    content=types.TextContent(type="text", text=message["content"]["text"]),
                )
                for message in payload["messages"]
            ],
        )

    return server


def _sdk_access_token():
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except ImportError:
        return None
    return get_access_token()


def sdk_streamable_http_available() -> bool:
    try:
        from mcp.server.auth.middleware.auth_context import AuthContextMiddleware  # noqa: F401
        from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware  # noqa: F401
        from mcp.server.auth.provider import AccessToken, TokenVerifier  # noqa: F401
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: F401
    except Exception:
        return False
    return True


def require_sdk_streamable_http() -> None:
    if not sdk_streamable_http_available():
        raise RuntimeError("Install NEExT[workbench-mcp] with mcp>=1.27,<2 to enable SDK-backed Workbench MCP.")


def create_mcp_token_verifier(store: WorkbenchStore):
    require_sdk_streamable_http()
    from mcp.server.auth.provider import AccessToken, TokenVerifier

    class WorkbenchMcpTokenVerifier(TokenVerifier):
        async def verify_token(self, token: str) -> AccessToken | None:
            scopes = store.mcp_scopes_for_token(token)
            if scopes is None:
                return None
            return AccessToken(token=token, client_id="neext-workbench-local", scopes=scopes)

    return WorkbenchMcpTokenVerifier()


def create_streamable_http_session_manager(service: WorkbenchMcpService):
    require_sdk_streamable_http()
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    return StreamableHTTPSessionManager(
        create_mcp_server(service),
        json_response=True,
        stateless=True,
    )


MCP_CLI_MODULE = "NEExT.workbench.mcp_cli"

# macOS TCC-protected locations Claude Desktop's sandbox cannot read.
PROTECTED_DIR_NAMES = ("Desktop", "Documents", "Downloads")


def path_under_protected_dir(path) -> Optional[str]:
    """Return a human label for the macOS protected folder containing ``path``.

    Claude Desktop runs stdio servers sandboxed and cannot read ``~/Desktop``,
    ``~/Documents``, ``~/Downloads`` or iCloud Drive. Returns ``None`` on non-macOS
    platforms (Windows/Linux have no such sandbox) so they are never falsely blocked.
    """
    if sys.platform != "darwin":
        return None
    try:
        # Normalize ``..``/``.`` and make absolute WITHOUT following symlinks: a venv's
        # python is a symlink to a base interpreter outside the protected folder, but
        # its unreadable ``lib``/``site-packages`` (the actual TCC failure) live at the
        # literal venv path under ~/Desktop. Resolving symlinks would hide that.
        candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
        home = Path(os.path.abspath(os.path.expanduser("~")))
    except (OSError, RuntimeError, ValueError):
        return None
    protected: list[tuple[Path, str]] = [(home / name, name) for name in PROTECTED_DIR_NAMES]
    protected.append((home / "Library" / "Mobile Documents", "iCloud Drive"))
    for root, label in protected:
        if candidate == root or root in candidate.parents:
            return label
    return None


def stdio_launch_command(workspace_path) -> list[str]:
    """Build the local stdio launch command for the Workbench MCP server.

    Uses the absolute path to the interpreter already running Workbench plus
    ``-m NEExT.workbench.mcp_cli`` so the command always resolves to the exact
    environment that has NEExT + the MCP SDK installed. This avoids PATH/console-script
    fragility and works on macOS, Windows (``python.exe``), and Linux.
    """
    return [sys.executable, "-m", MCP_CLI_MODULE, "--workspace", str(workspace_path)]


def evaluate_stdio_readiness(workspace_path) -> McpStdioReadiness:
    """Validate that a local stdio MCP server can launch from this environment.

    Returns a blocked readiness with concrete remediation when the interpreter or
    MCP launcher is unavailable, or when the environment/workspace sits inside a
    macOS protected folder Claude Desktop cannot read.
    """
    command = stdio_launch_command(workspace_path)
    command_preview = " ".join(command)
    issues: list[str] = []
    remediation: list[str] = []

    interpreter = sys.executable
    if not interpreter or not Path(interpreter).exists():
        issues.append("The Python interpreter running Workbench could not be located.")
        remediation.append("Reinstall NEExT[workbench-mcp] in a standard virtual environment and relaunch Workbench.")

    if not sdk_streamable_http_available():
        issues.append("The MCP SDK is not importable in this environment.")
        remediation.append('Install the MCP extra: pip install -e ".[workbench-mcp]" (requires Python 3.10+).')

    if importlib.util.find_spec(MCP_CLI_MODULE) is None:
        issues.append("The Workbench MCP launcher module is not importable.")
        remediation.append("Reinstall NEExT so 'NEExT.workbench.mcp_cli' is importable on the Python path.")

    interpreter_protected = path_under_protected_dir(interpreter)
    if interpreter_protected:
        issues.append(f"The Python environment lives inside your macOS {interpreter_protected} folder, " "which Claude Desktop cannot read.")
        remediation.append(
            f"Recreate the virtual environment outside ~/{interpreter_protected} "
            "(for example ~/neext-dev/.venv) and reinstall NEExT[workbench-mcp]."
        )
        remediation.append(
            "Or grant Claude Full Disk Access under System Settings → Privacy & Security → " "Full Disk Access, then restart Claude Desktop."
        )

    workspace_protected = path_under_protected_dir(workspace_path)
    if workspace_protected and workspace_protected != interpreter_protected:
        issues.append(f"The workspace folder is inside your macOS {workspace_protected} folder, " "which Claude Desktop cannot read.")
        remediation.append(f"Move the Workbench workspace outside ~/{workspace_protected} " "(the default ~/NEExT-Workbench location is fine).")

    seen: set[str] = set()
    remediation = [item for item in remediation if not (item in seen or seen.add(item))]

    ok = not issues
    return McpStdioReadiness(
        status="ready" if ok else "blocked",
        ok=ok,
        interpreter=interpreter,
        command_preview=command_preview,
        issues=issues,
        remediation=remediation,
    )
