"""Serializers for web-safe NEExT object summaries."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def dataframe_payload(df: pd.DataFrame, limit: int = 200, offset: int = 0) -> dict[str, Any]:
    safe_df = df.replace({float("inf"): None, float("-inf"): None}).where(pd.notnull(df), None)
    page = safe_df.iloc[offset : offset + limit]
    return {
        "columns": list(safe_df.columns),
        "rows": page.to_dict(orient="records"),
        "total_rows": int(len(safe_df)),
        "offset": int(offset),
        "limit": int(limit),
    }


def collection_summary(collection: Any) -> dict[str, Any]:
    graphs = []
    for graph in collection.graphs:
        info = graph.get_graph_info()
        info["sampled_nodes"] = len(graph.sampled_nodes or graph.nodes)
        graphs.append(info)
    summary = collection.describe()
    summary["total_nodes"] = collection.get_total_node_count()
    summary["total_edges"] = sum(len(graph.edges) for graph in collection.graphs)
    summary["graphs"] = graphs
    return summary


def graph_elements(graph: Any, max_nodes: int = 500) -> dict[str, Any]:
    nodes = list(graph.nodes)
    truncated = False
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        truncated = True
    node_set = set(nodes)

    node_payloads = []
    for node in nodes:
        attrs = dict(graph.node_attributes.get(node, {}))
        attrs.update({"id": str(node), "label": str(node)})
        node_payloads.append({"data": _safe_dict(attrs)})

    edge_payloads = []
    for index, (source, target) in enumerate(graph.edges):
        if source not in node_set or target not in node_set:
            continue
        attrs = dict(graph.edge_attributes.get((source, target), {}))
        attrs.update(
            {
                "id": f"e{index}",
                "source": str(source),
                "target": str(target),
            }
        )
        edge_payloads.append({"data": _safe_dict(attrs)})

    return {
        "graph": graph.get_graph_info(),
        "truncated": truncated,
        "max_nodes": max_nodes,
        "elements": {"nodes": node_payloads, "edges": edge_payloads},
    }


def model_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in result.items():
        if key == "model":
            payload["model_class"] = value.__class__.__name__
        elif hasattr(value, "tolist"):
            payload[key] = value.tolist()
        else:
            payload[key] = _safe_value(value)
    return payload


def _safe_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_value(value) for key, value in values.items()}


def _safe_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return _safe_dict(value)
    return str(value)
