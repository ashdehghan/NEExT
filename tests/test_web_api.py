from pathlib import Path

import pytest


def test_web_api_generates_dataset_and_serializes_graph_summary(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from NEExT.web.app import create_app

    client = fastapi_testclient.TestClient(create_app(Path(tmp_path)))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post(
        "/api/datasets/generate",
        json={
            "name": "tiny ER vs BA",
            "preset": "er_vs_ba",
            "seed": 7,
            "params": {"n_per_class": 1, "n_nodes": 8},
        },
    )
    assert created.status_code == 200
    artifact = created.json()["artifact"]
    assert artifact["type"] == "dataset"

    summary = client.get(f"/api/graphs/{artifact['id']}")
    assert summary.status_code == 200
    assert summary.json()["num_graphs"] == 2

    graph_id = summary.json()["graphs"][0]["graph_id"]
    elements = client.get(f"/api/graphs/{artifact['id']}/{graph_id}/elements")
    assert elements.status_code == 200
    assert "nodes" in elements.json()["elements"]
