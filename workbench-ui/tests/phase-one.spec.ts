import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const TITLE_ONLY_COMMANDS = ["Import", "Create"] as const;

async function clearProjects(page: Page) {
  const response = await page.request.get("/api/projects");
  const projects = (await response.json()) as { id: string }[];
  for (const project of projects) {
    await page.request.delete(`/api/projects/${project.id}`);
  }
}

async function createProject(page: Page, name: string, description: string) {
  const ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Create" }).click();
  await page.getByRole("textbox", { name: "Name" }).fill(name);
  await page.getByRole("textbox", { name: "Description" }).fill(description);
  await page.locator("#form-create-project").getByRole("button", { name: "Create" }).click();
}

function seedTinyEmbeddingProject(name: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const workspacePath = path.join(repoRoot, "sandbox", "workbench-e2e");
  const script = String.raw`
import json
import sys
from pathlib import Path

import pandas as pd

from NEExT.workbench.schemas import (
    DatasetDataFiles,
    DatasetManifest,
    DatasetStats,
    FeatureCreateParams,
    FeatureCreateRequest,
    OperationSpec,
    ProjectCreate,
)
from NEExT.workbench.storage import WorkbenchStore, utc_now

workspace_path = Path(sys.argv[1])
project_name = sys.argv[2]
store = WorkbenchStore(workspace_path)
project = store.create_project(ProjectCreate(name=project_name, description="tiny embedding project"))

dataset_id = store._new_dataset_id(project.id)
dataset_path = store.dataset_path(project.id, dataset_id)
(dataset_path / "prepared").mkdir(parents=True, exist_ok=False)

nodes = pd.DataFrame(
    [
        {"graph_id": "g1", "node_id": 0},
        {"graph_id": "g1", "node_id": 1},
        {"graph_id": "g1", "node_id": 2},
        {"graph_id": "g2", "node_id": 0},
        {"graph_id": "g2", "node_id": 1},
        {"graph_id": "g2", "node_id": 2},
    ]
)
edges = pd.DataFrame(
    [
        {"graph_id": "g1", "src_node_id": 0, "dest_node_id": 1},
        {"graph_id": "g1", "src_node_id": 1, "dest_node_id": 2},
        {"graph_id": "g2", "src_node_id": 0, "dest_node_id": 1},
        {"graph_id": "g2", "src_node_id": 1, "dest_node_id": 2},
    ]
)
graph_labels = pd.DataFrame([{"graph_id": "g1", "graph_label": 0}, {"graph_id": "g2", "graph_label": 1}])
nodes.to_parquet(dataset_path / "prepared" / "nodes.parquet", index=False)
edges.to_parquet(dataset_path / "prepared" / "edges.parquet", index=False)
graph_labels.to_parquet(dataset_path / "prepared" / "graph_labels.parquet", index=False)

now = utc_now()
stats = DatasetStats(
    graph_count=2,
    node_count=6,
    edge_count=4,
    has_graph_labels=True,
    has_node_features=False,
    has_edge_features=False,
)
prepared_files = DatasetDataFiles(
    nodes="prepared/nodes.parquet",
    edges="prepared/edges.parquet",
    graph_labels="prepared/graph_labels.parquet",
)
dataset = DatasetManifest(
    id=dataset_id,
    project_id=project.id,
    name="Tiny Embedding",
    description="tiny prepared graph collection",
    status="completed",
    created_at=now,
    updated_at=now,
    source_catalog_id="TINY_E2E",
    source_name="Tiny Embedding",
    source="E2E fixture",
    source_domain="Tests",
    operation=OperationSpec(
        operation_id="neext.prepare_graph_collection",
        operation_version="1",
        params={"graph_type": "networkx", "reindex_nodes": True, "filter_largest_component": False},
    ),
    source_stats=stats,
    prepared_stats=stats,
    prepared_data_files=prepared_files,
    data_files=prepared_files,
    stats=stats,
)
store._write_json(dataset_path / "artifact.json", dataset.model_dump())

params = FeatureCreateParams(feature_vector_length=2, normalize_features=False, n_jobs=1, parallel_backend="threading")
page_rank = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="page_rank", params=params))
degree = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="degree_centrality", params=params))

print(json.dumps({"project_id": project.id, "dataset_id": dataset_id, "feature_ids": [page_rank.id, degree.id]}))
`;
  return JSON.parse(execFileSync("python", ["-c", script, workspacePath, name], { cwd: repoRoot, encoding: "utf-8" })) as {
    project_id: string;
    dataset_id: string;
    feature_ids: string[];
  };
}

function seedTinyFeatureExploreProject(name: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const workspacePath = path.join(repoRoot, "sandbox", "workbench-e2e");
  const script = String.raw`
import json
import sys
from pathlib import Path

import pandas as pd

from NEExT.workbench.schemas import (
    DatasetDataFiles,
    DatasetManifest,
    DatasetMappingFiles,
    DatasetStats,
    FeatureCreateParams,
    FeatureCreateRequest,
    FeatureOutputFiles,
    FeatureOutputStats,
    OperationSpec,
    ProjectCreate,
)
from NEExT.workbench.storage import WorkbenchStore, utc_now

workspace_path = Path(sys.argv[1])
project_name = sys.argv[2]
store = WorkbenchStore(workspace_path)
project = store.create_project(ProjectCreate(name=project_name, description="feature explore project"))

dataset_id = store._new_dataset_id(project.id)
dataset_path = store.dataset_path(project.id, dataset_id)
(dataset_path / "prepared").mkdir(parents=True, exist_ok=False)
(dataset_path / "mappings").mkdir(parents=True, exist_ok=False)

nodes = pd.DataFrame(
    [
        {"graph_id": "g1", "node_id": 0},
        {"graph_id": "g1", "node_id": 1},
        {"graph_id": "g1", "node_id": 2},
        {"graph_id": "g2", "node_id": 0},
        {"graph_id": "g2", "node_id": 1},
        {"graph_id": "g2", "node_id": 2},
    ]
)
edges = pd.DataFrame(
    [
        {"graph_id": "g1", "src_node_id": 0, "dest_node_id": 1},
        {"graph_id": "g1", "src_node_id": 1, "dest_node_id": 2},
        {"graph_id": "g2", "src_node_id": 0, "dest_node_id": 1},
        {"graph_id": "g2", "src_node_id": 1, "dest_node_id": 2},
    ]
)
graph_labels = pd.DataFrame([{"graph_id": "g1", "graph_label": 0}, {"graph_id": "g2", "graph_label": 1}])
mapping = pd.DataFrame(
    [
        {
            "source_graph_id": row["graph_id"],
            "source_node_id": f"{row['graph_id']}-source-{row['node_id']}",
            "internal_graph_id": row["graph_id"],
            "internal_node_id": row["node_id"],
            "included": True,
            "drop_reason": None,
        }
        for row in nodes.to_dict(orient="records")
    ]
)
nodes.to_parquet(dataset_path / "prepared" / "nodes.parquet", index=False)
edges.to_parquet(dataset_path / "prepared" / "edges.parquet", index=False)
graph_labels.to_parquet(dataset_path / "prepared" / "graph_labels.parquet", index=False)
mapping.to_parquet(dataset_path / "mappings" / "node_mapping.parquet", index=False)
pd.DataFrame([{"source_graph_id": "g1", "internal_graph_id": "g1"}, {"source_graph_id": "g2", "internal_graph_id": "g2"}]).to_parquet(
    dataset_path / "mappings" / "graph_mapping.parquet",
    index=False,
)

now = utc_now()
stats = DatasetStats(
    graph_count=2,
    node_count=6,
    edge_count=4,
    has_graph_labels=True,
    has_node_features=False,
    has_edge_features=False,
)
prepared_files = DatasetDataFiles(
    nodes="prepared/nodes.parquet",
    edges="prepared/edges.parquet",
    graph_labels="prepared/graph_labels.parquet",
)
mapping_files = DatasetMappingFiles(node_mapping="mappings/node_mapping.parquet", graph_mapping="mappings/graph_mapping.parquet")
dataset = DatasetManifest(
    id=dataset_id,
    project_id=project.id,
    name="Tiny Feature Explore",
    description="tiny feature explore graph collection",
    status="completed",
    created_at=now,
    updated_at=now,
    source_catalog_id="TINY_FEATURE_EXPLORE",
    source_name="Tiny Feature Explore",
    source="E2E fixture",
    source_domain="Tests",
    operation=OperationSpec(
        operation_id="neext.prepare_graph_collection",
        operation_version="1",
        params={"graph_type": "networkx", "reindex_nodes": True, "filter_largest_component": False},
    ),
    source_stats=stats,
    prepared_stats=stats,
    prepared_data_files=prepared_files,
    mapping_files=mapping_files,
    data_files=prepared_files,
    stats=stats,
)
store._write_json(dataset_path / "artifact.json", dataset.model_dump())

params = FeatureCreateParams(feature_vector_length=2, normalize_features=False, n_jobs=1, parallel_backend="threading")
feature = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="page_rank", params=params))
feature_path = store.feature_path(project.id, feature.id)
(feature_path / "output").mkdir(parents=True, exist_ok=False)
features = pd.DataFrame(
    [
        {"node_id": 0, "graph_id": "g1", "page_rank_0": -1.0, "page_rank_1": 0.0},
        {"node_id": 1, "graph_id": "g1", "page_rank_0": 0.0, "page_rank_1": 0.0},
        {"node_id": 2, "graph_id": "g1", "page_rank_0": 1.0, "page_rank_1": 0.0},
        {"node_id": 0, "graph_id": "g2", "page_rank_0": 0.0, "page_rank_1": -1.0},
        {"node_id": 1, "graph_id": "g2", "page_rank_0": 0.0, "page_rank_1": 0.0},
        {"node_id": 2, "graph_id": "g2", "page_rank_0": 0.0, "page_rank_1": 1.0},
    ]
)
features.to_parquet(feature_path / "output" / "features.parquet", index=False)
feature.status = "completed"
feature.output_files = FeatureOutputFiles(features="output/features.parquet")
feature.output_stats = FeatureOutputStats(row_count=6, column_count=4)
feature.updated_at = utc_now()
store._write_json(feature_path / "artifact.json", feature.model_dump())

one_column = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="page_rank", params=params))
one_column_path = store.feature_path(project.id, one_column.id)
(one_column_path / "output").mkdir(parents=True, exist_ok=False)
one_column_features = pd.DataFrame(
    [
        {"node_id": 0, "graph_id": "g1", "single_0": -1.0},
        {"node_id": 1, "graph_id": "g1", "single_0": 0.0},
        {"node_id": 2, "graph_id": "g1", "single_0": 1.0},
        {"node_id": 0, "graph_id": "g2", "single_0": -2.0},
        {"node_id": 1, "graph_id": "g2", "single_0": 0.0},
        {"node_id": 2, "graph_id": "g2", "single_0": 2.0},
    ]
)
one_column_features.to_parquet(one_column_path / "output" / "features.parquet", index=False)
one_column.name = "Tiny Feature Explore - One Column"
one_column.status = "completed"
one_column.output_files = FeatureOutputFiles(features="output/features.parquet")
one_column.output_stats = FeatureOutputStats(row_count=6, column_count=3)
one_column.updated_at = utc_now()
store._write_json(one_column_path / "artifact.json", one_column.model_dump())

print(json.dumps({"project_id": project.id, "dataset_id": dataset_id, "feature_ids": [feature.id, one_column.id]}))
`;
  return JSON.parse(execFileSync("python", ["-c", script, workspacePath, name], { cwd: repoRoot, encoding: "utf-8" })) as {
    project_id: string;
    dataset_id: string;
    feature_ids: string[];
  };
}

function seedTinyEmbeddingExploreProject(name: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const workspacePath = path.join(repoRoot, "sandbox", "workbench-e2e");
  const script = String.raw`
import json
import sys
from pathlib import Path

import pandas as pd

from NEExT.workbench.schemas import (
    DatasetDataFiles,
    DatasetManifest,
    DatasetStats,
    EmbeddingCreateParams,
    EmbeddingCreateRequest,
    EmbeddingOutputFiles,
    EmbeddingOutputStats,
    FeatureCreateParams,
    FeatureCreateRequest,
    FeatureOutputFiles,
    FeatureOutputStats,
    OperationSpec,
    ProjectCreate,
)
from NEExT.workbench.storage import WorkbenchStore, utc_now

workspace_path = Path(sys.argv[1])
project_name = sys.argv[2]
store = WorkbenchStore(workspace_path)
project = store.create_project(ProjectCreate(name=project_name, description="embedding explore project"))

dataset_id = store._new_dataset_id(project.id)
dataset_path = store.dataset_path(project.id, dataset_id)
(dataset_path / "prepared").mkdir(parents=True, exist_ok=False)

nodes = pd.DataFrame(
    [
        {"graph_id": "g1", "node_id": 0},
        {"graph_id": "g1", "node_id": 1},
        {"graph_id": "g2", "node_id": 0},
        {"graph_id": "g2", "node_id": 1},
    ]
)
edges = pd.DataFrame(
    [
        {"graph_id": "g1", "src_node_id": 0, "dest_node_id": 1},
        {"graph_id": "g2", "src_node_id": 0, "dest_node_id": 1},
    ]
)
graph_labels = pd.DataFrame([{"graph_id": "g1", "graph_label": 0}, {"graph_id": "g2", "graph_label": 1}])
nodes.to_parquet(dataset_path / "prepared" / "nodes.parquet", index=False)
edges.to_parquet(dataset_path / "prepared" / "edges.parquet", index=False)
graph_labels.to_parquet(dataset_path / "prepared" / "graph_labels.parquet", index=False)

now = utc_now()
stats = DatasetStats(
    graph_count=2,
    node_count=4,
    edge_count=2,
    has_graph_labels=True,
    has_node_features=False,
    has_edge_features=False,
)
prepared_files = DatasetDataFiles(
    nodes="prepared/nodes.parquet",
    edges="prepared/edges.parquet",
    graph_labels="prepared/graph_labels.parquet",
)
dataset = DatasetManifest(
    id=dataset_id,
    project_id=project.id,
    name="Tiny Embedding Explore",
    description="tiny embedding explore graph collection",
    status="completed",
    created_at=now,
    updated_at=now,
    source_catalog_id="TINY_EMBEDDING_EXPLORE",
    source_name="Tiny Embedding Explore",
    source="E2E fixture",
    source_domain="Tests",
    operation=OperationSpec(
        operation_id="neext.prepare_graph_collection",
        operation_version="1",
        params={"graph_type": "networkx", "reindex_nodes": True, "filter_largest_component": False},
    ),
    source_stats=stats,
    prepared_stats=stats,
    prepared_data_files=prepared_files,
    data_files=prepared_files,
    stats=stats,
)
store._write_json(dataset_path / "artifact.json", dataset.model_dump())

feature_params = FeatureCreateParams(feature_vector_length=2, normalize_features=False, n_jobs=1, parallel_backend="threading")
feature = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="page_rank", params=feature_params))
feature_path = store.feature_path(project.id, feature.id)
(feature_path / "output").mkdir(parents=True, exist_ok=False)
features = pd.DataFrame(
    [
        {"node_id": 0, "graph_id": "g1", "page_rank_0": 0.1, "page_rank_1": 0.2},
        {"node_id": 1, "graph_id": "g1", "page_rank_0": 0.2, "page_rank_1": 0.3},
        {"node_id": 0, "graph_id": "g2", "page_rank_0": 0.3, "page_rank_1": 0.4},
        {"node_id": 1, "graph_id": "g2", "page_rank_0": 0.4, "page_rank_1": 0.5},
    ]
)
features.to_parquet(feature_path / "output" / "features.parquet", index=False)
feature.status = "completed"
feature.output_files = FeatureOutputFiles(features="output/features.parquet")
feature.output_stats = FeatureOutputStats(row_count=4, column_count=4)
feature.updated_at = utc_now()
store._write_json(feature_path / "artifact.json", feature.model_dump())

def complete_embedding(label, dimension, rows):
    embedding = store.create_embedding(
        project.id,
        EmbeddingCreateRequest(
            source_embedding_id="approx_wasserstein",
            source_feature_ids=[feature.id],
            params=EmbeddingCreateParams(embedding_dimension=dimension),
        ),
    )
    embedding_path = store.embedding_path(project.id, embedding.id)
    (embedding_path / "output").mkdir(parents=True, exist_ok=False)
    output = pd.DataFrame(rows)
    output.to_parquet(embedding_path / "output" / "embeddings.parquet", index=False)
    embedding.name = f"Tiny Embedding Explore - {label}"
    embedding.status = "completed"
    embedding.output_files = EmbeddingOutputFiles(embeddings="output/embeddings.parquet")
    embedding.output_stats = EmbeddingOutputStats(row_count=len(output), column_count=len(output.columns))
    embedding.updated_at = utc_now()
    store._write_json(embedding_path / "artifact.json", embedding.model_dump())
    return embedding.id

two_dim_id = complete_embedding(
    "Two Dimensions",
    2,
    [{"graph_id": "g1", "emb_0": 1.0, "emb_1": 2.0}, {"graph_id": "g2", "emb_0": 3.0, "emb_1": 4.0}],
)
one_dim_id = complete_embedding(
    "One Dimension",
    1,
    [{"graph_id": "g1", "emb_0": 1.0}, {"graph_id": "g2", "emb_0": 2.0}],
)

print(json.dumps({"project_id": project.id, "dataset_id": dataset_id, "embedding_ids": [two_dim_id, one_dim_id]}))
`;
  return JSON.parse(execFileSync("python", ["-c", script, workspacePath, name], { cwd: repoRoot, encoding: "utf-8" })) as {
    project_id: string;
    dataset_id: string;
    embedding_ids: string[];
  };
}

function seedTinyModelProject(name: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const workspacePath = path.join(repoRoot, "sandbox", "workbench-e2e");
  const script = String.raw`
import json
import sys
from pathlib import Path

import pandas as pd

from NEExT.workbench.schemas import (
    DatasetDataFiles,
    DatasetManifest,
    DatasetStats,
    EmbeddingCreateRequest,
    EmbeddingCreateParams,
    FeatureCreateParams,
    FeatureCreateRequest,
    ModelCreateParams,
    ModelCreateRequest,
    OperationSpec,
    ProjectCreate,
)
from NEExT.workbench.storage import WorkbenchStore, utc_now

workspace_path = Path(sys.argv[1])
project_name = sys.argv[2]
store = WorkbenchStore(workspace_path)
project = store.create_project(ProjectCreate(name=project_name, description="tiny model project"))

dataset_id = store._new_dataset_id(project.id)
dataset_path = store.dataset_path(project.id, dataset_id)
(dataset_path / "prepared").mkdir(parents=True, exist_ok=False)

node_records = []
edge_records = []
label_records = []
for graph_index in range(8):
    graph_id = f"g{graph_index}"
    label = graph_index % 2
    for node_id in range(3):
        node_records.append({"graph_id": graph_id, "node_id": node_id})
    edge_records.extend(
        [
            {"graph_id": graph_id, "src_node_id": 0, "dest_node_id": 1},
            {"graph_id": graph_id, "src_node_id": 1, "dest_node_id": 2},
        ]
    )
    label_records.append({"graph_id": graph_id, "graph_label": label})

nodes = pd.DataFrame(node_records)
edges = pd.DataFrame(edge_records)
graph_labels = pd.DataFrame(label_records)
nodes.to_parquet(dataset_path / "prepared" / "nodes.parquet", index=False)
edges.to_parquet(dataset_path / "prepared" / "edges.parquet", index=False)
graph_labels.to_parquet(dataset_path / "prepared" / "graph_labels.parquet", index=False)

now = utc_now()
stats = DatasetStats(
    graph_count=8,
    node_count=24,
    edge_count=16,
    has_graph_labels=True,
    has_node_features=False,
    has_edge_features=False,
)
prepared_files = DatasetDataFiles(
    nodes="prepared/nodes.parquet",
    edges="prepared/edges.parquet",
    graph_labels="prepared/graph_labels.parquet",
)
dataset = DatasetManifest(
    id=dataset_id,
    project_id=project.id,
    name="Tiny Model",
    description="tiny labeled graph collection",
    status="completed",
    created_at=now,
    updated_at=now,
    source_catalog_id="TINY_MODEL_E2E",
    source_name="Tiny Model",
    source="E2E fixture",
    source_domain="Tests",
    operation=OperationSpec(
        operation_id="neext.prepare_graph_collection",
        operation_version="1",
        params={"graph_type": "networkx", "reindex_nodes": True, "filter_largest_component": False},
    ),
    source_stats=stats,
    prepared_stats=stats,
    prepared_data_files=prepared_files,
    data_files=prepared_files,
    stats=stats,
)
store._write_json(dataset_path / "artifact.json", dataset.model_dump())

feature_params = FeatureCreateParams(feature_vector_length=2, normalize_features=False, n_jobs=1, parallel_backend="threading")
page_rank = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="page_rank", params=feature_params))
degree = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=dataset_id, source_feature_id="degree_centrality", params=feature_params))

embedding_params = EmbeddingCreateParams(embedding_dimension=1)
first_embedding = store.create_embedding(
    project.id,
    EmbeddingCreateRequest(source_embedding_id="approx_wasserstein", source_feature_ids=[page_rank.id], params=embedding_params),
)
second_embedding = store.create_embedding(
    project.id,
    EmbeddingCreateRequest(source_embedding_id="approx_wasserstein", source_feature_ids=[degree.id], params=embedding_params),
)

print(
    json.dumps(
        {
            "project_id": project.id,
            "dataset_id": dataset_id,
            "feature_ids": [page_rank.id, degree.id],
            "embedding_ids": [first_embedding.id, second_embedding.id],
        }
    )
)
`;
  return JSON.parse(execFileSync("python", ["-c", script, workspacePath, name], { cwd: repoRoot, encoding: "utf-8" })) as {
    project_id: string;
    dataset_id: string;
    feature_ids: string[];
    embedding_ids: string[];
  };
}

function seedLineageSelectionProject(name: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const workspacePath = path.join(repoRoot, "sandbox", "workbench-e2e");
  const script = String.raw`
import json
import sys
from pathlib import Path

from NEExT.workbench.schemas import (
    DatasetManifest,
    DatasetStats,
    EmbeddingCreateParams,
    EmbeddingCreateRequest,
    FeatureCreateParams,
    FeatureCreateRequest,
    ModelCreateParams,
    ModelCreateRequest,
    OperationSpec,
    ProjectCreate,
)
from NEExT.workbench.storage import WorkbenchStore, utc_now

workspace_path = Path(sys.argv[1])
project_name = sys.argv[2]
store = WorkbenchStore(workspace_path)
project = store.create_project(ProjectCreate(name=project_name, description="lineage selection project"))

def create_dataset(label):
    dataset_id = store._new_dataset_id(project.id)
    dataset_path = store.dataset_path(project.id, dataset_id)
    dataset_path.mkdir(parents=True, exist_ok=False)
    now = utc_now()
    stats = DatasetStats(
        graph_count=2,
        node_count=4,
        edge_count=2,
        has_graph_labels=True,
        has_node_features=False,
        has_edge_features=False,
    )
    dataset = DatasetManifest(
        id=dataset_id,
        project_id=project.id,
        name=label,
        description=f"{label} lineage fixture",
        status="completed",
        created_at=now,
        updated_at=now,
        source_catalog_id=label.upper().replace(" ", "_"),
        source_name=label,
        source="E2E fixture",
        source_domain="Tests",
        operation=OperationSpec(
            operation_id="neext.prepare_graph_collection",
            operation_version="1",
            params={"graph_type": "networkx", "reindex_nodes": True, "filter_largest_component": False},
        ),
        source_stats=stats,
        prepared_stats=stats,
        stats=stats,
    )
    store._write_json(dataset_path / "artifact.json", dataset.model_dump())
    return dataset

alpha = create_dataset("Lineage Alpha")
beta = create_dataset("Lineage Beta")
feature_params = FeatureCreateParams(feature_vector_length=2, normalize_features=False, n_jobs=1, parallel_backend="threading")
alpha_page = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=alpha.id, source_feature_id="page_rank", params=feature_params))
alpha_degree = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=alpha.id, source_feature_id="degree_centrality", params=feature_params))
beta_page = store.create_feature(project.id, FeatureCreateRequest(source_dataset_id=beta.id, source_feature_id="page_rank", params=feature_params))

embedding_params = EmbeddingCreateParams(embedding_dimension=2)
alpha_page_embedding = store.create_embedding(
    project.id,
    EmbeddingCreateRequest(source_embedding_id="approx_wasserstein", source_feature_ids=[alpha_page.id], params=embedding_params),
)
alpha_combined_embedding = store.create_embedding(
    project.id,
    EmbeddingCreateRequest(source_embedding_id="approx_wasserstein", source_feature_ids=[alpha_page.id, alpha_degree.id], params=embedding_params),
)
beta_embedding = store.create_embedding(
    project.id,
    EmbeddingCreateRequest(source_embedding_id="approx_wasserstein", source_feature_ids=[beta_page.id], params=embedding_params),
)

for embedding, label in [
    (alpha_page_embedding, "Alpha Page Embedding"),
    (alpha_combined_embedding, "Alpha Combined Embedding"),
    (beta_embedding, "Beta Page Embedding"),
]:
    embedding.name = label
    embedding.updated_at = utc_now()
    store._write_json(store.embedding_path(project.id, embedding.id) / "artifact.json", embedding.model_dump())

model_params = ModelCreateParams(
    task_type="classifier",
    sample_size=1,
    test_size=0.5,
    balance_dataset=False,
    n_jobs=1,
    parallel_backend="thread",
)
alpha_model = store.create_model(
    project.id,
    ModelCreateRequest(source_model_id="random_forest", source_embedding_ids=[alpha_combined_embedding.id], params=model_params),
)
beta_model = store.create_model(
    project.id,
    ModelCreateRequest(source_model_id="random_forest", source_embedding_ids=[beta_embedding.id], params=model_params),
)
alpha_model.name = "Alpha Classifier"
alpha_model.updated_at = utc_now()
store._write_json(store.model_path(project.id, alpha_model.id) / "artifact.json", alpha_model.model_dump())
beta_model.name = "Beta Classifier"
beta_model.updated_at = utc_now()
store._write_json(store.model_path(project.id, beta_model.id) / "artifact.json", beta_model.model_dump())

print(
    json.dumps(
        {
            "project_id": project.id,
            "datasets": {"alpha": alpha.id, "beta": beta.id},
            "features": {"alpha_page": alpha_page.id, "alpha_degree": alpha_degree.id, "beta_page": beta_page.id},
            "embeddings": {
                "alpha_page": alpha_page_embedding.id,
                "alpha_combined": alpha_combined_embedding.id,
                "beta": beta_embedding.id,
            },
            "models": {"alpha": alpha_model.id, "beta": beta_model.id},
        }
    )
)
`;
  return JSON.parse(execFileSync("python", ["-c", script, workspacePath, name], { cwd: repoRoot, encoding: "utf-8" })) as {
    project_id: string;
    datasets: Record<string, string>;
    features: Record<string, string>;
    embeddings: Record<string, string>;
    models: Record<string, string>;
  };
}

test.beforeEach(async ({ page }) => {
  await clearProjects(page);
});

test("loads the packaged Workbench shell and creates a project", async ({ page }) => {
  await page.goto("/");

  const ribbon = page.locator(".ribbon");
  await expect(page.getByRole("button", { name: "HOME", exact: true })).toHaveClass(/is-active/);
  await expect(ribbon.getByRole("button", { name: "Import" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Create" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Projects" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Trash" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Settings" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Help" })).toHaveCount(0);

  await expect(page.locator(".selection-panel")).toContainText("Project");
  await expect(page.locator(".selection-panel")).toContainText("Datasets");
  await expect(page.locator(".selection-panel")).toContainText("Features");
  await expect(page.locator(".selection-panel")).toContainText("Embeddings");
  await expect(page.locator(".selection-panel")).toContainText("Models");
  await expect(page.locator(".inspector-panel")).toContainText("Inspector");
  await expect(page.locator(".jobs-panel")).toContainText("Jobs");
  await expect(page.locator(".cmd")).toContainText("Command Window");

  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name: "Create Project" })).toBeVisible();

  const projectName = `Phase One Project ${Date.now()}`;
  await page.getByRole("textbox", { name: "Name" }).fill(projectName);
  await page.getByRole("textbox", { name: "Description" }).fill("project-backed create");
  await page.locator("#form-create-project").getByRole("button", { name: "Create" }).click();

  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Projects" })).toHaveClass(/is-active/);
  const createdRow = page.locator("table tbody tr", { hasText: projectName });
  const inspector = page.locator(".inspector-panel");
  await expect(createdRow).toBeVisible();
  await expect(createdRow).toHaveClass(/is-selected/);
  await expect(createdRow.getByText("active")).toBeVisible();
  await expect(inspector).toContainText("Project Details");
  await expect(inspector).toContainText(projectName);
  await expect(inspector).toContainText("project-backed create");
  await expect(inspector).toContainText("Active project");
  await expect(inspector).toContainText("Project ID");
  await expect(inspector).toContainText(/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/);
  await expect(inspector).toContainText("Created");
  await expect(inspector).toContainText("Updated");

  const secondProjectName = `Second Project ${Date.now()}`;
  await ribbon.getByRole("button", { name: "Create" }).click();
  await page.getByRole("textbox", { name: "Name" }).fill(secondProjectName);
  await page.getByRole("textbox", { name: "Description" }).fill("second project details");
  await page.locator("#form-create-project").getByRole("button", { name: "Create" }).click();

  const secondRow = page.locator("table tbody tr", { hasText: secondProjectName });
  await expect(secondRow).toBeVisible();
  await expect(secondRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText(secondProjectName);
  await expect(inspector).toContainText("second project details");

  await createdRow.click();
  await expect(createdRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText(projectName);
  await expect(inspector).toContainText("project-backed create");
  await expect(inspector).not.toContainText(secondProjectName);
});

test("Projects table deletes projects through confirmation", async ({ page }) => {
  await page.goto("/");

  const ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Create" }).click();

  const projectName = `Delete Project ${Date.now()}`;
  await page.getByRole("textbox", { name: "Name" }).fill(projectName);
  await page.getByRole("textbox", { name: "Description" }).fill("delete flow");
  await page.locator("#form-create-project").getByRole("button", { name: "Create" }).click();

  const createdRow = page.locator("table tbody tr", { hasText: projectName });
  const inspector = page.locator(".inspector-panel");
  await expect(createdRow).toBeVisible();
  await expect(createdRow).toHaveClass(/is-selected/);
  await expect(createdRow.getByRole("button", { name: `Delete ${projectName}` })).toBeVisible();
  await expect(inspector).toContainText(projectName);

  await createdRow.getByRole("button", { name: `Delete ${projectName}` }).click();
  const dialog = page.getByRole("dialog", { name: "Delete Project" });
  await expect(dialog).toContainText(projectName);
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(createdRow).toBeVisible();
  await expect(inspector).toContainText(projectName);

  await createdRow.getByRole("button", { name: `Delete ${projectName}` }).click();
  await dialog.getByRole("button", { name: "Delete project" }).click();
  await expect(page.locator("table tbody tr", { hasText: projectName })).toHaveCount(0);
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toHaveCount(0);
  await expect(page.locator(".selection-panel .sel-section").first().locator(".sel-count")).toHaveText("0");
  await expect(inspector).toContainText("No active project.");
  await expect(inspector).not.toContainText(projectName);
  await expect(page.locator(".status-bar")).not.toContainText(projectName);
  await expect(page.locator(".artifact-table-title")).toContainText("Projects");

  await ribbon.getByRole("button", { name: "Trash" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Trash");
  const trashProjectRow = page.locator("table tbody tr", { hasText: projectName });
  await expect(trashProjectRow).toBeVisible();
  await trashProjectRow.getByRole("button", { name: "Restore" }).click();
  await expect(trashProjectRow).toHaveCount(0);
  await ribbon.getByRole("button", { name: "Projects" }).click();
  await expect(page.locator("table tbody tr", { hasText: projectName })).toBeVisible();
});

test("Home commands switch center views", async ({ page }) => {
  await page.goto("/");

  const ribbon = page.locator(".ribbon");
  await expect(ribbon.getByRole("button", { name: "Import" })).toBeDisabled();
  await expect(page.locator(".artifact-table-title")).toContainText("Projects");

  await ribbon.getByRole("button", { name: "Settings" }).click();
  const settingsSurface = page.locator(".settings-surface");
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await expect(settingsSurface.getByRole("button", { name: "General" })).toHaveClass(/is-active/);
  await expect(settingsSurface).toContainText("No general settings yet.");
  await settingsSurface.getByRole("button", { name: "Docs" }).click();
  await expect(settingsSurface.getByRole("button", { name: "Docs" })).toHaveClass(/is-active/);
  await expect(settingsSurface.locator(".settings-docs-panel")).toContainText("Workbench Flow");
  await expect(settingsSurface.locator(".settings-doc-content")).toContainText("NEExT Workbench is a local, project-first interface");
  await expect(ribbon.getByRole("button", { name: "Help" })).toHaveCount(0);

  await ribbon.getByRole("button", { name: "Trash" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Trash");

  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name: "Create Project" })).toBeVisible();

  await ribbon.getByRole("button", { name: "Projects" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Projects");
});

test("Left Panel and Center Views scope artifacts by dataset and mark selected lineage relationships", async ({ page }) => {
  const projectName = `Lineage Project ${Date.now()}`;
  seedLineageSelectionProject(projectName);

  await page.goto("/");
  const ribbon = page.locator(".ribbon");
  const selection = page.locator(".selection-panel");
  const section = (title: string) => selection.locator(".sel-section", { hasText: title });
  const item = (title: string, name: string) => section(title).locator(".sel-item", { hasText: name });

  await expect(selection.locator(".sel-item-name", { hasText: projectName })).toBeVisible();
  await expect(selection.locator(".sel-context")).toHaveText(`Context: Project ${projectName}`);
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(section("Features").locator(".sel-count")).toHaveText("0");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("0");
  await expect(section("Models").locator(".sel-count")).toHaveText("0");
  await expect(item("Features", "Lineage Alpha - PageRank")).toHaveCount(0);
  await expect(item("Embeddings", "Alpha Page Embedding")).toHaveCount(0);
  await expect(item("Models", "Alpha Classifier")).toHaveCount(0);

  await item("Datasets", "Lineage Alpha").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Dataset Lineage Alpha");
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(item("Datasets", "Lineage Alpha")).toHaveClass(/is-active/);
  await expect(item("Datasets", "Lineage Beta")).toBeVisible();
  await expect(section("Features").locator(".sel-count")).toHaveText("2");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("2");
  await expect(section("Models").locator(".sel-count")).toHaveText("1");
  await expect(item("Features", "Lineage Beta - PageRank")).toHaveCount(0);
  await expect(item("Embeddings", "Beta Page Embedding")).toHaveCount(0);
  await expect(item("Models", "Beta Classifier")).toHaveCount(0);
  await expect(page.locator(".artifact-table-title")).toContainText("Datasets");
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Beta" })).toBeVisible();

  await item("Datasets", "Lineage Beta").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Dataset Lineage Beta");
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(item("Datasets", "Lineage Beta")).toHaveClass(/is-active/);
  await expect(item("Datasets", "Lineage Alpha")).toBeVisible();
  await expect(section("Features").locator(".sel-count")).toHaveText("1");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("1");
  await expect(section("Models").locator(".sel-count")).toHaveText("1");
  await expect(item("Features", "Lineage Beta - PageRank")).toBeVisible();
  await expect(item("Features", "Lineage Alpha - PageRank")).toHaveCount(0);
  await expect(item("Embeddings", "Beta Page Embedding")).toBeVisible();
  await expect(item("Models", "Beta Classifier")).toBeVisible();

  await item("Datasets", "Lineage Alpha").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Dataset Lineage Alpha");

  await page.getByRole("button", { name: "FEATURES" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Features · 2 features");
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - PageRank" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - Degree Centrality" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Beta - PageRank" })).toHaveCount(0);
  await ribbon.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Feature Explore");
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - PageRank" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - Degree Centrality" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Beta - PageRank" })).toHaveCount(0);

  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embeddings · 2 embeddings");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Page Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Combined Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Page Embedding" })).toHaveCount(0);
  await ribbon.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embedding Explore");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Page Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Combined Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Page Embedding" })).toHaveCount(0);

  await page.getByRole("button", { name: "MODELS" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Models - 1 model");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Classifier" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Classifier" })).toHaveCount(0);
  await ribbon.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Model Explore");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Classifier" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Classifier" })).toHaveCount(0);

  await item("Features", "Lineage Alpha - PageRank").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Feature Lineage Alpha - PageRank");
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(item("Datasets", "Lineage Alpha")).toHaveClass(/is-active/);
  await expect(item("Datasets", "Lineage Beta")).toBeVisible();
  await expect(section("Features").locator(".sel-count")).toHaveText("2");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("2");
  await expect(section("Models").locator(".sel-count")).toHaveText("1");
  await expect(item("Features", "Lineage Alpha - Degree Centrality")).toBeVisible();
  await expect(item("Embeddings", "Alpha Page Embedding")).toHaveClass(/is-related/);
  await expect(item("Embeddings", "Alpha Combined Embedding")).toHaveClass(/is-related/);
  await expect(item("Models", "Alpha Classifier")).toHaveClass(/is-related/);
  await expect(page.locator(".artifact-table-title")).toContainText("Features");
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - PageRank" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Alpha - Degree Centrality" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Lineage Beta - PageRank" })).toHaveCount(0);

  await item("Embeddings", "Alpha Combined Embedding").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Embedding Alpha Combined Embedding");
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(item("Datasets", "Lineage Alpha")).toHaveClass(/is-active/);
  await expect(item("Datasets", "Lineage Beta")).toBeVisible();
  await expect(section("Features").locator(".sel-count")).toHaveText("2");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("2");
  await expect(section("Models").locator(".sel-count")).toHaveText("1");
  await expect(item("Features", "Lineage Alpha - PageRank")).toBeVisible();
  await expect(item("Features", "Lineage Alpha - Degree Centrality")).toBeVisible();
  await expect(item("Embeddings", "Alpha Page Embedding")).toBeVisible();
  await expect(item("Models", "Alpha Classifier")).toHaveClass(/is-related/);
  await expect(page.locator(".artifact-table-title")).toContainText("Embeddings");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Page Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Combined Embedding" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Page Embedding" })).toHaveCount(0);

  await item("Models", "Alpha Classifier").click();
  await expect(selection.locator(".sel-context")).toHaveText("Context: Model Alpha Classifier");
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(item("Datasets", "Lineage Alpha")).toHaveClass(/is-active/);
  await expect(item("Datasets", "Lineage Beta")).toBeVisible();
  await expect(section("Features").locator(".sel-count")).toHaveText("2");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("2");
  await expect(section("Models").locator(".sel-count")).toHaveText("1");
  await expect(item("Embeddings", "Alpha Combined Embedding")).toBeVisible();
  await expect(item("Models", "Beta Classifier")).toHaveCount(0);
  await expect(page.locator(".artifact-table-title")).toContainText("Models");
  await expect(page.locator(".document table tbody tr", { hasText: "Alpha Classifier" })).toBeVisible();
  await expect(page.locator(".document table tbody tr", { hasText: "Beta Classifier" })).toHaveCount(0);

  await item("Project", projectName).click();
  await expect(selection.locator(".sel-context")).toHaveText(`Context: Project ${projectName}`);
  await expect(section("Datasets").locator(".sel-count")).toHaveText("2");
  await expect(section("Features").locator(".sel-count")).toHaveText("0");
  await expect(section("Embeddings").locator(".sel-count")).toHaveText("0");
  await expect(section("Models").locator(".sel-count")).toHaveText("0");
  await expect(item("Features", "Lineage Alpha - PageRank")).toHaveCount(0);
  await expect(item("Embeddings", "Alpha Page Embedding")).toHaveCount(0);
  await expect(item("Models", "Alpha Classifier")).toHaveCount(0);

  await page.getByRole("button", { name: "FEATURES" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Features · 0 features");
  await expect(page.locator(".artifact-table-empty")).toContainText("No features.");
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embeddings · 0 embeddings");
  await expect(page.locator(".artifact-table-empty")).toContainText("No embeddings.");
  await page.getByRole("button", { name: "MODELS" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Models - 0 models");
  await expect(page.locator(".artifact-table-empty")).toContainText("No models.");
});

test("Artifact lifecycle deletes, cascades, and restores from Home Trash", async ({ page }) => {
  const projectName = `Lifecycle Project ${Date.now()}`;
  seedLineageSelectionProject(projectName);

  await page.goto("/");
  const ribbon = page.locator(".ribbon");
  const selection = page.locator(".selection-panel");
  const alphaDatasetItem = selection.locator(".sel-section", { hasText: "Datasets" }).locator(".sel-item", { hasText: "Lineage Alpha" });
  const betaDatasetItem = selection.locator(".sel-section", { hasText: "Datasets" }).locator(".sel-item", { hasText: "Lineage Beta" });

  await alphaDatasetItem.click();
  await page.getByRole("button", { name: "MODELS" }).click();
  const alphaModelRow = page.locator("table tbody tr", { hasText: "Alpha Classifier" });
  await expect(alphaModelRow).toBeVisible();
  await alphaModelRow.getByRole("button", { name: "Delete Alpha Classifier" }).click();
  let dialog = page.getByRole("dialog", { name: "Delete Artifact" });
  await expect(dialog).toContainText('Move "Alpha Classifier" to project trash?');
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(alphaModelRow).toBeVisible();

  await alphaModelRow.getByRole("button", { name: "Delete Alpha Classifier" }).click();
  dialog = page.getByRole("dialog", { name: "Delete Artifact" });
  await dialog.getByRole("button", { name: "Delete artifact" }).click();
  await expect(page.locator("table tbody tr", { hasText: "Alpha Classifier" })).toHaveCount(0);
  await expect(page.locator(".artifact-table-empty")).toContainText("No models.");

  await betaDatasetItem.click();
  await page.getByRole("button", { name: "MODELS" }).click();
  await expect(page.locator("table tbody tr", { hasText: "Beta Classifier" })).toBeVisible();

  await page.getByRole("button", { name: "HOME", exact: true }).click();
  await ribbon.getByRole("button", { name: "Trash" }).click();
  let trashRow = page.locator("table tbody tr", { hasText: "Alpha Classifier" });
  await expect(trashRow).toBeVisible();
  await trashRow.getByRole("button", { name: "Restore" }).click();
  await expect(trashRow).toHaveCount(0);

  await alphaDatasetItem.click();
  await page.getByRole("button", { name: "MODELS" }).click();
  await expect(page.locator("table tbody tr", { hasText: "Alpha Classifier" })).toBeVisible();

  await page.getByRole("button", { name: "DATASETS" }).click();
  const alphaDatasetRow = page.locator("table tbody tr", { hasText: "Lineage Alpha" });
  await expect(alphaDatasetRow).toBeVisible();
  await alphaDatasetRow.getByRole("button", { name: "Delete Lineage Alpha" }).click();
  dialog = page.getByRole("dialog", { name: "Delete Artifact Bundle" });
  await expect(dialog).toContainText('Deleting "Lineage Alpha" will also move downstream artifacts to project trash.');
  await expect(dialog).toContainText("Lineage Alpha - PageRank");
  await expect(dialog).toContainText("Alpha Combined Embedding");
  await expect(dialog).toContainText("Alpha Classifier");
  await dialog.getByRole("button", { name: "Delete bundle" }).click();
  await expect(page.locator("table tbody tr", { hasText: "Lineage Alpha" })).toHaveCount(0);
  await expect(page.locator("table tbody tr", { hasText: "Lineage Beta" })).toBeVisible();

  await page.getByRole("button", { name: "HOME", exact: true }).click();
  await ribbon.getByRole("button", { name: "Trash" }).click();
  trashRow = page.locator("table tbody tr", { hasText: "Lineage Alpha" });
  await expect(trashRow).toBeVisible();
  await trashRow.getByRole("button", { name: "Restore" }).click();
  await expect(trashRow).toHaveCount(0);

  await page.getByRole("button", { name: "DATASETS" }).click();
  await expect(page.locator("table tbody tr", { hasText: "Lineage Alpha" })).toBeVisible();
  await expect(page.locator("table tbody tr", { hasText: "Lineage Beta" })).toBeVisible();
});

test("Home Settings enables, regenerates, and disables local MCP setup", async ({ page }) => {
  await page.request.post("/api/mcp-settings/disable");
  await page.goto("/");

  const ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Settings" }).click();
  const settingsSurface = page.locator(".settings-surface");

  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await expect(settingsSurface.getByRole("button", { name: "General" })).toHaveClass(/is-active/);
  await expect(settingsSurface).toContainText("No general settings yet.");

  await settingsSurface.getByRole("button", { name: "Docs" }).click();
  await expect(settingsSurface.getByRole("button", { name: "Docs" })).toHaveClass(/is-active/);
  const docsPanel = settingsSurface.locator(".settings-docs-panel");
  await expect(docsPanel.locator(".settings-doc-topic")).toHaveCount(8);
  await expect(docsPanel.locator(".settings-doc-topic", { hasText: "Overview" })).toHaveClass(/is-active/);
  await expect(docsPanel.locator(".settings-doc-content")).toContainText("Project Create and custom Feature Create are the active Create workflows.");
  await docsPanel.locator(".settings-doc-topic", { hasText: "Features and Custom Features" }).click();
  await expect(docsPanel.locator(".settings-doc-content")).toContainText("compute_feature(graph)");
  await expect(docsPanel.locator(".settings-doc-content")).toContainText("trusted local Python, not sandboxed");
  await docsPanel.locator(".settings-doc-topic", { hasText: "NEExT Library Quickstart" }).click();
  await expect(docsPanel.locator(".settings-doc-content")).toContainText("embedding_dimension=8");
  await expect(docsPanel.locator(".settings-doc-content")).toContainText("graph_labels.csv uses graph_id and graph_label.");

  await settingsSurface.getByRole("button", { name: "Agentic", exact: true }).click();
  await expect(settingsSurface.getByRole("button", { name: "Agentic", exact: true })).toHaveClass(/is-active/);
  await expect(page.getByText("MCP disabled")).toBeVisible();
  await expect(page.getByText("make neext-workbench")).toBeVisible();
  await expect(page.getByText("http://127.0.0.1:8765/mcp")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Setup Checklist" })).toBeVisible();
  await expect(page.getByText("Verify the client can list tools")).toBeVisible();
  await expect(page.getByRole("heading", { name: "ChatGPT" })).toHaveCount(0);

  await page.getByRole("button", { name: "Enable MCP" }).click();
  const token = page.locator(".settings-token", { hasText: "One-time token" }).locator("code");
  await expect(token).toContainText("nxt_mcp_");
  const firstToken = await token.textContent();
  const configPanel = page.locator(".settings-configs");
  const clientTabs = configPanel.locator(".settings-client-tabs");
  // HTTP clients are always offered, regardless of stdio readiness.
  await expect(clientTabs.getByRole("button", { name: "Claude Code", exact: true })).toBeVisible();
  await expect(clientTabs.getByRole("button", { name: "Cursor", exact: true })).toBeVisible();
  await expect(clientTabs.getByRole("button", { name: "Generic Streamable HTTP", exact: true })).toBeVisible();

  // Claude Desktop is stdio-only: the snippet is shown only when the local
  // environment can host the server. When stdio is blocked (e.g. the interpreter
  // or workspace sits inside a macOS protected folder), the UI suppresses the
  // snippet and surfaces remediation instead. Assert whichever state applies.
  const claudeDesktopTab = clientTabs.getByRole("button", { name: "Claude Desktop local", exact: true });
  const stdioBlocked = (await page.locator(".settings-readiness.is-blocked").count()) > 0;
  if (stdioBlocked) {
    await expect(claudeDesktopTab).toHaveCount(0);
    await expect(page.locator(".settings-readiness.is-blocked")).toContainText("Claude Desktop");
    await expect(page.locator(".settings-readiness-list").first()).not.toHaveCount(0);
  } else {
    await expect(claudeDesktopTab).toHaveClass(/is-active/);
    await expect(configPanel.locator(".settings-snippet")).toHaveCount(1);
    await expect(configPanel.locator(".settings-snippet")).toContainText("\"command\"");
    await expect(configPanel.locator(".settings-snippet")).toContainText("NEExT.workbench.mcp_cli");
    await expect(configPanel.locator(".settings-snippet")).toContainText("NEEXT_WORKBENCH_MCP_TOKEN");
    await expect(configPanel.locator(".settings-snippet")).toContainText("nxt_mcp_");
  }
  await expect(page.locator(".settings-mcp-panel .settings-mcp-meta")).toContainText("Transport:");
  await expect(page.locator(".settings-capabilities")).toContainText("tools available");

  await clientTabs.getByRole("button", { name: "Cursor", exact: true }).click();
  await expect(configPanel.locator(".settings-snippet")).toContainText("Cursor");
  await expect(configPanel.locator(".settings-snippet")).toContainText(".cursor/mcp.json");

  await clientTabs.getByRole("button", { name: "Claude Code", exact: true }).click();
  await expect(configPanel.locator(".settings-snippet")).toContainText("claude mcp add --transport http");

  await clientTabs.getByRole("button", { name: "Generic Streamable HTTP", exact: true }).click();
  await expect(configPanel.locator(".settings-snippet")).toContainText("mcp.json");
  await expect(configPanel.locator(".settings-snippet")).toContainText("streamable-http");
  await expect(configPanel.locator(".settings-snippet")).toContainText("Authorization");

  await page.getByRole("button", { name: "Regenerate Token" }).click();
  await expect.poll(async () => token.textContent()).not.toBe(firstToken);

  await page.getByRole("button", { name: "Disable MCP" }).click();
  await expect(page.getByText("MCP disabled")).toBeVisible();
  await expect(page.locator(".settings-snippet")).toHaveCount(0);
});

test("Models Import remains title-only and Create is hidden", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "MODELS" }).click();
  const ribbon = page.locator(".ribbon");
  await expect(page.locator(".artifact-table-title")).toContainText("Models");

  await ribbon.getByRole("button", { name: "Import" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Import");
  await expect(ribbon.getByRole("button", { name: "Create" })).toHaveCount(0);
});

test("Feature Create saves, runs, and previews custom Python features", async ({ page }) => {
  const projectName = `Custom Feature Project ${Date.now()}`;
  seedTinyEmbeddingProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.locator(".selection-panel .sel-section", { hasText: "Datasets" }).locator(".sel-item", { hasText: "Tiny Embedding" }).click();
  await page.getByRole("button", { name: "FEATURES" }).click();
  const ribbon = page.locator(".ribbon");

  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name: "Create Custom Feature" })).toBeVisible();
  await expect(page.getByText("Dataset: Tiny Embedding")).toBeVisible();
  await expect(page.getByLabel("Dataset")).toHaveCount(0);
  await expect(page.getByLabel("Parallel Jobs")).toHaveCount(0);
  await expect(page.getByLabel("Parallel Backend")).toHaveCount(0);
  const customCode = `import pandas as pd

def compute_feature(graph):
    nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
    values = [float(graph.G.degree(node)) for node in nodes]
    df = pd.DataFrame({
        "node_id": nodes,
        "graph_id": graph.graph_id,
        "custom_degree": values,
    })
    return df[["node_id", "graph_id", "custom_degree"]]
`;
  await page.getByLabel("Name").fill("Custom UI Degree");
  await page.getByLabel("Python Code").fill(customCode);
  await page.getByLabel("Normalize Features").uncheck();
  await page.getByRole("button", { name: "Show custom feature guide" }).click();
  await expect(page.getByRole("heading", { name: "Custom Feature Guide" })).toBeVisible();
  const guideCard = page.locator(".feature-guide-card");
  await expect(guideCard).toContainText("compute_feature(graph)");
  await expect(guideCard).toContainText("trusted local Python, not sandboxed");
  await expect(guideCard).toContainText("Missing Python packages are reported clearly");
  await expect(guideCard).toContainText("n_jobs=1");
  await page.getByRole("button", { name: "Back to custom feature form" }).click();
  await expect(page.getByRole("heading", { name: "Create Custom Feature" })).toBeVisible();
  await expect(page.getByLabel("Name")).toHaveValue("Custom UI Degree");
  await expect(page.getByLabel("Python Code")).toHaveValue(customCode);
  await expect(page.getByLabel("Normalize Features")).not.toBeChecked();
  await page.locator(".card-foot").getByRole("button", { name: "Validate" }).click();
  await expect(page.getByText("Valid feature output: custom_degree")).toBeVisible();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();

  const customRow = page.locator("table tbody tr", { hasText: "Custom UI Degree" }).first();
  await expect(customRow).toBeVisible();
  await expect(customRow).toContainText("Tiny Embedding");
  await expect(customRow).toContainText("Custom Python");
  await expect(customRow).toContainText("planned");
  await customRow.getByRole("button", { name: "Run" }).click();
  await expect(customRow.locator(".status-pill")).toHaveText("completed", { timeout: 30_000 });
  await customRow.getByRole("button", { name: "Preview" }).click();

  await expect(page.locator(".artifact-table-title")).toContainText("Custom UI Degree Explore");
  const featureExplore = page.locator(".feature-explore");
  await featureExplore.getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("custom_degree");
});

test("Dataset Explore graph search and navigation update Inspector", async ({ page }) => {
  const projectName = `Explore Project ${Date.now()}`;
  seedTinyEmbeddingProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.getByRole("button", { name: "DATASETS" }).click();
  const ribbon = page.locator(".ribbon");
  const inspector = page.locator(".inspector-panel");
  const datasetRow = page.locator("table tbody tr", { hasText: "Tiny Embedding" }).first();
  await expect(datasetRow).toBeVisible();
  await datasetRow.getByRole("button", { name: "Preview" }).click();

  const exploreView = page.locator(".dataset-explore");
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toHaveClass(/is-active/);
  await exploreView.getByRole("button", { name: "Graph", exact: true }).click();
  await expect(exploreView.getByLabel("Search graphs and nodes")).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Previous graph" })).toBeDisabled();
  await expect(exploreView.getByRole("button", { name: "Next graph" })).toBeEnabled();
  await expect(inspector).toContainText("Dataset Graph Details");
  await expect(inspector).toContainText("Tiny Embedding");
  await expect(inspector).toContainText("g1");

  await exploreView.getByRole("button", { name: "Next graph" }).click();
  await expect(inspector).toContainText("g2");
  await expect(exploreView.getByRole("button", { name: "Next graph" })).toBeDisabled();

  await exploreView.locator(".graph-tab-panel").focus();
  await page.keyboard.press("ArrowLeft");
  await expect(inspector).toContainText("g1");

  await exploreView.getByLabel("Search graphs and nodes").fill("2");
  const nodeResult = exploreView.locator(".graph-search-result", { hasText: "graph g1" }).first();
  await expect(nodeResult).toBeVisible();
  await nodeResult.click();
  await expect(inspector).toContainText("Dataset Node Details");
  await expect(inspector).toContainText("Node ID");
  await expect(inspector).toContainText("2");
  await expect(inspector).toContainText("Degree");
  await expect(inspector).toContainText("Visible In Visual");

  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    exploreView.getByRole("button", { name: "Export CSV" }).click()
  ]);
  expect(download.suggestedFilename()).toBe("Tiny_Embedding_nodes.csv");
  const downloadedPath = await download.path();
  expect(downloadedPath).toBeTruthy();
  const csv = fs.readFileSync(String(downloadedPath), "utf-8");
  expect(csv).toContain("graph_id,node_id");
  expect(csv).toContain("g1,0");

  await exploreView.getByRole("button", { name: "Back to Datasets" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Datasets");
  await expect(datasetRow).toHaveClass(/is-selected/);
});

test("Feature Explore shows statistics, PCA, data, and node inspector details", async ({ page }) => {
  const projectName = `Feature Explore Project ${Date.now()}`;
  seedTinyFeatureExploreProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page
    .locator(".selection-panel .sel-section", { hasText: "Datasets" })
    .locator(".sel-item", { hasText: "Tiny Feature Explore" })
    .click();
  await page.getByRole("button", { name: "FEATURES" }).click();

  const ribbon = page.locator(".ribbon");
  const featureAnalysisGroup = ribbon.locator(".tool-group", { hasText: "Feature Analysis" });
  await expect(featureAnalysisGroup.getByRole("button", { name: "Explore" })).toBeVisible();

  await featureAnalysisGroup.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Feature Explore");
  await expect(page.locator("table tbody tr", { hasText: "Tiny Feature Explore - PageRank" })).toBeVisible();

  await ribbon.getByRole("button", { name: "Features" }).click();
  const featureRow = page.locator("table tbody tr", { hasText: "Tiny Feature Explore - PageRank" }).first();
  await expect(featureRow).toBeVisible();
  await featureRow.getByRole("button", { name: "Preview" }).click();

  const exploreView = page.locator(".feature-explore");
  const inspector = page.locator(".inspector-panel");
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Feature Explore - PageRank Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "Choose Feature" })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "PCA", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Data", exact: true })).toBeVisible();
  await expect(exploreView).toContainText("Rows");
  await expect(exploreView).toContainText("Graph Labels");

  await exploreView.getByRole("button", { name: "PCA", exact: true }).click();
  await expect(exploreView.locator(".feature-pca-control-band")).toBeVisible();
  await expect(exploreView.locator(".feature-pca-control-band")).toContainText("Direct 2D");
  const chart = exploreView.locator(".feature-pca-chart");
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toBeVisible();
  const pointPosition = await chart.evaluate((element) => {
    const chartElement = element as HTMLDivElement & {
      __featurePcaChart?: {
        getOption: () => {
          xAxis?: { name?: string }[];
          yAxis?: { name?: string }[];
          series?: { data?: { graph_id: string; value: [number, number] }[] }[];
        };
        convertToPixel: (finder: { seriesIndex: number }, value: [number, number]) => [number, number];
      };
    };
    const chartInstance = chartElement.__featurePcaChart;
    if (!chartInstance) throw new Error("Feature PCA chart instance is not attached.");
    const option = chartInstance.getOption();
    if (option.xAxis?.[0]?.name !== "page_rank_0") throw new Error("Raw 2D x-axis label was not applied.");
    if (option.yAxis?.[0]?.name !== "page_rank_1") throw new Error("Raw 2D y-axis label was not applied.");
    const datum = option.series?.[0]?.data?.find((point) => point.graph_id === "g1") || option.series?.[0]?.data?.[0];
    if (!datum) throw new Error("Feature PCA chart has no plotted points.");
    const [x, y] = chartInstance.convertToPixel({ seriesIndex: 0 }, datum.value);
    return { x, y };
  });
  await chart.click({ position: pointPosition });
  await expect(inspector).toContainText("Feature Graph Details");
  await expect(inspector).toContainText("Plotted In Chart");
  await expect(inspector).toContainText("Node Count");
  await expect(inspector).toContainText("page_rank_0");

  await exploreView.getByLabel("Search feature graph IDs and labels").fill("g1");
  const graphResult = exploreView.locator(".graph-search-result", { hasText: "g1" }).first();
  await expect(graphResult).toBeVisible();
  await graphResult.click();
  await expect(inspector).toContainText("Feature Graph Details");
  await expect(inspector).toContainText("Graph ID");
  await expect(inspector).toContainText("g1");
  await expect(inspector).toContainText("Aggregation");
  await expect(inspector).toContainText("Mean");

  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("page_rank_0");
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(6);

  await exploreView.getByRole("button", { name: "Choose Feature" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Feature Explore");
  await expect(page.locator("table tbody tr", { hasText: "Tiny Feature Explore - One Column" })).toBeVisible();
  await page.locator("table tbody tr", { hasText: "Tiny Feature Explore - One Column" }).getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Feature Explore - One Column Explore");
  await expect(exploreView.getByRole("button", { name: "PCA", exact: true })).toBeDisabled();
  await expect(exploreView).toContainText("Feature Columns");
});

test("Embedding Explore shows statistics, PCA, data, and graph inspector details", async ({ page }) => {
  const projectName = `Embedding Explore Project ${Date.now()}`;
  seedTinyEmbeddingExploreProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page
    .locator(".selection-panel .sel-section", { hasText: "Datasets" })
    .locator(".sel-item", { hasText: "Tiny Embedding Explore" })
    .click();
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();

  const ribbon = page.locator(".ribbon");
  const embeddingAnalysisGroup = ribbon.locator(".tool-group", { hasText: "Embedding Analysis" });
  await expect(embeddingAnalysisGroup.getByRole("button", { name: "Explore" })).toBeVisible();

  await embeddingAnalysisGroup.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embedding Explore");
  await expect(page.locator("table tbody tr", { hasText: "Tiny Embedding Explore - Two Dimensions" })).toBeVisible();

  await ribbon.getByRole("button", { name: "Embeddings" }).click();
  const embeddingRow = page.locator("table tbody tr", { hasText: "Tiny Embedding Explore - Two Dimensions" }).first();
  await expect(embeddingRow).toBeVisible();
  await embeddingRow.getByRole("button", { name: "Preview" }).click();

  const exploreView = page.locator(".embedding-explore");
  const inspector = page.locator(".inspector-panel");
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Embedding Explore - Two Dimensions Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "Choose Embedding" })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "PCA", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Data", exact: true })).toBeVisible();
  await expect(exploreView).toContainText("Graphs");
  await expect(exploreView).toContainText("Graph Labels");

  await exploreView.getByRole("button", { name: "PCA", exact: true }).click();
  await expect(exploreView.locator(".feature-pca-control-band")).toBeVisible();
  await expect(exploreView.locator(".feature-pca-control-band")).toContainText("Direct 2D");
  const chart = exploreView.locator(".embedding-pca-chart");
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toBeVisible();
  const pointPosition = await chart.evaluate((element) => {
    const chartElement = element as HTMLDivElement & {
      __embeddingPcaChart?: {
        getOption: () => {
          xAxis?: { name?: string }[];
          yAxis?: { name?: string }[];
          series?: { data?: { graph_id: string; value: [number, number] }[] }[];
        };
        convertToPixel: (finder: { seriesIndex: number }, value: [number, number]) => [number, number];
      };
    };
    const chartInstance = chartElement.__embeddingPcaChart;
    if (!chartInstance) throw new Error("Embedding PCA chart instance is not attached.");
    const option = chartInstance.getOption();
    if (option.xAxis?.[0]?.name !== "emb_0") throw new Error("Raw 2D x-axis label was not applied.");
    if (option.yAxis?.[0]?.name !== "emb_1") throw new Error("Raw 2D y-axis label was not applied.");
    const datum = option.series?.[0]?.data?.find((point) => point.graph_id === "g1") || option.series?.[0]?.data?.[0];
    if (!datum) throw new Error("Embedding PCA chart has no plotted points.");
    const [x, y] = chartInstance.convertToPixel({ seriesIndex: 0 }, datum.value);
    return { x, y };
  });
  await chart.click({ position: pointPosition });
  await expect(inspector).toContainText("Embedding Graph Details");
  await expect(inspector).toContainText("Plotted In Chart");
  await expect(inspector).toContainText("emb_0");

  await exploreView.getByLabel("Search embedding graph IDs and labels").fill("g1");
  const graphResult = exploreView.locator(".graph-search-result", { hasText: "g1" }).first();
  await expect(graphResult).toBeVisible();
  await graphResult.click();
  await expect(inspector).toContainText("Embedding Graph Details");
  await expect(inspector).toContainText("Graph ID");
  await expect(inspector).toContainText("g1");
  await expect(inspector).toContainText("Graph Label");

  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("emb_0");
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(2);
  await expect(exploreView).toContainText("1-2 of 2");

  await exploreView.getByRole("button", { name: "Choose Embedding" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embedding Explore");
  await expect(page.locator("table tbody tr", { hasText: "Tiny Embedding Explore - One Dimension" })).toBeVisible();
  await page.locator("table tbody tr", { hasText: "Tiny Embedding Explore - One Dimension" }).getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Embedding Explore - One Dimension Explore");
  await expect(exploreView.getByRole("button", { name: "PCA", exact: true })).toBeDisabled();
  await expect(exploreView).toContainText("Dimensions");
});

test("Dataset Library configures single graph egonet datasets", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");

  const projectName = `Single Graph Dataset ${Date.now()}`;
  const ribbon = page.locator(".ribbon");
  await createProject(page, projectName, "single graph egonet flow");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();

  await page.getByRole("button", { name: "DATASETS" }).click();
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Type", "Source", "Size", "Status", "Actions"]);
  const karateCatalogRow = page.locator("table tbody tr", { hasText: "Zachary Karate Club" }).first();
  const inspector = page.locator(".inspector-panel");
  await expect(karateCatalogRow).toBeVisible();
  await expect(karateCatalogRow).toContainText("Single Graph");
  await karateCatalogRow.click();
  await expect(inspector).toContainText("Catalog Dataset Details");
  await expect(inspector).toContainText("Single Graph");
  await expect(inspector).toContainText("club");

  await karateCatalogRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure Zachary Karate Club" })).toBeVisible();
  await expect(page.getByLabel("K-Hop")).toHaveValue("1");
  await expect(page.getByLabel("Node Selection")).toHaveValue("all_nodes");
  await expect(page.getByLabel("Target Attribute")).toContainText("club");
  await page.getByRole("button", { name: "Back" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Dataset Library");
  await expect(karateCatalogRow).toBeVisible();

  await karateCatalogRow.getByRole("button", { name: "Configure" }).click();
  await page.getByLabel("Target Attribute").selectOption("club");
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();
  await expect(page.locator(".sel-section", { hasText: "Datasets" }).locator(".sel-count")).toHaveText("1");

  const datasetRow = page.locator("table tbody tr", { hasText: "Zachary Karate Club" }).first();
  await expect(datasetRow).toBeVisible();
  await expect(datasetRow).toContainText("planned");
  await datasetRow.getByRole("button", { name: "Run" }).click();
  await expect(page.locator(".cmd")).toContainText("Computing 1-hop egonets for Zachary Karate Club", { timeout: 20_000 });
  await expect(datasetRow.locator(".status-pill")).toHaveText("completed", { timeout: 30_000 });
  await datasetRow.getByRole("button", { name: "Preview" }).click();

  const exploreView = page.locator(".dataset-explore");
  await expect(page.locator(".artifact-table-title")).toContainText("Zachary Karate Club Explore");
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toHaveClass(/is-active/);
  await expect(exploreView).toContainText("Source Graph");
  await expect(exploreView).toContainText("Prepared Egonet Collection");
  await expect(exploreView).toContainText("Egonet Generation");
  await expect(exploreView).toContainText("Target Attribute");
  await expect(exploreView).toContainText("Graph Labels");
  await exploreView.getByRole("button", { name: "Graph", exact: true }).click();
  await expect(exploreView.getByLabel("Search graphs and nodes")).toBeVisible();
  await expect(exploreView.locator(".graph-label-badge")).toContainText("Label");
  await expect(exploreView.locator(".graph-center-badge")).toContainText("Center Source Node");
  await expect(exploreView.locator(".graph-meta-badge", { hasText: "1-hop egonet" })).toBeVisible();
  await expect(exploreView.locator(".graph-meta-badge", { hasText: "Target club" })).toBeVisible();
  const firstCenterBadge = await exploreView.locator(".graph-center-badge").textContent();
  await exploreView.getByRole("button", { name: "Next graph" }).click();
  await expect.poll(async () => exploreView.locator(".graph-center-badge").textContent()).not.toBe(firstCenterBadge);
  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  await page.getByLabel("Dataset table").selectOption("graph_labels");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("graph_label");
  await page.getByLabel("Dataset table").selectOption("graph_mapping");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("source_node_id");
});

test("Datasets and Features run through planned artifacts and jobs", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");

  await page.getByRole("button", { name: "FEATURES" }).click();
  let ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Select a Dataset");
  await expect(page.locator(".artifact-table-empty")).toContainText("Select a dataset in the Left Panel");

  await page.getByRole("button", { name: "DATASETS" }).click();
  ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Type", "Source", "Size", "Status", "Actions"]);
  const mutagCatalogRow = page.locator("table tbody tr", { hasText: "MUTAG" }).first();
  const inspector = page.locator(".inspector-panel");
  await expect(mutagCatalogRow).toBeVisible();
  await expect(mutagCatalogRow).toContainText("188 graphs, 3,371 nodes, 7,442 edges");
  await expect(mutagCatalogRow.getByRole("button", { name: "Configure" })).toBeDisabled();
  await expect(page.locator(".sel-section", { hasText: "Datasets" }).locator(".sel-count")).toHaveText("0");
  await mutagCatalogRow.click();
  await expect(mutagCatalogRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Dataset Details");
  await expect(inspector).toContainText("Molecule graph collection for graph classification workflows.");
  await expect(inspector).toContainText("AnomalyPoint/NEExT_datasets");
  await expect(inspector).toContainText("Molecules");
  await expect(inspector).toContainText("graph_collection");
  await expect(inspector).toContainText("No active project");
  await expect(inspector).toContainText("188");
  await expect(inspector).toContainText("3371");
  await expect(inspector).toContainText("7442");

  const projectName = `Dataset Project ${Date.now()}`;
  await page.getByRole("button", { name: "HOME", exact: true }).click();
  await createProject(page, projectName, "dataset prepare flow");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();

  await page.getByRole("button", { name: "DATASETS" }).click();
  await ribbon.getByRole("button", { name: "Import" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Import");
  await expect(ribbon.getByRole("button", { name: "Create" })).toHaveCount(0);

  await ribbon.getByRole("button", { name: "Library" }).click();
  await mutagCatalogRow.click();
  await expect(mutagCatalogRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Dataset Details");
  await expect(inspector).toContainText("Not configured");
  await mutagCatalogRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure MUTAG" })).toBeVisible();
  await expect(page.getByLabel("Graph Backend")).toHaveValue("networkx");
  await expect(page.getByLabel("Filter Largest Component")).toBeChecked();
  await expect(page.getByLabel("Reindex Nodes")).toBeChecked();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();
  await expect(page.locator(".sel-section", { hasText: "Datasets" }).locator(".sel-count")).toHaveText("1");

  const leftPanelDatasetItem = page.locator(".selection-panel .sel-section", { hasText: "Datasets" }).locator(".sel-item", { hasText: "MUTAG" });
  const leftPanelProjectItem = page.locator(".selection-panel .sel-section", { hasText: "Project" }).locator(".sel-item", { hasText: projectName });
  await expect(leftPanelDatasetItem).toBeVisible();

  await leftPanelProjectItem.click();
  await page.getByRole("button", { name: "DATASETS" }).click();
  await ribbon.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Dataset Explore");
  await expect(page.locator("table tbody tr", { hasText: "MUTAG" })).toBeVisible();
  await expect(page.locator("table tbody tr", { hasText: "MUTAG" }).locator(".status-pill")).toHaveText("planned");

  await leftPanelDatasetItem.click();
  await expect(leftPanelDatasetItem).toHaveClass(/is-active/);
  await expect(inspector).toContainText("Dataset Details");
  await expect(inspector).toContainText("planned");
  await expect(inspector).toContainText("MUTAG");

  await ribbon.getByRole("button", { name: "Datasets" }).click();
  const datasetRow = page.locator("table tbody tr", { hasText: "MUTAG" }).first();
  await expect(datasetRow).toBeVisible();
  await expect(datasetRow).toContainText("MUTAG");
  await expect(datasetRow).toContainText("planned");
  await datasetRow.click();
  await expect(datasetRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Dataset Details");
  await expect(inspector).toContainText("MUTAG");
  await expect(inspector).toContainText("planned");
  await expect(inspector).toContainText("Dataset ID");
  await expect(inspector).toContainText(/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/);
  await expect(inspector).toContainText("Graphs");
  await expect(inspector).toContainText("Nodes");
  await expect(inspector).toContainText("Edges");
  await datasetRow.getByRole("button", { name: "Run" }).click();
  await expect(page.locator(".jobs-panel")).toContainText(/queued|running|completed/i);
  await expect(page.locator(".cmd")).toContainText("Preparing dataset MUTAG", { timeout: 20_000 });
  await expect(datasetRow.locator(".status-pill")).toHaveText("completed", { timeout: 90_000 });
  await expect(datasetRow.getByRole("button", { name: "Preview" })).toBeVisible();
  await datasetRow.getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("MUTAG Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  const exploreView = page.locator(".dataset-explore");
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Graph", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Data", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Statistics", exact: true })).toHaveClass(/is-active/);
  await expect(inspector).toContainText("MUTAG");
  await exploreView.getByRole("button", { name: "Graph", exact: true }).click();
  await expect(exploreView.getByLabel("Search graphs and nodes")).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Previous graph" })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Next graph" })).toBeVisible();
  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.getByLabel("Dataset table")).toHaveValue("nodes");
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(50);
  await page.getByLabel("Dataset table").selectOption("edges");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("src_node_id");
  await page.getByLabel("Dataset table").selectOption("node_mapping");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("source_node_id");

  await page.getByRole("button", { name: "FEATURES" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Features");
  await expect(page.locator(".artifact-table-empty")).toContainText("No features.");
  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name: "Create Custom Feature" })).toBeVisible();
  await expect(page.getByText("Dataset: MUTAG")).toBeVisible();
  await expect(page.getByLabel("Dataset")).toHaveCount(0);
  await expect(page.getByLabel("Parallel Jobs")).toHaveCount(0);
  await expect(page.getByLabel("Parallel Backend")).toHaveCount(0);

  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Type", "Output", "Actions"]);
  await expect(page.getByText("Dataset: MUTAG")).toBeVisible();
  const pageRankRow = page.locator("table tbody tr", { hasText: "PageRank" }).first();
  await expect(pageRankRow).toBeVisible();
  await pageRankRow.click();
  await expect(pageRankRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Feature Details");
  await expect(inspector).toContainText("PageRank scores with neighborhood aggregation.");
  await expect(inspector).toContainText("neext.compute_node_features");

  await pageRankRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure PageRank" })).toBeVisible();
  await expect(page.getByText("Dataset: MUTAG")).toBeVisible();
  await expect(page.getByLabel("Dataset")).toHaveCount(0);
  await expect(page.getByLabel("Feature Vector Length")).toHaveValue("3");
  await expect(page.getByLabel("Normalize Features")).toBeChecked();
  await expect(page.getByLabel("Parallel Jobs")).toHaveValue("1");
  await expect(page.getByLabel("Parallel Backend")).toHaveValue("loky");

  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();
  const featureRow = page.locator("table tbody tr", { hasText: "MUTAG - PageRank" }).first();
  await expect(featureRow).toBeVisible();
  await expect(featureRow).toContainText("MUTAG");
  await expect(featureRow).toContainText("PageRank");
  await expect(featureRow).toContainText("planned");
  await expect(featureRow).toHaveClass(/is-selected/);

  const leftPanelFeatureItem = page.locator(".selection-panel .sel-section", { hasText: "Features" }).locator(".sel-item", { hasText: "MUTAG - PageRank" });
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Features" }).locator(".sel-count")).toHaveText("1");
  await expect(leftPanelFeatureItem).toBeVisible();
  await leftPanelFeatureItem.click();
  await expect(leftPanelFeatureItem).toHaveClass(/is-active/);
  await expect(inspector).toContainText("Feature Details");
  await expect(inspector).toContainText("MUTAG - PageRank");
  await expect(inspector).toContainText("planned");
  await expect(inspector).toContainText("MUTAG");
  await expect(inspector).toContainText("page_rank_0, page_rank_1, page_rank_2");
  await expect(inspector).toContainText("neext.compute_node_features");
  await expect(inspector).toContainText("Feature ID");
  await featureRow.getByRole("button", { name: "Run" }).click();
  await expect(page.locator(".cmd")).toContainText("Computing features: page_rank", { timeout: 20_000 });
  await expect(featureRow.locator(".status-pill")).toHaveText("completed", { timeout: 90_000 });
  await featureRow.getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("MUTAG - PageRank Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  const featureExploreView = page.locator(".feature-explore");
  await expect(featureExploreView.getByRole("button", { name: "Statistics", exact: true })).toBeVisible();
  await expect(featureExploreView.getByRole("button", { name: "PCA", exact: true })).toBeVisible();
  await expect(featureExploreView.getByRole("button", { name: "Data", exact: true })).toBeVisible();
  await featureExploreView.getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("page_rank_0");
});

test("Embeddings library configures, batches, runs, and previews persisted artifacts", async ({ page }) => {
  test.setTimeout(120_000);
  const projectName = `Embedding Project ${Date.now()}`;
  seedTinyEmbeddingProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  const ribbon = page.locator(".ribbon");
  const inspector = page.locator(".inspector-panel");

  await expect(page.locator(".artifact-table-title")).toContainText("Embeddings");
  await expect(page.locator(".artifact-table-empty")).toContainText("No embeddings.");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Features" }).locator(".sel-count")).toHaveText("0");
  await expect(ribbon.getByRole("button", { name: "Create" })).toHaveCount(0);

  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Select a Dataset");
  await expect(page.locator(".artifact-table-empty")).toContainText("Select a dataset in the Left Panel before configuring embeddings.");
  await expect(page.locator(".artifact-table .tbl")).toHaveCount(0);

  await page
    .locator(".selection-panel .sel-section", { hasText: "Datasets" })
    .locator(".sel-item", { hasText: "Tiny Embedding" })
    .click();
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.getByText("Dataset: Tiny Embedding")).toBeVisible();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Algorithm", "Output", "Actions"]);
  const approxRow = page.locator("table tbody tr", { hasText: "Approx Wasserstein" }).first();
  await expect(approxRow).toBeVisible();
  await approxRow.click();
  await expect(approxRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Embedding Details");
  await expect(inspector).toContainText("approx_wasserstein");
  await expect(inspector).toContainText("neext.compute_graph_embeddings");

  await approxRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure Approx Wasserstein" })).toBeVisible();
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeDisabled();
  await expect(page.getByLabel("Embedding Dimension")).toHaveValue("3");
  await page.getByLabel("Embedding Dimension").fill("2");
  const pageRankFeaturePickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Embedding - PageRank" });
  const degreeFeaturePickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Embedding - Degree Centrality" });
  await pageRankFeaturePickerRow.click();
  await degreeFeaturePickerRow.click();
  await expect(page.getByText("Dataset: Tiny Embedding")).toBeVisible();
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeEnabled();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();

  const approxEmbeddingRow = page.locator("table tbody tr", { hasText: "Tiny Embedding - Approx Wasserstein Embedding" }).first();
  await expect(approxEmbeddingRow).toBeVisible();
  await expect(approxEmbeddingRow).toContainText("Tiny Embedding");
  await expect(approxEmbeddingRow).toContainText("Approx Wasserstein");
  await expect(approxEmbeddingRow).toContainText("Tiny Embedding - PageRank");
  await expect(approxEmbeddingRow).toContainText("Tiny Embedding - Degree Centrality");
  await expect(approxEmbeddingRow).toContainText("planned");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Embeddings" }).locator(".sel-count")).toHaveText("1");
  await approxEmbeddingRow.click();
  await expect(inspector).toContainText("Embedding Details");
  await expect(inspector).toContainText("Tiny Embedding - Approx Wasserstein Embedding");
  await expect(inspector).toContainText("Tiny Embedding - PageRank");
  await expect(inspector).toContainText("Tiny Embedding - Degree Centrality");
  await expect(inspector).toContainText("Dimension");
  await expect(inspector).toContainText("2");

  await page
    .locator(".selection-panel .sel-section", { hasText: "Features" })
    .locator(".sel-item", { hasText: "Tiny Embedding - PageRank" })
    .click();
  await expect(page.locator(".selection-panel .sel-context")).toContainText("Context: Feature Tiny Embedding - PageRank");
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await ribbon.getByRole("button", { name: "Library" }).click();
  const secondApproxRow = page.locator("table tbody tr", { hasText: "Approx Wasserstein" }).first();
  await expect(secondApproxRow).toBeVisible();
  await secondApproxRow.getByRole("button", { name: "Configure" }).click();
  await page.getByLabel("Embedding Dimension").fill("1");
  const preselectedPageRankRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Embedding - PageRank" });
  const unselectedDegreeRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Embedding - Degree Centrality" });
  await expect(preselectedPageRankRow.locator("input[type='checkbox']")).toBeChecked();
  await expect(unselectedDegreeRow.locator("input[type='checkbox']")).not.toBeChecked();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();

  const approxEmbeddingRows = page.locator("table tbody tr", { hasText: "Tiny Embedding - Approx Wasserstein Embedding" });
  await expect(approxEmbeddingRows).toHaveCount(2);
  await expect(page.locator(".selection-panel .sel-context")).toContainText("Context: Embedding Tiny Embedding - Approx Wasserstein Embedding");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Embeddings" }).locator(".sel-count")).toHaveText("2");
  await approxEmbeddingRows.nth(0).locator("input[type='checkbox']").check();
  await approxEmbeddingRows.nth(1).locator("input[type='checkbox']").check();
  await page.getByRole("button", { name: "Run Selected" }).click();
  await expect(page.locator(".jobs-panel")).toContainText("neext.compute_graph_embeddings", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing features:", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing embedding", { timeout: 30_000 });
  await expect(approxEmbeddingRows.nth(0).locator(".status-pill")).toHaveText("completed", { timeout: 60_000 });
  await expect(approxEmbeddingRows.nth(1).locator(".status-pill")).toHaveText("completed", { timeout: 60_000 });
  await approxEmbeddingRows.nth(0).getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Embedding - Approx Wasserstein Embedding Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  await page.locator(".embedding-explore").getByRole("button", { name: "Data", exact: true }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("graph_id");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("emb_0");
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(2);
});

test("Models library configures, runs, batches, and previews persisted artifacts", async ({ page }) => {
  test.setTimeout(180_000);
  const projectName = `Model Project ${Date.now()}`;
  const seeded = seedTinyModelProject(projectName);

  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.getByRole("button", { name: "MODELS" }).click();
  const ribbon = page.locator(".ribbon");
  const inspector = page.locator(".inspector-panel");
  const modelAnalysisGroup = ribbon.locator(".tool-group", { hasText: "Model Analysis" });
  await expect(modelAnalysisGroup.getByRole("button", { name: "Explore" })).toBeVisible();

  await expect(page.locator(".artifact-table-title")).toContainText("Models");
  await expect(page.locator(".artifact-table-empty")).toContainText("No models.");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Embeddings" }).locator(".sel-count")).toHaveText("0");
  await expect(ribbon.getByRole("button", { name: "Create" })).toHaveCount(0);

  await modelAnalysisGroup.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Model Explore");
  await expect(page.locator(".artifact-table-empty")).toContainText("No models.");

  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Select a Dataset");
  await expect(page.locator(".artifact-table-empty")).toContainText("Select a dataset in the Left Panel before configuring models.");
  await expect(page.locator(".artifact-table .tbl")).toHaveCount(0);

  await page
    .locator(".selection-panel .sel-section", { hasText: "Datasets" })
    .locator(".sel-item", { hasText: "Tiny Model" })
    .click();
  await page
    .locator(".selection-panel .sel-section", { hasText: "Embeddings" })
    .locator(".sel-item", { hasText: "Tiny Model - Approx Wasserstein Embedding" })
    .nth(0)
    .click();
  await expect(page.locator(".selection-panel .sel-context")).toContainText("Context: Embedding Tiny Model - Approx Wasserstein Embedding");
  await page.getByRole("button", { name: "MODELS" }).click();
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.getByText("Dataset: Tiny Model")).toBeVisible();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Algorithm", "Output", "Actions"]);
  const randomForestRow = page.locator("table tbody tr", { hasText: "Random Forest" }).first();
  await expect(randomForestRow).toBeVisible();
  await randomForestRow.click();
  await expect(randomForestRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Model Details");
  await expect(inspector).toContainText("random_forest");
  await expect(inspector).toContainText("neext.train_graph_model");

  await randomForestRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure Random Forest" })).toBeVisible();
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeEnabled();
  await expect(page.getByLabel("Task Type")).toHaveValue("classifier");
  await expect(page.getByLabel("Sample Size")).toHaveValue("5");
  await expect(page.getByLabel("Test Size")).toHaveValue("0.3");
  await page.getByLabel("Sample Size").fill("1");
  await page.getByLabel("Test Size").fill("0.5");

  const firstEmbeddingPickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Model - Approx Wasserstein Embedding" }).nth(0);
  const secondEmbeddingPickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Model - Approx Wasserstein Embedding" }).nth(1);
  await expect(firstEmbeddingPickerRow.locator("input[type='checkbox']")).toBeChecked();
  await expect(secondEmbeddingPickerRow.locator("input[type='checkbox']")).not.toBeChecked();
  await secondEmbeddingPickerRow.click();
  await expect(page.getByText("Dataset: Tiny Model")).toBeVisible();
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeEnabled();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();

  const modelRow = page.locator("table tbody tr", { hasText: "Tiny Model - Random Forest Classifier" }).first();
  await expect(modelRow).toBeVisible();
  await expect(modelRow).toContainText("Tiny Model");
  await expect(modelRow).toContainText("Random Forest");
  await expect(modelRow).toContainText("Classifier");
  await expect(modelRow).toContainText("planned");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Models" }).locator(".sel-count")).toHaveText("1");
  await modelRow.click();
  await expect(inspector).toContainText("Model Details");
  await expect(inspector).toContainText("Tiny Model - Random Forest Classifier");
  await expect(inspector).toContainText("Tiny Model");
  await expect(inspector).toContainText("Sample Size");
  await expect(inspector).toContainText("Expected Metrics");

  await modelAnalysisGroup.getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Model - Random Forest Classifier Explore");
  await expect(page.locator(".artifact-table-empty")).toContainText("Run model training before exploring this model.");
  await ribbon.getByRole("button", { name: "Models" }).click();

  await modelRow.getByRole("button", { name: "Run" }).click();
  await expect(page.locator(".jobs-panel")).toContainText("neext.train_graph_model", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing upstream embeddings for model", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing embedding", { timeout: 45_000 });
  await expect(page.locator(".cmd")).toContainText("Training model", { timeout: 90_000 });
  await expect(modelRow.locator(".status-pill")).toHaveText("completed", { timeout: 120_000 });
  await expect(modelRow.getByRole("button", { name: "Preview" })).toBeVisible();
  await modelRow.getByRole("button", { name: "Preview" }).click();
  const exploreView = page.locator(".model-explore");
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Model - Random Forest Classifier Explore");
  await expect(ribbon.getByRole("button", { name: "Explore" })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "Choose Model" })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Overview", exact: true })).toHaveClass(/is-active/);
  await expect(exploreView.getByRole("button", { name: "Metrics", exact: true })).toBeVisible();
  await expect(exploreView.getByRole("button", { name: "Data", exact: true })).toBeVisible();
  await expect(exploreView).toContainText("Metric Summary");
  await expect(exploreView).toContainText("Random Forest");
  await expect(inspector).toContainText("Metrics File");
  await expect(inspector).toContainText("Model File");

  await exploreView.getByRole("button", { name: "Metrics", exact: true }).click();
  const chart = exploreView.locator(".model-metrics-chart");
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toBeVisible();
  const metricPointPosition = await chart.evaluate((element) => {
    const chartElement = element as HTMLDivElement & {
      __modelMetricsChart?: {
        getOption: () => {
          series?: { data?: { iteration: number; metric: string; value: [number, number] }[] }[];
        };
        convertToPixel: (finder: { seriesIndex: number }, value: [number, number]) => [number, number];
      };
    };
    const chartInstance = chartElement.__modelMetricsChart;
    if (!chartInstance) throw new Error("Model metrics chart instance is not attached.");
    const option = chartInstance.getOption();
    const datum = option.series?.[0]?.data?.[0];
    if (!datum) throw new Error("Model metrics chart has no plotted points.");
    if (datum.iteration !== 0) throw new Error("Model metrics chart did not preserve iteration metadata.");
    const [x, y] = chartInstance.convertToPixel({ seriesIndex: 0 }, datum.value);
    return { x, y };
  });
  await chart.click({ position: metricPointPosition });
  await expect(inspector).toContainText("Model Iteration Details");
  await expect(inspector).toContainText("Iteration");
  await expect(inspector).toContainText("Accuracy");

  await exploreView.getByRole("button", { name: "Data", exact: true }).click();
  await expect(exploreView.locator(".artifact-table-scroll .tbl thead")).toContainText("Accuracy");
  const metricsDataRow = exploreView.locator(".artifact-table-scroll .tbl tbody tr").first();
  await expect(metricsDataRow).toBeVisible();
  await metricsDataRow.click();
  await expect(inspector).toContainText("Model Iteration Details");

  await exploreView.getByRole("button", { name: "Choose Model" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Model Explore");
  await expect(page.locator("table tbody tr", { hasText: "Tiny Model - Random Forest Classifier" })).toBeVisible();
  await ribbon.getByRole("button", { name: "Models" }).click();

  for (let index = 0; index < 2; index += 1) {
    const response = await page.request.post(`/api/projects/${seeded.project_id}/models`, {
      data: {
        source_model_id: "random_forest",
        source_embedding_ids: seeded.embedding_ids,
        params: {
          task_type: "classifier",
          sample_size: 1,
          test_size: 0.5,
          balance_dataset: false,
          n_jobs: 1,
          parallel_backend: "thread"
        }
      }
    });
    expect(response.ok()).toBeTruthy();
  }

  await page.reload();
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page
    .locator(".selection-panel .sel-section", { hasText: "Datasets" })
    .locator(".sel-item", { hasText: "Tiny Model" })
    .click();
  await page.getByRole("button", { name: "MODELS" }).click();
  await ribbon.getByRole("button", { name: "Models" }).click();
  const modelRows = page.locator("table tbody tr", { hasText: "Tiny Model - Random Forest Classifier" });
  await expect(modelRows).toHaveCount(3);
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Models" }).locator(".sel-count")).toHaveText("3");
  await modelRows.nth(0).locator("input[type='checkbox']").check();
  await modelRows.nth(1).locator("input[type='checkbox']").check();
  await page.getByRole("button", { name: "Run Selected" }).click();
  await expect(page.locator(".jobs-panel")).toContainText("neext.train_graph_model", { timeout: 20_000 });
  await expect(modelRows.nth(0).locator(".status-pill")).toHaveText("completed", { timeout: 90_000 });
  await expect(modelRows.nth(1).locator(".status-pill")).toHaveText("completed", { timeout: 90_000 });
});

test("removed workflow controls are absent", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("Generate Synthetic Dataset")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Import CSV" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "NetworkX" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Visualize" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Importance" })).toHaveCount(0);
  await expect(page.getByText("Default n_jobs")).toHaveCount(0);
  await expect(page.getByText("Clear visible log")).toHaveCount(0);
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await expect(page.locator(".ribbon").getByRole("button", { name: "Create" })).toHaveCount(0);
});
