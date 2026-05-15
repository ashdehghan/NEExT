import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
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
  await expect(ribbon.getByRole("button", { name: "Settings" })).toBeVisible();
  await expect(ribbon.getByRole("button", { name: "Help" })).toBeVisible();

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
});

test("Home commands switch center views", async ({ page }) => {
  await page.goto("/");

  const ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Import" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Import");

  await ribbon.getByRole("button", { name: "Settings" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Settings");

  await ribbon.getByRole("button", { name: "Help" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Help");

  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name: "Create Project" })).toBeVisible();

  await ribbon.getByRole("button", { name: "Projects" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Projects");
});

test("Models Import and Create commands remain title-only", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "MODELS" }).click();
  const ribbon = page.locator(".ribbon");
  await expect(page.locator(".artifact-table-title")).toContainText("Models");

  for (const command of TITLE_ONLY_COMMANDS) {
    await ribbon.getByRole("button", { name: command }).click();
    await expect(page.locator(".document .title-only")).toHaveText(command);
  }
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
});

test("Datasets and Features run through planned artifacts and jobs", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");

  await page.getByRole("button", { name: "FEATURES" }).click();
  let ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Type", "Output", "Actions"]);
  const initialPageRankRow = page.locator("table tbody tr", { hasText: "PageRank" }).first();
  await expect(initialPageRankRow).toBeVisible();
  await initialPageRankRow.click();
  await expect(page.locator(".inspector-panel")).toContainText("Catalog Feature Details");
  await expect(page.locator(".inspector-panel")).toContainText("neext.compute_node_features");
  await initialPageRankRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure PageRank" })).toBeVisible();
  await expect(page.getByLabel("Dataset")).toBeVisible();
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeDisabled();
  await expect(page.getByText("An active project is required.")).toBeVisible();

  await page.getByRole("button", { name: "DATASETS" }).click();
  ribbon = page.locator(".ribbon");
  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Source", "Size", "Status", "Actions"]);
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
  await ribbon.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".document .title-only")).toHaveText("Create");

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
  await expect(page.locator(".document .title-only")).toHaveText("Create");

  await ribbon.getByRole("button", { name: "Library" }).click();
  await expect(page.locator(".artifact-table .tbl thead th")).toHaveText(["Name", "Type", "Output", "Actions"]);
  const pageRankRow = page.locator("table tbody tr", { hasText: "PageRank" }).first();
  await expect(pageRankRow).toBeVisible();
  await pageRankRow.click();
  await expect(pageRankRow).toHaveClass(/is-selected/);
  await expect(inspector).toContainText("Catalog Feature Details");
  await expect(inspector).toContainText("PageRank scores with neighborhood aggregation.");
  await expect(inspector).toContainText("neext.compute_node_features");

  await pageRankRow.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByRole("heading", { name: "Configure PageRank" })).toBeVisible();
  const datasetSelect = page.getByLabel("Dataset");
  await expect(datasetSelect).toBeVisible();
  await expect(datasetSelect).toContainText("MUTAG");
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
  await expect(page.locator(".artifact-table-title")).toContainText("MUTAG - PageRank Preview");
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
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Features" }).locator(".sel-count")).toHaveText("2");

  await ribbon.getByRole("button", { name: "Library" }).click();
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

  await ribbon.getByRole("button", { name: "Library" }).click();
  const secondApproxRow = page.locator("table tbody tr", { hasText: "Approx Wasserstein" }).first();
  await expect(secondApproxRow).toBeVisible();
  await secondApproxRow.getByRole("button", { name: "Configure" }).click();
  await page.getByLabel("Embedding Dimension").fill("1");
  await page.locator(".feature-picker table tbody tr", { hasText: "Tiny Embedding - PageRank" }).click();
  await page.locator(".card-foot").getByRole("button", { name: "Save" }).click();

  const approxEmbeddingRows = page.locator("table tbody tr", { hasText: "Tiny Embedding - Approx Wasserstein Embedding" });
  await expect(approxEmbeddingRows).toHaveCount(2);
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Embeddings" }).locator(".sel-count")).toHaveText("2");
  await approxEmbeddingRows.nth(0).locator("input[type='checkbox']").check();
  await approxEmbeddingRows.nth(1).locator("input[type='checkbox']").check();
  await page.getByRole("button", { name: "Run Selected" }).click();
  await expect(page.locator(".jobs-panel")).toContainText("neext.compute_graph_embeddings", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing upstream features for embedding", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing embedding", { timeout: 30_000 });
  await expect(approxEmbeddingRows.nth(0).locator(".status-pill")).toHaveText("completed", { timeout: 60_000 });
  await expect(approxEmbeddingRows.nth(1).locator(".status-pill")).toHaveText("completed", { timeout: 60_000 });
  await approxEmbeddingRows.nth(0).getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Embedding - Approx Wasserstein Embedding Preview");
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

  await expect(page.locator(".artifact-table-title")).toContainText("Models");
  await expect(page.locator(".artifact-table-empty")).toContainText("No models.");
  await expect(page.locator(".selection-panel .sel-section", { hasText: "Embeddings" }).locator(".sel-count")).toHaveText("2");

  await ribbon.getByRole("button", { name: "Library" }).click();
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
  await expect(page.locator(".card-foot").getByRole("button", { name: "Save" })).toBeDisabled();
  await expect(page.getByLabel("Task Type")).toHaveValue("classifier");
  await expect(page.getByLabel("Sample Size")).toHaveValue("5");
  await expect(page.getByLabel("Test Size")).toHaveValue("0.3");
  await page.getByLabel("Sample Size").fill("1");
  await page.getByLabel("Test Size").fill("0.5");

  const firstEmbeddingPickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Model - Approx Wasserstein Embedding" }).nth(0);
  const secondEmbeddingPickerRow = page.locator(".feature-picker table tbody tr", { hasText: "Tiny Model - Approx Wasserstein Embedding" }).nth(1);
  await firstEmbeddingPickerRow.click();
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

  await modelRow.getByRole("button", { name: "Run" }).click();
  await expect(page.locator(".jobs-panel")).toContainText("neext.train_graph_model", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing upstream embeddings for model", { timeout: 20_000 });
  await expect(page.locator(".cmd")).toContainText("Computing embedding", { timeout: 45_000 });
  await expect(page.locator(".cmd")).toContainText("Training model", { timeout: 90_000 });
  await expect(modelRow.locator(".status-pill")).toHaveText("completed", { timeout: 120_000 });
  await expect(modelRow.getByRole("button", { name: "Preview" })).toBeVisible();
  await modelRow.getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Tiny Model - Random Forest Classifier Results");
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("Accuracy");
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(1);
  await expect(inspector).toContainText("Metrics File");
  await expect(inspector).toContainText("Model File");

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
});
