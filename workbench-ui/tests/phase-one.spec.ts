import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const TITLE_ONLY_SPACES = ["EMBEDDINGS", "MODELS"] as const;
const COMMANDS = ["Import", "Library", "Create"] as const;

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

test("Embeddings and Models spaces expose structural title-only commands", async ({ page }) => {
  await page.goto("/");

  for (const space of TITLE_ONLY_SPACES) {
    await page.getByRole("button", { name: space }).click();
    const ribbon = page.locator(".ribbon");
    const pluralTitle = space.charAt(0) + space.slice(1).toLowerCase();
    await expect(page.locator(".document .title-only")).toHaveText(pluralTitle);

    for (const command of COMMANDS) {
      await ribbon.getByRole("button", { name: command }).click();
      await expect(page.locator(".document .title-only")).toHaveText(command);
    }

    await ribbon.getByRole("button", { name: pluralTitle }).click();
    await expect(page.locator(".document .title-only")).toHaveText(pluralTitle);
  }
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
  await expect(leftPanelDatasetItem).toBeVisible();
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
  await expect(page.locator(".artifact-table-title")).toContainText("MUTAG Preview");
  await expect(page.getByRole("button", { name: "nodes" })).toHaveClass(/is-active/);
  await expect(page.locator(".artifact-table .tbl tbody tr")).toHaveCount(20);
  await page.getByRole("button", { name: "edges" }).click();
  await expect(page.locator(".artifact-table .tbl thead")).toContainText("src_node_id");
  await page.getByRole("button", { name: "Node Mapping" }).click();
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
