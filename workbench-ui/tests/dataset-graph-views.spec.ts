import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const SINGLE_GRAPH_NODES_CSV =
  "node_id,role\n" + "101,left\n" + "102,right\n" + "103,left\n" + "104,right\n" + "105,left\n" + "106,right\n" + "107,left\n" + "108,right\n";
const SINGLE_GRAPH_EDGES_CSV =
  "src_node_id,dest_node_id\n" + "101,102\n" + "102,103\n" + "103,104\n" + "104,105\n" + "105,106\n" + "106,107\n" + "107,108\n";

function largePathGraphEdgesCsv(nodeCount: number): string {
  const rows = ["src_node_id,dest_node_id"];
  for (let nodeId = 1; nodeId < nodeCount; nodeId += 1) {
    rows.push(`${nodeId},${nodeId + 1}`);
  }
  return `${rows.join("\n")}\n`;
}

async function createProjectViaApi(page: Page, name: string): Promise<string> {
  const response = await page.request.post("/api/projects", { data: { name, description: "graph views e2e" } });
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { id: string }).id;
}

async function seedPreparedDataset(page: Page, projectId: string, intakePayload: Record<string, unknown>): Promise<string> {
  const created = await page.request.post(`/api/projects/${projectId}/dataset-intake/create`, { data: intakePayload });
  expect(created.ok()).toBeTruthy();
  const datasetId = ((await created.json()) as { id: string }).id;
  const run = await page.request.post(`/api/projects/${projectId}/datasets/${datasetId}/run`);
  expect(run.ok()).toBeTruthy();
  const jobId = ((await run.json()) as { id: string }).id;
  await expect
    .poll(
      async () => {
        const jobResponse = await page.request.get(`/api/projects/${projectId}/jobs/${jobId}`);
        return ((await jobResponse.json()) as { status: string }).status;
      },
      { timeout: 40_000 }
    )
    .toBe("completed");
  return datasetId;
}

async function openGraphTab(page: Page, projectName: string, datasetName: string) {
  await page.goto("/");
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.getByRole("button", { name: "DATASETS" }).click();
  const datasetRow = page.locator("table tbody tr", { hasText: datasetName }).first();
  await expect(datasetRow).toBeVisible();
  await datasetRow.getByRole("button", { name: "Preview" }).click();
  const exploreView = page.locator(".dataset-explore");
  await exploreView.getByRole("button", { name: "Graph", exact: true }).click();
  return exploreView;
}

test("Egonet dataset graph views: stable toolbar, Overview centroids, Grid tiles", async ({ page }) => {
  test.setTimeout(60_000);
  const projectName = `Graph Views ${Date.now()}`;
  const projectId = await createProjectViaApi(page, projectName);
  await seedPreparedDataset(page, projectId, {
    name: "Views Graph",
    description: "single graph for view modes",
    source_graph_shape: "single_graph",
    tables: {
      nodes: { format: "csv", csv: SINGLE_GRAPH_NODES_CSV },
      edges: { format: "csv", csv: SINGLE_GRAPH_EDGES_CSV }
    },
    params: { k_hop: 1, node_selection: "all_nodes", target_node_attribute: "role" }
  });

  const exploreView = await openGraphTab(page, projectName, "Views Graph");

  const segControl = exploreView.locator(".seg-control");
  await expect(segControl.getByRole("tab", { name: "Overview" })).toBeVisible();
  await expect(segControl.getByRole("tab", { name: "Grid" })).toBeVisible();
  await expect(segControl.getByRole("tab", { name: "Single" })).toHaveClass(/is-active/);
  await expect(exploreView.locator(".graph-meta-strip .graph-id-badge")).toContainText("Graph");

  const toolbarRow = exploreView.locator(".graph-toolbar-row");
  const boxBefore = await toolbarRow.boundingBox();
  const centerBefore = await exploreView.locator(".graph-center-badge").textContent();
  await exploreView.getByRole("button", { name: "Next graph" }).click();
  await expect.poll(async () => exploreView.locator(".graph-center-badge").textContent()).not.toBe(centerBefore);
  const boxAfter = await toolbarRow.boundingBox();
  expect(boxAfter).toEqual(boxBefore);

  await segControl.getByRole("tab", { name: "Overview" }).click();
  await expect(exploreView.getByRole("button", { name: "Previous graph" })).toBeDisabled();
  await expect(exploreView.getByRole("button", { name: "Next graph" })).toBeDisabled();
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("Source graph");
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("8 nodes");
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("8 centroids");
  await expect(exploreView.locator(".graph-meta-action")).toHaveCount(0);
  await expect(exploreView.getByLabel("Source graph overview")).toBeVisible();

  await segControl.getByRole("tab", { name: "Grid" }).click();
  await expect(exploreView.locator(".graph-tile")).toHaveCount(8);
  await expect(exploreView.locator(".graph-position")).toHaveText("1 / 1");
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("Graphs 1–8 of 8");
  await expect(exploreView.getByRole("button", { name: "Previous page" })).toBeDisabled();
  await expect(exploreView.getByRole("button", { name: "Next page" })).toBeDisabled();

  await exploreView.getByLabel("Filter graphs").fill("3");
  await expect(exploreView.locator(".graph-tile")).toHaveCount(1);
  await expect(exploreView.locator(".graph-tile-caption").first()).toContainText("Graph 3");
  await expect(exploreView.locator(".graph-meta-strip")).toContainText('filtered by "3"');
  await exploreView.getByLabel("Filter graphs").fill("");
  await expect(exploreView.locator(".graph-tile")).toHaveCount(8);

  await exploreView.locator(".graph-tile", { hasText: "Graph 0" }).first().click();
  await expect(segControl.getByRole("tab", { name: "Single" })).toHaveClass(/is-active/);
  await expect(exploreView.locator(".graph-meta-strip .graph-id-badge")).toHaveText("Graph 0");
});

test("Collection dataset shows Grid and Single without Overview", async ({ page }) => {
  test.setTimeout(60_000);
  const projectName = `Collection Views ${Date.now()}`;
  const projectId = await createProjectViaApi(page, projectName);
  await seedPreparedDataset(page, projectId, {
    name: "Views Collection",
    description: "graph collection for view modes",
    tables: {
      node_graph_mapping: { format: "csv", csv: "node_id,graph_id\n1,g1\n2,g1\n3,g2\n4,g2\n" },
      edges: { format: "csv", csv: "src_node_id,dest_node_id\n1,2\n3,4\n" },
      graph_labels: { format: "csv", csv: "graph_id,graph_label\ng1,0\ng2,1\n" }
    },
    params: { graph_type: "networkx", filter_largest_component: false }
  });

  const exploreView = await openGraphTab(page, projectName, "Views Collection");
  const segControl = exploreView.locator(".seg-control");
  await expect(segControl.getByRole("tab", { name: "Single" })).toHaveClass(/is-active/);
  await expect(segControl.getByRole("tab", { name: "Grid" })).toBeVisible();
  await expect(segControl.getByRole("tab", { name: "Overview" })).toHaveCount(0);

  await segControl.getByRole("tab", { name: "Grid" }).click();
  await expect(exploreView.locator(".graph-tile")).toHaveCount(2);
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("Graphs 1–2 of 2");
});

test("Overview samples large source graphs and loads the full graph on demand", async ({ page }) => {
  test.setTimeout(90_000);
  const projectName = `Overview Sampling ${Date.now()}`;
  const projectId = await createProjectViaApi(page, projectName);
  await seedPreparedDataset(page, projectId, {
    name: "Large Graph",
    description: "large single graph for sampled overview",
    source_graph_shape: "single_graph",
    tables: {
      edges: { format: "csv", csv: largePathGraphEdgesCsv(600) }
    },
    params: { k_hop: 1, node_selection: "sample_fraction", sample_fraction: 0.05, random_seed: 13 }
  });

  const exploreView = await openGraphTab(page, projectName, "Large Graph");
  await exploreView.locator(".seg-control").getByRole("tab", { name: "Overview" }).click();
  await expect(exploreView.locator(".graph-meta-strip")).toContainText("600 nodes");
  await expect(exploreView.locator(".graph-meta-note")).toContainText("showing 500 of 600 nodes");
  const loadFull = exploreView.getByRole("button", { name: "Load full graph" });
  await expect(loadFull).toBeVisible();
  await loadFull.click();
  await expect(exploreView.locator(".graph-meta-note")).toHaveCount(0);
  await expect(exploreView.getByRole("button", { name: "Load full graph" })).toHaveCount(0);
});
