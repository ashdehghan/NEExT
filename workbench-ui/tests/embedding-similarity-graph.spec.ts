import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const SINGLE_GRAPH_NODES_CSV =
  "node_id,role\n" + "101,left\n" + "102,right\n" + "103,left\n" + "104,right\n" + "105,left\n" + "106,right\n" + "107,left\n" + "108,right\n";
const SINGLE_GRAPH_EDGES_CSV =
  "src_node_id,dest_node_id\n" + "101,102\n" + "102,103\n" + "103,104\n" + "104,105\n" + "105,106\n" + "106,107\n" + "107,108\n";
const COLLECTION_EDGES_CSV = "src_node_id,dest_node_id\n" + "1,2\n" + "2,3\n" + "4,5\n" + "5,6\n";
const COLLECTION_MAPPING_CSV = "node_id,graph_id\n" + "1,g1\n" + "2,g1\n" + "3,g1\n" + "4,g2\n" + "5,g2\n" + "6,g2\n";

async function createProjectViaApi(page: Page, name: string): Promise<string> {
  const response = await page.request.post("/api/projects", { data: { name, description: "similarity graph e2e" } });
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { id: string }).id;
}

async function waitForJob(page: Page, projectId: string, jobId: string) {
  await expect
    .poll(
      async () => {
        const jobResponse = await page.request.get(`/api/projects/${projectId}/jobs/${jobId}`);
        return ((await jobResponse.json()) as { status: string }).status;
      },
      { timeout: 60_000 }
    )
    .toBe("completed");
}

async function seedCompletedEmbedding(page: Page, projectId: string, intakePayload: Record<string, unknown>): Promise<string> {
  const created = await page.request.post(`/api/projects/${projectId}/dataset-intake/create`, { data: intakePayload });
  expect(created.ok()).toBeTruthy();
  const datasetId = ((await created.json()) as { id: string }).id;

  const featureCreated = await page.request.post(`/api/projects/${projectId}/features`, {
    data: {
      source_dataset_id: datasetId,
      source_feature_id: "page_rank",
      params: { feature_vector_length: 2, normalize_features: false, n_jobs: 1, parallel_backend: "threading" }
    }
  });
  expect(featureCreated.ok()).toBeTruthy();
  const featureId = ((await featureCreated.json()) as { id: string }).id;

  const embeddingCreated = await page.request.post(`/api/projects/${projectId}/embeddings`, {
    data: {
      source_embedding_id: "approx_wasserstein",
      source_feature_ids: [featureId],
      params: { embedding_dimension: 2 }
    }
  });
  expect(embeddingCreated.ok()).toBeTruthy();
  const embeddingId = ((await embeddingCreated.json()) as { id: string }).id;

  // Embedding run auto-runs the Draft dataset preparation and feature computation upstream.
  const run = await page.request.post(`/api/projects/${projectId}/embeddings/${embeddingId}/run`);
  expect(run.ok()).toBeTruthy();
  await waitForJob(page, projectId, ((await run.json()) as { id: string }).id);
  return embeddingId;
}

async function openEmbeddingExplore(page: Page, projectName: string, datasetName: string, embeddingName: string) {
  await page.goto("/");
  // Activate the seeded project explicitly (auto-select can tie-break to another
  // same-second project), then select the dataset branch: embedding lists are dataset-first.
  await page.getByRole("button", { name: "HOME" }).click();
  await page.locator(".ribbon").getByRole("button", { name: "Projects" }).click();
  await page.locator("table tbody tr", { hasText: projectName }).first().click();
  await expect(page.locator(".selection-panel .sel-item-name", { hasText: projectName })).toBeVisible();
  await page.locator(".selection-panel").getByRole("button", { name: `${datasetName} Dataset` }).click();
  await page.getByRole("button", { name: "EMBEDDINGS" }).click();
  await page.locator(".ribbon").getByRole("button", { name: "Explore" }).click();
  await expect(page.locator(".artifact-table-title")).toContainText("Embedding Explore");
  await page.locator("table tbody tr", { hasText: embeddingName }).first().getByRole("button", { name: "Explore" }).click();
}

test("Embedding similarity Graph tab: reference modes, threshold, ground truth", async ({ page }) => {
  test.setTimeout(120_000);
  const projectName = `Similarity Graph ${Date.now()}`;
  const projectId = await createProjectViaApi(page, projectName);
  await seedCompletedEmbedding(page, projectId, {
    name: "Similarity Source",
    description: "single graph for similarity view",
    source_graph_shape: "single_graph",
    tables: {
      nodes: { format: "csv", csv: SINGLE_GRAPH_NODES_CSV },
      edges: { format: "csv", csv: SINGLE_GRAPH_EDGES_CSV }
    },
    params: { k_hop: 1, node_selection: "all_nodes", target_node_attribute: "role" }
  });

  await openEmbeddingExplore(page, projectName, "Similarity Source", "Similarity Source - Approx Wasserstein Embedding");
  const graphTab = page.locator(".tab-strip").getByRole("button", { name: "Graph" });
  await expect(graphTab).toBeEnabled();
  await graphTab.click();

  const panel = page.locator(".similarity-graph-panel");
  await expect(panel).toBeVisible();
  await expect(panel.locator(".similarity-hint")).toContainText("Click a node");
  await expect(panel.locator(".similarity-chip-empty")).toHaveText("No reference selected");
  await expect(panel.locator(".similarity-stats")).toContainText("embedded");

  // Node mode via search: set reference 101.
  await panel.getByLabel("Reference node id").fill("101");
  await panel.getByRole("button", { name: "Set", exact: true }).click();
  await expect(panel.locator(".similarity-chip", { hasText: "101" })).toBeVisible();
  await expect(panel.locator(".similarity-hint")).toHaveCount(0);
  await expect(panel.locator(".similarity-stats span", { hasText: "reference" })).toContainText("1");

  // Selection mode: build a two-node reference.
  await panel.getByRole("tab", { name: "Selection" }).click();
  await expect(panel.locator(".similarity-chip-empty")).toBeVisible();
  await panel.getByLabel("Reference node id").fill("101");
  await panel.getByRole("button", { name: "Add", exact: true }).click();
  await panel.getByLabel("Reference node id").fill("103");
  await panel.getByRole("button", { name: "Add", exact: true }).click();
  await expect(panel.locator(".similarity-chip")).toHaveCount(2);

  // Scaling + threshold controls respond.
  await panel.getByRole("tab", { name: "Percentile" }).click();
  await expect(panel.getByRole("tab", { name: "Percentile" })).toHaveClass(/is-active/);
  await panel.getByLabel("Similarity threshold").fill("0.6");
  await expect(panel.locator(".similarity-threshold-value")).toHaveText("0.60");

  // Label-seeded sample with ground truth.
  await panel.getByRole("tab", { name: "Label", exact: true }).click();
  await expect(panel.getByLabel("Label value")).toBeVisible();
  await panel.getByLabel("Label value").selectOption("left");
  await panel.getByLabel("Sample fraction").fill("0.5");
  await panel.getByRole("button", { name: "Apply" }).click();
  await expect(panel.locator(".similarity-chip")).toHaveCount(2);
  const truthToggle = panel.getByLabel("Show ground truth");
  await expect(truthToggle).toBeEnabled();
  await truthToggle.check();

  // Clear returns to the unreferenced state.
  await panel.getByRole("button", { name: "Clear reference" }).click();
  await expect(panel.locator(".similarity-chip-empty")).toBeVisible();
});

test("Embedding similarity Graph tab is disabled for graph-collection embeddings", async ({ page }) => {
  test.setTimeout(120_000);
  const projectName = `Similarity Gating ${Date.now()}`;
  const projectId = await createProjectViaApi(page, projectName);
  await seedCompletedEmbedding(page, projectId, {
    name: "Collection Source",
    description: "graph collection for gating",
    tables: {
      edges: { format: "csv", csv: COLLECTION_EDGES_CSV },
      node_graph_mapping: { format: "csv", csv: COLLECTION_MAPPING_CSV }
    }
  });

  await openEmbeddingExplore(page, projectName, "Collection Source", "Collection Source - Approx Wasserstein Embedding");
  const graphTab = page.locator(".tab-strip").getByRole("button", { name: "Graph" });
  await expect(graphTab).toBeDisabled();
  await expect(graphTab).toHaveAttribute("title", /single-graph egonet datasets only/);
});
