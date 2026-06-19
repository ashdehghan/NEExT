import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "../sandbox/playwright-test-results",
  timeout: 30_000,
  expect: {
    timeout: 5_000
  },
  use: {
    baseURL: "http://127.0.0.1:8877",
    trace: "on-first-retry"
  },
  webServer: {
    command: "cd .. && python -m NEExT.workbench.cli --workspace sandbox/workbench-e2e --port 8877 --no-browser",
    url: "http://127.0.0.1:8877",
    reuseExistingServer: !process.env.CI,
    timeout: 20_000
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chrome" }
    }
  ]
});
