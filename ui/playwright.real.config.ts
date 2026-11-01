import { defineConfig } from "@playwright/test";

const baseUrl = process.env.REAL_RUNNER_BASE_URL;
if (!baseUrl) throw new Error("REAL E2E requires explicit REAL_RUNNER_BASE_URL; refusing test-only fallback.");
if (!process.env.REAL_TWIN_VIDEO) throw new Error("REAL E2E requires explicit REAL_TWIN_VIDEO.");

export default defineConfig({
  testDir: "./e2e",
  testMatch: /real-runner\.spec\.ts/,
  timeout: 30 * 60 * 1000,
  expect: { timeout: 20 * 60 * 1000 },
  workers: 1,
  outputDir: "test-results/playwright",
  use: { baseURL: "http://127.0.0.1:5173", viewport: { width: 1920, height: 1080 }, colorScheme: "dark", headless: true },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173",
    env: { ...process.env, VITE_RUNNER_BASE_URL: baseUrl },
    url: "http://127.0.0.1:5173",
    reuseExistingServer: false,
  },
});
