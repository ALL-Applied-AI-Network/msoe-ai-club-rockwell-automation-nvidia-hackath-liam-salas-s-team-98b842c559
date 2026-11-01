import { defineConfig } from "@playwright/test";

const baseUrl = process.env.REAL_RUNNER_BASE_URL;
if (!baseUrl) throw new Error("REAL E2E requires explicit REAL_RUNNER_BASE_URL; refusing test-only fallback.");
if (!process.env.REAL_TWIN_VIDEO) throw new Error("REAL E2E requires explicit REAL_TWIN_VIDEO.");

export default defineConfig({
  testDir: "./e2e",
  testMatch: /real-runner\.spec\.ts/,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:4173", viewport: { width: 1920, height: 1080 }, colorScheme: "dark", headless: true },
  webServer: {
    command: `VITE_RUNNER_BASE_URL=${baseUrl} npm run dev -- --host 127.0.0.1 --port 4173`,
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
  },
});
