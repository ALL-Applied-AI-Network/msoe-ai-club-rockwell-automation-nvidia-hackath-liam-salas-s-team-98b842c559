import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:4173",
    viewport: { width: 1920, height: 1080 },
    colorScheme: "dark",
    reducedMotion: "reduce",
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  expect: { timeout: 10_000 },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
  }
});
