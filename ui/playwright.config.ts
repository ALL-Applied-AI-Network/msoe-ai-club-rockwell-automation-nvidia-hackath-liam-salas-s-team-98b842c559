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
    launchOptions: { args: ["--disable-features=IsolateOrigins"] },
  },
  expect: { timeout: 10_000 },
  projects: [
    { name: "test-only", testMatch: /contract-simulator\.spec\.ts/ },
    { name: "real-service", testMatch: /real-runner\.spec\.ts/, use: { colorScheme: "dark" } }
  ],
  webServer: [
    {
      command: "python3 e2e/test-only-server.py",
      url: "http://127.0.0.1:5173/api/health",
      reuseExistingServer: false,
      timeout: 10_000,
    },
    {
      command: "VITE_RUNNER_BASE_URL=http://127.0.0.1:5173 npm run dev -- --host 127.0.0.1 --port 4173",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: false,
    }
  ]
});
