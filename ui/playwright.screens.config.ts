import { defineConfig } from "@playwright/test";

/**
 * Screenshot run of all four screens against a REAL runner or the development recorded replay
 * (ui/tools/replay_runner.py). Never uses the test-only simulator.
 *   SCREENS_RUNNER_BASE_URL  runner origin, e.g. http://127.0.0.1:8765 (live) or :8775 (replay)
 *   SCREENS_VIDEO            development video to upload and play locally
 *   SCREENS_LABEL            file-name label, e.g. "replay-2344f3c" or "live-10c106f"
 *   VITE_DEV_ALLOWED_COMMITS optional unpinned-commit override (banner is shown)
 */
const baseUrl = process.env.SCREENS_RUNNER_BASE_URL;
if (!baseUrl) throw new Error("Screens run requires SCREENS_RUNNER_BASE_URL.");
if (!process.env.SCREENS_VIDEO) throw new Error("Screens run requires SCREENS_VIDEO.");

export default defineConfig({
  testDir: "./e2e",
  testMatch: /screens\.spec\.ts/,
  timeout: 10 * 60 * 1000,
  workers: 1,
  outputDir: "test-results/playwright",
  use: { baseURL: "http://127.0.0.1:5173", viewport: { width: 1920, height: 1080 }, colorScheme: "light", headless: true },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173 --strictPort",
    env: { ...process.env, VITE_RUNNER_BASE_URL: baseUrl },
    url: "http://127.0.0.1:5173",
    reuseExistingServer: false,
  },
});
