import { expect, test } from "@playwright/test";

test("REAL runner processes the explicitly authorized V3/V4 video", async ({ page }) => {
  const baseUrl = process.env.REAL_RUNNER_BASE_URL;
  const video = process.env.REAL_TWIN_VIDEO;
  if (!baseUrl || !video) {
    throw new Error("REAL E2E requires explicit REAL_RUNNER_BASE_URL and REAL_TWIN_VIDEO; no test-only fallback is permitted.");
  }
  await page.addInitScript(() => { window.fetch = window.fetch.bind(window); });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Test-only service", { exact: true })).toHaveCount(0);
  await page.getByLabel("Operator run ID").fill("real-e2e-run");
  await page.getByLabel("Tray ID").fill("real-e2e-tray");
  await page.getByLabel("Video file").setInputFiles(video);
  await page.getByRole("button", { name: "Start live inspection" }).click();
  await expect(page.getByText(/Measuring/)).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("Waiting for reference measurements.", { exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/Keep|Check|Discard|Insufficient evidence/).first()).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("What this means", { exact: true })).toBeVisible({ timeout: 120_000 });
});
