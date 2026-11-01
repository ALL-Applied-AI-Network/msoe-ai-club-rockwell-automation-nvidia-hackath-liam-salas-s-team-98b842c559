import { expect, type Locator, type Page, test } from "@playwright/test";

const FINAL_RESULT_TIMEOUT = 25 * 60 * 1000;
const FAILURE_BANNERS = /Pipeline error|Event stream stopped|commit mismatch|Service request failed|Run failed/i;

test.describe.configure({ mode: "serial" });

async function expectProductionService(page: Page) {
  await expect(page.getByText("Service verified · healthy", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Test-only service", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Pipeline commit")).toHaveText(/^[0-9a-f]{40}$/);
  await expect(page.getByText("Development data, frozen-cohort hybrid, service 2344f3c", { exact: true })).toBeVisible();
}

async function startVideoRun(page: Page, file: string | { name: string; mimeType: string; buffer: Buffer }, suffix: string) {
  await page.getByLabel("Operator run ID").fill(`real-e2e-${suffix}`);
  await page.getByLabel("Tray ID").fill(`real-e2e-${suffix}`);
  await page.getByLabel("Recipe ID (optional)").fill("5");
  await page.getByLabel("Video file").setInputFiles(file);
  await page.getByRole("button", { name: "Start live inspection" }).click();
}

async function expectTerminalLifecycle(page: Page) {
  await expect(page.getByRole("status")).toHaveText(/^(Run completed|Run stopped)$/, { timeout: FINAL_RESULT_TIMEOUT });
  await expect(page.locator(".failure-banner")).toHaveCount(0);
  await expect(page.getByText(FAILURE_BANNERS)).toHaveCount(0);
}

async function waitForResultOrTerminal(page: Page, result: Locator) {
  const terminal = page.getByRole("status").filter({ hasText: /^(Run completed|Run stopped|Run failed)$/ });
  await expect(result.or(terminal).first()).toBeVisible({ timeout: FINAL_RESULT_TIMEOUT });
}

async function expectInViewport(locator: Locator) {
  const box = await locator.boundingBox();
  expect(box, "expected screenshot evidence to have a rendered bounding box").not.toBeNull();
  expect(box!.y, "expected all screenshot evidence below the top edge").toBeGreaterThanOrEqual(0);
  expect(box!.y + box!.height, "expected all screenshot evidence above the bottom edge").toBeLessThanOrEqual(1080);
}

async function expectVisibleThroughClipping(locator: Locator) {
  const visibleRatio = await locator.evaluate(element => {
    const elementBox = element.getBoundingClientRect();
    let top = Math.max(0, elementBox.top);
    let right = Math.min(window.innerWidth, elementBox.right);
    let bottom = Math.min(window.innerHeight, elementBox.bottom);
    let left = Math.max(0, elementBox.left);
    for (let ancestor = element.parentElement; ancestor; ancestor = ancestor.parentElement) {
      const style = getComputedStyle(ancestor);
      if (![style.overflow, style.overflowX, style.overflowY].some(value => /auto|clip|hidden|scroll/.test(value))) continue;
      const ancestorBox = ancestor.getBoundingClientRect();
      top = Math.max(top, ancestorBox.top);
      right = Math.min(right, ancestorBox.right);
      bottom = Math.min(bottom, ancestorBox.bottom);
      left = Math.max(left, ancestorBox.left);
    }
    const visibleArea = Math.max(0, right - left) * Math.max(0, bottom - top);
    return visibleArea / Math.max(1, elementBox.width * elementBox.height);
  });
  expect(visibleRatio, "expected screenshot evidence to be visible through every clipping ancestor").toBeGreaterThanOrEqual(0.95);
}

async function fitScreenshotEvidence(page: Page, evidence: readonly Locator[]) {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await evidence[0].evaluate(element => element.scrollIntoView({ block: "start" }));
    const boxes = await Promise.all(evidence.map(locator => locator.boundingBox()));
    if (boxes.every(box => box && box.y >= 0 && box.y + box.height <= 1080)) return;
    await page.keyboard.press("Control+-");
  }
  for (const locator of evidence) await expectInViewport(locator);
}

test("REAL runner processes the explicitly authorized V4 video", async ({ page }) => {
  const baseUrl = process.env.REAL_RUNNER_BASE_URL;
  const video = process.env.REAL_TWIN_VIDEO;
  if (!baseUrl || !video) {
    throw new Error("REAL E2E requires explicit REAL_RUNNER_BASE_URL and REAL_TWIN_VIDEO; no test-only fallback is permitted.");
  }

  await page.addInitScript(() => { window.fetch = window.fetch.bind(window); });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expectProductionService(page);
  await startVideoRun(page, video, "accepted-v4");

  const alarm = page.getByRole("alert").filter({ hasText: /Discard alarm/ });
  await waitForResultOrTerminal(page, alarm);
  await expect(alarm).toBeVisible({ timeout: 1_000 });
  await expect(alarm.getByRole("button", { name: "Acknowledge" })).toBeVisible();

  const tray = page.getByRole("region", { name: "Tray order" });
  const trayVials = tray.getByRole("button", { name: /Select tray position/ });
  await expect(trayVials).toHaveCount(3, { timeout: FINAL_RESULT_TIMEOUT });
  await expect(tray).toContainText("3 vials");
  for (const [index, vial] of (await trayVials.all()).entries()) {
    await expect(vial).toHaveAccessibleName(new RegExp(`^Select tray position ${index + 1}, Vial ${index + 1},`));
  }

  const discardVial = tray.getByRole("button", { name: /Select tray position .*Discard/ }).first();
  await expect(discardVial).toBeVisible();
  await discardVial.click();

  const vialDetails = page.getByRole("region", { name: "Vial details" });
  await expect(vialDetails.locator(".large-state")).toContainText("Discard");
  await vialDetails.getByText("Engineer details", { exact: true }).click();

  const modelVotes = vialDetails.getByRole("region", { name: "Model votes" });
  const voteParagraphs = modelVotes.locator("p");
  await expect(voteParagraphs).toHaveCount(3);
  for (const vote of await voteParagraphs.all()) {
    await expect(vote.locator("strong")).toBeVisible();
    await expect(vote.locator(".source-tag")).toBeVisible();
    await expect(vote).toContainText(/· Admitted \(admitted\) · discard(?: ·|$)/);
    await expect(vote).not.toContainText("Not admitted");
  }
  await expect(vialDetails).toContainText(/Average \(equal weight\)\s*0\.90/s);
  await expect(vialDetails).toContainText(/Lab \(CIELAB; L\* 0–100\)\s*L\* -?\d+\.\d · a\* -?\d+\.\d · b\* -?\d+\.\d/s);
  await expect(vialDetails).toContainText(/Tracker volume\s*\d+\.\d ml/s);
  await expect(vialDetails).toContainText(/Delta E \(current event stage\)\s*Not provided for .*runner event does not expose stage ΔE\./s);
  await expect(vialDetails).toContainText(/Case ID\s*CASE_/s);

  const operatorExplanation = vialDetails.locator(".operator-explanation");
  await expect(operatorExplanation).toContainText(/Qwen/i);
  await expect(operatorExplanation).toContainText("Model-written");
  const explanationSources = vialDetails.getByRole("region", { name: "Explanation sources" });
  const fallbackLabels = explanationSources.getByText("Deterministic fallback", { exact: true });
  await expect(fallbackLabels).toHaveCount(2);
  for (const label of await fallbackLabels.all()) await expect(label).toBeVisible();
  await expect(page.locator(".failure-banner")).toHaveCount(0);

  await voteParagraphs.first().evaluate(element => element.scrollIntoView({ block: "center" }));
  await fitScreenshotEvidence(page, [alarm, modelVotes, tray]);
  await expectInViewport(alarm);
  await expectInViewport(modelVotes);
  await expectInViewport(tray);
  for (const vote of await voteParagraphs.all()) await expectVisibleThroughClipping(vote);
  await page.screenshot({ path: "test-results/screenshots/phase3-real-development-2344f3c-dark-1920x1080.png" });

  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.getByRole("button", { name: "Switch to dark theme" })).toBeVisible();
  await fitScreenshotEvidence(page, [alarm, modelVotes, tray]);
  await expectInViewport(alarm);
  await expectInViewport(modelVotes);
  await expectInViewport(tray);
  for (const vote of await voteParagraphs.all()) await expectVisibleThroughClipping(vote);
  await page.screenshot({ path: "test-results/screenshots/phase3-real-development-2344f3c-light-1920x1080.png" });

  await alarm.getByRole("button", { name: "Acknowledge" }).click();
  await expect(alarm).toHaveCount(0);
  await expectTerminalLifecycle(page);
  await expect(trayVials.nth(1)).toHaveAccessibleName(/Vial 2, Not assessed \(run stopped\)$/);
  await expect(trayVials.nth(2)).toHaveAccessibleName(/Vial 3, Not assessed \(run stopped\)$/);
});

test("REAL runner abstains for an unrecognized video source", async ({ page }) => {
  if (!process.env.REAL_RUNNER_BASE_URL || !process.env.REAL_TWIN_VIDEO) {
    throw new Error("REAL E2E requires explicit REAL_RUNNER_BASE_URL and REAL_TWIN_VIDEO; no test-only fallback is permitted.");
  }

  await page.addInitScript(() => { window.fetch = window.fetch.bind(window); });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expectProductionService(page);
  await startVideoRun(page, {
    name: "unrecognized-camera-profile.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("arbitrary unrecognized video bytes")
  }, "unrecognized-source");

  const vialDetails = page.getByRole("region", { name: "Vial details" });
  const insufficientEvidence = vialDetails.locator(".large-state").filter({ hasText: "Insufficient evidence" });
  await waitForResultOrTerminal(page, insufficientEvidence);
  await expect(insufficientEvidence).toBeVisible({ timeout: 1_000 });
  await expect(vialDetails.getByText("Can't assess this video: camera setup not recognised", { exact: true })).toBeVisible();
  await vialDetails.getByText("Engineer details", { exact: true }).click();
  await expect(vialDetails).toContainText(/Reason\s*tracker_source_profile_unavailable/s);
  await expect(vialDetails.getByRole("region", { name: "Model votes" })).toContainText(/^Model votes\s*Unavailable$/s);
  await expect(vialDetails).toContainText(/Average \(equal weight\)\s*Unavailable/s);
  await expect(vialDetails.getByRole("region", { name: "Explanation sources" })).toContainText(/^Explanation sources\s*Unavailable$/s);
  await expect(vialDetails.locator(".operator-explanation")).toHaveCount(0);
  await expect(page.getByText("Test-only service", { exact: true })).toHaveCount(0);
  await expectTerminalLifecycle(page);
});
