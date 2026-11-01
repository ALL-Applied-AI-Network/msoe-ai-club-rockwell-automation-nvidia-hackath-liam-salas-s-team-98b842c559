import { expect, test } from "@playwright/test";

test.describe("TEST-ONLY runner contract simulator", () => {
  test.use({ colorScheme: "dark" });

  test("renders event-derived HMI, alarm acknowledgement, locator and engineer detail", async ({ page }) => {
    await page.addInitScript(() => { window.fetch = window.fetch.bind(window); });
    await page.goto("/");
    await expect(page.getByText("Test-only service", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Pipeline commit")).toHaveText(/^[0-9a-f]{40}$/);
    await page.getByLabel("Operator run ID").fill("test-run-1");
    await page.getByLabel("Tray ID").fill("tray-test-1");
    await page.getByLabel("Video file").setInputFiles({ name: "dummy-test-fixture.avi", mimeType: "video/x-msvideo", buffer: Buffer.from("not decoded") });
    await page.getByRole("button", { name: "Start live inspection" }).click();

    const eventTypes = await page.evaluate(async () => {
      const response = await fetch("http://127.0.0.1:5173/api/sessions/session-1/events?scenario=error");
      const body = await response.text();
      return ["measuring", "waiting_for_references", "vial_decision", "model_vote", "averaged_decision", "explanation", "rate_health"].filter(type => body.includes(`event: ${type}`));
    });
    expect(eventTypes).toEqual(["measuring", "waiting_for_references", "vial_decision", "model_vote", "averaged_decision", "explanation", "rate_health"]);
    await expect(page.getByText("Discard", { exact: false }).first()).toBeVisible();
    await expect(page.getByText("Deterministic fixture explanation for rendering tests.", { exact: true })).toBeVisible();
    await expect(page.getByRole("img", { name: /normalized bounding box/ })).toBeVisible();
    await page.getByRole("button", { name: /Acknowledge/ }).click();
    await expect(page.getByRole("button", { name: /Acknowledge/ })).toHaveCount(0);
    await page.getByText("Engineer details", { exact: true }).click();
    await expect(page.getByRole("heading", { name: "Model votes" })).toBeVisible();
    await expect(page.getByText(/Deterministic_test_fixture/i)).toBeVisible();
    await expect(page.getByText("fixture-model", { exact: false })).toBeVisible();

    await page.screenshot({ path: "test-results/screenshots/phase3-test-only-dark-1920x1080.png" });
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await expect(page.getByRole("button", { name: "Switch to dark theme" })).toBeVisible();
    await page.screenshot({ path: "test-results/screenshots/phase3-test-only-light-1920x1080.png" });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920);
  });
});
