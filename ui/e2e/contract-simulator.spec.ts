import { expect, test } from "@playwright/test";

test.describe("TEST-ONLY runner contract simulator", () => {
  test("renders all four screens and the alarm drawer from simulator events", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Test-only service", { exact: true })).toBeVisible();
    await expect(page.getByLabel(/^Pipeline commit [0-9a-f]{40}$/)).toBeVisible();
    await page.getByLabel("Operator run ID").fill("test-run-1");
    await page.getByLabel("Tray ID").fill("tray-test-1");
    await page.getByLabel("Video file").setInputFiles({ name: "dummy-test-fixture.avi", mimeType: "video/x-msvideo", buffer: Buffer.from("not decoded") });
    await page.getByRole("button", { name: "Start live inspection" }).click();
    await expect(page.getByRole("status")).toHaveText("Run completed", { timeout: 10_000 });
    await expect(page.getByText(/Pipeline error|Event stream stopped|commit mismatch|Illegal invocation/i)).toHaveCount(0);

    // L2 station
    const current = page.getByRole("region", { name: "Current vial" });
    await expect(current.getByText("Discard", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("region", { name: "Tray" }).getByRole("button").first()).toBeVisible();
    await current.getByRole("button", { name: "Agree" }).click();
    await expect(current.getByText(/the system decision is unchanged/i)).toBeVisible();

    // Alarm drawer
    await page.getByRole("button", { name: /^Alarms \(1\)$/ }).click();
    const drawer = page.getByRole("dialog", { name: "Alarm list" });
    await drawer.getByRole("button", { name: "Acknowledge" }).first().click();
    await expect(page.getByRole("button", { name: /^Alarms \(0\)$/ })).toBeVisible();
    await page.keyboard.press("Escape");

    // L3, L1, L4
    await page.getByRole("button", { name: /Vial detail/ }).click();
    await expect(page.getByRole("region", { name: "Model sources" })).toBeVisible();
    await page.getByRole("button", { name: /Line overview/ }).click();
    await expect(page.getByRole("region", { name: /ΔE trend/ })).toBeVisible();
    await page.getByRole("button", { name: /^Engineer$/ }).click();
    await page.getByRole("button", { name: /Diagnostics/ }).click();
    await expect(page.getByRole("region", { name: "Event log" })).toBeVisible();

    await page.getByRole("button", { name: /Inspection station/ }).click();
    await page.screenshot({ path: "test-results/screenshots/redesign-test-only-light-1920x1080.png" });
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.screenshot({ path: "test-results/screenshots/redesign-test-only-dark-1920x1080.png" });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920);
  });
});
