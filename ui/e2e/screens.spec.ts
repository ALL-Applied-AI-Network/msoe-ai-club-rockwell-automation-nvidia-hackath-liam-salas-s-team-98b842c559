import { expect, type Page, test } from "@playwright/test";

const label = process.env.SCREENS_LABEL ?? "runner";
const shot = (page: Page, name: string, theme: string) =>
  page.screenshot({ path: `test-results/screenshots/redesign-${label}-${name}-${theme}-1920x1080.png` });

async function seekVideo(page: Page) {
  const video = page.locator("video");
  await expect(video).toHaveCount(1);
  await video.evaluate(el => new Promise<void>(resolve => {
    const v = el as HTMLVideoElement;
    const seek = () => { v.addEventListener("seeked", () => resolve(), { once: true }); v.currentTime = 11.6; };
    if (v.readyState >= 1) seek(); else v.addEventListener("loadedmetadata", seek, { once: true });
  }));
  await page.waitForTimeout(300);
}

test("four screens and alarm drawer against a real runner or recorded replay", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/Service verified|Service ok/)).toBeVisible({ timeout: 30_000 });
  await page.getByLabel("Operator run ID").fill(`ui-screens-${label}`);
  await page.getByLabel("Tray ID").fill("DEV-V4-TRAIN-01");
  await page.getByLabel("Recipe ID (optional)").fill("5");
  await page.getByLabel("Video file").setInputFiles(process.env.SCREENS_VIDEO!);
  await page.getByRole("button", { name: "Start live inspection" }).click();
  await expect(page.getByRole("status")).toHaveText(/^(Run completed|Run stopped)$/, { timeout: 8 * 60 * 1000 });
  await expect(page.locator(".failure-banner")).toHaveCount(0);
  for (const theme of ["light", "dark"]) {
    if (theme === "dark") await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await page.getByRole("button", { name: /Inspection station/ }).click();
    await expect(page.getByRole("region", { name: "Current vial" })).toBeVisible();
    await seekVideo(page);
    await shot(page, "L2-station", theme);
    await page.getByRole("button", { name: /Vial detail/ }).click();
    await expect(page.getByRole("region", { name: "Model sources" })).toBeVisible();
    await page.waitForLoadState("networkidle");
    await shot(page, "L3-detail", theme);
    await page.getByRole("button", { name: /Line overview/ }).click();
    await expect(page.getByRole("region", { name: "ΔE trend" })).toBeVisible();
    await shot(page, "L1-overview", theme);
    if (theme === "light") await page.getByRole("button", { name: /^Engineer$/ }).click();
    await page.getByRole("button", { name: /Diagnostics/ }).click();
    await expect(page.getByRole("region", { name: "Event log" })).toBeVisible();
    await shot(page, "L4-diagnostics", theme);
    await page.getByRole("button", { name: /^Alarms \(/ }).click();
    await expect(page.getByRole("dialog", { name: "Alarm list" })).toBeVisible();
    await shot(page, "alarms", theme);
    await page.keyboard.press("Escape");
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920);
});
