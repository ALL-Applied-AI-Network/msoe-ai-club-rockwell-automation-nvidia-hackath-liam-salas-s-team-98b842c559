import { expect, test } from "@playwright/test";

const shots = "test-results/screenshots";

async function revealThrough(page: import("@playwright/test").Page, seconds: number) {
  await page.locator("video").evaluate(async (video, time) => {
    await new Promise<void>(resolve => video.readyState >= 2 ? resolve() : video.addEventListener("canplay", () => resolve(), { once: true }));
    video.pause();
    video.currentTime = time;
    video.dispatchEvent(new Event("timeupdate"));
  }, seconds);
}

test("development operator view stays honest across cue boundaries", async ({ page }) => {
  await page.goto("/");
  await page.addStyleTag({ content: "*, *::before, *::after { animation: none !important; transition: none !important; caret-color: transparent !important; } .header-actions button { border-radius: 0 !important; }" });
  await expect(page.getByRole("heading", { level: 1, name: /development data/i })).toBeVisible();
  const video = page.getByRole("region", { name: "Live run" }).locator("video");
  await expect(video).toBeVisible();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.readyState)).toBeGreaterThanOrEqual(2);
  await expect(video.locator("source")).toHaveAttribute("src", /ec609e6a200991921a25784646aa0b0bad0c4b9c7d1c3680a391f5e247b3b07f\.mp4$/);
  await expect(page.getByRole("region", { name: "Vial timeline" }).getByRole("button")).toHaveCount(18);
  await expect(page.getByText("waiting for top-down feed", { exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  });
  // Keep native controls available in normal runtime; move focus away and let them fade for a clean capture.
  await page.mouse.move(1900, 20);
  await page.waitForTimeout(2500);
  // Chromium keeps controls visible while this paused capture is focused; hide only the native
  // shadow controls in the test screenshot. The application keeps `controls` in normal mode.
  await page.addStyleTag({ content: "video::-webkit-media-controls { display: none !important; }" });
  await page.screenshot({ path: `${shots}/01-development-operator.png` });
  await revealThrough(page, 15.1);
  await expect(page.getByRole("button", { name: /select vial slot 2/i })).toHaveText(/insufficient_evidence/);
  await expect(page.getByRole("button", { name: "Select vial slot 1", exact: true })).toHaveText(/insufficient_evidence/);
  await expect(page.getByText(/18 \/ 18 revealed/)).toHaveCount(0);
  await revealThrough(page, 269.9);
  await expect(page.getByText("18 / 18 revealed")).toBeVisible();
  await expect(page.locator(".vial-strip .decision-chip--insufficient_evidence")).toHaveCount(18);
  await expect(page.locator(".vial-strip .decision-chip--keep")).toHaveCount(0);
  await expect(page.locator(".decision-chip--discard")).toHaveCount(0);
  await expect(page.locator(".decision-chip--keep")).toHaveCount(0);
  await expect(page.locator(".vial-strip .decision-chip--insufficient_evidence")).toHaveCount(18);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920);
});

test("development recording mode autoplays and remains all abstentions", async ({ page }) => {
  await page.goto("/?demo=1");
  await expect(page.getByRole("heading", { level: 1, name: /development data/i })).toBeVisible();
  const video = page.getByRole("region", { name: "Live run" }).locator("video");
  await expect(video).toHaveAttribute("autoplay", "");
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.muted)).toBe(true);
  await revealThrough(page, 30.1);
  await expect(page.getByText("3 / 18 revealed")).toBeVisible();
  await expect(page.locator(".decision-chip--discard")).toHaveCount(0);
  await page.screenshot({ path: `${shots}/02-development-recording.png` });
  await revealThrough(page, 269.9);
  await expect(page.getByText("18 / 18 revealed")).toBeVisible();
  await expect(page.locator(".decision-chip--discard")).toHaveCount(0);
});

test("narrow layout remains reachable", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 900 });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: /development data/i })).toBeVisible();
  const vial = page.getByRole("button", { name: "Select vial slot 1", exact: true });
  await vial.focus();
  await vial.press("Enter");
  await expect(vial).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(900);
});
