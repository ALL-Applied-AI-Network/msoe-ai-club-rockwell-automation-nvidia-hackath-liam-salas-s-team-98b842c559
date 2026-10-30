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

test("operator dashboard evidence flow", async ({ page }) => {
  await page.goto("/");
  await page.addStyleTag({ content: "*, *::before, *::after { animation: none !important; transition: none !important; caret-color: transparent !important; }" });
  await expect(page.getByRole("heading", { name: /synthetic twin data/i })).toBeVisible();
  const video = page.getByRole("region", { name: "Live run" }).locator("video");
  await expect(video).toBeVisible();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.readyState)).toBeGreaterThanOrEqual(2);
  await video.evaluate(async node => { node.pause(); if (Math.abs(node.currentTime) > 0.01) await new Promise<void>(resolve => { node.addEventListener("seeked", () => resolve(), { once: true }); node.currentTime = 0; }); });
  // Keep the verified video visible and add a deterministic local frame over it for byte-stable deliverable captures.
  await video.evaluate(node => {
    const parent = node.parentElement;
    if (!parent) return;
    parent.style.position = "relative";
    const poster = document.createElement("img");
    poster.src = "/assets/vials/vial-001-rgb.jpg";
    poster.alt = "Deterministic local run frame";
    poster.dataset.testPoster = "true";
    poster.style.cssText = `position:absolute;left:${node.offsetLeft}px;top:${node.offsetTop}px;width:${node.offsetWidth}px;height:${node.offsetHeight}px;object-fit:contain;background:#020608;z-index:1;pointer-events:none`;
    parent.appendChild(poster);
  });
  await expect(video.locator("source")).toHaveAttribute("src", /run-01-composite\.mp4$/);
  await expect(page.getByRole("region", { name: "Vial timeline" }).getByRole("button")).toHaveCount(9);
  await expect(page.getByText("waiting for top-down feed", { exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  });
  await page.screenshot({ path: `${shots}/01-live-run.png` });

  const pendingTopDown = page.getByRole("button", { name: /top-down position.*slot 7/i });
  await expect(pendingTopDown).toHaveAccessibleName(/awaiting decision/);
  await revealThrough(page, 32.6);
  await expect(page.getByRole("button", { name: /select vial slot 7/i })).toHaveText(/DISCARD/);
  await expect(page.getByRole("article", { name: "Vial slot 7 discard explanation" })).toBeVisible();
  await expect(page.getByRole("button", { name: /top-down position.*slot 7.*discard.*Delta E/i })).toBeVisible();
  await page.getByRole("button", { name: /select vial slot 7/i }).press("Enter");
  await expect(page.getByRole("button", { name: /top-down position.*slot 7.*selected/i })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("region", { name: "Vial detail" })).toContainText(/Delta E|Threshold|Limit codes/);
  const detailImages = page.getByRole("region", { name: "Vial detail" }).locator("img");
  await expect(detailImages).toHaveCount(2);
  for (const image of await detailImages.all()) {
    await expect.poll(() => image.evaluate((node: HTMLImageElement) => node.naturalWidth)).toBeGreaterThan(0);
  }
  await page.screenshot({ path: `${shots}/02-vial-detail.png` });

  await revealThrough(page, 44);
  await expect(page.getByRole("region", { name: "Retained discard explanations" }).getByRole("article")).toHaveCount(3);
  for (const slot of [7, 8, 9]) await expect(page.getByRole("article", { name: `Vial slot ${slot} discard explanation` })).toBeVisible();
  await page.screenshot({ path: `${shots}/03-discard-explanations.png` });

  const evidence = page.getByRole("button", { name: /Evidence .*predicted_mask/i }).first();
  await evidence.click();
  await expect(page.getByRole("img", { name: /vial 7 predicted mask overlay/i })).toHaveAttribute("data-evidence-highlight", "true");
  await page.getByRole("button", { name: "Open evidence panel" }).click();
  const dialog = page.getByRole("dialog", { name: "Evidence panel" });
  await expect(dialog).toContainText(/Reference records/);
  await page.screenshot({ path: `${shots}/04-history-panel.png` });
  await page.getByRole("tab", { name: "History" }).press("ArrowRight");
  await expect(page.getByRole("table")).toBeVisible();
  await expect(page.getByRole("table")).toContainText("limited all-abstention benchmark, no accuracy claim");
  await expect(page.getByRole("table").locator("tbody tr")).toHaveCount(4);
  await expect(dialog).toContainText("INN-67");
  await page.screenshot({ path: `${shots}/05-model-pane.png` });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Open evidence panel" })).toBeFocused();

  await page.getByRole("button", { name: /Switch to light theme/i }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByRole("button", { name: /Switch to dark theme/i })).toBeVisible();
  await page.screenshot({ path: `${shots}/06-light-theme.png` });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920);
});

test("narrow layout remains reachable", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 900 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /synthetic twin data/i })).toBeVisible();
  const vial = page.getByRole("button", { name: /select vial slot 1/i });
  await vial.focus();
  await vial.press("Enter");
  await expect(vial).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(900);
});
