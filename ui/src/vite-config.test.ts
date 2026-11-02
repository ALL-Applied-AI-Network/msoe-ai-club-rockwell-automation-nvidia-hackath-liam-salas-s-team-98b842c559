import type { UserConfig } from "vite";
import { describe, expect, it } from "vitest";

import viteConfig from "../vite.config";

describe("Vite runner proxy", () => {
  it("forwards same-origin API requests to the local runner service", () => {
    const config = viteConfig as UserConfig;

    expect(config.server?.proxy?.["/api"]).toMatchObject({
      target: "http://127.0.0.1:8766",
    });
  });
});
