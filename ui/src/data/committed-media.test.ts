import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { resolveFfmpeg } from "./ffmpeg-resolver";

describe("committed RUN-01 media", () => {
  it("resolves ffmpeg from a declared cross-platform project dependency", () => {
    const resolver = resolve(process.cwd(), "src/data/ffmpeg-resolver.ts");
    expect(existsSync(resolver), "Missing project ffmpeg resolver").toBe(true);
    const source = readFileSync(resolver, "utf8");
    expect(source).toContain('from "ffmpeg-static"');
    expect(source).not.toContain(["/", "tmp", "/"].join(""));
    expect(source).not.toContain(["aar", "ch64"].join(""));
    expect(source).not.toContain("process.env.HOME");
  });

  it("fails clearly for a missing dependency or invalid explicit override", () => {
    expect(() => resolveFfmpeg({}, null)).toThrow(
      "Install UI dev dependencies with `npm ci` in ui/ to provide ffmpeg-static."
    );
    expect(() => resolveFfmpeg({ FLUIDMIX_FFMPEG: "/definitely/missing/ffmpeg" }, undefined)).toThrow(
      "FLUIDMIX_FFMPEG does not exist: /definitely/missing/ffmpeg"
    );
  });

  it("fully decodes with the accepted browser playback metadata", () => {
    const ffmpeg = resolveFfmpeg();
    const media = resolve(process.cwd(), "../ui-data/assets/run/run-01-composite.mp4");
    const result = spawnSync(ffmpeg, ["-v", "info", "-i", media, "-f", "null", "-"], {
      encoding: "utf8"
    });
    expect(result.status, result.stderr).toBe(0);
    expect(result.stderr).toContain("Duration: 00:00:49.00");
    expect(result.stderr).toMatch(/Video: h264 .*yuv420p.*640x480.*10 fps/);
    expect(result.stderr).not.toContain("Audio:");
    expect(result.stderr).toMatch(/frame=\s*490\b/);
    expect(result.stderr).toContain("time=00:00:49.00");
  });
});
