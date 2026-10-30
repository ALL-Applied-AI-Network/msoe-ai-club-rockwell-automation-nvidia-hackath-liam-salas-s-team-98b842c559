import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const PORTABLE_FFMPEG = "/tmp/frontend-pydeps/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2";

describe("committed RUN-01 media", () => {
  it("fully decodes with the accepted browser playback metadata", () => {
    const explicit = process.env.FLUIDMIX_FFMPEG;
    const ffmpeg = explicit ?? PORTABLE_FFMPEG;
    expect(
      existsSync(ffmpeg),
      explicit
        ? `FLUIDMIX_FFMPEG does not exist: ${explicit}`
        : `Portable ffmpeg is unavailable; set FLUIDMIX_FFMPEG (checked ${PORTABLE_FFMPEG})`
    ).toBe(true);
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
