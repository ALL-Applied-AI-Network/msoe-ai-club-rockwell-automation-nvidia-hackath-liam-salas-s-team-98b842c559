import { describe, expect, it } from "vitest";
import { statSync } from "node:fs";
import { resolve } from "node:path";
import mockRun from "../../../ui-data/mock-run.json";
import { parseDemoData } from "./adapter";

describe("committed RUN-01 mock", () => {
  it("contains the nine traceable WP5 vials and documented mock decisions", () => {
    const data = parseDemoData(mockRun);
    expect(data.vials).toHaveLength(9);
    expect(data.vials.map((v) => v.slot)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(data.vials.map((v) => v.decision.outcome)).toEqual([
      "keep", "keep", "insufficient_evidence", "keep", "keep",
      "keep", "discard", "discard", "discard"
    ]);
    expect(data.vials.every((v) => v.decision.explanation.source === "deterministic_template")).toBe(true);
    expect(data.vials.every((v) => v.decision.explanation.evidence_refs.length > 0)).toBe(true);
    expect(data.top_down.observations).toHaveLength(9);
    expect(data.playback.cues).toHaveLength(9);
    expect(data.wp6.disclaimer).toBe("limited all-abstention benchmark, no accuracy claim");
    expect(data.wp6.inn67).toMatchObject({ status: "placeholder", label: "INN-67 placeholder" });
    expect(data.playback.video.duration_seconds.value).toBeGreaterThan(35);
    expect(data.playback.cues[8].end_seconds).toBeGreaterThan(data.playback.cues[0].start_seconds);
    expect(statSync(resolve(process.cwd(), "../ui-data/assets/run/run-01-composite.mp4")).size).toBeGreaterThan(20000);
  });
});
