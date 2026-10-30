import { describe, expect, it } from "vitest";
import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import mockRun from "../../../ui-data/mock-run.json";
import { parseDemoData } from "./adapter";

const PAIRS = [
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V2", 6.699801316215474],
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V3", 6.2970588604995275],
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V2", 6.438614236887922],
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V3", 6.980229689829405],
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V2", 4.494294414282912],
  ["PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V3", 3.7824264172035935],
  ["PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V2", 29.07563861068292],
  ["PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V3", 28.067765245336204],
  ["PAIR-BATCH_5a4a586e3ec3b085-dispense-2-t400-V1-V2", 28.92631630447299]
] as const;

const OUTCOMES = [
  "keep", "keep", "insufficient_evidence", "keep", "keep",
  "keep", "discard", "discard", "discard"
] as const;

describe("committed RUN-01 mock", () => {
  it("binds all nine literal outcomes and measured pair values", () => {
    const data = parseDemoData(mockRun);
    expect(data.vials.map((vial) => vial.slot)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(data.vials.map((vial) => vial.decision.outcome)).toEqual(OUTCOMES);
    expect(data.vials.map((vial) => [
      vial.decision.delta_e.provenance.source_id,
      vial.decision.delta_e.value
    ])).toEqual(PAIRS);
    expect(data.vials[2].decision.limit_codes).toEqual(["mock_quality_gate_unavailable"]);
  });

  it("preserves first and last measured Lab values and exact asset bindings", () => {
    const data = parseDemoData(mockRun);
    expect(data.vials[0].target_lab).toMatchObject({ l: 22.828592642911804, a: 27.388127716053035, b: 18.17593016522053, provenance: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V2-dispense-1-t200" } });
    expect(data.vials[0].reference_lab).toMatchObject({ l: 23.396501671707384, a: 20.72270814823629, b: 18.546063920423567, provenance: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V1-dispense-1-t200" } });
    expect(data.vials[8].target_lab).toMatchObject({ l: 12.23870104478932, a: -4.371363586950144, b: 1.3356067605913458, provenance: { kind: "measured", source_id: "MEAS-BATCH_5a4a586e3ec3b085-V2-dispense-2-t400" } });
    expect(data.vials[8].reference_lab).toMatchObject({ l: 26.54861195009829, a: 14.510343301429335, b: 17.931972923395247, provenance: { kind: "measured", source_id: "MEAS-BATCH_5a4a586e3ec3b085-V1-dispense-2-t400" } });
    expect(data.vials[0].rgb_frame).toMatchObject({ src: "/assets/vials/vial-001-rgb.jpg", provenance: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V2-dispense-1-t200" } });
    expect(data.vials[0].rgb_frame.provenance.note).toContain("c7aad9b956e8682d018f15adb7bdb58c364158c2e678e1e4ca9811dd1ce81fec");
    expect(data.vials[0].predicted_mask).toMatchObject({ src: "/assets/vials/vial-001-mask.png", provenance: { kind: "measured", source_id: "accepted-empty-reference-rgb-predictor" } });
    expect(data.vials[0].predicted_mask.provenance.note).toContain("60fa50193338ce74bad66f245cdddd8764f1c2d2d633e0aa1d81cac98ae5f84c");
  });

  it("gives each evidence target its own correct provenance", () => {
    const data = parseDemoData(mockRun);
    const byTarget = Object.fromEntries(data.vials[0].evidence.map((record) => [record.target, record.provenance]));
    expect(byTarget).toMatchObject({
      rgb_frame: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V2-dispense-1-t200" },
      predicted_mask: { kind: "measured", source_id: "accepted-empty-reference-rgb-predictor" },
      target_lab: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V2-dispense-1-t200" },
      reference_lab: { kind: "measured", source_id: "MEAS-BATCH_0ebf3f287cc1291a-V1-dispense-1-t200" },
      delta_e: { kind: "measured", source_id: PAIRS[0][0] },
      threshold: { kind: "documented", source_id: "phase1-mock-threshold-10" }
    });
    expect(Object.keys(byTarget).sort()).toEqual(["delta_e", "predicted_mask", "reference_lab", "rgb_frame", "target_lab", "threshold"]);
    expect(data.vials[0].recipe.rgb.provenance.source_id).toMatch(/^MEAS-/);
  });

  it("uses measured left identities for inferred mock history", () => {
    const data = parseDemoData(mockRun);
    expect(data.vials[0].history[0]).toMatchObject({ run_id: "RUN-TRAIN-04", vial_id: "V1", recipe_id: "8", decision: "keep", provenance: { kind: "inferred", source_id: PAIRS[0][0] }, time_seconds: { value: 1, provenance: { kind: "inferred", source_id: "run-01-composite" } } });
    expect(data.vials[8].history[0]).toMatchObject({ run_id: "RUN-TRAIN-01", vial_id: "V1", recipe_id: "5", decision: "discard", provenance: { kind: "inferred", source_id: PAIRS[8][0] }, time_seconds: { value: 9 } });
    expect(new Set(data.vials.map((vial) => vial.history[0].time_seconds.value)).size).toBe(9);
    expect(data.vials[0].history[0].explanation.evidence_refs).toEqual([`evidence-${PAIRS[0][0]}-reference_lab`]);
  });

  it("binds the four WP6 rows to exact source files and parser outcomes", () => {
    const data = parseDemoData(mockRun);
    expect(data.wp6.disclaimer).toBe("limited all-abstention benchmark, no accuracy claim");
    expect(data.wp6.rows.map((row) => [row.method, row.passed, row.total, row.provenance.source_id])).toEqual([
      ["template", 16, 16, "template-score.json#sha256=ce036c2af554c201d45f2a600bdde6e301f9566c76e7232324f04553722c5843"],
      ["qwen3_vl", 16, 16, "qwen-score.json#sha256=3cd9ccd37af3453063f4aed803bc0c35ee3e2db5832ad10ef6c107a6df9040f9"],
      ["anomalythink_free_form", 0, 16, "anomalythink-score.json#sha256=a0dfd8aa795f1c6e3e6aad4b7b5b3c3a12f8ec956a1e976f8fdec31bbea52be3"],
      ["cosmos_strict_parse", 0, 16, "cosmos-score.json#sha256=3b986e50d1ec0a7140c9ecff54fee5f94b256234a715876e225625e8b70fd2d9"]
    ]);
    expect(data.wp6.rows[0].method_note).toContain("Template 16/16");
    expect(data.wp6.rows[1].method_note).toContain("Qwen3-VL 16/16");
    expect(data.wp6.rows[2].failure_note).toContain("free-form outputs failed the parser contract");
    expect(data.wp6.rows[3].failure_note).toContain("Markdown fences");
    expect(data.wp6.inn67).toMatchObject({ status: "placeholder", label: "INN-67 placeholder" });
  });

  it("aligns cues to fill sections after entry and preserves transport gaps", () => {
    const data = parseDemoData(mockRun);
    expect(data.playback.video.duration_seconds).toMatchObject({ value: 49, provenance: { kind: "inferred", source_id: "run-01-composite" } });
    expect(data.playback.cues.map((cue) => [cue.start_seconds, cue.end_seconds])).toEqual([
      [1.3, 4.6], [5.9, 9.2], [10.5, 13.8], [15.1, 19.6], [20.9, 25.4],
      [26.7, 31.2], [32.5, 36.7], [38, 42.2], [43.5, 47.7]
    ]);
    expect(data.playback.cues[0].start_seconds).toBeGreaterThan(0);
    expect(data.playback.cues[8].end_seconds).toBeLessThanOrEqual(data.playback.video.duration_seconds.value);
    for (let index = 1; index < data.playback.cues.length; index += 1) {
      expect(data.playback.cues[index].start_seconds - data.playback.cues[index - 1].end_seconds).toBeCloseTo(1.3, 6);
    }
    expect(data.playback.cues.every((cue) => cue.provenance.note.includes("derived local composite"))).toBe(true);
  });

  it("commits all local assets with valid signatures and no absolute source paths", () => {
    const data = parseDemoData(mockRun);
    for (const [index, vial] of data.vials.entries()) {
      const ordinal = String(index + 1).padStart(3, "0");
      expect(vial.rgb_frame.src).toBe(`/assets/vials/vial-${ordinal}-rgb.jpg`);
      expect(vial.predicted_mask.src).toBe(`/assets/vials/vial-${ordinal}-mask.png`);
      const rgb = readFileSync(resolve(process.cwd(), `../ui-data${vial.rgb_frame.src}`));
      const mask = readFileSync(resolve(process.cwd(), `../ui-data${vial.predicted_mask.src}`));
      expect([...rgb.subarray(0, 3)]).toEqual([0xff, 0xd8, 0xff]);
      expect([...mask.subarray(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
      expect(rgb.length).toBeGreaterThan(1_000);
      expect(mask.length).toBeGreaterThan(100);
    }
    expect(statSync(resolve(process.cwd(), "../ui-data/assets/run/run-01-composite.mp4")).size).toBeGreaterThan(5_000_000);
    expect(JSON.stringify(mockRun)).not.toMatch(/\/home\/|\/tmp\//);
  });
});
