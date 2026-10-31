import fixture from "../../../ui-data/example.json";
import { describe, expect, expectTypeOf, it } from "vitest";

import { DemoDataValidationError, parseDemoData } from "./adapter";

function copyFixture(): Record<string, any> {
  return structuredClone(fixture);
}

describe("parseDemoData", () => {
  it("accepts the documented synthetic twin fixture", () => {
    const data = parseDemoData(fixture);

    expect(data.dataset_id).toBe("phase1-synthetic-demo");
  });

  it("accepts a variable-count final v2 record with unavailable comparison values and WP7 codes", () => {
    const input = copyFixture();
    const vial = input.vials[0];
    input.label = "Final WP7 data";
    input.data_origin = "wp7_final";
    input.source = { mode: "final", origin: "wp7_final", label: "Final WP7 data" };
    input.vials = [vial];
    input.top_down.observations = [input.top_down.observations[0]];
    input.run.cue_times = [input.run.cue_times[0]];
    input.history = [];
    vial.history_refs = [];
    input.evidence = [{
      evidence_id: "citation-decision-001-1",
      vial_id: vial.vial_id,
      target: "stage_citation",
      label: "target stage 1",
      provenance: {
        kind: "documented",
        source_id: "source-stage-evidence-id",
        note: "Exact WP7 citation identifier."
      }
    }];
    input.wp6 = {
      disclaimer: "WP6 benchmark unavailable for this source",
      rows: [],
      inn67: { status: "unavailable", label: "INN-67 unavailable", note: "Not supplied." }
    };
    vial.target_lab = null;
    vial.reference_lab = null;
    vial.decision.outcome = "insufficient_evidence";
    vial.decision.delta_e = null;
    vial.decision.threshold = null;
    vial.decision.limit_codes = ["arbitrary_honest_wp7_limit_code"];
    vial.decision.evidence_refs = ["citation-decision-001-1"];
    vial.decision.explanation.evidence_refs = ["citation-decision-001-1"];

    const data = parseDemoData(input);

    expect(data.vials).toHaveLength(1);
    expect(data.vials[0].target_lab).toBeNull();
    expect(data.vials[0].delta_e).toBeNull();
    expect(data.vials[0].decision.limit_codes).toEqual([
      "arbitrary_honest_wp7_limit_code"
    ]);
    expect(data.source).toEqual(input.source);
    expect(data.wp6).toMatchObject({
      disclaimer: "WP6 benchmark unavailable for this source",
      inn67: { status: "unavailable", label: "INN-67 unavailable" }
    });
    expectTypeOf(data.wp6.disclaimer).toEqualTypeOf<string>();
    expectTypeOf(data.wp6.inn67.status).toEqualTypeOf<string>();
    expectTypeOf(data.wp6.inn67.label).toEqualTypeOf<string>();
  });

  it("rejects duplicate vial identifiers", () => {
    const input = copyFixture();
    input.vials.push(structuredClone(input.vials[0]));

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError('Duplicate vial_id "vial-001" in vials')
    );
  });

  it("rejects dangling evidence references", () => {
    const input = copyFixture();
    input.vials[0].decision.evidence_refs = ["evidence-missing"];

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Decision "decision-001" references missing evidence "evidence-missing"'
      )
    );
  });

  it("rejects a top-down observation linked to another decision", () => {
    const input = copyFixture();
    input.top_down.observations[0].decision_id = "decision-other";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Top-down observation for "vial-001" links decision "decision-other"; expected "decision-001"'
      )
    );
  });

  it("rejects a feed state without a top-down asset", () => {
    const input = copyFixture();
    input.top_down.state = "feed";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError("Top-down feed state requires an asset")
    );
  });

  it("rejects a waiting state that references a top-down asset", () => {
    const input = copyFixture();
    input.top_down.asset = {
      asset_id: "unexpected",
      src: "/assets/top-down/unexpected.png",
      provenance: { kind: "documented", source_id: "test", note: "Test asset." }
    };

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError("Top-down waiting state must not reference an asset")
    );
  });

  it("rejects a normalized bounding box that extends past the image", () => {
    const input = copyFixture();
    input.top_down.observations[0].bbox.x = 0.95;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Bounding box for "vial-001" extends outside normalized image bounds'
      )
    );
  });

  it.each([
    {
      name: "top-down vial_id",
      mutate(input: Record<string, any>) {
        input.top_down.observations.push(
          structuredClone(input.top_down.observations[0])
        );
      },
      message: 'Duplicate vial_id "vial-001" in top_down.observations'
    },
    {
      name: "evidence_id",
      mutate(input: Record<string, any>) {
        input.evidence.push(structuredClone(input.evidence[0]));
      },
      message: 'Duplicate evidence_id "evidence-delta-vial-001" in evidence'
    }
  ])("rejects duplicate $name values", ({ mutate, message }) => {
    const input = copyFixture();
    mutate(input);

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(message)
    );
  });

  it("rejects a top-down observation for an unknown vial", () => {
    const input = copyFixture();
    input.top_down.observations[0].vial_id = "vial-missing";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Top-down observation references unknown vial_id "vial-missing"'
      )
    );
  });

  it("rejects evidence assigned to a different vial", () => {
    const input = copyFixture();
    input.evidence[0].vial_id = "vial-other";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Decision "decision-001" for "vial-001" references evidence "evidence-delta-vial-001" owned by "vial-other"'
      )
    );
  });

  it("rejects an evidence record for an unknown vial", () => {
    const input = copyFixture();
    const orphan = structuredClone(input.evidence[0]);
    orphan.evidence_id = "evidence-orphan";
    orphan.vial_id = "vial-missing";
    input.evidence.push(orphan);

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Evidence "evidence-orphan" references unknown vial_id "vial-missing"'
      )
    );
  });

  it("rejects a normalized position outside its vial bounding box", () => {
    const input = copyFixture();
    input.top_down.observations[0].position.x = 0.9;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Position for "vial-001" lies outside its normalized bounding box'
      )
    );
  });

  it.each([
    {
      name: "decision outcome",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.outcome = "rework";
      }
    },
    {
      name: "explanation source",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.explanation.source = "free_text_model";
      }
    },
    {
      name: "provenance kind",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.delta_e.provenance.kind = "assumed";
      }
    }
  ])("rejects an unsupported $name", ({ mutate }) => {
    const input = copyFixture();
    mutate(input);

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects evidence-bearing numbers without provenance", () => {
    const input = copyFixture();
    delete input.top_down.observations[0].bbox.provenance;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects zero-area normalized geometry", () => {
    const input = copyFixture();
    input.top_down.observations[0].bbox.width = 0;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects data not labelled as synthetic twin data", () => {
    const input = copyFixture();
    input.label = "Operator demo data";

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it.each([
    ["keep", "within_delta_e_limit", 3, 3],
    ["discard", "exceeds_delta_e_limit", 3.01, 3],
    ["insufficient_evidence", "measurement_unavailable", 0, 3]
  ])(
    "accepts the valid %s decision contract",
    (outcome, limitCode, deltaE, threshold) => {
      const input = copyFixture();
      input.vials[0].decision.outcome = outcome;
      input.vials[0].decision.limit_codes = [limitCode];
      input.vials[0].decision.delta_e.value = deltaE;
      input.vials[0].decision.threshold.value = threshold;

      expect(parseDemoData(input).vials[0].decision.outcome).toBe(outcome);
    }
  );

  it("treats equality as within the limit and keep", () => {
    const input = copyFixture();
    input.vials[0].decision.delta_e.value = 3;
    input.vials[0].decision.threshold.value = 3;

    expect(parseDemoData(input).vials[0].decision.outcome).toBe("keep");
  });

  it.each([
    ["keep", "exceeds_delta_e_limit", 1, 3],
    ["discard", "within_delta_e_limit", 4, 3],
    ["insufficient_evidence", "within_delta_e_limit", 1, 3],
    ["keep", "within_delta_e_limit", 3.01, 3],
    ["discard", "exceeds_delta_e_limit", 3, 3]
  ])("rejects contradictory outcome/limit/numeric contract %#", (outcome, limitCode, deltaE, threshold) => {
    const input = copyFixture();
    input.vials[0].decision.outcome = outcome;
    input.vials[0].decision.limit_codes = [limitCode];
    input.vials[0].decision.delta_e.value = deltaE;
    input.vials[0].decision.threshold.value = threshold;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it.each(["deterministic_template", "cosmos_reason2"])(
    "accepts the supported %s explanation source",
    (source) => {
      const input = copyFixture();
      input.vials[0].decision.explanation.source = source;

      expect(parseDemoData(input).vials[0].decision.explanation.source).toBe(
        source
      );
    }
  );

  it("maps all nine ordered vial slots into complete view models", () => {
    const input = copyFixture();
    input.vials.reverse();
    const data = parseDemoData(input);

    expect(data.vials).toHaveLength(9);
    expect(data.vials.map((vial) => vial.slot)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9
    ]);
    expect(data.vials[0]).toMatchObject({
      vial_id: "vial-001",
      recipe: {
        recipe_id: "sim-recipe-001",
        label: "Synthetic amber recipe",
        rgb: { r: 184, g: 112, b: 48 }
      },
      predicted_mask: {
        asset_id: "mask-vial-001",
        src: "/assets/masks/vial-001.png"
      },
      target_lab: { l: 58, a: 24, b: 46 },
      reference_lab: { l: 57, a: 23, b: 45 },
      threshold: { value: 3 },
      decision: {
        outcome: "keep",
        limit_codes: ["within_delta_e_limit"]
      }
    });
    expect(data.vials[0].recipe.rgb.provenance.kind).toBe("measured");
    expect(data.vials[0].target_lab!.provenance.kind).toBe("measured");
    expect(data.vials[0].top_down.position).toMatchObject({ x: 0.1, y: 0.5 });
    expect(data.top_down.state).toBe("waiting");
    expect(
      data.vials.every(
        (vial) => vial.decision.explanation.source === "deterministic_template"
      )
    ).toBe(true);
  });

  it("maps playback metadata and orders cues by vial slot", () => {
    const input = copyFixture();
    input.run.cue_times.reverse();
    const data = parseDemoData(input);

    expect(data.playback).toMatchObject({
      run_id: "run-synthetic-001",
      title: "Synthetic twin operator run",
      video: {
        asset_id: "run-video-001",
        src: "/assets/run/run-001.mp4",
        duration_seconds: { value: 36 }
      }
    });
    expect(data.playback.cues.map((cue) => cue.vial_id)).toEqual(
      Array.from({ length: 9 }, (_, index) => `vial-${String(index + 1).padStart(3, "0")}`)
    );
    expect(data.playback.cues[0]).toMatchObject({
      slot: 1,
      start_seconds: 0,
      end_seconds: 4
    });
    expect(data.playback.cues[8]).toMatchObject({
      slot: 9,
      start_seconds: 32,
      end_seconds: 36
    });
  });

  it("rejects overlapping playback cues", () => {
    const input = copyFixture();
    input.run.cue_times[1].start_seconds = 3;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Playback cue for "vial-002" overlaps the previous vial slot'
      )
    );
  });

  it("joins same-recipe history with time provenance and prior explanations", () => {
    const data = parseDemoData(fixture);
    const history = data.vials[0].history[0];

    expect(history).toMatchObject({
      history_id: "history-vial-001-01",
      run_id: "run-synthetic-prior",
      vial_id: "prior-vial-amber-01",
      time_seconds: {
        value: 12.5,
        provenance: { kind: "documented", source_id: "synthetic-history-001" }
      },
      decision: "discard",
      explanation: {
        source: "deterministic_template",
        text: "Prior synthetic twin vial exceeded the documented demo limit."
      }
    });
    expect(history.provenance.kind).toBe("documented");
  });

  it("rejects a history explanation with no evidence references", () => {
    const input = copyFixture();
    input.history[0].explanation.evidence_refs = [];

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects a dangling history explanation evidence reference", () => {
    const input = copyFixture();
    input.history[0].explanation.evidence_refs = ["evidence-missing"];

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'History "history-vial-001-01" references missing evidence "evidence-missing"'
      )
    );
  });

  it("rejects history evidence outside the linked vial context", () => {
    const input = copyFixture();
    input.history[0].explanation.evidence_refs = ["evidence-delta-vial-002"];

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("joins clickable evidence records in explanation order", () => {
    const data = parseDemoData(fixture);

    expect(data.vials[0].evidence).toEqual([
      expect.objectContaining({
        evidence_id: "evidence-delta-vial-001",
        target: "delta_e",
        vial_id: "vial-001"
      }),
      expect.objectContaining({
        evidence_id: "evidence-frame-vial-001",
        target: "rgb_frame",
        vial_id: "vial-001"
      }),
      expect.objectContaining({
        evidence_id: "evidence-mask-vial-001",
        target: "predicted_mask",
        vial_id: "vial-001"
      })
    ]);
  });

  it("maps the documented WP6 benchmark rows and INN-67 placeholder", () => {
    const data = parseDemoData(fixture);

    expect(data.wp6.disclaimer).toBe(
      "limited all-abstention benchmark, no accuracy claim"
    );
    expect(data.wp6.rows.map((row) => [row.method, row.passed, row.total])).toEqual([
      ["template", 16, 16],
      ["qwen3_vl", 16, 16],
      ["anomalythink_free_form", 0, 16],
      ["cosmos_strict_parse", 0, 16]
    ]);
    expect(data.wp6.rows[2].failure_note).toContain("free-form");
    expect(data.wp6.rows[3].failure_note).toContain("strict-parse");
    expect(data.wp6.inn67).toMatchObject({
      status: "placeholder",
      label: "INN-67 placeholder"
    });
  });

  it("rejects a 0/16 WP6 row without a failure note", () => {
    const input = copyFixture();
    input.wp6.rows[2].failure_note = null;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'WP6 row "anomalythink_free_form" requires a failure note for 0/16'
      )
    );
  });

  it("maps a required local RGB frame beside every predicted mask", () => {
    const data = parseDemoData(fixture);

    expect(data.vials[0].rgb_frame).toMatchObject({
      asset_id: "rgb-frame-vial-001",
      src: "/assets/frames/vial-001.jpg",
      provenance: { kind: "documented", source_id: "synthetic-frame-001" }
    });
    expect(
      data.vials.every(
        (vial) =>
          vial.rgb_frame.src.startsWith("/assets/") &&
          vial.evidence.some((record) => record.target === "rgb_frame") &&
          vial.evidence.some((record) => record.target === "predicted_mask")
      )
    ).toBe(true);
  });

  it("rejects a vial with no RGB frame reference", () => {
    const input = copyFixture();
    delete input.vials[0].rgb_frame;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects a vial whose cited frame evidence has the wrong target", () => {
    const input = copyFixture();
    const frameEvidence = input.evidence.find(
      (record: Record<string, any>) =>
        record.evidence_id === "evidence-frame-vial-001"
    );
    frameEvidence.target = "predicted_mask";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Decision "decision-001" must cite rgb_frame and predicted_mask evidence targets'
      )
    );
  });
});
