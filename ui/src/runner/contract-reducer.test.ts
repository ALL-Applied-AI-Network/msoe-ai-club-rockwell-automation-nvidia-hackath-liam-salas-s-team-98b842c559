import { describe, expect, it } from "vitest";

import { initialRunnerState, reduceRunnerEvent, type RunnerEvent } from "./contract";

const source = { source_id: "source-1", kind: "file" as const, label: "three-pour.avi" };
const position = { tray_index: 7, vial_number: "vial-7", slot_label: "Slot 8" };
const locator = { status: "unavailable" as const, bbox_normalized: null, centroid_normalized: null, evidence_ids: [], unavailable_reason: "No top-down feed" };
const tracker = {
  state: "tracking" as const, geometry_source: "pipeline" as const, fill_fraction: 0.4, confidence: 0.91,
  tank_id: "T-200", fill_step_id: "step-2", measured_volume_ml: 4.2, volume_provenance: "measured" as const,
  unavailable_reason: null, locator
};
const measurement = {
  measurement_id: "measurement-7", status: "measured" as const, rgb: [100, 90, 80] as const,
  lab: [51.2, 12.4, -3.1] as const, delta_e: 2.8, reference_ids: ["reference-1"], evidence_ids: ["evidence-7"], unavailable_reason: null
};

function event(sequence: number, type: RunnerEvent["type"], payload: unknown, eventId = `event-${sequence}`): RunnerEvent {
  return {
    schema_version: "runner-event.v1", event_id: eventId, session_id: "session-1", sequence,
    emitted_at: `2026-10-31T12:00:${String(sequence).padStart(2, "0")}Z`,
    pipeline_commit: "7bf35aeea21c36acf1a390661bd05fea2b07b190", source, type, payload
  } as RunnerEvent;
}

describe("runner-event.v1 reduction", () => {
  it("derives vial and physical tray order from nested positions", () => {
    const state = [
      event(1, "measuring", { position, tracker, measurement, stage: "measuring" }),
      event(2, "vial_decision", {
        position: { tray_index: 1, vial_number: "vial-1", slot_label: "Slot 2" }, disposition: "keep", decision_id: "decision-1",
        alarm: null, measurement, evidence_ids: ["evidence-1"], unavailable_reason: null
      })
    ].reduce(reduceRunnerEvent, initialRunnerState("session-1", true));

    expect(state.trayOrder).toEqual(["vial-1", "vial-7"]);
    expect(state.vials["vial-7"].tracker?.measured_volume_ml).toBe(4.2);
    expect(state.vials["vial-7"].measurement?.lab).toEqual([51.2, 12.4, -3.1]);
    expect(state.testOnly).toBe(true);
  });

  it("retains nested model, equal-weight average, final alarm and plain explanation", () => {
    const alarm = { alarm_id: "alarm-7", state: "unacknowledged" as const, acknowledged_at: null, acknowledged_by: null };
    const events = [
      event(1, "model_vote", { position, vote_id: "vote-1", model: { model_id: "cnn", revision: "r4", role: "cnn", source: "inferred" }, admission: { admitted: true, reason: "valid" }, disposition: "discard", confidence: 0.82, evidence_ids: ["evidence-7"], unavailable_reason: null }),
      event(2, "averaged_decision", { position, decision_id: "average-7", method: "equal_weight", admitted_vote_ids: ["vote-1"], scores: { keep: 0.1, check: 0.08, discard: 0.82 }, disposition: "discard", alarm, evidence_ids: ["evidence-7"], unavailable_reason: null }),
      event(3, "explanation", { position, explanation_id: "explanation-7", plain_language: "The measured color differs from the reference.", limits: ["Reference quality is limited."], evidence_ids: ["evidence-7"], producer: { model_id: "explain", revision: "r2", source: "inferred" }, unavailable_reason: null }),
      event(4, "rate_health", { status: "healthy", processing_fps: 7.5, source_fps: 10, queue_depth: 0, active_model_id: "cnn", gpu_free_mib: 12000, gpu_status: "available", limits: [] })
    ];
    const state = events.reduce(reduceRunnerEvent, initialRunnerState("session-1"));

    expect(state.vials["vial-7"].votes[0].model.revision).toBe("r4");
    expect(state.vials["vial-7"].average?.scores.discard).toBe(0.82);
    expect(state.vials["vial-7"].average?.alarm?.alarm_id).toBe("alarm-7");
    expect(state.vials["vial-7"].explanation?.plain_language).toContain("differs");
    expect(state.rate?.processing_fps).toBe(7.5);
  });

  it("fails closed on a sequence gap or a conflicting replay", () => {
    const gap = reduceRunnerEvent(initialRunnerState("session-1"), event(2, "rate_health", { status: "healthy" }));
    expect(gap.protocolError).toMatch(/expected sequence 1/i);

    const first = reduceRunnerEvent(initialRunnerState("session-1"), event(1, "rate_health", { status: "healthy" }));
    const corrupt = reduceRunnerEvent(first, event(1, "rate_health", { status: "stalled" }, "different-event"));
    expect(corrupt.protocolError).toMatch(/sequence 1.*conflict/i);
    expect(reduceRunnerEvent(first, event(1, "rate_health", { status: "healthy" }))).toBe(first);
  });
});
