import { describe, expect, it } from "vitest";

import { initialRunnerState, reduceRunnerEvent, type RunnerEvent } from "../runner/contract";
import { runnerEvent } from "../test/fakeClient";
import { alarmList, dispositionView, labToHex } from "./selectors";

const reduce = (events: RunnerEvent[]) => events.reduce(reduceRunnerEvent, initialRunnerState("session-live"));

describe("alarmList", () => {
  it("does not raise the rule's discard alarm when the equal-weight average keeps the vial", () => {
    const runner = reduce([
      runnerEvent(1, "vial_decision", { disposition: "discard" }, "vial-a", 0),
      runnerEvent(2, "averaged_decision", { disposition: "keep", average: 0.6 }, "vial-a", 0)
    ]);
    expect(alarmList(runner, new Set())).toEqual([]);
  });

  it("keeps critical error alarms after the capped event log has rolled over", () => {
    const events: RunnerEvent[] = [runnerEvent(1, "error", { code: "pipeline_failed", message: "boom", stage: "inference", retryable: false, evidence_ids: [] })];
    for (let i = 2; i <= 450; i += 1) events.push(runnerEvent(i, "measuring", {}, "vial-a", 0));
    const runner = reduce(events);
    expect(runner.eventLog.some(e => e.type === "error")).toBe(false);
    expect(alarmList(runner, new Set()).map(a => a.priority)).toContain("critical");
  });

  it("raises a medium can't-assess alarm that is acknowledged locally", () => {
    const runner = reduce([runnerEvent(1, "vial_decision", { disposition: "insufficient_evidence", unavailable_reason: "measurement_stage_unavailable" }, "unavailable", 0)]);
    const [alarm] = alarmList(runner, new Set());
    expect(alarm).toMatchObject({ priority: "medium", origin: "ui", acknowledged: false });
    expect(alarmList(runner, new Set([alarm.id]))[0].acknowledged).toBe(true);
  });
});

describe("display helpers", () => {
  it("keeps keep neutral and colours only abnormal states", () => {
    expect(dispositionView("keep").tone).toBe("keep");
    expect(dispositionView("discard")).toMatchObject({ icon: "✖", tone: "discard" });
    expect(dispositionView("insufficient_evidence")).toMatchObject({ icon: "?", tone: "cant_assess" });
  });

  it("maps CIELAB white and black to sRGB extremes", () => {
    expect(labToHex([100, 0, 0])).toBe("#ffffff");
    expect(labToHex([0, 0, 0])).toBe("#000000");
  });
});
