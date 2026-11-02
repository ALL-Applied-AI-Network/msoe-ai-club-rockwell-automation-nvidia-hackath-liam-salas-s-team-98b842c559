/** TEST-ONLY: build an InspectionModel from events without the session hook, for screen tests. */
import { vi } from "vitest";

import type { RunnerEvent } from "../runner/contract";
import { initialRunnerState, reduceRunnerEvent } from "../runner/contract";
import type { InspectionModel } from "../model/useInspection";
import { HYBRID_COMMIT, HYBRID_SESSION_ID, hybridBundle, hybridDecisions, hybridEvents } from "./fakeClient";

export function modelFromEvents(events: readonly RunnerEvent[], overrides: Partial<InspectionModel> = {}): InspectionModel {
  const sessionId = events[0]?.session_id ?? "session-live";
  const runner = events.reduce(reduceRunnerEvent, initialRunnerState(sessionId, false));
  return {
    health: { status: "healthy", service: "fluidmix-pipeline-runner", api_version: "v1", pipeline_commit: HYBRID_COMMIT, test_only: false, inference: true, active_sessions: 1, limits: [] },
    healthError: null, pinStatus: "verified", expectedCommit: HYBRID_COMMIT, disclosedHybrid: true, runner, runPhase: "completed",
    connection: "Run completed", startError: null, streamFailure: null, busy: false,
    runContext: { operator_run_id: "RUN-TEST", recipe_id: "5", tray_id: "TRAY-TEST" }, sourceKind: "video", sourceLabel: "accepted-v4-train01-cohort.mp4",
    videoUrl: null, decisions: null, decisionsError: null, bundle: null, vialCount: 3, fps: 10, framesPerVial: null,
    selectedVialId: runner.trayOrder[0] ?? null, localAcks: new Set(), operatorLog: [],
    start: vi.fn(), attach: vi.fn(), setVideoFile: vi.fn(), selectVial: vi.fn(), acknowledge: vi.fn(),
    recordOperatorAction: vi.fn(() => null), imageUrl: vi.fn((id: string) => `test://${sessionId}/${id}.png`), refreshDecisions: vi.fn(),
    ...overrides
  };
}

/** The recorded hybrid development session with its decision receipts and evidence bundle. */
export function hybridModel(overrides: Partial<InspectionModel> = {}): InspectionModel {
  const events = hybridEvents();
  return modelFromEvents(events, { decisions: hybridDecisions(), bundle: hybridBundle(), selectedVialId: "CASE_97a8c6cb8161881c", ...overrides });
}
export { HYBRID_SESSION_ID };
