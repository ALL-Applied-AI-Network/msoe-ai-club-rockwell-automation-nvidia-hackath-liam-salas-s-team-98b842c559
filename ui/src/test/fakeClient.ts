/**
 * TEST-ONLY runner client and fixtures. Never imported by the operator path.
 *
 * `hybridSession()` returns the recorded development stream from session e7383eef (service 2344f3c),
 * with sequences renumbered for the reducer; see `fixtures/hybrid-session-events.json` provenance.
 */
import { vi } from "vitest";

import type { RunnerClient, RunnerHealth, RunnerSession, RunnerSource, SessionState, StreamNotice } from "../runner/client";
import type { RunnerEvent } from "../runner/contract";
import { parseDecisionRecords, parseEvidenceBundle, type DecisionRecords, type EvidenceBundle } from "../runner/evidence";
import bundleJson from "./fixtures/hybrid-session-bundle.json";
import decisionsJson from "./fixtures/hybrid-session-decisions.json";
import eventsJson from "./fixtures/hybrid-session-events.json";

export const TEST_COMMIT = "cf6829b99486b41f4190874dbd701ad812648217";
export const HYBRID_COMMIT = "2344f3c655d9aa255eb576e6d2a280bd0fc9bdff";
export const HYBRID_SESSION_ID = "e7383eef1fbc40d5b5192cd8bd1a195c";
export const HYBRID_TARGET_VIAL = "CASE_97a8c6cb8161881c";

export const TEST_HEALTH: RunnerHealth = {
  status: "healthy", service: "fluidmix-pipeline-runner", api_version: "v1", pipeline_commit: TEST_COMMIT,
  test_only: true, inference: false, active_sessions: 0, limits: []
};

export function hybridEvents(): RunnerEvent[] {
  return (eventsJson as unknown as { events: RunnerEvent[] }).events;
}
export function hybridDecisions(): DecisionRecords {
  return parseDecisionRecords(decisionsJson);
}
export function hybridBundle(): EvidenceBundle {
  return parseEvidenceBundle(bundleJson) as EvidenceBundle;
}

export interface FakeClientOptions {
  readonly health?: RunnerHealth | Error;
  readonly sessionState?: SessionState;
  readonly sessionId?: string;
  readonly sessionCommit?: string;
  readonly decisions?: DecisionRecords | Error;
  readonly bundle?: EvidenceBundle | null;
  readonly sessionLookup?: (sessionId: string) => Promise<RunnerSession>;
}

export class FakeRunnerClient implements RunnerClient {
  readonly sources: RunnerSource[] = [];
  readonly acknowledgements: string[] = [];
  private onEvent?: (event: RunnerEvent) => void;
  private onDisconnect?: (notice: StreamNotice) => void;
  private sessionCount = 0;
  decisions: DecisionRecords | Error;
  bundle: EvidenceBundle | null;

  constructor(private readonly options: FakeClientOptions = {}) {
    this.decisions = options.decisions ?? { rules: [], finals: [], abstentions: [], unparsed: 0 };
    this.bundle = options.bundle ?? null;
  }

  private get commit() { return this.options.sessionCommit ?? (this.options.health instanceof Error ? TEST_COMMIT : this.options.health?.pipeline_commit ?? TEST_COMMIT); }

  async getHealth() {
    const health = this.options.health ?? TEST_HEALTH;
    if (health instanceof Error) throw health;
    return health;
  }

  async createSession(source: RunnerSource) {
    this.sources.push(source);
    this.sessionCount += 1;
    const base = this.options.sessionId ?? "session-live";
    return { session_id: this.sessionCount === 1 ? base : `${base}-${this.sessionCount}`, test_only: true, pipeline_commit: this.commit };
  }

  async getSession(sessionId: string): Promise<RunnerSession> {
    if (this.options.sessionLookup) return this.options.sessionLookup(sessionId);
    return { session_id: sessionId, state: this.options.sessionState ?? "running", pipeline_commit: this.commit, test_only: true, last_sequence: 0 };
  }

  subscribe(_sessionId: string, _after: number, onEvent: (event: RunnerEvent) => void, onDisconnect: (notice: StreamNotice) => void) {
    this.onEvent = onEvent;
    this.onDisconnect = onDisconnect;
    return vi.fn();
  }

  async acknowledge(_sessionId: string, alarmId: string, _operatorId: string) { this.acknowledgements.push(alarmId); }
  async getDecisions() { if (this.decisions instanceof Error) throw this.decisions; return this.decisions; }
  async getEvidenceBundle() { return this.bundle; }
  evidenceImageUrl(sessionId: string, imageId: string) { return `test://${sessionId}/${imageId}.png`; }

  emit(event: RunnerEvent) { this.onEvent?.(event); }
  /** Emit events re-stamped with the given session ID (the stream checks session IDs in the reducer). */
  emitAll(events: readonly RunnerEvent[], sessionId = "session-live") { for (const e of events) this.onEvent?.({ ...e, session_id: sessionId } as RunnerEvent); }
  disconnect() { this.onDisconnect?.({ kind: "transport", message: "Runner connection interrupted" }); }
  rejectStream(message = "Runner sent an invalid event") { this.onDisconnect?.({ kind: "terminal", message }); }
}

/** Compact event builder for unit tests (shapes match runner-event.v1). */
export function runnerEvent(sequence: number, type: RunnerEvent["type"], payload: Record<string, unknown>, vialId?: string, trayPosition?: number, sessionId = "session-live"): RunnerEvent {
  const position = { tray_index: trayPosition ?? 0, vial_number: vialId ?? "vial-unknown", slot_label: `Slot ${(trayPosition ?? 0) + 1}` };
  const locator = payload.locator ?? { status: "unavailable", bbox_normalized: null, centroid_normalized: null, evidence_ids: [], unavailable_reason: "No top-down feed" };
  const tracker = { state: "tracking", geometry_source: "pipeline", fill_fraction: 0.4, confidence: 0.9, tank_id: payload.tank_id ?? "t200", fill_step_id: payload.fill_step_id ?? "1", measured_volume_ml: payload.tracker_volume_ml ?? 4.2, volume_provenance: "measured", unavailable_reason: null, locator };
  const measurement = { measurement_id: payload.measurement_id ?? "measurement-7", status: payload.source ?? "measured", rgb: [100, 90, 80], lab: payload.lab ?? [51.2, 12.4, -3.1], delta_e: "delta_e" in payload ? payload.delta_e : 2.8, reference_ids: ["reference-1"], evidence_ids: payload.evidence_ids ?? ["measurement-7"], unavailable_reason: payload.unavailable_reason ?? null };
  const disposition = payload.disposition ?? "keep";
  const alarm = disposition === "discard" ? { alarm_id: `alarm-${vialId}`, state: "unacknowledged", acknowledged_at: null, acknowledged_by: null } : null;
  const exactPayload: Record<string, unknown> = type === "measuring" ? { position, tracker, measurement, stage: "measuring" }
    : type === "waiting_for_references" ? { position, tracker, missing_reference_types: ["clean_recipe"], available_reference_ids: [], reason: "later_vials_not_closed" }
    : type === "vial_decision" ? { position, disposition, decision_id: `decision-${sequence}`, alarm, measurement, evidence_ids: measurement.evidence_ids, unavailable_reason: payload.unavailable_reason ?? null }
    : type === "model_vote" ? { position, vote_id: `vote-${sequence}`, model: { model_id: payload.model_id, revision: payload.model_revision ?? "r1", role: "vlm", source: payload.source ?? "inferred" }, admission: { admitted: payload.admitted ?? true, reason: payload.admission_reason ?? "admitted" }, disposition, confidence: payload.score ?? null, evidence_ids: ["measurement-7"], unavailable_reason: null }
    : type === "averaged_decision" ? { position, decision_id: `average-${sequence}`, method: "equal_weight", admitted_vote_ids: ["vote-2"], scores: { keep: disposition === "keep" ? payload.average : 0.1, check: 0.08, discard: disposition === "discard" ? payload.average : 0.1 }, disposition, alarm, evidence_ids: ["measurement-7"], unavailable_reason: null }
    : type === "explanation" ? { position, explanation_id: `explanation-${sequence}`, plain_language: payload.text, limits: payload.limits ?? [], evidence_ids: payload.evidence_ids ?? [], producer: { model_id: payload.model_id ?? "explain", revision: payload.model_revision ?? "r2", source: payload.source ?? "inferred" }, unavailable_reason: null }
    : type === "rate_health" ? { status: payload.status ?? "healthy", processing_fps: payload.processing_rate_fps ?? 10, source_fps: 10, queue_depth: 0, active_model_id: "cnn", gpu_free_mib: 12000, gpu_status: "available", limits: [] }
    : payload;
  return {
    schema_version: "runner-event.v1", event_id: `event-${sequence}`, type, session_id: sessionId, sequence,
    emitted_at: `2026-10-31T12:00:${String(sequence).padStart(2, "0")}Z`, pipeline_commit: TEST_COMMIT,
    source: { source_id: "source-1", kind: "file", label: "three-pour.avi" }, payload: exactPayload
  } as RunnerEvent;
}
