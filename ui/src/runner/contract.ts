export type Disposition = "keep" | "check" | "discard" | "insufficient_evidence";
export type EvidenceSource = "measured" | "inferred" | "unavailable";
export type RunnerEventType = "measuring" | "waiting_for_references" | "vial_decision" | "model_vote" | "averaged_decision" | "explanation" | "error" | "rate_health";

export interface SourceIdentity { readonly source_id: string; readonly kind: "file" | "v4l2" | "rtsp"; readonly label: string }
export interface Position { readonly tray_index: number; readonly vial_number: string; readonly slot_label: string }
export interface Locator {
  readonly status: "located" | "unavailable";
  readonly bbox_normalized: readonly [number, number, number, number] | null;
  readonly centroid_normalized: readonly [number, number] | null;
  readonly evidence_ids: readonly string[];
  readonly unavailable_reason: string | null;
}
export interface Tracker {
  readonly state: "unavailable" | "searching" | "tracking" | "lost";
  readonly geometry_source: "pipeline" | "top_down_locator" | "unavailable";
  readonly fill_fraction: number | null;
  readonly confidence: number | null;
  readonly tank_id: string | null;
  readonly fill_step_id: string | null;
  readonly measured_volume_ml: number | null;
  readonly volume_provenance: EvidenceSource;
  readonly unavailable_reason: string | null;
  readonly locator: Locator;
}
export interface Measurement {
  readonly measurement_id: string;
  readonly status: EvidenceSource;
  readonly rgb: readonly [number, number, number] | null;
  readonly lab: readonly [number, number, number] | null;
  readonly delta_e: number | null;
  readonly reference_ids: readonly string[];
  readonly evidence_ids: readonly string[];
  readonly unavailable_reason: string | null;
}
export interface Alarm { readonly alarm_id: string; readonly state: "unacknowledged" | "acknowledged"; readonly acknowledged_at: string | null; readonly acknowledged_by: string | null }
export interface DecisionPayload {
  readonly position: Position; readonly disposition: Disposition; readonly decision_id: string; readonly alarm: Alarm | null;
  readonly measurement: Measurement; readonly evidence_ids: readonly string[]; readonly unavailable_reason: string | null;
}
export interface ModelVotePayload {
  readonly position: Position; readonly vote_id: string;
  readonly model: { readonly model_id: string; readonly revision: string; readonly role: "cnn" | "vlm" | "rule"; readonly source: string };
  readonly admission: { readonly admitted: boolean; readonly reason: string };
  readonly disposition: Disposition; readonly confidence: number | null; readonly evidence_ids: readonly string[]; readonly unavailable_reason: string | null;
}
export interface AveragePayload {
  readonly position: Position; readonly decision_id: string; readonly method: "equal_weight"; readonly admitted_vote_ids: readonly string[];
  readonly scores: { readonly keep: number; readonly check: number; readonly discard: number };
  readonly disposition: Disposition; readonly alarm: Alarm | null; readonly evidence_ids: readonly string[]; readonly unavailable_reason: string | null;
}
export interface ExplanationPayload {
  readonly position: Position; readonly explanation_id: string; readonly plain_language: string; readonly limits: readonly string[];
  readonly evidence_ids: readonly string[]; readonly producer: { readonly model_id: string; readonly revision: string; readonly source: string }; readonly unavailable_reason: string | null;
}
export interface RateHealthPayload {
  readonly status: "healthy" | "degraded" | "stalled"; readonly processing_fps: number; readonly source_fps: number | null;
  readonly queue_depth: number; readonly active_model_id: string | null; readonly gpu_free_mib: number | null;
  readonly gpu_status: "available" | "unavailable" | "not_required"; readonly limits: readonly string[];
}

interface EventEnvelope<T extends RunnerEventType, P> {
  readonly schema_version: "runner-event.v1"; readonly event_id: string; readonly session_id: string; readonly sequence: number;
  readonly emitted_at: string; readonly pipeline_commit: string; readonly source: SourceIdentity; readonly type: T; readonly payload: P;
}
export type RunnerEvent =
  | EventEnvelope<"measuring", { readonly position: Position; readonly tracker: Tracker; readonly measurement: Measurement; readonly stage: "locating" | "measuring" }>
  | EventEnvelope<"waiting_for_references", { readonly position: Position; readonly tracker: Tracker; readonly missing_reference_types: readonly ("clear" | "clean_recipe" | "previous_recipe")[]; readonly available_reference_ids: readonly string[]; readonly reason: string }>
  | EventEnvelope<"vial_decision", DecisionPayload>
  | EventEnvelope<"model_vote", ModelVotePayload>
  | EventEnvelope<"averaged_decision", AveragePayload>
  | EventEnvelope<"explanation", ExplanationPayload>
  | EventEnvelope<"error", { readonly code: string; readonly message: string; readonly stage: string; readonly retryable: boolean; readonly evidence_ids: readonly string[] }>
  | EventEnvelope<"rate_health", RateHealthPayload>;

/** One streamed tracker/measurement sample, kept in arrival order for timelines and the evidence grid. */
export interface StageSample {
  readonly sequence: number; readonly emitted_at: string; readonly stage: "locating" | "measuring" | "waiting_for_references";
  readonly tracker: Tracker; readonly measurement?: Measurement;
}
/** Compact record of a non-rate event, for the alarm list and diagnostics log. */
export interface LoggedEvent {
  readonly sequence: number; readonly emitted_at: string; readonly type: RunnerEventType; readonly vialId: string | null; readonly summary: string;
}
export interface VialState {
  readonly vialId: string; readonly position: Position; readonly phase: "locating" | "measuring" | "waiting_for_references" | "decided";
  readonly history: readonly StageSample[]; readonly firstSeenAt: string; readonly decidedAt?: string;
  readonly tracker?: Tracker; readonly measurement?: Measurement; readonly decision?: DecisionPayload;
  readonly votes: readonly ModelVotePayload[]; readonly average?: AveragePayload; readonly explanation?: ExplanationPayload;
  readonly explanations: readonly ExplanationPayload[];
}
export interface RunnerState {
  readonly sessionId: string; readonly lastSequence: number; readonly appliedEvents: Readonly<Record<number, string>>;
  readonly vials: Readonly<Record<string, VialState>>; readonly trayOrder: readonly string[]; readonly rate?: RateHealthPayload;
  readonly testOnly: boolean; readonly error?: string; readonly protocolError?: string;
  readonly eventLog: readonly LoggedEvent[];
  /** Uncapped: every error event, and the first non-healthy rate report, so alarms never age out of the capped logs. */
  readonly errorLog: readonly LoggedEvent[]; readonly firstUnhealthy?: { readonly emitted_at: string; readonly processing_fps: number; readonly status: RateHealthPayload["status"] }; readonly rateHistory: readonly { readonly emitted_at: string; readonly processing_fps: number; readonly status: RateHealthPayload["status"] }[];
}
const EVENT_LOG_LIMIT = 400;
const RATE_HISTORY_LIMIT = 240;

export function initialRunnerState(sessionId: string, testOnly = false): RunnerState {
  return { sessionId, lastSequence: 0, appliedEvents: {}, vials: {}, trayOrder: [], testOnly, eventLog: [], errorLog: [], rateHistory: [] };
}

function eventFingerprint(event: RunnerEvent): string { return JSON.stringify(event); }
function eventPosition(event: RunnerEvent): Position | undefined {
  return event.type === "error" || event.type === "rate_health" ? undefined : event.payload.position;
}

function updateVial(state: RunnerState, event: RunnerEvent): RunnerState {
  const position = eventPosition(event);
  if (!position) return state;
  const vialId = position.vial_number;
  const current: VialState = state.vials[vialId] ?? { vialId, position, phase: "locating", votes: [], explanations: [], history: [], firstSeenAt: event.emitted_at };
  let next: VialState = { ...current, position };
  switch (event.type) {
    case "measuring": next = {
      ...next, phase: event.payload.stage, tracker: event.payload.tracker, measurement: event.payload.measurement,
      history: [...current.history, { sequence: event.sequence, emitted_at: event.emitted_at, stage: event.payload.stage, tracker: event.payload.tracker, measurement: event.payload.measurement }]
    }; break;
    case "waiting_for_references": next = {
      ...next, phase: "waiting_for_references", tracker: event.payload.tracker,
      history: [...current.history, { sequence: event.sequence, emitted_at: event.emitted_at, stage: "waiting_for_references", tracker: event.payload.tracker }]
    }; break;
    case "vial_decision": next = { ...next, phase: "decided", decision: event.payload, measurement: event.payload.measurement, decidedAt: current.decidedAt ?? event.emitted_at }; break;
    case "model_vote": next = { ...next, votes: [...current.votes.filter(vote => vote.vote_id !== event.payload.vote_id), event.payload] }; break;
    case "averaged_decision": next = { ...next, average: event.payload }; break;
    case "explanation": next = {
      ...next,
      explanation: event.payload,
      explanations: [...current.explanations.filter(item => item.explanation_id !== event.payload.explanation_id), event.payload]
    }; break;
  }
  const vials = { ...state.vials, [vialId]: next };
  const trayOrder = Object.values(vials).sort((a, b) => a.position.tray_index - b.position.tray_index || a.vialId.localeCompare(b.vialId)).map(vial => vial.vialId);
  return { ...state, vials, trayOrder };
}

function logEntry(event: RunnerEvent): LoggedEvent {
  const vialId = eventPosition(event)?.vial_number ?? null;
  let summary: string;
  switch (event.type) {
    case "measuring": summary = `${event.payload.stage} · ${event.payload.measurement.measurement_id} · ${event.payload.measurement.status}`; break;
    case "waiting_for_references": summary = `waiting for ${event.payload.missing_reference_types.join(", ") || "references"} (${event.payload.reason})`; break;
    case "vial_decision": summary = `rule decision ${event.payload.disposition}${event.payload.unavailable_reason ? ` (${event.payload.unavailable_reason})` : ""}`; break;
    case "model_vote": summary = `${event.payload.model.model_id} vote ${event.payload.disposition}${event.payload.confidence == null ? "" : ` @ ${event.payload.confidence}`}`; break;
    case "averaged_decision": summary = `average ${event.payload.disposition} over ${event.payload.admitted_vote_ids.length} admitted votes`; break;
    case "explanation": summary = `explanation from ${event.payload.producer.model_id} (${event.payload.producer.source})`; break;
    case "error": summary = `${event.payload.code}: ${event.payload.message}`; break;
    default: summary = event.type;
  }
  return { sequence: event.sequence, emitted_at: event.emitted_at, type: event.type, vialId, summary };
}

export function reduceRunnerEvent(state: RunnerState, event: RunnerEvent): RunnerState {
  if (state.protocolError || event.session_id !== state.sessionId) return state;
  const existing = state.appliedEvents[event.sequence];
  const fingerprint = eventFingerprint(event);
  if (existing) return existing === fingerprint ? state : { ...state, protocolError: `Sequence ${event.sequence} conflicts with an applied event. Live decisions are unavailable.` };
  const expected = state.lastSequence + 1;
  if (event.sequence !== expected) return { ...state, protocolError: `Expected sequence ${expected}, received ${event.sequence}. Live decisions are unavailable.` };
  let next = updateVial(state, event);
  if (event.type === "rate_health") {
    const sample = { emitted_at: event.emitted_at, processing_fps: event.payload.processing_fps, status: event.payload.status };
    if (!next.firstUnhealthy && sample.status !== "healthy") next = { ...next, firstUnhealthy: sample };
    next = { ...next, rate: event.payload, rateHistory: [...next.rateHistory, { emitted_at: event.emitted_at, processing_fps: event.payload.processing_fps, status: event.payload.status }].slice(-RATE_HISTORY_LIMIT) };
  } else {
    const entry = logEntry(event);
    next = { ...next, eventLog: [...next.eventLog, entry].slice(-EVENT_LOG_LIMIT), errorLog: event.type === "error" ? [...next.errorLog, entry] : next.errorLog };
  }
  if (event.type === "error") next = { ...next, error: event.payload.message };
  return { ...next, lastSequence: event.sequence, appliedEvents: { ...next.appliedEvents, [event.sequence]: fingerprint } };
}
