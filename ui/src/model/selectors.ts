/**
 * Pure view-model helpers shared by every screen. Everything here derives from runner events,
 * runner decision receipts or the optional evidence bundle. Nothing is invented: a missing value
 * stays `null` and the caller must render it as unavailable.
 */
import type { Disposition, RunnerState, VialState } from "../runner/contract";
import type { DecisionRecords, FinalRecord, ModelSourceRecord, RuleReceipt } from "../runner/evidence";

export type Tone = "keep" | "discard" | "cant_assess" | "attention" | "running" | "pending" | "reference";
export interface DispositionView { readonly icon: string; readonly label: string; readonly tone: Tone }

/** ISA-101 style: colour only for discard (red), can't assess (amber) and running (blue outline). */
export function dispositionView(disposition?: Disposition | string | null): DispositionView {
  switch (disposition) {
    case "keep": return { icon: "✓", label: "Keep", tone: "keep" };
    case "discard": return { icon: "✖", label: "Discard", tone: "discard" };
    case "insufficient_evidence": return { icon: "?", label: "Can't assess", tone: "cant_assess" };
    case "check": return { icon: "?", label: "Check by hand", tone: "attention" };
    default: return { icon: "○", label: "Waiting", tone: "pending" };
  }
}

/** Default when the runner does not report its profile: the frozen recipe-5 profile uses 3 vials. */
export const DEFAULT_VIAL_COUNT = 3;

export type VialRole = "target" | "reference";
/** In each cohort of `vialCount` vials, the first is judged and the rest are its references. */
export function vialRole(vial: VialState, vialCount = DEFAULT_VIAL_COUNT): VialRole {
  return vial.position.tray_index % vialCount === 0 ? "target" : "reference";
}

export function vialLabel(vial: VialState): string {
  return `Vial ${vial.position.tray_index + 1}`;
}

/** The disposition the operator should act on: equal-weight average if present, otherwise the rule. */
export function finalDisposition(vial: VialState): Disposition | undefined {
  return vial.average?.disposition ?? vial.decision?.disposition;
}

export function vialView(vial: VialState, opts: { readonly runEnded: boolean; readonly vialCount?: number }): DispositionView {
  const disposition = finalDisposition(vial);
  if (disposition) return dispositionView(disposition);
  if (vialRole(vial, opts.vialCount) === "reference" && (vial.phase === "waiting_for_references" || opts.runEnded || vial.history.length > 0)) {
    return { icon: "◎", label: "Reference", tone: "reference" };
  }
  if (opts.runEnded) return { icon: "—", label: "Not assessed (run ended)", tone: "pending" };
  if (vial.phase === "waiting_for_references") return { icon: "…", label: "Waiting for references", tone: "running" };
  return { icon: "…", label: "Measuring", tone: "running" };
}

export type SourceKey = "rule" | "qwen" | "cosmos" | "anomalythink";
export const SOURCE_ORDER: readonly SourceKey[] = ["rule", "qwen", "cosmos", "anomalythink"];
export const SOURCE_NAMES: Readonly<Record<SourceKey, string>> = { rule: "Rule", qwen: "Qwen", cosmos: "Cosmos", anomalythink: "AnomalyThink" };

export function sourceKeyForModel(modelId: string): SourceKey | null {
  const id = modelId.toLowerCase();
  if (id.includes("anomalythink")) return "anomalythink";
  if (id.includes("cosmos")) return "cosmos";
  if (id.includes("qwen")) return "qwen";
  if (id === "deterministic" || id.includes("rule")) return "rule";
  return null;
}

export interface SourceAgreement {
  readonly key: SourceKey;
  readonly name: string;
  readonly disposition: Disposition | null;
  readonly confidence: number | null;
  readonly admitted: boolean | null;
  readonly admissionReason: string | null;
  /** null when either side is unavailable. */
  readonly agreesWithRule: boolean | null;
}

/** Rule plus the three VLM votes for one vial, in a fixed order, from streamed events only. */
export function sourceAgreement(vial: VialState): readonly SourceAgreement[] {
  const rule = vial.decision?.disposition ?? null;
  return SOURCE_ORDER.map(key => {
    if (key === "rule") {
      return { key, name: SOURCE_NAMES[key], disposition: rule, confidence: null, admitted: rule ? true : null, admissionReason: null, agreesWithRule: rule ? true : null };
    }
    const vote = vial.votes.find(v => sourceKeyForModel(v.model.model_id) === key);
    if (!vote) return { key, name: SOURCE_NAMES[key], disposition: null, confidence: null, admitted: null, admissionReason: null, agreesWithRule: null };
    return {
      key, name: SOURCE_NAMES[key], disposition: vote.disposition, confidence: vote.confidence, admitted: vote.admission.admitted,
      admissionReason: vote.admission.reason, agreesWithRule: rule && vote.admission.admitted ? vote.disposition === rule : null
    };
  });
}

export interface Counts {
  readonly inspected: number; readonly keep: number; readonly discard: number; readonly cantAssess: number; readonly check: number;
  readonly discardRate: number | null; readonly cantAssessRate: number | null;
}

/** Counts judged vials only (vials with a rule or averaged disposition). References are not counted. */
export function counts(runner: RunnerState | null): Counts {
  const decided = runner ? runner.trayOrder.map(id => runner.vials[id]).filter(v => finalDisposition(v)) : [];
  const by = (d: Disposition) => decided.filter(v => finalDisposition(v) === d).length;
  const inspected = decided.length;
  const discard = by("discard"); const cantAssess = by("insufficient_evidence");
  return {
    inspected, keep: by("keep"), discard, cantAssess, check: by("check"),
    discardRate: inspected ? discard / inspected : null, cantAssessRate: inspected ? cantAssess / inspected : null
  };
}

export type AlarmPriority = "critical" | "high" | "medium" | "low";
export interface AlarmItem {
  readonly id: string;
  readonly priority: AlarmPriority;
  readonly at: string;
  readonly vialId: string | null;
  readonly vialLabel: string | null;
  readonly message: string;
  /** "runner" alarms are acknowledged through the runner API; "ui" alarms only locally. */
  readonly origin: "runner" | "ui";
  readonly acknowledged: boolean;
}

const PRIORITY_RANK: Readonly<Record<AlarmPriority, number>> = { critical: 0, high: 1, medium: 2, low: 3 };

/**
 * ISA-18.2 style list: discard = high (runner alarm); can't assess = medium; unhealthy service = medium;
 * pipeline error / protocol error = critical. `localAcks` holds IDs acknowledged in this browser.
 */
export function alarmList(runner: RunnerState | null, localAcks: ReadonlySet<string>): readonly AlarmItem[] {
  if (!runner) return [];
  const items: AlarmItem[] = [];
  for (const vialId of runner.trayOrder) {
    const vial = runner.vials[vialId];
    const runnerAlarm = vial.average?.alarm ?? vial.decision?.alarm;
    const disposition = finalDisposition(vial);
    if (runnerAlarm) {
      items.push({
        id: runnerAlarm.alarm_id, priority: "high", at: vial.decidedAt ?? vial.firstSeenAt, vialId, vialLabel: vialLabel(vial),
        message: "Discard: remove this vial from the keep path.", origin: "runner",
        acknowledged: runnerAlarm.state === "acknowledged" || localAcks.has(runnerAlarm.alarm_id)
      });
    } else if (disposition === "insufficient_evidence") {
      const id = `ui-cant-assess-${vialId}`;
      items.push({
        id, priority: "medium", at: vial.decidedAt ?? vial.firstSeenAt, vialId, vialLabel: vialLabel(vial),
        message: `Can't assess: ${vial.decision?.unavailable_reason ?? vial.average?.unavailable_reason ?? "not enough reliable evidence"}.`, origin: "ui",
        acknowledged: localAcks.has(id)
      });
    }
  }
  const unhealthy = runner.rateHistory.find(r => r.status !== "healthy");
  if (unhealthy) {
    const id = `ui-service-${unhealthy.status}`;
    items.push({ id, priority: "medium", at: unhealthy.emitted_at, vialId: null, vialLabel: null, message: `Service ${unhealthy.status}.`, origin: "ui", acknowledged: localAcks.has(id) });
  }
  for (const entry of runner.eventLog.filter(e => e.type === "error")) {
    const id = `ui-error-${entry.sequence}`;
    items.push({ id, priority: "critical", at: entry.emitted_at, vialId: entry.vialId, vialLabel: null, message: `Pipeline error: ${entry.summary}`, origin: "ui", acknowledged: localAcks.has(id) });
  }
  return items.sort((a, b) => Number(a.acknowledged) - Number(b.acknowledged) || PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority] || b.at.localeCompare(a.at));
}

/** Rule receipt for a vial, from the final record or a standalone rule receipt. */
export function ruleReceiptFor(records: DecisionRecords | null, vialId: string): RuleReceipt | null {
  if (!records) return null;
  return records.finals.find(f => f.vial_id === vialId)?.rule ?? records.rules.find(r => r.vial_id === vialId) ?? null;
}

export function finalRecordFor(records: DecisionRecords | null, vialId: string): FinalRecord | null {
  return records?.finals.find(f => f.vial_id === vialId) ?? null;
}

export function modelRecordFor(records: DecisionRecords | null, vialId: string, key: SourceKey): ModelSourceRecord | null {
  return finalRecordFor(records, vialId)?.models.find(m => sourceKeyForModel(m.source_id) === key) ?? null;
}

/**
 * The ΔE plotted per vial on the trend: the largest stage ΔE76 between the target and either reference,
 * taken from the runner's rule receipt. Falls back to the streamed measurement ΔE. `null` if neither exists.
 */
export function vialDeltaE(vial: VialState, records: DecisionRecords | null): { readonly value: number | null; readonly origin: "rule_receipt" | "event" | null } {
  const rule = ruleReceiptFor(records, vial.vialId);
  const stageValues = rule?.stages.flatMap(s => [s.delta_e76_vial2, s.delta_e76_vial3]).filter((x): x is number => x != null) ?? [];
  if (stageValues.length) return { value: Math.max(...stageValues), origin: "rule_receipt" };
  const streamed = vial.decision?.measurement.delta_e ?? vial.measurement?.delta_e ?? null;
  return streamed != null ? { value: streamed, origin: "event" } : { value: null, origin: null };
}

/** Thresholds shown on the trend and the evidence grid (keep ≤ 1.0 < check ≤ 2.0 < discard). */
export const DELTA_E_KEEP_LIMIT = 1.0;
export const DELTA_E_DISCARD_LIMIT = 2.0;

/** Frame number encoded in an evidence or measurement ID such as `v4-CASE_x-F0060`; null if absent. */
export function frameFromId(id: string): number | null {
  const match = /-F(\d{3,6})$/.exec(id);
  return match ? Number(match[1]) : null;
}

/** Approximate CIELAB (D65) → sRGB hex for colour swatches. Display only; not a measurement. */
export function labToHex(lab: readonly [number, number, number]): string {
  const [L, a, b] = lab;
  const fy = (L + 16) / 116; const fx = fy + a / 500; const fz = fy - b / 200;
  const inv = (t: number) => (t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787);
  const X = 0.95047 * inv(fx); const Y = 1.0 * inv(fy); const Z = 1.08883 * inv(fz);
  const lin = [3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.204 * Y + 1.057 * Z];
  const gamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  return `#${lin.map(c => Math.round(Math.min(1, Math.max(0, gamma(c))) * 255).toString(16).padStart(2, "0")).join("")}`;
}

export function formatPercent(value: number | null): string {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

export function shortCommit(commit: string | null | undefined): string {
  return commit ? commit.slice(0, 7) : "unknown";
}
