/**
 * Read-only evidence that the runner exposes outside the SSE stream.
 *
 * - `GET /api/sessions/{id}/decisions` exists on the live runner and returns the deterministic
 *   rule receipt plus a final record whose `sources` carry each model's structured vote,
 *   guarded explanation and per-stage comparison.
 * - `GET /api/sessions/{id}/evidence-bundle` and `.../evidence/images/{image_id}` are optional.
 *   Only the development replay tool (`ui/tools/replay_runner.py`) serves them today. A 404 means
 *   "not exposed by this runner" and the UI must say so instead of substituting anything.
 *
 * Parsers are tolerant: they keep fields they recognise and drop the rest, so a newer VLM schema
 * cannot crash the operator screen. Unknown or malformed records become `null`.
 */

export interface RuleStage {
  readonly stage_id: string;
  readonly evidence_id: string | null;
  readonly delta_e76_vial2: number | null;
  readonly delta_e76_vial3: number | null;
  readonly disposition: string;
  readonly limit_codes: readonly string[];
}

export interface RuleReceipt {
  readonly kind: "rule";
  readonly vial_id: string;
  readonly disposition: string;
  readonly discard_score: number | null;
  readonly stages: readonly RuleStage[];
  readonly limit_codes: readonly string[];
  readonly receipt_sha256: string | null;
  readonly config_sha256: string | null;
}

export interface StageComparison {
  readonly stage_id: string;
  readonly bundle_ids: readonly string[];
  readonly delta_e76_vial2: number | null;
  readonly delta_e76_vial3: number | null;
  readonly vial1_vs_vial2: string | null;
  readonly vial1_vs_vial3: string | null;
  readonly reliability: string | null;
  readonly phase_and_fill_match: string | null;
  readonly stream_tank_consistency: string | null;
  readonly previous_recipe_direction: string | null;
  readonly deterministic_stage_disposition: string | null;
}

/** One VLM source as recorded in the runner's final decision record. */
export interface ModelSourceRecord {
  readonly kind: "model";
  readonly source_id: string;
  readonly repository: string | null;
  readonly revision: string | null;
  readonly disposition: string;
  readonly confidence: number | null;
  readonly discard_score: number | null;
  readonly guard_status: string | null;
  readonly guard_reasons: readonly string[];
  /** Text shown to operators: the model's own text if admitted, otherwise the deterministic fallback. */
  readonly explanation: string | null;
  readonly explanation_status: string | null;
  readonly explanation_guard_reasons: readonly string[];
  /** The model's raw text, kept even when the guard rejected it (engineering view only). */
  readonly model_explanation: string | null;
  readonly observations: readonly { readonly stage_id: string; readonly visual_assessment: string }[];
  readonly per_stage_comparison: readonly StageComparison[];
  readonly image_citations: readonly string[];
  readonly evidence_citations: readonly string[];
  /** Optional VLM v2 fields; present only when the VLM agent's schema emits them. */
  readonly physics_cited: readonly string[];
  readonly elapsed_seconds: number | null;
}

export interface FinalRecord {
  readonly kind: "final";
  readonly vial_id: string;
  readonly final_disposition: string;
  readonly mean_discard_score: number | null;
  readonly valid_source_count: number | null;
  readonly weighting: string | null;
  readonly rule: RuleReceipt | null;
  readonly models: readonly ModelSourceRecord[];
}

export interface AbstentionRecord {
  readonly kind: "abstention";
  readonly vial_id: string;
  readonly disposition: string;
  readonly limit_codes: readonly string[];
}

export interface DecisionRecords {
  readonly rules: readonly RuleReceipt[];
  readonly finals: readonly FinalRecord[];
  readonly abstentions: readonly AbstentionRecord[];
  readonly unparsed: number;
}

/** Subset of the WP7 evidence bundle that the evidence grid and video overlay use. */
export interface BundleImage {
  readonly image_id: string;
  readonly stage_id: string;
  readonly vial_position: number;
  readonly phase: string | null;
  readonly crop_box_xyxy: readonly [number, number, number, number] | null;
  readonly source_native_dimensions: readonly [number, number] | null;
}
export interface BundleVialStage {
  readonly evidence_id: string;
  readonly stage_id: string;
  readonly vial_position: number;
  readonly evidence_role: string | null;
  readonly median_lab: readonly [number, number, number] | null;
  readonly delta_e76_vial2: number | null;
  readonly delta_e76_vial3: number | null;
  readonly deterministic_disposition: string | null;
}
export interface BundleStage {
  readonly stage_id: string;
  readonly stage_ordinal: number;
  readonly tank_id: string | null;
  readonly programmed_ml: number | null;
  readonly expected_cumulative_ml: number | null;
  readonly measured_volume_ml: number | null;
  readonly fill_agreement: boolean | null;
  readonly stage_timing_seconds: readonly [number, number] | null;
  readonly stream_median_lab: readonly [number, number, number] | null;
}
export interface EvidenceBundle {
  readonly cohort_id: string | null;
  readonly recipe_id: string | null;
  readonly vial_roi_xyxy: readonly [number, number, number, number] | null;
  readonly frame_size: readonly [number, number] | null;
  readonly frame_rate_hz: number | null;
  readonly stages: readonly BundleStage[];
  readonly vialStages: readonly BundleVialStage[];
  readonly images: readonly BundleImage[];
}

type Json = Record<string, unknown>;
const isObj = (v: unknown): v is Json => !!v && typeof v === "object" && !Array.isArray(v);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
function tuple<N extends number>(v: unknown, n: N): number[] | null {
  return Array.isArray(v) && v.length === n && v.every(x => typeof x === "number" && Number.isFinite(x)) ? (v as number[]) : null;
}

function parseRuleStage(v: unknown): RuleStage | null {
  if (!isObj(v) || !str(v.stage_id)) return null;
  return {
    stage_id: v.stage_id as string, evidence_id: str(v.evidence_id), delta_e76_vial2: num(v.delta_e76_vial2), delta_e76_vial3: num(v.delta_e76_vial3),
    disposition: str(v.disposition) ?? "unknown", limit_codes: strs(v.limit_codes)
  };
}

export function parseRuleReceipt(v: unknown): RuleReceipt | null {
  if (!isObj(v) || v.type !== "deterministic") return null;
  return {
    kind: "rule", vial_id: str(v.vial_id) ?? str(v.target_vial_id) ?? "unknown", disposition: str(v.disposition) ?? "unknown",
    discard_score: num(v.discard_score), stages: (Array.isArray(v.stages) ? v.stages : []).map(parseRuleStage).filter((s): s is RuleStage => !!s),
    limit_codes: strs(v.limit_codes), receipt_sha256: str(v.receipt_sha256), config_sha256: str(v.config_sha256)
  };
}

function parseComparison(v: unknown): StageComparison | null {
  if (!isObj(v) || !str(v.stage_id)) return null;
  return {
    stage_id: v.stage_id as string, bundle_ids: strs(v.bundle_ids), delta_e76_vial2: num(v.delta_e76_vial2), delta_e76_vial3: num(v.delta_e76_vial3),
    vial1_vs_vial2: str(v.vial1_vs_vial2), vial1_vs_vial3: str(v.vial1_vs_vial3), reliability: str(v.reliability),
    phase_and_fill_match: str(v.phase_and_fill_match), stream_tank_consistency: str(v.stream_tank_consistency),
    previous_recipe_direction: str(v.previous_recipe_direction), deterministic_stage_disposition: str(v.deterministic_stage_disposition)
  };
}

export function parseModelSource(v: unknown): ModelSourceRecord | null {
  if (!isObj(v) || !str(v.source_id)) return null;
  const telemetry = isObj(v.runtime_telemetry) ? v.runtime_telemetry : {};
  const physics = strs(v.physics_cited).length ? strs(v.physics_cited) : strs(v.physics);
  return {
    kind: "model", source_id: v.source_id as string, repository: str(v.repository), revision: str(v.revision),
    disposition: str(v.disposition) ?? "unknown", confidence: num(v.disposition_confidence) ?? num(v.vote), discard_score: num(v.discard_score),
    guard_status: str(v.guard_status), guard_reasons: strs(v.guard_reasons), explanation: str(v.explanation),
    explanation_status: str(v.explanation_status), explanation_guard_reasons: strs(v.explanation_guard_reasons),
    model_explanation: str(v.model_explanation),
    observations: (Array.isArray(v.observations) ? v.observations : []).flatMap(o => isObj(o) && str(o.stage_id) ? [{ stage_id: o.stage_id as string, visual_assessment: str(o.visual_assessment) ?? "unknown" }] : []),
    per_stage_comparison: (Array.isArray(v.per_stage_comparison) ? v.per_stage_comparison : []).map(parseComparison).filter((c): c is StageComparison => !!c),
    image_citations: strs(v.image_citations), evidence_citations: strs(v.evidence_citations), physics_cited: physics,
    elapsed_seconds: num(telemetry.elapsed_seconds)
  };
}

export function parseDecisionRecords(value: unknown): DecisionRecords {
  const rows = Array.isArray(value) ? value : [];
  const rules: RuleReceipt[] = []; const finals: FinalRecord[] = []; const abstentions: AbstentionRecord[] = [];
  let unparsed = 0;
  for (const row of rows) {
    if (!isObj(row)) { unparsed += 1; continue; }
    if (row.type === "deterministic") { const r = parseRuleReceipt(row); if (r) rules.push(r); else unparsed += 1; continue; }
    if (row.type === "abstention") {
      abstentions.push({ kind: "abstention", vial_id: str(row.vial_id) ?? "unavailable", disposition: str(row.disposition) ?? "insufficient_evidence", limit_codes: strs(row.limit_codes) });
      continue;
    }
    if (row.type === "final" || Array.isArray(row.sources)) {
      const sources = Array.isArray(row.sources) ? row.sources : [];
      finals.push({
        kind: "final", vial_id: str(row.vial_id) ?? "unknown", final_disposition: str(row.final_disposition) ?? "unknown",
        mean_discard_score: num(row.mean_discard_score), valid_source_count: num(row.valid_source_count), weighting: str(row.weighting),
        rule: sources.map(parseRuleReceipt).find((r): r is RuleReceipt => !!r) ?? null,
        models: sources.map(parseModelSource).filter((m): m is ModelSourceRecord => !!m)
      });
      continue;
    }
    unparsed += 1;
  }
  return { rules, finals, abstentions, unparsed };
}

export function parseEvidenceBundle(value: unknown): EvidenceBundle | null {
  if (!isObj(value)) return null;
  const camera = isObj(value.camera_metadata) ? value.camera_metadata : {};
  const stages: BundleStage[] = (Array.isArray(value.stages) ? value.stages : []).flatMap(s => {
    if (!isObj(s) || !str(s.stage_id)) return [];
    const fill = isObj(s.camera_fill_state) ? s.camera_fill_state : {};
    const stream = isObj(s.stream) ? s.stream : {};
    return [{
      stage_id: s.stage_id as string, stage_ordinal: num(s.stage_ordinal) ?? 0, tank_id: str(s.tank_id), programmed_ml: num(s.programmed_ml),
      expected_cumulative_ml: num(fill.expected_cumulative_ml), measured_volume_ml: num(fill.measured_volume_ml),
      fill_agreement: typeof fill.agreement === "boolean" ? fill.agreement : null,
      stage_timing_seconds: tuple(s.stage_timing_seconds, 2) as [number, number] | null,
      stream_median_lab: tuple(stream.median_lab, 3) as [number, number, number] | null
    }];
  });
  const vialStages: BundleVialStage[] = (Array.isArray(value.deterministic_evidence) ? value.deterministic_evidence : []).flatMap(e => {
    if (!isObj(e) || !str(e.evidence_id) || !str(e.stage_id) || num(e.vial_position) == null) return [];
    return [{
      evidence_id: e.evidence_id as string, stage_id: e.stage_id as string, vial_position: num(e.vial_position) as number,
      evidence_role: str(e.evidence_role), median_lab: tuple(e.median_lab, 3) as [number, number, number] | null,
      delta_e76_vial2: num(e.delta_e76_vial2), delta_e76_vial3: num(e.delta_e76_vial3), deterministic_disposition: str(e.deterministic_disposition)
    }];
  });
  const images: BundleImage[] = (Array.isArray(value.images) ? value.images : []).flatMap(i => {
    if (!isObj(i) || !str(i.image_id) || !str(i.stage_id) || num(i.vial_position) == null) return [];
    return [{
      image_id: i.image_id as string, stage_id: i.stage_id as string, vial_position: num(i.vial_position) as number, phase: str(i.phase),
      crop_box_xyxy: tuple(i.crop_box_xyxy, 4) as [number, number, number, number] | null,
      source_native_dimensions: tuple(i.source_native_dimensions, 2) as [number, number] | null
    }];
  });
  return {
    cohort_id: str(value.cohort_id), recipe_id: str(value.recipe_id),
    vial_roi_xyxy: tuple(camera.vial_roi, 4) as [number, number, number, number] | null,
    frame_size: tuple(camera.resolution, 2) as [number, number] | null, frame_rate_hz: num(camera.frame_rate_hz),
    stages: stages.sort((a, b) => a.stage_ordinal - b.stage_ordinal), vialStages, images
  };
}
