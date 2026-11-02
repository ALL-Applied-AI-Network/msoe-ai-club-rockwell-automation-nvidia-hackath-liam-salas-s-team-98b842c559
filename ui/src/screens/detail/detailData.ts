/** Pure data helpers for the vial detail screen. Everything comes from runner events, decision receipts or the evidence bundle. */
import type { StageSample, VialState } from "../../runner/contract";
import type { BundleStage, EvidenceBundle } from "../../runner/evidence";
import { DELTA_E_DISCARD_LIMIT, DELTA_E_KEEP_LIMIT, ruleReceiptFor } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";

export interface StageRef { readonly id: string; readonly label: string; readonly tank: string | null; readonly bundle: BundleStage | null }

const tankOf = (stageId: string): string | null => /-(t\d+)$/i.exec(stageId)?.[1].toLowerCase() ?? null;

/** Stages in order: bundle stages, else rule-receipt stages, else distinct tanks seen in the vial's history. */
export function stagesFor(model: InspectionModel, vial: VialState): readonly StageRef[] {
  if (model.bundle?.stages.length) {
    return model.bundle.stages.map(s => { const tank = s.tank_id ?? tankOf(s.stage_id); return { id: s.stage_id, label: (tank ?? s.stage_id).toUpperCase(), tank, bundle: s }; });
  }
  const rule = ruleReceiptFor(model.decisions, vial.vialId);
  if (rule?.stages.length) return rule.stages.map(s => { const tank = tankOf(s.stage_id); return { id: s.stage_id, label: (tank ?? s.stage_id).toUpperCase(), tank, bundle: null }; });
  const tanks = [...new Set(vial.history.map(h => h.tracker.tank_id).filter((t): t is string => !!t))];
  return tanks.map(t => ({ id: t, label: t.toUpperCase(), tank: t.toLowerCase(), bundle: null }));
}

export const samplesFor = (vial: VialState, stage: StageRef): readonly StageSample[] =>
  vial.history.filter(h => h.tracker.tank_id != null && h.tracker.tank_id.toLowerCase() === stage.tank);

/** Runner vial occupying cohort position 1..n relative to the selected target. */
export function vialAtPosition(model: InspectionModel, target: VialState, position: number): VialState | null {
  const runner = model.runner; if (!runner) return null;
  const start = Math.floor(target.position.tray_index / model.vialCount) * model.vialCount;
  const id = runner.trayOrder.find(v => runner.vials[v].position.tray_index === start + position - 1);
  return id ? runner.vials[id] : null;
}

export function bundleImageId(bundle: EvidenceBundle | null | undefined, stageId: string, position: number): string | null {
  const imgs = bundle?.images.filter(i => i.stage_id === stageId && i.vial_position === position) ?? [];
  return (imgs.find(i => i.phase === "late") ?? imgs.find(i => i.phase === "end") ?? imgs[imgs.length - 1])?.image_id ?? null;
}

export function bundleLab(bundle: EvidenceBundle | null | undefined, stageId: string, position: number) {
  return bundle?.vialStages.find(v => v.stage_id === stageId && v.vial_position === position)?.median_lab ?? null;
}

export function streamedLab(vial: VialState | null, stage: StageRef): readonly [number, number, number] | null {
  if (!vial) return null;
  const last = [...samplesFor(vial, stage)].reverse().find(s => s.measurement?.status === "measured" && s.measurement.lab);
  return last?.measurement?.lab ?? null;
}

export type Band = "keep" | "check" | "discard";
export const bandOf = (de: number): Band => (de <= DELTA_E_KEEP_LIMIT ? "keep" : de <= DELTA_E_DISCARD_LIMIT ? "check" : "discard");
export const BAND_TEXT: Readonly<Record<Band, string>> = { keep: "within keep limit", check: "check band", discard: "outside limits" };

export interface StageDeltaE { readonly v2: number | null; readonly v3: number | null; readonly origin: "rule receipt" | "bundle" }
export function deltaEFor(model: InspectionModel, vialId: string, stageId: string): StageDeltaE | null {
  const rule = ruleReceiptFor(model.decisions, vialId)?.stages.find(s => s.stage_id === stageId);
  if (rule && (rule.delta_e76_vial2 != null || rule.delta_e76_vial3 != null)) return { v2: rule.delta_e76_vial2, v3: rule.delta_e76_vial3, origin: "rule receipt" };
  const b = model.bundle?.vialStages.find(v => v.stage_id === stageId && v.vial_position === 1);
  return b && (b.delta_e76_vial2 != null || b.delta_e76_vial3 != null) ? { v2: b.delta_e76_vial2, v3: b.delta_e76_vial3, origin: "bundle" } : null;
}

export const fmt1 = (n: number | null | undefined): string => (n == null ? "—" : n.toFixed(1));
export const fmtLab = (lab: readonly number[]): string => `L* ${fmt1(lab[0])}  a* ${fmt1(lab[1])}  b* ${fmt1(lab[2])}`;
