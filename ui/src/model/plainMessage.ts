import { finalDisposition } from "./selectors";
import type { ExplanationPayload, VialState } from "../runner/contract";

/** Operator-readable reason for an insufficient-evidence result, from the runner's unavailable_reason code. */
export function cantAssessMessage(code: string | null): string {
  if (code === "measurement_stage_unavailable") return "Can't assess: the camera did not capture a usable measurement for every stage.";
  if (code === "tracker_source_profile_unavailable") return "Can't assess this video: camera setup not recognised";
  return "Can't assess: not enough reliable evidence.";
}

export function unavailableCode(vial: VialState): string | null {
  return vial.decision?.unavailable_reason ?? vial.average?.unavailable_reason ?? null;
}

export function isProducedExplanation(e: ExplanationPayload): boolean {
  return e.producer.source !== "deterministic_fallback" && e.producer.model_id !== "deterministic-template";
}

/** First model-written explanation, else the first (fallback) one, else null. */
export function pickExplanation(vial: VialState): { readonly item: ExplanationPayload; readonly fallback: boolean } | null {
  const list = vial.explanations.length ? vial.explanations : vial.explanation ? [vial.explanation] : [];
  const good = list.find(isProducedExplanation);
  if (good) return { item: good, fallback: false };
  return list[0] ? { item: list[0], fallback: true } : null;
}

export function isInsufficient(vial: VialState): boolean { return finalDisposition(vial) === "insufficient_evidence"; }
