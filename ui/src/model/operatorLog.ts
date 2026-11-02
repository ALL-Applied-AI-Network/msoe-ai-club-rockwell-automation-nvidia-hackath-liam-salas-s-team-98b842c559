/**
 * Operator agree/override records. They are a human-in-the-loop log only: they never change the
 * system decision, and the live runner has no endpoint for them, so they live in this browser's
 * localStorage and are labelled that way wherever they are shown.
 */
import type { Disposition } from "../runner/contract";

export interface OperatorRecord {
  readonly record_id: string;
  readonly recorded_at: string;
  readonly session_id: string;
  readonly vial_id: string;
  readonly decision_id: string | null;
  readonly system_disposition: Disposition | null;
  readonly action: "agree" | "override";
  /** For overrides: what the operator believes is right. */
  readonly operator_disposition: Disposition | null;
  readonly reason: string;
  readonly operator: string;
}

const STORAGE_KEY = "fluidmix.operator-log.v1";

export function loadOperatorLog(): OperatorRecord[] {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY);
    const value: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(value) ? (value as OperatorRecord[]) : [];
  } catch {
    return [];
  }
}

export function saveOperatorLog(records: readonly OperatorRecord[]): void {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Storage can be unavailable (private mode); the in-memory log still works for this page.
  }
}

export function newRecordId(): string {
  return `op-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
