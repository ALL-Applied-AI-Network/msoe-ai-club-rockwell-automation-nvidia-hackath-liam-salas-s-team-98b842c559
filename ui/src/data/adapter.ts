import Ajv from "ajv";

import schema from "../../../ui-data/schema.json";

export type ProvenanceKind = "measured" | "documented" | "inferred";
export type DecisionOutcome = "keep" | "discard" | "insufficient_evidence";
export type ExplanationSource = "deterministic_template" | "cosmos_reason2";

export interface Provenance {
  readonly kind: ProvenanceKind;
  readonly source_id: string;
  readonly note: string;
}

export interface Position {
  readonly x: number;
  readonly y: number;
  readonly provenance: Provenance;
}

export interface BoundingBox extends Position {
  readonly width: number;
  readonly height: number;
}

export interface TopDownObservation {
  readonly vial_id: string;
  readonly decision_id: string;
  readonly position: Position;
  readonly bbox: BoundingBox;
}

export interface EvidenceNumber {
  readonly value: number;
  readonly provenance: Provenance;
}

export interface Decision {
  readonly record_id: string;
  readonly outcome: DecisionOutcome;
  readonly delta_e: EvidenceNumber;
  readonly evidence_refs: readonly string[];
  readonly explanation: {
    readonly source: ExplanationSource;
    readonly text: string;
  };
}

export interface Vial {
  readonly vial_id: string;
  readonly recipe_id: string;
  readonly decision: Decision;
}

export interface EvidenceRecord {
  readonly evidence_id: string;
  readonly vial_id: string;
  readonly label: string;
  readonly provenance: Provenance;
}

export interface DemoData {
  readonly schema_version: "1.0.0";
  readonly dataset_id: string;
  readonly label: string;
  readonly data_origin: "synthetic_twin";
  readonly top_down: {
    readonly asset_id: string;
    readonly observations: readonly TopDownObservation[];
  };
  readonly vials: readonly Vial[];
  readonly evidence: readonly EvidenceRecord[];
}

const validateSchema = new Ajv({ allErrors: true }).compile(schema);

export class DemoDataValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DemoDataValidationError";
  }
}

function assertUniqueValues(
  values: readonly string[],
  field: string,
  collection: string
): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      throw new DemoDataValidationError(
        `Duplicate ${field} "${value}" in ${collection}`
      );
    }
    seen.add(value);
  }
}

function assertUniqueIdentifiers(data: DemoData): void {
  assertUniqueValues(data.vials.map((vial) => vial.vial_id), "vial_id", "vials");
  assertUniqueValues(
    data.top_down.observations.map((observation) => observation.vial_id),
    "vial_id",
    "top_down.observations"
  );
  assertUniqueValues(
    data.vials.map((vial) => vial.decision.record_id),
    "record_id",
    "decisions"
  );
  assertUniqueValues(
    data.evidence.map((record) => record.evidence_id),
    "evidence_id",
    "evidence"
  );
}

function assertEvidenceReferences(data: DemoData): void {
  const evidenceById = new Map(
    data.evidence.map((record) => [record.evidence_id, record])
  );
  for (const vial of data.vials) {
    for (const evidenceRef of vial.decision.evidence_refs) {
      const evidence = evidenceById.get(evidenceRef);
      if (!evidence) {
        throw new DemoDataValidationError(
          `Decision "${vial.decision.record_id}" references missing evidence "${evidenceRef}"`
        );
      }
      if (evidence.vial_id !== vial.vial_id) {
        throw new DemoDataValidationError(
          `Decision "${vial.decision.record_id}" for "${vial.vial_id}" references evidence "${evidenceRef}" owned by "${evidence.vial_id}"`
        );
      }
    }
  }
  const vialIds = new Set(data.vials.map((vial) => vial.vial_id));
  for (const evidence of data.evidence) {
    if (!vialIds.has(evidence.vial_id)) {
      throw new DemoDataValidationError(
        `Evidence "${evidence.evidence_id}" references unknown vial_id "${evidence.vial_id}"`
      );
    }
  }
}

function assertTopDownDecisionLinks(data: DemoData): void {
  const decisionsByVial = new Map(
    data.vials.map((vial) => [vial.vial_id, vial.decision.record_id])
  );
  for (const observation of data.top_down.observations) {
    const expectedDecision = decisionsByVial.get(observation.vial_id);
    if (!expectedDecision) {
      throw new DemoDataValidationError(
        `Top-down observation references unknown vial_id "${observation.vial_id}"`
      );
    }
    if (observation.decision_id !== expectedDecision) {
      throw new DemoDataValidationError(
        `Top-down observation for "${observation.vial_id}" links decision "${observation.decision_id}"; expected "${expectedDecision}"`
      );
    }
  }
  const observedVials = new Set(
    data.top_down.observations.map((observation) => observation.vial_id)
  );
  for (const vial of data.vials) {
    if (!observedVials.has(vial.vial_id)) {
      throw new DemoDataValidationError(
        `Vial "${vial.vial_id}" has no top-down observation`
      );
    }
  }
}

function assertGeometry(data: DemoData): void {
  for (const observation of data.top_down.observations) {
    const { bbox } = observation;
    if (bbox.x + bbox.width > 1 || bbox.y + bbox.height > 1) {
      throw new DemoDataValidationError(
        `Bounding box for "${observation.vial_id}" extends outside normalized image bounds`
      );
    }
    const { position } = observation;
    if (
      position.x < bbox.x ||
      position.x > bbox.x + bbox.width ||
      position.y < bbox.y ||
      position.y > bbox.y + bbox.height
    ) {
      throw new DemoDataValidationError(
        `Position for "${observation.vial_id}" lies outside its normalized bounding box`
      );
    }
  }
}

export function parseDemoData(input: unknown): DemoData {
  if (!validateSchema(input)) {
    const details = validateSchema.errors
      ?.map((error) => `${error.instancePath || "/"} ${error.message}`)
      .join("; ");
    throw new DemoDataValidationError(`Schema validation failed: ${details}`);
  }

  const data = input as unknown as DemoData;
  assertUniqueIdentifiers(data);
  assertEvidenceReferences(data);
  assertTopDownDecisionLinks(data);
  assertGeometry(data);
  return data;
}
