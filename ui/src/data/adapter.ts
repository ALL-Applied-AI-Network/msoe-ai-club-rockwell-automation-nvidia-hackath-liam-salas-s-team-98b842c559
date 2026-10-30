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

export interface RgbValue {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly provenance: Provenance;
}

export interface LabValue {
  readonly l: number;
  readonly a: number;
  readonly b: number;
  readonly provenance: Provenance;
}

export interface AssetReference {
  readonly asset_id: string;
  readonly src: string;
  readonly provenance: Provenance;
}

export interface Explanation {
  readonly source: ExplanationSource;
  readonly text: string;
  readonly evidence_refs: readonly string[];
}

export interface Decision {
  readonly record_id: string;
  readonly outcome: DecisionOutcome;
  readonly delta_e: EvidenceNumber;
  readonly threshold: EvidenceNumber;
  readonly limit_codes: readonly (
    | "within_delta_e_limit"
    | "exceeds_delta_e_limit"
    | "measurement_unavailable"
    | "mask_unavailable"
    | "mock_quality_gate_unavailable"
  )[];
  readonly evidence_refs: readonly string[];
  readonly explanation: Explanation;
}

interface VialWire {
  readonly slot: number;
  readonly vial_id: string;
  readonly recipe_id: string;
  readonly rgb_frame: AssetReference;
  readonly predicted_mask: AssetReference;
  readonly target_lab: LabValue;
  readonly reference_lab: LabValue;
  readonly decision: Decision;
  readonly history_refs: readonly string[];
}

export interface EvidenceRecord {
  readonly evidence_id: string;
  readonly vial_id: string;
  readonly target:
    | "recipe_rgb"
    | "rgb_frame"
    | "predicted_mask"
    | "target_lab"
    | "reference_lab"
    | "delta_e"
    | "threshold";
  readonly label: string;
  readonly provenance: Provenance;
}

export interface RecipeViewModel {
  readonly recipe_id: string;
  readonly label: string;
  readonly rgb: RgbValue;
}

export interface HistoryViewModel {
  readonly history_id: string;
  readonly recipe_id: string;
  readonly run_id: string;
  readonly vial_id: string;
  readonly time_seconds: EvidenceNumber;
  readonly decision: DecisionOutcome;
  readonly explanation: Explanation;
  readonly provenance: Provenance;
}

export interface PlaybackCueViewModel {
  readonly vial_id: string;
  readonly slot: number;
  readonly start_seconds: number;
  readonly end_seconds: number;
  readonly provenance: Provenance;
}

export interface PlaybackViewModel {
  readonly run_id: string;
  readonly title: string;
  readonly video: {
    readonly asset_id: string;
    readonly src: string;
    readonly media_type: "video/mp4";
    readonly duration_seconds: EvidenceNumber;
    readonly provenance: Provenance;
  };
  readonly cues: readonly PlaybackCueViewModel[];
}

export interface VialViewModel {
  readonly slot: number;
  readonly vial_id: string;
  readonly recipe: RecipeViewModel;
  readonly rgb_frame: AssetReference;
  readonly predicted_mask: AssetReference;
  readonly target_lab: LabValue;
  readonly reference_lab: LabValue;
  readonly delta_e: EvidenceNumber;
  readonly threshold: EvidenceNumber;
  readonly decision: Decision;
  readonly evidence: readonly EvidenceRecord[];
  readonly history: readonly HistoryViewModel[];
  readonly top_down: TopDownObservation;
  readonly cue: PlaybackCueViewModel;
}

export type Wp6Method =
  | "template"
  | "qwen3_vl"
  | "anomalythink_free_form"
  | "cosmos_strict_parse";

export interface Wp6RowViewModel {
  readonly method: Wp6Method;
  readonly label: string;
  readonly passed: number;
  readonly total: 16;
  readonly method_note: string;
  readonly failure_note: string | null;
  readonly provenance: Provenance;
}

interface DemoDataWire {
  readonly schema_version: "1.0.0";
  readonly dataset_id: string;
  readonly label: string;
  readonly data_origin: "synthetic_twin";
  readonly run: {
    readonly run_id: string;
    readonly title: string;
    readonly video: PlaybackViewModel["video"];
    readonly cue_times: readonly Omit<PlaybackCueViewModel, "slot">[];
  };
  readonly recipes: readonly RecipeViewModel[];
  readonly history: readonly HistoryViewModel[];
  readonly top_down: {
    readonly state: "waiting" | "feed";
    readonly asset: AssetReference | null;
    readonly observations: readonly TopDownObservation[];
  };
  readonly vials: readonly VialWire[];
  readonly evidence: readonly EvidenceRecord[];
  readonly wp6: {
    readonly disclaimer: "limited all-abstention benchmark, no accuracy claim";
    readonly rows: readonly Wp6RowViewModel[];
    readonly inn67: {
      readonly status: "placeholder";
      readonly label: "INN-67 placeholder";
      readonly note: string;
    };
  };
}

export interface DemoData {
  readonly schema_version: "1.0.0";
  readonly dataset_id: string;
  readonly label: string;
  readonly data_origin: "synthetic_twin";
  readonly playback: PlaybackViewModel;
  readonly top_down: DemoDataWire["top_down"];
  readonly vials: readonly VialViewModel[];
  readonly wp6: DemoDataWire["wp6"];
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

function assertUniqueIdentifiers(data: DemoDataWire): void {
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
  assertUniqueValues(
    data.recipes.map((recipe) => recipe.recipe_id),
    "recipe_id",
    "recipes"
  );
  assertUniqueValues(
    data.history.map((record) => record.history_id),
    "history_id",
    "history"
  );
  assertUniqueValues(
    data.run.cue_times.map((cue) => cue.vial_id),
    "vial_id",
    "run.cue_times"
  );
}

function assertEvidenceReferences(data: DemoDataWire): void {
  const evidenceById = new Map(
    data.evidence.map((record) => [record.evidence_id, record])
  );
  for (const vial of data.vials) {
    const citedTargets = new Set<EvidenceRecord["target"]>();
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
      citedTargets.add(evidence.target);
    }
    if (
      !citedTargets.has("rgb_frame") ||
      !citedTargets.has("predicted_mask")
    ) {
      throw new DemoDataValidationError(
        `Decision "${vial.decision.record_id}" must cite rgb_frame and predicted_mask evidence targets`
      );
    }
    if (
      vial.decision.evidence_refs.join("\0") !==
      vial.decision.explanation.evidence_refs.join("\0")
    ) {
      throw new DemoDataValidationError(
        `Decision "${vial.decision.record_id}" evidence_refs do not match its explanation`
      );
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

  const vialsByHistory = new Map<string, Set<string>>();
  for (const vial of data.vials) {
    for (const historyRef of vial.history_refs) {
      const linkedVials = vialsByHistory.get(historyRef) ?? new Set<string>();
      linkedVials.add(vial.vial_id);
      vialsByHistory.set(historyRef, linkedVials);
    }
  }
  for (const history of data.history) {
    if (history.explanation.evidence_refs.length === 0) {
      throw new DemoDataValidationError(
        `History "${history.history_id}" must cite at least one evidence record`
      );
    }
    const linkedVials = vialsByHistory.get(history.history_id) ?? new Set<string>();
    for (const evidenceRef of history.explanation.evidence_refs) {
      const evidence = evidenceById.get(evidenceRef);
      if (!evidence) {
        throw new DemoDataValidationError(
          `History "${history.history_id}" references missing evidence "${evidenceRef}"`
        );
      }
      if (!linkedVials.has(evidence.vial_id)) {
        throw new DemoDataValidationError(
          `History "${history.history_id}" references evidence "${evidenceRef}" outside its linked vial context`
        );
      }
    }
  }
}

function assertTopDownDecisionLinks(data: DemoDataWire): void {
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

function assertGeometry(data: DemoDataWire): void {
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

function assertDownstreamContract(data: DemoDataWire): void {
  if (data.top_down.state === "feed" && data.top_down.asset === null) {
    throw new DemoDataValidationError(`Top-down feed state requires an asset`);
  }
  if (data.top_down.state === "waiting" && data.top_down.asset !== null) {
    throw new DemoDataValidationError(`Top-down waiting state must not reference an asset`);
  }
  if (data.vials.length !== 9) {
    throw new DemoDataValidationError(
      `Expected exactly 9 vial slots; received ${data.vials.length}`
    );
  }
  const orderedSlots = data.vials.map((vial) => vial.slot).sort((a, b) => a - b);
  if (orderedSlots.join(",") !== "1,2,3,4,5,6,7,8,9") {
    throw new DemoDataValidationError(
      `Vial slots must be unique and cover 1 through 9`
    );
  }

  const recipeIds = new Set(data.recipes.map((recipe) => recipe.recipe_id));
  const historyById = new Map(
    data.history.map((record) => [record.history_id, record])
  );
  const vialsById = new Map(data.vials.map((vial) => [vial.vial_id, vial]));
  const duration = data.run.video.duration_seconds.value;

  for (const vial of data.vials) {
    if (!recipeIds.has(vial.recipe_id)) {
      throw new DemoDataValidationError(
        `Vial "${vial.vial_id}" references missing recipe "${vial.recipe_id}"`
      );
    }
    // Phase 1 decisions are deterministic: equality is within the limit.
    const decisionLimits = vial.decision.limit_codes.filter((code) =>
      code === "within_delta_e_limit" ||
      code === "exceeds_delta_e_limit" ||
      code === "measurement_unavailable"
    );
    const expectedLimit = vial.decision.outcome === "keep"
      ? "within_delta_e_limit"
      : vial.decision.outcome === "discard"
        ? "exceeds_delta_e_limit"
        : "measurement_unavailable";
    if (decisionLimits.length !== 1 || decisionLimits[0] !== expectedLimit) {
      throw new DemoDataValidationError(
        `Decision "${vial.decision.record_id}" outcome "${vial.decision.outcome}" requires limit code "${expectedLimit}"`
      );
    }
    if (
      (vial.decision.outcome === "keep" && vial.decision.delta_e.value > vial.decision.threshold.value) ||
      (vial.decision.outcome === "discard" && vial.decision.delta_e.value <= vial.decision.threshold.value)
    ) {
      throw new DemoDataValidationError(
        `Decision "${vial.decision.record_id}" contradicts Delta E ${vial.decision.delta_e.value} and threshold ${vial.decision.threshold.value}`
      );
    }
    for (const historyRef of vial.history_refs) {
      const history = historyById.get(historyRef);
      if (!history) {
        throw new DemoDataValidationError(
          `Vial "${vial.vial_id}" references missing history "${historyRef}"`
        );
      }
      if (history.recipe_id !== vial.recipe_id) {
        throw new DemoDataValidationError(
          `History "${historyRef}" does not match recipe "${vial.recipe_id}"`
        );
      }
    }
  }

  if (data.run.cue_times.length !== 9) {
    throw new DemoDataValidationError(
      `Expected exactly 9 playback cues; received ${data.run.cue_times.length}`
    );
  }
  for (const cue of data.run.cue_times) {
    if (!vialsById.has(cue.vial_id)) {
      throw new DemoDataValidationError(
        `Playback cue references unknown vial_id "${cue.vial_id}"`
      );
    }
    if (cue.start_seconds >= cue.end_seconds || cue.end_seconds > duration) {
      throw new DemoDataValidationError(
        `Playback cue for "${cue.vial_id}" has invalid timing`
      );
    }
  }
  const orderedCues = [...data.run.cue_times].sort(
    (left, right) =>
      requireLink(vialsById.get(left.vial_id), "Missing cue vial").slot -
      requireLink(vialsById.get(right.vial_id), "Missing cue vial").slot
  );
  for (let index = 1; index < orderedCues.length; index += 1) {
    if (orderedCues[index].start_seconds < orderedCues[index - 1].end_seconds) {
      throw new DemoDataValidationError(
        `Playback cue for "${orderedCues[index].vial_id}" overlaps the previous vial slot`
      );
    }
  }

  const expectedWp6: readonly [Wp6Method, number][] = [
    ["template", 16],
    ["qwen3_vl", 16],
    ["anomalythink_free_form", 0],
    ["cosmos_strict_parse", 0]
  ];
  if (
    data.wp6.rows.length !== expectedWp6.length ||
    data.wp6.rows.some(
      (row, index) =>
        row.method !== expectedWp6[index][0] ||
        row.passed !== expectedWp6[index][1] ||
        row.total !== 16
    )
  ) {
    throw new DemoDataValidationError(
      `WP6 rows must be template 16/16, Qwen3-VL 16/16, AnomalyThink 0/16, and Cosmos 0/16`
    );
  }
  for (const row of data.wp6.rows) {
    if (row.passed === 0 && !row.failure_note) {
      throw new DemoDataValidationError(
        `WP6 row "${row.method}" requires a failure note for 0/16`
      );
    }
  }
}

export function explanationSourceLabel(source: ExplanationSource): string {
  return source === "deterministic_template"
    ? "Deterministic template"
    : "Cosmos-Reason2";
}

function requireLink<T>(value: T | undefined, message: string): T {
  if (value === undefined) {
    throw new DemoDataValidationError(message);
  }
  return value;
}

function mapViewModel(data: DemoDataWire): DemoData {
  const recipes = new Map(data.recipes.map((recipe) => [recipe.recipe_id, recipe]));
  const history = new Map(data.history.map((record) => [record.history_id, record]));
  const evidence = new Map(
    data.evidence.map((record) => [record.evidence_id, record])
  );
  const observations = new Map(
    data.top_down.observations.map((record) => [record.vial_id, record])
  );
  const slots = new Map(data.vials.map((vial) => [vial.vial_id, vial.slot]));
  const cues = data.run.cue_times
    .map((cue) => ({
      ...cue,
      slot: requireLink(slots.get(cue.vial_id), `Missing slot for "${cue.vial_id}"`)
    }))
    .sort((left, right) => left.slot - right.slot);
  const cuesByVial = new Map(cues.map((cue) => [cue.vial_id, cue]));

  const vials = [...data.vials]
    .sort((left, right) => left.slot - right.slot)
    .map<VialViewModel>((vial) => ({
      slot: vial.slot,
      vial_id: vial.vial_id,
      recipe: requireLink(
        recipes.get(vial.recipe_id),
        `Missing recipe "${vial.recipe_id}"`
      ),
      rgb_frame: vial.rgb_frame,
      predicted_mask: vial.predicted_mask,
      target_lab: vial.target_lab,
      reference_lab: vial.reference_lab,
      delta_e: vial.decision.delta_e,
      threshold: vial.decision.threshold,
      decision: vial.decision,
      evidence: vial.decision.explanation.evidence_refs.map((evidenceRef) =>
        requireLink(evidence.get(evidenceRef), `Missing evidence "${evidenceRef}"`)
      ),
      history: vial.history_refs.map((historyRef) =>
        requireLink(history.get(historyRef), `Missing history "${historyRef}"`)
      ),
      top_down: requireLink(
        observations.get(vial.vial_id),
        `Missing top-down observation for "${vial.vial_id}"`
      ),
      cue: requireLink(
        cuesByVial.get(vial.vial_id),
        `Missing playback cue for "${vial.vial_id}"`
      )
    }));

  return {
    schema_version: data.schema_version,
    dataset_id: data.dataset_id,
    label: data.label,
    data_origin: data.data_origin,
    playback: {
      run_id: data.run.run_id,
      title: data.run.title,
      video: data.run.video,
      cues
    },
    top_down: data.top_down,
    vials,
    wp6: data.wp6
  };
}

export function parseDemoData(input: unknown): DemoData {
  if (!validateSchema(input)) {
    const details = validateSchema.errors
      ?.map((error) => `${error.instancePath || "/"} ${error.message}`)
      .join("; ");
    throw new DemoDataValidationError(`Schema validation failed: ${details}`);
  }

  const data = input as unknown as DemoDataWire;
  assertUniqueIdentifiers(data);
  assertEvidenceReferences(data);
  assertTopDownDecisionLinks(data);
  assertGeometry(data);
  assertDownstreamContract(data);
  return mapViewModel(data);
}
