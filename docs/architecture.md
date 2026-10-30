# Pipeline interfaces and units

The repository implements physics helpers plus a synthetic V3 path for stage-image
measurement, conservative three-vial triage and local Cosmos evaluation. Those contracts are
described below. It does **not** contain a deployable RGB-only liquid-mask estimator, a trained
camera detector, a frozen held-out real-transfer result or an operator-facing explanation
service. The later plant-facing stage handoffs remain proposals for INN-11 to review, not
existing APIs. See [plan](plan.md) and [proposal](proposal.md).

## Implemented contracts

### INN-67 predicted-store source adapter

`src/fluidmix/inn67_predicted_store_producer.py` is the CPU-only producer paired with the
INN-67 reference-bank materializer. It requires the complete accepted 18-case development
render inventory and deterministically derives each same-case prefill RGB and planned stage
RGB from the accepted renderer receipt. Receipt case/plan identity and every selected path and
hash must agree; hand-authored RGB locator manifests are not accepted. It checks the frozen
CNN checkpoint, candidate receipt, deterministic replay and review identities before
inference. Only the receipt-declared current and prefill RGB enter the injected or frozen
six-channel predictor;
semantic masks, labels and programmed recipe volumes are not accepted inference fields.

The producer publishes a fresh no-clobber SQLite file, an
`inn67_predicted_evidence_store_population` receipt, and the exact
`inn67_predicted_store_export` consumed by `inn67_bank_materializer`. Event cursors preserve
plan order within B01, B02 and B03, with all 32 donor observations preceding the 16 B03
configuration observations. The public API has no predictor-injection argument: accepted
publication always loads the frozen CNN after checking the checkpoint, configuration, replay,
review and runtime-source identities. Tests replace that private loader only inside the test
process to prove boundary and schema composition; they do not establish CNN accuracy, physical
cleanliness or sim-to-real validity.

The runtime files `cnn_predictor.py` and `empty_reference_rgb_predictor.py` are byte-for-byte
the accepted implementations pinned by WP5 materializer commit
`d89a3f9fca31e7a2a37c89290189b3a5fa0583da`; the CNN source originated at
`a5f29ea90272a3424d8c7de09dbf435509d28105`. Their SHA-256 identities are checked before
checkpoint loading. PyTorch and torchvision retain their BSD-3-Clause terms; ImageNet
pretraining has separate terms recorded in the accepted WP5 lineage at `a5f29ea`.

All functions below live in src/fluidmix/absorbance.py. Vectors use R/G/B channel order.

| Function | Inputs | Output |
|---|---|---|
| srgb_to_linear | sRGB channels, 0–255 | Linear intensity, nominally 0–1 |
| linear_to_srgb | Linear intensity | sRGB floats clipped to 0–255 |
| volume_fractions | Tank volumes in mL | Fractions summing to one |
| mix_absorbance | Recipe and tank absorbance vectors | Dimensionless mixture absorbance |
| expected_color | Recipe, tanks, clear reference i0_lin | Linear intensity |
| carryover_absorbance | New/old recipes, tanks, old fraction 0–1 | Mixture absorbance |
| fit_tank_absorbance | List of (ml_dict, measured_linear_rgb) | tank_a, i0_lin, per-channel R² |
| fit_carryover_fraction | Measured intensity, recipes, tanks, reference | Old fraction and RMS absorbance residual |
| tank_absorbance_from_image | Tank/clear intensities, tank/vial path in mm | Absorbance rescaled to vial path |
| delta_e | Two sRGB 0–255 vectors | CIE76 color distance |

Recipes use t100, t200 and t400 keys. Absorbance is dimensionless; fraction 0.15 means 15%.
The API rejects unknown tanks, nonfinite/negative volumes and missing/nonfinite calibration
for used colored tanks. T-100 defaults to clear. Fitted residue fractions are still clipped
to [0,1]; matched references and physical validity require separate checks.
A clipped endpoint or small residual does not establish a causal diagnosis.
The forward carryover helper accepts one `numbers.Real` scalar in [0,1] (Python or NumPy
int/float, Fraction). It raises TypeError for bools, arrays (including 0-d and NumPy masked
values), NumPy time deltas and anything that is not a `numbers.Real` (for example Decimal,
complex, None or text), and ValueError for NaN, infinity or values outside [0,1], including
integers too large for a float. A per-channel fraction is not supported. See the
[acceptance audit](acceptance-inn-23-24.md#inn-11-follow-up-audit).

Calibration needs full-rank recipe fractions for the intercept and two colored tanks;
three vials alone do not guarantee identifiability. The fit rejects rank-deficient designs
and invalid intensity vectors, but can return negative absorbance for diagnostics.
See [physical limits](absorbance.md).

### Deterministic synthetic stage decision

`src/fluidmix/stage_decision.py` implements a decision-only boundary for three-stage,
three-vial synthetic batches. `decide_vial_stages` consumes pipeline-produced CIE Lab
statistics and explicit reference provenance; it does not read images, labels or Cosmos
output. Its frozen config supplies the peer-cluster and distance thresholds. A disposition
requires a unique tight peer mode containing at least two eligible reference-bank peers
from at least two batches. Later vials from the target batch may add context, but neither
same-batch evidence alone nor one independent reference batch can authorize `keep` or
`discard`.

The output is `keep`, `discard` or `insufficient_evidence`, with the target and peer evidence
IDs, per-stage delta E76, threshold, reference center and spread, and validity reason codes.
Target stages must belong to one vial and one partition, reference-bank rows cannot reuse
the target batch, and all references must match recipe, stage and split. The function has no
vial-position cleanliness rule and emits no operator-control advice. This is synthetic
colour-distance evidence only: the thresholds are not calibrated, and the result makes no
physical-carryover, defect, cause or sim-to-real claim.

### Verified stage-capture binding

`src/fluidmix/capture_manifest_adapter.py` binds one synthetic `vial_fill` batch of
three vials by three dispense events to the existing measurement and explanation
contracts. The current schema-v2 boundary opens and hash-checks one local Brev F2 runtime
receipt on Isaac Sim 6.2.0.0, and requires its runtime ID and source SHA to match the raw
capture declaration. The accepted receipt must report true scene-parity, finite-particle,
containment, zero-leakage, volume-height, stream/source-colour, numeric/visual-mixing,
semantics/alignment and timing gates. Its path and body remain artifact provenance and are
not copied into capture measurements, model inputs or measured evidence; those carry only
the receipt hash/runtime ID binding. Each event carries its ordinal, tank ID/order,
physics step
or frame ID, event-relative timestamp and final-observation offset. The older schema-v1
Isaac 6.0.1/time-code record remains readable as historical provenance but cannot
represent schema-v2 F2 rows. Missing, unreadable or hash-mismatched artifacts, duplicate
events, runtime drift and event drift fail binding with a named reason.

The realtime count-driven renderer also applies a fail-closed FillCam visible-height gate.
Its primary decision is the unchanged numeric height interval. A below-lower-bound surface
may pass only as `occlusion_censored` when the frozen source-stage, geometry-layer, camera
and active-carrier clamp raster all match and the dense liquid surface contacts the first
two-row release below a fully occluded acceptable band. The renderer reads back and hashes
the composed clamp mesh, topology, world transform and visibility after session edits;
source-file hashes alone are not treated as effective-geometry proof. Camera and clamp
local-to-world transforms are derived values: their raw states and hashes remain in the
receipt, while a versioned comparator requires every element of the finite 4x4 binary64
matrix to match the source-bound expected matrix with relative tolerance `1e-12` and
absolute tolerance `1e-12`. All camera authored
state and every nonmatrix clamp field remain exact. Exact count,
signed-volume, absolute body geometry, containment and body-height proofs remain required.
This handles the real carrier occlusion without moving scene parts or widening the gate; an
unoccluded or genuinely short fill still fails. Offline validation recomputes both legacy
and already-censored results from mask bytes, derives an in-memory receipt view, and never
rewrites the historical renderer receipt.

### INN-67 post-render producer

`src/fluidmix/inn67_postrender_producer.py` is the CPU-only handoff between the frozen
INN-67 development render plan and reference-bank materialization. It resolves exactly six
planned subruns and 18 cases beneath an explicit render root, verifies the exact frozen plan
and profile plus render-receipt, supervisor and accepted-validator hashes, and calls the
canonical
`validate_physx_f2_artifacts.validate_output` implementation for each case. Only after all
18 results pass does it publish 18 `inn67_occlusion_aware_artifact_validation` wrappers and
one `render-inventory.json` in a new output directory beneath the render root. Existing
outputs, symlinks, paths outside the render root, partial denominators and duplicate attempt
receipts fail closed.

The render-plan file SHA authenticates the frozen file while its canonical JSON SHA binds
the supervisor and runtime receipt. Real schema-20 receipts bind cases through
`arguments.train_case_id` and the execution schedule's model-input and renderer-case records;
they do not duplicate case, plan or known-good-profile hashes as top-level fields. The producer
also rehashes and exposes only the declared prefill and stage RGB files for a later deterministic
source manifest. Renderer semantic-mask paths remain validation provenance and are not projected
as inference inputs.

This producer does not render, alter completed case artifacts, export or populate an evidence
store, open WP1/video-5 data, or establish physical accuracy. A passing inventory is artifact
integrity evidence for the later CPU materializer only.

The count-driven qualification video's source-frame paths remain unique. Exact RGB bytes
may repeat only when every repeated capture is at or after fill end, the stream is hidden,
the liquid masks are byte-equal, and the saved effective-render fingerprint is equal. The
fingerprint binds hashed twin layers, the read-back camera world transform and projection,
render sampling, every reviewed light's effective transform/intensity/exposure, nuisance
state, the source-equivalent RGB glass/annotation state, the bound body mesh/material and
all authored OmniSurface inputs, plus visible stream material and cylinder geometry.
Active-pour
repeats fail even when those fingerprints happen to match. This narrow exception records
the repeat group and does not apply to the independently bound target/reference images
below.

### No-physics rigid vial transport artifact

The approved implementation boundary is the immutable 13-state authored overlay in
[`transport-animation-handoff.md`](transport-animation-handoff.md). It samples the
source twin's existing 1.2-second conveyor ease, composes every state at source time code
1, reconstructs the accepted outgoing liquid once, and keeps camera/light state fixed.
All 13 state layers and all 13 consumed snapshot arrays are bound before output and
revalidated after the independent checker. Transport frames, masks, arrays and receipts
remain Part A provenance and are excluded from model, measured-colour and plant inputs.

The earlier 2D compositor and anonymous-session renderer are rejected evidence. The 2D
route exposed occluded background and produced inpaint bands. The session route mixed
scene edits with Replicator's live `/Render` authoring; frozen commit `86596b6` still
omitted the 13 consumed arrays from its final evidence closure. Those implementations
must not be used or patched. Their reviewed mover allowlists, source distance measurements,
time-code-1 guard, count-body reconstruction and acceptance vocabulary are reused by the
authored-layer implementation.

### WP1 post-render acceptance boundary

`src/fluidmix/wp1_postrender.py` is the CPU-only consumer for the frozen two-run WP1
collection. It does not render fills or transport. The DP3 function opens the exact run-1
vial-5/vial-6 RGB/mask pairs named by accepted renderer receipts, requires the three ordered
T200/T400/T100 stage pairs and identical corresponding mask hashes, and applies the
pre-registered maximum same-stage CIE76 rule. Its decision binds both the canonical plan
hash and the different on-disk plan-file hash and never rewrites a source manifest. Invalid
or incomplete evidence publishes a new no-clobber `insufficient_evidence` decision with
reasons; output-publication failures remain errors.

When that original result is exactly `fallback_required`, the aggregate has one explicit
additive route: pass `--fallback-case-root` exactly twice for fresh 3.0 mL cases registered
against original run-1 target ordinals 5 (`S0440`) and 9 (`S0441`).
`scripts/wp1_dp3_fallback.py plan` deterministically creates two unrendered source-bound
one-vial plans accepted by the existing cohort supervisor. Each uses a same-recipe
predecessor, the exact frozen recipe volumes and tank palette, a distinct frozen seed and
nuisance ID, and a new source-derived case ID. The source plan's truthful renderer vial
ordinal is 1; ordinal 5 or 9 identifies the immutable original control, not a fabricated
position inside the one-vial execution.

After an authorized render and the existing offline validator, the script's `register`
phase requires the frozen manifest root and writes a closed
`fluidmix_wp1_dp3_fallback_case_acceptance_v3` receipt. It independently recomputes the
target bundle from the exact frozen run-1 plan and requires the locally copied source spec,
full render plan, model row and label provenance to equal that recomputation. The receipt
publishes file and canonical hashes for the frozen collection manifest and run-1 plan, then
binds the actual renderer
schedule and case binding, schedule/plan/file hashes, 3.0 mL residue, recipe context,
seed/nuisance identity, L40S/Isaac/Kit profile, existing offline result, and the camera,
light and nuisance attestations carried by the renderer receipt. A copied 1.5 mL receipt
or an internally consistent plan with changed volumes, run identity, source/split hashes,
palette, predecessor context, seed, nuisance identity or derived case ID cannot satisfy
these checks. The consumer repeats the frozen-run recomputation rather than trusting the
registration sidecar. It compares each fresh case with its immutable
original 1.5 mL control, requires identical corresponding masks and reports all three
same-stage CIE76 values under the frozen 2.0 threshold.

The frozen manifest does not say whether both fallback-pair maxima must pass or one maximum
across both pairs is enough. Therefore valid fallback evidence currently publishes
`approval_required` with both per-pair measurements; it cannot make aggregate WP1 pass
until that cross-case rule is explicitly approved. Missing, extra, duplicate, run-2,
wrong-recipe, wrong-context, wrong-residue, seed/nuisance-drifted, lineage-drifted,
hash-drifted or failed-validator evidence fails closed. Neither phase edits or substitutes
the original 18 cases, and plan creation does not authorize rendering or compute spend.

The aggregate validator deterministically rebuilds both child manifest triples, validates
the two complete nine-case roots, and requires separate leakage, input-boundary, actual
`fluidmix_wp1_fill_readiness_v1`/`fluidmix_wp1_l40s_host_profile_v1`, alternative-4
transport and lead-review evidence. The two real host names must differ, while their frozen
L40S, Isaac/Kit, source, stage, renderer and offline-validator identities must match. The
transport seam directly consumes each `rendered_transport_run_sequence_v2` composer receipt
and its ten hash-bound `fluidmix_transport_stage_render_receipt_v1` candidates. It checks
entry/between/exit null rules, exact accepted-fill endpoint hashes, decoded frame counts and
fps, encoded transition hashes, and independently recomputes the normal centering,
background, seam, full-exit and uniqueness gates. There is no separate WP1 transport
attestation that can replace those producer artifacts. The validator never repairs or
weakens transport evidence. Missing or unverifiable claims produce a failed
`wp1-acceptance.json`.

The dedicated packager accepts only an intact passing aggregate receipt. It copies the two
reviewed composed MP4 runs and the 18 accepted MJPG/AVI fills to the exact `videos/test/`
names without an implicit transcode. Composer and segment receipts, accepted fill clips and
lead-review inputs remain hash-bound under `review-inputs/`. The INDEX disposition and
preset columns are explicitly metadata and are not a model-input interface.

### Frozen Part A / Part B contracts

The pipeline now has two deliberately separate strict schemas. The flat per-stage
[`evidence-record-v1`](contracts/evidence-record-v1.md) is the Part B boundary: camera
artifacts, programmed recipe/run context, measurements, decisions, history bindings and
guarded explanation provenance. Storage can identify an oracle mask for segregated
synthetic regression, but production validation rejects it; writing `predicted` alone does
not create trusted model provenance.

The per-case [`sim-metadata-v1`](contracts/sim-metadata-v1.md) is Part A only: resolved
tank colour, renderer seed, prescribed residue and label, particle summary and host
profile. It is always excluded from model inputs. A hash-bound orchestration receipt joins
it to a case, and scoring-only evaluation may read that label only after inference. Part B
modules and prompts must not import or project the Part A contract. Complete simulation
metadata is published only at runtime case finalization; plan-time colour settings are not
filled out with placeholder residue, particle, label or host values.

#### Part ownership and read isolation

`src/fluidmix/part_boundary_manifest.py` is the authoritative reviewed inventory for this
boundary. It explicitly and disjointly registers current Part A-only modules, production
Part B roots, evaluation-only modules, the two frozen legacy-oracle diagnostic serializers,
and production/evaluation script entry points. The production inventory contains the
strict evidence validator, deploy-input policy, predicted-mask validator, trusted measurement
adapter, Part B projector and the WP7 carryover scaffold described below. Current synthetic capture adapters, model-input builders, direct
mask measurement, triage, stage decision and Cosmos scripts are registered evaluation-only;
classification does not turn their oracle-capable historical routes into deployable inputs.
The V3 `f2_fill_clip_inputs` adapter is also evaluation-only. It revalidates accepted raw
receipts and projects only hash-bound video/operator context, with active-fill end derived
by exact source-step bisection. Raw source-step arrays, their hashes, fill/stage physics
steps, simulation clocks and simulator/private fields remain outside the published
Part-B-safe manifest; external stage still indices are never treated as video frame indices.
The completed projection is checked against a closed recursive schema: root, denominator,
record, recipe-volume and stage objects admit only their declared fields and validated value
shapes. Unknown aliases fail closed rather than relying on a forbidden-word substring scan.
When cases are built beneath a larger atomic staging directory, `published_output_dir` names
the post-rename case root so persisted pixel locators do not retain the temporary path.
Every current Python module is accounted for; common byte-loading, schema and adjacent-analysis
helpers have an explicit shared-support role instead of being silently treated as production.
Explanation/history capabilities have no independent production roots: WP7 must reach measured
records through `part_b_evidence`, and absent support remains insufficient evidence.

The manifest's source audit follows ordinary local imports and literal
`importlib.import_module(...)` / `__import__(...)` calls transitively. Production roots and
active evaluation/model scripts fail the audit if they reach a Part A-only or frozen
legacy-oracle module; production additionally cannot reach evaluation-only modules, including
the label join. A fresh-interpreter finder repeats the check at import time so conditional or
runtime import behavior cannot silently weaken the static result. These checks establish code
ownership and dependency separation; they do not establish model accuracy or physical
validity.

Fresh-process read tests install an audit hook over resolved **exact** files and explicit
roots. The protected inputs are the preset catalog, supplied sim-metadata fixtures, and the
supplied Part A receipt and label roots. The guard does not reject filenames or prose merely
because they contain words such as `label`, `preset`, or `metadata`; ordinary camera RGB and
recipe files remain readable. Runtime callers must supply the actual receipt/label roots from
their orchestration binding rather than asking the guard to guess them from names.

`synthetic_decision_scoring` is the only registered post-inference label-join module. The
manifest authorizes that join only after a regular saved inference artifact matches its
declared SHA-256, and rechecks the bytes immediately before label access. No production root
may import that scorer. The two frozen diagnostic serializers remain golden-test inputs only;
active production, evaluation/model and future WP7 roots cannot import or invoke them. WP7 may
depend on `project_part_b_model_evidence_v1`, but its registration check rejects either legacy
serializer.

#### WP7 one-pass production scaffold

`scripts/run_carryover_pipeline.py` exposes `freeze-validation`, `run-test` and
`run-real-check`. Validation freezing accepts one exact validation-result schema and binds the
pipeline commit, recipe schema, mask runtime/config, measurement contract, frozen 2.0/1.0
decision config, DP5 choice, explanation method/model/prompt/guard, evidence-store schema and
validation denominators. The two run phases require the exact canonical freeze hash and create
distinct owner-tagged claims before opening camera-video or programmed-recipe bytes. Claims and
terminal receipts are atomic, exclusive and never reusable; the adjacent journal is append-only.

The programmed recipe schema contains only run/case/vial ordering, current and previous recipe
IDs, programmed tank millilitres and dispense order. The output binds the video, recipe, freeze,
claim and store identities and reports deterministic count equations. The predictor is a
separately pluggable live capability. The run CLI accepts exact checkpoint, replay, review and
camera-capture-receipt paths and hashes, then defers concrete frozen-CNN construction until the
controller has acquired the one-pass claim and bound the recipe/video inputs. The frozen runtime
configuration identity binds the accepted checkpoint and CNN configuration but is independent of
the phase capture receipt; receipt, video and origin remain phase-specific provenance. A missing
or freeze-mismatched runtime emits one `insufficient_evidence` record per vial. A real
check whose freeze has no accepted RGB-only route seals an explicit denominator-bound skip.
That explicit skip binds its 30-vial recipe and video evidence without loading a runtime.
These source and synthetic boundary tests do not constitute a validation freeze, held-out run,
camera measurement, detector result or physical-accuracy result.

`src/fluidmix/wp7_rgb_runtime.py` supplies the bounded frozen-CNN `PredictedMaskRuntime` used by
the one-pass controller. The controller invokes its live `analyze` capability, projects the
result through `TrustedMeasuredEvidenceResult` and the existing Part-B projector, then ingests
the measured records before leaving later history/decision work explicitly unavailable. The
runtime derives each half-open fill window from the composed transport receipt:
the inclusive start is the preceding encoded transition stop minus one, because that transition
already contains the first fill frame, and the exclusive stop is the following transition
start. The window must exactly equal the accepted fill frame count. That seam frame is also the
same-run pre-pour reference and must match the transition's decoded-frame hash. Fixed-nozzle-ROI
RGB change locates maximal stream bouts, and a stage endpoint is selected only when the first
post-bout CNN-visible body frame agrees with its next frame. Missing or count-ambiguous bouts and
missing or unstable endpoints abstain. Programmed volume is absent from this selection API.

The loader delegates to the existing reviewed CNN lineage loader after independently checking
the frozen checkpoint hash. The existing CNN continues to own the fixed vial ROI and clamp
exclusion. A separate `wp7-camera-rgb-mask-registry-v1` factory verifies the exact accepted
checkpoint, replay and review bytes plus each current RGB, same-run reference RGB and derived
mask locator. It reruns that loaded CNN on every bound current/reference pair and requires the
persisted mask pixels to match exactly. An all-black same-run reference rejects the sequence.
The receipt records truthful `synthetic_camera` or `real_camera` capture origin outside model
inputs and binds that origin into the registry capability and its hash. Provenance requires the
registry origin to equal the capture receipt. The factory validates exact complete per-case frame
coverage inside a multi-vial registry, and selector mask hashes use the persisted PNG-byte
representation. The terminal receipt publishes the resulting endpoint/measurement/runtime
provenance. Unknown, oracle, simulation, label and Part-A truth fields remain rejected. This is a
distinct capability; it does not relax or reinterpret the existing twin-only
`rgb_mask_prediction` schema.

The runtime measurement factory consumes only that opaque camera-RGB capability. It rechecks
all bound pixels, binary masks, contiguous frame/reference indices and the accepted vial ROI and
clamp exclusion. The existing `predict_fill_sequence` function remains the sole temporal
authority: only its visible-entry/growth, sustained pixel-stream and visible-exit evidence can
mark a CNN lower-body mask as `clamp_hidden_inferred`. Programmed volume is accepted only by
that pre-existing bounded interface and does not change the selected mask, surface or height;
synthetic changed-volume controls produce identical capability records. Empty masks and black
frames remain explicit abstentions. The module binds video, transport receipt, checkpoint,
configuration and code hashes plus per-vial reference/selected indices, mask hashes and
measurement citations. It does not call the decision or explanation routes and does not make a
camera-accuracy or physical-validity claim.

The frozen explanation method must be `deterministic_guarded_template` with model ID
`none`; another model cannot be selected as the WP7 DP4/default explanation backend through
validation metadata. Disposition logic and the frozen DP5 choice remain separate.
Every normal terminal receipt records that policy plus the completed limited all-abstention
diagnostic as non-comparative metadata. The learned-model rows are parse/abstention diagnostics,
not keep/discard accuracy, and the metadata therefore records both accuracy and winner as null.
The full comparison remains separate under INN-67 on Friday. This policy change does not alter
the accepted fill inputs, predictor interface or evidence-store contract.

`preflight_explanation_route` is a CPU-only policy boundary that runs before either one-pass
phase. `evaluation_only` requires the reviewed zero-row evaluation-store receipt, explicitly
selects 18 test abstentions and 30 real-check skips, and rejects any supplied history inputs.
`history_backed` never falls back: it rejects that evaluation store and requires an exact
reviewed RGB-only predicted-mask provenance record, an approval-bound production-history store
record, and nonempty factory-bound `HistoryQuery` values returned by the WP5 store. Those inputs
remain explicitly pending while INN-62 section 6 is unresolved. Enabling additionally requires a
separate canonical route-decision artifact, a caller-declared matching hash, exact INN-62/section
6 history scope and Liam (`@salasl`) approval; store approval is not route approval. The history
store record admits only camera RGB and predicted masks; evaluation/test query members and paths
with `oracle` or `sim_metadata` provenance are rejected.

Both one-pass APIs and CLI phases require the complete preflight receipt, its declared SHA-256
and the selected mode. The pipeline revalidates their exact shape, canonical hash, activation
state and 18/30 denominator semantics before loading the freeze, creating a claim, or reading
recipe/video bytes. Claim, every journal event, source bindings and terminal receipts carry the
preflight hash and selected mode. Before loading the preflight JSON, the CLI also resolves its
path and rejects an alias with the video, recipe, freeze, store, output, claim or derived journal
path. This preflight does not run a predictor or establish that a real route-decision artifact,
predicted-mask source or production-history store is approved.

The separate `development_reference_backed` route records Liam's INN-64 option-3 choice for
one WP1 pass and the video-5 one-shot against the frozen INN-67 development bank/config. It
accepts only factory `development_reference_coverage` queries from a separate minimal SQLite
store. Each query contains the explicit `development_reference` role and exactly one donor from
each of the two independent INN-67 donor batches. The normal `EvidenceStore` schema does not
admit this role, and the development route does not admit normal production-history queries. The route receipt binds
the exact accepted RGB-only CNN, predicted-store database/population tuple, development-bank,
frozen-config and materialization-receipt hashes. It also binds the accepted ordered 18-vial
video SHA-256 plus its composition receipt and final review identities, preserves the 18/30
denominators, and labels the evidence as synthetic rather than plant or physical truth.

`wp7_development_reference.validate_development_reference_inputs` validates the accepted
predicted-store database/population/export tuple and the materializer's bank/config/receipt
cross-bindings, including the receipt-derived source manifest. The materializer derives and
hash-verifies the 32 accepted RGB/mask locators, then publishes a fresh append-only store with
32 donor records, 16 frozen configurations and 16 pre-access coverage queries. It deliberately
omits generic fields that are neither published nor needed for donor citation: fill clips and
frame ranges, programmed recipe/order fields, MAD/IQR and pairwise measurements. Exact
evidence/image citations are available from every sealed query. Target-bound reference queries
require a factory-bound runtime target and select its donor configuration uniquely by recipe,
validated stage ordinal and tank. Runtime stage IDs remain target citations rather than donor
selectors, and cursors from the independent runtime and donor stores are not compared. The frozen
pre-access donor snapshot establishes chronology. Queries are computed read-only in memory so the
receipt-hashed donor SQLite snapshot never changes. Citation
members carry the upstream render/validation/plan/profile hashes and accepted predictor,
predicted-export, source-store and donor-snapshot bindings. Runtime
predictor/one-shot integration remains pending, so this foundation does not open WP1 or video-5.

#### WP7 development-history preparation (sealed)

`wp7_history_preparation` and `scripts/prepare_wp7_development_history.py` implement a
preparation-only path over the accepted synthetic twin development store. The builder verifies
the fixed receipt/database/review tuple, copies the database to a fresh owner-tagged root, makes
one same-run, strictly-earlier, train-only `vial_history` query per vial, and appends only
validated decision and explanation rows. All 18 vial decisions abstain as
`insufficient_evidence`; there is no threshold, peer-batch result, keep/discard result, cause,
defect, or plant action. The copied store contains 45 explained stage rows, while the population
receipt reports the 18 query hashes used separately from the store's total query-row count.

The separate preparation preflight binds that population, the exact reviewed ordered
18-vial video (7,322 frames, 732.2 seconds), and the reviewed INN-67 development-bank tuple.
Publication requires caller-declared hashes for the INN-67 materialization receipt, 32-record
two-donor reference bank, and 16-configuration frozen CIE76 development config. It verifies the
receipt's CPU/no-access claims, exact output hashes, shared provenance, B03-only selection gate,
two earlier peers per configuration, and development-only denominators before emitting the
canonical absolute-path `inn67_binding`. Hashing the already accepted video is a preparation
integrity check, not opening it as a one-shot execution input. The preflight remains
`sealed_pending_liam_decision`, records `one_shot_inputs_opened=false`, contains no route-decision
artifact, and is intentionally rejected by `_validate_execution_preflight`, `run_test`, and
`run_real_check` before claims or inputs are opened.

The v2 preparation receipt proves only that the accepted INN-67 inputs are bound; option-3
execution remains conditional on the separate Liam access decision. The historical fallback
record remains option 1: all 18 vials remain `insufficient_evidence`; option 2 is not approved.
None of these preparation records is production history, a held-out result, or permission to run
the one-pass phase.

#### Sealed RCA-bench model execution

`rca_bench_controller.run_model_phase` is the oracle-free execution boundary for the
remaining frozen WP6 comparison. It accepts the validated public plan, hash-verifying image
locators, digest-only freeze receipt, accepted Part B provenance, exact frozen source/config
paths, one local model snapshot, and new ledger/raw destinations. The runner has no oracle
argument. The scorer remains the first phase allowed to open hidden score data.

The runnable identities are pinned exactly: untuned `Qwen/Qwen3-VL-8B-Instruct` at
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, AnomalyThink at
`8331bc79ecf01e3466f0e6a27cec8c77f182f430`, and Cosmos-Reason2-8B at
`a9fae2cf89dc64db96b12860417f0eb403013bb9`. All three must declare
`Qwen3VLForConditionalGeneration`, model type `qwen3_vl`, and no `auto_map`; loading uses
the stock local Transformers classes with `trust_remote_code=False` and
`local_files_only=True`. The audited AD-Copilot load failure is not a runnable identity.

Before model loading, the controller rejects locator byte drift, frozen source/config drift,
repository/revision drift, a non-stock config, snapshot symlinks, or a snapshot inventory
digest that differs from `model_weight_sha256`. The inventory is the SHA-256 of canonical,
relative-path-ordered rows containing each regular file's relative path, byte length and
SHA-256. It covers immutable repository payload files and prunes only the snapshot root's
Hugging Face `.cache` control subtree, whose local acquisition metadata is not repository
content; payload symlinks and non-regular entries still fail closed. Every validation-only
row is rejected and journaled before runtime loading. Model-load wall time and reset CUDA
peak allocated/reserved bytes are bound into the ledger and each model record's runtime
provenance; unavailable CUDA values remain null with an explicit reason. Each
model row receives its frozen prompt and all four or six ordered images exactly once; an
item-level generation/decode error is sealed as `inference_error` without retry. A global
preflight/load failure seals the run as fatal/incomplete. The unchanged
`rca-bench-raw-v1` record binds ordered image identities/hashes, model identity, raw response,
latency, CUDA availability, snapshot inventory and processor/model classes. This is an
execution and provenance interface, not a model-quality, physical-accuracy or successful-GPU
result.

RGB and liquid masks are bound independently for every image. In the current V3 artifacts,
the masks are Isaac renderer semantic oracle masks aligned to synthetic RGB; they are not
predictions from camera RGB. They support synthetic measurement and label checks but are not
a deployable camera input. `rgb_mask_prediction.validate_rgb_mask_prediction_locators` now
defines the separate schema-v2 intake for future RGB-only predictions. It requires `input_fields`
to be exactly `["rgb"]`, binds frozen model/config hashes, code commit and validation-threshold
source, and rejects label, oracle, residual, palette and composition fields. Valid predicted masks
then use the same hash, decode, alignment and empty-mask checks; an unusable prediction makes the
vial decision `insufficient_evidence`. This is an implemented interface, not evidence that a model
has produced accurate masks: no RGB-only prediction artifact has been accepted or scored yet, and
oracle-mask results must not be reported as segmenter or detector performance. Image IDs and
artifact paths must be unique, and each declared hash must match its file. Equal content hashes
are reported rather than hidden:
repeated mask bytes are valid for the same fixed camera and stage geometry, while reuse of
target RGB bytes as a peer or duplicate RGB bytes within one peer set makes that reference
case `insufficient_evidence`. Per-image decode, shape/alignment, binary usability, coverage
and height-quality checks feed the measurement validity and limit codes. Runtime mask-quality
failures therefore cause the evaluator to abstain; artifact-integrity failures stop capture
binding.

For the accepted Varied V3 cohort, `build_f2_varied_stage_measurements.py` exposes this boundary
as `--predicted-image-locators`. The prediction artifact must contain exactly the image IDs from
the verified six-run capture manifests, and every RGB path and hash must match those manifests;
only the mask path/hash may be replaced. The output records the prediction-manifest hash and
producer provenance. Omitting the option preserves the existing semantic-mask diagnostic byte
contract. No accepted prediction artifact exists yet.

Schema v3, `rgb_temporal_liquid_mask_prediction_locators`, covers a pixel-only temporal predictor.
It adds an explicit path and SHA-256 for the same-vial pre-pour RGB. The V3 consumer verifies that
reference as a readable 640x480 RGB PNG, requires one stable reference binding per vial, and rejects
cross-vial reuse or aliasing to any stage RGB. Reference pixels are provenance for mask generation;
the colour measurement still uses only the stage RGB under the predicted mask. This closes the
second-input provenance gap but does not make the rejected temporal baseline deployable.

The per-batch `stage_capture_measurements` artifact is an intermediate record, not a
decision. `build_measured_evidence` requires the exact image set referenced by an
allowlisted model-input plan, binds every target and peer to its actual family, recipe,
partition, batch and stage, and then delegates to `vision_measurement.evaluate_model_plan`.
Its output is the canonical deterministic `measured_evidence` consumed by the stage
Cosmos boundary. Peer batches may differ, but recipe, family, partition and stage may
not; this prevents metadata from manufacturing the required independent reference
batches. Oracle/render parameters and labels are rejected from the capture manifest,
and pixel paths remain outside the deployable model-input rows.

`src/fluidmix/model_input_plan_builder.py` and
`scripts/build_model_input_plan_from_manifests.py` build that allowlisted plan from a
strict companion `stage_batch_context` plus verified capture manifests. The context
declares, rather than infers, each three-vial batch's `reference_only` or
`scored_target` role, run ID, previous/current recipe IDs, programmed tank volumes and
dispense order. One invocation is deliberately confined to one family, current recipe
and split. A disposition requires at least two external `reference_bank` batch IDs from
at least two external run IDs. When the bank is thinner, the plan remains measurable but
its primary case cannot borrow the
target batch to manufacture independence, so the deterministic evaluator abstains and
counts that vial. Reference batches sharing a run ID count as one source, and a reference
from the target run cannot establish external corroboration. With a sufficient bank,
later vials from the target batch are included as same-run evidence. Same-batch and
one-reference cases remain diagnostic, and only the full cross-run case can drive a
disposition.
Stage-event identities come from the manifests and must agree across all batches and with
the declared tank order. Cluster and threshold constants are serialized with a canonical
SHA-256 and checked before evaluation. `deploy_input_policy.py` owns the recursive field
allowlist: programmed recipe mL/order/sequence and pixel-derived evidence are allowed,
while composition, alpha, particle/source counts, expected disposition/volume/height,
render/light/exposure settings and nuisance IDs/seeds remain provenance-only. The plan
contains hashes and IDs but no pixel paths, render parameters or defect labels.

`scripts/build_stage_measured_evidence.py` consumes that plan and the same strict raw
manifests. It publishes two no-clobber artifacts: canonical `measured_evidence`, with
source hashes and no file paths, and a separate exact image-locator map required by the
existing stage-Cosmos input audit and runner. Publication validates the complete image
set before writing either artifact; an existing destination is never overwritten.

`scripts/score_synthetic_stage_decisions.py` is a separate post-hoc scoring boundary.
It opens immutable-hash-declared `measured_evidence` and the separated train-family
`label-provenance` artifact only after decisions exist. It requires the frozen 12-target
join (six prescribed keep and six prescribed discard), rejects missing, extra or duplicate
case/batch/vial identities and any oracle field in measured evidence, and reports keep/discard/
abstention confusion and coverage. The label file is never an input to measurement,
decision generation or Cosmos. These scores describe prescribed synthetic controls only;
they do not define a real rejection threshold or establish sim-to-real validity.

The frozen Nano comparison has a separate denominator contract of twelve vial decisions:
four three-vial target batches produce its 36 model attempts and 12 deterministic-template
controls. A smaller plan is still valid for deterministic keep/discard/abstain evidence,
but is not the frozen Nano evaluation cohort.
`scripts/build_stage_cosmos_attempt_plan.py` performs the next handoff without loading a
model: it publishes the exact 48-row prompt/image/control matrix and deterministic parse
and content-policy receipts for all 12 template controls. The processor audit, Nano runner
and scorer require that same canonical artifact, recompute it from the bound measured
evidence and prompt, and carry its SHA-256 into raw-result provenance. This prevents GPU
execution from silently rebuilding a different matrix. Passing the CPU preflight says
nothing about Nano's output quality or the physical validity of the synthetic evidence.

This contract verifies file identity, image shape, mask usability and evidence routing.
It does **not** establish physical-fluid accuracy, that any reference is clean in the real
plant, or that a synthetic decision transfers to real footage. For the varied-run V3 render
path, the count-driven body height from programmed volume is the image and label height of
record. A finite particle-height miss is recorded as
`diagnostic_warning_not_a_physics_pass`, while leakage, failed containment, missing/non-finite
artifacts, mask/render failures and contradictory receipts remain hard failures. Such a case
may be training-render eligible after validation, but it must not be called a particle-physics
pass or evidence of physical carryover or sim-to-real validity.

### Varied-run V3/V4 measurement and explanation boundary

`f2_varied_stage_measurement.py` consumes only corrected schema-3 capture manifests and their
hash-bound RGB/mask locators. It supports two or three active dispense stages, computes robust
masked Lab statistics for each of three vials and publishes all three pairwise delta E76 values
at each same-run stage. It does not open label, residual-volume or palette provenance.

`f2_varied_triage.py` compares each vial with the other two at the same stages without a vial-
position prior. It can record a relative outlier, a loose peer mode or consistency with a peer
mode, but a two-vial mode may itself be jointly affected. Without an independent clean anchor it
therefore emits `insufficient_evidence`; the relative assessment is diagnostic and never
authorizes keep or discard.

`f2_varied_explanation.py` and the current V3 Nano runtime form the explanation-only boundary. Numeric
values, units, stage/tank IDs and evidence IDs are copied from deterministic evidence; model text
owns only wording, abstention and a bounded evidence-collection suggestion. The lenient parser
does not repair or count raw strict format, and content checks still reject invented numbers or
IDs, unsupported cause/carryover claims and process-control advice. The accepted fallback is the
deterministic template whenever Nano fails those checks.

The attempt planner accepts exactly the supported split/count pairs: V3 has 18 decisions in six
three-vial runs and V4 has 30 decisions in ten three-vial runs. It rejects mixed decision versions,
crossed V3/V4 counts and denominator drift. V4 support is an interface contract only; no accepted
V4 Nano execution or model-quality result is claimed.

## Proposed stage handoffs

| Stage | Input evidence | Proposed output |
|---|---|---|
| Recipe/context | Recipe ID, tank mL, previous recipe, vial ID, corrected time | Aligned record with source IDs |
| Camera measurement | Camera/frame ID, liquid mask, matched I0, exposure checks | Linear RGB, optional uncalibrated absorbance, validity and uncertainty |
| Drainage model | Recipe sequence, tank volumes, predecessor at every dispense and residual-volume/flush assumptions | Volume drawn per tank and predicted leading-residue fraction; no camera input |
| Forward physics | Valid calibration, recipe/context, assumed residue fraction | Expected clean/carryover absorbance and intensity |
| Anomaly comparison | Clean-recipe expected value, measured value and frozen threshold | Score, threshold, flag and invalid-measurement status |
| Cause evidence | Valid intensity, current/previous recipes | Fitted fraction, residual and validity limits |
| Cosmos explanation | Quantitative evidence, recipe and uncertainty | Operator explanation linked to evidence IDs |

The tank-camera route needs measured path lengths and a suitable clear reference. The twin
has no tank images. Appearance estimates still need validation as transmission measurements.
Explanation must preserve invalid/missing evidence instead of advising plant changes from it.

### Proposed record contract for INN-11 review

These fields are a documentation proposal, not implemented classes, a serialized schema,
or team agreement. Revision 2 (2026-10-19) adapts it to the synthetic-only decision of
2026-10-16 in [evaluation](evaluation.md#partition-rules).

Every record carries `record_id`, source IDs, producer/config version, `validity`
(`valid`, `uncertain`, `invalid`), reason codes and an uncertainty description.
Use an explicit unavailable value/reason when uncertainty is not measured; do not imply zero.
Derived records link their upstream record IDs and calibration/model/threshold versions.

Every record also carries its partition, so evaluation code can refuse to mix sets:

| Field | Values | Rule |
|---|---|---|
| `data_origin` | `twin`, `real_check` | Twin renders vs real Dataset 2 video. Never inferred from file paths at scoring time |
| `split_role` | twin: `train`, `val`, `test`, `buffer`; real: `real_validation`, `calibration` | Copied from the recipe CSV bound by `split_version` (`sim_recipes.csv` for v2 or `sim_recipes_v3.csv` for WP1 v3), or from `data/real_check_windows.csv`; never assigned by a model at scoring time. `buffer` is never used |
| `split_version` | `inn-7-splits-v2`, `inn-7-splits-v3` | Must match the hash-bound split source for that artifact. Scoring must not pool records across versions |

A derived record inherits all three from its inputs. Inputs that disagree make it invalid
(reason `partition_mismatch`), because a mixed record could carry real or test information
into training or tuning. Twin test metrics and real-check results are reported apart.

| Record | Stage-specific fields and units |
|---|---|
| Recipe | Recipe ID (`sim_recipe_id` for twin), vial ID, tank volumes (mL), previous recipe ID or missing reason, video/frame IDs, corrected interval (s), timing-map version |
| Ground-truth label | Recipe ID, vial ID, label source, injected tank volumes (mL), per-dispense predecessor source and injected leading-residual volume (mL), previous recipe, injected fault type or `none`, defect flags from the frozen definitions |
| Measurement | Camera/ROI/reference IDs, exposure/quality checks, linear RGB (required, nominal 0–1), absorbance (optional, dimensionless) with `absorbance_status` (`uncalibrated` until a calibration is agreed), reference/path provenance (mm), measurement version |
| Drainage prediction | Recipe sequence and dispense IDs, preceding-dispense source, volume drawn per tank (mL), predicted leading-residue fraction (0–1), residual-volume/flush assumptions, drainage model and config version |
| Clean prediction | Recipe/measurement context, calibration ID, expected clean RGB and/or absorbance, prediction uncertainty and domain limits |
| Anomaly | Measurement and clean-prediction IDs, score definition/units, frozen threshold/config, decision or abstention reason |
| Cause | Anomaly ID, previous-recipe evidence, carryover fraction (0–1), RMS absorbance residual, clipping/identifiability status and alternative explanations |
| Explanation | Evidence IDs, reported status/limits and explanation-model version; no new quantitative or plant-action claim without supporting evidence |

Ground-truth labels are answers, not evidence. Twin labels come from the render parameters
passed to scripts/twin_liquid_variant.py, never from a colour score (label source
`render_parameters`); see [labels](labels.md). Video 5 has no defect labels, so real-check
records have none. Labels are stored apart from measurements; only scoring code reads them.
Models, thresholds and the drainage model never take a label as input.

Absorbance stays `uncalibrated` because real calibration is deferred and the only real fit
gives negative values (see [absorbance](absorbance.md)). An uncalibrated value can be shown
for diagnosis but cannot set a threshold, score a test set or support a cause claim.

The drainage prediction (INN-35) uses the recipe sequence only, so it is available even when a
camera measurement is invalid. It follows Liam's 2026-10-26 description: tanks are not premixed;
the first portion of each dispense is residue from its predecessor, followed by current-tank
fluid, and mixing occurs in the vial. Its residual-volume and flush values remain uncalibrated
assumptions. The drainage-only baseline (INN-34) and the cause stage may read the prediction;
the anomaly stage does not. A synthetic residual-volume value is label provenance and must not
be a deployed model input.

**Circularity risk.** scripts/twin_liquid_variant.py sets each twin vial's material from
`mix_absorbance` with guessed tank values. If the twin's clean prediction also came from
`expected_color` or `mix_absorbance` with those values, a twin comparison would test the
formula against itself and could look accurate even when the physics is wrong. Proposed
rule: in the twin, the clean prediction for a recipe is the measured colour of a clean
render of that recipe (zero injected residue and fault), extracted by the same measurement
code, with `calibration ID` set to `twin_clean_render` and the render IDs as sources. The
twin then tests detection, measurement and the drainage model under a known colour
mapping, not that mapping. Twin results cannot show that `fluidmix.absorbance` or the
guessed tank values describe real fluid. Only `real_check` results are evidence for or
against that, and the write-up must say so next to every twin metric.

Anomaly scoring must compare with the clean recipe. It must not subtract an estimated
carryover contribution before deciding whether the vial is anomalous. Missing previous
recipe blocks that carryover attribution, but need not block a clean-recipe comparison
when its own recipe, measurement, calibration and threshold are valid. Invalid camera or
reference evidence blocks absorbance inference; downstream records carry the invalid
state and reasons. An explanation cannot turn missing or invalid evidence into a cause.
INN-26 and INN-27 own implementation of those decisions and abstention behavior.

## Script handoffs

- `encode_palette_cut_reel.py`: an ordered JSON list of at least two V3/V4 run MP4s plus
  exact package- and run-receipt hashes -> one H.264 review reel with a fixed two-second
  labelled card between runs, an atomic receipt and `SHA256SUMS`. The accepted package's
  inventory must bind each run receipt, MP4 and `videos/INDEX.csv`; the index supplies a
  canonical palette identity and adjacent identities must differ. Inputs must be silent,
  contiguous CFR with identical geometry, frame rate and known codec profile. Decoded output
  is checked for source order, timing and exact H.264/yuv420p profile. The readable card also
  renders every length-delimited UTF-8 literal byte as an injective visible identity raster;
  every decoded card frame must exactly match an independently encoded canonical reference,
  with zero pixel tolerance. Source segments must be uniquely nearest their declared run with
  a fixed separation margin or the encoder abstains. Output is restricted to the centrally defined
  `outputs/part-a/palette-cut-reels` root, which every production Part-B read guard blocks;
  no output may overlap a bound accepted package. Execution requires a clean Git worktree,
  and the receipt binds Git commit/tree plus exact hashes for the module, CLI and local
  behavior-dependency closure. The encoder adds no transport frames and the
  per-run package remains the acceptance artifact. WP1 packages without immutable per-run
  palette identity are intentionally rejected until that field is added upstream.
- `part_boundary_manifest.py`: authoritative ownership inventory and reusable static import,
  fresh-import, exact-read, post-inference-join and future-WP7 gates. Its registered production
  path currently has no CLI; `build_model_input_plan_from_manifests.py`,
  `build_stage_measured_evidence.py`, `build_f2_varied_stage_measurements.py` and
  `decide_f2_varied_triads.py` remain synthetic evaluation entry points. All evaluation/model
  scripts and modules are listed in the manifest rather than inferred from filenames.
- measure_video_clock.py -> verify_still_alignment.py -> align_still_labels.py: videos, still
  indexes and video-5 recipes -> clock fits, still-to-frame map and aligned video-5 labels in
  FLUIDMIX_OUTPUTS. See [video timing](video_timing.md). fixlabels.py is legacy.
- color.py: video/recipe endpoints -> rough colors.json using fresh temporary frames.
  It does not directly export data/video5_vial_colors.csv.
- fit_absorbance_video5.py: committed CSVs -> stdout diagnostic, no deployable calibration.
- twin_liquid_variant.py: external station, recipe, tank JSON -> override .usda.
  Rendering is a separate external command; see [twin](twin.md).
