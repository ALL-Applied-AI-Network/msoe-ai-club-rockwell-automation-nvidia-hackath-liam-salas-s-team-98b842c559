# WP7 image-grounded VLM decision boundary

Status: the CPU contract and synthetic tests are implemented. No V3/V4 development
freeze, VLM smoke, model decision or production result has been published.

`fluidmix.wp7_vlm_decision` scores only vial 1 in a complete three-vial cohort.
Vials 2 and 3 supply same-cohort reference measurements and images; their
deterministic dispositions are removed from the prompt payload. The component accepts
exactly three `wp7-camera-fill-step-receipt-v1` artifacts through the public
`validate_fill_step_receipt` boundary. It does not derive stream changes, select frames,
run a CNN or import Part-A geometry code.

Each accepted tracker stage supplies three distinct opaque early/mid/end image IDs,
tank identity, stream-step and fill-step agreement, stream Lab/spread, recipe amounts,
and a `twin_geometry_equivalent_volume_ml`. That volume is camera evidence interpreted
through frozen twin geometry; it is not a calibrated physical liquid volume or plant
truth. The tracker claim boundary and calibration, producer and receipt hashes remain
in the VLM bundle provenance. Local image paths and hashes are verified in a separate
locator binding and never enter prompt JSON or text.

The VLM response is strict JSON. One bounded three-backtick Markdown fence may be
removed; no other repair is allowed. Raw JSON names `disposition_confidence` (this
replaces the ambiguous raw `discard_score`). After all guards pass, a discard confidence
is copied to normalized `discard_score`, while a keep confidence is normalized as
`1 - disposition_confidence`; the receipt retains that normalized score.
Insufficient or guard-failed results have no vote. The deterministic source and each
admitted VLM have equal weight; two valid sources are required and a mean of exactly
0.5 is discard. Aggregation accepts only raw attempts from the three exact pinned
repository/revision pairs, rejects duplicate/unknown/caller-guarded rows, and performs
parse and guard internally.

The bundle recomputes target-to-vial-2 and target-to-vial-3 CIE76 distance from the
same-stage median Lab values. It also requires a closed, self-hashed deterministic
decision receipt binding every stage disposition and limit code to its evidence,
configuration hash and source-artifact hash; aggregation derives its one deterministic
vote from that bundle binding.

The development freeze accepts structured input manifests only. Every manifest must
identify V3 or V4, `synthetic_twin` origin, and `twin_train` or `twin_validation`; both
V3 and V4 must be present. Referenced artifacts are rehashed, while evaluation, test,
real and video-5 inputs are rejected. The freeze binds the decision source, prompt, response schema, parser,
aggregation constants, exact pinned model repositories/revisions, tracker receipts,
tracker calibration and producer hashes, and every named development input hash.
Production loading requires the reviewed freeze file's SHA-256 and validates it before
opening the production bundle path.

`run_sequential_vlm_runtime` is the executable adapter for the three models. It uses
the exact local-only pinned stock runtime, delivers the complete ordered early/mid/end
target plus vial-2/vial-3 image set without the generic plan loader's six-image cap,
routes raw text through parse/guard/aggregate, and releases each runtime reference
before loading the next model. Importing or testing this module does not load weights.

## Current evidence and remaining work

The accepted old WP7 endpoint route supported 0 of 18 development vials. That is the
baseline, not an empirical score of the replacement tracker. Accepted V3/V4 case trees
contain 18 and 30 fill videos, but the inspected published RGB/CNN artifacts do not
contain accepted through-time masks for all 48 fills. The earlier
`approach-a-predictions-v3.json` records only three development vials/six stage endpoints
and has status `prediction_gates_fail`; the V3 and V4 score files each cover six stage
predictions. Consequently this change does not fabricate the replacement numerator,
denominator or freeze.

The next authorized step is to generate hash-bound through-time CNN observations from
the accepted frozen checkpoint on V3/V4 only, run the dedicated tracker, publish its
before/after supported-vial receipt, and then create and review the VLM development
freeze. Synthetic unit tests prove fail-closed schemas and arithmetic only; they do not
validate camera accuracy, physical volume, carryover detection or model quality.
