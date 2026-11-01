# WP7 image-grounded VLM decision boundary

Status: the CPU contract, one-model runtime CLI and synthetic tests are implemented.
This change does not run a GPU model or publish a model decision.

`fluidmix.wp7_vlm_decision` scores only vial 1 in a complete three-vial cohort.
Vials 2 and 3 supply same-cohort reference measurements and images; their
deterministic dispositions are removed from the prompt payload. The component accepts
exactly three `wp7-camera-fill-step-receipt-v1` artifacts through the public
`validate_fill_step_receipt` boundary. It does not derive stream changes, select frames,
run a CNN or import Part-A geometry code.

Each accepted tracker stage supplies three distinct opaque early/mid/end image IDs,
tank identity, stream-step and fill-step agreement, stream Lab/spread, recipe amounts,
an explicit `measured_volume_ml`, and a `twin_geometry_equivalent_volume_ml`. The latter
is camera evidence interpreted through frozen twin geometry; neither field is calibrated
physical liquid volume or plant truth. The tracker claim boundary and calibration,
producer and receipt hashes remain in the VLM bundle provenance. Local image paths and
hashes are verified in a separate locator binding and never enter prompt JSON or text.

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
the exact accepted configuration hash and source-artifact hash. Validation independently
replays the frozen CIE76 rule: a stage is `keep` only when both reference distances are at
most 1.0, `discard` only when both are at least 2.0, and otherwise
`insufficient_evidence`; any discard controls the overall result, while overall keep
requires every stage to be keep. Aggregation derives its one deterministic vote from that
validated bundle binding.

The v4 development freeze accepts structured input manifests only. Every manifest must
identify V3 or V4, `synthetic_twin` origin, and `twin_train` or `twin_validation`; both
V3 and V4 must be present. Referenced artifacts are rehashed, while evaluation, test,
real and video-5 inputs are rejected. Freeze creation also requires
`--materialization-receipt`; its self-hash, manifest hashes, tracker receipt hashes,
tracker source hash and deterministic configuration must match the supplied inputs. The
freeze binds that receipt's file SHA-256, the exact structured deterministic policy, the
decision source, prompt, response schema, parser, aggregation constants, exact pinned
model repositories/revisions, tracker receipts, tracker calibration and producer hashes,
and every named development input hash. Earlier freezes must be regenerated.
It also binds the stock adapter's exact effective generation configuration:
`do_sample=false`, `max_new_tokens=768`, and `batch_size=1`. Freeze validation rejects
any missing, additional or changed generation field. Each model-attempt receipt repeats
that structured configuration in its binding, and the one-model runner passes the bound
768-token limit explicitly to the stock adapter. Existing v3 freezes and attempts bound
to them must be regenerated; they cannot be migrated by editing their self-hashed JSON.
This freeze-time validation plus the exact receipt file hash is the transitive provenance
anchor: runtime requires the separately reviewed freeze file SHA-256, so changing the
materialization-receipt identity also changes the required reviewed freeze identity. Runtime
does not reopen the materialization receipt or accept a caller-supplied replacement.
Production loading requires the reviewed freeze file's SHA-256 and validates it before
opening the production bundle path.

`run_sequential_vlm_runtime` is the executable adapter for the three models. It uses
the exact local-only pinned stock runtime, delivers the complete ordered early/mid/end
target plus vial-2/vial-3 image set without the generic plan loader's six-image cap,
routes raw text through parse/guard/aggregate, and releases each runtime reference
before loading the next model. Importing or testing this module does not load weights.

For the real development speed path, `scripts/run_wp7_vlm_model.py` loads exactly one
model and publishes one no-clobber `wp7-vlm-model-attempt-receipt-v1`. It accepts the
path-free cohort, a separate internal image-locator file and exactly three tracker
receipts. Before model loading it verifies the caller-supplied reviewed freeze SHA-256,
the closed freeze contract and, in development mode, its exact frozen tracker-cohort
binding. `--production` instead validates that frozen policy before opening the new
cohort, requires complete production camera metadata, then binds the newly validated
tracker receipts into the attempt. The receipt retains
only prompt hashes, ordered image IDs/hashes, source-artifact hashes, the freeze hash,
the exact repository/revision and snapshot-inventory hash, raw response, latency and
memory telemetry; local image and snapshot paths are not published or prompted.

Qwen is the first allowed attempt. The first development Qwen receipt is retained as
failure evidence: generation completed at the old 384-token cap, but its raw JSON ended
mid image citation and therefore did not parse. The 768-token v4 limit is a bounded
correction to that observed truncation, not evidence that a future response will parse or
pass the guard. Cosmos Reason2 and AnomalyThink require a bound Qwen
attempt whose raw JSON parses and passes the existing response guard. All three use the
pinned local-only stock Qwen3-VL runtime with `trust_remote_code=False`. The separate
`scripts/aggregate_wp7_vlm_attempts.py` command is CPU-only: it reconstructs the bundle,
validates zero to three self-hashed attempt receipts against the exact same prompt,
images, freeze and sources, and then calls the existing equal-weight aggregator. A
malformed or guard-failed attempt remains excluded; it is never repaired into a vote.

## Current evidence and remaining work

The accepted tracker report now records 23 of 36 supported items and supplies three
complete three-vial cohorts to this boundary. Those are source-artifact observations,
not VLM accuracy or physical-volume results. This runtime does not reinterpret the
support denominator, select replacement frames, or create tracker observations.

The next runtime step is to review the development freeze hash, run Qwen on one accepted
cohort, and continue to Cosmos Reason2 or AnomalyThink only when the Qwen output parses
and passes the guard. Synthetic unit tests prove fail-closed schemas and arithmetic only;
they do not validate camera accuracy, physical volume, carryover detection or model
quality.
