# `evidence-record-v1`

`evidence-record-v1` is the frozen flat, per-stage Part B record. It contains camera- or
program-derived evidence only. It never contains a tank-colour preset, renderer palette,
residue label, particle state, host profile, or another simulation oracle. One vial is an
ordered sequence of two or three records, one after each active programmed dispense.

The JSON Schema is
[`src/fluidmix/contracts/evidence-record-v1.schema.json`](../../src/fluidmix/contracts/evidence-record-v1.schema.json).
It uses JSON Schema Draft 2020-12, requires every field below, and rejects extra fields at
every object boundary.

## Exact fields

| Group | Required fields | Meaning |
|---|---|---|
| Run and recipe | `run_id`, `case_id`, `vial_ordinal`, `block_recipe_id`, `block_position`, `recipe_id`, `previous_recipe_id`, `recipe_ml`, `dispense_order`, `split_role`, `split_version` | Production-run grouping and programmed recipe context. `recipe_ml` has exactly `t100`, `t200`, and `t400`; order contains exactly the positive-volume tanks. |
| Stage | `stage_id`, `tank_id`, `stage_ordinal` | State after one active tank dispense. Sequence ordinals are contiguous and tanks follow programmed order. |
| Clip | `fill_clip_path`, `fill_frame_range`, `transition_frame_ranges`, `clip_sha256` | A contained relative clip path and exact original-byte hash. Ranges are zero-based and half-open. Entry ends no later than fill start; exit starts no earlier than fill end. |
| Image | `image_id`, `rgb_sha256`, `mask_sha256`, `mask_source` | Exact original RGB and mask bytes. Storage supports `oracle` or `predicted`; production rejects oracle. Writing `predicted` does not establish trusted provenance. |
| Measurement | `measurement_id`, `median_lab`, `mad_lab`, `iqr_lab`, `pairwise_delta_e76`, `colour_regime_id` | Pixel-derived Lab statistics and CIE76 comparisons. Pairwise rows name another, non-self evidence ID. `colour_regime_id` remains `null` until WP5 supplies a separately reviewed measured binding. |
| Decision | `evidence_id`, `decision_id`, `disposition`, `limit_codes`, `history_refs` | A singular stage evidence ID, later decision, limits, and verified history references. Each history row is exactly `{evidence_id, query_sha256}`. |
| Explanation | `explanation_id`, `model_revision`, `prompt_sha256`, `image_ids_sent`, `guard_result` | Guarded explanation provenance. Visual attempts record the complete ordered four- or six-image delivery. A direct deterministic template records no images and uses `model_revision="deterministic-template:<revision>"`; its hash binds the exact canonical template input bytes. |

IDs are bounded, nonempty, and control-free. `image_id` is one safe filename component.
Hashes are lowercase SHA-256. Numeric values must be finite; MAD, IQR, and CIE76 values
are nonnegative. Supported split roles are `train`, `val`, `validation`, `evaluation`,
`test`, `real_validation`, and `calibration`. `buffer` is not rendered or scored and is
not a record role.

## Non-serialized phase and sequence rules

Callers pass `phase="measured"|"decided"|"explained"`; phase is not serialized.

- Measured records have measurement/evidence values, null decision/explanation values,
  and empty limit, history, and delivered-image lists.
- Decided records add a decision and disposition. They may add sorted limits and verified
  history, but explanation fields remain null.
- Explained records retain the decision and add explanation, revision, content hash,
  exact delivered-image list, and guard result.

`validate_evidence_sequence_v1` requires two or three rows with the same vial, recipe,
split, clip, decision/history, and explanation bindings. Stage, image, measurement, and
evidence IDs are unique. Stage ordinals and tanks exactly follow `dispense_order`.

Phase changes publish a new record; they never overwrite an earlier one. The exact
orchestration-only receipt is:

```text
schema_version = evidence-record-derivation-v1
predecessor_phase, derived_phase, predecessor_sha256, derived_sha256
```

Only `measured -> decided` and `decided -> explained` are allowed. The verifier hashes the
exact published bytes and checks that fields outside the new phase family are unchanged.
If record publication succeeds but receipt publication fails, an identical retry verifies
the existing bytes and publishes only the missing receipt. A mismatch fails closed.

## Artifact verification and trust boundary

The clip path is stored in the record. Segregated oracle regression storage uses
`images/<image_id>.png` and `masks/oracle/<image_id>.png`. Predicted storage instead uses
the exact paths and hashes returned by the existing locator validators, normalized to:

```text
Mapping[image_id, {rgb_path, rgb_sha256, mask_path, mask_sha256}]
```

All paths must remain below the authorized artifact root. Verifiers open regular files
through no-follow directory descriptors and hash original bytes; they do not transcode or
rename masks. This normalized map locates storage only. Task 4a must wrap it in trusted
model/config/code/threshold lineage before production projection. Oracle masks remain
synthetic regression/evaluation evidence and cannot enter production decisions, history,
prompts, or WP7 inference.
