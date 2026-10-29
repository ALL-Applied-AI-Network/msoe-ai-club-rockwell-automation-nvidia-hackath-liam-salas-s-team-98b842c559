# WP4 fill clips and measurement matrix

This is a CPU-only Part-B implementation record for `src/fluidmix/fill_clips.py`.
Inputs are a camera video, the programmed recipe/tank order/vial sequence, and optional
frame/timing evidence. No tank-colour preset, palette, residue, particle state, label,
or other simulator metadata is read. Synthetic receipts can locate an artifact, but
cannot provide a measurement value.

| Area | Status | Evidence / limit |
|---|---|---|
| Half-open per-fill source frame ranges | Works | `FillBoundary`; all fill and transition intervals are globally checked for overlap/order before output. |
| Review MP4 per-fill clips | Works | `build_fill_clips`; each clip is written and SHA-256 hashed. MP4 is review-only. |
| Transport handling | Works | Transition ranges are tagged `transition`/`transport` and are never sent to measurement. |
| Stage after active tank dispense | Works | `StageBoundary` requires every programmed active ordinal exactly once, with matching tank/order; incomplete or duplicate sets fail closed. |
| Masked crop/still artifacts | Works with a supplied mask | RGB and binary mask bytes produce derived still/crop artifacts, while evidence locators retain the untouched source RGB/mask path and SHA-256. |
| RGB→Lab median/MAD/IQR | Works with supplied mask | Delegates to the existing `vision_measurement.measure_image` implementation. |
| Same-stage CIE76 pairs | Works | Pair rows carry both evidence IDs and fixed six-decimal display values. |
| Triad thresholds | Works as constants only | Frozen `2.0` peer diameter and `1.0` margin are exposed; disposition remains downstream. |
| Cross-batch threshold | Works as helper only | `max(2.0, max_radius + 1.0)`; reference selection/decision remains downstream. |
| Real final-mixture window | Works as timing helper | Implements `end + 0.3 s` through `0.1 s` before departure and the three-frame eligibility gate. It does not claim physical truth. |
| INN-52 timing evidence | Needs validation | Only the exact `{fills:[...]}` Part-B timing projection is accepted; unknown fields and recursively nested Part-A/simulation metadata fail closed. Burst timing itself remains prior real-video evidence, not a learned detector. |
| `evidence-record-v1` measured records | Works for storage mode | Each measured stage is validated through the authoritative `validate_evidence_record_v1(..., phase="measured", mode="storage")`; pair rows carry contextual stage/run/recipe fields. |
| Accepted V3 regression | Executed read-only and matched | `remeasure_preserved_v3` read only the accepted measured-evidence and image-locator artifacts: 42/42 Lab rows and 42/42 ΔE rows matched; source hashes remained unchanged. |
| Accepted V3 clip-input projection | Works, evaluation-only | `build_f2_fill_clip_input_manifest` revalidates the exact six-request/18-receipt publication graph and public artifact-validator result, then publishes one safe record per case. The active interval is `[0, bisect_right(video_timing_source_frame_steps, fill_end_step))`; whole-video fallback is forbidden. Source-step arrays stay in the receipt, and the 42 external stage still IDs carry no video frame index. |
| RGB-only predicted masks | Not available | No accepted predicted-mask artifact exists. This implementation records oracle masks only for synthetic regression and does not report segmenter performance. |
| Evidence-record-v1 publication / evidence store | Needs integration | This module emits keyed/hash-bound Part-B artifacts; WP5 must persist them and WP7 must connect decisions/explanations. |

The generated JSON records the source video hash, recipe sequence, fill/stage IDs,
artifact hashes, measurement contract, and these limits. It does not serialize any
simulation truth.

The separate V3 clip-input manifest records video path/hash/fps/decoded count and the
active-fill range needed by the clip operator. It contains no mask path/source, label,
palette, preset, residue, particle, semantic, or oracle field. Oracle masks may be joined
only later by the segregated synthetic evaluation runner; this manifest is not a production
model feature table and does not establish detector performance.
