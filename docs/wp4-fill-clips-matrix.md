# WP4 fill clips and measurement matrix

This is a CPU-only Part-B implementation record for `src/fluidmix/fill_clips.py`.
Inputs are a camera video, the programmed recipe/tank order/vial sequence, and optional
frame/timing evidence. No tank-colour preset, palette, residue, particle state, label,
or other simulator metadata is read. Synthetic receipts can locate an artifact, but
cannot provide a measurement value.

| Area | Status | Evidence / limit |
|---|---|---|
| Half-open per-fill source frame ranges | Works | `FillBoundary`; ranges are checked against the source video and recipe sequence. |
| Review MP4 per-fill clips | Works | `build_fill_clips`; each clip is written and SHA-256 hashed. MP4 is review-only. |
| Transport handling | Works | Transition ranges are tagged `transition`/`transport` and are never sent to measurement. |
| Stage after active tank dispense | Works | `StageBoundary` requires ordinal, tank ID and programmed active order. |
| Masked crop/still artifacts | Works with a supplied mask | RGB and binary mask bytes are copied to hash-bound stills; a masked crop is also emitted. |
| RGB→Lab median/MAD/IQR | Works with supplied mask | Delegates to the existing `vision_measurement.measure_image` implementation. |
| Same-stage CIE76 pairs | Works | Pair rows carry both evidence IDs and fixed six-decimal display values. |
| Triad thresholds | Works as constants only | Frozen `2.0` peer diameter and `1.0` margin are exposed; disposition remains downstream. |
| Cross-batch threshold | Works as helper only | `max(2.0, max_radius + 1.0)`; reference selection/decision remains downstream. |
| Real final-mixture window | Works as timing helper | Implements `end + 0.3 s` through `0.1 s` before departure and the three-frame eligibility gate. It does not claim physical truth. |
| INN-52 timing evidence | Needs validation | Optional locator schema can derive frame boundaries; burst timing itself remains prior real-video evidence, not a learned detector. |
| Accepted V3 regression | Executed read-only and matched | `remeasure_preserved_v3` read only the accepted measured-evidence and image-locator artifacts: 42/42 Lab rows and 42/42 ΔE rows matched; source hashes remained unchanged. |
| RGB-only predicted masks | Not available | No accepted predicted-mask artifact exists. This implementation records oracle masks only for synthetic regression and does not report segmenter performance. |
| Evidence-record-v1 publication / evidence store | Needs integration | This module emits keyed/hash-bound Part-B artifacts; WP5 must persist them and WP7 must connect decisions/explanations. |

The generated JSON records the source video hash, recipe sequence, fill/stage IDs,
artifact hashes, measurement contract, and these limits. It does not serialize any
simulation truth.
