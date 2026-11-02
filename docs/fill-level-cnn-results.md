# Fill-level CNN results

Status: **measured through Step 10; final Step 12 results write-up.** The frozen
Revision 2 model was evaluated once on 560 held-out synthetic renders and once on
1,470 approved frames from 27 real-check vials. Neither result was used to tune the
model, rule, ruler or evaluator.

The result is mixed and does not establish production-ready physical accuracy. On the
held-out synthetic test, typical visible error was 0 mL and passed the frozen 1 mL
target, but p95 error was 2.20 mL against a 2 mL target and hidden success was 94.4%
against a 95% target. On the real check, 19 vials produced visible readings with median
absolute error 2.43 mL, but p95 was 23.0 mL and the four worst visible readings were
14.9-23.6 mL high. Eight vials were called hidden; their approximately 30.3 mL ranges
contained the recipe value in five cases.

The synthetic validation gates and real calibration both failed. Calibration and Steps
9 and 10 nevertheless ran under explicit user decisions, with the failures retained and
disclosed below.

## Evidence labels and scope

- **Measured** means produced by a recorded run and tied to a committed compact summary
  or an ignored checksum manifest.
- **Source-file observation** means read from the organizer twin, plan, recipe file or
  another versioned input; it is not a model result.
- **Qualitative diagnostic** means a bounded visual inspection used to explain a failure;
  it is not an accuracy score.
- **Interpretation** explains measured evidence but is not itself a measurement.
- **Proposal** is an untested future direction, not a result or authorization.

No model was trained on real frames. Video-5 vials 1 and 13 were calibration evidence,
not independent validation. The held-out synthetic test and 27-vial real check were each
measured once after their evaluators were frozen. Raw renders, masks, model weights and
real frames remain outside Git.

## Evidence inventory

| Claim | Evidence label | Versioned or checksum-bound source |
|---|---|---|
| Camera targets and organizer-twin hashes | Source-file observation and measured projection | [`camera-match.json`](../assets/fill-level/camera-match.json), commit `c3544ba` |
| Label semantics, variation and split contract | Source-file observation and implementation decision | [`render-contract.json`](../assets/fill-level/render-contract.json), commit `7e8f98f` |
| Full 3,800-frame Step 6 batch | Measured | [`full-batch-summary.json`](../assets/fill-level/full-batch-summary.json), commit `a8592bf` |
| Original training and exception audit | Measured | [`step7-training-summary.json`](../assets/fill-level/step7-training-summary.json), [`step7-exception-review.json`](../assets/fill-level/step7-exception-review.json), commit `366249e` |
| Original validation and first calibration failure | Measured | [`step8-validation-surface-summary.json`](../assets/fill-level/step8-validation-surface-summary.json), [`step8-calibration-failure-summary.json`](../assets/fill-level/step8-calibration-failure-summary.json), commit `74e713b` |
| Revision 1 and second calibration failure | Measured development result | [`step8-revision1-failure-summary.json`](../assets/fill-level/step8-revision1-failure-summary.json), commit `cada81e` |
| Revision 2 combined data and training | Measured synthetic-only; validation-selected | [`step8-r2-combined-batch-summary.json`](../assets/fill-level/step8-r2-combined-batch-summary.json), SHA-256 `bfe6650e...d6b5`, commit `fe545ae`; [`step8-r2-training-summary.json`](../assets/fill-level/step8-r2-training-summary.json), SHA-256 `728991a8...0930`, commit `1f5148b` |
| Revision 2 validation and failed calibration | Measured on synthetic validation / permitted real calibration | [`step8-r2-validation-summary.json`](../assets/fill-level/step8-r2-validation-summary.json), compact/full SHA-256 `0a176ffc...92f5` / `04bb91db...cde61d`; [`step8-r2-calibration-failure-summary.json`](../assets/fill-level/step8-r2-calibration-failure-summary.json), SHA-256 `bc34399a...8729`; commit `87cbf32` |
| Hand-measured two-point ruler | Source-file observation from Step 3 hand measurements | SHA-256 `c1d2f112...1333`, frozen in `45292f1` |
| Frozen held-out test | Measured once on held-out synthetic test and untuned | [`step9-r2-test-summary.json`](../assets/fill-level/step9-r2-test-summary.json), compact/full SHA-256 `e748bfda...199d` / `0eb50c97...8b1f`; evaluator freeze `45292f1`, result `a2a1ee9` |
| Frozen real check | Measured once on 27 reserved real-check vials and untuned | [`step10-real-check-summary.json`](../assets/fill-level/step10-real-check-summary.json), compact/full/frame-ledger SHA-256 `5a0ad63b...036` / `c91bdddd...db32` / `c65f6b9c...78256`; evaluator freeze `4d9a34a`, result `fa298ee` |

The detailed chronological record is
[`fill-level-cnn-log.md`](work/fill-level-cnn-log.md). Ignored evidence is bound by the
hash manifests recorded there.

## Synthetic data and rendering

- **Measured, Steps 2-5:** Isaac Sim 6.0.1 on an NVIDIA L40S produced 640x480 FillCam
  renders with `empty`, `liquid`, `foam`, `blocked` and `ignore` labels. Deterministic
  variation covered level, colour and darkness, foam, sticker, wet walls, lighting,
  camera and holder appearance. All 20 pilot frames passed the offline audit and visual
  inspection; PathTracing at 32 subframes was retained.
- **Measured, Step 6:** all 20 chunks completed: 3,000 train, 300 validation and 500
  reserved test frames. Their seed ranges were disjoint. All stored and rerun checks
  passed, source inputs were unchanged, and aggregate integrity reported zero errors.
- **Measured, Revision 2:** the original population was augmented with an exact-camera
  v4 cohort of 480 train, 60 validation and 60 test renders. The combined contract was
  therefore 3,480 train, 360 validation and 560 sealed test frames. The new cohort was
  retained separately in every evaluation summary so its behavior could not be hidden by
  the larger original cohort.
- **Measured coverage:** the Step 6 frames included 1,897 foam, 2,274 sticker, 1,166
  wet-wall, 1,863 edge-zone and 991 holder-hidden cases. One 0.065117 mL training frame
  had no visible liquid pixels because the sub-mL liquid was below the visible base edge
  with foam and sticker occlusion; its ignored overlays were reviewed and it was retained
  as a legitimate occlusion.
- **Limitation:** the holder projected to about 48.1-66.5 mL in the twin versus 53-69 mL
  in the real reference. Sampling followed the twin zones and recorded this mismatch;
  organizer geometry was not moved.

## Original model and transfer failures

The model is a four-level U-Net with 1,942,628 parameters and a fixed 112x416 vial crop
`(256,48,368,464)`. It scores four classes and excludes ignore pixels. Training used only
synthetic train frames; checkpoint selection used only synthetic validation frames.

**Measured, original training:** 30 epochs selected epoch 28 with validation loss
0.026975 and macro IoU 0.970575. Per-class IoU was 0.990214 empty, 0.988638 liquid,
0.935690 foam and 0.967756 blocked. The selected ignored weight hashes to
`86e89644...ddbeb`.

**Measured, first calibration failure:** the original model and frozen center-strip rule
reported all 98 permitted calibration frames at row 48. The equal medians failed the
ordering guard, so no ruler was created. The real liquids were much darker than rendered
validation liquid and approximately as dark as the holder, identifying a major appearance
gap.

**Measured development result, Revision 1:** one disclosed synthetic-only appearance
revision selected epoch 27 at macro IoU 0.965030. A fixed darkened synthetic check improved,
but the one permitted repeat calibration still failed: vial 1 had 23 visible and five
hidden predictions, while all 70 vial-13 predictions were hidden. No ruler was created.
Six already-permitted calibration frames were inspected as bounded qualitative diagnostics;
they showed patchy real-domain segmentation, including holder, dark-liquid and large-label
confusion.

## Revision 2 training and validation gates

Revision 2 added the exact-camera v4 cohort, retained train-only liquid recolouring and
darkening, and froze the full `holder_band_v2` surface rule with measurement rows
`[83,427)`. It opened no sealed-test image and no real frame during training or validation.

**Measured training:** all 3,480 train and 360 validation samples passed preflight. Thirty
epochs completed in 477.59 seconds, selecting epoch 29 at validation loss 0.046605 and
macro IoU 0.948626. The selected ignored weight is `best-epoch-029.pt`, SHA-256
`1949bdea...b094`.

**Measured validation:** both frozen validation gates failed.

| Cohort | Frames | Visible predictions | Visible absolute error, rows median / p95 / max | Hidden truth-in-range | Non-hidden called hidden |
|---|---:|---:|---:|---:|---:|
| Original Step 6 validation | 300 | 132 | 0 / 1.45 / 34 | 69/72 (95.83%) | 98 |
| New exact-v4 validation | 60 | 26 | 2.5 / 60 / 71 | 17/18 (94.44%) | 17 |

- The overall visible p95 was 17.05 rows against a maximum of 3 rows: **failed**.
- Hidden success required each cohort to reach 95%. The original cohort passed, while the
  exact-v4 cohort reached 94.44%: **failed**, though the user accepted that result against
  their practical 90% bar.
- All 115 false-hidden calls and the reported 92-row (30.263 mL) range width were retained.
  A hidden success only means the truth lies inside that broad range; it flags possible
  large errors rather than providing a precise reading.
- Eight of 26 visible exact-v4 predictions missed by more than six rows; all eight had foam
  or a sticker. The single crop-edge truth fallback was recorded.

**Decision disclosure:** despite both validation-gate failures, the user explicitly directed
the one frozen calibration run to proceed. No threshold, rule setting or model weight was
changed in response to validation.

## Revision 2 calibration failure

Only calibration vial 1 (30 mL, 28 frames) and vial 13 (80 mL, 70 frames) were opened.
All 98 frames were scored once as development use 3; access counters recorded zero train,
validation, test and real-check frames during this run. An earlier provenance preflight
failed before opening an output root or scoring a frame, so it did not count as a
calibration use.

| Vial | Visible / hidden / abstain | Predicted row median (range) | Frozen target | Absolute miss |
|---|---:|---:|---:|---:|
| 1, 30 mL | 28 / 0 / 0 | 224 (224-224) | 329 | 105 rows |
| 13, 80 mL | 70 / 0 / 0 | 169 (168-174) | 177 | 8 rows |

**Measured failure:** the ordering was correct (`224 > 169` in image coordinates), but
both medians were outside the frozen +/-4-row tolerance. No system-fitted ruler was created.

**Decision disclosure:** the user explicitly directed Steps 9 and 10 to run without a fix,
using the unchanged selected weight and rule. The Step 10 conversion instead uses the
hand-measured Step 3 camera targets: 30 mL at row 329 and 80 mL at row 177, or exactly
3.04 rows/mL. This ruler comes only from calibration vials 1 and 13 and was not fitted to
the failed calibration predictions.

## Step 9: held-out synthetic test

The evaluator and ruler were committed before production discovery. All 560 sealed test
renders were scored once in 8.41 seconds: 500 from the original Step 6 cohort and 60 from
the new exact-v4 cohort. No train, validation, calibration or real-check image was loaded.

**Measured overall result:** 239 frames produced visible readings. Absolute row-derived
error was median 0 mL, p95 2.204 mL and maximum 33.882 mL. Of 142 ground-truth-hidden
frames, 134 were called hidden with truth inside the range (94.37%). There were 183 false
hidden calls among 418 non-hidden frames (43.78%), four abstentions and three truth-crop
fallbacks. Synthetic segmentation macro IoU was 0.953417.

| Cohort | Visible n | Visible error, mL median / p95 / max | Hidden success | False hidden | Abstain / fallback |
|---|---:|---:|---:|---:|---:|
| Original Step 6 test (500) | 211 | 0 / 0.987 / 25.987 | 123/128 (96.09%) | 163 | 3 / 2 |
| New exact-v4 test (60) | 28 | 0.329 / 12.862 / 33.882 | 11/14 (78.57%) | 20 | 1 / 1 |
| Combined (560) | 239 | 0 / 2.204 / 33.882 | 134/142 (94.37%) | 183 | 4 / 3 |

Against the frozen targets:

- typical visible error was 0 mL against a maximum of 1 mL: **passed**;
- visible p95 was 2.204 mL against a maximum of 2 mL: **failed**; and
- hidden success was 94.37% against a minimum of 95%: **failed**, while exceeding the
  user's separate practical bar of 90%.

Every predicted hidden interval was 92 rows, or 30.263 mL. Thus the high containment rate
shows that the rule catches many large errors behind the holder; it does not show precise
hidden-level measurement. The sharp cohort split also shows that aggregate behavior was
dominated by the larger original cohort.

## Step 10: measured real check

The real-check evaluator was committed before any production frame access. It used the
same frozen weight and rule plus the hand-measured two-point ruler disclosed above. The
exact 27 eligible vials other than calibration vial 13 were evaluated once. Exactly 1,470
approved settled-window frames were decoded and scored; 1,464 frame predictions contributed
to the vial medians. No train, validation, test or calibration frame was opened during the
run. There were no abstentions and no no-clean-frame outcomes.

**Measured overall result:** 19 vials produced visible readings and eight were called
hidden.

| Result | Count or value |
|---|---:|
| Visible absolute error median / p95 / max | 2.434 / 22.961 / 23.553 mL |
| Visible signed-error median | +1.447 mL |
| Visible within 1 / 2 / 5 mL | 4 / 7 / 11 of 19 |
| Hidden recipe inside reported range | 5 of 8 |
| Hidden range | 42.171-72.434 mL (30.263 mL wide) |

**Measured by recipe zone:**

Here, zone is the source/reference grouping from the recipe and camera geometry, not a
model-detected edge flag; the evaluator recorded zero edge-flagged frames.

| Zone | Vials | Visible / hidden | Visible error median / p95 / max, mL | Within 1 / 2 / 5 mL | Hidden containment |
|---|---:|---:|---:|---:|---:|
| Visible | 10 | 7 / 3 | 2.237 / 23.355 / 23.553 | 2 / 3 / 4 | 0/3 |
| Edge | 15 | 12 / 3 | 2.599 / 14.934 / 14.934 | 2 / 4 / 7 | 3/3 |
| Hidden | 2 | 0 / 2 | n/a | n/a | 2/2 |

**Measured by recipe transition:**

| Group | Vials | Visible / hidden | Visible error median / p95 / max, mL | Within 1 / 2 / 5 mL | Hidden containment |
|---|---:|---:|---:|---:|---:|
| Repeat | 20 | 15 / 5 | 2.434 / 17.322 / 22.895 | 3 / 6 / 9 | 3/5 |
| Transition | 7 | 4 / 3 | 8.684 / 22.260 / 23.553 | 1 / 1 / 2 | 2/3 |

Transitions were materially worse than repeats at the median. The four worst visible
readings were all high: vial 4, +23.553 mL (transition, visible zone); vial 5, +22.895 mL
(repeat, visible zone); vial 8, +14.934 mL (repeat, edge); and vial 25, +14.934 mL
(transition, edge).

**Interpretation:** the median real-vial point error is useful evidence that the frozen
system sometimes transfers, but the tail errors are too large to support a general physical
accuracy claim. Hidden containment is also intentionally forgiving: every hidden result
uses the same 30.263 mL interval, and all three visible-zone vials incorrectly called hidden
fell outside it. Hidden success therefore flags only large errors.

## Bias and exposure disclosures

- **Source-file observation:** the Oct 31 camera-target notes in the plan referred to stills
  from real-check vials 8, 12, 14, 17 and 18 while matching framing. The numerical ruler
  itself uses only calibration vials 1 and 13.
- **Source-file observation:** at about 18:50 CST, after the Step 10 evaluator was frozen at
  about 18:37 and while its run was in progress, Claude viewed WP5 masks and frames for
  real-check vials 6, 18 and 28 on a separate GB10 system for an unrelated WP5 check. That
  exposure could not change the already-frozen Step 10 model, rule, ruler or evaluator, but
  it is disclosed because those vials are in the measured real-check population. The
  evaluation was therefore not pristine-blind to every reviewer even though its frozen
  result could not be changed.
- **Measured and decision disclosure:** validation gates failed; the user directed
  calibration to run anyway. Calibration then failed; the user directed the frozen Step 9
  and Step 10 evaluations to run anyway. No reported Step 9 or Step 10 metric was used for
  tuning or rerun.

## What would likely help

**Proposal, not a measured result or authorization:** future work could render physically
dark translucent liquids, include realistic large labels, reduce false-hidden calls, and
separate transition behavior from settled repeats. Any real-frame training or hand-labelled
real dataset would require a policy change, a new disjoint evaluation population and explicit
approval; the current project did not do this.

## Conclusion

The project produced a reproducible synthetic pipeline, a frozen Revision 2 model, a
sealed synthetic test and a measured real-vial check. The honest conclusion is not a
single pass or fail:

- synthetic typical visible accuracy passed, while synthetic p95 and hidden targets failed;
- the newer exact-v4 cohort was substantially harder than the original cohort;
- the system calibration failed and supplied no ruler;
- the hand-ruler real check had a 2.43 mL median visible error but approximately 23 mL p95
  and maximum errors; and
- hidden outputs were broad 30.3 mL flags, not precise estimates.

These results support further research but not a production-ready physical accuracy claim.
