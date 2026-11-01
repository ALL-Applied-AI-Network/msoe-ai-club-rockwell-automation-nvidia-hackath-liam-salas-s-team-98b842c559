# Fill-level CNN results (draft through Step 8)

Status: **draft through Step 8; not a final project result.** The project remains
STOPped pending the user's decision. Step 9 has not started. No test-render manifest or
image and no real-check vial was opened or scored for this draft; those results are
unchecked.

The current result is negative for real-world transfer. A four-class segmentation model
trained only on synthetic renders reached about 0.97 macro intersection-over-union (IoU)
on synthetic validation, but it could not produce two valid calibration points from the
permitted video-5 vials. One disclosed revision improved a fixed synthetic darkness test
but also failed the single permitted repeat calibration. No row-to-mL ruler was created,
and no physical or real-world accuracy is claimed.

## Evidence labels and scope

- **Measured** means produced by a recorded run and tied to a committed compact summary or
  an ignored checksum manifest.
- **Source-file observation** means read from the organizer twin, the plan or another
  versioned input; it is not a model result.
- **Qualitative diagnostic** means a bounded visual inspection used to explain a failure;
  it is not an accuracy score.
- **Interpretation** explains the evidence but is not itself a measurement.
- **Unchecked** means the reserved test or real-check result does not exist yet.

Video-5 vials 1 and 13 are calibration evidence, not independent validation. Revision 1
was informed by the first calibration failure, so its second calibration result is a
disclosed development result rather than an unbiased test. No model was trained on real
frames.

## Evidence inventory

| Claim | Evidence label | Versioned or checksum-bound source |
|---|---|---|
| Camera targets and organizer-twin hashes | Source-file observation and measured projection | [`assets/fill-level/camera-match.json`](../assets/fill-level/camera-match.json), SHA-256 `aeee0c5e...afc2f8`, commit `c3544ba` |
| Label semantics, variation and split contract | Source-file observation and implementation decision | [`assets/fill-level/render-contract.json`](../assets/fill-level/render-contract.json), SHA-256 `70d87087...a3258`, commit `7e8f98f` |
| Twenty-frame pilot and renderer choice | Measured | [`assets/fill-level/pilot-summary.json`](../assets/fill-level/pilot-summary.json), SHA-256 `6fc753fa...f29ca`, commit `31beb9f` |
| Full 3,800-frame synthetic batch | Measured | [`assets/fill-level/full-batch-summary.json`](../assets/fill-level/full-batch-summary.json), SHA-256 `e736d17e...d88c0`, commit `a8592bf` |
| Original training and exception audit | Measured | [`assets/fill-level/step7-training-summary.json`](../assets/fill-level/step7-training-summary.json), SHA-256 `799fc3af...5c88`; [`assets/fill-level/step7-exception-review.json`](../assets/fill-level/step7-exception-review.json), SHA-256 `6ef12c4a...6130f`; commit `366249e` |
| Frozen validation rule and first calibration failure | Measured | [`assets/fill-level/step8-validation-surface-summary.json`](../assets/fill-level/step8-validation-surface-summary.json), SHA-256 `797a2bb8...7e12`; [`assets/fill-level/step8-calibration-failure-summary.json`](../assets/fill-level/step8-calibration-failure-summary.json), SHA-256 `8ac1ff33...775e3`; commit `74e713b` |
| Revision 1 and second calibration failure | Measured development result | [`assets/fill-level/step8-revision1-failure-summary.json`](../assets/fill-level/step8-revision1-failure-summary.json), SHA-256 `4b3375d6...7cbb9`, commit `cada81e` |
| Six-frame original/revised overlays and row profiles | Qualitative diagnostic, ignored | `outputs/fill-level-cnn/step8-diagnostics-20261101-codex-01/diagnostics-attempt-01-20261101T103833Z/`; summary SHA-256 `1d1adb1c...dd21`, evidence manifest SHA-256 `1d19bab1...9176` |

The detailed chronological record is the
[`fill-level-cnn-log.md`](work/fill-level-cnn-log.md). Raw renders, masks, real frames and
model weights remain outside Git.

## Synthetic data and rendering

- **Measured, Step 1:** the local Python environment was created and the synthetic
  absorbance smoke passed. The full local suite did not pass cleanly because OpenCV was then
  missing from the development requirements and the Windows checkout did not materialize the
  required skills symlink; those setup limits are not model evidence.
- **Measured, Step 2:** Isaac Sim 6.0.1 on the existing NVIDIA L40S produced a recognizable
  640x480 FillCam smoke render. The successful warm-cache render took 31.262 seconds; its raw
  PNG and logs remain ignored.
- **Measured and source-file observation, Step 3:** video-5 reference measurements put the
  30 mL surface at row 329 and the 80 mL surface at about row 177, approximately 3.04
  rows/mL. An anonymous-session-layer camera override projected the exact-height synthetic
  surfaces to those targets without editing the organizer twin. Final rendered RGB edges
  were rows 329 and 179, both within the three-row camera checkpoint tolerance.
- **Measured, Steps 4-5:** the renderer produced `empty`, `liquid`, `foam`, `blocked` and
  `ignore` labels, with deterministic variation in level, colour/darkness, foam, sticker,
  wet walls, lighting, camera and holder appearance. All 20 pilot frames passed the offline
  audit and visual inspection. PathTracing at 32 subframes was retained.
- **Measured, Step 6:** all 20 chunks completed: 3,000 train, 300 validation and 500 reserved
  test frames at 640x480. Their disjoint seed ranges are `100000-102999`,
  `1100000-1100299` and `2100000-2100499`. All 20 stored and rerun checks passed, source
  inputs were unchanged, and aggregate integrity reported zero errors. Successful render
  time was 25,618.8 seconds, or 533.98 frames/hour.
- **Measured coverage:** the 3,800 frames include 1,897 foam, 2,274 sticker, 1,166 wet-wall,
  1,863 edge-zone and 991 holder-hidden cases. Continuous sampling produced no exactly
  0 mL frame. One 0.065117 mL training frame has no visible liquid pixels because the
  sub-mL liquid is below the visible base edge with foam and sticker occlusion; its ignored
  overlays were reviewed and it was retained as a legitimate occlusion.
- **Limitation:** the holder projects to about 48.1-66.5 mL in the twin versus 53-69 mL in
  the real reference. Sampling follows the twin zones and records this mismatch; organizer
  geometry was not moved.

## Synthetic-only training

The original model is a four-level U-Net with 1,942,628 parameters. It uses the fixed
112x416 vial crop `(256,48,368,464)`, four scored classes and ignore pixels. Training used
only the 3,000 synthetic train frames; model selection used only the 300 synthetic validation
frames. Class weights came from train masks only. Training augmentation used JPEG 4:2:0,
blur and noise without changing mask geometry.

**Measured:** the full run completed 30 epochs in 303.49 seconds. Validation-only selection
chose epoch 28 with loss 0.026975 and macro IoU 0.970575. Per-class IoU was 0.990214 empty,
0.988638 liquid, 0.935690 foam and 0.967756 blocked. The selected ignored weight hashes to
`86e89644...ddbeb`.

**Limitation:** this is synthetic validation. It measures agreement with synthetic labels;
it does not establish physical or real-world accuracy.

## Synthetic surface validation

Before any real calibration frame was used, the original center-strip rule was frozen from
all 300 synthetic validation renders. It searches columns `[298,316)`, requires at least
five liquid pixels per row and uses blocked pixels to report hidden spans.

- **Measured visible cases:** 215 predictions; median absolute surface error 1 row
  (0.329 mL), p95 2 rows (0.658 mL) and maximum 60 rows (19.737 mL).
- **Measured hidden cases:** among 72 true-hidden frames, the rule reported hidden for 38
  (52.78%). The true geometry row was inside the reported range for 21/72 (29.17%), or
  21/38 (55.26%) conditional on a hidden prediction.
- **Interpretation:** the median and p95 visible results were strong, but the maximum error
  and hidden-case coverage already showed important synthetic failure modes.

## Calibration failure 1: original model and rule

Only two settled video-5 calibration windows were opened: vial 1 at 30 mL, frames 816-843
(28 frames), and vial 13 at 80 mL, frames 19700-19769 (70 frames). The inputs are lossless
640x480 PNGs bound by index SHA-256 `73181d39...f9b2`.

**Measured failure:** all 98 frames were reported visible at row 48, the first row of the
crop. The affine-ruler guard rejected the equal medians because the 30 mL row must lie below
the 80 mL row. No ruler was created. Read-only diagnosis found all 18 center-strip pixels at
row 48 classified as liquid in every frame. At the expected row 177, vial 13 had a median
of zero liquid and 18 blocked pixels.

**Interpretation:** extraction, dimensions, preprocessing, checkpoint and input hashes
matched. The frozen synthetic rule searched above the vial interior and the CNN did not
transfer to the much darker real liquids. This was a genuine model/rule domain-transfer
failure, not a transport or retry failure.

**Measured (Claude review, ~03:35 entry):** in the vial-centre columns, the real liquid mean
HSV value-channel brightness (`V`) was 0.155 for vial 1 and 0.088 for vial 13, while the real
holder plate was 0.184. Rendered validation liquid, measured under its liquid mask, had a
5th-percentile `V` of 0.26 and a median of 0.56; no rendered validation liquid was below
0.15. Both real-liquid means were far below the rendered 5th percentile and were as dark as
the holder or darker. Vial 13 was below every rendered value, while vial 1 lay near the
reported lower bound. This is the clearest measured appearance gap behind the transfer
failure.

## One disclosed revision

Exactly one revision was authorized after that calibration failure. It was informed only by
the two calibration vials and is disclosed here.

- The rule's measurement rows were restricted to `[83,427)`, derived from synthetic
  validation vial outlines; columns and pixel thresholds remained frozen.
- Synthetic train-only augmentation recoloured liquid-mask pixels in about half of frames,
  with value sampled from `[0.03,0.35]`, while retaining local texture. Global brightness,
  contrast, gamma, saturation and white-balance jitter were added; JPEG 4:2:0, blur and
  noise remained.
- Training retained the Step 7 data/settings and added cosine learning-rate decay. No real
  frame was used for training or selection.

**Measured revised training:** 30 epochs selected epoch 27 on validation at macro IoU
0.965030. The ignored selected weight hashes to `51f4e3cf...f0197`.

**Measured synthetic comparison:** on normal validation, original/revised macro IoU was
0.970578/0.965033 and visible-surface p95 was 2.0/2.55 rows, so the revised model passed the
required three-row ceiling. On a fixed-seed darkened validation copy, original/revised macro
IoU was 0.964944/0.978789 and p95 was 14.8/2.0 rows.

**Limitation:** the fixed synthetic darkening test showed the intended augmentation effect;
it did not demonstrate transfer to real video.

## Calibration failure 2 and bounded qualitative diagnostics

Exactly one repeat calibration used Revision 1 on the same 98 frames.

- **Measured, vial 1:** 23 visible and five hidden predictions. Twenty visible predictions
  were at row 99 and three at row 224, giving median row 99 versus target row 329.
- **Measured, vial 13:** zero visible and 70 hidden predictions. Without three visible rows,
  calibration stopped before fitting. No ruler was created and no second revision or repeat
  run was made.

The stopped-review diagnostic allowance opened exactly six already-permitted calibration
frames: vial 1 frames 816, 830 and 843, and vial 13 frames 19700, 19735 and 19769. For each
frame, both models produced a predicted-mask-on-RGB overlay and a center-strip row profile;
all 44 payloads in the preserved diagnostic evidence passed host and local checksum checks.

**Qualitative diagnostic:** Revision 1 correctly identifies empty glass near the top of both
vials, the holder front as blocked, liquid near the bottom of vial 1, and some liquid near
the top of vial 13 inside the holder hole. It also labels much of vial 13's dark liquid
column as blocked, labels vial 1's large white BRADY sticker as foam, and creates stray
liquid patches high in vial 1. Predictions outside the vial are not interpreted because
those pixels are `ignore` and were never trained.

**Interpretation:** these bounded overlays correct an earlier, more optimistic rule-only
reading. The failure is mainly a CNN real-domain transfer problem, not only a surface-rule
problem. A rule-only change might fit the two calibration vials but would not repair the
patchy segmentation itself.

## What would likely help

**Proposal, not a measured result or authorization:** the failure evidence suggests these
future directions, but none has been implemented or tested here.

- Render physically dark, translucent liquids so path-traced training images reproduce the
  real appearance gap, rather than relying only on 2D recolouring.
- Include realistic large labels such as the white BRADY sticker so label appearance and
  occlusion are represented in synthetic training.
- Explore unsupervised style adaptation that uses unlabelled real frames for appearance only,
  while preserving the synthetic labels and an independent evaluation split.
- Consider a small hand-labelled real training set only after a policy change and explicit
  approval from both the user and Liam. The current project rules do not permit training on
  real frames.

## Conclusion and limits

The synthetic pipeline produced a large, internally checked dataset and a model with strong
synthetic validation scores. That success did not transfer to the two permitted real
calibration vials. The original model/rule failed, and one disclosed synthetic-only revision
also failed. Therefore:

- no row-to-mL ruler is frozen;
- no physical fill-level accuracy is established;
- the 500 reserved test renders are unscored;
- the 27 real-check vials are unopened and unscored;
- Step 9 and Step 10 results are unchecked; and
- the project remains STOPped pending the user's choice. Do not start Step 9.

This draft can support an honest negative result if the project ends here, or it can be
extended only after a newer explicit decision authorizes the next path.
