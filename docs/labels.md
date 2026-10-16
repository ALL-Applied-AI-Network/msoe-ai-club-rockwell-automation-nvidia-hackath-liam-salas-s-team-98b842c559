# Label protocol (INN-7 T5)

Written 2026-10-16. A **label** is the answer a frame should get (which recipe, which vial,
whether a fault is present). This page says where labels come from, what stays unknown, and
which real frames are left out of scoring. Counts below were measured on the local copy in
[data inventory](data_inventory.md); the twin example describes files, not renders.

Design (project description: "look at the real data, but train on the digital twin"): the twin
generates both training and test data. Real video is for understanding the station and, now and
then, validating assumptions such as absorbance metrics. It is not a training or test set.

## 1. Where labels come from

| Source | Label fields | Known how | Used for |
|---|---|---|---|
| Twin renders | recipe (mL per tank), tank colors, time code (frame), liquid mask, any injected fault | By construction: they are the inputs to the render, plus the `synthetic_class` attributes on fill-station prims | Training and test |
| Real video 5, vial_fill camera | recipe ID, mL per tank, dispense order, fill window (start/end video s) per vial | Recipe log `data/video5_vial_recipes.csv`; stills placed by `scripts/align_still_labels.py` | Validation only |
| Real videos 1–4 | none | No recipe or vial data exists | Station context only |

The twin's per-prim attributes are `userProperties:synthetic_class`, `synthetic_class_id` and
`synthetic_instance_id`; the liquid is class `liquid` (`Fill_liquid`, class id 2). See [twin](twin.md).
Previous-recipe context for each real vial is in `data/video5_transitions.csv` (built by
`scripts/build_transitions.py`; see [data inventory](data_inventory.md)).

## 2. Independence from the detector

A label is useless for scoring if it was derived from the score it is checking.

- Twin labels come from what was rendered (the parameters passed in), never from colors
  measured on the rendered image.
- Real validation labels come from the recipe log and fill windows (time), never from color.
  The fill windows were placed by eye from the video, so they are human judgments, but about
  flow timing, not about vial color.
- Do not "fix" a label because the detector disagrees with it. Record the disagreement instead.
  Example: one aligned still of vial 22 shows teal liquid under an orange stream
  ([video timing](video_timing.md)); it keeps its recipe-12 label.

## 3. What is unknown (no label exists)

- **Defect cause** on any real vial. No defect or cause labels exist.
- **Residue/carryover truth** on the real line: how much previous fluid stayed in the line.
- **Stills for vials 26–30**: still coverage ends at video second 1395.5 (the last ~18% of the
  recording; see [video timing](video_timing.md)). Their final-colour windows come from video frames.
- **Recipe before vial 1**: not recorded; recipe 5 is noted "swapped in".
- **Videos 1–4**: no recipes or vial windows. Out of scope, because test data comes from the twin.
- **Fill-window accuracy**: spot-checked on vial 25 only (flow start and stop within ~1–2 s).

## 4. Ambiguous cases in real validation frames

Apply these rules before any validation score, and report the excluded counts beside the score.

| Case | Rule | Video-5 count (vial_fill stills) |
|---|---|---|
| Within 2 s of a fill start or end (`boundary_margin_s` < 2) | "uncertain": excluded from scores, counted separately | 114 of 844 placed (56 in a fill, 58 between) |
| Before or after the video (`status` not `in_video`) | excluded | 1 (19:47:09.6, ~1 s before frame 0) |
| In a recording gap or within the uncertainty of an unresolved clock break | excluded | 0 in video 5 (no jumps); applies to cap_seal video 1 |
| Vial 1 (unknown prior recipe) | excluded from carryover validation; allowed for clean-color checks | 1 vial |
| Duplicate stills (`(1).jpg` files) | excluded | 2 vial_fill files; not indexed, so not among the 844 |

The 2 s margin is a judgment, not a measured error bound: it covers the ~1–2 s window error
seen on vial 25 plus about one frame of still timing error. Change it only before scoring.

These rules decide which vial a still belongs to. They do not choose the frames used to measure a
vial's final colour. Those come from video frames in a per-vial window after the fill ends and before
the cart moves (`data/real_check_windows.csv`; see [evaluation](evaluation.md)). Stills are too
sparse for that: the window lasts 0.5–5.3 video seconds, and 24 of vials 1–25 get at most one still in it.

## 5. Worked example: real still

Clock model from [video timing](video_timing.md): wall time at video 0 ≈ 2026-09-23 19:47:10.55,
1.21578 wall s per video s. So `video_s = (wall − 19:47:10.55) / 1.21578`.

**Clear case.** `vial_fill_20260923_200407_402315.jpg`
1. Filename time: 20:04:07.402 → 1016.85 wall s after 19:47:10.55.
2. 1016.85 / 1.21578 = **836.38 video s**, frame 25,091 at 30 fps (aligned CSV: 836.377, frame 25091).
3. Vial windows: vial 16 runs 817.671–855.988 s. The still is inside, 18.7 s from the nearest edge.
4. Label: in_fill = 1, vial 16, recipe 10 (T100 45 / T200 15 / T400 15 mL, 75 mL).
   From `data/video5_transitions.csv`: previous recipe 9, recipe_changed = 1, position_in_run = 1,
   so it is eligible for carryover validation.
5. Contrast: the source frame labels set video_s = 1017.40 and call this vial 19. That is the
   1-second-per-second error the alignment fixes.

**Ambiguous case.** `vial_fill_20260923_200429_525630.jpg` maps to 854.573 s, inside vial 16 but
1.42 s before its end at 855.988 s. Label: vial 16, recipe 10, **uncertain**; excluded from scores
and counted in the 114.

## 6. Worked example: twin frame (what is actually known)

No renders exist yet, and color and fault injection are not calibrated or implemented. At the level
of files that exist today:

- Recipe 7 (T100 20, T400 50 mL) and tank colors are passed to `scripts/twin_liquid_variant.py`,
  which writes a USD layer that overrides only the liquid material color and opacity. That color
  mapping is an uncalibrated guess.
- The fill animation is fixed: time codes 1–686 at 30 fps, ~48 mL blue by 368 and ~67 mL by 686.
  Time code **686 = finished vial**; 12–368 and 586–661 show the stream.
- A render of that layer at time code 686 would be labeled: recipe 7, the tank colors used,
  time code 686 (finished), liquid pixels = prims with synthetic_class `liquid`, fault = none
  (no fault parameter exists yet). Every field comes from the render inputs, none from the image.
- Limit: recipe volumes do not resize the animation, so the mL label describes the intended
  recipe, not the rendered liquid height.

## 7. Dropped criterion: second-person agreement

The original issue asked for a second person to label a sample and report agreement. That
criterion does not apply under this design:

- Twin labels are known by construction, so two people would only copy the same render inputs.
- Real video is not a test set, and the team is not labeling real frames. The only real labels
  (recipe log and fill windows) came with the dataset; spot-checking windows against video
  (as done for vial 25) is the useful check, not inter-rater agreement.
