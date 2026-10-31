# Fill-level CNN: decisions and build plan

Written 2026-10-31 by Claude from a planning session with the user (loducaj). This is a
structured record of the decisions, not a verbatim transcript. Codex implements; Claude
reviews each checkpoint (see "Working arrangement"). Base: `a1129eb` (origin/main).
Proposed task branch: `codex/fill-level-cnn`. Deadline: Nov 1 submission, 23:59 CT.

Evidence labels follow [AGENTS.md](../../AGENTS.md): **measured**, **source-file observation**,
**proposal**, **unchecked**. Nothing below has been implemented yet.

## Goal

A CNN that reads where the true liquid surface is in a `vial_fill` frame, ignoring foam,
so the system can check the amount in each vial against the recipe. The vial is always
the same size, so a liquid height converts to mL through one fixed scale.

## Pipeline

```
video frames
  -> stream burst detector (scripts/stream_sequence.py, INN-52): find the pause after each dose
  -> CNN (trained on twin renders only): label each pixel liquid / foam / empty / blocked
  -> rule: top row of the liquid, or a range when the surface is hidden
  -> ruler: pixel row -> mL (fixed, fitted once)
  -> compare with the recipe
```

Only the CNN is learned. The ruler and comparison are fixed arithmetic.

## Decision log

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | When must it work? | Nov 1 submission | User requirement |
| 2 | When to measure? | Once after each tank's dose ("stage"), every vial; 2 or 3 per vial. Not every frame | Matches the per-tank wrong-volume defect and the pipeline's stage definition (INN-61) |
| 3 | How to find stage ends? | Sebastien's burst detector. Burst count != recipe tank count -> "timing uncertain" | Independent of recipe mL; already measured 28/29 vials |
| 4 | CNN output? | Per-pixel mask over a vial crop: liquid, foam, empty, blocked | Foam excluded by design; hidden surface detected, not guessed; inspectable; also fills INN-61's missing RGB liquid mask |
| 5 | Surface hidden? | Report a range: mL at bottom and top of the blocked span above the highest visible liquid; check if the recipe value is inside. Burst-timing estimate recorded separately. "Watch the pour" is a later upgrade | Honest; still catches large errors |
| 6 | Foam? | Ignore only: excluded from volume, not reported. Foam stays a training class | User requirement; a foam fault has no definition yet |
| 7 | Which frames? | Median level over the frames in the pause after a dose: skip the first 0.3 s, need >= 3 frames, else "no clean frame". Last dose: frames before the cart moves | Reuses the frozen final-mixture window rule; avoids the falling stream |
| 8 | Pixel row -> mL? | Fit a ruler on real calibration vials 1 (30 mL) and 13 (80 mL); role `calibration`; frozen before the real check | Absorbs camera angle and curved surface; no camera model needed |
| 9 | Per dose or running total? | Per dose eventually, but the current focus is an accurate level reading | User priority |
| 10 | How to prove it? | Twin test (main) plus real video 5 vs recipe log. Hand-marked real lines optional later, only with Liam's OK | Twin truth is exact; real check is the only real-world evidence |
| 11 | Accuracy target | Visible surfaces: typical error <= 1 mL, 95% <= 2 mL. Same on foamy frames. Truly hidden: "hidden" or a range in >= 95%. 70 mL edge zone reported separately. Frozen before twin test scoring | Measured camera/plant repeatability is about 1 mL |
| 12 | Baseline? | None; report the CNN's accuracy only (user decision) | Flag: [plan](../plan.md) lists "measured against a baseline" as a judging point and [evaluation](../evaluation.md) asks each part to beat one |
| 13 | Training data source | Fresh Isaac Sim renders on Brev, team credits (user decision and authorization) | V3/V4 renders are not in Git (checked main and all 28 remote branches) |
| 14 | Fallback | Hard cutoff noon Nov 1: no labeled frames by then -> 2D synthetic generator in code (needs Liam's OK: not twin data) | Brev rendering of this scene failed before (below) |
| 15 | Camera | Match the twin camera to the real view, using the measured targets below | The hard cases are geometric |
| 16 | Render variation | See "Render variation" | User accepted the list as is |
| 17 | Labels | See "Label rules" | Teaches the holder-through-glass trap |
| 18 | Real check scope | Final volumes first; per-stage on real video is the stretch goal | Final-volume windows already exist |
| 19 | Delivery | **Open:** Liam's OK for the submission, and PR vs Liam's combined push | Code freeze was Oct 28 |

## Measured on real video 5 (Claude, scratch analysis, 2026-10-31)

Measured from brightness/colour row profiles down the vial centre (columns 298-315) on the
19 finished-vial stills that fall inside `data/real_check_windows.csv` windows. No script is
committed for this; re-measure with a committed script before relying on it.

| Item | Value | Source |
|---|---|---|
| Scale | 3.04 rows per mL (0.33 mL/row), linear 30-80 mL | 30 mL at row 329 (vial 1), 80 mL at row 177 (vial 13) |
| Linearity check | 70 mL predicted 207, seen ~206 (vial 8); 75 mL predicted 192, seen 193-195 (vials 17, 18) | within 1-3 rows |
| Repeatability | Same recipe, different vials: 2-3 rows (0.7-1 mL) | vials 17 vs 18, 13 vs 14 |
| Edge sharpness | Liquid edge spans 6-9 rows; centre locatable to about 1-2 rows | vial 13 |
| Holder | Dark rows 210-259 beside the vial (x=272). Hides about 53-69 mL | vial 13 frame |
| Edge zone | 69-73 mL: a sliver of surface shows over the holder top (clear in blue vial 8) | vials 7, 8, 9, 19, 24 |
| Trap | Through empty glass the holder looks as dark as liquid from row ~198 (vial 12, 60 mL, low colour saturation) | vial 12 |
| Video-5 stage readings (78) | 9 hidden (60 mL stages), 18 at the edge (70 mL), rest visible unless a sticker covers | recipe log + zones |
| Stickers | Brady barcode labels cover the line in some vials (14, 15, 16) and vary in position | stills |
| Foam | Bubbly, uneven surface on vial 13 about 0.3 s before fill end (`vial_fill_20260923_200028_117016.jpg`) | still |
| Not measured | 40 mL vials (no still in window), mid-fill stages, pause length between doses | needs video decoding |

Source-file observations: the `cap_seal` camera looks down on opaque yellow caps and cannot
show fill level. Neither README gives vial dimensions; the twin models a 26.41 mm inner
cylinder, 0.548 mL per mm ([twin](../twin.md)). The finished vial stays in view only
0.07-4.87 s before the cart moves ([labels](../labels.md)).

## Brev and Isaac Sim

- Instance `isaac-launchable-d70fec`, org UWM/MSOE-Team-11, AWS `g6e.4xlarge` (L40S 46 GB,
  driver 595.71.05), deployed from the organizer launchable `env-35JP2ywERLgqtD0b0MIeK1HnF46`
  (Isaac Sim 6.0.1). User `ubuntu`; `~/isaac-launchable` exists; `~/.cache` links to
  `/ephemeral/cache`; 180 GB free. Where Isaac Sim's Python lives is **unchecked**.
- Billing runs while it is on (about $3.60/h). Stop it when idle:
  `wsl -e bash -lc "brev stop isaac-launchable-d70fec"`. Do not touch teammates' instances.
- Brev CLI v0.6.337 is installed in WSL Ubuntu on the user's laptop and logged in. Use
  `wsl -e bash -lc "brev exec isaac-launchable-d70fec '<cmd>'"` and `brev copy`.
  `brev launch` from the CLI failed with "Internal Server Error" on three L40S types;
  the web console deploy worked.
- **History** (branch `codex/twin-recovery-7-20261024`, `docs/morning-review/2026-10-25.md`,
  `docs/fillcam-recovery.md`): FillCam rendering on this 6.0.1 launchable failed on Oct 24:
  unwritable container cache, then a blank image, then no image. It later worked on the GB10
  with Isaac Sim 6.2 and watchdog retries (INN-41).
- **Reuse** [render_tank_station.py](../../scripts/render_tank_station.py): headless
  `SimulationApp`, a private Kit data/cache/log root (`--portable-root`), retries, semantic
  labels in the session layer, and a separate geometric liquid-label pass with shells hidden.
  Validate pairs the way [check_tank_capture.py](../../scripts/check_tank_capture.py) does.

## Render variation (decision 16)

| Factor | Range |
|---|---|
| Liquid level | 0-85 mL continuous; oversample 45-53, 53-69 and 69-75 mL |
| Liquid colour | Any hue and darkness: clear, pale, orange, blue, green, brown, near-black |
| Foam | About half the renders: 0-10 mm layer on top, white to tinted, bumpy, partly see-through |
| Stickers | About 60%: white barcode label, random rotation around the vial and height |
| Lighting | Brightness and colour temperature about +/-20% |
| Camera | Small jitter around the matched view: a few pixels, +/-2% zoom |
| Wet walls | Sometimes faint droplets or film above the liquid |

Training-time augmentation, not rendered: video-style compression, chroma at half resolution
(the real video is `yuv420p`), blur, noise. No falling stream in v1.

## Label rules (decision 17)

| Pixel shows | Label |
|---|---|
| Liquid through the glass, including the sliver over the holder at 70 mL | liquid |
| Foam | foam |
| Empty vial, including the holder seen through empty glass | empty |
| Holder front face, metal post, sticker | blocked |
| Outside the vial | ignored (not scored) |

Check on the first render whether the standard semantic labels mark the glass instead of what
lies behind it. If so, derive liquid/foam from a separate pass with the glass hidden.

## Camera targets (decision 15)

Real `vial_fill` pixel targets: holder dark rows 210-259 beside the vial; 30 mL surface at row
329; 80 mL at row 177; about 3.04 rows per mL. Done when renders at known liquid heights land
within +/-3 rows of these. Log this as informed by video 5.

## Build steps and checkpoints

Claude reviews at each **checkpoint** before the next step starts.

1. **Setup.** Task worktree/branch off origin/main. Python env per [quickstart](../quickstart.md).
2. **Smoke render** on Brev: unchanged FillCam, frame 686, writable private cache, outer
   timeout. Pass: a non-blank 640x480 PNG resembling the twin preview. Record seconds per
   frame. Two failures -> stop and report. **Checkpoint.**
3. **Camera match** in an override layer (never edit organizer files). **Checkpoint.**
4. **Variation and label layer** per the tables above, plus a per-frame manifest: seed, role,
   level mL, foam thickness, sticker pose, lighting, camera jitter.
5. **Pilot batch** of about 20 frames across all zones; visual and label-integrity check;
   choose renderer mode and counts from measured speed. **Checkpoint.**
6. **Full batch.** Seed ranges by role (train 0-999,999; val 1,000,000-1,999,999; test
   2,000,000-2,999,999). Target about 3,000 train, 300 val and 500 test if speed allows.
   Outputs under ignored `outputs/`. Stop the instance afterwards.
7. **Train** a small U-Net on the vial crop, 4 classes plus ignore, on the Brev GPU. Select on
   val only. **Checkpoint.**
8. **Ruler and rule.** Fit the ruler with the system's own row rule on vials 1 and 13, freeze
   it, and cross-check against the matched twin camera's geometry.
9. **Twin test** on test renders, split by case: clean, foam, sticker, edge zone, hidden.
   **Checkpoint.**
10. **Real check** on final-volume windows: eligible vials minus calibration (27). Report
    agreement with the recipe log, abstentions and counts. **Checkpoint.**
11. Stretch: per-stage real check with the burst detector.
12. Results note with evidence labels. No push or PR without the user's OK.

## Rules that apply

- Train on twin renders only. Real video 5 is validation only; never train on real frames.
- Log every change made by looking at video 5 (camera targets, layout, ruler).
- Raw renders, model weights and real frames stay out of Git.
- Thresholds and targets are frozen before test scoring.
- External messages, pushes, PRs and new spend need the user's OK.

## Open items

- Liam: OK for the submission, and PR vs combined push (decision 19).
- Pause length between doses (affects per-stage frames; readable from burst output).
- Render speed on the L40S, which sets the batch size.
- Glass labeling behaviour in Isaac Sim (step 2).

## Working arrangement

One file per writer, so the two agents never edit the same file:

| Channel | Writer | Reader | Content |
|---|---|---|---|
| Commits on the task branch | Codex | Claude | The code; Claude reviews diffs directly |
| `docs/work/fill-level-cnn-log.md` | Codex | Claude | Newest first: step, commands run (especially on Brev), results, problems, questions for Claude |
| Brev instance outputs | Codex | Claude | Claude inspects and copies back renders independently |
| [fill-level-cnn-review.md](fill-level-cnn-review.md) | Claude | Codex | Per checkpoint: findings (blocking / non-blocking) and go / stop |

Codex commits after each step and updates its log in the same commit. Codex reads the review
file before starting each step. Claude watches the branch for new commits. The user relays
"continue" to Codex when a review entry says go.
