# Dispense stream in the fill-station video (INN-52)

Measured 2026-10-26 on `vial_fill_video_5.mp4`; its `vial_fill` files match
`data/dataset_manifest.csv`. Real video, inspection only: nothing here trains a model.
Thresholds were chosen while looking at the same 30 vials, so the counts below describe the
signal; they are not a held-out evaluation (see [evaluation](evaluation.md)).

    python scripts/stream_sequence.py           # writes outputs/inn52/stream_*.csv and a timeline PNG

## What the camera shows (measured)

- All five vial_fill videos are 640×480 H.264, `yuv420p`, 30 fps, at roughly 5–51 kbit/s.
  `yuv420p` stores colour at half resolution (320×240), so colour edges are about 2 px wide.
- **Plumbing (known, team report 2026-10-26).** Each tank has its own tube, and the tubes join
  into one shared dispensing tube that ends at the nozzle. This matches [plan](plan.md): three
  tanks feed one nozzle. The tank tubes and the junction are out of frame in every video. The
  camera sees only the metal nozzle at x≈293–315 (20–23 px wide), with the same framing in
  videos 1–5, so it cannot show which tank tube is flowing. Still unmeasured: the volume of the
  shared tube from the junction to the tip.
- **No unobstructed stream.** The nozzle tip ends at about the vial rim (y≈75), so the whole
  visible stream is behind the vial wall. Below y≈198 the black vial holder hides it.
- **Stream width.** 5–8 px wide in luma (brightness), 6–10 px in colour, where colour is
  smeared by half-resolution storage and compression. That is only 2.5–4 colour samples across,
  so at most one or two are free of background.

## What the stream tells us (measured, 30 vials)

`scripts/stream_sequence.py` compares each frame with a frame 15 frames before the labelled fill
start, finds a thin vertical stripe under the nozzle, and groups flowing frames into bursts.
Vial 1 has no pre-pour frame (its window starts at frame 0), so the counts use 29 vials.

| Check | Result |
|---|---|
| Burst count = tanks in the recipe | 28 of 29 vials. Vial 30 merges or misses its 10 mL T-400 burst |
| Burst length per mL, logged order | T-100 3.85, T-200 10.94, T-400 7.47 video frames/mL; spread (CV) 6%, 5%, 12% |
| Same, order reversed | spread 101%, 77%, 77% |
| Order from timing alone (rates from the other recipes) | 26 of 28 vials. Both misses are recipe 10, where 15 mL T-200 and 45 mL T-100 take about the same time |
| Burst colour nearest the dispensing tank's colour | 24 of 73 bursts (median a\*b\* distance 5.2) |
| Burst colour nearest the previous dose's colour | 25 of 73 bursts (median a\*b\* distance 5.2) |
| Measurement time | ~0.6 ms per frame on a laptop CPU, excluding video decoding |

What this means:

- **Timing works.** Each tank pours at its own steady rate, and only the logged order gives
  steady rates. So stream timing alone recovers which tank pours when, and roughly how much,
  given the recipe's tank list. Converted with the measured clock rate (1.2157 wall s per video s,
  [video timing](video_timing.md)) and assuming the logged mL were dispensed: about 6.4, 2.3 and
  3.3 mL/s for T-100, T-200 and T-400. The T-400 spread includes vial 29's partly detected first
  burst.
- **Order can be read from timing.** The dispense order changes between recipes (team report,
  2026-10-26), and recovering it is part of the problem. Given each recipe's tanks and mL, and
  per-tank rates measured on the other recipes, timing picks the logged order in 26 of 28
  vials. It fails when two tanks' doses take about the same time (recipe 10); there another cue
  has to break the tie (stream colour cannot, see below). Every video-5 recipe logs an order drawn from
  T-200 > T-400 > T-100, so this checks the method against one order family only.
- **How colour is measured.** No tank colour is assumed, so the method works for any dyes.
  Each burst gets its measured colour: the CIELAB shift of the stream against the pre-pour
  frame (ΔL\*, Δa\*, Δb\*), its hue angle and chroma, and the median sRGB of the stream pixels.
  To test a hypothesis, each tank's reference colour is the median shift of the bursts that
  hypothesis assigns to it, taken from the other recipes' vials (leave-one-recipe-out). A burst
  matches when its nearest reference, by a\*b\* distance, is the expected tank.
- **Colour does not identify the tank.** The nearest reference is the dispensing tank in 24 of
  73 bursts and the previous dose in 25 of 73, about what guessing among two or three tanks
  gives. Examples: in vials 25–27 (T-200 then T-400) both bursts have hues of 68–81° (orange);
  in vial 28 (T-400 then T-100) both are 70–82°. An earlier version named colours with
  hand-set thresholds (clear/orange/blue) and got 17 and 33 of 73. That lead for the previous
  dose does not survive once tank colours are fitted from the data.
- **The colour is in the stream itself.** Frame checks rule out the liquid already in the vial
  as the source: in vial 8 the 20 mL T-100 (clear) burst is visibly blue above the holder for
  its whole length (frame 10760, hue 204°), and in vial 28 the T-400 (blue) burst is visibly
  pale orange (frame 47440, hue 70°).
- **The shared dispensing tube explains this.** Liquid left in the shared tube comes out at
  the start of the next dose, so a clear-tank dose can come out blue. The vial 8 case suggests
  the shared tube holds about 20 mL or more. A simple one-dose lag ("each burst shows only the
  previous dose") fits no better than the current dose (25 vs 24 of 73), so the liquid in the
  tube mixes more gradually than a clean push-out. Modelling that needs the tube's volume.
- The stream is pale and thin, seen through the vial wall, and compressed. Its colour is an
  uncalibrated reference only, as the ticket warned. Do not feed it into absorbance math.

## Vial colour next to stream colour (measured, 30 vials)

`stream_sequence.py --store` also reads the vial: the median colour of the liquid below the
holder after each burst and after the fill, against the same region of the pre-pour frame.
Both are stored in `data/` (see [data](data.md)). After a 10 mL first dose the liquid is still
below that region; `vial_region_changed` flags those rows, and the numbers here use the 70
burst rows where at least 90% of the region changed.

- **Stream and vial colours differ.** Median a\*b\* gap 10.7 (middle half 5.7–14.0); hues are
  within 30° in only 5 of 70 bursts. That is expected: the vial holds everything poured so far,
  and the stream holds the current dose plus what the shared tube carried over.

`scripts/vial_identify.py` asks how well the final vial colour says what is in the vial. It
fits colour as a volume-weighted mix of one fitted feature vector per tank (no preset tank
colours) on nine recipes and ranks the ten-recipe menu for the held-out recipe's vials. Guessing
gets about 2 of 20. Steady 2nd/3rd vials:

| Colour source | Nearest vial same recipe | Unseen recipe right | Top 3 | Worst-tank fraction error |
|---|---|---|---|---|
| Inherited `video5_vial_colors.csv` | 15 of 20 | 10–15 of 20, see below | 18 of 20 | 0.12 |
| Measured here, whole region vs empty vial | 13 of 20 | 8 of 20 | 14 of 20 | 0.21 |

- **The inherited result is unstable.** Many inherited colours have a channel at exactly 0,
  so the log in the absorbance feature depends on its floor: 10, 13, 15 and 13 of 20 for
  floors of 1e-2 to 1e-5. Do not quote one of those as the accuracy.
- **The measured colours trade errors.** They identify recipe 14 (60 mL clear + 10 mL blue)
  in both vials, which the inherited colours rank 4th and 9th, because the whole-region median
  keeps the dilution that the most-saturated-pixel method drops. They do worse on the even and
  dark mixes (recipes 5, 8, 9, 10, 11). Vials 8 and 9 read almost 0 in red, so dark vials lose
  a channel here too.
- **First vial after a recipe change:** 0–1 of 10 for both sources. Its colour is often
  closer to the previous recipe (carryover), so a mismatch here is a carryover clue, not yet a
  test.
- Small sample, one video, and the feature and region were chosen on these vials. This shows
  which tanks dominate a settled vial; it does not measure concentrations.

## Liquid level, including clear liquid (measured, 30 vials)

T-100 is colourless, so colour can't show how much of it is in a vial. The level can: clear
liquid still changes brightness where it bends the background. `liquid_level` in
`stream_sequence.py` compares the vial column (x 296–328) with the empty-vial frame row by
row, scans up from the base, skips the holder rows (240–262) and stops at the first 15 dry
rows. `scripts/vial_level.py` turns level into volume with a straight line fitted on other
recipes.

| Readings after a burst | Volume error (median, 90th percentile) |
|---|---|
| Surface below the holder, clear view (40) | 7.5 mL, 25.9 mL |
| Surface above the holder (33) | 19.0 mL, 46.3 mL |
| Clear view, last dose T-100 / T-200 / T-400 | 8.3 / 7.5 / 6.4 mL (n = 5 / 22 / 13) |

- **Only roughly 20–50 mL is in clear view.** Above that the surface is behind the holder top
  and, on many vials, a barcode label (rows ~150–235). Final levels still rank 30 < 40 <
  60–75 < 80 mL, but 60, 70 and 75 mL overlap.
- **Clear liquid is seen about as well as coloured.** Errors after a T-100 dose match the
  others, but only 5 such readings are in clear view.
- **The settings are fragile.** Of 24 threshold settings tried, errors ranged from 5 to 38
  mL. The chosen one was picked on recipes 5, 7, 9, 11 and 13 and gave 7.4 mL on 6, 8, 10,
  12 and 14. Picking on the other half chose a slightly different setting (7.7 mL). The table
  uses the chosen setting on all recipes, so it is a little optimistic.
- **Why:** the measure is change against the empty vial, and that depends on what is behind
  the vial. A dark conveyor band (rows ~300–330) hides most of the change, and wet walls or
  foam above the surface add change. A meniscus edge detector, a plain backdrop behind the
  vial, or the SAM mask height (`docs/sam.md`, PR #14) are the next things to compare.

## While a dose pours (measured, 30 vials)

The checks above use one median colour per burst and one vial reading after it. That hides any
change inside a dose, which is where carryover from the shared tube should show: the previous
liquid comes out first. `stream_sequence.py` now keeps the vial colour, liquid level and burst
number for every frame (`outputs/inn52/stream_frames.csv`), and `scripts/burst_trace.py` splits
each burst's flowing frames into 5 equal parts in time.

    python scripts/stream_sequence.py && python scripts/burst_trace.py   # --store writes data/video5_burst_trace.csv

412 of 13,247 flowing frames are dropped because the level had reached the stream rows (row
190 or above), where the vial's contents sit behind the stream. 71 of the 73 tank-matched
bursts have a usable first and last part.

| Check | Result |
|---|---|
| Colour drift, first to last part | median 2.2 a\*b\* (middle half 1.1–4.4); above 3× that burst's frame-to-frame floor (median 0.17) in 59 of 71 |
| Drift by burst position in the vial | 1st 1.3, 2nd 2.9, 3rd 2.6 (n = 28, 28, 15) |
| After a different tank: first part nearer the previous tank's reference | 37 of 70 (guessing 35) |
| Last part nearer the current tank's reference | 30 of 70 (guessing 35) |
| First part nearer the previous dose's last part than its own last part | 24 of 43 |
| Clear-view level rises within the burst | 56 of 62 (T-100 14/15, T-200 23/23, T-400 19/24) |

What this means:

- **The colour does change during a dose.** The change is well above frame-to-frame noise in
  most bursts. Examples: in vials 13–15 the T-100 dose starts at b\* +8 (the T-400 colour
  before it) and ends near +3; in vial 7 the T-400 dose moves from +b\* to −a\*.
- **Carryover does not explain the direction.** If the tube pushed out the previous liquid
  first, the start of a dose would sit near the previous dose's end and move towards the
  tank's own colour. Neither test beats guessing (37 of 70, 24 of 43). Some vials fit the idea
  (10–14, T-200 then T-400: start within 0.5–2.6 of the previous end), others do not (19–21
  and 23–24, T-400 then T-100: 4.0–6.9 away).
- **Even the end of a dose does not identify its tank.** The tanks' last-part references are
  only 0.9–2.6 a\*b\* apart, and a single burst's last part is a median 5.1 from its own tank's
  reference. So the stream does not settle to a tank colour that carryover could be measured
  against. The whole-burst result above (24 of 73) has the same cause.
- **Drift grows with what is already in the vial.** First bursts drift less (1.3) than later
  ones (2.6–2.9), although first bursts also follow a tank change (the previous vial's last
  dose). Carryover alone would not predict that difference. The background behind the stream
  changing (wet walls, splashes, light through the coloured liquid below) would. This is a
  suspicion, not a test.
- **The vial level follows the dose.** In clear view the surface rises during 56 of 62 bursts.
  Its speed in rows per frame does not follow the pour rates, though: T-200 pours slowest
  (10.9 frames/mL) but rises fastest (−0.27 rows per frame), T-400 slowest (−0.04). T-200 is
  usually the first dose, low in the vial, and rows per mL change with height and with the
  conveyor band behind the vial, so this is a rise/no-rise check, not a flow rate.
- **Next:** measure the shared tube's volume (that sets how much carryover to expect), view
  the stream against a plain backdrop below the vial wall, or use the SAM stream mask (PR #14)
  so the stream pixels exclude the background.

## Limits

- Fixed regions were placed by hand on video 5; videos 1–4 share the framing but have no
  recipes, so the burst checks were not run there.
- Tank reference colours are fitted on these same 30 vials (leave-one-recipe-out). They are
  not calibrated colours, and each is only as good as the burst-to-tank assignment under test.
- Fill windows come from the recipe log and were checked by eye on one vial ([labels](labels.md)).
  A burst near a window edge can be cut short.
- The twin's stream is cyan for every liquid (INN-45), so none of this can be trained on twin
  data yet.

Generated per-frame, per-burst and per-vial CSVs and the timeline image stay in
`outputs/inn52/` (ignored by Git).

## Assumed colour labels

Both colour CSVs include assumed_correct_colour: 1 when the recipe ID matches
the previous vial in the full recorded sequence, otherwise 0. The first recorded
vial is assigned 0 because no previous vial is available. Labels are computed
before --vials filtering. Every burst row inherits its vial label. These are
assumptions, not measured colour validity or verified contamination labels.
