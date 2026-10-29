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
