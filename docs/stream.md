# Dispense stream in the fill-station video (INN-52)

Measured 2026-10-26 on `vial_fill_video_5.mp4`; its `vial_fill` files match
`data/dataset_manifest.csv`. Real video, inspection only: nothing here trains a model.
Thresholds were chosen while looking at the same 30 vials, so the counts below describe the
signal; they are not a held-out evaluation (see [evaluation](evaluation.md)).

    python scripts/stream_sequence.py           # writes outputs/inn52/stream_*.csv and a timeline PNG

## What the camera shows (measured)

- All five vial_fill videos are 640×480 H.264, `yuv420p`, 30 fps, at roughly 5–51 kbit/s.
  `yuv420p` stores colour at half resolution (320×240), so colour edges are about 2 px wide.
- **One nozzle, no tubes.** Every video shows a single metal nozzle at x≈293–315 (20–23 px wide)
  with the same framing. The tanks and their tubes are out of frame. This matches the plant
  description in [plan](plan.md): three tanks feed one nozzle. So the camera cannot show which
  tube is flowing.
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
| Burst colour = tank now dispensing | 17 of 73 bursts |
| Burst colour = previous dose | 33 of 73 bursts |
| Measurement time | ~0.6 ms per frame on a laptop CPU, excluding video decoding |

What this means:

- **Timing works.** Each tank pours at its own steady rate, and only the logged order gives
  steady rates. So stream timing alone recovers which tank pours when, and roughly how much,
  given the recipe's tank list. Converted with the measured clock rate (1.2157 wall s per video s,
  [video timing](video_timing.md)) and assuming the logged mL were dispensed: about 6.4, 2.3 and
  3.3 mL/s for T-100, T-200 and T-400. The T-400 spread includes vial 29's partly detected first
  burst.
- **Colour does not identify the tank.** The stream colour matches the dispensing tank in only
  17 of 73 bursts. Examples: in vials 25–27 (T-200 then T-400), both bursts read orange; in
  vial 28 (T-400 then T-100, no T-200) both read orange.
- **The stream may show what sits in a shared line.** A one-dose lag ("each burst pushes out
  the previous dose") matches more bursts (33 of 73) but still fewer than half, so the hold-up
  volume, if that is the cause, is probably not one dose. That is the carryover mechanism the
  project is about, but the plumbing is unconfirmed: we do not know the line volume or where the
  three tank lines join. Treat this as a hypothesis.
- The stream is pale and thin, seen through the vial wall, and compressed. Its colour is an
  uncalibrated reference only, as the ticket warned. Do not feed it into absorbance math.

## Limits

- Fixed regions were placed by hand on video 5; videos 1–4 share the framing but have no
  recipes, so the burst checks were not run there.
- The colour names (clear/orange/blue) are thresholds on a CIELAB shift, chosen by eye.
- Fill windows come from the recipe log and were checked by eye on one vial ([labels](labels.md)).
  A burst near a window edge can be cut short.
- The twin's stream is cyan for every liquid (INN-45), so none of this can be trained on twin
  data yet.

Generated per-frame, per-burst and per-vial CSVs and the timeline image stay in
`outputs/inn52/` (ignored by Git).
