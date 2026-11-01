# INN-52 revised scope: dispense timing and stream colour

Proposed rewording of INN-52 for the team to review. The Linear issue has not been changed;
copy this text there once the team agrees. Measurements behind it are in PR #11
(`docs/stream.md`, `scripts/stream_sequence.py`).

## Why the original scope changes

The original ticket asked which **tube** dispenses, from the fill-station camera, and used the
stream above the vial as the main colour reference. Video 5 measurements show:

- The plumbing is known: each tank has its own tube, and the tubes join into one shared
  dispensing tube that ends at the nozzle (team, 2026-10-26; agrees with [plan](plan.md)). The
  tank tubes and the junction are out of frame in all five videos, and the camera sees only the
  nozzle, so a tube → tank map cannot be drawn from video.
- No unobstructed stream. The nozzle tip ends at the vial rim, so the stream is only visible
  through the vial wall, 5–8 px wide, in compressed video that stores colour at half resolution.
- Flow **timing** does carry the sequence. Bursts match the recipe's tank count in 28 of 29
  vials, and burst length per mL is steady for each tank only in the logged order.
- Stream **colour** does not identify the dispensing tank. With tank colours fitted from the
  data rather than preset, a burst's colour is nearest the dispensing tank in 24 of 73 bursts
  and the previous dose in 25 of 73, about chance. A clear-tank dose can come out visibly blue
  (vial 8) because liquid left in the shared dispensing tube comes out first. That is the
  carryover mechanism.

## Goal (revised)

From the fill-station camera, find when each tank dispenses in each vial, and check the
dispense sequence and amounts against the recipe log. Record the measured stream colour per
burst (CIELAB shift, hue, chroma, sRGB) as an uncalibrated clue to what the shared dispensing
tube holds, not as tank identity.

## Approach

1. Done in PR #11: detect the stream under the nozzle, group flowing frames into bursts,
   compare burst count, order and length with the recipe.
2. Get the volume of the shared dispensing tube, from the junction to the tip, from Liam.
3. With that volume, predict the colour of each burst from the recipe history and compare it
   with the measured colours. That turns the stream into carryover evidence for the first
   vial after a recipe change.
4. Optional: flow-rate drift per tank across the run, as a pump health signal.

Dropped: tube boxes, the tube → tank map, the "before the stream enters the vial" colour
region, and YOLO/SAM segmentation (a fixed 80×200 px crop is enough; ~0.6 ms per frame).

## Acceptance criteria

- [x] Nozzle, stream and holder regions drawn over video-5 frames; framing checked in videos 1–5.
- [x] Per vial: detected bursts vs recipe tanks, with match counts.
- [x] Per-tank flow rate (frames per mL) with spread, logged order vs reversed order.
- [x] Colour per burst, kept separate from tank identity, with limits stated.
- [x] Inference time per frame.
- [ ] Shared dispensing tube volume from Liam recorded in docs/stream.md.
- [ ] Measured stream and vial colours stored per burst and per vial (separate PR).
- [ ] A short note on how the liquid-level / actual-volume model (INN-35/36) would use burst
      timing as its per-tank volume input.

## Data rules (unchanged)

Real videos 1–5 for inspection only; never training data. Labels from the recipe log, never
from colour ([labels](labels.md)). Raw frames and generated overlays stay out of Git.

## Open questions

- Shared dispensing tube volume (above). Owner: Liam.
- Who owns the actual-volume model referred to as "Joseph's"? It needs its own ticket if it is
  not INN-35/36.
- Does the twin's USD model the nozzle line? If not, stream checks stay real-video only.
