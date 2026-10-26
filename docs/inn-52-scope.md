# INN-52 revised scope: dispense timing and stream colour

Proposed rewording of INN-52 for the team to review. The Linear issue has not been changed;
copy this text there once the team agrees. Measurements behind it are in PR #11
(`docs/stream.md`, `scripts/stream_sequence.py`).

## Why the original scope changes

The original ticket asked which **tube** dispenses, from the fill-station camera, and used the
stream above the vial as the main colour reference. Video 5 measurements show:

- One shared nozzle, no tubes. The tanks and their tubes are out of frame in all five videos,
  so a tube → tank map cannot be drawn. This agrees with [plan](plan.md): three tanks feed one
  nozzle.
- No unobstructed stream. The nozzle tip ends at the vial rim, so the stream is only visible
  through the vial wall, 5–8 px wide, in compressed video that stores colour at half resolution.
- Flow **timing** does carry the sequence. Bursts match the recipe's tank count in 28 of 29
  vials, and burst length per mL is steady for each tank only in the logged order.
- Stream **colour** does not identify the dispensing tank (17 of 73 bursts). It may show liquid
  held in a shared line, which is the carryover mechanism, but the plumbing is unconfirmed.

## Goal (revised)

From the fill-station camera, find when each tank dispenses in each vial, and check the
dispense sequence and amounts against the recipe log. Record the stream colour per burst as an
uncalibrated clue to what is in the shared line, not as tank identity.

## Approach

1. Done in PR #11: detect the stream under the nozzle, group flowing frames into bursts,
   compare burst count, order and length with the recipe.
2. Confirm the plumbing with Liam: where the three tank lines join, the shared line volume
   from the junction to the nozzle, and whether a dose pushes out the previous one first.
3. If a hold-up volume is confirmed, predict the colour of each burst from the recipe history
   and that volume, and compare with the measured colours. That turns the stream into
   carryover evidence for the first vial after a recipe change.
4. Optional: flow-rate drift per tank across the run, as a pump health signal.

Dropped: tube boxes, the tube → tank map, the "before the stream enters the vial" colour
region, and YOLO/SAM segmentation (a fixed 80×200 px crop is enough; ~0.6 ms per frame).

## Acceptance criteria

- [x] Nozzle, stream and holder regions drawn over video-5 frames; framing checked in videos 1–5.
- [x] Per vial: detected bursts vs recipe tanks, with match counts.
- [x] Per-tank flow rate (frames per mL) with spread, logged order vs reversed order.
- [x] Colour per burst, kept separate from tank identity, with limits stated.
- [x] Inference time per frame.
- [ ] Plumbing answers from Liam recorded in docs/stream.md.
- [ ] A short note on how the liquid-level / actual-volume model (INN-35/36) would use burst
      timing as its per-tank volume input.

## Data rules (unchanged)

Real videos 1–5 for inspection only; never training data. Labels from the recipe log, never
from colour ([labels](labels.md)). Raw frames and generated overlays stay out of Git.

## Open questions

- Plumbing (above). Owner: Liam.
- Who owns the actual-volume model referred to as "Joseph's"? It needs its own ticket if it is
  not INN-35/36.
- Does the twin's USD model the nozzle line? If not, stream checks stay real-video only.
