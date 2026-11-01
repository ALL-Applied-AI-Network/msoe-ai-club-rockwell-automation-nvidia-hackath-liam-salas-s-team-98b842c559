# Top-down cap segmenter: synthetic-only training, measured on the real line

**Measured result.** A 275k-parameter U-Net trained only on synthetic twin renders of the station finds
vial caps in the real top-down cap_seal camera videos with **precision 0.957 / recall 0.920** across all
five recordings (105k frames), ahead of the YOLO baseline trained on the real videos (0.945 / 0.873).
No real frame was used for training, checkpoint selection or the confidence threshold. The draft labels
behind the score were human-reviewed on videos 2 and 4 (147 decisions, one label error found; scores moved
by at most 0.001).

| model | training data | precision | recall | false boxes / 105k frames | reference-line exits found |
|---|---|---|---|---|---|
| train12 YOLO baseline | real cap_seal videos | 0.945 | 0.873 | 2170 | 35 / 49 |
| **dense53** (this work) | **synthetic renders only** | **0.957** | **0.920** | 1795 | 29 / 49 |
| dense60 (reproducible EMA variant) | synthetic renders only | 0.998 | 0.851 | 89 | 4 / 49 |

Scored by the frozen evaluation harness (`heldout_tracking.score_tracking`, trajectory-scoped boxes,
IoU 0.5, v4 guarded tracker). Per-run reports: `data/topdown/harness-report-dense53.json`,
`data/topdown/harness-report-dense60.json`, `data/topdown/harness-report-train12-baseline.json`.
Interactive scoreboard with the figures: [docs/assets/topdown/scoreboard.html](assets/topdown/scoreboard.html).

## Why this component matters

Carryover attribution needs to know *which vial* a colour or fill measurement belongs to. The top-down
camera is the one view that sees every vial travel the inspection lane, so cap identity from this camera
is what ties the other cameras' measurements to the recipe sequence. A plant adopting the pipeline would
have to label its own line footage to get a detector; this component shows the detector can come from the
digital twin instead, with the real footage used only to verify it.

## How it works

1. **Synthetic twin renders** (dataset `edge28/dataset02`, 5160 train / 1280 validation frames, 640 x 480,
   per-cap instance masks) are the only training input.
2. **Augmentation for the real world**, all synthetic: hue/blur/noise/JPEG jitter; non-cap patches, drawn
   dark polylines and light rectangles pasted into the lane as negatives; synthetic caps pasted onto that
   structure and onto drawn dark squares (the station holder) as positives, scaled 0.8 to 1.6x.
3. **DenseCapNet** (widths 12/24/48/96, GroupNorm, BCE with positive weight 50 plus Dice, instance-area
   weighting) outputs one cap-probability map at native resolution.
4. **Components and confidence**: pixels above 0.5 form 8-connected components of at least 9 px; each has a
   box, a mask and a confidence equal to its mean probability.
5. **Synthetic-derived floor**: components below 0.90 are dropped. The rule (1st percentile of true-cap
   confidence on synthetic validation, rounded down to 0.05) was fixed before any real scoring.
6. **Tracker**: the fixed `ConveyorGuardedTracker` turns per-frame caps into camera IDs with states, lane
   entries and reference-line exits.

Source: `src/fluidmix/topdown/segmenter.py` (model, tensor prep,
component rule) and the tracker modules beside it. Training script:
`scripts/topdown/train_topdown_dense.py` (the exact dense53
source; it fails closed on any protocol, dataset or source-hash mismatch).

## Run it

```bash
python scripts/topdown/run_topdown_segmenter.py external/external/csi-vial-fill-dataset/cap_seal/videos/cap_seal_video_2.mp4 \
    --weights <path to dense53-width12-ema-none.pt> --out outputs/topdown-demo/video2
```

Writes `overlay.mp4` (boxes, camera IDs, exit flashes), `detections.jsonl` and `summary.json`. Runs at about
65 fps on the DGX Spark GB10 and works on CPU. Weights are kept out of git per the repository rules; the two
checkpoints (1.1 MB each) live on the Spark at
`/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/train01/best.pt` and
`.../dense60/train01/best.pt`, with their training protocols and frozen-model receipts in
`models/topdown/`.

## Where it still fails

- **Short dropouts fragment tracks.** Dense53 misses fewer frames than the baseline in total, but its misses
  come as 1 to 5 frame dropouts mid-track (40% are confidence dips to 0.87, just under the floor; 35% are
  partial masks on 16 px caps). The tracker scores a fragmentation on each one: 249 versus 83 for the baseline.
- **Reference-line exits trail the baseline** (29 versus 35 of 49). In 19 of the 20 misses the cap was
  detected through the crossing but its track had died earlier: long dropouts while the cap sits in the station
  holder, or the guarded tracker rejecting re-observations after a stop-and-restart. Widening the tracker's gap
  tolerance in a replay recovered at most 3 exits, so this is the stationary-cap detection and the tracker's
  restart logic, not a threshold.
- **The recipe is an epoch lottery.** Neighbouring epochs of the dense53 recipe transfer anywhere from 0.22 to
  0.99 precision and no synthetic metric predicts which. Dense53's selected epoch was a fortunate one. The
  EMA-averaged variant (dense60) is reproducible but its tighter synthetic confidences push the floor to 0.95
  and recall to 0.85.
- **Labels are draft trajectories**, reviewed by a teammate on videos 2 and 4 only, and all five videos were
  seen during development (never for training or selection). There is no unexposed real recording.

## Evidence trail

- [docs/topdown-iteration-log.md](topdown-iteration-log.md): every run from Owen's Run38 through dense64,
  what changed, what it scored, and the diagnostics (26 training runs, 5 harness runs).
- `data/topdown/reviewed-labels-videos-2-4.json`: the
  147 human review decisions with provenance.
- Scripts in `scripts/topdown/`: `real_proxy.py` (stride-10 screening),
  `conf_analysis.py` (floor derivation), `rescore_reviewed.py`, `tracker_gap_sim.py` (tracker replay),
  `exit_miss_causes.py`, `demo_overlay.py` (three-panel demo clips).
- Figures in [docs/assets/topdown/](assets/topdown/): real-versus-synthetic frames, dense41's collapse on
  real fixtures, the augmented training frames, the false boxes the floor removes, dense53's dropouts, the
  one fixture the recipe argues with itself about.

Lineage: Owen's Codex thread built the dense U-Net family (dense31 to dense38), the frozen evaluation harness,
the trajectory labels and the tracker; Liam's session took it from Run38 to dense64 and ran the real-video
evaluations.
