# Top-down cap segmenter: synthetic-only training, measured on the real line

**Measured result.** Both detectors were trained only on twin renders; ours (DenseCapNet, a 275k-parameter
U-Net) is ahead of the team's YOLO baseline (train12) on per-frame detection on the five real top-down
cap_seal videos (105k frames), **precision 0.957 / recall 0.920** versus 0.945 / 0.873, and behind it on
track identity coverage (0.65 versus 0.78; dense60 0.44). No real frame was used for training, checkpoint
selection or the confidence threshold. All five real videos were scored during the recipe search (25
training runs, dense39 to dense64; videos 1, 3 and 5 through a stride-10 proxy for screening): checkpoint
selection within a run and the floor were synthetic-only, but the choice of augmentation recipe among runs
used those real scores, so 0.957 / 0.920 is exposed-validation evidence, not a held-out test. The draft
labels behind the score were reviewed by Liam, who also ran the iteration, on videos 2 and 4 (one frame
per run of disagreement plus a spot check every 90 frames: 147 of 148 items answered, one missing cap
found; scores moved by at most 0.001); videos 1, 3 and 5, which carry most of the score, remain
unreviewed drafts.

| model | training data | precision | recall | false boxes / 105k frames | reference-line exits found | track identity coverage |
|---|---|---|---|---|---|---|
| train12 YOLO baseline (team) | synthetic twin renders only (random init, selection on synthetic VAL) | 0.945 | 0.873 | 2170 | 35 / 49 | 0.78 |
| **dense53** (this work) | **synthetic twin renders only** | **0.957** | **0.920** | 1795 | 29 / 49 | 0.65 |
| dense60 (reproducible EMA variant) | synthetic twin renders only | 0.998 | 0.851 | 89 | 4 / 49 | 0.44 |

Track identity coverage is `track_point_TP / (track_point_TP + track_point_FN)` summed over the five videos
in each harness report. The baseline's provenance is its receipt on the Spark,
`/home/hackathon5/outputs/topdown-01a17f15-20261029/train12/frozen-model.json` (random init,
`real_inputs_used` false, selection on synthetic VAL); it was never trained on real video.

Scored by the frozen evaluation harness (`heldout_tracking.score_tracking`, trajectory-scoped boxes,
IoU 0.5, v4 guarded tracker). The frozen scorer and the draft trajectory labels are not in git; they live on
the team's DGX Spark at `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense-real53/`
(`source/heldout_tracking.py` sha256 `8ddf32c00ace6326e4e0cc1d94a0fba7bf841296802c291f646b1dadb60218ca`,
`source/evaluate_topdown_segmentation.py` sha256
`a2eb4e16681e569ef3eac005ac3719f8bd534a9406a75308b028ba2e2de57e94`, `trajectories.json` sha256
`9fdda3cbbaed66fd15100a9a91e3c37cfe8b8c840d5d1a3324a6143e6e9608ac`), and the committed harness reports in
`data/topdown/` are their output: `data/topdown/harness-report-dense53.json`,
`data/topdown/harness-report-dense60.json`, `data/topdown/harness-report-train12-baseline.json`.
Interactive scoreboard with the figures: [docs/assets/topdown/scoreboard.html](assets/topdown/scoreboard.html).

## Why this component matters

Carryover attribution needs to know *which vial* a colour or fill measurement belongs to. The top-down
camera is the one view that sees every vial travel the inspection lane, so cap identity from this camera
is what ties the other cameras' measurements to the recipe sequence. Neither detector needed line footage:
both were trained on the digital twin's renders alone. The difference between them is the
augmentation-for-real recipe, the synthetic-derived confidence floor and the reviewed evaluation, with the
real footage used only to measure the result.

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
5. **Synthetic-derived floor**: components below 0.90 are dropped. The rule is derived from synthetic
   validation only (1st percentile of true-cap confidence, rounded down to 0.05); the rule was written after
   Run41's collapse on the real videos had been analysed and before any floored run was scored on real video.
6. **Tracker**: the fixed `ConveyorGuardedTracker` turns per-frame caps into camera IDs with states, lane
   entries and reference-line exits.

Source: `src/fluidmix/topdown/segmenter.py` (model, tensor prep,
component rule) and the tracker modules beside it. Training script:
`scripts/topdown/train_topdown_dense.py` (the exact dense53
source; it fails closed on any protocol, dataset or source-hash mismatch), with its companion modules
beside it (see the evidence trail).

## Run it

```bash
python scripts/topdown/run_topdown_segmenter.py external/external/csi-vial-fill-dataset/cap_seal/videos/cap_seal_video_2.mp4 \
    --weights <path to dense53-width12-ema-none.pt> --out outputs/topdown-demo/video2
```

Writes `overlay.mp4` (boxes, camera IDs, exit flashes), `detections.jsonl` and `summary.json`. Runs at
70.7 fps on the DGX Spark GB10 over video 5 (measured) and works on CPU. The runner uses the same model,
floor, lane and tracker settings as the scored run, but it is not a bit-exact replay of the harness:
floor-edge components (confidence within about 0.005 of the 0.90 floor) can flip because GPU inference is
not bit-reproducible, and a full replay of video 2 differed from the frozen harness output on 82 of 4802
frames, all of them such floor-edge boxes. The scored numbers come from the harness reports, not from this
runner. Weights are kept out of git per the repository rules; the two checkpoints (1.1 MB each) live on the
Spark at `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/train01/best.pt` and
`.../dense60/train01/best.pt`, with their training protocols and sha256 receipts in `models/topdown/`.

## Where it still fails

- **Short dropouts fragment tracks.** Dense53 misses fewer frames than the baseline in total, but its misses
  come as 1 to 5 frame dropouts mid-track (40% are confidence dips to 0.87, just under the floor; 35% are
  partial masks on 16 px caps). The tracker scores a fragmentation on each one: 249 versus 83 for the baseline.
  This is why track identity coverage trails the baseline (0.65 versus 0.78) although per-frame detection is ahead.
- **Reference-line exits trail the baseline** (29 versus 35 of 49). In 19 of the 20 misses the cap was
  detected through the crossing but its track had died earlier: long dropouts while the cap sits in the station
  holder, or the guarded tracker rejecting re-observations after a stop-and-restart. Widening the tracker's gap
  tolerance in a replay recovered at most 3 exits, so this is the stationary-cap detection and the tracker's
  restart logic, not a threshold.
- **The recipe is an epoch lottery.** Epochs of the dense53 run swing between recall 0.43 and 0.92, and a
  single-factor sibling (dense56) between precision 0.22 and 0.99; no synthetic metric predicts which.
  Dense53's selected epoch was a fortunate one; dense60 is the reproducible variant. The EMA-averaged dense60's
  tighter synthetic confidences push the floor to 0.95 and recall to 0.85.
- **Labels are draft trajectories**, reviewed by Liam, who also ran the iteration, on videos 2 and 4 (one
  frame per run of disagreement plus a spot check every 90 frames: 147 of 148 items answered, one missing
  cap found); videos 1, 3 and 5, which carry most of the score, remain unreviewed drafts. All five videos
  were scored during the recipe search (never for training or checkpoint selection; the choice of recipe
  among runs was made on these same videos, so the 0.957 / 0.920 is exposed-validation evidence, not a
  held-out test). There is no unexposed real recording.

## Evidence trail

- [docs/topdown-iteration-log.md](topdown-iteration-log.md): every run from Owen's Run38 through dense64,
  what changed, what it scored, and the diagnostics (25 training runs, dense39 to dense64, there is no
  dense50; 5 harness runs).
- `data/topdown/reviewed-labels-videos-2-4.json`: the
  147 human review decisions with provenance.
- Scripts in `scripts/topdown/`: `real_proxy.py` (stride-10 screening),
  `conf_analysis.py` (floor derivation), `rescore_reviewed.py`, `tracker_gap_sim.py` (tracker replay),
  `exit_miss_causes.py`, `demo_overlay.py` (three-panel demo clips), `review_items.py` (review item
  selection). These diagnostics are bound to the Spark run tree at
  `/home/hackathon5/outputs/topdown-01a17f15-20261029` (paths at the top of each file) and to the harness
  sources there; they are kept for provenance, not as a clone-runnable pipeline, and do not run from a clean
  checkout.
- Training: `scripts/topdown/train_topdown_dense.py` and its companion modules
  `topdown_dense_segmenter.py`, `dense_validation_metrics.py`, `instance_balanced_loss.py`,
  `epoch_schedule.py` and `score_instance_masks.py` (imported by `dense_validation_metrics.py`) are byte
  copies of `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/source/`; each copy's sha256 was
  verified against `source_sha256` in `models/topdown/dense53-frozen-model.json`. Training works on the
  Spark with `python scripts/topdown/train_topdown_dense.py --root /home/hackathon5/outputs/topdown-01a17f15-20261029/dense53`:
  it requires the run root with `protocol.json`, the preflight receipts (`cpu-proof.json`,
  `data-audit.json`, `cpu-fixture.pt`) and the synthetic dataset the protocol names
  (`edge28/dataset02`, manifest sha256 in the receipt) at that location, so it is not runnable from a
  clean checkout.
- Figures in [docs/assets/topdown/](assets/topdown/): real-versus-synthetic frames, dense41's collapse on
  real fixtures, the augmented training frames, the false boxes the floor removes, dense53's dropouts, the
  one fixture the recipe argues with itself about.

Lineage: Owen's Codex thread built the dense U-Net family (dense31 to dense38), the frozen evaluation harness,
the trajectory labels and the tracker; Liam's session took it from Run38 to dense64 and ran the real-video
evaluations.
