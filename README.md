# Fluid Mixing Station 3: carryover evidence from the cameras

Team **Mixture Of Experts**. Rockwell Automation + NVIDIA Innovation Lab, Fall 2026, Dataset 2 (Fluid
Mixing). Submission materials: [one-page summary](docs/submission/one-page-summary.md),
[video script](docs/submission/video-script.md), [references and credits](docs/submission/credits.md).

## The problem

Station 3 dispenses three tank fluids through one shared path into each vial. When the recipe changes,
residue left in that path tints the first vials of the new batch. The line's existing Teledyne colour
check passes a known colour and fails an unknown one; it does not compare a vial against its own recipe
and it cannot warn that the *next* recipe will be contaminated. Today that means scrap, a manual flush
decision, and no record of which vial was affected or why.

## What we built

A fail-closed evidence pipeline that combines the three camera views with the programmed recipe and
returns `keep`, `discard` or `insufficient_evidence` with the measurements behind it; anything it cannot
support returns `insufficient_evidence`, and simulator masks or synthetic measurements are never passed
off as plant truth. Its stages, and their state as of Nov 1, 2026:

| stage | what it does | state |
|---|---|---|
| Digital twin of the station | Isaac Sim renders of vials and the tank station; frozen train / val / test splits over 438 twin recipes | built; renders and a 9-state tank sweep exist, used as a pipeline canary and as the only training data |
| Top-down cap identity | a U-Net trained **only on twin renders** finds every cap in the real top-down camera and the fixed tracker assigns camera IDs, lane entries and reference-line exits | **measured on all five real videos against the team's YOLO baseline, which was also trained only on twin renders**, see below |
| Dispense stream timing | per-burst segmentation of the dispense stream in the real video: burst count, dispense order | measured on video 5 (30 vials): burst count matches the recipe in 28 of the 29 vials with a pre-pour frame; dispense order recovered from timing in 26 of 28; stream colour does not identify the tank (24 of 73 bursts) |
| Vial colour and fill | masked CIE Lab measurement of the vial; SAM 2.1 zero-shot fill level | implemented on synthetic fills; the SAM 2 notebook runs on real stills, no numbers claimed |
| Beer-Lambert carryover prediction | predict the expected vial absorbance from tank colours and recipe, fit the carryover fraction | implemented and unit-tested; the real-video fit returns negative absorbance, so it is **not usable for plant predictions** |
| Decide | compare the vial's measured colour with the settled colour of its own recipe (leave-one-out median of its position>=2 peers) against the CIE76 2.0 threshold the twin pipeline froze before any real measurement; link the decision to the top-down camera ID | **implemented and measured on video 5** (30 vials, in-sample demonstration, not a validation), see below |
| Explain | write the operator message from the record's evidence | the operator message is a deterministic template (vial, recipe, previous recipe, dE, tau, decision); an optional Cosmos Reason client exists in `src/fluidmix/carryover/explain_cosmos.py` but is not called by the pipeline and has not been run |

## Measured result on the real line

The cap segmenter is the component with a full real-video evaluation. Both detectors were trained only on
twin renders; ours (DenseCapNet, 275k parameters) is ahead of the team's YOLO baseline on per-frame
detection on the five real videos and behind it on track identity coverage:

| model | training data | precision | recall | false boxes in 105k frames | track identity coverage | reference-line exits found |
|---|---|---|---|---|---|---|
| YOLO baseline (train12) | synthetic twin renders only (random init, selection on synthetic val) | 0.945 | 0.873 | 2170 | 0.78 | 35 / 49 |
| **ours, dense53** | **synthetic twin renders only** | **0.957** | **0.920** | 1795 | 0.65 | 29 / 49 |
| ours, dense60 (reproducible variant) | synthetic twin renders only | 0.998 | 0.851 | 89 | 0.44 | 4 / 49 |

Track identity coverage is track_point_TP / (TP + FN) from the harness reports. Frozen harness, all five
real recordings, no real frame used for training, checkpoint selection or the threshold. All five real
videos were scored during the recipe search (25 training runs, dense39 to dense64; videos 1, 3 and 5
through a stride-10 proxy for screening); checkpoint selection within a run and the floor were
synthetic-only, but the choice of augmentation recipe among runs used those real scores, so 0.957 / 0.920
is exposed-validation evidence, not a held-out test. The confidence floor was derived from synthetic
validation only (1st percentile of true-cap confidence, rounded down to 0.05); the rule was written after
Run41's collapse on the real videos had been analysed and before any floored run was scored on real video.
Labels were reviewed by Liam, who also ran the iteration, on videos 2 and 4 (one frame per run of disagreement
plus a spot check every 90 frames: 147 of 148 items answered, one missing cap found); videos 1, 3 and 5,
which carry most of the score, remain unreviewed drafts.

Where the coverage gap comes from, and a protocol comparison: the frozen tracker births a track only in the
entry band, so a cap whose track died during a long dropout in the station holder is never re-born and its later
detections are discarded. A tracker variant declared before scoring (births anywhere in the lane, provisional
until three observations, never inheriting an ID; `src/fluidmix/topdown/conveyor_rebirth_tracking.py`) was
replayed on every model's saved detections with the frozen scorer: coverage baseline 0.78 to 0.86, ours 0.65 to
**0.90**; reference-line exits baseline 35 to 37 of 49, ours 29 to **48 of 49**; false track points 1238 (baseline)
against 821 (ours). Every re-birth is a new camera ID (313 for 49 vials), so the frozen protocol stays the headline;
see [docs/topdown-cap-segmenter.md](docs/topdown-cap-segmenter.md) for the full table.

Where it still fails: short dropouts fragment tracks (249 versus 83) under the frozen tracker, traced to the cap
sitting still in the station holder and to the tracker's entry-band birth rule. Epochs of the dense53 run swing between recall 0.43 and 0.92, and a single-factor
sibling (dense56) between precision 0.22 and 0.99; dense53's selected epoch was a fortunate one; dense60 is
the reproducible variant. The frozen scorer and the draft trajectory labels are not in git; they live on
the team's DGX Spark at `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense-real53/`
(`source/heldout_tracking.py` sha256 8ddf32c0..., `source/evaluate_topdown_segmentation.py` sha256
a2eb4e16..., `trajectories.json` sha256 9fdda3cb...), and the committed harness reports in `data/topdown/`
are their output. Method, failure analysis and reproduction: [docs/topdown-cap-segmenter.md](docs/topdown-cap-segmenter.md);
every run: [docs/topdown-iteration-log.md](docs/topdown-iteration-log.md); scoreboard:
[docs/assets/topdown/scoreboard.html](docs/assets/topdown/scoreboard.html).

### End-to-end carryover flag on video 5

The decision stage was run end to end on the 30 vials of real video 5, whose three cameras are treated as
frame-synchronized (same frame count, same frame index, timestamps agree): the vial's final-mixture colour is measured from the fill camera over its settled
window, compared with the leave-one-out settled colour of its own recipe, flagged by a fixed rule, written
up as an operator message, and linked to the camera ID the top-down tracker gave that vial in the lane:
27 of 30 vials linked to a camera ID (unverified: no label checks the links, and a missed lane entry can
shift the greedy assignment by one vial); 18 of the linked vials have a recorded reference-line crossing.
The recipe schedule is a hand label standing in for the PLC's programmed recipe. **The carryover
signature, measured (no thresholds involved):**

| vial position in a run | distance to the recipe's settled colour (CIE76), median [range] | deviation points at the previous recipe (direction cosine, median) |
|---|---|---|
| first vial after a recipe change (n=8) | **22.9** [3.5 to 50.4] | **0.97** (5 of 8 at or above 0.9) |
| second vial (n=10) | 2.3 [0.7 to 15.3] | the three that still deviate (recipes 6, 12, 14) point at the pre-run recipe; the rest are noise |

In this analysis each recipe's settled colour is its single position-3 vial (n=1 per recipe). The first vial
after a changeover looks like a mix of its own recipe and the one before it (fitted share of the previous
recipe median 0.68); the first-vial error grows with the size of the recipe change (Spearman 0.72, n=8) and stays below it in 6 of 8 changeovers (vials 16 and 25 exceed it), and the error is gone by the second or
third vial. That is attribution to a cause, not only detection. Figure and per-vial table:
`outputs/carryover-video5/attribution.md`, regenerated by `scripts/carryover_attribution.py`.

**The fixed decision rule** (tau = 2.0, the CIE76 colour-visibility threshold the team's twin pipeline froze on
twin renders before any real measurement; the reference is the leave-one-out median of position>=2 peers, so
for a position-2 vial it is the position-3 vial alone)
flags 8 of 8 first vials after a changeover, against 7 of 8 for a stand-in for the line's known-colour
check (pass if the colour is within tau of any recipe's reference colour; the real Teledyne outputs are not
in the dataset), which passes vial 22 because its wrong colour happens to be a known one. The rule also
flags 5 of 10 second vials and 5 of 10 later vials; three of those point at the pre-run recipe (residual
carryover), the rest sit just above a threshold that is tight for camera colour. The earlier noise-based rule,
kept as the secondary, gives the same decisions; the twin threshold was adopted after the run, knowing that.
Video 5 has no defect labels, so these counts are agreement with the process assumption, not detection
accuracy: an in-sample demonstration. Per-vial table, limits, rule provenance, camera-ID links and demo
clips: [docs/carryover-video5.md](docs/carryover-video5.md).

## Why other manufacturers would care

- **No line labelling.** Neither detector needed line footage: both came from the twin. The difference
  between them is the augmentation-for-real recipe, the synthetic-derived floor and the reviewed
  evaluation, and a plant with a twin of its station can get a detector before the line is instrumented.
- **Carryover is a sequencing problem.** Knowing which vial is which, and which recipe came before it, is
  what turns a colour reading into "flush before the next recipe" instead of a scrapped batch.
- **Fail closed.** Every decision carries its evidence or abstains, so operators can trust a flag without
  trusting a model.

## Run it

Environment: [docs/quickstart.md](docs/quickstart.md); CPU is enough for everything except training.
`<dataset>` is the organizers' CSI vial-fill dataset root ([docs/data.md](docs/data.md)). Steps marked
**Spark** need the dense53 checkpoint, which is only on the team's DGX Spark (`run_topdown_segmenter.py`
finds it there by default; elsewhere pass `--weights`), or the tracker output that step writes.

```bash
pip install -r requirements-dev.txt   # includes opencv-python, matplotlib and torch (CPU wheels suffice)
python -m pytest tests -q   # 120 of 121 pass; tests/test_tank_station.py::test_committed_asset_rebuilds_exactly fails on this host independently of these branches (a USD snapshot byte-compare)
python scripts/topdown/run_topdown_segmenter.py <dataset>/cap_seal/videos/cap_seal_video_5.mp4 --out outputs/carryover-video5/topdown   # Spark: tracker IDs, 720 to 833 s on the GB10
python scripts/carryover_video5.py --dataset <dataset> --out outputs/carryover-video5   # decision on video 5: records.json + results.md
python scripts/carryover_attribution.py                                             # attribution.md + attribution.png
python scripts/carryover_demo.py --dataset <dataset>                                # Spark: camera-ID links + five three-camera clips
python scripts/carryover_lane_clip.py 25 13 17                                         # Spark: lane-pass clips for vials 25 and 17
```

Datasets and model weights stay out of git ([docs/operations.md](docs/operations.md)). The two segmenter
checkpoints (1.1 MB each) are on the Spark; their sha256 receipts and training protocols are in
[models/topdown/](models/topdown/).

## Repository map

| path | contents |
|---|---|
| [docs/proposal.md](docs/proposal.md), [docs/plan.md](docs/plan.md) | intent and current work |
| [docs/architecture.md](docs/architecture.md) | implemented functions versus proposed stage handoffs |
| [docs/evaluation.md](docs/evaluation.md), [docs/absorbance.md](docs/absorbance.md) | splits, pre-registered metrics, physics limits |
| [docs/stream.md](docs/stream.md), [docs/inn-52-scope.md](docs/inn-52-scope.md) | dispense-stream measurement |
| [docs/tank-station.md](docs/tank-station.md), [docs/twin.md](docs/twin.md) | the Isaac Sim twin |
| [docs/topdown-cap-segmenter.md](docs/topdown-cap-segmenter.md) | the cap segmenter and its real-video evaluation |
| [docs/carryover-video5.md](docs/carryover-video5.md) | the end-to-end carryover decision on video 5: rule, results, limits, camera-ID links, demo clips |
| `src/fluidmix/` | absorbance math, tank-station capture, top-down segmenter and tracker, carryover measure / decide / explain |
| `scripts/`, `tests/`, `data/` | entry points, checks, small reviewed data with provenance |

## Team

Liam Salas (data audit, splits, video timing, segmenter real-video iteration and label review), Owen
Pacetti (agent workflows, tank-station twin, dense U-Net family, frozen harness and YOLO baseline),
Sebastien Huelke (record contract, carryover validation, dispense-stream measurement, SAM 2 fill notebook).
Built with Claude Code and Codex as pair programmers; [docs/submission/credits.md](docs/submission/credits.md) lists data, models, tools and compute.
