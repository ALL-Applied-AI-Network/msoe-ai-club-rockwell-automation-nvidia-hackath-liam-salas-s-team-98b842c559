# Fluid Mixing Station 3: carryover evidence from the cameras

Team **Mixture Of Experts**. Rockwell Automation + NVIDIA Innovation Lab, Fall 2026, Dataset 2 (Fluid
Mixing). Final submission materials are in [docs/submission/](docs/submission/): the
[one-page summary](docs/submission/one-page-summary.md), the [video script](docs/submission/video-script.md)
and the [references and credits](docs/submission/credits.md).

## The problem

Station 3 dispenses three tank fluids through one shared path into each vial. When the recipe changes,
residue left in that path tints the first vials of the new batch. The line's existing Teledyne colour
check passes a known colour and fails an unknown one; it does not compare a vial against its own recipe
and it cannot warn that the *next* recipe will be contaminated. Today that means scrap, a manual flush
decision, and no record of which vial was affected or why.

## What we built

A fail-closed evidence pipeline that combines the three camera views with the programmed recipe and
returns `keep`, `discard` or `insufficient_evidence` with the measurements behind it. Its stages, and
their state as of Nov 1, 2026:

| stage | what it does | state |
|---|---|---|
| Digital twin of the station | Isaac Sim renders of vials and the tank station; frozen train / val / test splits over 438 twin recipes | built; renders and a 9-state tank sweep exist, used as a pipeline canary and as the only training data |
| Top-down cap identity | a U-Net trained **only on twin renders** finds every cap in the real top-down camera and the fixed tracker assigns camera IDs, lane entries and reference-line exits | **measured on all five real videos against a real-trained YOLO baseline**, see below |
| Dispense stream timing | per-burst segmentation of the dispense stream in the real video: burst count, dispense order | measured on 28 vials: burst count matches the recipe in 28/29, order recovered in 26/28; stream colour does not identify the tank (24/73 bursts) |
| Vial colour and fill | masked CIE Lab measurement of the vial; SAM 2.1 zero-shot fill level | implemented on synthetic fills; the SAM 2 notebook runs on real stills, no numbers claimed |
| Beer-Lambert carryover prediction | predict the expected vial absorbance from tank colours and recipe, fit the carryover fraction | implemented and unit-tested; the real-video fit returns negative absorbance, so it is **not usable for plant predictions** |
| Decide | compare the vial's measured colour with the settled colour of its own recipe (leave-one-out reference) under a fixed rule, tau = max(2.3, 3 x median measurement noise); link the decision to the top-down camera ID | **implemented and measured on video 5** (30 vials, in-sample demonstration, not a validation), see below |
| Explain | write the operator message from the record's evidence | implemented as a deterministic template (vial, recipe, previous recipe, dE, tau, decision); Cosmos is **not used** |

The pipeline does not turn simulator masks or synthetic measurements into plant truth: anything it cannot
support returns `insufficient_evidence`.

## Measured result on the real line

The cap segmenter is the component with a full real-video evaluation. Trained on synthetic renders only,
it is ahead of the baseline that was trained on the real videos:

| model | training data | precision | recall | false boxes in 105k frames | reference-line exits found |
|---|---|---|---|---|---|
| YOLO baseline (train12) | real cap_seal videos | 0.945 | 0.873 | 2170 | 35 / 49 |
| **ours, dense53** | **synthetic renders only** | **0.957** | **0.920** | 1795 | 29 / 49 |
| ours, dense60 (reproducible variant) | synthetic renders only | 0.998 | 0.851 | 89 | 4 / 49 |

Frozen harness, all five real recordings, no real frame used for training, checkpoint selection or the
threshold; labels human-reviewed on videos 2 and 4. Where it still fails: short dropouts fragment tracks
(249 versus 83) and reference-line exits trail the baseline, traced to the cap sitting still in the station
holder and to the tracker's restart logic. Full method, failure analysis and reproduction:
[docs/topdown-cap-segmenter.md](docs/topdown-cap-segmenter.md); every run:
[docs/topdown-iteration-log.md](docs/topdown-iteration-log.md); interactive scoreboard:
[docs/assets/topdown/scoreboard.html](docs/assets/topdown/scoreboard.html).

### End-to-end carryover flag on video 5

The decision stage was run end to end on the 30 vials of real video 5, whose three cameras are
frame-synchronized: the vial's final-mixture colour is measured from the fill camera over its settled
window, compared with the leave-one-out settled colour of its own recipe, flagged by a fixed rule, written
up as an operator message, and linked to the camera ID the top-down tracker gave that vial in the lane
(27 of 30 linked). The recipe schedule is a hand label standing in for the PLC's programmed recipe.

**The carryover signature, measured (no thresholds involved):**

| vial position in a run | distance to the recipe's settled colour (CIE76), median [range] | deviation points at the previous recipe (direction cosine, median) |
|---|---|---|
| first vial after a recipe change (n=8) | **22.9** [3.5 to 50.4] | **0.97** (5 of 8 at or above 0.9) |
| second vial (n=10) | 2.3 [0.7 to 15.3] | the three that still deviate (recipes 6, 12, 14) point at the pre-run recipe; the rest are noise |

The first vial after a changeover looks like a mix of its own recipe and the one before it (fitted share
of the previous recipe median 0.68), the error scales with how different the two recipes are, and it is
gone by the second or third vial. That is attribution to a cause, not only detection. Figure and
per-vial table: `outputs/carryover-video5/attribution.md`, regenerated by `scripts/carryover_attribution.py`.

**The fixed decision rule** (tau = max(2.3, 3 x median within-window measurement noise), leave-one-out
reference) flags 8 of 8 first vials after a changeover, against 7 of 8 for the line's known-colour check,
which passes vial 22 because its wrong colour happens to be a known one. The rule also flags 5 of 10
second vials and 5 of 10 later vials: the noise term collapsed to zero on integer pixel medians, so tau
fell to its hand-chosen floor, and that floor coincides with an older video-5 statistic. Video 5 has no
defect labels, so these counts are agreement with the process assumption, not detection accuracy, and the
run is an in-sample demonstration. Per-vial table, limits, provenance of the rule, the camera-ID link and
the five three-camera demo clips: [docs/carryover-video5.md](docs/carryover-video5.md).

## Why other manufacturers would care

- **No line labelling.** The detector came from the twin, and the real footage was only used to verify it.
  A plant with a twin of its station can get a working detector before the line is instrumented.
- **Carryover is a sequencing problem.** Knowing which vial is which, and which recipe came before it, is
  what turns a colour reading into "flush before the next recipe" instead of a scrapped batch.
- **Fail closed.** Every decision carries its evidence or abstains, so operators can trust a flag without
  trusting a model.

## Run it

Environment: [docs/quickstart.md](docs/quickstart.md) (CPU is enough for everything except training).

```bash
python -m pytest tests -q                       # synthetic math and record-contract checks
python scripts/smoke_absorbance.py              # Beer-Lambert known-answer test
python scripts/topdown/run_topdown_segmenter.py external/external/csi-vial-fill-dataset/cap_seal/videos/cap_seal_video_2.mp4 \
    --weights <dense53-width12-ema-none.pt> --out outputs/topdown-demo/video2   # overlay clip + detections
python scripts/carryover_video5.py              # end-to-end decision on video 5: records.json + results.md
python scripts/carryover_demo.py                # camera-ID links + five three-camera demo clips (cv2 only)
```

Datasets and model weights stay out of git ([docs/operations.md](docs/operations.md)). The CSI dataset and
twin come from the organizers (see [docs/data.md](docs/data.md)); the two segmenter checkpoints (1.1 MB each)
are on the team's DGX Spark and are listed in [models/topdown/](models/topdown/).

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

Liam Salas (data audit, splits, video timing, segmenter real-video iteration), Owen Pacetti (agent
workflows, tank-station twin, dense U-Net family, frozen evaluation harness and YOLO baseline),
Sebastien Huelke (record contract, carryover validation, dispense-stream measurement, SAM 2 fill notebook).
Built with Claude Code and Codex as pair programmers; see [docs/submission/credits.md](docs/submission/credits.md)
for data, models, tools and compute.
