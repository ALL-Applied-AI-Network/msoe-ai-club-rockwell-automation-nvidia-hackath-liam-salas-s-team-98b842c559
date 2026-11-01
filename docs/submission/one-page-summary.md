# Fluid Mixing Station 3: carryover evidence from the cameras

**Team Mixture Of Experts** (Liam Salas, Owen Pacetti, Sebastien Huelke). Dataset 2, Fluid Mixing.
Repository: github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559

## The problem on this line

Station 3 dispenses three tank fluids through one shared path. When the recipe changes, residue in that
path tints the first vials of the new batch. The existing Teledyne check passes a known colour and fails
an unknown one; it cannot say whether a vial matches its own recipe, which vial was affected, or that the
next recipe needs a flush. The cost is scrap, a manual flush decision with no evidence, and no record.

## What we built

A fail-closed evidence pipeline over the three synchronized cameras plus the programmed recipe. It tracks
every vial through the inspection lane from the top-down camera, measures the dispense bursts and the vial
colour, compares the vial's colour with the settled colour of its own recipe, measured from the later
vials of that recipe, and returns `keep`, `discard` or `insufficient_evidence` with the measurements
attached. It abstains rather than guess, and its learned parts are trained on an Isaac Sim digital twin
of the station, not on line footage.

## Measured results against a baseline

**Cap identity from the top-down camera (scored on all five real videos, 105k frames, frozen harness).**
Both detectors were trained only on twin renders; ours (DenseCapNet, 275k parameters) is ahead of the
team's YOLO baseline on per-frame detection and behind it on track identity coverage:

| model | training data | precision | recall | track identity coverage | reference-line exits found |
|---|---|---|---|---|---|
| YOLO baseline (train12) | twin renders only (random init, selection on synthetic val) | 0.945 | 0.873 | 0.78 | 35 / 49 |
| **ours (dense53)** | **twin renders only** | **0.957** | **0.920** | 0.65 | 29 / 49 |
| ours, reproducible variant (dense60) | twin renders only | 0.998 | 0.851 | 0.44 | 4 / 49 |

No real frame was used for training, checkpoint selection or the confidence threshold. All five real
videos were scored during the recipe search (25 training runs, dense39 to dense64; videos 1, 3 and 5
through a stride-10 proxy for screening); checkpoint selection within a run and the floor were
synthetic-only, but the choice of augmentation recipe among runs used those real scores, so 0.957 / 0.920
is exposed-validation evidence, not a held-out test. Track identity coverage is track_point_TP / (TP + FN)
from the harness reports. The draft labels were reviewed by Liam, who also ran the iteration, on videos 2
and 4 (one frame per run of disagreement plus a spot check every 90 frames: 147 of 148 items answered, one
missing cap found); videos 1, 3 and 5, which carry most of the score, remain unreviewed drafts.

**Dispense-stream timing (video 5, 30 vials):** burst count matches the recipe in 28 of the 29 vials with
a pre-pour frame; dispense order recovered from timing in 26 of 28. Stream colour does not identify the
tank (24 of 73 bursts), so tank identity comes from timing, not colour.

**End-to-end on video 5 (30 real vials, three frame-synchronized cameras):** each vial's settled
colour is measured from the fill camera and compared with its own recipe's settled colour, then linked
to the top-down camera ID: 27 of 30 vials linked (unverified: no label checks the links, and a missed lane
entry can shift the greedy assignment by one vial); 18 of the linked vials have a recorded reference-line
crossing. The first vial after a recipe change sits a median 22.9 CIE76 units from its recipe's settled
colour (the single position-3 vial of that recipe), the second a median 2.3, and the first vial's
deviation points at the previous recipe (direction cosine median 0.97, fitted share of the previous recipe
median 0.68): carryover attributed to its cause, decaying within two vials. A fixed rule flags 8 of 8 such
vials against 7 of 8 for a stand-in for the line's known-colour check (pass if the colour is within tau of
any recipe's reference colour; the real Teledyne outputs are not in the dataset), which passes vial 22
because its wrong colour is a known one; the rule also flags half of the later vials, because its threshold
floor decided every vial. Video 5 has no defect labels, so this is agreement with the process assumption
on one recording, not a validation.

## Where it still fails

Short detection dropouts split tracks three times as often as the baseline (249 versus 83), so track
identity coverage (0.65 against 0.78) and confirmed reference-line exits (29 of 49 against 35) trail it;
the causes are the cap sitting still in the station holder and the tracker's stop-and-restart logic, both
reproducible. Epochs of the dense53 run swing between recall 0.43 and 0.92, and a single-factor sibling
(dense56) between precision 0.22 and 0.99; dense53's selected epoch was a fortunate one; dense60 is the
reproducible variant. The Beer-Lambert fit on real video returns negative absorbance and is not used. The
carryover threshold's noise term collapsed to zero, so its hand-chosen floor decided every vial and
flagged half of the later vials. The operator message is a deterministic template; an optional Cosmos
Reason client exists in `src/fluidmix/carryover/explain_cosmos.py` but is not called by the pipeline and
has not been run. All five real videos were seen during development, so every result is a verification on
exposed footage, not a blind test.

## Why it matters to other manufacturers

Carryover is a sequencing problem: the value is in knowing which vial is which and which recipe came
before it. Neither detector needed line footage; the difference between them is the augmentation-for-real
recipe, the synthetic-derived floor and the reviewed evaluation, so a plant with a twin of its station
can have a working vision stage before anyone labels line footage, and every flag carries its evidence or
abstains, which is what makes an operator trust it. Adoption needs the existing cameras, a twin of the
station, the detector's confidence floor (synthetic-derived, already frozen) and a carryover colour
threshold tau frozen on the twin (ours is a hand-chosen floor, not yet frozen on the twin); no retraining
on the plant's product.

## Compute and tools

NVIDIA Isaac Sim 6.0 and OpenUSD for the twin; PyTorch and OpenCV; SAM 2.1 for fill level; DGX Spark
GB10 for training and evaluation, plus roughly five to six instance-hours on two Brev L40S instances (runs
dense47, dense49, dense52, dense55, dense63, dense64; exact credits are in the Brev console, not recorded
here); Claude Code and Codex as pair programmers. Data: the CSI vial-fill dataset and twin from the
organizers.
