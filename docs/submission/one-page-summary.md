# Fluid Mixing Station 3: carryover evidence from the cameras

**Team Mixture Of Experts** (Liam Salas, Owen Pacetti, Sebastien Huelke). Dataset 2, Fluid Mixing.
Repository: github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559

## The problem on this line

Station 3 dispenses three tank fluids through one shared path; after a recipe change, residue in that path
tints the first vials of the new batch. The existing colour check passes a known colour and fails an
unknown one; it cannot say whether a vial matches its own recipe, which vial was affected, or that the next
recipe needs a flush. The cost is scrap, a manual flush with no evidence, and no record.

## What we built

A fail-closed evidence pipeline over the three cameras plus the programmed recipe: it tracks every vial
from the top-down camera, measures the dispense bursts and the vial colour, compares that colour with the
settled colour of the vial's own recipe, and returns `keep`, `discard` or `insufficient_evidence` with the
measurements attached. Every part we trained was trained on an Isaac Sim twin, never on line footage;
the only pretrained model is SAM 2, used as is.

## Measured results against a baseline

**Cap identity from the top-down camera**, scored on all five real videos (105k frames) with a frozen
harness. Both detectors were trained only on twin renders:

| model | precision | recall | track identity coverage |
|---|---|---|---|
| team YOLO baseline (train12) | 0.945 | 0.873 | 0.78 |
| **ours (DenseCapNet, 275k parameters)** | **0.957** | **0.920** | 0.65 |

Ahead on per-frame detection, behind on track identity. No real frame was used for training, checkpoint
selection or the confidence floor; all five videos were scored while choosing the augmentation recipe
among 25 runs, so this is exposed validation, not a blind test. Labels were reviewed by Liam on videos 2
and 4 (147 decisions, one missing cap found).

**Dispense-stream timing** (real video 5): burst count matches the recipe in 28 of the 29 vials with a
pre-pour frame; dispense order recovered from timing in 26 of 28; stream colour does not identify the tank.

**End to end on video 5** (30 real vials, three cameras treated as frame-synchronized): each vial's settled
colour is measured from the fill camera and compared with its own recipe's settled colour, then linked to
its top-down camera ID (27 of 30, unverified by labels). The first vial after a recipe change sits a median
22.9 CIE76 units from its recipe's settled colour, the second a median 2.3, and the first vial's deviation
points at the previous recipe (direction cosine median 0.97): carryover attributed to its cause, decaying
within two vials. A fixed rule (CIE76 2.0, the threshold the twin pipeline froze before any real measurement) flags 8 of 8
such vials against 7 of 8 by a stand-in for the line's known-colour check, which passes vial 22 because its
wrong colour is a known one. The rule also flags half of the later vials, three of them residual carryover. Video 5 has no defect labels, so
this is agreement with the process assumption on one recording, not a validation.

## Where it still fails

Short detection dropouts split tracks three times as often as the baseline, which costs track identity and
reference-line exits (29 of 49 against 35). The headline model's epoch was a fortunate one; a reproducible
variant scores 0.998 / 0.851. The carryover threshold is twin-frozen but was adopted after the run; its decisions match the earlier
hand-chosen rule. The Beer-Lambert fit on
real video returns negative absorbance and is not used. The operator message is a deterministic template;
an optional Cosmos Reason client exists but is not called by the pipeline and has not been run.

## Why it matters to other manufacturers

Carryover is a sequencing problem: the value is in knowing which vial is which and which recipe came
before it. Neither detector needed line footage, so a plant with a twin of its station can have a vision
stage before anyone labels its line, and every flag carries its evidence or abstains. Adoption needs the
existing cameras, a twin, and a carryover threshold frozen on the twin before plant data, as ours is.

## Compute and tools

Isaac Sim 6.0 and OpenUSD (twin); PyTorch, OpenCV; SAM 2.1 (fill level); DGX Spark GB10 and two Brev L40S
instances; Claude Code and Codex as pair programmers. Data: the organizers' CSI vial-fill dataset and twin.
