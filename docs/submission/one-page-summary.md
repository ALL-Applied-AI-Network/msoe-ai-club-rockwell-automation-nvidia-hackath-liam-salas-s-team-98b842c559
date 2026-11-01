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
colour, predicts the expected colour from the recipe, and returns `keep`, `discard` or
`insufficient_evidence` with the measurements attached. It abstains rather than guess, and its learned
parts are trained on an Isaac Sim digital twin of the station, not on line footage.

## Measured results against a baseline

**Cap identity from the top-down camera (synthetic training only, scored on all five real videos,
105k frames, frozen harness, labels human-reviewed on two videos):**

| model | training data | precision | recall | reference-line exits found |
|---|---|---|---|---|
| YOLO baseline | real line videos | 0.945 | 0.873 | 35 / 49 |
| **ours** | **digital-twin renders only** | **0.957** | **0.920** | 29 / 49 |
| ours, reproducible variant | digital-twin renders only | 0.998 | 0.851 | 4 / 49 |

No real frame was used for training, checkpoint selection or the confidence threshold. The detector that
has never seen the line beats the one trained on it.

**Dispense-stream timing (real video, 28 vials):** burst count matches the recipe in 28 of 29 vials and
the dispense order is recovered in 26 of 28. Stream colour alone does not identify the dispensing tank
(24 of 73 bursts), so tank identity comes from timing, not colour.

**End-to-end on video 5 (30 real vials, three frame-synchronized cameras):** each vial's settled
colour is measured from the fill camera and compared with its own recipe's settled colour, then linked
to the top-down camera ID (27 of 30). The first vial after a recipe change sits a median 22.9 CIE76
units from its recipe's settled colour, the second a median 2.3, and the first vial's deviation points
at the previous recipe (direction cosine median 0.97, fitted share of the previous recipe median 0.68):
carryover attributed to its cause, decaying within two vials. A fixed rule flags 8 of 8 such vials
against 7 of 8 by the line's known-colour check, which passes vial 22 because its wrong colour is a known
one; the rule also flags half of the later vials, because its threshold floor decided every vial. Video 5
has no defect labels, so this is agreement with the process assumption on one recording, not a validation.

## Where it still fails

Short detection dropouts split tracks three times as often as the baseline, so fewer reference-line exits
are confirmed (29 of 49 against 35); the causes are the cap sitting still in the station holder and the
tracker's stop-and-restart logic, both reproducible. The Beer-Lambert fit on real video returns negative
absorbance and is not used. The carryover threshold's noise term collapsed to zero, so its hand-chosen
floor decided every vial and flagged half of the later vials; the operator message is a template, with an
optional Cosmos Reason call that has not been exercised live. All five real videos were seen during development, so every result is a verification on
exposed footage, not a blind test.

## Why it matters to other manufacturers

Carryover is a sequencing problem: the value is in knowing which vial is which and which recipe came
before it. Our detector came from the twin, so a plant with a twin of its station can have a working
vision stage before anyone labels line footage, and every flag carries its evidence or abstains, which is
what makes an operator trust it. Adoption needs the existing cameras, a twin of the station, and a
threshold frozen on synthetic data (ours is not yet); no retraining on the plant's product.

## Compute and tools

NVIDIA Isaac Sim 6.0 and OpenUSD for the twin; PyTorch and OpenCV; SAM 2.1 for fill level; DGX Spark
GB10 and Brev L40S (about 6 GPU-hours) for training and evaluation; Claude Code and Codex as pair
programmers. Data: the CSI vial-fill dataset and twin from the organizers.
