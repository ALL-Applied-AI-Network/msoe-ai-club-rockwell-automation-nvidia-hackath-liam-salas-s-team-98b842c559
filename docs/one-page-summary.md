# Fluid Mixing Carryover — Evidence Pipeline Summary

## The problem

After a recipe change, residue in the shared dispensing path can leave the nozzle with a
later dispense and alter one or more vials. Per Liam's 2026-10-26 correction of the organizer
description, the tanks dispense sequentially and the fluids mix **inside the vial**, not
upstream. The residue volume is currently unmeasured. The intended system uses only the
`vial_fill`, `tank_level` and `cap_seal` camera feeds plus programmed recipe volumes, tank
order and vial sequence. It should return `keep`, `discard` or `insufficient_evidence` for
each vial and explain the quantitative evidence; Cosmos does not detect defects.

## What is implemented

The current end-to-end diagnostic starts with synthetic `vial_fill` evidence. Isaac Sim 6.2
renders the fixed twin scene and FillCam view while the recipe's active tanks dispense in order. One
synthetic run contains three vials; tank colours stay fixed within a run and vary between
runs. A leading, uncalibrated residue portion is tracked only in label provenance and never
enters the decision inputs. Streams, rising fill level and height-banded in-vial colour are
visible, but the rendered body is volume-driven and the particle fluid remains an
approximation.

Every stage RGB and aligned liquid mask is hash-bound. The CPU path measures median CIE Lab
inside the liquid mask and computes same-stage dE76 colour differences. Conservative triage
compares a vial with its two same-run peers. It does not assume that an agreeing pair is clean,
so thin reference evidence causes an explicit abstention. Cosmos3-Nano receives supplied
numbers, units and evidence IDs only for explanation. A deterministic template is the
accepted fallback.

The present measurements use Isaac semantic masks, which are simulator oracles rather than
camera predictions. A strict schema-v2 intake is implemented for future RGB-only predicted
masks. It binds frozen model/config hashes and rejects labels, residual parameters and other
simulator truth. A hash-bound temporal extension also covers a same-vial pre-pour RGB reference.
The first fixed temporal baseline was rejected on validation: mean Dice was 0.9495, but mean mask-
height error was 30.17 px versus a frozen 5 px limit. Evaluation stayed unopened, and no accepted
predicted-mask artifact exists yet.

## Measured synthetic result — 2026-10-27

| Check | Observed result | What it does **not** establish |
|---|---|---|
| Rendering | 18/18 cases passed the current hard render gates; 18 videos, 5,886 decoded frames and 42 stage RGB/mask pairs | 15 cases retain 16 particle-height warnings, so these are render passes, not particle-physics passes |
| Measurement | 42 masked-Lab stage rows and 42 same-stage pairwise dE76 rows across six whole-run groups | The semantic masks are not RGB-model predictions |
| Vial decisions | 18/18 `insufficient_evidence`; 0 `keep`, 0 `discard` | There is no keep/discard coverage or detector-accuracy claim |
| Post-hoc diagnostic | Relative-outlier flag: TP=5, FN=1, FP=1, TN=11 against separately joined synthetic labels | This exploratory flag is not a disposition or frozen detector metric |
| Cosmos3-Nano | 72/72 attempts completed; strict content passed 0/54 model attempts and required abstention passed 1/54 | Negative evidence for this frozen prompt/runtime treatment only |
| Deterministic explanation | Parsing, content and required abstention passed 18/18 | A diagnostic fallback, not a deployed operator service |

The accepted same-host review bundle is
`/home/hackathon5/liamreporoot/f2-varied-v3-demo-v2-20261027/`. It links 18 abstaining
records to 42 rehashed images and can be rebuilt with the command in the repository README.

## Claim boundary and next actions

This is synthetic-only evidence. It does not prove physical carryover accuracy, chemistry,
cause identification, real-camera transfer or plant performance. Real calibration,
validation and test partitions must stay disjoint, and learned thresholds must freeze before
test scoring. The immediate next step is for the separate training track to publish RGB-only
mask predictions through the schema-v2 contract and score them on held-out whole-run synthetic
families. After that, the same deterministic measurement/abstention boundary can be exercised
without oracle masks. Real labelled defect runs are still required before any plant claim.

Detailed provenance and limitations are in [data](data.md), [architecture](architecture.md),
[evaluation](evaluation.md) and the [current plan](plan.md#twins-job).
