# Fluid Mixing Carryover — one-page evidence summary

## Problem and intended boundary

When a recipe changes, residue in the shared dispensing path may alter a later vial. In
Dataset 2 the tanks dispense sequentially and the fluids mix in the vial; upstream residue
volume is not measured. The intended software combines `vial_fill`, `tank_level` and
`cap_seal` camera evidence with programmed recipe order and volumes, then returns `keep`,
`discard` or `insufficient_evidence` with quantitative support. It is a fail-closed evidence
pipeline, not permission to operate equipment or change a recipe. Cosmos is an explanation
consumer, not a defect detector.

## Implemented pipeline

The accepted path renders and transports fixed synthetic fills, binds each RGB/mask artifact
by hash, measures masked RGB in CIE Lab, compares same-stage colours with CIE76, and applies
peer-based triage. It has strict schemas, provenance and abstention guards. An RGB-only CNN
runtime and six-run development population are accepted for development evidence; the older
V3 regression remains an oracle-mask software check. A deterministic guarded template is the
approved fail-closed explanation fallback. The pipeline does not silently turn simulator
masks, labels or synthetic measurements into camera or plant truth.

## Accepted measured state

- **Fills and transport:** 18/18 L40S **WP1 test fills** passed their accepted render/copy-back
  checks. Across the WP1 test plus V3/V4 development/demo cohorts, **81/84** joins were
  accepted: WP1 **20/20**, V3 **24/24** (6/6 complete runs), and V4 **37/40** (9/10
  complete packaged runs). V4 ordinal 81 begins deferred `RUN-V4-EVALUATION-03`; ordinals
  82–84 are absent. These transport/package checks establish synthetic visual and artifact
  integrity, not physical-liquid validation. There are currently **zero Brev hosts**.
- **RGB-only CNN development:** the accepted six-run synthetic population contains 45
  predicted-mask/stage records across 18 clips, 18 vials and six runs. This supports pipeline
  development only; it does not establish real-camera accuracy, a real-camera or plant
  threshold, or a model winner.
- **WP1 frozen one-pass:** the immutable pass ran once and produced **18 abstentions and 0
  keep/discard decisions** (coverage 0/18). It must not be rerun. The result is an honest
  evidence-availability outcome, not detector-accuracy evidence.
- **DP3 visibility:** the measured 1.5 mL-residue versus 0 mL pair reached maximum CIE76
  1.707241, below the frozen 2.0 visibility threshold. Liam accepted **1.5 mL as the measured
  detection limit** and parked aggregate acceptance; no fallback render or aggregate attempt-2
  is authorized.
- **Video 5:** attempt 1 is a terminal fatal run after RGB/CNN capture, before terminal
  scoring/evidence-store publication, so it has no real-video score. Attempt 2 is prepared but
  **held**; it has not run and requires explicit authorization. INN-64 remains held.

## What is not measured or complete

WP3 required baselines are **not run (0 attempts)**, and the full WP6 model comparison is not
run. The limited all-abstention diagnostic is not a winner or accuracy comparison. WP5 §6
image-grounding/explanation completion is not claimed: the development evidence chain does not
meet the literal referenced-image requirement, so **INN-62 is held**. Synthetic results do
not prove chemistry, physical carryover, real-camera transfer, cause identification, plant
performance, or keep/discard accuracy. Negative absorbance diagnostics and oracle-mask checks
are not plant calibration.

## Next decisions

1. Decide whether to authorize the held video-5 attempt-2 recovery; do not treat the fatal
   attempt-1 artifacts as a score.
2. Decide whether to finish V4 ordinals 82–84 and the remaining three transport joins via a
   separately approved GB10/Brev route; no host is running now.
3. Resolve INN-62's WP5 §6 image-grounding requirement and INN-64's held real-video route.
4. Only after accepted inputs and an authorized frozen plan exist, decide whether to run the
   WP3 baselines/full WP6 comparison. Until then, use the guarded template and
   `insufficient_evidence` for unsupported decisions.

Evidence: local-only, non-portable external ignored artifacts
`/home/hackathon5/liamreporoot/outputs/pipeline-20261028/FINAL-REPORT.md` and
`/home/hackathon5/liamreporoot/outputs/pipeline-20261028/PROGRESS.md`, plus the [WP4 matrix](wp4-fill-clips-matrix.md),
[WP6 comparison record](wp6-model-comparison.md), [evaluation limits](evaluation.md), and
[physics limits](absorbance.md).
