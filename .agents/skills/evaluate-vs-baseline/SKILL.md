---
name: evaluate-vs-baseline
description: Score a real-data detector or report team result numbers against matched baselines.
---
# Evaluate against baselines

## Inputs
Corrected vial IDs, independent labels, frozen partitions/thresholds, method outputs
and baseline outputs. Read [evaluation](../../../docs/evaluation.md).

## Workflow
1. Audit timing, eligible transitions and partition overlap.
2. Score the method, first-vial rule, vision-only colour check and drainage-only tank model on
   identical eligible vials.
3. Report false alarms, misses, counts, exclusions and provenance.
   Cause/fraction accuracy requires independent cause/residue truth.

## Output and validation
Matched-vial comparison with split, model, calibration and threshold IDs.
Check that the held-out recipe was excluded from calibration and threshold tuning.

## Limits
Teledyne is not a baseline (docs/evaluation.md); rough color differences are not defect truth.
The fit script reports prediction error, not detector performance. No scoring CLI exists.
