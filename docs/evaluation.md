# Evaluation and frozen partitions

No detector scoring pipeline is implemented. The fit script reports clean-vial color
prediction error only. Synthetic math checks are not real generalization evidence.

## Partition rules

Train learned models on twin synthetic data only.
Before real-data work, record a manifest with source video, recipe, transition, vial IDs,
corrected frame intervals, measurement/reference provenance and assigned role:
calibration, validation or test.

For each leave-one-recipe-out fold, exclude the test recipe from calibration and threshold
tuning. Choose separate calibration and validation groups among the remaining recipes;
fit tanks/I0 on calibration only, tune the threshold on validation only, then freeze both
before scoring the held-out test recipe. No vial or repeated frame of it may cross roles.
Keep adjacent transition sequences grouped to avoid leakage. Record any previous-recipe
context exposed to prediction. Report when limited recipes prevent an identifiable fit
or independent validation. Do not silently reuse test vials to make the fit work.

The full-data diagnostic fit in fit_absorbance_video5.py is not a deployable calibration;
its first-vial fits are not held-out evaluation. Clean-vial leave-one-recipe-out results
also lack a detector threshold/validation stage.

## Labels and denominators

Committed CSVs have 30 vials in 10 recipe groups. They expose nine transitions between
groups. Vial 1 has no recorded previous recipe. The inherited "10 changeovers, 8/10"
claim includes an initial batch and was based on rough color comparisons; it is not an
independently verified defect-label set. Define eligible transitions and labels before
reporting a detector score. See [data](data.md).

## Matched baseline comparison

| Baseline | Behavior | Current evidence |
|---|---|---|
| First-vial rule | Flag every eligible first vial after a known recipe change | Rule defined; independent defect scoring pending |
| Teledyne color check | Compare with plant color-check output on those same vials | Outputs unavailable; not measured |

Score method and both baselines on identical eligible vials with independent labels.
Report false alarms, misses, denominators and excluded/invalid measurements.
Cause accuracy requires verified cause labels; fraction error requires known residue truth.
Counts accompany percentages because the sample is small.
Freeze measurement method, calibration, thresholds, model/config IDs and partitions.
Use [timing notes](data.md). Bench experiments and CSI tests are proposed until run
and require the corresponding authorization.
