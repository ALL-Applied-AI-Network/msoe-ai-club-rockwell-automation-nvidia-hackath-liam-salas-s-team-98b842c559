# INN-32 feasibility protocol: draft, not frozen

Prepared 2026-10-16 UTC against main e4146b10671c38ed866e2e77728f329fa978b0fb.
Status: **awaiting team agreement and upstream evidence**. No numeric targets or route
are approved. Historical video-5 errors and negative absorbance were already known when
this draft was written. This is not a retrospective preregistration of those results.

Source: [INN-32](https://linear.app/innovation-lab-2026/issue/INN-32), under
[INN-8](https://linear.app/innovation-lab-2026/issue/INN-8). Use this draft for the next
comparisons only after team review, a timestamped freeze and identification of unseen
evaluation inputs. Final held-out evaluation remains INN-17.

## Required freeze record

Before inspecting new comparative outcomes, save the agreed revision/commit and UTC time,
reviewer names, input hashes, eligibility rules, split IDs, measurement/ROI/reference
versions, model/calibration versions, threshold method and all tolerances below.
Record which historical results each reviewer already saw. If fresh unseen inputs are
unavailable, state that limitation; do not claim prospective validation.

| Criterion | Quantity to agree before comparison | Current value |
|---|---|---|
| Clean prediction | Error metric, units, aggregation, acceptable error and comparison uncertainty | UNSET |
| Repeatability | Repeated-measurement unit, matched conditions, variability metric and limit | UNSET |
| False alarms | Independent clean-label denominator, acceptable count/rate and uncertainty summary | UNSET |
| Misses | Independent defect-label denominator, acceptable count/rate and uncertainty summary | UNSET |
| Invalid coverage | Eligible/valid/uncertain/invalid counts, abstention treatment and acceptable coverage | UNSET |
| Useful improvement | Paired improvement required over simple color, including uncertainty and coverage tradeoffs | UNSET |

UNSET values block freezing and route selection. Do not fill them from observed method
performance. Changes after the freeze require a new revision, reason and fresh evaluation
plan; retain the original record and label prior outcomes exploratory.

## Evidence needed for each claim

| Claim | Required evidence | Cannot establish the claim |
|---|---|---|
| Appearance prediction | Paired predictions/errors on the same eligible held-out mixtures, fixed splits, versions and uncertainty | Low training error or a fit on all real vials |
| Physical absorbance | Matched clear reference, exposure/ROI/path checks, positive/identifiable calibration and repeatability audit | Synthetic recovery, clipped negative absorbance or a good color fit alone |
| Defect detection | Independent labels, frozen thresholds, matched eligibility, false alarms/misses and invalid coverage | Assuming first vials are defective or labeling from detector output |
| Residue/cause | Independent residue/cause truth and an identifiable fit with residual/uncertainty/alternative-cause checks | A small mixture residual or a detector flag alone |

INN-29 owns reference/ROI/exposure validity. Re-extraction waits for INN-7's verified
timing/mapping and partition work; do not duplicate it. INN-30 owns paired clean prediction:
keep equivalent mixtures 5 and 8 together and protect INN-17's final test. INN-31 owns
independent-label detection comparison. See [evaluation](evaluation.md) and
[physical limits](absorbance.md). Negative fitted absorbance, unsupported references,
rank deficiency or unstable exposure block physical adoption; clipping does not repair them.

## Comparison and result table

Run the absorbance and simple-color methods on the same eligible evidence. Freeze which
comparisons each invalidity blocks. Report the full eligible denominator, per-method
valid/uncertain/invalid counts, exclusions/reasons and a paired valid intersection; do not
hide unequal coverage by reporting only that intersection. Preserve per-vial source IDs,
measurement validity, predictions, independent labels, decisions and method versions.
Clean anomaly scoring and carryover attribution stay separate; see [interfaces](architecture.md).

For every table record input/manifest checksums, exact command, environment, code/config
version, output path/hash and reproduction result. No comparative runner exists yet.
Do not describe the inherited diagnostic as this protocol's implementation.

| Evidence | Source/result location | Status now |
|---|---|---|
| Measurement audit (INN-29) | Awaiting verified audit | Missing |
| Paired prediction (INN-30) | Awaiting grouped, held-out comparison | Missing |
| Paired detection (INN-31) | Awaiting independent labels and frozen thresholds | Missing |
| Team tolerance review | Names, date and accepted revision required | Missing |
| Independent table reproduction | Commands, environment and actual comparison required | Missing |

## Decision record to complete after review

Select one supported outcome: proceed with physically supported absorbance; retain simple
color; use a stated limited absorbance scope; fix measurements; or insufficient evidence.
Record the reason, evidence IDs, passed/failed/blocked criteria, excluded claims, smallest
missing evidence, team reviewers and approved implementation route. **No outcome is selected
in this draft.**

State the resulting permission or restriction for INN-13/14/25/27 individually. A no-go or
inconclusive result does not approve an unsupported absorbance pipeline. Route conditional
tank/CSI/live-hardware needs through INN-9; this document creates no visit or hardware job.
Compute access/budget remains INN-10. Label validation remains INN-25. Do not render or
train in bulk on the strength of this draft, available hardware or proposal GPU estimates.
