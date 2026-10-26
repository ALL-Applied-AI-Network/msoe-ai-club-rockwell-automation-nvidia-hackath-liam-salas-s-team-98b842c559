# F2 composition-substitution cohort contract

Status: **CPU contract implemented; unrendered and not training eligible**. The plan and
per-vial schedules are deterministic inputs for a future Isaac runner integration. They are
not render receipts or evidence that the cases are visually separable.

## Why this replaces the residual-slug pilot

Source-file observation: the earlier residual adapter starts with extra particles and removes
the same count only from the final T100 pour. For the train recipe S0117, a 277-particle case
therefore reaches 1,758 particles after T200 and 2,684 after T400. Those counts represent
47.466 and 72.468 mL, not the programmed 40 and 65 mL, so both intermediate stages would miss
the programmed-height gate by more than 6 mm. That adapter also freezes 60 Hz timing and has
no binding in `scripts/physx_three_pour_sequence.py`. It cannot be used for the accepted
120 Hz cohort merely by changing metadata.

The replacement keeps the clean particle count at every stage. A vial-specific synthetic
parameter `alpha` changes only the tracked particle composition:

`(1-alpha) * current-tank one-hot + alpha * previous-recipe final fraction`.

The previous S0437 recipe fraction is `[0.125, 0.125, 0.75]` in T200/T400/T100 order.
`alpha` is an uncalibrated render oracle, not a measured residue volume or physical fraction.
It is excluded from model and deploy inputs. The render-only falling stream retains the
current tank's source colour.

## Frozen train-only cohort

The exact train transition is S0437 to S0117. There are two separate clean
`reference_only` batches and four `scored_target` batches, each containing three vials:

| target profile | alpha by vial | keep/discard labels |
|---|---:|---|
| clean | 0 / 0 / 0 | keep / keep / keep |
| first only | .10 / 0 / 0 | discard / keep / keep |
| first two | .10 / .05 / 0 | discard / discard / keep |
| all three | .10 / .05 / .025 | discard / discard / discard |

The 18 planned videos contain 54 stage captures. Only 12 target vials are scored: six keep
and six discard. Two distinct reference batch IDs exist for every target stage, but observed
reference validity remains `0/36` until the images are rendered and measured. Planned
availability is not a successful reference measurement.

The two reference batches use disjoint train-range seed triplets 330001–330003 and
330004–330006. All four target profiles reuse the same per-vial vector
330101/330102/330103, with one nuisance-group ID per vial position. These are matched
counterfactuals: profile differences are not confounded with the PBD seed, and the four
target batches are not independent samples. The complete matched groups stay in train. No
validation or test family is included. Recipe, transition and split files remain hash-bound.
Batch IDs are opaque `BATCH_<12 hex>` values; profile names occur only in separated label
provenance, never in model input.

## Timing and volume acceptance

The schedule uses 120 Hz physics and 10 Hz capture. T200, T400 and T100 run back to back for
2,704, 1,556 and 1,072 steps. Stage captures are steps 2,704, 4,260 and 5,368; the final
capture is 0.3 seconds after fill end. There is no initial residual slug and particle IDs are
contiguous from 0 through 2,776.

| stage | programmed mL | required count | programmed height | count-derived height |
|---|---:|---:|---:|---:|
| after T200 | 40 | 1,481 | 76.489 mm | 76.464 mm |
| after T400 | 65 | 2,407 | 124.295 mm | 124.273 mm |
| final +0.3 s | 75 | 2,777 | 143.417 mm | 143.377 mm |

Heights use the frozen 522.951416 mm² dodecagon area and a +/-6 mm gate. Every rendered
stage must additionally prove emitted/active/inside counts, zero in-flight/outside/escaped
particles, finite z min/Q05/Q95/max, particle bulk height, semantic visible height, body
volume closure, and the top-level RGB check against the visible inner bottom. The collider
floor top and copied liquid bottom are geometrically aligned, but opaque-base visibility
still requires the actual RGB crop.

## Exact runner handoff

`scripts/build_physx_composition_cohort.py --output-dir <new-external-directory>` writes
exactly three files: `render-schedule-plan.json`, `model-input-plan.json`, and
`label-provenance.json`. It refuses an existing directory. The model-input artifact is a
recursive allowlist and contains no alpha, profile, disposition, label, oracle or render
seed. The renderer parameters and scoring provenance are never emitted as one consumer
bundle.

Rendering remains fail-closed for two source-file reasons. The Isaac runner does not yet
accept the schedule, and it hard-codes `physxParticle:maxParticles=2221`, while every new
schedule requires 2,777 particles. Merely generating a schedule does not change that USD
capacity.

The next implementation must add both arguments below to
`scripts/physx_three_pour_sequence.py` and reject any mismatch:

```text
--execution-schedule <json>
--execution-schedule-sha256 <lowercase_sha256>
```

The runner must bind the schedule hash into its receipt and verify source-plan hashes,
120 Hz timing, event particle IDs/counts/compositions, all three programmed-volume gates,
and the accepted A34 full-run receipt. It must author `maxParticles` from the validated
schedule's `totals.maximum_particle_count`, read it back at no less than 2,777, and bind the
required, authored and read-back values plus the capacity decision into the receipt. Until
all checks exist, the render plan reports both `runner_schedule_ingest_not_implemented` and
`runner_particle_capacity_not_schedule_driven`, with `render_authorized=false` and
`training_eligible=false`.

## Verification and limits

Focused CPU tests cover exact denominators, split/seed discipline, timings, stage volumes,
all 18 schedules, contiguous IDs, composition sums, matched target seeds, opaque IDs,
recursive oracle rejection, capacity fail-closure, hash binding, tamper rejection and CLI
overwrite refusal. These synthetic checks do not validate PBD as
Navier-Stokes, mixing rate, liquid colour, carryover physics, real defects or causes, or
sim-to-real transfer.
