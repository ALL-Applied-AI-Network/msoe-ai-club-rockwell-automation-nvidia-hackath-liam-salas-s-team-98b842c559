# Feed-derived reference contract

Status: **stopped as a detector; retained synthetic contract only**. This is not a detector,
clean-colour source, carryover estimate or deployed camera pipeline.

## Problem and decision

The deployed system has only the three fixed camera feeds and programmed recipe data. A prior
vial can therefore be stored as an appearance reference, but no allowed input proves that vial
is clean or that tank contents remained unchanged. Schedule timing also does not by itself prove
which observed vial belongs to a transition.

This bounded scaffold will answer only: given explicit linear-RGB region arrays and a versioned
event ledger, is a descriptive comparison available, and what feed-derived appearance distance
does it produce? It will preserve missing, uncertain and invalid evidence. It will not select a
reference from colour, set a threshold, emit a defect flag or infer a cause.

Three implementation shapes were considered:

1. Wrapping the existing absorbance functions was rejected because that would inherit an
   uncalibrated camera-transmission model and does not solve reference provenance.
2. An append-only JSONL service was rejected because persistence and deployment I/O are outside
   this contract and would obscure the numerical and metadata boundary being tested.
3. Chosen: immutable records plus pure measurement, ledger-validation and comparison functions
   in a new module. Callers own storage and reference selection.

## Measurement model

For channel `c` in R/G/B, let `L_c` be the median of a named liquid ROI and `P_c` the median of a
named same-frame background patch, both in nominal linear RGB on `[0, 1]`. The descriptor is

```text
z_c = ln((L_c + epsilon) / (P_c + epsilon)).
```

Version `feed-reference-contract-v1` fixes `epsilon=1e-6`, at least 16 pixels in each region,
underexposure at a channel median `<=1e-4`, saturation at pixel values `>=0.99` with more than 5%
of either region saturated, and patch spatial median absolute deviation above `0.08` in any
channel as uncertain. These values define synthetic fixtures only; they are not camera settings
or plant thresholds and must not be tuned on video 5.

Malformed shape, nonfinite values or values outside `[0, 1]` are invalid. Quality-limit failures
are uncertain. Neither state yields a descriptor for downstream comparison. The record retains
frame/ROI/patch IDs, areas, per-channel medians, saturation fractions and patch deviation. It
also retains the no-patch descriptive baseline `L / sum(L)`; that baseline is not a correction or
detector.

The descriptor cancels a common multiplicative scalar or per-channel gain only when the same
gain acts on both regions, neither region clips, and `epsilon` is negligible. It is not invariant
to different regional illumination, nonlinear tone mapping, focus/blur mixing, occlusion, dirt,
clipping or compression. Synthetic checks must expose those limits rather than label their
result a defect.

## Event and reference records

Each event carries explicit record, schedule, sequence, transition, vial, frame/window, timing,
phase and appearance-domain IDs. Scheduled and observed positions must agree and the supplied
ledger must be unique, monotone and contiguous. Timing uncertainty must be finite and no larger
than its predeclared tolerance. Only `final_mixture` phase is comparable.

A `feed_derived_reference_candidate` points to one prior event and measurement and records its
selection-rule version. It is always marked `cleanliness_unverified`; the code has no state named
`clean_reference`. Reference and current frames must be disjoint, while using the same fixed ROI
and patch IDs is required and expected. A comparison is unavailable on a missing prior,
ambiguous/off-by-one/missing/extra event mapping, invalid phase, unknown or changed appearance
domain, invalid measurement, mismatched geometry or overlapping frame evidence.

When all structural checks pass, the output contains Euclidean distance in natural-log descriptor
units plus normalized-RGB baseline distance. Its validity remains `uncertain` with reason
`reference_cleanliness_unverified`. It contains no threshold, decision or cause. Coverage is
reported with one primary bucket per requested current event: first occurrence, reference
unavailable, reference stale/invalid, current invalid, event mapping invalid, or compared repeat.

## Acceptance checks and falsifiers

Synthetic tests must cover common scalar and per-channel gains, liquid-only change, independent
liquid/patch gain, shading, gamma, focus/blur mixing, occlusion/dirt, clipping, quantization,
near-black and nonfinite inputs. They must also show both contaminated-reference failure modes:
a shifted reference disagrees with a later unshifted sample, while the same shifted reference can
mask a later matching shift.

Ledger tests cover first occurrence, no prior, repeated recipe, off-by-one, missing and extra vial,
pause/clock uncertainty, stale or unknown state, invalid phase, frame overlap and an invalid
current measurement. A deterministic script will report every case and the mutually exclusive
coverage denominator; it will use no video or image data.

Stop and salvage only record validation if common-gain behavior fails under the exact stated
model, contaminated-reference masking is hidden, invalid evidence receives a numeric comparison,
or ledger/reference exclusion cannot be enforced without an outcome-derived choice. Passing
these checks proves only the declared synthetic contract. It cannot establish ROI correctness,
camera robustness, clean colour, defect/carryover detection, physical validity or sim-to-real
transfer.

Implemented files are `src/fluidmix/feed_reference.py`, `tests/test_feed_reference.py`, and
`scripts/check_feed_reference_contract.py`. Verification is the focused test file, full test
suite, synthetic absorbance smoke, contract script and documentation checker.
