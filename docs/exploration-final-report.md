# Final exploration report — 2026-10-25

## Outcome

The mission is **not achieved**. The repository does not contain a deployable camera detector, a
validated carryover/cause estimator, or an operator explanation service. It contains useful
provenance checks, synthetic contracts, a local Cosmos runtime smoke, and several carefully
preserved negative results. For any future integration, the required safe policy is to reject
unsupported evidence and emit `insufficient_evidence`; this repository does not currently
implement an end-to-end output path. It cannot yet tell an operator which defect occurred or what
caused it.

This report is an evidence synthesis, not a new experiment. Detailed commands, branches, hashes,
review findings and process deviations remain in the [exploration log](agent-brief.md). Immutable
release reference: `explore/final-synthesis-20261025`, based on main
`40d31b974786c28cdb151e35b86804d3f3cf6a25`.

## Evidence at the stopping point

| Stage | Strongest retained evidence | Decision and limit |
|---|---|---|
| Input identity | Repeated checks matched all 4,688 manifest files by path, size and SHA-256. | The team is reading the expected bytes. This says nothing about detector accuracy or physical validity. |
| Timing and windows | A regenerated check found 31/31 informative still matches within +/-2 frames, with median absolute offset 1 frame; none came from `vial_fill` video 5. The committed final-mixture windows were byte-current for 28/30 vials under the recorded tools. | Useful provenance only. Window boundaries remain heuristic and are not physical ground truth. |
| Real colour signal | The brief reports a post-hoc video-5 audit: first-to-last-vial median dE76 29.3 versus vial-2/vial-3 median 5.1 at nine changeovers, p=0.001, AUC 0.91, and shift toward the previous recipe in 9/9. | Association only. The ROI was chosen after inspection, the exact producing commit is not retained here, video 5 has no defect/cause labels, and run 6 contradicts the assumption that vial 2 is clean. |
| Camera measurement | The frozen feed-reference contract passed 26 named constructed-array/metadata checks; six constructed current events were assigned once across six coverage/abstention buckets. | A synthetic reproducibility record, not 26 observations or detector trials. There is no deployed ROI, pixel extractor, reference selector, real threshold, or verified clean reference. See tag `explore/feed-reference-contract-frozen-20261024`. |
| Detector | The three-vial run-relative route is structurally ambiguous; the stored-reference route cannot verify that its reference is clean; the saturation-extrema selector fragmented on the inspected windows. | No detector exists. No false-alarm, miss, recall, accuracy, or real coverage claim is supported. |
| Cause evidence | The historical Beer-Lambert fit reproduces negative absorbance on real video 5. Recipe targets and order are available, but actual dispensed volume and retained path volume are not measured. | No physical residue fraction or cause estimator exists. Recipe context can describe a hypothesis, not prove carryover. |
| Cosmos | A pinned local Cosmos3 Edge path produced a non-empty completion for one synthetic FillCam preview. In two frozen strict-contract runs, deterministic controls passed 6/6 while raw model outputs passed strict schema 0/18 in each run; v1 reached the token cap 17/18 and v2 2/18. | Local image-input code-path compatibility is established narrowly. Visual dependence, explanation quality, safe operator advice and mechanism behind the cap difference are not established. See frozen commits `71bf5c9...` and `b3411ce...`. |
| Twin rendering | In the earlier smoke, the original process failed before pixels and an infrastructure retry produced a decodable, uniformly black 640x480 image. A separate frozen recovery used one renderer process, captured RGB shape `(0,)`, and wrote no image; an outer replay was blocked before a second spawn. | Both are failed feasibility results. Root cause, renderer reliability, liquid appearance and sim-to-real transfer remain unknown. See tags `explore/brev-fillcam-black-stopped-20261024` and `explore/fillcam-recovery-frozen-20261024`. |
| Integration | No implemented camera estimator, threshold service, cause stage, training pipeline or Cosmos client connects the allowed inputs to an operator record. | No end-to-end run or deployment claim is available. |

## What worked

- Data and source provenance were made reproducible. The cited frozen commits and tags preserve
  the retained failed-work evidence; their hashes are recorded in the exploration log.
- Strict validators, hidden-oracle scoring, explicit denominators, no-overwrite protections and
  abstention buckets exposed unsupported records instead of silently repairing them.
- The local GB10 could execute one pinned Cosmos image-input code path. This is useful runtime
  feasibility evidence, even though it is not a quality result.
- The feed-reference scaffold now rejects forged descriptors and denominator relabeling. It is a
  reusable prerequisite if real ROI/event/reference provenance becomes available.
- Brev work stayed bounded and stopped. The last recorded lifecycle was 1,071 seconds at a quoted
  $3.605088/hour, a $1.0725 upper-bound estimate rather than an invoice.

## What failed and was stopped

- Camera-RGB Beer-Lambert calibration is physically invalid on the available feed.
- Three-vial run-relative scoring cannot identify which of two disagreeing vials is wrong and can
  hide two matching faults; colour alone also misses total-volume errors that preserve ratios.
- Hand-edited preview polygons produced visible seams and holes, so they are not training data.
- The existing saturation-extrema predicate is not a usable liquid mask on the inspected slice.
- Both bounded Isaac/FillCam branches failed to produce usable pixels.
- Cosmos direct generation did not satisfy the frozen strict record contract in either look.
- A stored feed reference cannot be promoted to a clean target from schedule order alone. The
  available assets lack the real event mapping, ROI provenance, appearance continuity and labels
  needed to evaluate that detector.

These failures are not patched into positive claims. Their useful contracts and safety checks are
retained; their detector, physical and sim-to-real interpretations are not.

## Narrowest honest pipeline

1. Accept only `vial_fill`, `tank_level`, `cap_seal`, and programmed recipe mL/order/sequence.
2. Validate source identity, event provenance and image quality. The repository has pieces of this
   gate, but not a deployed three-feed event mapper or validated ROI registry.
3. Compute a camera measurement only when a frozen real ROI, disjoint reference/current frames and
   quality checks are valid. That real measurement path is currently missing.
4. A future integrated system may emit defect or cause evidence only after an independently frozen
   detector is evaluated against labels. No such detector or real defect/cause labels exist; any
   current integration would therefore have to abstain, but none is implemented here.
5. Let Cosmos explain only supplied quantitative evidence, never detect or invent it. Until the
   raw record contract and explanation criteria pass, use at most a deterministic conservative
   template for internal testing—not operator guidance.

The retained design is therefore a guarded scaffold whose future interface would require
abstention, not a working system.

## Three next experiments

### 1. Feed-only observability and nuisance stress test

Freeze one candidate descriptor and exercise it locally on constructed fixed-view arrays under
common/per-channel gain, gamma, blur, clipping, compression, ROI drift, dirt/occlusion, near-black
values and contaminated references. Include pairs with identical observations but different
latent causes. Stop the descriptor if it emits a numeric comparison for invalid or observationally
ambiguous cases. This cheap CPU test can falsify method classes; a pass still cannot validate a
real ROI, carryover or camera transfer. Brev is unnecessary.

### 2. Frozen cross-feed event and quality audit

Develop the event-ledger schema, fixed geometry and quality buckets without video-5 outcome
tuning, freeze them, then apply them once to the three real feeds plus recipe schedule. Report only
one-to-one mapping coverage, disjointness, missing/ambiguous events and mutually exclusive
abstention counts. Stop if evidence overlaps, mapping is ambiguous, or invalid frames enter the
comparison denominator. This can establish deployment coverage, not defect recall or cause
accuracy. Run locally; do not use Brev.

### 3. Separate fixed-view renderer capture-path smoke

On a fresh branch, use a materially different documented capture path rather than patching either
stopped renderer. Freeze one unchanged scene/view and require a nonblack RGB image plus a semantic
liquid mask before any material variant, carryover scenario or batch. Prefer local execution; if a
compatible local renderer is unavailable, bound one recommended Brev instance to one short
smoke-and-stop window. Failure to produce both outputs stops the route. Success would establish
only render/label feasibility, not liquid realism, robustness or sim-to-real transfer.

## Human review

The concrete morning actions are kept separately in
[the 2026-10-25 review note](morning-review/2026-10-25.md). The immediate operational action is to
verify and delete the retained Brev instance as planned. Do not present any branch as a working
defect detector or restart a stopped approach without a new frozen design.
