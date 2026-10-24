# Run-relative colour score: stopped detector approach

Status: **stopped after adversarial decision review**, 2026-10-24. No implementation or detector
result exists. The incomplete draft proposal remains below as evidence of what was reviewed; its
exact fixtures, threshold grid and denominators were never frozen.
Base: `40d31b974786c28cdb151e35b86804d3f3cf6a25`. Video 5 is not used in this experiment.

## Review verdict

A fresh-context reviewer rejected adoption as an individual-vial detector. With exactly three
vials, one bad repeat vial makes both repeats equally distant from each other, and two matching bad
repeat vials make each other look normal. The score also cannot detect a total-volume error that
preserves mixture ratio and colour. These are missing-information failures, not threshold-tuning
problems for a method using only these three Lab vectors, so the detector approach was stopped
rather than patched.

The equations below may later be implemented only as property tests or a batch-consistency
diagnostic. They cannot support per-vial rejection, wrong-volume coverage, cause attribution,
real-time decisions, physical carryover estimates, sim-to-real validity or superiority over the
required matched baselines. Scaled MAD remains a candidate heuristic; it was not adopted.

## Purpose and boundary

The proposed score compares a finished vial with other vials from the same recipe run. It is an
offline colour-consistency check: it needs later run-mates, so it cannot decide whether the first
vial should be rejected in real time. CIELAB is used only as a camera-colour coordinate system;
this model does not estimate absorbance, residue volume or physical dye concentration.

## Equation

For recipe run `r`, let `x[r,i]` be vial `i`'s measured CIELAB vector and let position 1 be the
post-changeover vial. For vial `i`, form the reference index set

    J[r,i] = {j : position[r,j] >= 2 and j != i}.

The reference is the coordinate-wise median of those vectors,

    q[r,i,c] = median({x[r,j,c] : j in J[r,i]}),  c in {L*, a*, b*},

and the score is CIE76 distance,

    s[r,i] = sqrt(sum_c (x[r,i,c] - q[r,i,c])^2)  [Delta-E-76 units].

If `J[r,i]` is empty or a measurement is non-finite, the result abstains. The vial is flagged only
when `s[r,i] > tau(k)`; equality is not a flag.

Thresholds are fit only from independently labelled clean synthetic-validation scores
`S_clean,val`. Let

    m = median(S_clean,val)
    d = median({abs(s - m) : s in S_clean,val})
    tau(k) = m + k * 1.4826 * d,  k >= 0.

The factor 1.4826 makes MAD comparable with standard deviation for Gaussian noise. It does not
make the data Gaussian. Fewer than three finite calibration scores or `d = 0` makes the threshold
invalid; the implementation must fail rather than silently invent a noise floor. Scores from the
same run are correlated, so later uncertainty resampling must use runs, not individual vials.

## Assumptions and falsifiers

Assumptions:

1. A preregistered liquid region yields comparable Lab measurements within one run.
2. At least two post-changeover vials exist, and most post-changeover vials represent the recipe.
3. Illumination and exposure do not change differentially between vials within a run, or a
   separately measured drift control marks the run invalid.
4. Synthetic-validation labels come from render inputs, never from this score.

Limit checks fixed before coding:

- Adding the same Lab offset to every vial must leave every score unchanged.
- A first vial identical to its run-mates must score zero.
- In a three-vial run, positions 2 and 3 each use the other as their sole reference. A single
  position-2 fault therefore gives positions 2 and 3 identical scores; the method cannot identify
  which one is wrong.
- If positions 2 and 3 share the same fault, they define each other as normal and both score zero;
  the clean first vial can instead look anomalous.

The draft proposed treating those behaviors as implementation falsifiers and later challenging the
detector against clean changeovers, carryover and mid-run wrong-volume cases. Review rejected that
broader criterion: colour alone cannot cover general wrong-volume faults, and the three-vial score
cannot identify an individual repeat fault. No detector work proceeds under that criterion. A
future property test could still verify the algebra, but a twin pass would show behavior only under
the chosen renderer and fault mapping, not physical carryover accuracy.

## Draft Experiment 01: deterministic identifiability stress test

Hypotheses proposed before any run:

- H1: the score detects a shifted first vial and is invariant to uniform run-wide Lab drift.
- H2: with only three vials, it cannot uniquely locate one bad post-changeover vial and can miss
  two matching bad post-changeover vials.

Fixtures use simple one-axis Lab values with labels assigned directly by construction: clean run,
first-vial shift, uniform drift, one position-2 shift, and matching shifts in positions 2 and 3.
Expected values are hand-computed from the equations above. The first-vial rule is the behavioral
control on the same cases. We report false alarms, misses and abstentions as counts with
denominators; these deterministic fixtures are proofs of code behavior, not performance estimates,
so no confidence interval is appropriate.

Stopping condition: run each fixed case once, preserve any negative result, and do not change the
fixtures or threshold in response. No real images, committed rough colours, video-5 frames, Cosmos
model or plant recommendation are in scope.

## Rejected draft implementation acceptance criteria

These criteria are retained only to show the proposal that was stopped; they are not an active
implementation plan.

1. One public pure function scores a Lab run and represents abstention explicitly.
2. One public function computes the centred, scaled-MAD threshold and rejects invalid inputs or a
   zero MAD.
3. Tests are written first and independently assert the limit cases above, strict threshold
   behavior, and invalid input handling.
4. A script writes the experiment command, revision plus dirty state, fixed cases, raw scores and
   matched counts under `outputs/20261024-run-relative-contract/`.
5. `python -m pytest tests -q`, `scripts/smoke_absorbance.py` and
   `scripts/check_agent_docs.py` remain green.

Camera extraction, real-data scoring, bootstrap intervals, a drainage model and Cosmos integration
are explicit follow-on work, not hidden inside this change.
