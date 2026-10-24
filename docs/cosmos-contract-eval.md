# Cosmos explanation-contract experiment v1

Status: **design accepted with corrections; no model evaluation has run**.

## Question and claim boundary

Can the pinned local Cosmos3 Edge checkpoint return a strict, evidence-bounded record when it is
given camera-pipeline facts and recipe context, and does attaching one irrelevant but valid fixed
FillCam twin image change that record?

This is a contract-compliance and image-distraction test. It cannot measure defect detection,
cause accuracy, image understanding, operator usefulness, physical validity or sim-to-real
transfer. The deterministic template is a policy control, not a competing model baseline.

## Allowed inputs and policy

Fixture inputs use only fields the deployed system could have:

- `vial_fill` appearance-comparison values and quality state, with references identified as coming
  from that feed;
- current and previous control-panel recipe tank mL and dispense order; and
- explicit unavailable values. No label, scale, flow/level sensor, configured tank colour,
  Teledyne output, lab measurement or fitted absorbance is present.

The fixtures are neutral `synthetic_contract` records, not twin-test or real-validation records.
Their appearance numbers are interface examples, not camera measurements or physical claims.
Even a colour shift toward a previous-recipe feed reference is association only. Therefore every
response must keep `defect`, `cause` and `next_vial_effect` as `unknown`, and confidence as
`not_assigned`.

| Evidence state | Allowed appearance statement | Required advisory state |
|---|---|---|
| Valid comparison above its supplied threshold | `difference_observed` | `manual_review_required` |
| Valid comparison at or below its supplied threshold | `no_difference_observed` | `no_automatic_action` |
| Uncertain/invalid camera evidence, missing previous recipe or omitted numbers | `unknown` | `manual_review_required` |
| Image plus recipe with no quantitative evidence | `unknown` | `manual_review_required` |

The output cites only supplied evidence IDs and reproduces numeric strings, units and sources
exactly. Missing values stay null/absent; they never become zero. Unsupported causal or future
claims fail the contract.

## Frozen matrix and scoring

Use six neutrally named fixtures: two valid differences (one toward and one not toward the
previous reference), one valid no-difference comparison, one uncertain-quality comparison, one
invalid-camera record and one missing-previous-recipe record. Expected constraints live in a
separately hashed oracle that the inference runner never reads.

Run exactly four conditions once per fixture (`F = 6`, `4F = 24` planned attempts):

1. the full evidence/recipe text plus the canonical 640 x 480 synthetic FillCam preview;
2. byte-identical full evidence/recipe text without an image;
3. recipe text plus the image, with all quantitative camera evidence omitted; and
4. the deterministic policy template.

The image is the same hashed preview in every image attempt and is declared irrelevant to the
fixture. Conditions 1 and 2 are a paired distraction test; condition 3 must abstain and is not a
visual-performance test. The template should pass by construction and is reported separately.

Report these fixed denominators:

- schema, evidence/numeric fidelity, advisory safety and unsupported-claim policy: all 24 attempts;
- completion/error counts: six attempts per condition;
- required abstention: the six image-only attempts plus all full-input conditions for the three
  insufficient/invalid fixtures (15 attempts total);
- paired image invariance: six fixture pairs; and
- template policy control: six attempts.

Invalid JSON, duplicate/unknown fields or IDs, numeric changes, fenced/repairable JSON and
inference errors fail. Generate and preserve all responses before running the separate scorer.
There is no retry, repair, reprompt, output selection or human scoring.

## Freeze and execution

Before inference, one clean commit and tag must freeze the JSON contract, canonicalization,
fixtures, sealed oracle, prompt bytes, image hash, template, scorer, model revision and selected
weight manifest, runtime lock, greedy decoding/seed/token limit, mode matrix and output path. The
runner refuses a dirty tree or a tag that does not point to `HEAD`.

Pinned model candidate: `nvidia/Cosmos3-Edge` revision
`344d602b128d1bbdacb43b08d0a3626f46343e29`, using the already verified local snapshot. Canonical
image SHA-256: `c4886724246a6dbef76c6658b07c8568c5e8e4372f7a9a623021a3ddeb3997af`.

Stop before inference if labels/answer words enter runner-visible fixture data, the oracle can be
read by the runner, image and no-image text differs, outputs can be repaired/retried, or the full
pin set cannot be verified. A failed or incomplete frozen run is retained as the result; any
change is a separately tagged second look.

The fresh-context reviewer returned `PROCEED-WITH-CORRECTIONS`. It removed all defect, carryover,
wrong-volume, future-effect and plant-action targets, required the sealed oracle and paired image
invariance denominator, and narrowed the template to a reference control.
