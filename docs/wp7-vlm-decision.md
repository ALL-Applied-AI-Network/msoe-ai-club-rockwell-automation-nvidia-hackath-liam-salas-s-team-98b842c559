# WP7 image-grounded VLM decision boundary

Status: response contract v2, its CPU guard, independent one-model attempt path and synthetic
tests are implemented in this branch. No v13 development freeze has been reviewed, no v2 GPU
attempt has run, and no model-quality or physical-validity result follows from the CPU work.

`fluidmix.explanation.vlm.decision` judges only vial 1 in a complete three-vial cohort. The
prompt describes vials 2 and 3 as later same-recipe references after vial 1 flushed the line.
That is a comparison assumption, not measured proof that either reference is physically clean.
The bundle supplies early, mid and settled end images, camera-derived Lab measurements, recipe
context and the deterministic rule result. It supplies no Beer-Lambert expectation, computed
clean colour, hidden simulator state, oracle mask, residue amount or prescribed label.

## Response contract v2

The model reasons from the images and its physical knowledge instead of copying deterministic
labels. Its ordered JSON contains one `stages` row per bundle stage, then `disposition`,
`confidence`, `agrees_with_deterministic`, `why_agree_or_disagree`, `physics_used`,
`explanation`, `evidence_citations` and `image_citations`. Each stage reports what the
model sees, its physical reasoning, the vial-1/reference comparison, image quality and optional
stage-local image citations.

The three long reasoning fields are limited to 600 characters each and the operator explanation
to 400 characters. The parser retains its 65,536-byte response ceiling. Citation arrays may be
empty; when an ID is supplied, it must exist in the bundle, and a stage citation must belong to
that stage. The guard requires exactly one stage row for every bundle stage in bundle order.

The deterministic rule is a separate, equal-weight source, not an independent measurement. A VLM
may agree, partly agree or disagree with it. The guard does not require copied relations, copied
DeltaE values, agreement, threshold recomputation or confidence caps.

The factual guard excludes the whole model source, with no score and no replacement explanation,
when it finds invalid structure, an unsupported ID, nonfinite or out-of-range confidence, an
overlength field, hidden simulator state, a quantified residue claim, an operator instruction,
an asserted physical cause, or an explicit prose disposition that conflicts with the structured
one. Qualified hypotheses such as “consistent with stratification” or “could reflect carryover”
are allowed. Numeric prose checking is deliberately narrow: explicit DeltaE and confidence claims
must match the bundle/response; other numbers are not checked. Raw output and guard reasons remain
available for review.

After admission, discard contributes `confidence`, keep contributes `1 - confidence`, and
`insufficient_evidence` contributes no vote. The deterministic source contributes 1 for discard,
0 for keep and no vote for insufficient evidence. Sources have equal arithmetic weight, at least
two numeric votes are required, and a mean discard score of exactly 0.5 is discard. These sources
are correlated because they use overlapping images and Lab evidence.

## Inputs, provenance and frozen artifacts

The component accepts exactly three `wp7-camera-fill-step-receipt-v1` artifacts through the
public tracker validation boundary. Each stage binds early/mid/end image identities, tank and fill
step, stream Lab/spread, recipe amounts and camera-derived fill state. Local paths are verified
through a separate locator and never enter prompt text.

The bundle independently recomputes vial-1-to-reference CIE76 distances and validates the closed,
self-hashed deterministic receipt against the frozen 1.0/2.0 policy. This validation protects the
separate deterministic vote; it no longer dictates VLM output fields.

Breaking artifacts are versioned rather than rewritten:

- development freeze: `wp7-vlm-development-freeze-v13`;
- one-model attempt: `wp7-vlm-model-attempt-receipt-v3`;
- aggregation receipt/publication: v2;
- prompt evidence payload: v5, unchanged because its data shape is unchanged.

A v13 freeze binds the exact decision source, prompts, response schema, parser, guard, aggregation
constants, pinned models, tracker inputs, materialization receipt and generation configuration.
Earlier freezes and attempts are immutable historical evidence and must not be edited into the new
versions.

## Model execution

Qwen3-VL-8B constrains the whole response to the v2 JSON schema. Cosmos-Reason2-8B and
AnomalyThink-Qwen3-VL-8B keep free `<think>` text and constrain only the JSON inside
`<answer>`. All use the existing pinned revisions, local-only stock runtime,
`trust_remote_code=False`, deterministic settings and the 10,000-token request.

Each model attempt is operationally independent. Cosmos and AnomalyThink no longer require a Qwen
receipt or a Qwen guard pass. The one-model command is:

    python3 cli/fluidmix/explanation/vlm/run_wp7_vlm_model.py ...

It verifies the reviewed freeze before opening runtime inputs and publishes one no-clobber v3
attempt receipt. CPU aggregation accepts zero to three unique pinned attempt receipts:

    python3 cli/fluidmix/explanation/vlm/aggregate_wp7_vlm_attempts.py ...

Malformed and guard-failed responses remain recorded but do not vote. The runner service maps the
raw v2 `confidence` field to its existing HMI event contract and emits explanations only for
admitted model-written sources. There is no deterministic fallback presented as model output.

Constrained decoding uses `xgrammar==0.2.8`. A tokenizer/grammar compile probe is CPU-compatible,
but schema compatibility is not evidence that any model will finish generation or pass the guard.

## Verification boundary and next gate

CPU tests can establish schema closure, receipt hashing, stage coverage, guard behavior,
aggregation arithmetic and service-event compatibility. They cannot establish camera accuracy,
carryover detection, causal accuracy, model quality, sim-to-real transfer or plant readiness.

The next gated step is to review the code and a newly generated v13 V3/V4 development freeze.
Only after separate compute authorization should the three pinned models run independently on the
same frozen cases. Record parse status, guard admission and reasons, disposition/confidence,
deterministic agreement, aggregation, latency, memory and human review of the physics text. Any
prompt, schema, guard or runtime change after observation requires a new code identity and freeze;
failed attempts remain evidence and are never overwritten.
