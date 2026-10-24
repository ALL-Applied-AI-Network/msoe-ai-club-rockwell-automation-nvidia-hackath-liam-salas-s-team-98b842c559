# Cosmos contract v2 result

Status: **result shell written before inference; the frozen model matrix has not run**.

V2 is a transparent post-v1 second look selected after v1's negative strict-contract result. Its
only inference-behavior change is the pinned template variable `enable_thinking=false`; it is not
a preregistered confirmation of v1, a repair of v1, or a reasoning ablation. The processor-only
input audit is reported in the design document and is not model-output evidence.

## Frozen provenance and integrity

Fill these fields only from the frozen run and tracked score. Do not copy raw response text into
Git.

| Field | Frozen result |
|---|---|
| Freeze commit/tag | NOT RUN |
| Raw artifact SHA-256 | NOT RUN |
| Score artifact SHA-256 | NOT RUN |
| Matrix integrity | NOT RUN |
| Inference errors | NOT RUN |
| Prompt-pair integrity | NOT RUN |

## Fixed-denominator outcome

| Measure | Frozen denominator | Result |
|---|---:|---:|
| Matrix attempts | 24 | NOT RUN |
| Per-mode attempts/completions/errors | 6 per mode | NOT RUN |
| Strict unwrapped schema | 24 | NOT RUN |
| Schema-dependent policy counters | 24 each | NOT RUN |
| Required abstention | 15 | NOT RUN |
| Schema-gated image invariance | 6 pairs | NOT RUN |
| Deterministic template control | 6 | NOT RUN |
| Generations reaching the 384-token cap | 18 model attempts | NOT RUN |

If either member of an image/text pair is invalid, image invariance is **not established** for
that pair. Do not report a failed schema-gated pair as an image effect.

## Descriptive v1/v2 comparison

After scoring, state only whether direct schema completion, completion/error counts, and cap
reaches changed under the same 24-attempt matrix and 384-token ceiling after the one template
switch. A difference may motivate a hypothesis that the template condition interacted with direct
completion, but cannot isolate reasoning, formatting, or token budget as the cause. A tie or
negative result is retained without changing the prompt, cap, parser, controls, or cases.

## Claim boundary and stopping decision

This experiment cannot establish defect detection, carryover, wrong-volume detection, causal
attribution, future-vial effect, image understanding, operator usefulness, safe plant action,
physical validity, camera robustness, or sim-to-real transfer. A strict pass would establish only
that the raw completion matched this synthetic evidence-to-explanation interface.

Stop after the complete v2 result. Preserve malformed, capped, or errored attempts in their fixed
denominators. Do not retry, extract or repair JSON, select outputs, reprompt, add stop strings,
change the token cap, or rerun a subset. Any later generation change is a new branch and separately
frozen experiment.
