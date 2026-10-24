# Cosmos contract v2 result

Status: **frozen v2 completed and stopped after a negative strict-contract result**.

V2 was an exploratory second look selected after v1's result. Its only inference-behavior change
was the pinned template variable `enable_thinking=false`; it was not a preregistered confirmation,
a repair of v1 or a reasoning ablation. The processor audit in the design document establishes
input control only.

## Frozen provenance and integrity

| Field | Frozen result |
|---|---|
| Freeze commit/tag | `b3411ce260604332ff89a32d319138dce5b83e99` / `explore/cosmos-contract-v2-frozen-20261024` |
| Raw artifact SHA-256 | `7b7f75dae01c7d5b404fd395bee0845ca307d6d7c798d0a2ab09fdc5854b8036` |
| Score artifact SHA-256 | `1c1ca3b8a161a0a2b33d9e21e104c4f5af6129ccad3e492b8d687411c356cf8c` |
| Input-audit SHA-256 | `b18e0830f7ee69847e14b1cf1882a5291e92db10525cf06bb6eb157b086f553e` |
| Matrix integrity | 24/24 attempted and completed |
| Inference errors | 0/24 |
| Prompt-pair integrity | 6/6 |

The raw artifact remains ignored at `outputs/20261024-cosmos-contract-v2/raw-results.json` and no
raw response text is tracked. Its exact raw-free score is tracked in
`data/cosmos_contract_result_v2.json`. A fresh reviewer independently recomputed the score from the
unrepaired raw artifact and sealed oracle and matched it exactly.

## Fixed-denominator outcome

| Measure | Frozen denominator | Result |
|---|---:|---:|
| Matrix attempts/completions/errors | 24 | 24 / 24 / 0 |
| Each of four modes | 6 per mode | 6 attempted / 6 completed / 0 errors |
| Strict unwrapped schema | 24 | 6/24 overall; **0/18 model** |
| Each schema-dependent policy counter | 24 | 6/24 overall; **0/18 model** |
| Required abstention | 15 | 3/15; the three passes are controls |
| Schema-gated image invariance | 6 pairs | 0/6; **not established** |
| Deterministic template control | 6 | 6/6, by construction |
| Generations reaching the 384-token cap | 18 model attempts | 2/18 |

All six strict/schema-dependent passes are deterministic `template_reference` controls. None of
the 18 Cosmos model responses passed the strict raw schema. The `unsupported_claim` counter is
also schema-gated, so its 6/24 result is not evidence that malformed model text was safe or unsafe.
Because neither member of any model image/text pair was schema-valid, `0/6` invariance is not six
image effects or a robustness rate. Prompt-pair integrity only establishes matched text bytes.

## Descriptive v1/v2 comparison

Under the same stated 24-attempt matrix and 384-token ceiling, both runs completed 24/24 attempts
with zero inference errors, and strict model schema completion remained 0/18. The tracked v1
result reports 17/18 model generations reaching the cap; the independently reproduced v2 count
was 2/18 after the template switch. This descriptive association does not isolate reasoning,
formatting, token budget or any other mechanism, and it did not produce a strict model pass.

## Review, claim boundary and stopping decision

Fresh reviewer `review_v2_result` returned `VALID-WITH-NARROWER-WORDING`. The prompt supplies the
closed schema and transformation policy, and the six fixtures are synthetic interface cases.
Therefore even a pass would not demonstrate independent detection, causal inference, visual
understanding or generalization.

This experiment provides no evidence of defect or carryover detection, wrong-volume detection,
cause attribution, future-vial effect, image understanding, explanation quality, operator
usefulness, safe plant action, physical validity, real-camera robustness or sim-to-real transfer.
V2 stops here without retry, JSON extraction/repair, output selection, reprompting, cap change or
subset rerun. Any later generation change is a new, separately frozen experiment.
