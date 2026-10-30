# WP7 deterministic-template and limited-result integration

Status: implemented against WP7 scaffold base
`7577373d4e30e0abefd442991f5db6db6b3df980`. Verification here is synthetic and CPU-only.
No frozen one-pass input or video5 input was opened or executed.

## Active policy

The deterministic guarded template is the official DP4/default for both the disposition and
its explanation. `freeze_validation` rejects any explanation method other than
`deterministic_guarded_template` or any model ID other than `none`. A missing valid history does
not select a model: the test path emits an explicit `insufficient_evidence` abstention and the
evaluation-only real-check path emits an explicit denominator-bound skip.

Normal terminal receipts include `wp7-explanation-policy-v1`. The metadata is bound through the
pipeline source hash and repeats neither predictions nor store rows. It does not change accepted
fill inputs, the predictor interface or the evidence-store contract.

## Completed limited diagnostic

The completed pre-access run contained 16 all-abstention attempts per method: eight ordered and
eight shuffled. It was a strict parsing, abstention and unsupported-cause-refusal diagnostic,
not a keep/discard evaluation.

| Method | Strict parse | Correct abstention | Unsupported-cause refusal | Parsed-pair flips | Runtime metadata |
|---|---:|---:|---:|---:|---|
| deterministic template | 16/16 | 16/16 | 16/16 | 0/8 abstention; 0/8 refusal | CPU template |
| Qwen3-VL-8B-Instruct | 16/16 | 16/16 | 16/16 | 0/8 abstention; 0/8 refusal | 5.589 s median; 18,824,709,120 B peak CUDA allocated |
| AnomalyThink-Qwen3-VL-8B-KCR | 0/16 | not comparable after parse failure | not comparable after parse failure | no parsed pairs | free-form thinking and yes/no; 15.214 s median |
| Cosmos-Reason2-8B | 0/16 | not comparable after parse failure | not comparable after parse failure | no parsed pairs | JSON fenced; no repair; 7.619 s median |

The receipt metadata sets `keep_discard_accuracy` and `winner` to null. All targets were already
abstentions, so these observations cannot support keep/discard accuracy, physical validity,
model adoption or a winner claim. The full comparison remains scheduled for Friday under
INN-67.

## Remaining boundary

This integration does not authorize a frozen test or real-check run. It does not establish valid
production history, an accepted RGB-only runtime, detector accuracy, explanation quality on
decisions, or sim-to-real transfer. Those evidence and execution gates remain unchanged.

## Synthetic verification

The initial focused test run failed twice: validation accepted a model-backed explanation, and
the normal receipt still reported explanation support as unavailable. After the minimal policy
change, the same two tests passed. The completed checks were:

- focused red/green: `2 passed, 24 deselected` after the fix;
- WP7 pipeline and route-preflight tests: `39 passed`;
- full synthetic/unit suite: `2781 passed, 14 skipped`;
- `scripts/smoke_absorbance.py`: `synthetic smoke passed`;
- `scripts/check_agent_docs.py`: 11 skills and 54 document targets passed;
- `git diff --check`: no errors.

These checks exercise schemas, receipts and synthetic bytes only. They do not validate camera
measurements, physical carryover, a model decision or a frozen production execution.
