# WP6 model comparison and DP4 fallback record

**Decision date:** 2026-10-29

**Record base:** `e4d93e4c24c274ec3e99df220204e5f86443f777`

**Result:** no RCA-bench v1 baseline or model execution ran. DP4 uses the deterministic
guarded-template default as Liam's pre-approved fail-closed policy fallback. It is not an
empirical winner.

## Why the comparison did not run

RCA-bench v1 permits model-facing evidence only when every liquid mask is an accepted
RGB-derived prediction with `mask_source="predicted"`. The sole bounded RGB predictor was
rejected during independent review on an authorized real blank-vial development image: its
direct production API returned 5,647 liquid pixels, a 149-pixel height and a recipe-like
48.8889 mL value for the empty vial. The same mask at programmed 10, 50 and 70 mL showed that
the earlier recipe circularity was fixed, but blank-vial correctness was not.

That blocking finding ended the one-attempt predictor route before held-out evaluation. The
held-out inputs, evaluation media and video 5 remained unopened. Existing synthetic
measurement uses oracle masks and is therefore ineligible for RCA-bench model input; it was
not relabelled. No accepted `mask_source="predicted"` evidence exists.

Liam's already approved option 3 required this exact outcome if the predictor review or
held-out pass failed: do not refreeze the benchmark and do not make a second predictor or
evaluation attempt; retain the oracle-mask synthetic diagnostic, explain the real-check
skip, and select DP4's deterministic guarded template. The decision is recorded in the
progress log as INN-61 failure comment `3451c2b8-926b-46ca-970a-5c64f866132f` and INN-63
resolution comment `8504ea53-e07a-467d-a7f6-81872b1790d5`. Those Linear facts are cited
only as already recorded in `outputs/pipeline-20261028/PROGRESS.md`; this work did not read
or change Linear.

## Planned and executed denominators

The frozen protocol does not inherit the historical INN-51 denominator of 90 primary and
10 shuffled cases. It requires the actual eligible counts in a freeze receipt.

| Frozen protocol unit | Planned denominator | Executed denominator | Result |
|---|---:|---:|---|
| Development selection | One primary per eligible V3/V4 vial, with only eligible controls | Not materialized; 0 attempts | **NOT RUN** |
| Image-grounding qualification | Exactly 12 ordered/shuffled pairs across four runs, or 24 raw model attempts per qualifying model | 0/12 pairs; 0/24 attempts | **NOT RUN** |
| WP1 test | One primary per eligible WP1 item per method; exact count had to be sealed, with at least one required-abstention and one non-required-abstention primary | Plan not materialized; 0 attempts for every method | **NOT RUN** |
| Real check | One post-freeze RGB-only agreement check if eligible | 0 | **SKIPPED**: no accepted RGB-only mask path |

The input-builder remediation was still awaiting independent re-review, generated no real
WP1 plan, oracle, freeze receipt or model run, and could not overcome the missing predictor
provenance. Therefore an exact nonzero test denominator was never validly established.
Reporting a zero rate would be misleading: all quality rates below are **not measured**
because their denominators were never executed.

## Method-by-method execution record

Liam's final comparison set was the three candidates AnomalyThink, AD-Copilot and
Cosmos-Reason2, plus deterministic-template and Qwen base controls. Cosmos3-Edge was
removed from this round.

| Method | Exact identity | Role | Execution and no-run reason |
|---|---|---|---|
| Deterministic template | Frozen RCA-bench template sentinel and source bindings | Baseline (a); final DP4 fallback | **NOT RUN (0 attempts).** Even baseline (a) required the absent accepted public plan, locators, provenance and freeze receipt. DP4 selection is a policy fallback, not a scored baseline win. |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Baseline (b): base VLM + deterministic guard + template fallback | **NOT RUN (0 attempts).** Exact weights were acquired, but accepted predicted-mask inputs and a sealed executable plan did not exist. No processor/model load occurred. |
| AnomalyThink-Qwen3-VL-8B-KCR | `aacudad/AnomalyThink-Qwen3-VL-8B-KCR@8331bc79ecf01e3466f0e6a27cec8c77f182f430` | Requested post-trained candidate | **NOT RUN (0 attempts).** Exact weights were acquired, but the same input gate stopped execution. It remained inference-only; no post-training or distillation occurred. |
| AD-Copilot | `jiang-cc/AD-Copilot@0081f9363ad871e84475a250cf371cf4b1e53c42` | Requested paired-reference candidate | **AUDITED OUT-OF-BOX LOAD FAILURE; 0 attempts.** The processor adds comparison placeholders, but the exported generation class instantiates the stock Qwen path without the comparison-aware model, producing a token/feature mismatch before generation. The official repository and Space return to the same broken Hub module. Per Liam, it was skipped without weights, import, execution or GPU time; a local repair would violate the out-of-box-only decision. |
| Cosmos-Reason2-8B | `nvidia/Cosmos-Reason2-8B@a9fae2cf89dc64db96b12860417f0eb403013bb9` | Requested candidate | **NOT RUN (0 attempts).** Exact gated weights were acquired after Liam accepted access and the gate/terms archive was sealed, but the same input gate stopped execution. No processor/model load occurred. |

For clarity, AD-Copilot's result is an audited loader/architecture disposition, not an
RCA-bench quality score. The other requested candidates and both controls have no inference
result at all.

## Metrics record

No raw or scored RCA-bench artifact exists. Consequently all requested explanation-quality
and resource metrics are **not measured / not applicable**, rather than zero:

| Metric family | Recorded result |
|---|---|
| Supported-explanation pass and supported-statement precision | Not measured; 0 executed primary items |
| Unsupported-claim rate | Not measured; 0 executed primary items |
| Required-abstention recall and unnecessary-abstention rate | Not measured; neither executed denominator exists |
| Correct-cause rate and cause support | Not measured; no executed cause denominator |
| Numeric fidelity and parse success | Not measured; no responses |
| Image sensitivity / grounding | Not measured; 0/12 qualification pairs executed |
| Per-class metrics (`keep`, `discard`, `insufficient_evidence`) | Not measured; no executed class denominator |
| Paired transitions against baseline (b) | Not measured; baseline (b) and candidates did not run |
| Guard fallback count and reasons | Not measured; no model output reached the guard |
| False-commit or guarded-output commitment outcomes | Not measured; no raw or final method output was committed |
| p50/p95 attempt latency and model-load latency | Not measured |
| GPU allocated/reserved memory and peak memory | Not measured; no model was loaded |

There are therefore no empirical claims about which model is more accurate, more
image-sensitive, faster or smaller. There is also no evidence that any candidate beats
baseline (b) by the frozen 10-percentage-point threshold, holds unsupported-claim rate, or
preserves required-abstention recall.

## Acquisition and legal/provenance record

Acquisition is evidence of local availability, not evaluation:

- AnomalyThink: 21 repository files, 17,564,967,290 bytes.
- Cosmos-Reason2: 19 repository files, 17,546,593,337 bytes.
- Qwen base: 16 repository files, 17,545,915,883 bytes.
- Total: 56 repository files, 52,657,476,510 bytes. Hugging Face verification reported
  zero checksum mismatches.
- `FILE-INVENTORY.tsv` SHA-256:
  `44d54d4602bfb88264e583d8143d8442a9f6129ec35d05df3aa7051f6b0c69b5`.
- Weight-acquisition receipt SHA-256:
  `4e1b6146dd52705a12eb15e5725c0102a858ff7fc1d1b82b63f9393856955c93`.
- Earlier metadata/source/licence receipt SHA-256:
  `439955a912540c70abe2f9ffdb1134e9a70f1bd159266c5188c0a1585fd32a69`.
- Cosmos gate/terms receipt SHA-256:
  `e080618a3a0b02f7cd039012b6e452996fd45015a99b60c964408df7b59a5e5e`.
  Liam, not an agent, accepted the Hub gate; the gate page, NVIDIA Open Model License and
  Trustworthy AI terms were archived before the Cosmos download.

AnomalyThink was allowed only as a read-only, out-of-the-box research comparator. Its model
card says Apache-2.0, but the retained training-source chain includes Real-IAD's
research-only/non-commercial limitation. That unresolved chain is a release and derivative
use limitation: do not fine-tune, redistribute, publish derivatives or copy IAD-VLMs code
without legal review.

The acquired snapshots and all weights remain under ignored `outputs/`; Git tracks none of
the weight-root files. This record does not copy model contents into the repository.

## DP4 and post-training recommendation

**DP4 decision:** use the deterministic guarded-template default. This is the safe behavior
selected by the approved fallback rule after the input prerequisite failed. It is not an
empirical ranking and does not turn the unexecuted deterministic template into a measured
winner.

**Recommendation: do not post-train now.** No post-training or distillation occurred in
this round. WP8a's gate was not met: there is no valid predicted-input benchmark, no scored
base-versus-candidate comparison and no credible evidence that tuning can be judged before
the deadline. Any later post-training or distillation requires all of the following before
work starts:

1. Liam's new explicit approval for the method, compute and spend;
2. independently accepted RGB-only predicted inputs and a sealed, non-leaky RCA-bench plan;
3. a fresh validation/test design that preserves the one-pass boundary; and
4. legal review, especially for any AnomalyThink/Real-IAD-derived data or weights.

Until then, the guarded template should fail closed and state insufficient evidence where
the accepted Part B evidence is unavailable.

## Evidence boundary

This record was written from source files and ignored artifact metadata only. It did not
open held-out/evaluation/video-5 inputs, inspect weight contents beyond inventory metadata,
run a model, use a GPU, change Linear, or perform post-training.

Key evidence snapshot:

| Evidence | SHA-256 |
|---|---|
| Frozen WP3 protocol, `outputs/pipeline-20261028/wp3-rca-bench-v1/wp3-eval-protocol.md` | `6a6465048e3839bdcce60c39686f6edf382dfff0b6d23493d0d9b4a4e4d3a4f3` |
| Predictor independent review | `3d809d9b95c9717a2567f81358abc140b96238f27a311ec696049643b23e2d2d` |
| WP6 real-input gate audit | `7f926a1831e43a5379c900a3bcd3537341059b1ba756a6dfb119defc45fa06ed` |
| AD-Copilot audit | `b3f43cf4c3cefced7b7cd80c384a17fe1c0e419dd3d5de0757d5ab5b738207a8` |
| AD-Copilot alternatives report | `a2d27c5a1df812e00d6980c429b6ce87957b855410f8c6b4b7301c300fa85150` |
| Input-builder first review (failed candidate) | `faec31157eb88eb5d3bf82409e0936472e10a1955018008db9a5e2e8bc1ae54b` |
| Progress-log snapshot used for approved decisions | `9a0901bb240eb046de0623c1d11b3e177ba615006fc53dc09be83ad4c11270ca` |
