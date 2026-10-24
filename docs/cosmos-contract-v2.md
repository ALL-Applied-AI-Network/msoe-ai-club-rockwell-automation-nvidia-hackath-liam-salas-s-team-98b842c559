# Cosmos contract v2: no-thinking second look

Status: **accepted with corrections; processor audit and model inference have not run on this
branch**.

## Question and boundary

V1 preserved a complete negative strict-contract result, but 17/18 model outputs reached the
384-token ceiling while the pinned chat template opened a thinking segment. V2 asks one narrow,
post-v1 question: under the identical contract matrix and output cap, does the model's direct raw
contract completion change when the template-native `enable_thinking` flag is set to `false`?

This is not a repair or replacement of v1, a clean reasoning ablation, or an explanation/vision
quality result. The suffix can change model behavior, not only formatting. No result can support
defect detection, cause, future-vial effect, safety, operator usefulness, physics, camera
robustness or sim-to-real transfer.

## Exact treatment and inherited bytes

The sole inference-behavior change is:

```python
processor.apply_chat_template(..., enable_thinking=False)
```

The pinned template renders `<think></think>` instead of the default `<think>\n` generation
suffix. V2 inherits unchanged from the v1 freeze tag:

- six fixture bytes, sealed oracle, prompt text and strict scorer;
- the fixed image and attachment matrix, four modes and 24/15/6/6 denominators;
- model revision/manifest, runtime lock/platform, seed 0, greedy decoding and 384-token cap; and
- no retry, repair, prose stripping, reprompt, output selection or subset rerun.

Only v2 experiment/tag/output identifiers and provenance fields differ administratively. The
template-reference mode does not use the flag.

## Pre-inference audit and kill gate

Before a freeze tag or generation, render every one of the 18 model fixture/mode inputs through
the pinned processor with both flag values. Record raw prompt hash, image attachment/order,
rendered-string and token-list hashes, lengths, common-prefix length and both suffixes. Each pair
must have equal token count and differ only by `<think>\n` versus `<think></think>`. The v1 fixture,
oracle, prompt, image, scorer, model and runtime hashes must remain unchanged.

The runner must recompute that audit before inference and match its tracked bytes. Kill v2 if any
other rendered difference, token-count change, hash/pin change, prompt-pair loss, runner-visible
oracle, dirty/tag mismatch, unsafe/existing output path or runtime/model mismatch appears.

## Frozen run and reporting

Freeze a clean v2 commit/tag before generation, execute all 24 attempts once locally, preserve
every raw response, then run the copied scorer once. Any malformed, capped or errored response
remains in its denominator. Report cap reaches as `x/18` model attempts and per-mode completion as
`x/6`.

The only permitted v1/v2 comparison is descriptive: with the same stated matrix and 384-token
ceiling, direct-output/completion/cap counts changed or did not after the one template switch.
Schema-gated image invariance is “not established” whenever either pair member is invalid. Stop
after the complete v2 result without adapting the cap, prompt, controls or parser.

Fresh reviewer `review_no_thinking_decision` returned `PROCEED-WITH-CORRECTIONS` under exactly
these limits.
