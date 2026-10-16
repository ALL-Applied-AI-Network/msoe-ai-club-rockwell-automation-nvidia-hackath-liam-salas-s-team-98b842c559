---
name: diagnose-fluidmix
description: Diagnose a reproducible failure in Fluid Mixing math, data scripts or setup; isolate the cause before fixing. Use for a failing command or a fix that did not hold. Use review-change for reviewing a diff and tdd-fluidmix for an already understood behavior change.
---
# Diagnose a Fluid Mixing failure

## Inputs
Failing command, expected behavior, exact input/configuration, source revision and environment.
Read the affected interface in [architecture](../../../docs/architecture.md) and use
[data provenance](../../../docs/data.md) for CSV/video failures.

## Workflow
1. Run the smallest command that exercises the reported failure. Capture the exit status
   and the relevant actual/expected output; distinguish setup failures from wrong results.
2. Reduce the input without losing the symptom. Preserve source data; use a small fixture
   or an ignored output directory. Pin randomness and record dependency versions if relevant.
3. Rank plausible causes, each with an observable prediction. Change one variable per probe;
   keep evidence that rules a cause out. If reproduction needs unavailable data or hardware,
   report the missing input instead of treating a synthetic substitute as reproduction.
4. Once a cause is isolated, state the input/output contract before an authorized fix.
   Add a regression through the public function or script and show it fails on the old code
   for the expected reason, then passes with the fix.
5. Re-run the original command and affected checks. Remove temporary instrumentation.
   If fixes expose new failures repeatedly, revisit the contract before another patch.

## Output and validation
Report the reproduction, supported cause, probes, changed files and before/after evidence.
Run the affected tests, then the CPU checks in [quickstart](../../../docs/quickstart.md).
Label software correctness separately from physical validity.

## Limits
Negative fitted absorbance is an existing diagnostic result, not automatically a new bug.
Do not alter labels, splits or thresholds to make a failure disappear. This workflow grants
no source-data download, equipment operation, compute allocation or publication permission.
