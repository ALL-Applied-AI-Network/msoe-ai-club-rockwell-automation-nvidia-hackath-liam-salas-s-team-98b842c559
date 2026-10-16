---
name: tdd-fluidmix
description: Implement an agreed Fluid Mixing function or script behavior test-first, one failing test and minimal fix at a time. Use for test-first or red-green-refactor work. Use diagnose-fluidmix when the cause or reproduction is still unknown.
---
# Test-first Fluid Mixing changes

## Inputs
Agreed behavior and accepted input types, public function/script interface, and affected tests.
Read [architecture](../../../docs/architecture.md) for current contracts; proposals there
are not approved APIs. Resolve consequential missing-value or type decisions before coding.

## Workflow
1. Choose a public boundary that exercises the behavior: a function called by production
   code or the actual script with a tiny fixture. State what this boundary cannot verify.
2. Write one test with an independently specified expected result, including the intended
   exception for invalid input. Run it and confirm the behavior fails; import/syntax failures
   do not demonstrate the regression.
3. Make the smallest implementation change that passes. Run the affected test file before
   adding the next behavior. Keep valid endpoints and ordinary inputs as positive controls.
4. Add boundary cases from the agreed contract. Check masks/missing values and type validity
   before conversions that could erase them; do not silently broaden the accepted types.
5. Run the full tests and synthetic smoke. Refactor only while green and rerun checks after
   changing code. For a bug fix, retain evidence that the regression fails on the old code.

## Output and validation
Report the contract, tested boundary, red/green evidence and remaining limits.
Use python -m pytest tests -q and python scripts/smoke_absorbance.py from the repo root
in the CPU environment described in [quickstart](../../../docs/quickstart.md).

## Limits
Do not mirror implementation formulas to construct expected values or mock away the failing
code path. Synthetic tests cannot validate camera measurements or real carryover claims.
No test may weaken the physical-calibration or data-partition requirements to pass.
