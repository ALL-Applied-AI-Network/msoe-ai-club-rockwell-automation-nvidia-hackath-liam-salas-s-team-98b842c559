---
name: experiment-cycle
description: Run a bounded experiment, benchmark or ablation from an agreed hypothesis and measurement.
---
# Experiment cycle

## Inputs
Hypothesis, metric, control, data/split, configuration and authorized resource budget.
Check prior project results before repeating work.

## Workflow
1. Write the falsifier, confounds and stopping condition before running.
2. Save under outputs/<run-id>/ the command, code revision plus dirty diff,
   environment, config, seed, data/split IDs and raw results. Keep secrets out.
3. Run method and control through the same scorer. Check loaded data, units,
   denominators and leakage; vary seeds when randomness matters.
4. Preserve surprising/negative results. Diagnose within budget instead of tuning them away.
5. Record conclusion, limits and next question in existing project docs.

## Output and validation
A traceable run record and conclusion. Reproduce important notebook results from a fresh
state or script; label unreproduced findings provisional.

## Limits
This skill supplies no runner. It does not grant compute spend or override
[data separation](../../../docs/evaluation.md). Use [evaluate-vs-baseline](../evaluate-vs-baseline/SKILL.md)
for detector scoring.
