---
name: plan-change
description: Plan a feature or refactor before implementation; scale the plan to the change.
---
# Plan a change

## Inputs
Requested behavior, repository entry points and constraints.

## Workflow
1. Inspect current behavior and ownership; ask only about consequential unknowns.
2. Define observable acceptance criteria, scope and exclusions.
3. Compare plausible approaches; record the chosen interfaces, affected files,
   risks and verification commands.
4. Keep decisions separate from assumptions. Resolve required answers before dependent work.
5. Hand off the plan with the next implementation step and unresolved decisions.
   Proceed to implementation only if it is within the user's request.

## Output and validation
A compact plan sized to the task. Each requirement maps to a change and a way to prove it.

## Limits
A plan is not evidence of completion. Do not add delegation, model changes or publishing
permission. Follow [AGENTS.md](../../../AGENTS.md).
