---
name: review-change
description: Review a code or documentation diff for concrete defects; use before accepting a substantial change.
---
# Review a change

## Inputs
Acceptance criteria, base revision and changed files.

## Workflow
1. Inspect branch diff, staged/unstaged edits and untracked files; preserve other work.
2. Choose relevant lenses: behavior, interfaces, data integrity or evidence.
3. For each finding, state starting conditions, action, wrong result and source location.
   Reproduce when possible; label unresolved hypotheses and pre-existing defects separately.
4. Run relevant checks and inspect their actual output. When fixes are authorized,
   repair the cause and recheck the failing path and affected neighbors.

## Output and validation
Ranked actionable findings with evidence, or a bounded clean review with tested limits.
Label self-review; use an independent reviewer only when authorized.

## Limits
Do not manufacture blockers from preferences, weaken tests or change scope.
For publication, also use [pre-push-review](../pre-push-review/SKILL.md).
