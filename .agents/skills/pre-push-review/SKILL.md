---
name: pre-push-review
description: Check project-specific risks and readiness before a repository push or PR.
---
# Pre-push review

## Inputs
Acceptance criteria, intended base and complete change. Follow
[review-change](../review-change/SKILL.md) for diff coverage and findings.

## Workflow
1. Run python -m pytest tests -q, python scripts/smoke_absorbance.py and
   python scripts/check_agent_docs.py, plus checks relevant to changed behavior.
2. Check RGB/absorbance errors, hidden invalid calibration, data leakage,
   timing assumptions and unsupported results against [physics](../../../docs/absorbance.md),
   [data](../../../docs/data.md) and [evaluation](../../../docs/evaluation.md).
3. Inspect staging for secrets/large artifacts. Resolve findings or document limits.

## Output and validation
Review findings, actual check results and unresolved acceptance criteria.
Host discovery, rendering and physical validity need their own evidence.

## Limits
Passing checks does not authorize publication. Follow [AGENTS.md](../../../AGENTS.md)
and the user's review boundary.
