# Shared agent skills

.agents/skills/<name>/SKILL.md is authoritative; .claude/skills points to it for Claude.
Codex discovers the canonical directory. Keep one copy. CLAUDE.md imports AGENTS.md.
If symlinks are unsupported, manually load the canonical skill and report auto-discovery
as unverified.

## General workflows

| Skill | Use it for | Output |
|---|---|---|
| [plan-change](../.agents/skills/plan-change/SKILL.md) | Any feature/refactor | Scope, acceptance criteria and verification plan |
| [review-change](../.agents/skills/review-change/SKILL.md) | Reviewing code/docs | Concrete findings and checked limits |
| [experiment-cycle](../.agents/skills/experiment-cycle/SKILL.md) | Experiments/benchmarks/ablations | Traceable run record and evidence-backed conclusion |

Adapted principles from the personal plan-then-ship, adversarial-review and research-loop
skills. These team versions contain no personal paths, model assignments, mandatory
delegation or external helpers. Project-writing also uses the source-backed current-state
approach from project-status-doc.

## Project workflows

| Skill | Use it for | Output |
|---|---|---|
| [video-timing](../.agents/skills/video-timing/SKILL.md) | Still/frame alignment | Corrected labels and evidence |
| [expected-vial-color](../.agents/skills/expected-vial-color/SKILL.md) | Absorbance/calibration/residue | Prediction/fit with validity |
| [twin-liquid-variant](../.agents/skills/twin-liquid-variant/SKILL.md) | Synthetic USD overrides | Layer and metadata checks |
| [evaluate-vs-baseline](../.agents/skills/evaluate-vs-baseline/SKILL.md) | Detector scoring | Matched baseline comparison |
| [project-writing](../.agents/skills/project-writing/SKILL.md) | Docs/status/review prose | Sources, limits and next action |
| [pre-push-review](../.agents/skills/pre-push-review/SKILL.md) | Publication readiness | Project risk/check evidence |

Each skill states inputs, workflow, output/validation and limits. Shared policy stays in
[AGENTS.md](../AGENTS.md); detailed facts stay in docs.

## Validation

Run python scripts/check_agent_docs.py. It discovers immediate skill folders dynamically
and validates complete YAML frontmatter (including duplicate keys), names/descriptions,
the shared pointer and local file targets. Heading fragments are not checked.
New valid skills need no checker inventory edit. This proves layout, not host loading.

In a fresh Codex/Claude session, check the startup catalog before enumerating files.
Select expected-vial-color normally and run python scripts/smoke_absorbance.py.
Record host/version, discovered names, selected skill, command and actual output.
See [review evidence](review-inn-22-24.md) for host results and untested changes.
