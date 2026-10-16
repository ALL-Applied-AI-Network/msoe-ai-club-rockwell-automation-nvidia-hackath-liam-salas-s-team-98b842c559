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
| [diagnose-fluidmix](../.agents/skills/diagnose-fluidmix/SKILL.md) | Reproducing and isolating math/script/setup failures | Minimal reproduction, supported cause and regression evidence |
| [tdd-fluidmix](../.agents/skills/tdd-fluidmix/SKILL.md) | Implementing an agreed behavior test-first | Checked input/output contract and red/green evidence |

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
See the [Oct 16 acceptance audit](acceptance-inn-23-24.md) for fresh host results and
limits; the [Oct 14 review](review-inn-22-24.md) retains the earlier evidence.


## Personal-workflow adaptations added Oct 16

The two Fluid Mixing skills adapt Owen's diagnose and tdd workflows into self-contained
team procedures. Project-specific names distinguish them from globally installed personal
skills. They need no private files, personal checkout, external helper, delegated agent or
particular model. The existing plan-change, review-change and experiment-cycle already
cover the corresponding personal planning, review and experiment loops; avoid duplicates.

Source principles: the diagnose and tdd workflows in
[Owen's skills repository](https://github.com/oap22/skills), themselves adapted from
[Matt Pocock's skills](https://github.com/mattpocock/skills). The team procedures here are
rewritten for the repository's public interfaces, CPU commands and physical-evidence limits.

Routing examples (manual review, not a fresh-host routing evaluation):

| Request | Workflow |
|---|---|
| Why does this CSV command fail on my input? | diagnose-fluidmix |
| The fraction fix still accepts missing values; reproduce it | diagnose-fluidmix |
| Implement this agreed scalar-input contract test-first | tdd-fluidmix |
| Plan a camera estimator | plan-change |
| Review the current patch for defects | review-change |
| Does this calibration prove physical absorbance? | expected-vial-color |

The [Oct 16 fresh-host receipts](acceptance-inn-23-24.md) cover the original nine
skills only. The two additions have file/frontmatter/link checks; automatic discovery,
routing and an end-to-end workflow in a fresh Codex/Claude session remain unverified.
The next fresh session can check the eleven-name catalog before file enumeration and
exercise the new workflow when an authorized bug or implementation task is available.
