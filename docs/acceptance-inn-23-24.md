# INN-23/24 acceptance audit, 2026-10-16

Base: main e4146b10671c38ed866e2e77728f329fa978b0fb (merged PR #3).
This audit follows the [Oct 14 evidence](review-inn-22-24.md). It records new checks and
remaining gates; it does not close either issue or certify a teammate trial.

## Current source and feedback

Read-only Git SSH refs on Oct 16 matched the base above. Main and INN-7_Audit_Dataset2
pointed to that commit. The signed-in GitHub browser showed zero open PRs; PRs #2 and #3
were merged, each with zero conversation comments and no reviews. Linear INN-23 and
INN-24 remained In Progress, assigned to Owen, and both comment lists were empty.
These observations do not imply that organizer feedback is absent from other channels.

The GitHub connector returned 404 and local gh authentication was invalid. Browser and
Git SSH provided the observations above without changing accounts or credentials.

## INN-23 coverage

| Acceptance criterion | Evidence/status |
|---|---|
| One authoritative source and compatible host discovery | .agents/skills contains nine skills; .claude/skills is a relative symlink to it; docs/skills.md explains fallback limits |
| Six original workflows reviewed | Inventory below; shared policy stays in AGENTS.md |
| Trigger, inputs, steps, outputs, validation, limits | All six have a trigger in frontmatter, Inputs, Workflow, Output and validation, and Limits |
| Commands match the repository | CPU commands rerun below; source-video extraction and USD/render execution remain untested in this audit |
| AGENTS.md routes to index and workflows | Existing contract links docs/skills.md and names all nine workflows |
| Fresh Claude discovery/workflow | Nine project skills exposed; expected-vial-color invoked through Skill; synthetic smoke passed on the retry described below |
| Fresh Codex discovery/workflow | New model-backed attempt blocked by CLI/account model incompatibility; no workflow success claimed |

| Reviewed skill | Checked purpose and boundary |
|---|---|
| video-timing | Own clocks per video; source-video confirmation required; no general all-video timing claim |
| twin-liquid-variant | Ignored override output; no render/physical-absorption success claim |
| expected-vial-color | Physical input checks; negative calibration blocks use; synthetic recovery is not camera accuracy |
| evaluate-vs-baseline | Matched eligibility, independent labels, frozen splits/thresholds; no implemented scoring CLI claimed |
| project-writing | Source-backed prose and dated conflicts; drafting grants no sending permission |
| pre-push-review | Actual checks and limits; passing tests grant no publication permission |

The Oct 14 Codex local-discovery result remains historical evidence. The Oct 16 fresh
CLI 0.147.0 attempt used its configured gpt-6.1-sol model and read-only sandbox. The
ChatGPT-account endpoint rejected that model before a model response. It also logged
local MCP startup errors; none of these prove project discovery. No model was substituted.

Fresh Claude Code 2.1.285 exposed the nine project skills before scanning skill files.
It invoked expected-vial-color through Skill and read its linked physics document.
The retry did not read architecture.md or run the skill's full pytest command; the test
scope was catalog selection plus the representative synthetic smoke. It reported missing
catalog descriptions for review-change, twin-liquid-variant and video-timing despite
valid frontmatter in the repository. Discovery of their names passed; that display limit
was not diagnosed or repaired by this audit.
The first attempt added shell activation/diagnostic wrappers and was denied by the test
harness's exact-command permission. The fresh retry retained that narrow permission,
used the already configured PATH and executed only python scripts/smoke_absorbance.py:

    old-fluid fraction 0.150000; RMS absorbance residual 0.000e+00
    synthetic smoke passed

This is agent-host evidence, not a human teammate trial. The host reported claude-sonnet-5-5;
no model override was supplied.

## INN-24 coverage

| Acceptance criterion | Evidence/status |
|---|---|
| Index, environment/dependencies, small data, tests and smoke | docs/README.md and quickstart.md; fresh local venv and commands rerun |
| Pipeline interfaces/modules/units and stage relationships | architecture.md: implemented API table plus explicitly proposed records, validity and anomaly/cause separation |
| Provenance, timing, splits, baselines and limits | data.md, evaluation.md and absorbance.md; equivalent recipes 5/8 and inherited fold limitation now explicit |
| Configuration, secrets, output locations and shutdown | operations.md; no credentials or large datasets added |
| Proposal/project links and decisions versus evidence | Index links merged proposal and Linear; plan.md now distinguishes the older proposal from feasibility gates; INN-32 protocol remains unapproved |
| Teammate clean checkout and corrections | **Pending**; teammate-quickstart.md is an unfilled receipt template |

## CPU evidence

Fresh Python 3.14.5 venv, macOS arm64. Installed declared requirements: NumPy 2.5.4,
pytest 9.1.1, PyYAML 6.0.3, Pillow 12.3.0. No GPU, raw video, twin or large download used.

| Command | Actual result |
|---|---|
| python -m pytest tests -q | 24 passed |
| python scripts/check_agent_docs.py | Shared pointer, nine skills and local document file targets passed |
| python scripts/smoke_absorbance.py | Fraction 0.150000; residual 0; passed |
| python scripts/fit_absorbance_video5.py | 20 presumed-clean vials; negative absorbance; median/max Delta E 12.6/26.4 |

The existing diagnostic's recipe-ID folds split equivalent mixtures 5/8. The numbers
above reproduce that diagnostic, not a corrected unseen-mixture or detection comparison.
No thresholds, physical go/no-go or cause labels are inferred from them.

## Remaining gates

INN-23 needs a supported fresh Codex session that demonstrates catalog discovery and a
representative workflow. INN-24 needs a real teammate's clean-checkout receipt and any
corrections. INN-11 record fields and INN-32 tolerances/route still need team agreement.
The draft protocol does not change those issues' status or authorize downstream compute.
Review and publication of this private docs diff are separate from these checks.
