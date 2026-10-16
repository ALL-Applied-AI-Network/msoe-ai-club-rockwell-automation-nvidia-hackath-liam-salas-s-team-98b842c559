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
| Fresh Codex discovery/workflow | Fresh native app task passed startup catalog discovery, selected expected-vial-color, read required docs, ran smoke and passed 24 tests; the separate CLI route remains incompatible |

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
local MCP startup errors; that CLI attempt did not prove discovery or a workflow.

A separate fresh native Codex app task, "Verify fresh Innovation Lab skill discovery"
(thread 01a142fc-ab71-7b50-8d37-8e97dba0e738), completed in the saved source checkout at
e4146b1. It reported all nine project names in its first commentary, before any file reads;
then selected expected-vial-color, read the skill and required physics/interfaces, ran
the synthetic smoke and passed all 24 tests (0.87 s). Source status was clean before/after.
The app task was explicitly created with gpt-6.1-sol selected. Its own runtime exposed
only the generic GPT-6 family, so the report does not independently attest the model
variant. No model substitution was requested or performed.

The task had no inherited conversation or supplied skill contents. It used the lead's
ready venv executable with PYTHONDONTWRITEBYTECODE=1 and pytest's cache disabled.
This proves repository discovery/workflow, not independent dependency setup. It ran in
the saved source checkout rather than the private docs worktree; the skill files, shared
pointer, AGENTS contract, code and tests were identical between them at test time. No
source was edited by that task. Process limitation: the native check ran in the saved
source checkout without entering the required isolated worktree. This does not establish
worktree-procedure compliance; it still records startup discovery and the actual smoke
output on clean source. The final follow-up only corrects a causal claim in a
function docstring; its executable body and the representative smoke remain unchanged.

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

The fresh native Codex test resolves INN-23's previous host-workflow evidence gap for
the unchanged shared skills. INN-23's acceptance evidence is now covered; issue status
was not changed by this private audit. The CLI/account route remains incompatible.
INN-24 needs a real teammate's clean-checkout receipt and any corrections.
INN-11 record fields and INN-32 tolerances/route still need team agreement.
The draft protocol does not change those issues' status or authorize downstream compute.
Review and publication of this private docs diff are separate from these checks.

## INN-11 follow-up audit

The scaffold's usable components remain the CPU math, diagnostic and data-tool checks
recorded above. The proposed record schema/validity handoffs still need team review.
The forward carryover API does not enforce its documented single fraction in [0,1].
For a clear new mixture, old T-200 absorbance [0.1,0.6,1.5] and fraction -0.1, it returns
negative absorbance [-0.01,-0.06,-0.15]. Nonfinite and per-channel array inputs also lack
a suitable boundary check. This is a reproducible software-contract gap, not new evidence
about real fluid behavior or an adopted physical route.

A bounded validation candidate was tested privately but was **not retained**. Fifteen
protective cases failed on the exact pre-fix source while four positive controls passed.
The candidate passed those cases and the resulting 43-test suite; fresh verification
then showed NumPy masked inputs lost their missing-value mask before validation and
became ordinary fractions. That unresolved case prevents calling the guard complete.
The private patch/probes are review material, not a shipped fix or passing acceptance gate.
No forward/fitted math implementation, calibration, measurement floors, CSVs or partitions
were changed. The function docstring now correctly says residuals do not establish cause.

INN-32's protocol was checked against its prerequisites: upstream INN-29/30/31 evidence,
independent labels, equivalent-mixture grouping, INN-17 final-test protection, a timestamped
pre-comparison freeze, uncertainty/coverage accounting, independent reproduction and team
route review. All six numeric tolerance fields remain UNSET. "Nonnegative calibration"
allows the clear tank's existing zero-absorbance model; it introduces no error tolerance.
Neither this audit nor the template supplies missing team agreement or scientific evidence.


## Local completion review, 2026-10-16

Reviewed the complete overnight diff from e4146b1 through 368632c and the follow-up docs
corrections on `codex/inn-23-docs-review`. This is a self-review; no new independent
reviewer was used. The earlier independent review applies to the overnight tree only.

The six original skills still have trigger descriptions, inputs, workflow, outputs,
validation and limits. Their commands match the script interfaces; source-video and USD
execution remain untested. Canonical skills, the Claude symlink and AGENTS routing are
unchanged from the fresh-host checks. The private native Codex and Claude receipts were
inspected directly, including their incomplete/untested portions described above.

Corrected stale evidence links, disclosed the native check's worktree limitation and the
unfixed forward-fraction boundary, aligned the quickstart's diagnostic terminology, and
removed conflicting clean-baseline/data-partition instructions from the plan. Dynamic
threshold benefits now remain a hypothesis requiring validation. Deadline dates were
checked against the Linear project's delivery phases.

Checks rerun in the isolated review worktree using the existing overnight Python venv:
24 tests passed (1.19 s), smoke recovered 0.150000 with zero residual, and the docs checker
passed nine skills and 27 document file-target checks. The diagnostic reproduced negative
absorbance and median/max Delta E 12.6/26.4. No new dependency installation, physical
validation or Linux portability check is claimed. Whitespace checks passed.

INN-23 acceptance criteria are satisfied by the existing shared skills and fresh-host
evidence. The local docs are ready for Owen's later PR; nothing has been pushed or merged.
INN-24 remains pending the human teammate trial. INN-11 interfaces and INN-32 tolerances
remain proposals, and the carryover guard remains unfixed. Reflection: none, no failure
in this completion review; historical failed attempts and their limits are retained above.


## INN-23 closure and portable skill additions

Owen explicitly authorized the tracker update on Oct 16. Linear confirmed INN-23 Done
with the completion evidence appended. INN-24 remains open. The closure covers the
original shared-skill acceptance criteria and the fresh-host receipts above.

A subsequent authorized addition adapts diagnose and tdd into diagnose-fluidmix and
tdd-fluidmix. These team skills are self-contained and link only repository docs; the
skill index records source attribution, routing examples and discovery limits. Both pass
the skill-creator validator, and the repo checker now passes eleven skills and 29 document
file targets. This does not extend the earlier nine-skill fresh-host receipts to the two
new skills. New-host discovery and actual end-to-end use remain unverified for them.

Owen authorized pushing the reviewed branch after checks and will open the PR himself.
Earlier no-push observations above describe the state at their respective review times.
No runtime code, physical calibration, source data or data partitions changed in this
follow-up. The skill-discovery regression originally hard-coded ten skills including its
fixture; adding two skills exposed that stale assertion. It now checks a baseline count
and requires exactly one additional discovered skill, preserving its original purpose. The two new workflows are self-reviewed; no independent reviewer was used.
