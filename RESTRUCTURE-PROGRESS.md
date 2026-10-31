# Onion restructure progress

## Authorities and boundaries

- Documented: worktree `/home/hackathon5/liamreporoot/onion-restructure-20261031`, branch `codex/onion-restructure-20261031`, base `baebe514c669bf3b311c3b8091ca2934807c10f5`.
- Documented: no push, PR, GPU, Brev, protected-render/output writes, video 5 access, or WP1 test-data access.
- Received 2026-10-31: read-only target structure `/home/hackathon5/liamreporoot/overnight-ops-20261028/ONION-STRUCTURE.md`; it replaces the inaccessible private rationale link and is subordinate to the thread goal.
- Mapping source: read-only `/home/hackathon5/liamreporoot/overnight-ops-20261028/ONION-MAPPING.json`.

## Baseline at `baebe51`

Measured environment: worktree-local Python 3.12 virtual environment with `requirements-dev.txt`, `requirements-vision.txt`, `opencv-python-headless`, `scikit-image`, and the quickstart-documented optional `usd-core`. Tests require `PYTHONPATH=src`; without it, newly added modules fail collection with `ModuleNotFoundError: fluidmix`.

Measured collection:

```text
env PYTHONPATH=src .venv/bin/python -m pytest tests --collect-only -q
3256 tests collected
```

The unrestricted suite was split into disjoint file partitions so subprocess-bearing tests could be pinned to `/tmp` and observed without touching shared outputs or other worktrees:

```text
# Files containing subprocess calls or hard-coded shared input paths
env PYTHONPATH=src rg -l 'subprocess|/home/hackathon5' tests/test_*.py |
  xargs .venv/bin/python -m pytest -q --basetemp=/tmp/onion-baseline-pure

# Complement: files containing neither pattern
env PYTHONPATH=src rg --files-without-match 'subprocess|/home/hackathon5' tests/test_*.py |
  xargs .venv/bin/python -m pytest -q --basetemp=/tmp/onion-baseline-pure-final
```

Measured baseline result after installing `usd-core`: **3210 passed, 3 failed, 43 skipped (3256 collected)**.

- Pure partition, directly measured after dependency completion: 1580 passed, 0 failed, 8 skipped.
- Integration partition: initial direct measurement was 1612 passed, 21 failed, 35 skipped. A focused rerun of the five OpenUSD-dependent files measured 76 passed and proved the 18 missing-`pxr` failures from that partition were dependency-only. Adjusted integration baseline: 1630 passed, 3 failed, 35 skipped.
- Source observation: the three remaining failures already exist at the untouched base:
  1. `tests/test_wp7_history_preparation.py::test_preflight_is_sealed_and_rejected_before_one_pass_access` — `run_test()` no longer accepts the test's `store=` argument.
  2. `tests/test_part_boundary.py::test_part_boundary_manifest_registers_the_complete_current_boundary_inventory` — `fluidmix.wp7_evaluation_only_preparation` is not ownership-registered.
  3. `tests/test_part_boundary.py::test_authoritative_manifest_entries_resolve_and_unknown_roots_fail_closed` — the same missing ownership registration.

Parity gate for the move-only result: **3210 passed**, with the same three named baseline failures and 43 skips unless the replay base changes.

## Work ledger

- Baseline recorded before move-script implementation.
- Task 1 measured: `tools/restructure/onion_move.py` and focused temporary-repository tests implemented. A fresh controller run, separate from the implementer's reported 1.37-second run, measured `.venv/bin/python -m pytest tools/restructure/test_onion_move.py -q` as `14 passed in 1.19s`; controller `py_compile` and `git diff --check` both exited 0.
- Task 1 reviewed: independent read-only review completed five fix-round slots—code-changing Rounds 1, 2, 4 and 5 plus a blocked, no-change Round 3. The final verdict was 4 remaining findings addressed, 0 open, no new Critical/Important breakage, Task quality Approved.
- Documented limit: the tool has executed against this real worktree only in non-mutating dry-run mode; no move/apply has occurred.
- Initial dry-run report: repository-maintenance `scripts/check_agent_docs.py` and the mapping-new `src/fluidmix/wp7_evaluation_only_preparation.py` / `scripts/prepare_wp7_evaluation_only.py` pair were the only unmapped paths before Task 2 recorded responsibility rulings.
- Task 2 dry run measured on commit `366e766`: 423 planned moves, exit 2, and exactly those three unmapped paths; no files were moved.
- Ruling: route `src/fluidmix/wp7_evaluation_only_preparation.py` to `src/fluidmix/pipeline/evaluation_preparation.py` and its CLI to `cli/fluidmix/pipeline/prepare_evaluation_only.py`. Source observation: the module orchestrates evaluation-only publication and imports the pipeline; placing it in `pipeline` preserves behavior without claiming it is pure storage. Cost if wrong: the later linter may expose a persistence boundary violation, which must be reported/fixed in Tasks 3–4 rather than hidden in the move.
- Ruling: route repository-maintenance `scripts/check_agent_docs.py` to `cli/fluidmix/core/check_agent_docs.py` so no legacy `scripts/` file remains and the required CLI tree stays within the approved product/twin/evaluation families. Cost if wrong: this developer command is grouped with product-core CLIs despite validating repository metadata, but no production module dependency is introduced.
- Task 2 route fix measured: fresh controller verification was `17 passed in 1.06s`; `py_compile` and `git diff --check` exited 0. Independent read-only review approved the active supplemental source route, both CLI routes, tests, and corrected progress wording.
- Task 2 second dry run measured: exit 0, 426 planned moves, no `UNMAPPED` marker, no files moved.
- Next: commit the reviewed explicit routes, then apply the move and verify move-only parity.

## Implementation plan

Spec authorities: the active thread goal, then `ONION-STRUCTURE.md`, then `ONION-MAPPING.json`.

Global constraints:

- Work only in this worktree and branch. Everything else is read-only.
- No push, PR, merge, GPU, Brev, `outputs/`, protected render folders, video 5, or WP1 test-data access.
- Preserve behavior. A move-only change may alter paths/imports but not logic.
- Mechanical moves, import rewrites, and `PATH-MAP.md` are delegated to Luna or Sol-low. Component fixes use Sol-medium. Every commit receives an independent read-only review before acceptance.
- The script must be deterministic and replayable on a newer commit, report unmapped new files, use `git mv`, and be safe to re-run.

### Task 1 — Build the replayable move tool

Create `tools/restructure/onion_move.py` and focused tests under `tools/restructure/` test-first. The tool must:

- Read an explicit mapping path, defaulting to the read-only parent-workspace `overnight-ops-20261028/ONION-MAPPING.json` when resolvable.
- Translate mapping keys exactly as `ONION-STRUCTURE.md` specifies into `src/fluidmix/`, `src/twin/`, and `src/evaluation/`.
- Contain an explicit responsibility-based rename table; do not derive ticket-prefix removal heuristically.
- Plan and perform `git mv` for mapped production modules, `scripts/` to responsibility-mirrored `cli/`, and tests to the source-mirrored `tests/` tree.
- Rewrite absolute and relative Python imports in `src/`, `cli/`, and `tests/` without changing other logic.
- Generate complete old-path to new-path rows for `docs/architecture/PATH-MAP.md`.
- Offer a non-mutating dry-run that reports every tracked source/script/test file not covered by the mapping/rename plan and exits nonzero for unmapped files unless an explicit reporting-only option is selected.
- Refuse dirty worktrees or unsafe/out-of-repository paths, avoid overwrites/collisions, and make a second run a no-op rather than an error.
- Be tested in temporary Git repositories. Do not run the move against this real worktree in Task 1 and do not commit.

Verification: focused tool tests demonstrate RED before implementation and GREEN after; `python -m py_compile tools/restructure/onion_move.py`; self-review of the diff. The repository baseline suite remains a later move-parity gate.

### Task 2 — Execute and commit the move-only tree

After Task 1 review approval, commit the reviewed tool, execute it on this worktree, add the product-first architecture tour, verify complete path mapping/import parity and baseline pass-count parity, obtain independent read-only review, then make the move-only commit with no logic changes.

### Task 3 — Add and report architecture enforcement

Add the supplied import-linter contracts plus an AST purity test; run them and list every violation without fixing it in the same commit.

### Task 4 — Fix violations one cause per reviewed commit

Implement the four named cause groups from the goal and the 21-break inventory in `ONION-STRUCTURE.md`, with parity tests and Sol-medium implementation/review separation.

### Task 5 — Consolidate duplicate colour/hash helpers only after Tasks 1–4 are green

Add one `colour.delta_e76` and one hash helper with parity tests against every old copy, then delete the copies.
