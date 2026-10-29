# WP7 dual-path preflight implementation report

Status: implementation and synthetic verification complete; no WP7 one-pass phase was
authorized or executed. Liam's INN-62 section 6 decision remains open, so neither route is
activated by this work.

## Revision and isolation

- Base: `3028cc4bea7275fd9b393576c429045d3ddbfeb4`
- Initial implementation head: `d1cb38254b1d5542514c3c98881d97078d3b2846`
- Round-1 corrected implementation head: `008667203bfe4d9e54029b6a09998302167b59e0`
- Branch: `codex/wp7-dual-path-01a18128`
- Owner worktree: `/home/hackathon5/liamreporoot/wp7-dual-path-01a18128`
- The initial implementation head is a direct child of the requested base. The corrected head is
  its descendant through the initial documentation receipt.
- This report is a documentation-only child of the corrected implementation commit; the final
  branch head is reported in the handoff because a commit cannot contain its own object ID.

## Requirements mapping

| Requirement | Implemented evidence |
|---|---|
| Preserve the reviewed evaluation-only outcome | `preflight_explanation_route` requires an exact reviewed zero-row evaluation-store receipt and returns 18 test abstentions plus 30 real-check skips. |
| Prepare the history-backed route without activating it | The separate `history_backed` branch requires exact predicted-mask and production-history provenance plus nonempty trusted WP5 `HistoryQuery` capabilities. |
| Require valid predicted-mask provenance | Exact schema, reviewed RGB-only source class, test-and-real split role, RGB/mask/runtime/config/validation hashes, and `accepts_real_rgb_only=true` are mandatory. |
| Require production-history evidence | The store purpose must be `production_history`, schema version must equal accepted WP5 version 3, sources must be exactly camera RGB and predicted masks, query hashes must match, and every trusted query must contain at least one member. |
| Reject unsafe history inputs | Evaluation-only stores, `evaluation`/`test` query members, oracle/sim-metadata path vocabulary, non-predicted source classes, forged query dictionaries, empty queries, schema drift, and query-hash drift fail closed. |
| Never silently fall back | `history_backed` validation errors raise; every successful receipt records `selected_mode == requested_mode` and `fallback_used=false`. |
| Keep unresolved history route dormant | Valid history/mask/store evidence without a separate route decision returns `pending_inn62_section6_approval` and `history_backed_explanations_enabled=false`. |
| Require exact Liam route approval for future enablement | The separate canonical decision must name `INN-62`, `section_6_history_backed_explanations`, approved `history_backed` mode, `Liam`, and `@salasl`; its caller-declared SHA-256 must match. Store approval cannot substitute. |
| Prevent one-pass bypass | Both APIs and CLI phases require the exact preflight receipt, declared SHA-256 and selected mode. Validation precedes freeze loading, claims and recipe/video reads. |
| Preserve the execution binding | Claim, every journal event, source bindings, fatal/normal terminal receipts and provenance inventory bind the preflight SHA-256 and selected mode. Recipe denominators must equal the bound 18/30 counts before video read. |
| CPU-only synthetic verification | Tests create only temporary synthetic bytes, predicted-mask fixtures, and WP5 stores. No real, held-out, oracle, or video5 input is used. |

## Changed files and SHA-256

- `src/fluidmix/carryover_pipeline.py`: `494bce91a49efd175b2793456431094355ca240eb380ceab2d9dd64c03eab686`
- `scripts/run_carryover_pipeline.py`: `c99611eb0c85d3270f8b401241a17623f443781cdaff7009d6be325689d68e71`
- `tests/test_carryover_pipeline.py`: `b75ddad3e0df214ac3a71becb5e86aabb48a5dbfb8fdfe89f27103816834ab8c`
- `tests/test_wp7_route_preflight.py`: `db34a61c01eec462dca482812f78da30d3a343a5760e50345dc24c855fe2bf8b`
- `docs/architecture.md`: `9ff7a0c8fcd669c0a26d8cabfad328fda29e658dabec67919fe7a1d9d0d71827`
- `docs/evaluation.md`: `ce5e24a8b45989e8e74ab870fba713526b9a016dd6df8d8eabc0fbb2dd78af04`

## Test evidence

Red evidence before implementation:

- `python -m pytest tests/test_wp7_route_preflight.py -q`
- Result: 10 failures, all because `preflight_explanation_route` did not exist.

Green evidence after implementation:

- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python -m pytest tests/test_wp7_route_preflight.py -q`
- Result: `11 passed`.
- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python -m pytest tests -q`
- Result: `2767 passed, 14 skipped in 390.45s`; exit 0.
- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python scripts/smoke_absorbance.py`
- Result: `synthetic smoke passed`; exit 0.
- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python scripts/check_agent_docs.py`
- Result: `shared pointer, 11 skills and 54 document file-target checks passed`; exit 0.
- `git diff --check`
- Result: no output; exit 0.

Round-1 independent review and corrective RED evidence:

- Review: `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-dual-path-preflight-01a18128/INDEPENDENT-REVIEW-d1cb382.md`
- Verdict: BLOCK on two findings: history enabled without the unresolved INN-62 section 6
  decision, and optional preflight bypass through both one-pass entry points/CLI.
- Pending/approval tests initially produced two failures: the unapproved route returned
  `history_backed_requirements_satisfied`, and the API accepted no route-decision arguments.
- Missing-preflight execution test initially failed after creating a claim and attempting the
  synthetic recipe read, proving the helper was bypassable.
- Mismatched-hash/mode execution tests initially produced two failures by reaching the synthetic
  recipe parser instead of rejecting the preflight.

Round-1 corrective GREEN evidence:

- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python -m pytest tests/test_wp7_route_preflight.py tests/test_carryover_pipeline.py tests/test_evidence_store.py -q`
- Result: `57 passed in 12.58s`; exit 0.
- `/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python -m pytest tests -q`
- Result: `2773 passed, 14 skipped in 390.83s`; exit 0 against corrective head
  `008667203bfe4d9e54029b6a09998302167b59e0`.
- Synthetic smoke, agent-doc checks and `git diff --check` all exited 0.

The video5 diagnostic fit was intentionally not run because this task forbids opening video5.

## Protected-input non-access statement

- The frozen exact worktree
  `/home/hackathon5/liamreporoot/wp7-onepass-exec-01a18128` was seen only as a path in the
  repository-wide `git worktree list` inventory. No command used it as a working directory,
  opened a file beneath it, or modified it.
- No frozen execution root or one-pass claim was created. Neither frozen `run-test` nor frozen
  `run-real-check` was invoked. Existing unit tests exercised only temporary synthetic calls.
- No protected test, held-out, real-check, video5, oracle, or `sim_metadata` input was opened.
- No real media was decoded, no GPU command was run, no Linear operation was performed, and no
  push or PR was attempted.

## Exact future activation conditions

Evaluation-only may be selected only when all of the following are true:

1. Liam explicitly resolves INN-62 section 6 in favor of the evaluation-only route.
2. The supplied receipt has the exact `wp7-evaluation-only-store-v1` schema, purpose
   `evaluation_only`, accepted WP5 schema version 3, reviewed schema/snapshot/receipt hashes,
   and `stage_row_count=0`.
3. The frozen denominators remain exactly 18 test vials and 30 real-check vials.
4. The resulting preflight receipt selects 18 abstentions and 30 skips with no history inputs.

History-backed may be selected only when all of the following are true:

1. A separate canonical `wp7-route-decision-v1` artifact exists with exact artifact type
   `wp7_route_decision`, issue `INN-62`, scope `section_6_history_backed_explanations`, approved
   mode `history_backed`, status `approved`, approver `Liam`, and handle `@salasl`; the declared
   route-decision SHA-256 matches its canonical bytes. No actual artifact was created here.
2. A separately reviewed RGB-only predicted-mask source exists and its actual bytes, runtime,
   config, validation receipt, split role, and real-RGB acceptance match the exact provenance
   hashes supplied to preflight.
3. A separately approved store has purpose `production_history`, accepted WP5 schema version 3,
   an independently checked snapshot and approval receipt, and only camera-RGB/predicted-mask
   source classes. An evaluation-only, oracle, or sim-metadata store is not eligible.
4. WP5 returns at least one genuine, nonempty `HistoryQuery`; its ordered query hashes exactly
   match the store receipt, all citations revalidate, and no member has `evaluation`/`test` role
   or forbidden oracle/sim-metadata path provenance.
5. `preflight_explanation_route(requested_mode="history_backed", ...)` returns enabled rather
   than pending. Any error remains terminal; it must not trigger evaluation-only fallback.
6. The execution owner supplies that exact receipt, its declared preflight SHA-256 and selected
   mode to the one-pass API/CLI. Execution revalidates them before any claim or recipe/video read.
7. A separately authorized execution owner rechecks the freeze and inputs before creating any
   one-pass claim. This implementation report is not that authorization.

## Limits and review

- Provenance receipts are exact schema-and-hash bindings. This preflight does not itself prove
  that a referenced review/approval receipt is legitimate; that independent approval and byte
  recheck are explicit activation conditions above.
- Nonempty synthetic history proves route gating only. It does not prove full 18/30 coverage,
  explanation quality, detector accuracy, physical validity, or real-data suitability.
- Self-review under `review-change` inspected committed, staged, unstaged and untracked state
  through the implementation commit with behavior, interface, data-integrity and evidence
  lenses. No P1/P2 defect was found. The bounded limitation is the external approval check just
  stated.
- Round-1 independent review blocked the initial candidate on two High findings. Corrective head
  `008667203bfe4d9e54029b6a09998302167b59e0` addresses exactly those findings and awaits scoped
  re-review; this report does not claim reviewer acceptance.
