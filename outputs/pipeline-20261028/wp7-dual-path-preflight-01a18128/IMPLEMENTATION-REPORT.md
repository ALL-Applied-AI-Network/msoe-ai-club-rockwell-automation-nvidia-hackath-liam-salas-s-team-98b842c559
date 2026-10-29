# WP7 dual-path preflight implementation report

Status: implementation and synthetic verification complete; no WP7 one-pass phase was
authorized or executed. Liam's INN-62 section 6 decision remains open, so neither route is
activated by this work.

## Revision and isolation

- Base: `3028cc4bea7275fd9b393576c429045d3ddbfeb4`
- Reviewed implementation head: `d1cb38254b1d5542514c3c98881d97078d3b2846`
- Branch: `codex/wp7-dual-path-01a18128`
- Owner worktree: `/home/hackathon5/liamreporoot/wp7-dual-path-01a18128`
- The implementation head is a direct child of the requested base.
- This report is a documentation-only child of that reviewed implementation commit; the final
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
| CPU-only synthetic verification | Tests create only temporary synthetic bytes, predicted-mask fixtures, and WP5 stores. No real, held-out, oracle, or video5 input is used. |

## Changed files and SHA-256

- `src/fluidmix/carryover_pipeline.py`: `e28908256b27a0ebb1ab75abaec6be9e1d373c89013a684b056bcd496b2df170`
- `tests/test_wp7_route_preflight.py`: `d4fc5220a6bbe82ebc727e744a058cd287173b9e6d8e3d68bb157c65faca7c9a`
- `docs/architecture.md`: `0a5f749d781481ce173893efaecd9feba71edc69009ea364f359ab02a2cf9a36`
- `docs/evaluation.md`: `d3a94b599ebdc79da26677366f5b24190b49ee77f9652d1134b6615cb5070115`

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

1. Liam explicitly resolves INN-62 section 6 in favor of history-backed explanations.
2. A separately reviewed RGB-only predicted-mask source exists and its actual bytes, runtime,
   config, validation receipt, split role, and real-RGB acceptance match the exact provenance
   hashes supplied to preflight.
3. A separately approved store has purpose `production_history`, accepted WP5 schema version 3,
   an independently checked snapshot and approval receipt, and only camera-RGB/predicted-mask
   source classes. An evaluation-only, oracle, or sim-metadata store is not eligible.
4. WP5 returns at least one genuine, nonempty `HistoryQuery`; its ordered query hashes exactly
   match the store receipt, all citations revalidate, and no member has `evaluation`/`test` role
   or forbidden oracle/sim-metadata path provenance.
5. `preflight_explanation_route(requested_mode="history_backed", ...)` succeeds. Any error
   remains terminal for that selection attempt; it must not trigger evaluation-only fallback.
6. A separately authorized execution owner rechecks the freeze and inputs before creating any
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
- Independent review could not be dispatched from this worker because all four agent slots were
  occupied. The lead will assign that review after this worker exits.
