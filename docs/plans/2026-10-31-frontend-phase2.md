# Phase 2 real-output adapter and recording mode plan

Spec authority: Liam's approved Phase 2 request relayed on thread
`01a18b55-d34b-7261-9429-b389ab070b0d` on 2026-10-31.

Scaffold base: `b596d11` on `codex/frontend-ui-20261030`.

Upstream contracts are read-only observations from commit `42909d3` on
`codex/pipeline-final-integration-wp7-01a18128`:

- `src/fluidmix/wp7_final_decision.py` returns `wp7-final-decision-v1` records and nested
  `wp7-operator-record-v1` records from `decide_wp7_final`.
- `src/fluidmix/evidence_store.py` defines WP5 SQLite schema version 3.

## Global constraints

- Work only in this worktree. Do not push, open a PR, edit outputs, check out the upstream
  branch, use GPU/Brev, or modify another worktree.
- Read upstream code only with `git show`. Open SQLite inputs with a URI using
  `mode=ro&immutable=1`, set `query_only`, and deny access to `sim_metadata`.
- Never read WP1 test data, video 5, grounding qualification, sealed inputs, oracle data,
  or simulator truth. The adapter must not discover or traverse those paths.
- Development inputs are synthetic development evidence, never plant, production, physical
  accuracy, or final-run evidence. Preserve source IDs, hashes, split versions, claim boundaries,
  and unavailable values.
- WP7 decision records are authoritative for disposition, per-stage Delta E/threshold,
  limit codes, explanation text, and citations. SQLite supplies identity, images, measured Lab,
  recipe/run context, and earlier evidence. The UI must reject cross-source disagreement.
- No published `wp7-final-decision-v1` file exists at planning time. The accepted published
  V3/V4 development-history store contains 18 vial decisions, all
  `insufficient_evidence`; do not invent a development keep/discard result.
- Liam decision relayed 2026-10-31: demo, development, and final views show source-backed data
  only. If WP7 publishes no discard, the recording shows the real abstentions and evidence
  honestly. Never add a synthetic or invented discard to a development/final view.
- Ruling: the development E2E and recording use the honest 18/18 abstention fixture. Mock data is
  available only to component/contract tests with its existing synthetic label; it is not a demo
  or screenshot source. If this ruling is wrong, no pre-publication recording demonstrates the
  discard animation, although the source-driven UI path remains ready to render one.
- Phase 1 behavior remains available through the same source switch. All normalized output must
  validate against the committed UI schema before the browser loads it.

## Chosen interface

`ui-data/source.json` contains exactly one configurable value:

```json
{"source":"development"}
```

Supported values are:

- `mock` — materialize the existing Phase 1 mock for component/contract tests only;
- `development` — materialize the committed V3/V4 development fixture;
- a path to a published WP7 output directory — final mode.

Changing that one `source` string is the only configuration edit needed when WP7 publishes.
The final directory adapter discovers one evidence database named `evidence.sqlite3`, accepts
final decisions from an unambiguous `wp7-final-decisions.json`, JSONL equivalent, or
`decisions/*.json`, and uses a published MP4 when present. If no MP4 is present, it derives a
clearly labelled presentation reel from hash-verified target RGB frames. Ambiguous or missing
inputs fail closed.

A Python materializer reads the selected source without mutation and atomically produces the
browser-facing `ui-data/active-run.json` plus local derived assets. Vite always fetches
`/active-run.json`; pre-dev, pre-build and E2E preparation invoke the materializer.

The UI contract advances to version 2 and supports variable vial counts, development/final
labels, WP7 limit-code strings, and unavailable stage comparison values. For a multi-stage WP7
record the detail focus is deterministic: first outlier stage for discard, otherwise the valid
stage with the largest target-to-threshold ratio, otherwise the first stage. All stage citations
remain available even though the existing detail card shows one focus stage.

## Acceptance criteria

1. The real adapter accepts the exact WP7 record shape, joins it to schema-v3 evidence by target
   evidence/measurement/image identity, verifies hashes and values, and rejects drift, ambiguity,
   missing citations, unsafe paths, wrong SQLite versions, and writes.
2. The committed fixture is derived only from approved V3/V4 development outputs, excludes
   `sim_metadata` contents and forbidden sources, is labelled `Development data`, and has a
   hash/provenance manifest.
3. `source.json` is the only input-selection setting. `mock`, `development`, and a temporary
   final-output directory all materialize valid `active-run.json` without source edits.
4. The dashboard renders any positive vial count, unavailable metrics, contract-driven source
   labels, and final WP7 limit codes without weakening pending-state privacy.
5. `?demo=1` is a clean 1920x1080 recording mode for development/final sources: muted autoplay,
   progressive decision cues, and automatic focus/highlight if a source-backed discard arrives.
   An all-abstention source remains all-abstention on screen. Runtime playback remains
   user-controllable outside demo mode.
6. Unit tests cover JSON/JSONL/directory decision discovery, read-only SQLite joins, both committed
   fixtures and the UI schema, final-like keep/discard/insufficient records, switch modes, source
   hash immutability, and invalid inputs. Component tests cover variable counts, nullable values,
   autoplay, streaming discard explanations, and evidence highlight.
7. Playwright runs only on development data at 1920x1080 and produces fresh source-backed
   screenshots. Mock discard streaming/highlight behavior remains covered at the component and
   contract-test boundary, not in demo screenshots.
8. `ui-PROGRESS.md`, the UI-data README, and architecture documentation distinguish measured
   source observations, derived development fixtures, final-format contract tests, and unverified
   future final output.

## Task 1 — real adapter, v2 contract, and one-value switch (sol-medium)

Implement the Python materializer boundary and source configuration test-first. Add small temporary
test fixtures that exercise exact `wp7-final-decision-v1` keep/discard/insufficient records and a
schema-v3 SQLite store. Add v2 normalized schema support and preserve mock mode. The materializer
must never instantiate the writable upstream `EvidenceStore` class.

Verification: focused Python tests with RED/GREEN evidence, schema validation, source-before/after
hash equality, existing Vitest adapter tests, typecheck, and build.

## Task 2 — published V3/V4 development fixture and presentation reel (sol-medium)

Add a reproducible fixture builder that reads only:

- `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-development-history-preparation-01a18128-attempt01/evidence.sqlite3`
  SHA-256 `88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7`;
- its safe `explanation-records.json`, SHA-256
  `c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9`.

Derive a small schema-v3 fixture with no `sim_metadata` rows, actual V3/V4 evidence and all 18
published abstention decisions. Copy only the hash-verified presentation frames/masks needed by the
UI and build an approximately 4.5-minute H.264 development reel from them. Publish a manifest with
source/derived hashes and explicit `Development data` labelling. Never open the sealed-preflight
file in that source directory.

Verification: deterministic rebuild into two fresh roots, exact 6-run/18-vial/V3+V4 provenance,
18/18 insufficient-evidence outcome, asset hashes, source immutability, SQLite quick check, full
media decode, and v2 schema/adapter parse.

## Task 3 — dashboard generalization and recording mode (sol-medium)

Switch the browser loader to `/active-run.json`; render the normalized source label and variable
vial count; support unavailable Lab/Delta E/threshold cleanly; preserve pending privacy and the
top-down hook. Implement `?demo=1` muted autoplay and automatic selection/evidence highlight when a
discard arrives. Keep explanation labels contract-driven and show final WP7 limit codes verbatim.

Verification: component tests first, full Vitest suite, typecheck, and production build.

## Task 4 — development E2E, screenshots, and handoff (luna)

Default `source.json` to `development`, update package preparation scripts, and run the operator
and recording flows against development data only. Capture fresh useful 1920x1080 source-backed
development screenshots without transient media states; do not capture mock data. Update
`ui-PROGRESS.md`, `ui-data/README.md`, and `docs/architecture.md` with exact commands, hashes,
boundaries, and the one-line final switch.

Verification: Python adapter/fixture tests, full Vitest, typecheck, build, development Playwright,
development demo Playwright, screenshot dimensions/inspection, full media decode, docs checker, source
immutability rehash, `git diff --check`, and clean status.

## Final integration

Review each task independently, resolve findings through its original implementer, then run a
whole-branch review and fresh controller verification. No push or PR.
