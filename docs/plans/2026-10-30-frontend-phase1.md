# Phase 1 single-page operator UI implementation plan

## Spec

Binding brief: `/home/hackathon5/liamreporoot/overnight-ops-20261028/FRONTEND-AGENT-BRIEF.md`.
Repository evidence rules: `AGENTS.md` and `docs/architecture.md`.
Scaffold base: `a1129eb8ed12cf2cf05eb3617596577a1bfe67e1` on
`codex/frontend-ui-20261030`.

## Current state and decisions

- This worktree has no UI, Node runtime, or `outputs/` mount. Published artifacts live in
  `/home/hackathon5/liamreporoot/outputs/pipeline-20261028` and remain read-only.
- Build a Vite + React + TypeScript static app in `ui/`. Use a portable Node runtime only
  for development; commit source and lockfiles, not the runtime or dependency cache.
- The app loads one schema-validated static JSON document through a small adapter. Phase 2
  replaces only that adapter/input, not the components.
- Phase 1 is an explicitly labelled synthetic-twin composite demo. WP5 colour measurements
  and Delta E values retain their evidence IDs and are labelled `measured`. Mock threshold,
  decisions, timing, and top-down positions are labelled `documented` or `inferred`; they
  are not presented as WP5 outputs or plant evidence.
- Use RUN-01 WP1 fill clips and transport states only to derive a compact local MP4 and
  thumbnails. Never read or display video 5, test oracle/labels, sealed artifacts, simulator
  truth, or protected render inputs.

## Global constraints

- Everything lives on one live page: no route navigation and no business-case screen.
- Required states are the live run video and progressive vial strip; in-place vial detail
  with RGB/mask overlay, Lab comparison, Delta E, threshold, decision, and limit codes;
  discard explanations beside their vial with clickable evidence highlights and an explicit
  source label; a same-page history drawer; a model-pane toggle; and a top-down placeholder.
- `ui-data/schema.json` must encode `vial_id` linked to top-down normalized bbox/position,
  decision, and Delta E without requiring component changes when detections arrive.
- Decisions are exactly `keep`, `discard`, or `insufficient_evidence`. Explanation sources
  are exactly `deterministic_template` or `cosmos_reason2`; Phase 1 uses only the former.
- The WP6 table is the documented limited all-abstention benchmark: template 16/16, Qwen3-VL
  16/16, AnomalyThink 0/16 free-form, Cosmos 0/16 strict-parse. It always carries the visible
  label `limited all-abstention benchmark, no accuracy claim` and space for INN-67.
- Every synthetic item is visibly labelled `synthetic twin data`; key numbers expose
  `measured`, `documented`, or `inferred` provenance in accessible labels/tooltips.
- No backend, external service, analytics, upload, GPU, Brev, push, or PR. Keep source inputs
  read-only and copy only small derived assets into this worktree.
- Tests must be written and observed failing before production behavior is implemented.
  Required verification is unit/contract tests, component tests, a production build, and a
  headless Playwright smoke test at 1920x1080 with screenshots.
- Each implementation task commits locally. Reviews are read-only. No agent may dispatch a
  subagent of its own.

## Acceptance criteria

1. `ui-data/schema.json` validates the committed mock and rejects invalid decisions, missing
   evidence provenance, broken vial/top-down links, and out-of-range normalized geometry.
2. A typed adapter loads the mock, checks cross-record invariants, and exposes stable view
   models for all nine vials and the WP6 rows.
3. Playback time progressively reveals vial decisions; selecting a vial updates detail in
   place and a discard explanation streams beside the selected/live vial. Clicking an
   evidence ID visibly highlights its cited number/image.
4. History opens and shows reference provenance plus earlier decisions/explanations. The
   same side area toggles to the limited WP6 model pane.
5. The top-down panel renders a waiting state and nine mock positions keyed by `vial_id`;
   selecting a vial highlights the same physical position and decision/Delta E.
6. The page remains usable at 1920x1080 and narrow widths, with keyboard-operable controls,
   labelled regions, focus styles, reduced-motion support, and light/dark mode.
7. `ui-PROGRESS.md` records sources, provenance limits, commands/results, screenshots, and
   the Phase 2 adapter handoff. It does not claim physical accuracy.

## Task 1: Contract, typed adapter, and unit tests (sol-medium)

Create the Vite/React/TypeScript test scaffold in `ui/`, JSON Schema in
`ui-data/schema.json`, a documented example fixture, and a typed adapter in `ui/src/data/`.
Write Vitest tests first and record RED/GREEN evidence. Validate schema structure and the
cross-record hook `vial_id -> top_down.position/bbox -> decision/delta_e`; reject duplicates,
dangling evidence references, mismatched links, invalid geometry, and unsupported enum values.
Add scripts for unit/component tests, build, and E2E without implementing page components.

Verification: focused adapter tests, then the complete current UI test suite and TypeScript
typecheck/build if a minimal entry exists.

## Task 2: Reproducible mock data and derived local assets (luna)

Using only the authorized WP1 RUN-01 batch, WP5 audited bridge, and WP6 limited score files,
add a small deterministic build script and committed `ui-data/mock-run.json`. Select nine
traceable WP5 pairs, preserve source measurement/evidence IDs and Lab/Delta E values, and
label mock scenario decisions and timing honestly. Derive local thumbnails/mask presentation
assets and a compact browser-playable RUN-01 composite MP4 from the bound fill clips and
transport states. Tests first must fail for missing/incorrect source mapping, provenance,
decision mix, nine-vial linkage, or missing local assets. Do not copy raw source artifacts.

Verification: generator/fixture tests, schema/adapter tests against the committed mock, asset
hash/size checks, and a media probe confirming MP4 playback metadata.

## Task 3: Single live page and component tests (sol-medium)

Implement focused React components for run playback/timeline, vial detail and overlay,
streaming explanation/evidence highlights, history/model side panel, top-down placeholder,
theme control, and the composed one-page app. Write React Testing Library tests first for each
observable interaction and progressive reveal behavior. Keep state in the page; add no router.

Verification: focused component tests, complete unit/component suite, typecheck, and production
build with pristine output.

## Task 4: E2E smoke, screenshots, progress handoff (luna)

Add Playwright configuration and a headless smoke test at 1920x1080. Exercise initial run,
playback-driven vial reveal, vial filter/detail, discard explanation, clickable evidence
highlight, history drawer, model toggle, top-down link/highlight, theme, and keyboard access.
Save screenshots for the required states under `ui/test-results/screenshots/`. Write
`ui-PROGRESS.md` from actual evidence with screenshot links and exact run commands.

Verification: complete UI unit/component suite, build, full Playwright smoke run, screenshot
existence/dimensions, and repository documentation checks.

## Final integration and review

The lead reviews every task diff, resolves review findings through the original implementer,
then runs a whole-branch independent review. The lead performs fresh final verification:
schema/mock validation, full UI tests, production build, E2E smoke, screenshot inspection,
repository checks, `git status`, and a requirement-by-requirement audit. No push or PR.
