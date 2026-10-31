# Phase 2 Task 3 report — dashboard generalization and recording mode

Base: `937e3635ed15e6a43366fea824debf030fee37d2`

## Implemented contract

- The browser fetches only `/active-run.json`. Load, HTTP, and schema failures remain visible and never fall back to mock data.
- The dashboard renders `source.label`, any schema-valid positive vial count, dynamic counts/grid columns, and the contract outcome, explanation text, WP6 label/status, and limit-code strings without rewriting them.
- Vial images, decision outcomes, explanations, metrics, evidence IDs, limit codes, and the optional top-down feed remain absent before their applicable cue. Nullable Lab, Delta E, and threshold values render as `Unavailable` only after reveal.
- The top-down controls preserve vial ID, slot/position, decision, Delta E, reveal state, and selection in their accessible names for variable-count sources.
- Runtime `?demo=1` applies only to development/final sources. It requests muted autoplay, catches autoplay rejection, advances reveals from the source cue timing, selects the latest landed vial, and highlights every cited evidence-ID control for a landed discard. Mock remains user-controlled at runtime and is used only through an explicit component-test prop for the discard path.
- The exact committed development source is materialized into a temporary test directory. Tests confirm its `Development data` label, 18 vials, 18/18 `insufficient_evidence` outcomes, unavailable comparisons, progressive selection, and absence of any fabricated discard.
- `ui-data/source.json` remains unchanged (`mock`).

## RED evidence

Initial focused command (repository `ui/` directory):

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vitest/vitest.mjs run --exclude e2e/** src/App.test.tsx src/components/OperatorDashboard.test.tsx
```

The default reporter emitted 1,304 lines because Testing Library included DOM dumps. The complete concise result was:

```text
Test Files  2 failed (2)
Tests  10 failed | 17 passed (27)
```

The ten behavioral failures were the `/active-run.json` loader and error copy (3), pending image privacy, 18-vial count/source label, verbatim insufficient status/nullable values, variable-count top-down linkage, discard citation highlight, muted/autoplay progressive selection, and all-abstention development progression. These were assertion failures against the pre-change behavior, not import or syntax failures.

Additional focused RED for the source explanation:

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vitest/vitest.mjs run --exclude e2e/** src/components/OperatorDashboard.test.tsx -t "reveals nullable Development values"

Test Files  1 failed (1)
Tests  1 failed | 23 skipped (24)
```

Failure: the detail did not contain the exact source text `UNKNOWN; abstain; collect evidence; limit_codes=frozen_decision_config_unavailable,matched_peer_batches_unavailable,prior_reference_unavailable`.

Self-review privacy RED:

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vitest/vitest.mjs run --exclude e2e/** src/components/OperatorDashboard.test.tsx -t "renders a feed asset"

Test Files  1 failed (1)
Tests  1 failed | 23 skipped (24)
```

Failure: the optional top-down source image reported `top-down feed connected` and was present before decision cues. The fix withholds it until all vials are revealed.

## GREEN and final verification

Focused component GREEN:

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vitest/vitest.mjs run --exclude e2e/** src/App.test.tsx src/components/OperatorDashboard.test.tsx --reporter=verbose

Test Files  2 passed (2)
Tests  27 passed (27)
Duration  1.16s (tests 59%, environment 19%, import 8%, transform 8%, setup 6%)
```

Final full Vitest:

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vitest/vitest.mjs run --exclude e2e/**

 RUN  v5.0.5 /home/hackathon5/liamreporoot/frontend-ui-20261030/ui

 Test Files  6 passed (6)
      Tests  88 passed (88)
   Start at  14:16:11
   Duration  1.08s (tests 46%, environment 33%, setup 9%, import 8%, transform 4%)
```

Typecheck (no stdout, exit 0):

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/typescript/bin/tsc --noEmit
```

Production build:

```text
/home/hackathon5/.vscode-server/cli/servers/Stable-e646868227c3e3a9e3f24277c8c7220be42c0be7/server/node node_modules/vite/bin/vite.js build

vite v8.3.5 building client environment for production...
transforming...
✓ 91 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.41 kB │ gzip:   0.28 kB
dist/assets/index-CD_JgHpX.css   11.10 kB │ gzip:   3.25 kB
dist/assets/index-CAUMffkK.js   371.40 kB │ gzip: 111.92 kB

✓ built in 136ms
```

The host did not expose `npm` on `PATH`, so the same pinned package binaries were invoked with the installed Node executable. No dependency or source configuration was changed.

## Accessibility and self-review

Self-review used the repository `review-change` workflow across behavior, interfaces, privacy, and evidence integrity.

- Preserved semantic regions, timeline articles, pressed selection state, live announcements, keyboard vial selection, dialog close/Escape behavior, tab relationships, and opener focus restoration.
- Pending evidence dialogs focus their close button; revealed dialogs focus the active tab.
- Outcomes and Delta E remain present in top-down accessible names after reveal, so color is not the only cue.
- Autoplay rejection is caught and tested with a rejected `play()` promise, preventing an unhandled rejection.
- Mock query mode remains non-autoplay with visible native controls; controls are hidden only in recording mode.
- `git diff --check` passed, and `git diff -- ui-data/source.json` was empty.

No Critical or Important findings remain in the bounded self-review. Task 4 still owns real-browser Playwright, 1920×1080 screenshot inspection, development default selection, preparation scripts, and documentation handoff. No published final source existed here, so final-mode behavior is exercised through the same validated v2 view model and source-mode branch but was not recorded against future final data.
