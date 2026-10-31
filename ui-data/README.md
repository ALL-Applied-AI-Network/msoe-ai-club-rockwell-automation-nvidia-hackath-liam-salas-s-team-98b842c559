# Phase 1 UI data

`example.json` is documented synthetic twin data for the operator-demo UI. It is not a
plant measurement, physical calibration, or operating recommendation. It demonstrates the
complete contract with nine ordered vial slots, local run-video cues, recipes and colour
evidence, paired local RGB-frame and predicted-mask references, same-recipe history,
top-down state, and the limited WP6 all-abstention benchmark.

From a clean checkout, install the UI dependencies and run the committed-data/media checks:

    cd ui
    npm ci
    npm test

`ffmpeg-static` is a declared development dependency and supplies the platform-specific
ffmpeg executable used by the committed-media test. `FLUIDMIX_FFMPEG=/path/to/ffmpeg` is
an optional explicit override; an invalid override or missing dependency fails with a setup
message. Tests do not search `/tmp`, home directories, or architecture-specific host paths.

`schema.json` defines the wire structure. The TypeScript adapter additionally validates
cross-record links and maps the normalized records into page-facing playback, vial,
history, evidence, top-down, and WP6 view models so components do not perform joins.

## Phase 2 source boundary

The browser consumes only the materialized `ui-data/active-run.json`. The Python adapter
reads the published development SQLite and development-records fixture through a read-only,
immutable SQLite connection; its authorizer denies `sim_metadata`. It does not instantiate
the writable WP5 adapter and does not discover sealed, WP1, video-5, grounding, oracle, or
simulator-truth inputs. The final WP7 adapter is read-only and accepts only an approved
`wp7-final-decision-v1` source directory; no published final decision file exists here.

The one-line final switch is the path string in [`source.json`](source.json):
`{"source":"development"}` for this checked-in source, or an approved final-source
directory path when final WP7 output is published. Mock remains a component/contract-test
input only and is never used for the demo or screenshots. Preparation is centralized in
`scripts/materialize_ui_data.py`: from `ui/`, `npm run prepare:data` writes the validated
`active-run.json`; `npm run dev`, `npm run build`, `npm run preview`, and the Playwright
web server call that same preparation command. Runtime code never edits source fixtures.

The checked-in Development data fixture is derived from the hash-pinned V3/V4 six-run,
18-vial source and its approved explanation records. Its source-backed reel is 270 seconds
(a presentation artifact, not plant timing); E2E seeks to cue boundaries instead of waiting.
All 18 decisions are the measured development abstention records: `insufficient_evidence`.
No keep/discard is invented. Lab reference, Delta E, and threshold values remain unavailable
where the contract says they are unavailable. The reel, fixture, and manifest are derived
artifacts, not final WP7 output; final behavior remains unverified until an approved final
decision artifact exists.
