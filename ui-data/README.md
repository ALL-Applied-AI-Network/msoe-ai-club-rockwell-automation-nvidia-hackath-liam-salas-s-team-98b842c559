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
