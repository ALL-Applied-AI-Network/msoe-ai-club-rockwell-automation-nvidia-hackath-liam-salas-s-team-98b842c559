# Phase 1 UI data

`example.json` is documented synthetic twin data for the operator-demo UI. It is not a
plant measurement, physical calibration, or operating recommendation. It demonstrates the
complete contract with nine ordered vial slots, local run-video cues, recipes and colour
evidence, paired local RGB-frame and predicted-mask references, same-recipe history,
top-down state, and the limited WP6 all-abstention benchmark.

`schema.json` defines the wire structure. The TypeScript adapter additionally validates
cross-record links and maps the normalized records into page-facing playback, vial,
history, evidence, top-down, and WP6 view models so components do not perform joins.
