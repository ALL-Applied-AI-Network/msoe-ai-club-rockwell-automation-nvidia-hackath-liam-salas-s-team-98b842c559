# Import and purity rules

The onion boundaries are executable in `.importlinter`; inner-ring I/O purity is
checked by `tests/architecture/test_inner_ring_purity.py`. Import-linter 2.x requires
multi-value INI fields on separate lines, so the supplied three package names and
module names are represented as equivalent multiline values.

The structure specification also explicitly places the strict local-file snapshot
helpers in `fluidmix.core.io`. That module is the sole named exception to the generic
inner-ring `open()` scan. Its surface is limited to `hash_regular_file` and
`load_strict_json_snapshot`, and parity with the frozen twin implementations is checked
separately. All other core, colour, perception, measurement, and decision modules remain
inside the purity scan.

Run the checks from the repository root in the CPU environment:

```text
PYTHONPATH=src .venv/bin/lint-imports --config .importlinter --no-cache
PYTHONPATH=src .venv/bin/python -m pytest tests/architecture/test_inner_ring_purity.py -q
```

## Measured pre-fix result

Measured on 2026-10-31 immediately after move commit `a391c73`:

- Import graph: 164 files and 238 dependencies; one contract kept and two broken.
- `nothing imports evaluation`: kept.
- `product rings only import inward`: broken by 10 direct import edges.
- `twin and product share no code`: broken by 32 direct import edges.
- Inner-ring purity: 24 modules passed and 10 modules failed, with 36 findings.

These are source observations, not fixed behavior. They are intentionally recorded
before the cause-specific repair commits.

### Product ring violations

1. `fluidmix.explanation.vlm.model_input_plan_builder` imports `fluidmix.storage.artifacts.capture_manifest_adapter` (line 18).
2. `fluidmix.memory.history.history` imports `fluidmix.storage.sqlite.evidence_store` (line 16).
3. `fluidmix.memory.history.sixth_development_run` imports `fluidmix.storage.sqlite.evidence_store` (line 620).
4. `fluidmix.storage.artifacts.part_b_evidence` imports `fluidmix.pipeline.rgb_runtime` (line 16).
5. `fluidmix.decision.vial` imports `fluidmix.memory.history.development_reference` (line 16).
6. `fluidmix.measurement.vision_measurement` imports `fluidmix.memory.history.vision_dataset_plan` (line 746).
7. `fluidmix.measurement.varied_stage_measurement` imports `fluidmix.storage.artifacts.capture_manifest_adapter` (line 14).
8. `fluidmix.measurement.varied_stage_measurement` imports `fluidmix.perception.masks` (line 19).
9. `fluidmix.measurement.vision_measurement` imports `fluidmix.perception.masks` (lines 113, 449, and 910).
10. `fluidmix.measurement.vision_measurement` imports `fluidmix.pipeline.rgb_runtime` (lines 114 and 517).

### Product-to-twin violations

1. `fluidmix.memory.reference_bank.bank` imports `twin.cohorts.one_run_manifest_v1` (line 16).
2. `fluidmix.memory.reference_bank.bank` imports `twin.physx.run_variation` (line 17).
3. `fluidmix.explanation.vlm.model_input_plan_builder` imports `twin.render.render_process` (line 23).
4. `fluidmix.measurement.validation_measurement_binding` imports `twin.render.twin_video` (line 18).
5. `fluidmix.measurement.validation_measurement_binding` imports `twin.render.brev_video_cohort` (line 11).
6. `fluidmix.storage.artifacts.capture_manifest_adapter` imports `twin.render.render_process` (line 11).
7. `fluidmix.measurement.varied_stage_measurement` imports `twin.render.render_process` (line 23).
8. `fluidmix.measurement.validation_measurement_binding` imports `twin.render.render_process` (line 12).
9. `fluidmix.measurement.validation_stage_measurement` imports `twin.render.render_process` (line 11).
10. `fluidmix.core.contracts.sim_metadata` imports `twin.scene.tank_colour_presets` (line 142).

### Twin-to-product violations

1. `twin.cohorts.varied_image_locator` imports `fluidmix.measurement.vision_measurement` (line 15).
2. `twin.physx.residual_pilot` imports `fluidmix.core.paths` (line 15).
3. `twin.cohorts.full_transition_manifest` imports `fluidmix.core.contracts._common` (line 25).
4. `twin.cohorts.full_transition_manifest` imports `fluidmix.core.paths` (line 26).
5. `twin.cohorts.varied_image_locator` imports `fluidmix.storage.artifacts.capture_manifest_adapter` (line 10).
6. `twin.cohorts.test_recipe_generator` imports `fluidmix.colour.absorbance` (line 22).
7. `twin.postrender.run_sequence_dataset` imports `fluidmix.core.policy` (line 13).
8. `twin.cohorts.train_residual_manifest_v2` imports `fluidmix.core.paths` (line 19).
9. `twin.physx.case_plan` imports `fluidmix.core.paths` (line 13).
10. `twin.probes.pilot_dataset` imports `fluidmix.core.paths` (line 25).
11. `twin.render.palette_cut_reel` imports `fluidmix.storage.artifacts.part_boundary_manifest` (line 25).
12. `twin.cohorts.stage_cohort_materializer` imports `fluidmix.core.policy` (line 18).
13. `twin.cohorts.receipt_manifest_publisher` imports `fluidmix.core.policy` (line 18).
14. `twin.cohorts.varied_cohort_manifest_v3` imports `fluidmix.core.paths` (line 17).
15. `twin.cohorts.full_transition_manifest` imports `fluidmix.colour.absorbance` (line 24).
16. `twin.cohorts.receipt_manifest_publisher` imports `fluidmix.storage.artifacts.capture_manifest_adapter` (line 1049).
17. `twin.physx.composition_cohort` imports `fluidmix.core.paths` (line 16).
18. `twin.cohorts.varied_stage_materializer` imports `fluidmix.core.policy` (line 17).
19. `twin.cohorts.varied_stage_materializer` imports `fluidmix.core.paths` (line 27).
20. `twin.cohorts.varied_cohort_manifest_v4` imports `fluidmix.core.paths` (line 17).
21. `twin.probes.pilot_diagnostic` imports `fluidmix.core.paths` (line 18).
22. `twin.postrender.wp1_postrender` imports `fluidmix.measurement.vision_measurement` (line 34).

### Inner-ring purity violations

- `fluidmix/core/contracts/_common.py`: calls `open()` on lines 28, 36, 178, 188, 245, 255, 337, and 349.
- `fluidmix/perception/baselines/spatial_rgb_mask_predictor.py`: imports `subprocess` on line 20; calls `open()` on lines 374, 388, 802, 1103, and 1190.
- `fluidmix/perception/baselines/temporal_rgb_mask_baseline.py`: imports `subprocess` on line 20; calls `open()` on line 289.
- `fluidmix/perception/cnn/cnn_training.py`: calls `open()` on lines 89, 134, 135, and 138.
- `fluidmix/perception/materializer.py`: imports `subprocess` on line 16; calls `open()` on lines 95, 139, and 149.
- `fluidmix/measurement/cap_delay.py`: calls `open()` on line 21.
- `fluidmix/measurement/fill_clips.py`: calls `open()` on lines 83 and 485.
- `fluidmix/measurement/real_photometric_range.py`: calls `open()` on lines 42 and 241.
- `fluidmix/measurement/varied_stage_measurement.py`: calls `open()` on lines 57, 62, and 289.
- `fluidmix/measurement/vision_measurement.py`: calls `open()` on lines 382, 387, 1157, and 1173.
