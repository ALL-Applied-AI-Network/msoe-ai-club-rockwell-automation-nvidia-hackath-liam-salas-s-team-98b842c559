# Tank-colour presets, evidence contracts, and Part A/Part B boundary plan

> Binding work unit: INN-58. Approved goal prompt SHA-256
> `1339f253f7eba2b696fb32711e4f7d3cb273db2a310114f4376b8bef10c7344f`.
> Implement test-first, one task and one independent review gate at a time.

## Outcome and non-negotiable rulings

Freeze the interfaces required by later work packages without changing accepted V3/V4
artifacts or weakening D16. New synthetic runs may select a named tank-colour preset;
the resolved colours and hashes remain Part A provenance. Part B accepts camera RGB and
programmed recipe/run context only.

- Preserve the existing V3/V4 builders, legacy `run_palette_id` / `run_palette_seed`
  behavior, canonical output, and hashes byte-for-byte.
- Use the approved `base` transmission values from `physx_run_variation.py`, not the
  older nominal constants in `physx_three_pour.py` or `particle_palette.py`.
- Validate every transmission channel in `(0, 1]`. Never clip zero: the renderer applies
  `-log(transmission)`.
- Keep `mask_source = oracle | predicted` in the storage/evaluation schema required by
  evidence-record-v1. The production Part B intake must reject `oracle`; oracle masks are
  restricted to segregated synthetic regression/evaluation diagnostics and never reach
  WP7 inference, history, decisions, or prompts.
- `colour_regime_id` is derived from measured camera evidence only, never a preset,
  palette, seed, label, residue, particle state, or host profile. INN-58 permits `null`
  but does not accept an arbitrary non-null caller value; WP5 must later add and test the
  measured derivation before retrieval can consume it.
- Do not modify cameras, geometry, materials, lighting, physics, fill behavior,
  qualified validators, or accepted receipts.

## Task 1 — Pure preset catalog and resolver

Files:

- add `configs/tank_colour_presets.json`
- add `src/fluidmix/tank_colour_presets.py`
- add `tests/test_tank_colour_presets.py`

RED tests first:

1. Catalog schema and named `base`, `amber_teal`, and `crimson_indigo` presets.
2. Exact base values; finite channels in `(0, 1]`; exact tank key set.
3. T100 clear lock defaults on and rejects a non-clear override instead of replacing it.
4. Per-tank overrides, finite dye-strength range `[0.5, 1.5]`, bounded deterministic
   extinction-space randomization, seed divergence, and T100 exclusion while locked.
5. Free-text provenance label is at most 200 Unicode characters and has no control
   characters.
6. Canonical JSON (`sort_keys`, compact separators, ASCII, `allow_nan=False`), resolved
   object SHA-256, renderer-palette SHA-256, and compatibility alias are stable.

Implement the smallest immutable catalog/resolver API and an adapter to the existing
renderer palette shape. Do not edit legacy `build_run_palette()` in this task.

Review gate: focused tests, neighboring `test_physx_run_variation.py`, full suite, then
an independent diff review. Fix Important-or-higher findings before Task 2.

## Task 2 — Dual legacy/new renderer binding

Files:

- modify `src/fluidmix/physx_run_variation.py`
- modify `scripts/physx_three_pour_sequence.py`
- modify `scripts/run_physx_train_residual_cohort.py`
- modify the smallest relevant renderer/supervisor tests

RED tests first:

1. Existing V3/V4 manifest fixtures, canonical hashes, and supervisor commands remain
   unchanged.
2. New-plan input resolves a named preset once, stores the complete resolved object and
   hashes, and the renderer consumes the stored object without reopening mutable config.
3. `--tank-colours` is simple entry; advanced overrides are explicit and off by default.
4. Receipt/label provenance contains the resolved preset and hashes and marks it
   `excluded_from_model_inputs`; model-input plans do not.
5. Source-bound hash mismatches, unknown presets, invalid channels, and inconsistent
   hashes fail before launch/output creation.

Do not migrate accepted V3/V4 plans or change qualified renderer behavior. Review and
verify as in Task 1. No simulator run is needed for the pure binding task; any later GPU
smoke requires a preregistered experiment and a separate output owned by the lead thread.

## Task 3 — Frozen JSON Schemas, validator APIs, and contract documentation

Files:

- add `src/fluidmix/contracts/__init__.py`
- add `src/fluidmix/contracts/evidence-record-v1.schema.json`
- add `src/fluidmix/contracts/sim-metadata-v1.schema.json`
- add `docs/contracts/evidence-record-v1.md`
- add `docs/contracts/sim-metadata-v1.md`
- add `tests/test_contract_schemas.py`
- modify the new-plan receipt writer from Task 2
- modify `requirements-dev.txt`, `docs/README.md`, and `docs/architecture.md`

Use JSON Schema Draft 2020-12 with `additionalProperties: false` at every object boundary.
Declare `jsonschema>=4.23,<5` as a development dependency. Freeze these root shapes rather
than leaving the table as an unspecified flat projection:

- `sim-metadata-v1` is one strict flat Part A record containing exactly the §4.3 Part A
  vocabulary plus required `excluded_from_model_inputs=true`: resolved preset/toggles,
  renderer palette/seed, residue/source, particle-state summary, label, and host profile.
  Case association and schema version live in the writer's external artifact-binding
  receipt/path, not new model-visible keys. Task 2's new-plan receipt writer calls
  `validate_sim_metadata_v1(...)` immediately before publication. Accepted V3/V4 receipts
  remain byte-stable and are not rewritten into this schema.
- `evidence-record-v1` is one strict flat **per-stage** Part B record containing exactly
  the serialized keys listed in §4.3—no `record_state`, wrapper, observation, registry, or
  lineage keys. A vial with 2–3 active stages has an ordered list of 2–3 records validated
  both independently and as one sequence. Every approved key is present. The validator receives a non-serialized
  phase argument (`measured`, `decided`, `explained`): later decision/explanation keys are
  required `null`/empty in earlier phases and required non-null at their owning phase.
  Missing values are never encoded as empty strings or magic hashes. Storage/regression
  validation recognizes `mask_source=oracle|predicted`; production validation rejects
  oracle. `colour_regime_id` is a required `null` until WP5 supplies a trusted measured
  binding; omission and untrusted strings fail.
- Phase transitions never mutate or overwrite a published record. The orchestrator writes
  a new no-clobber record, then an orchestration-only binding receipt containing phase,
  predecessor SHA-256 and derived SHA-256. The receipt is verified before the next stage
  but is never a pipeline/model content input or prompt field. `history_refs` is empty in
  measured records and populated only in a newly published decided record after verified
  retrieval; explanation fields are populated only in the derived explained record.

Expose explicit Python APIs in `src/fluidmix/contracts/__init__.py`:

- `validate_sim_metadata_v1(record)` and
  `verify_sim_metadata_artifacts(record, artifact_root)`;
- `validate_evidence_record_v1(record, phase=..., mode="storage"|"production",
  verified_colour_regime_binding=None)`;
- `validate_evidence_sequence_v1(records, phase=..., mode="storage"|"production",
  verified_colour_regime_bindings=None)`; and
- `verify_evidence_artifacts(record, artifact_root, predicted_mask_registry=...)`.

Schema validation owns requiredness, types, enums and cardinalities. Python validation owns
positive recipe totals, frame ordering, state-dependent nullability, and
derivation/boundary modes. Sequence validation owns the 2–3 record cardinality, shared
run/case/vial/recipe/split identity and phase, unique contiguous stage ordinals, and exact
agreement between stage tank order and the positive-volume tanks in programmed
`dispense_order`; mixed-vial/run records, duplicate/missing stages, and inactive tanks
fail before projection. Canonical in-record hashes are
recomputed from their documented canonical subobject. External clip/RGB/mask/state hashes
are syntax-checked by the schema and byte-verified by the named artifact verifier before a
record becomes trusted. Tests cover one valid fixture and one mutation per owner/rule,
including attempts to omit `colour_regime_id` or pass a preset name. The reserved
`verified_colour_regime_binding` is an opaque typed result, never serialized: it can be
constructed only by the later reviewed WP5 measured-clustering implementation and must
bind the record's measurement IDs/algorithm hash. Retrieval rejects null; INN-58 rejects
all non-null values because no such constructor exists yet.

Document the exact §4.3 evidence fields and Part A fields. Explicitly distinguish the
schema's historical/evaluation support for oracle mask provenance from the stricter Part B
input validator. Keep large raw particle/host payloads in hash-bound Part A artifacts;
the schema carries only a strict summary. Golden tests keep the two frozen legacy oracle
serializers and hashes byte-stable; they are historical regression fixtures only, are not
rewritten into the new production contract, and cannot be invoked by an active model or
evaluation runner.

Review and verify before Task 4.

## Task 4a — Trusted predicted-mask provenance and production projection

Files:

- modify `src/fluidmix/deploy_input_policy.py`
- modify `src/fluidmix/model_input_plan_builder.py`
- add `src/fluidmix/part_b_evidence.py`
- modify the predicted-locator intake and measurement adapter at the smallest existing seam
- add `tests/test_part_boundary.py`
- add golden tests for `stage_cosmos_explanation.py` and `f2_varied_explanation.py`
- modify the smallest predicted-locator/measurement neighboring tests

RED tests first:

1. Validate by exact envelope/path/allowlist, not a global leaf-name denylist. Reject every
   sim-metadata envelope/property at Part B intake and before prompt formatting while a
   positive control containing legitimate `recipe_ml.t100/t200/t400` passes.
2. Create a trusted predicted-mask registry entry only from the existing
   `rgb_mask_prediction` locator validator. It binds image ID, source RGB/hash, mask/hash,
   frozen model/config/code hashes, and validation-threshold source. Measurement consumes
   that trusted entry and retains its registry/evidence binding; a caller cannot establish
   provenance merely by writing `mask_source="predicted"`.
3. Reject oracle, missing, unusable, hash-mismatched, and relabelled-oracle records before
   production projection. The outcome is `insufficient_evidence`, never oracle fallback.
4. Add a versioned `project_part_b_model_evidence_v1(...)` for the new production path.
   It first calls the sequence validator, then accepts only schema-valid production records
   plus trusted locator/history bindings,
   emits an ordered JSON array of flat per-stage records using only the exact §4.3
   camera/program-derived keys, and revalidates every record immediately before prompt
   serialization. Registry and artifact-binding receipts remain external verifier inputs.
5. Do not route the frozen legacy diagnostic serializers through the new projector.
   Preserve their bytes/hashes with pure golden serialization tests, classify them as
   historical/regression-only, and prohibit all active model, evaluation, and WP7 entry
   points from importing or invoking them.
6. `colour_regime_id` is present and exactly `null` until WP5 supplies the separately
   reviewed typed measured-derivation binding; omission, preset/hash aliases, and arbitrary
   strings fail. Retrieval continues to reject null.

Keep evaluation entry points separate; they may join labels only after saved inference.
Review and verify this runtime/provenance gate before Task 4b.

## Task 4b — Authoritative ownership manifest and dependency/read isolation

Files:

- add `src/fluidmix/part_boundary_manifest.py`
- extend `tests/test_part_boundary.py`
- modify `docs/architecture.md` and the relevant script/module index

The manifest explicitly classifies every current Part A-only module, production Part B
root, evaluation-only module, legacy oracle diagnostic, and production/evaluation script
entry point. The initial reviewed inventory must include measurement, capture adapters,
triage, stage decision, mask prediction, model-input, history/explanation entry points and
their scripts—not only the four files changed in Task 4a. New production roots must be
registered before their tests pass.

RED tests first:

1. Walk transitive local imports from every registered production Part B root and active
   evaluation/model script. Fail any edge into Part A-only or legacy-oracle modules. Parse
   ordinary imports plus constant `importlib.import_module` / `__import__` calls, then add
   a fresh-interpreter import smoke with a forbidden-module finder.
2. Fresh-process read guards reject access to the exact preset catalog, sim-metadata
   fixtures, and Part A receipt/label roots from every production entry point. Patch exact
   filesystem targets, not words in prose; legitimate RGB/recipe inputs pass.
3. Assert scoring-only evaluation roots may join Part A labels only after a sealed
   inference artifact. Active model/evaluation and production roots cannot invoke legacy
   oracle serializers; production roots also cannot call the post-inference join.
4. Assert WP7's future root list must be registered and may use the new production
   projector, never either frozen diagnostic serializer.

Review and verify this isolation gate independently before acceptance.

## Task 5 — Acceptance and handoff

Run fresh, uncached verification from the lead worktree:

```bash
PY=/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider \
  tests/test_tank_colour_presets.py tests/test_contract_schemas.py \
  tests/test_part_boundary.py tests/test_physx_run_variation.py \
  tests/test_physx_varied_cohort_manifest_v3.py \
  tests/test_physx_varied_cohort_manifest_v4.py \
  tests/test_physx_varied_cohort_supervisor.py \
  tests/test_model_input_plan_builder.py \
  tests/test_stage_cosmos_explanation.py tests/test_f2_varied_explanation.py -q
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider tests -q
"$PY" scripts/smoke_absorbance.py
"$PY" scripts/check_agent_docs.py
"$PY" scripts/fit_absorbance_video5.py
git diff --check
git status --short
```

The diagnostic fit's successful exit is not a physical-validity gate; record any negative
absorbance warning. Perform an independent review of committed, staged, unstaged, and
untracked scope. Record verification, limits, hashes, decisions, spend, and the resume line
in `outputs/pipeline-20261028/PROGRESS.md`. Only then close INN-58 with one evidence comment.

## Alternatives and why this plan is bounded

- **Rejected:** replace legacy palettes everywhere with named presets. It would invalidate
  accepted V3/V4 hash bindings and exceed the allowed change.
- **Rejected:** permit oracle masks because the storage schema names them. This violates
  the controlling D16 input boundary and makes the real pipeline non-deployable.
- **Chosen:** a dual legacy/new Part A entry path plus a stricter Part B intake. It preserves
  accepted evidence, supports new synthetic runs, and makes the product boundary testable.
