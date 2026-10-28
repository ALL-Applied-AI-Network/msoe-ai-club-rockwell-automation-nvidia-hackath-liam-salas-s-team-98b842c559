# `sim-metadata-v1`

`sim-metadata-v1` is the frozen per-case Part A record. Every field is simulation truth or
runtime provenance and is marked `excluded_from_model_inputs=true`. Part B must not import,
read, query, project, or prompt with this record; Task 4b adds transitive import/read
enforcement. A scoring-only evaluation may join its label after inference through the
separately verified case binding.

The JSON Schema is
[`src/fluidmix/contracts/sim-metadata-v1.schema.json`](../../src/fluidmix/contracts/sim-metadata-v1.schema.json).
It uses JSON Schema Draft 2020-12 and closes every object against extra fields.

## Exact root

| Field | Exact meaning |
|---|---|
| `tank_colour_preset` | Named Part A preset: `base`, `amber_teal`, or `crimson_indigo`. |
| `resolved_tank_transmission_1cm` | Exact T100/T200/T400 linear-RGB transmission vectors, each channel in `(0,1]`. |
| `advanced_toggles` | Exact override subset, dye scale, clear lock, randomization flag/seed, frozen jitter bounds, derived tank/channel jitter, and optional provenance label. |
| `tank_colour_preset_hash` | Task 1 canonical semantic-record SHA-256. |
| `renderer_palette` | Exact reference depth, transmission map, renderer hash, and compatibility hash alias. |
| `renderer_seed` | Nonnegative render seed. |
| `residue_ml` | Prescribed leading-residual mL per active dispense, not a vial total or a measured plant value. |
| `residue_source_liquid` | Previous recipe plus ordered per-dispense `{stage_ordinal, source_tank_id, composition_t200_t400_t100}` rows. |
| `particle_state_summary` | Relative manifest path/hash/byte count, particle count, finite flag, and required non-null resolution profile ID. Raw state stays external. |
| `label` | Exact `{disposition, rule_id}` synthetic oracle. |
| `host_profile` | Profile ID, local/Brev kind, GPU model/count, Isaac/Kit versions, and relative profile path/hash/bytes. It contains no hostname or credential. |
| `excluded_from_model_inputs` | Always `true`. |

The validator reconstructs the complete Task 1 resolution in memory, without reopening
the mutable catalog, and passes it through `validate_tank_colour_resolution`. That function
independently replays the fixed preset values, override/clear-lock rules, extinction-space
scale and seeded jitter, nine-decimal rounding, canonical JSON, and all three hashes.
Resolved and renderer transmissions must match; both palette hashes are the canonical
transmission-map hash.

Positive residue has two or three contiguous source rows. The first source is the previous
final tank and later sources are the preceding programmed dispense tanks; compositions are
one-hot in T200/T400/T100 order. Zero residue has no previous source and no source rows.
The label rule is exactly `prescribed_synthetic_latent_oracle`: disposition is `keep` iff
residue is zero and `discard` otherwise. This is a prescribed synthetic label, not evidence
that a real line has that residue.

## External bindings

Case identity and schema identity are deliberately not model-visible root fields. The
orchestrator binds exact metadata bytes with this four-field receipt:

```text
schema_version = sim-metadata-binding-v1
case_id, sim_metadata_path, sim_metadata_sha256
```

`sim_metadata_path` is contained relative POSIX. Before publication, ingest, or scoring,
`validate_sim_metadata_binding_v1` checks the expected case, exact bytes, schema, and the
external particle/host summaries through stable no-follow reads. Both external files must
be strict JSON: the manifest repeats particle count, finite status, and resolution profile;
the host file repeats profile ID, compute kind, GPU model/count, and Isaac/Kit versions.
Those values must equal the metadata summaries in addition to their bytes and hashes.

`validate_sim_metadata_runtime_schedule_v1` accepts only
`physx_f2_runtime_leading_residual_schedule`, never the non-runtime base schedule. It
cross-checks the per-dispense residue amount, source tanks, one-hot compositions, previous
recipe, and canonical schedule hash. The schedule must carry a same-case binding with the
same residue. Its contextual previous recipe must match the model-input record; for positive
residue it must also match the source-liquid predecessor. The same-case finalized label row
must agree on residue, disposition, `label_source`, schedule hash, and both model/deploy
exclusion flags.

For a run's approved clean-start first vial, case and model predecessor recipe IDs are both
null and the model has no previous final tank. The physical schedule may retain the
mechanically unused tank predecessor used to build zero-particle segments; it is not
serialized as source liquid. Non-clean-start context binds the model and physical previous
final tank, and positive residue additionally binds that tank to the first source row.

Plan-time colour resolution is not a complete `sim-metadata-v1` record. Residue, final
particle evidence, label, and host evidence must exist before runtime publication; the
pipeline must not invent placeholders. Large particle and host payloads stay external and
hash-bound.
