# Task 2 implementer report

## Outcome and boundaries

Implemented a deterministic builder at `scripts/build_development_fixture.py` and committed its
minimized output under `ui-data/development/`. The fixture is labelled exactly `Development data`.
It preserves the six source runs, 18 vials, 45 source stages, V3+V4 split versions, all 18 published
`insufficient_evidence` decisions, and the exact bytes of the approved development record file.

The source database is opened only as `mode=ro&immutable=1`, with `PRAGMA query_only=ON` and a
SQLite authorizer that denies reads or writes involving `sim_metadata`. The builder uses explicit
column lists for each copied table. It creates a new schema-v3 database and never uses SQLite
backup or byte-copy on the source database. The derived `sim_metadata` table exists for schema-v3
compatibility and has zero rows. The source `sim_metadata` contents were not queried or copied.

The only control/evidence inputs are the approved `evidence.sqlite3` and sibling
`explanation-records.json`. RGB and mask dependencies are selected exclusively through image and
artifact rows in that database, resolved without directory traversal, and copied only after their
recorded SHA-256 and byte size match. All 45 stages are joined by the reviewed materializer, so its
45 RGB and 45 mask files are required presentation dependencies. No fill clips or upstream videos
are copied.

`development-records.json` is an exact byte copy of the hash-verified published development
records. The development-only materializer consumes this shape directly and preserves its decision
IDs, dispositions, limit codes, guard text, model revision, run/vial association, and history
references. Unpublished Delta E, threshold, and reference Lab remain unavailable. No
`wp7-final-decision-v1` artifact is synthesized; final-directory mode remains strict.

The 269-second H.264 reel is based only on the first source-backed target RGB frame for each vial,
at an intended 15 seconds per vial. The one-frame-per-second encoding cadence quantizes the MP4
container duration by one second; the normalized UI presentation timeline remains 270 seconds.

## TDD evidence

Public boundary: the existing `materialize()` function with `{"source":"development"}`. This
proves the committed source can pass the reviewed joins and schema-v2 validation. It cannot prove
physical accuracy, real-fluid behavior, or final WP7 publication.

### RED

Command:

```text
.venv/bin/python -m pytest tests/test_development_fixture.py -q
```

Output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_committed_development_fixture_materializes_as_schema_v2 _________

tmp_path = PosixPath('/tmp/pytest-of-hackathon5/pytest-4649/test_committed_development_fix0')

    def test_committed_development_fixture_materializes_as_schema_v2(tmp_path: Path) -> None:
        repository = Path(__file__).parents[1]
        output = tmp_path / "ui-data"
        output.mkdir()
        config = output / "source.json"
        config.write_text('{"source":"development"}', encoding="utf-8")

>       destination = materialize(
            repository_root=repository,
            config_path=config,
            output_dir=output,
        )

tests/test_development_fixture.py:14:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
scripts/materialize_ui_data.py:1147: in materialize
    mode, source = resolve_source(config_path, repository_root=repository_root)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

config_path = PosixPath('/tmp/pytest-of-hackathon5/pytest-4649/test_committed_development_fix0/ui-data/source.json')
repository_root = PosixPath('/home/hackathon5/liamreporoot/frontend-ui-20261030')

    def resolve_source(config_path: Path, *, repository_root: Path) -> tuple[str, Path | None]:
        """Resolve the one-value source selector without opening source evidence."""

        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MaterializationError(f"cannot read source selector: {error}") from error
        if (
            not isinstance(value, dict)
            or set(value) != {"source"}
            or not isinstance(value.get("source"), str)
            or not value["source"].strip()
        ):
            raise MaterializationError(
                "source.json must contain exactly one non-empty string value named source"
            )
        source = value["source"]
        if source == "mock":
            return ("mock", None)
        if source == "development":
            development = repository_root.resolve() / "ui-data" / "development"
            if not development.is_dir():
>               raise MaterializationError(
                    f"development source directory does not exist: {development}"
                )
E               scripts.materialize_ui_data.MaterializationError: development source directory does not exist: /home/hackathon5/liamreporoot/frontend-ui-20261030/ui-data/development

scripts/materialize_ui_data.py:46: MaterializationError
=========================== short test summary info ============================
FAILED tests/test_development_fixture.py::test_committed_development_fixture_materializes_as_schema_v2
1 failed in 0.11s
```

### GREEN

Focused fixture and Task 1 adapter regression command:

```text
.venv/bin/python -m pytest tests/test_ui_materializer.py tests/test_development_fixture.py -q
```

Output:

```text
.............................                                            [100%]
29 passed in 4.89s
```

The fixture tests cover two fresh rebuild roots and identical whole-tree hashes, approved source
hashes before and after, six runs and 18 vials, V3+V4, 18/18 abstention, 90 individual asset hashes
and sizes, schema version 3, zero `sim_metadata` rows, SQLite `quick_check`, complete media decode,
approximately 4.5-minute duration, and schema-v2 materialization.

## Verification evidence

Complete Python suite and required smoke/doc checks:

```text
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/smoke_absorbance.py && .venv/bin/python scripts/check_agent_docs.py
```

Output:

```text
........................................................................ [ 67%]
..................................                                       [100%]
106 passed, 1 skipped in 8.33s
old-fluid fraction 0.150000; RMS absorbance residual 8.012e-18
synthetic smoke passed
shared pointer, 11 skills and 36 document file-target checks passed
```

The first full-suite attempt stopped at collection because the clean environment lacked `cv2`:

```text
E   ModuleNotFoundError: No module named 'cv2'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 skipped, 1 error in 0.11s
```

I installed `opencv-python-headless` into the ignored local `.venv` and reran the suite above.

Final syntax/database/provenance check command:

```text
.venv/bin/python -m py_compile scripts/build_development_fixture.py scripts/materialize_ui_data.py tests/test_development_fixture.py && .venv/bin/python -c '<database/provenance assertions>' && sha256sum <approved inputs and primary derived files> && stat -c '%s %n' <primary derived files> && git diff --check
```

Output:

```text
quick_check ok
counts 6 18 45 0
decisions 18 {'insufficient_evidence'}
source_ids ['RUN-TRAIN-01', 'RUN-TRAIN-02', 'RUN-TRAIN-03', 'RUN-TRAIN-04', 'RUN-V4-TRAIN-01', 'RUN-VALIDATION-01']
split_versions ['whole-run-varied-v3', 'whole-run-varied-v4']
derived_files 94 12130895
88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7  /home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-development-history-preparation-01a18128-attempt01/evidence.sqlite3
c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9  /home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-development-history-preparation-01a18128-attempt01/explanation-records.json
f989bf2fe2b31cc04b6984f58f46254a97631e09c9c0a7b1ebdacba240282552  ui-data/development/evidence.sqlite3
40e205939597232ac867be78c6975513daa526fa503257d929217f51d8508d1c  ui-data/development/wp7-final-decisions.json
c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9  ui-data/development/development-records.json
ec609e6a200991921a25784646aa0b0bad0c4b9c7d1c3680a391f5e247b3b07f  ui-data/development/development-reel.mp4
c96f6665ab8f36f209071908c8c6fc14c376c5b933234cd2be5fac35f09bbd73  ui-data/development/manifest.json
389120 ui-data/development/evidence.sqlite3
211693 ui-data/development/wp7-final-decisions.json
18165 ui-data/development/development-records.json
889067 ui-data/development/development-reel.mp4
26366 ui-data/development/manifest.json
```

The 90 individual RGB/mask hashes and sizes, plus all other allowed derived files, are recorded in
`ui-data/development/manifest.json`. The derived files listed there total 12,130,895 bytes; the
directory occupies about 13 MiB on disk.

UI `npm test`, `npm run typecheck`, and `npm run build` were attempted but could not start because
this environment has no Node/npm executable:

```text
/bin/bash: line 1: npm: command not found
```

No UI success is claimed. Task 2 does not modify TypeScript, and the Python schema materializer
validation is covered above.

## Materializer compatibility change

Final mode retains Task 1's exact WP7 record validation and strict single-run invariant.
Development mode has a separate exact published-development-record adapter. Its normalized
`run_id` is the explicitly inferred presentation ID
`development-v3-v4-six-run-presentation`; the six source run IDs remain in the fixture database
and manifest. Development cues are 15 seconds per vial and normalized duration is 270 seconds.

## Self-review

- Confirmed no code discovers or traverses the approved source directory. The builder opens only
  the two named inputs and DB-addressed assets under the fixed repository artifact root.
- Confirmed no code or command opened the sealed preflight, WP1 test, video 5, grounding
  qualification, oracle data, or simulator truth.
- Confirmed source SQLite access is immutable/read-only and `sim_metadata` is denied by authorizer.
- Confirmed fixture SQLite is rebuilt from explicit safe tables and contains zero `sim_metadata`
  rows; no source database backup/copy operation exists.
- Confirmed no keep/discard decision was synthesized. All normalized outcomes remain the source
  `insufficient_evidence` disposition.
- Corrected two issues found during implementation: relative FFmpeg concat paths and a manifest
  statement that initially called a value-preserving JSON serialization “verbatim.” The final
  builder resolves the output root and byte-copies the approved record file.
- `ui-data/source.json` remains `{"source":"mock"}` for Task 4 to change.
- `git diff --check` passed. No push or PR was performed.

## Fix Round 1

Controller ruling: development mode must consume the exact published development-record shape
directly. It must not manufacture a `wp7-final-decision-v1` translation, and final-directory mode
must remain unchanged and strict.

### RED

Command:

```text
.venv/bin/python -m pytest tests/test_development_fixture.py -q
```

Output:

```text
F...F.                                                                   [100%]
=================================== FAILURES ===================================
_________ test_committed_development_fixture_materializes_as_schema_v2 _________

tmp_path = PosixPath('/tmp/pytest-of-hackathon5/pytest-4693/test_committed_development_fix0')

    def test_committed_development_fixture_materializes_as_schema_v2(tmp_path: Path) -> None:
        repository = Path(__file__).parents[1]
        output = tmp_path / "ui-data"
        output.mkdir()
        config = output / "source.json"
        config.write_text('{"source":"development"}', encoding="utf-8")

        destination = materialize(
            repository_root=repository,
            config_path=config,
            output_dir=output,
        )

        normalized = json.loads(destination.read_text(encoding="utf-8"))
        assert normalized["schema_version"] == "2.0.0"
        assert normalized["source"] == {
            "mode": "development",
            "origin": "synthetic_development",
            "label": "Development data",
        }
        assert len(normalized["vials"]) == 18
        assert {vial["decision"]["outcome"] for vial in normalized["vials"]} == {
            "insufficient_evidence"
        }
        assert normalized["run"]["video"]["duration_seconds"]["value"] == 270.0

        source_records = json.loads(
            (repository / "ui-data" / "development" / "development-records.json").read_text()
        )
        database = sqlite3.connect(repository / "ui-data" / "development" / "evidence.sqlite3")
        try:
            associations = {
                (run_id, ordinal): (vial_id, recipe_id)
                for vial_id, run_id, ordinal, recipe_id in database.execute(
                    "SELECT vial_id,run_id,vial_ordinal,recipe_id FROM vials"
                )
            }
        finally:
            database.close()
        normalized_by_id = {
            vial["decision"]["record_id"]: vial for vial in normalized["vials"]
        }
        evidence_by_id = {item["evidence_id"]: item for item in normalized["evidence"]}
        assert set(normalized_by_id) == {record["decision_id"] for record in source_records}
        for record in source_records:
            vial = normalized_by_id[record["decision_id"]]
            decision = vial["decision"]
            assert (vial["vial_id"], vial["recipe_id"]) == associations[
                (record["run_id"], record["vial_ordinal"])
            ]
            assert decision["outcome"] == record["disposition"]
>           assert decision["limit_codes"] == record["limit_codes"]
E           AssertionError: assert ['unsupported...ent_selector'] == ['frozen_deci..._unavailable']
E
E             At index 0 diff: 'unsupported_development_selector' != 'frozen_decision_config_unavailable'
E             Right contains 2 more items, first extra item: 'matched_peer_batches_unavailable'
E             Use -v to get more diff

tests/test_development_fixture.py:87: AssertionError
________ test_builder_rejects_hash_matching_inputs_at_unapproved_paths _________

tmp_path = PosixPath('/tmp/pytest-of-hackathon5/pytest-4693/test_builder_rejects_hash_matc0')

    def test_builder_rejects_hash_matching_inputs_at_unapproved_paths(tmp_path: Path) -> None:
        repository = Path(__file__).parents[1]
        ffmpeg = _ffmpeg(repository)
        copied = tmp_path / "copied-inputs"
        copied.mkdir()
        database = copied / "evidence.sqlite3"
        records = copied / "explanation-records.json"
        shutil.copyfile(DEFAULT_SOURCE, database)
        shutil.copyfile(DEFAULT_SOURCE.with_name("explanation-records.json"), records)

>       with pytest.raises(FixtureBuildError, match="exact approved source path"):
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'exact approved source path'
E         Actual message: 'artifact is missing or a symlink: outputs/pipeline-20261028/wp5-cnn-swap-materialization-d89a3f9-01a18128-attempt02/masks/BATCH_5a4a586e3ec3b085-V2-dispense-2-t400.png'

tests/test_development_fixture.py:204: AssertionError
=========================== short test summary info ============================
FAILED tests/test_development_fixture.py::test_committed_development_fixture_materializes_as_schema_v2
FAILED tests/test_development_fixture.py::test_builder_rejects_hash_matching_inputs_at_unapproved_paths
2 failed, 4 passed in 2.33s
```

The first failure proves the semantic regression: the UI exposed an invented limit code instead
of the source values. The second proves the builder did not reject hash-identical inputs at an
unapproved path before trying to resolve assets.

### Implementation

- Removed `translate_records()` and removed the generated `wp7-final-decisions.json`.
- Added an exact development-record validator and development-only materializer path. It verifies
  each source record against every matching SQLite stage decision/explanation row and verifies each
  history reference against `history_members`.
- Mapped exact source decision ID, disposition, limit codes, guard result, explanation ID, model
  revision, status/action, prompt hash, history citations, and run/vial identity into schema v2.
  Delta E, threshold, and reference Lab are null because the published records do not provide them.
- Restored final mode to its original strict single-run `_validate_record()` path. A public
  materialization test proves raw development records are rejected as invalid final records.
- The CLI now accepts only `--output` and `--ffmpeg`. The library boundary rejects any resolved
  database/record paths other than the exact approved paths before hashing or asset access.

### GREEN

Focused builder/materializer command:

```text
.venv/bin/python -m pytest tests/test_ui_materializer.py tests/test_development_fixture.py -q
```

Output:

```text
...............................                                          [100%]
31 passed in 4.98s
```

Complete Python/smoke/docs command:

```text
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/smoke_absorbance.py && .venv/bin/python scripts/check_agent_docs.py
```

Output:

```text
........................................................................ [ 66%]
....................................                                     [100%]
108 passed, 1 skipped in 7.54s
old-fluid fraction 0.150000; RMS absorbance residual 8.012e-18
synthetic smoke passed
shared pointer, 11 skills and 36 document file-target checks passed
```

UI command:

```text
PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm test && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run typecheck && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run build
```

Output:

```text
> fluidmix-operator-demo@0.0.0 test
> vitest run --exclude e2e/**


 RUN  v5.0.5 /home/hackathon5/liamreporoot/frontend-ui-20261030/ui


 Test Files  6 passed (6)
      Tests  81 passed (81)
   Start at  13:53:06
   Duration  1.98s (tests 34%, environment 30%, import 17%, setup 10%, transform 9%, worker 1%)


> fluidmix-operator-demo@0.0.0 typecheck
> tsc --noEmit


> fluidmix-operator-demo@0.0.0 build
> tsc --noEmit && vite build

vite v8.3.5 building client environment for production...
transforming...
✓ 91 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.41 kB │ gzip:   0.27 kB
dist/assets/index-DkLFkvKW.css   10.58 kB │ gzip:   3.14 kB
dist/assets/index-Cgy6HHgB.js   370.16 kB │ gzip: 111.65 kB

✓ built in 323ms
```

### Corrected artifact evidence

```text
fake final artifact absent
f989bf2fe2b31cc04b6984f58f46254a97631e09c9c0a7b1ebdacba240282552  ui-data/development/evidence.sqlite3
c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9  ui-data/development/development-records.json
ec609e6a200991921a25784646aa0b0bad0c4b9c7d1c3680a391f5e247b3b07f  ui-data/development/development-reel.mp4
b41258bd866a351a4e8dcecee18d357cd220bfb9ac51e1e3ffd44af3f26b630b  ui-data/development/manifest.json
389120 ui-data/development/evidence.sqlite3
18165 ui-data/development/development-records.json
889067 ui-data/development/development-reel.mp4
26076 ui-data/development/manifest.json
quick_check ok
counts 6 18 45 0
derived_files 93 11919202
```

The source input hashes remain the approved
`88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7` and
`c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9`.

### Fix-round self-review

- Searched the corrected builder and fixture: no final-format development artifact or
  `unsupported_development_selector` remains.
- Confirmed the exact source bytes are retained and record-by-record tests compare all 18 source
  records with normalized results and SQLite associations.
- Confirmed final-mode tests still exercise keep/discard/insufficient exact final records and now
  explicitly reject the development record shape.
- Confirmed source database access and the zero-row `sim_metadata` boundary are unchanged.
- Confirmed `ui-data/source.json` remains mock for Task 4.
- No push or PR was performed.
