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
records. `wp7-final-decisions.json` is a documented deterministic translation for the Task 1 UI
contract. It preserves decision IDs, dispositions, target identities, measured Lab values, target
asset hashes, and citations derived from the schema-v3 evidence. It does not claim to be a
published final WP7 run. Historical fields that the final contract cannot represent remain intact
in `development-records.json`; the manifest documents this boundary.

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

Final mode retains Task 1's strict single-run invariant. Development mode now permits multiple
source runs only after each decision has independently passed the same per-record, per-vial,
SQLite, hash, value, and citation validation. Its normalized `run_id` is the explicitly inferred
presentation ID `development-v3-v4-six-run-presentation`; the six real source run IDs remain in the
fixture database and manifest. Development cues are 15 seconds per vial and normalized duration is
270 seconds. No Task 1 record validation was removed or weakened.

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
