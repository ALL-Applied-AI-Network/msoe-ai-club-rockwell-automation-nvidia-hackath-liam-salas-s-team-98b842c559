"""CPU contracts for the append-only Part-B evidence store."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8"))
    return _sha(path)


def _vial(
    root: Path,
    *,
    run: str,
    vial: int,
    recipe: str,
    previous: str | None,
    role: str,
    start_cursor: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]], dict[str, int]]:
    clip = root / "clips" / f"{run}-{vial}.mp4"
    clip_sha = _write(clip, f"clip:{run}:{vial}")
    records: list[dict[str, object]] = []
    locators: dict[str, dict[str, str]] = {}
    cursors: dict[str, int] = {}
    for ordinal, tank in enumerate(("t100", "t200"), start=1):
        evidence_id = f"E-{run}-{vial}-{ordinal}"
        image_id = f"I-{run}-{vial}-{ordinal}"
        rgb = root / "rgb" / f"{image_id}.png"
        mask = root / "mask" / f"{image_id}.png"
        rgb_sha = _write(rgb, f"rgb:{evidence_id}")
        mask_sha = _write(mask, f"mask:{evidence_id}")
        record = {
            "run_id": run,
            "case_id": f"CASE-{run}-{vial}",
            "vial_ordinal": vial,
            "block_recipe_id": f"BLOCK-{recipe}",
            "block_position": vial,
            "recipe_id": recipe,
            "previous_recipe_id": previous,
            "recipe_ml": {"t100": 1.0, "t200": 1.0, "t400": 0.0},
            "dispense_order": ["t100", "t200"],
            "split_role": role,
            "split_version": "fixture-v1",
            "stage_id": f"after-{tank}",
            "tank_id": tank,
            "stage_ordinal": ordinal,
            "fill_clip_path": clip.relative_to(root).as_posix(),
            "fill_frame_range": [vial * 10, vial * 10 + 5],
            "transition_frame_ranges": {"entry": None, "exit": None},
            "clip_sha256": clip_sha,
            "image_id": image_id,
            "rgb_sha256": rgb_sha,
            "mask_sha256": mask_sha,
            "mask_source": "predicted",
            "measurement_id": f"M-{run}-{vial}-{ordinal}",
            "median_lab": [50.0 + vial, float(ordinal), -float(ordinal)],
            "mad_lab": [1.0, 1.0, 1.0],
            "iqr_lab": [2.0, 2.0, 2.0],
            "pairwise_delta_e76": [],
            "colour_regime_id": None,
            "evidence_id": evidence_id,
            "decision_id": None,
            "disposition": None,
            "limit_codes": [],
            "history_refs": [],
            "explanation_id": None,
            "model_revision": None,
            "prompt_sha256": None,
            "image_ids_sent": [],
            "guard_result": None,
        }
        records.append(record)
        locators[image_id] = {
            "rgb_path": str(rgb),
            "rgb_sha256": rgb_sha,
            "mask_path": str(mask),
            "mask_sha256": mask_sha,
        }
        cursors[evidence_id] = start_cursor + ordinal - 1
    return records, locators, cursors


def _dataset(root: Path):
    specs = (
        ("RUN-A", 1, "R-A", None, "train", 10),
        ("RUN-A", 2, "R-B", "R-A", "train", 20),
        ("RUN-B", 1, "R-B", "R-A", "test", 30),
        ("RUN-C", 1, "R-B", "R-A", "evaluation", 40),
    )
    records: list[dict[str, object]] = []
    locators: dict[str, dict[str, str]] = {}
    cursors: dict[str, int] = {}
    for run, vial, recipe, previous, role, cursor in specs:
        batch = _vial(
            root,
            run=run,
            vial=vial,
            recipe=recipe,
            previous=previous,
            role=role,
            start_cursor=cursor,
        )
        records.extend(batch[0])
        locators.update(batch[1])
        cursors.update(batch[2])
    return records, locators, cursors


def test_store_creates_explicit_versioned_schema(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore, SCHEMA_VERSION

    database = tmp_path / "evidence.sqlite"
    EvidenceStore(database, artifact_root=tmp_path / "artifacts")

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "artifacts",
        "runs",
        "vials",
        "fills",
        "clips",
        "stages",
        "images",
        "measurements",
        "decisions",
        "explanations",
        "history_queries",
        "history_members",
        "regime_bindings",
        "sim_metadata",
    } <= tables

    wrong = tmp_path / "wrong-version.sqlite"
    with sqlite3.connect(wrong) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="unsupported evidence schema version"):
        EvidenceStore(wrong, artifact_root=tmp_path / "artifacts")


def test_ingest_is_idempotent_and_logical_collisions_fail(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    first = store.ingest(records, locators, phase="measured", event_cursors=cursors)
    second = store.ingest(records, locators, phase="measured", event_cursors=cursors)
    assert (first.inserted, first.existing) == (8, 0)
    assert (second.inserted, second.existing) == (0, 8)

    changed = deepcopy(records)
    changed[0]["median_lab"][0] = 99.0
    with pytest.raises(ValueError, match="logical collision"):
        store.ingest(changed, locators, phase="measured", event_cursors=cursors)

    alternate = root / "rgb" / "same-bytes-new-path.png"
    # Locator paths are content too: copying exact bytes to another path is not an
    # identical retry of the normalized ingest input.
    original_rgb = Path(locators[records[0]["image_id"]]["rgb_path"])
    alternate.write_bytes(original_rgb.read_bytes())
    changed_locators = deepcopy(locators)
    changed_locators[records[0]["image_id"]]["rgb_path"] = str(alternate)
    with pytest.raises(ValueError, match="logical collision"):
        store.ingest(records, changed_locators, phase="measured", event_cursors=cursors)


def test_incremental_ingest_cannot_backfill_an_earlier_same_run_vial(
    tmp_path: Path,
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    vial_one = [
        record
        for record in records
        if record["run_id"] == "RUN-A" and record["vial_ordinal"] == 1
    ]
    vial_two = [
        record
        for record in records
        if record["run_id"] == "RUN-A" and record["vial_ordinal"] == 2
    ]

    def inputs(rows):
        return (
            {row["image_id"]: locators[row["image_id"]] for row in rows},
            {row["evidence_id"]: cursors[row["evidence_id"]] for row in rows},
        )

    later_locators, later_cursors = inputs(vial_two)
    earlier_locators, earlier_cursors = inputs(vial_one)
    store = EvidenceStore(tmp_path / "out-of-order.sqlite", artifact_root=root)
    store.ingest(
        vial_two, later_locators, phase="measured", event_cursors=later_cursors
    )
    with pytest.raises(ValueError, match="incremental.*order"):
        store.ingest(
            vial_one,
            earlier_locators,
            phase="measured",
            event_cursors=earlier_cursors,
        )

    ordered = EvidenceStore(tmp_path / "ordered.sqlite", artifact_root=root)
    ordered.ingest(
        vial_one, earlier_locators, phase="measured", event_cursors=earlier_cursors
    )
    result = ordered.ingest(
        vial_two, later_locators, phase="measured", event_cursors=later_cursors
    )
    assert result.inserted == 2


def test_schema_rows_are_append_only_and_regime_binding_is_idempotent(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _vial(
        root, run="RUN-A", vial=1, recipe="R-A", previous=None,
        role="train", start_cursor=1,
    )
    database = tmp_path / "evidence.sqlite"
    store = EvidenceStore(database, artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    first = store.bind_measured_regime(records[0]["evidence_id"])
    second = store.bind_measured_regime(records[0]["evidence_id"])
    assert first.binding_sha256 == second.binding_sha256
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE stages SET stage_id = 'changed'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM stages")


def test_default_regime_is_same_run_and_arbitrary_membership_is_rejected(
    tmp_path: Path,
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)

    binding = store.bind_measured_regime("E-RUN-A-2-1")

    assert binding.colour_regime_id.startswith("measured-run-")
    assert set(binding.evidence_ids) == {
        record["evidence_id"]
        for record in records
        if record["run_id"] == "RUN-A"
    }
    with pytest.raises(TypeError, match="target evidence"):
        store.bind_measured_regime(
            "caller-chosen-regime",
            [record["evidence_id"] for record in records],
        )


def test_cross_run_regime_requires_measured_clean_reference_agreement(
    tmp_path: Path,
) -> None:
    from fluidmix.evidence_store import (
        CROSS_RUN_REGIME_TOLERANCE_DELTA_E76,
        EvidenceStore,
    )

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    clean_references = [
        "E-RUN-A-2-1",
        "E-RUN-B-1-1",
        "E-RUN-C-1-1",
    ]

    binding = store.bind_measured_regime(
        "E-RUN-C-1-1",
        cross_run_clean_reference_evidence_ids=clean_references,
    )

    assert CROSS_RUN_REGIME_TOLERANCE_DELTA_E76 == 2.0
    assert binding.colour_regime_id.startswith("measured-cross-")
    assert set(binding.evidence_ids) == {record["evidence_id"] for record in records}
    with pytest.raises(ValueError, match="same recipe and stage"):
        store.bind_measured_regime(
            "E-RUN-C-1-1",
            cross_run_clean_reference_evidence_ids=[
                "E-RUN-A-1-1",
                "E-RUN-C-1-1",
            ],
        )

    far_records = deepcopy(records)
    next(
        record
        for record in far_records
        if record["evidence_id"] == "E-RUN-B-1-1"
    )["median_lab"] = [80.0, 1.0, -1.0]
    far_store = EvidenceStore(tmp_path / "far.sqlite", artifact_root=root)
    far_store.ingest(far_records, locators, phase="measured", event_cursors=cursors)
    with pytest.raises(ValueError, match="frozen.*tolerance"):
        far_store.bind_measured_regime(
            "E-RUN-C-1-1",
            cross_run_clean_reference_evidence_ids=clean_references,
        )


@pytest.mark.parametrize("failure", ["hash", "outside", "symlink", "nonfinite"])
def test_ingest_rejects_unverified_artifacts_and_nonfinite_values(
    tmp_path: Path, failure: str
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _vial(
        root,
        run="RUN-A",
        vial=1,
        recipe="R-A",
        previous=None,
        role="train",
        start_cursor=1,
    )
    if failure == "hash":
        locators[records[0]["image_id"]]["rgb_sha256"] = "0" * 64
    elif failure == "outside":
        outside = tmp_path / "outside.png"
        outside_sha = _write(outside, "outside")
        locator = locators[records[0]["image_id"]]
        locator["rgb_path"] = str(outside)
        locator["rgb_sha256"] = outside_sha
        records[0]["rgb_sha256"] = outside_sha
    elif failure == "symlink":
        target = tmp_path / "target.png"
        target_sha = _write(target, "target")
        link = root / "rgb" / "unsafe.png"
        link.symlink_to(target)
        locator = locators[records[0]["image_id"]]
        locator["rgb_path"] = str(link)
        locator["rgb_sha256"] = target_sha
        records[0]["rgb_sha256"] = target_sha
    else:
        records[0]["median_lab"][0] = math.nan

    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    with pytest.raises((TypeError, ValueError), match="hash|SHA|below|unsafe|finite"):
        store.ingest(records, locators, phase="measured", event_cursors=cursors)


def test_retrieval_is_capability_bound_strictly_earlier_and_role_filtered(
    tmp_path: Path,
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    regime = store.bind_measured_regime(
        "E-RUN-C-1-1",
        cross_run_clean_reference_evidence_ids=[
            "E-RUN-A-2-1", "E-RUN-B-1-1", "E-RUN-C-1-1"
        ],
    )

    reference = store.reference_bank(
        "R-B",
        "after-t100",
        regime,
        before=40,
        allowed_roles={"train", "test"},
        target_evidence_id="E-RUN-C-1-1",
    )
    assert [member.evidence_id for member in reference.members] == [
        "E-RUN-A-2-1",
        "E-RUN-B-1-1",
    ]
    assert all(member.event_cursor < 40 for member in reference.members)
    assert all(member.evidence_id != "E-RUN-C-1-1" for member in reference.members)

    vial = store.vial_history(
        "RUN-A",
        before=21,
        allowed_roles={"train"},
        target_evidence_id="E-RUN-A-2-2",
    )
    assert [member.event_cursor for member in vial.members] == [10, 11, 20]

    switch = store.switch_history(
        "R-A",
        "R-B",
        before=31,
        allowed_roles={"train"},
        target_evidence_id="E-RUN-B-1-2",
    )
    assert [member.evidence_id for member in switch.members] == [
        "E-RUN-A-2-1",
        "E-RUN-A-2-2",
    ]
    assert {member.split_role for member in switch.members} == {"train"}
    assert all(member.rgb_path and member.rgb_sha256 for member in switch.members)
    assert all(member.mask_path and member.mask_sha256 for member in switch.members)
    assert all(member.clip_path and member.clip_sha256 for member in switch.members)
    assert all(len(member.median_lab) == 3 for member in switch.members)
    assert all(not hasattr(member, "disposition") for member in switch.members)


def test_query_hash_binds_filters_ordered_members_and_citations(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore, assert_history_citation

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    regime = store.bind_measured_regime(
        "E-RUN-C-1-1",
        cross_run_clean_reference_evidence_ids=[
            "E-RUN-A-2-1", "E-RUN-B-1-1", "E-RUN-C-1-1"
        ],
    )
    query = store.reference_bank(
        "R-B",
        "after-t100",
        regime,
        before=40,
        allowed_roles={"train", "test"},
        target_evidence_id="E-RUN-C-1-1",
    )
    changed_filter = store.reference_bank(
        "R-B",
        "after-t100",
        regime,
        before=40,
        allowed_roles={"train", "test", "calibration"},
        target_evidence_id="E-RUN-C-1-1",
    )
    changed_members = store.reference_bank(
        "R-B",
        "after-t100",
        regime,
        before=40,
        allowed_roles={"train"},
        target_evidence_id="E-RUN-C-1-1",
    )
    assert len({query.query_sha256, changed_filter.query_sha256, changed_members.query_sha256}) == 3
    member = query.members[0]
    assert assert_history_citation(query, member.evidence_id, member.image_id) is member
    with pytest.raises(ValueError, match="not a member"):
        assert_history_citation(query, "not-returned", member.image_id)
    with pytest.raises(TypeError, match="trusted history query"):
        assert_history_citation(
            {"query_sha256": query.query_sha256}, member.evidence_id, member.image_id
        )
    with pytest.raises(TypeError, match="measured-regime capability"):
        store.reference_bank(
            "R-B",
            "after-t100",
            {"colour_regime_id": "regime"},
            before=40,
            allowed_roles={"train"},
            target_evidence_id="E-RUN-C-1-1",
        )


def test_wp4_symmetric_pairs_do_not_leak_target_or_future_member(
    tmp_path: Path,
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    by_id = {record["evidence_id"]: record for record in records}
    earlier = "E-RUN-A-2-1"
    earlier_peer = "E-RUN-B-1-1"
    target = "E-RUN-C-1-1"
    by_id[earlier]["pairwise_delta_e76"] = [
        {"other_evidence_id": earlier_peer, "delta_e76": 0.75},
        {"other_evidence_id": target, "delta_e76": 1.25}
    ]
    by_id[earlier_peer]["pairwise_delta_e76"] = [
        {"other_evidence_id": earlier, "delta_e76": 0.75}
    ]
    by_id[target]["pairwise_delta_e76"] = [
        {"other_evidence_id": earlier, "delta_e76": 1.25}
    ]
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    regime = store.bind_measured_regime(
        target,
        cross_run_clean_reference_evidence_ids=[
            "E-RUN-A-2-1", "E-RUN-B-1-1", target
        ],
    )

    query = store.reference_bank(
        "R-B",
        "after-t100",
        regime,
        before=40,
        allowed_roles={"train", "test"},
        target_evidence_id=target,
    )

    member = next(item for item in query.members if item.evidence_id == earlier)
    assert member.pairwise_delta_e76 == ((earlier_peer, 0.75),)
    assert target not in json.dumps(
        [item.canonical_value() for item in query.members], sort_keys=True
    )


@pytest.mark.parametrize(
    ("query_kind", "selectors"),
    [
        ("reference", ("R-A", "after-t100")),
        ("reference", ("R-B", "after-t200")),
        ("vial", ("RUN-A",)),
        ("switch", ("R-X", "R-B")),
        ("switch", ("R-A", "R-X")),
    ],
)
def test_query_selectors_must_match_the_target_record(
    tmp_path: Path, query_kind: str, selectors: tuple[str, ...]
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    target = "E-RUN-C-1-1"
    regime = store.bind_measured_regime(target)

    with pytest.raises(ValueError, match="target.*selector"):
        if query_kind == "reference":
            store.reference_bank(
                selectors[0], selectors[1], regime, before=40,
                allowed_roles={"train"}, target_evidence_id=target,
            )
        elif query_kind == "vial":
            store.vial_history(
                selectors[0], before=40, allowed_roles={"train"},
                target_evidence_id=target,
            )
        else:
            store.switch_history(
                selectors[0], selectors[1], before=40,
                allowed_roles={"train"}, target_evidence_id=target,
            )


def test_roles_and_cursors_are_explicit_and_fail_closed(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=tmp_path / "artifacts")
    with pytest.raises(ValueError, match="allowed_roles"):
        store.vial_history("RUN-A", before=1, allowed_roles=set(), target_evidence_id="E")
    with pytest.raises(ValueError, match="allowed_roles"):
        store.vial_history("RUN-A", before=1, allowed_roles={"buffer"}, target_evidence_id="E")
    with pytest.raises((TypeError, ValueError), match="before"):
        store.vial_history("RUN-A", before=True, allowed_roles={"train"}, target_evidence_id="E")


def test_fill_clips_fixture_adapter_normalizes_only_store_inputs(tmp_path: Path) -> None:
    from fluidmix.evidence_store import adapt_fill_clips_v1

    root = tmp_path / "artifacts"
    records, locators, _ = _vial(
        root,
        run="RUN-A",
        vial=1,
        recipe="R-A",
        previous=None,
        role="train",
        start_cursor=1,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "part_b_fill_clip_measurements_v1",
        "evidence_records": records,
        "image_locators": [
            {"image_id": image_id, **locator, "derived_crop_path": "ignored"}
            for image_id, locator in locators.items()
        ],
    }
    adapted_records, adapted_locators = adapt_fill_clips_v1(payload)
    assert adapted_records == records
    assert adapted_locators == locators
    assert all(set(locator) == {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"} for locator in adapted_locators.values())


def test_production_retrieval_never_reads_opaque_sim_metadata(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    database = tmp_path / "evidence.sqlite"
    store = EvidenceStore(database, artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO sim_metadata VALUES (?, ?, ?)",
            ("secret", b"preset/residue/label", "0" * 64),
        )
        connection.commit()
    statements: list[str] = []
    store._trace_for_test = statements.append
    with store._connect() as connection, pytest.raises(
        sqlite3.DatabaseError, match="authoriz|prohibited"
    ):
        connection.execute("SELECT payload FROM sim_metadata").fetchall()
    with store._connect() as connection, pytest.raises(
        sqlite3.DatabaseError, match="authoriz|prohibited"
    ):
        connection.execute(
            "INSERT INTO sim_metadata VALUES ('forged', X'00', ?)", ("0" * 64,)
        )
    statements.clear()
    result = store.vial_history(
        "RUN-A",
        before=21,
        allowed_roles={"train"},
        target_evidence_id="E-RUN-A-2-2",
    )
    assert result.members
    assert all("sim_metadata" not in statement.lower() for statement in statements)
    assert not hasattr(result, "sim_metadata")


def test_retrieval_rechecks_cited_artifact_bytes(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    root = tmp_path / "artifacts"
    records, locators, cursors = _dataset(root)
    store = EvidenceStore(tmp_path / "evidence.sqlite", artifact_root=root)
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    Path(locators[records[0]["image_id"]]["rgb_path"]).write_bytes(b"drifted")
    with pytest.raises(ValueError, match="retrieved RGB image SHA-256 mismatch"):
        store.vial_history(
            "RUN-A",
            before=21,
            allowed_roles={"train"},
            target_evidence_id="E-RUN-A-2-2",
        )
