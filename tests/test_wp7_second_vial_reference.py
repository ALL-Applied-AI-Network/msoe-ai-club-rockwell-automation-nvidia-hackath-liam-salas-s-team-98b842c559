"""Synthetic contracts for the explicit same-camera cohort-goldens route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decision() -> dict[str, object]:
    return {
        "schema_version": "wp7-same-camera-cohort-goldens-route-decision-v1",
        "artifact_type": "wp7_same_camera_cohort_goldens_route_decision",
        "issue_id": "INN-67",
        "scope": "wp1_and_video5_same_camera_cohort_goldens",
        "approved_mode": "same_camera_cohort_goldens",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
        "reference_rule": (
            "closed_cohort_position_1_compares_to_golden_positions_2_and_3_"
            "positions_2_and_3_reference_control"
        ),
        "distance_metric": "CIE76",
        "keep_threshold_delta_e76": 1.0,
        "discard_threshold_delta_e76": 2.0,
        "aggregation_rule": (
            "keep_every_stage_both_le_1_discard_any_stage_both_ge_2_else_insufficient"
        ),
        "temporal_scope": "post_complete_same_run_three_vial_cohort_only",
        "evidence_boundary": (
            "runtime_rgb_cnn_measurements_and_programmed_recipe_metadata_only"
        ),
    }


def _sequence(block_count: int) -> dict[str, object]:
    runs = []
    global_block = 0
    for run_number, run_blocks in enumerate(
        (3, 3) if block_count == 6 else (10,), start=1
    ):
        vials = []
        for _ in range(run_blocks):
            global_block += 1
            recipe = f"R-{global_block:02d}"
            for position in (1, 2, 3):
                ordinal = len(vials) + 1
                vials.append({
                    "case_id": f"RUN-{run_number}-CASE-{ordinal:02d}",
                    "vial_ordinal": ordinal,
                    "block_recipe_id": recipe,
                    "block_position": position,
                    "recipe_id": recipe,
                    "previous_recipe_id": recipe if position > 1 else "R-PREV",
                    "programmed_tank_ml": {
                        "t100": 10.0, "t200": 20.0, "t400": 0.0,
                    },
                    "dispense_order": ["t200", "t100"],
                })
        runs.append({"run_id": f"RUN-{run_number}", "vials": vials})
    return {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "test" if block_count == 6 else "real_check",
        "split_version": "synthetic-three-vial-blocks-v1",
        "runs": runs,
    }


def test_cohort_goldens_route_preflight_is_distinct_and_executable() -> None:
    from fluidmix.carryover_pipeline import (
        _validate_execution_preflight,
        preflight_explanation_route,
    )

    decision = _decision()
    receipt = preflight_explanation_route(
        requested_mode="same_camera_cohort_goldens",
        expected_test_vials=18,
        expected_real_check_vials=30,
        route_decision=decision,
        expected_route_decision_sha256=_canonical_hash(decision),
    )

    assert receipt["history"] is None
    assert receipt["predicted_mask"] is None
    assert receipt["test"] == {
        "terminal_behavior": "same_camera_cohort_goldens",
        "vial_count": 18,
    }
    assert _validate_execution_preflight(
        receipt,
        expected_sha256=str(receipt["preflight_sha256"]),
        selected_mode="same_camera_cohort_goldens",
    ) == receipt


def test_cohort_goldens_route_decision_threshold_or_boundary_drift_fails() -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    decision = _decision()
    decision["keep_threshold_delta_e76"] = 1.01
    with pytest.raises(ValueError, match="route-decision"):
        preflight_explanation_route(
            requested_mode="same_camera_cohort_goldens",
            expected_test_vials=18,
            expected_real_check_vials=30,
            route_decision=decision,
            expected_route_decision_sha256=_canonical_hash(decision),
        )


def test_run_cli_exposes_distinct_cohort_goldens_mode() -> None:
    spec = importlib.util.spec_from_file_location(
        "same_camera_cohort_goldens_cli", ROOT / "scripts/run_carryover_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run_test = next(
        action for action in cli.build_parser()._actions
        if action.dest == "phase"
    ).choices["run-test"]
    selected = next(
        action for action in run_test._actions if action.dest == "selected_mode"
    )
    assert "same_camera_cohort_goldens" in selected.choices


def test_cohort_goldens_mode_cannot_bypass_capture_runtime_before_claim(
    tmp_path: Path, monkeypatch
) -> None:
    import fluidmix.carryover_pipeline as pipeline

    decision = _decision()
    preflight = pipeline.preflight_explanation_route(
        requested_mode="same_camera_cohort_goldens",
        expected_test_vials=18,
        expected_real_check_vials=30,
        route_decision=decision,
        expected_route_decision_sha256=_canonical_hash(decision),
    )
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda *_args: pytest.fail("capture bypass opened a protected input"),
    )
    claim = tmp_path / "claim.json"
    with pytest.raises(ValueError, match="complete capture boundary"):
        pipeline.run_test(
            video=tmp_path / "never-video",
            recipe_sequence=tmp_path / "never-recipe",
            freeze=tmp_path / "never-freeze",
            expected_freeze_sha256="0" * 64,
            evidence_out_store=tmp_path / "never-store",
            out=tmp_path / "never-out",
            claim=claim,
            owner_id="synthetic-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=str(preflight["preflight_sha256"]),
            selected_mode="same_camera_cohort_goldens",
        )
    assert not claim.exists()


@pytest.mark.parametrize(("blocks", "vials"), [(6, 18), (10, 30)])
def test_cohort_goldens_blocks_are_exact_ordered_triplets(blocks: int, vials: int) -> None:
    from fluidmix.carryover_pipeline import _same_camera_reference_blocks

    grouped = _same_camera_reference_blocks(_sequence(blocks), vials)

    assert len(grouped) == blocks
    assert all([vial["block_position"] for vial in group] == [1, 2, 3]
               for _, group in grouped)


def test_adjacent_same_recipe_triplets_remain_distinct() -> None:
    from fluidmix.carryover_pipeline import _same_camera_reference_blocks
    sequence = _sequence(6)
    for vial in sequence["runs"][0]["vials"][3:6]:
        vial["block_recipe_id"] = "R-01"
        vial["recipe_id"] = "R-01"
    assert len(_same_camera_reference_blocks(sequence, 18)) == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]), "ordered"),
        (lambda rows: rows[1].__setitem__("block_position", 1), "positions"),
        (lambda rows: rows[2].__setitem__("block_position", 2), "positions"),
        (lambda rows: rows[1].__setitem__("recipe_id", "R-OTHER"), "recipe identity"),
        (lambda rows: rows[2].__setitem__("dispense_order", ["t100"]), "stage-key"),
    ],
)
def test_cohort_goldens_blocks_reject_ambiguous_reference_or_stage_contract(
    mutation, message: str
) -> None:
    from fluidmix.carryover_pipeline import _same_camera_reference_blocks

    sequence = _sequence(6)
    rows = sequence["runs"][0]["vials"]
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        _same_camera_reference_blocks(sequence, 18)


def _measurements(sequence: dict[str, object], position_values: dict[int, float]):
    rows = {}
    cursor = 0
    for run in sequence["runs"]:
        for vial in run["vials"]:
            value = position_values[int(vial["block_position"])]
            key = (run["run_id"], vial["case_id"])
            rows[key] = []
            for ordinal, tank in enumerate(vial["dispense_order"], start=1):
                rows[key].append({
                    "evidence_id": f"E-{vial['case_id']}-{ordinal}",
                    "image_id": f"I-{vial['case_id']}-{ordinal}",
                    "event_cursor": cursor,
                    "run_id": run["run_id"],
                    "case_id": vial["case_id"],
                    "vial_ordinal": vial["vial_ordinal"],
                    "recipe_id": vial["recipe_id"],
                    "previous_recipe_id": vial["previous_recipe_id"],
                    "stage_ordinal": ordinal,
                    "tank_id": tank,
                    "median_lab": [value, 0.0, 0.0],
                })
                cursor += 1
    return rows


def test_cohort_goldens_decisions_are_causal_and_use_frozen_endpoints() -> None:
    from fluidmix.carryover_pipeline import _decide_same_camera_reference_vials

    sequence = _sequence(6)
    measurements = _measurements(sequence, {1: 50.0, 2: 10.0, 3: 11.0})
    records = _decide_same_camera_reference_vials(sequence, measurements, 18)

    first_block = records[:3]
    assert first_block[0]["disposition"] == "discard"
    assert len(first_block[0]["measurement"]["stage_citations"][0][
        "reference_evidence_ids"
    ]) == 2
    assert first_block[1]["disposition"] == "insufficient_evidence"
    assert first_block[1]["reason_codes"] == ["reference_control"]
    assert first_block[1]["limit_codes"] == ["reference_control"]
    assert first_block[2]["disposition"] == "insufficient_evidence"
    assert first_block[2]["reason_codes"] == ["reference_control"]
    assert all(
        set(citation) == {
            "stage_ordinal", "tank_id", "target_evidence_id", "target_image_id",
            "reference_evidence_ids", "reference_image_ids",
        }
        for citation in first_block[0]["measurement"]["stage_citations"]
    )
    discard = _measurements(sequence, {1: 14.0, 2: 10.0, 3: 11.0})
    assert _decide_same_camera_reference_vials(sequence, discard, 18)[0][
        "disposition"
    ] == "discard"


def test_incomplete_cohort_scores_nothing_and_preserves_target_measurement() -> None:
    from fluidmix.carryover_pipeline import _decide_same_camera_reference_vials
    sequence = _sequence(6)
    measured = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 10.0})
    measured[("RUN-1", sequence["runs"][0]["vials"][2]["case_id"])] = None
    records = _decide_same_camera_reference_vials(sequence, measured, 18)[:3]
    assert all(row["disposition"] == "insufficient_evidence" for row in records)
    assert records[0]["limit_codes"] == ["cohort_measurement_incomplete"]
    assert records[0]["measurement"]["evidence_ids"]


def test_cohort_goldens_projection_rejects_stage_drift_and_future_reference() -> None:
    from fluidmix.carryover_pipeline import (
        _decide_same_camera_reference_vials,
        _same_camera_stage_distances,
    )

    sequence = _sequence(6)
    measurements = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 10.0})
    third_case = sequence["runs"][0]["vials"][2]["case_id"]
    measurements[("RUN-1", third_case)][1]["tank_id"] = "t400"
    with pytest.raises(ValueError, match="stage-key parity"):
        _decide_same_camera_reference_vials(sequence, measurements, 18)

    first, reference = sequence["runs"][0]["vials"][:2]
    target_rows = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 10.0})[
        ("RUN-1", first["case_id"])
    ]
    reference_rows = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 10.0})[
        ("RUN-1", reference["case_id"])
    ]
    target_rows[-1]["event_cursor"] = reference_rows[-1]["event_cursor"]
    with pytest.raises(ValueError, match="event order"):
        _same_camera_stage_distances(
            first,
            reference,
            target_rows,
            reference_rows,
        )


def test_duplicate_recipe_ids_across_runs_never_share_a_reference() -> None:
    from fluidmix.carryover_pipeline import _decide_same_camera_reference_vials

    sequence = _sequence(6)
    repeated = sequence["runs"][1]["vials"][:3]
    for vial in repeated:
        vial["block_recipe_id"] = "R-01"
        vial["recipe_id"] = "R-01"
    measurements = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 10.0})
    records = _decide_same_camera_reference_vials(sequence, measurements, 18)

    second_run_target = records[9]
    assert second_run_target["run_id"] == "RUN-2"
    assert all(
            ref["evidence_id"].startswith("E-RUN-2-")
        for ref in second_run_target["history"]["reference_refs"]
    )


def test_runtime_projection_injects_event_cursors_before_causal_decision(
    monkeypatch,
) -> None:
    import fluidmix.carryover_pipeline as pipeline

    sequence = _sequence(6)
    measurements = _measurements(sequence, {1: 10.0, 2: 10.0, 3: 11.0})
    templates = {}
    event_cursors = {}
    bindings = {}
    for run in sequence["runs"]:
        for vial in run["vials"]:
            case_id = vial["case_id"]
            templates[(run["run_id"], case_id)] = [
                {"image_id": row["image_id"], "case_id": case_id}
                for row in measurements[(run["run_id"], case_id)]
            ]
            for row in measurements[(run["run_id"], case_id)]:
                event_cursors[row["evidence_id"]] = row["event_cursor"]
                bindings[row["image_id"]] = SimpleNamespace(validity="valid")

    monkeypatch.setattr(
        pipeline, "_trusted_runtime_bindings_by_image", lambda _output: bindings
    )
    monkeypatch.setattr(
        pipeline, "_validated_runtime_templates_by_vial", lambda *_args: templates
    )
    monkeypatch.setattr(
        pipeline,
        "project_runtime_measurements",
        lambda value: [
            dict(row) for row in measurements[next(
                key for key in measurements
                if key[1] == value.record_templates[0]["case_id"]
            )]
        ],
    )
    monkeypatch.setattr(
        pipeline, "_assert_runtime_stage_recipe_identity", lambda *_args, **_kwargs: None
    )
    output = pipeline.RuntimeProjectionInput(
        (), object(), {}, event_cursors, {"route": "synthetic"}
    )

    explained, records, count = pipeline._project_same_camera_reference_vials(
        sequence=sequence, runtime_output=output, expected_vials=18
    )

    assert count == 36
    assert len(explained) == 36
    assert records[0]["disposition"] == "keep"
    assert all(
        all(ref.startswith("E-RUN-1-") for ref in citation["reference_evidence_ids"])
        for citation in records[0]["measurement"]["stage_citations"]
    )

    unavailable_key = (
        "RUN-1", sequence["runs"][0]["vials"][2]["case_id"]
    )
    unavailable_templates = templates.pop(unavailable_key)
    unavailable_explained, unavailable_records, unavailable_count = (
        pipeline._project_same_camera_reference_vials(
            sequence=sequence, runtime_output=output, expected_vials=18
        )
    )
    assert unavailable_count == 34
    assert len(unavailable_explained) == 34
    assert all(
        record["disposition"] == "insufficient_evidence"
        for record in unavailable_records[:3]
    )
    assert unavailable_records[0]["measurement"]["evidence_ids"]
    templates[unavailable_key] = unavailable_templates

    target_evidence = measurements[("RUN-1", sequence["runs"][0]["vials"][1]["case_id"])][0][
        "evidence_id"
    ]
    event_cursors[target_evidence] = 0
    with pytest.raises(ValueError, match="cursor order"):
        pipeline._project_same_camera_reference_vials(
            sequence=sequence, runtime_output=output, expected_vials=18
        )


def test_dual_golden_shape_ingests_into_real_evidence_store(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore
    from test_evidence_store import _vial
    root = tmp_path / "artifacts"
    records, locators, cursors = [], {}, {}
    for vial, cursor in ((1, 10), (2, 20), (3, 30)):
        batch = _vial(root, run="RUN-1", vial=vial, recipe="R-1",
                      previous=None, role="test", start_cursor=cursor)
        records.extend(batch[0]); locators.update(batch[1]); cursors.update(batch[2])
    goldens = {
        (int(row["vial_ordinal"]), int(row["stage_ordinal"])): row
        for row in records if row["vial_ordinal"] in {2, 3}
    }
    for row in records:
        vial = int(row["vial_ordinal"])
        row.update({
            "decision_id": f"D-{vial}",
            "disposition": "keep" if vial == 1 else "insufficient_evidence",
            "limit_codes": [] if vial == 1 else ["reference_control"],
            "history_refs": [], "explanation_id": f"X-{vial}",
            "model_revision": "deterministic-template:closed-cohort-v1",
            "prompt_sha256": "1" * 64, "image_ids_sent": [],
            "guard_result": "pass" if vial == 1 else "fallback",
        })
        if vial == 1:
            stage = int(row["stage_ordinal"])
            row["pairwise_delta_e76"] = sorted([
                {"other_evidence_id": goldens[(golden, stage)]["evidence_id"],
                 "delta_e76": 1.0}
                for golden in (2, 3)
            ], key=lambda item: item["other_evidence_id"])
    result = EvidenceStore(tmp_path / "store.sqlite", artifact_root=root).ingest(
        records, locators, phase="explained", event_cursors=cursors
    )
    assert result.inserted == 6


@pytest.mark.parametrize(
    ("phase", "blocks", "expected_decisions", "expected_abstentions"),
    [
        ("run-test", 6, 6, 12),
        ("run-real-check", 10, 10, 20),
    ],
)
def test_full_phase_uses_claimed_runtime_and_publishes_causal_denominators(
    tmp_path: Path,
    monkeypatch,
    phase: str,
    blocks: int,
    expected_decisions: int,
    expected_abstentions: int,
) -> None:
    import fluidmix.carryover_pipeline as pipeline
    from test_carryover_pipeline import _freeze, _write_json

    decision = _decision()
    preflight = pipeline.preflight_explanation_route(
        requested_mode="same_camera_cohort_goldens",
        expected_test_vials=18,
        expected_real_check_vials=30,
        route_decision=decision,
        expected_route_decision_sha256=_canonical_hash(decision),
    )
    freeze, freeze_sha = _freeze(tmp_path / "freeze", accepts_real_rgb_only=True)
    sequence = _sequence(blocks)
    recipe = tmp_path / "recipe.json"
    video = tmp_path / "camera.bin"
    _write_json(recipe, sequence)
    video.write_bytes(b"synthetic claimed camera bytes")
    plan = tmp_path / "capture-plan.json"
    plan.write_text("synthetic deferred plan\n")
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    capture = tmp_path / "capture.json"
    provenance = tmp_path / "capture-provenance.json"
    output = tmp_path / "out.json"
    claim = tmp_path / "claim.json"
    store = tmp_path / "evidence.sqlite3"
    events = []

    case_ids = [
        vial["case_id"] for run in sequence["runs"] for vial in run["vials"]
    ]
    event_cursors = {f"E-{case_id}": index for index, case_id in enumerate(case_ids)}
    image_locators = {
        f"I-{case_id}": {"rgb_path": "unused", "mask_path": "unused"}
        for case_id in case_ids
    }

    class BoundRuntime:
        runtime_id = "reviewed-clamp-runtime-v1"
        runtime_sha256 = "2" * 64
        config_sha256 = "3" * 64
        accepts_real_rgb_only = True

        def analyze(self, **_kwargs):
            events.append("runtime_analyze")
            return pipeline.RuntimeProjectionInput(
                (), object(), image_locators, event_cursors,
                {"route": "same_camera_closed_cohort_goldens"},
            )

    def capture_loader(video_bytes, _sequence_value, actual_phase, video_sha):
        events.append("capture_loader")
        assert claim.exists()
        assert Path(f"{output}.journal.jsonl").exists()
        assert hashlib.sha256(video_bytes).hexdigest() == video_sha
        origin = "synthetic_camera" if actual_phase == "run-test" else "real_camera"
        receipt = {
            "schema_version": "wp7-camera-capture-v1",
            "artifact_type": "wp7_camera_capture",
            "capture_origin": origin,
            "video_sha256": video_sha,
            "encoded_transition_evidence": [],
            "fill_clips": [],
        }
        receipt["receipt_sha256"] = _canonical_hash(receipt)
        _write_json(capture, receipt)
        publication = {
            "schema_version": "wp7-phase-capture-publication-v1",
            "artifact_type": "wp7_phase_capture_publication",
            "status": "published",
            "phase": actual_phase,
            "capture_origin": origin,
            "video_sha256": video_sha,
            "plan_binding": {
                "path": str(plan.resolve()),
                "raw_file_sha256": plan_sha,
                "canonical_plan_sha256": "1" * 64,
            },
            "capture_receipt_binding": {
                "path": str(capture.resolve()),
                "raw_file_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "source_provenance": [{
                "role": "synthetic_test_metadata",
                "path": str(plan.resolve()),
                "sha256": plan_sha,
            }],
        }
        _write_json(provenance, publication)
        return BoundRuntime()

    def fake_project(*, sequence, expected_vials, **_kwargs):
        measurements = _measurements(sequence, {1: 50.0, 2: 10.0, 3: 11.0})
        records = pipeline._decide_same_camera_reference_vials(
            sequence, measurements, expected_vials
        )
        explained = [
            {"image_id": f"I-{case_id}", "evidence_id": f"E-{case_id}"}
            for case_id in case_ids
        ]
        return explained, records, len(explained)

    ingested = []

    class FakeStore:
        def __init__(self, path, **_kwargs):
            self.path = Path(path)
            self.path.write_bytes(b"synthetic same-camera evidence store")

        def ingest(self, records, locators, **kwargs):
            ingested.append((records, locators, kwargs))

    original_read = pipeline._read_production_bytes
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, role: (
            events.append(f"read:{role}"),
            pytest.fail("protected read occurred before claim")
            if not claim.exists() else original_read(path, role),
        )[-1],
    )
    monkeypatch.setattr(pipeline, "_project_same_camera_reference_vials", fake_project)
    monkeypatch.setattr(pipeline, "EvidenceStore", FakeStore)

    function = pipeline.run_test if phase == "run-test" else pipeline.run_real_check
    receipt = function(
        video=video,
        recipe_sequence=recipe,
        freeze=freeze,
        expected_freeze_sha256=freeze_sha,
        evidence_out_store=store,
        out=output,
        claim=claim,
        owner_id="synthetic-owner",
        preflight_receipt=preflight,
        expected_preflight_sha256=str(preflight["preflight_sha256"]),
        selected_mode="same_camera_cohort_goldens",
        capture_plan=plan,
        capture_plan_sha256=plan_sha,
        capture_receipt_out=capture,
        capture_provenance_out=provenance,
        capture_runtime_loader=capture_loader,
    )

    assert events[:4] == [
        "read:recipe sequence", "read:camera video", "capture_loader", "runtime_analyze"
    ]
    assert receipt["denominators"]["decision_count"] == expected_decisions
    assert receipt["denominators"]["abstention_count"] == expected_abstentions
    assert receipt["source_bindings"]["evidence_out_store_sha256"] == hashlib.sha256(
        store.read_bytes()
    ).hexdigest()
    assert ingested and len(ingested[0][0]) == blocks * 3

