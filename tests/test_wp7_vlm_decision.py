"""Focused synthetic contract tests for the WP7 image-grounded decision step."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.wp7_vlm_decision import (
    ContractError,
    aggregate_decision,
    build_vial1_bundle,
    create_development_freeze,
    guard_model_response,
    load_production_bundle,
    parse_model_response,
    run_sequential_vlm_runtime,
    validate_tracker_receipt,
)


PHASES = ("early", "mid", "end")


def _cohort(tmp_path: Path, *, partition: str = "development"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    locators = {}
    vials = []
    for vial_position in (1, 2, 3):
        vial_stages = []
        for stage_ordinal in (1, 2):
            frames = []
            for phase_index, phase in enumerate(PHASES):
                image_id = f"IMG-C1-V{vial_position}-S{stage_ordinal}-{phase}"
                path = tmp_path / f"{image_id}.bin"
                path.write_bytes(bytes((vial_position, stage_ordinal, phase_index)))
                locators[image_id] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                frames.append({
                    "image_id": image_id,
                    "phase": phase,
                    "timestamp_seconds": float(phase_index + 1),
                })
            vial_stages.append({
                "stage_id": f"stage-{stage_ordinal}",
                "frames": frames,
                "measurement": {
                    "evidence_id": f"EV-C1-V{vial_position}-S{stage_ordinal}",
                    "median_lab": [50.0, 2.0, 3.0],
                    "mad_lab": [1.0, 1.0, 1.0],
                    "iqr_lab": [2.0, 2.0, 2.0],
                    "delta_e76_vial2": 0.0,
                    "delta_e76_vial3": 0.0,
                    "deterministic_disposition": "keep",
                    "limit_codes": [],
                },
            })
        vials.append({"vial_position": vial_position, "vial_id": f"V{vial_position}", "stages": vial_stages})
    cohort = {
        "schema_version": "wp7-vlm-cohort-v1",
        "cohort_id": "cohort-1",
        "partition": partition,
        "recipe_id": "recipe-new",
        "previous_recipe_id": "recipe-old",
        "camera_metadata": {
            "resolution": [1920, 1080],
            "frame_rate_hz": 30.0,
            "exposure": None,
            "white_balance": None,
            "viewpoint_id": "fill-camera",
            "vial_roi": [10, 20, 100, 200],
        },
        "stages": [
            {
                "stage_id": f"stage-{ordinal}",
                "stage_ordinal": ordinal,
                "tank_id": f"t{ordinal}00",
                "programmed_ml": float(ordinal),
                "dispense_order": ordinal,
                "stage_timing_seconds": [float(ordinal), float(ordinal + 1)],
                "stream": {
                    "evidence_id": f"STREAM-S{ordinal}",
                    "median_lab": [45.0, 1.0, 2.0],
                    "spread_lab": [1.0, 1.0, 1.0],
                },
            }
            for ordinal in (1, 2)
        ],
        "vials": vials,
    }
    decision = {
        "schema_version": "wp7-deterministic-decision-receipt-v1",
        "artifact_type": "wp7_deterministic_decision_receipt",
        "cohort_id": "cohort-1",
        "target_vial_id": "V1",
        "disposition": "keep",
        "limit_codes": [],
        "config_sha256": "d" * 64,
        "source_artifact_sha256": "e" * 64,
        "stages": [
            {
                "stage_id": f"stage-{ordinal}",
                "evidence_id": f"EV-C1-V1-S{ordinal}",
                "disposition": "keep",
                "limit_codes": [],
                "delta_e76_vial2": 0.0,
                "delta_e76_vial3": 0.0,
            }
            for ordinal in (1, 2)
        ],
    }
    decision["receipt_sha256"] = hashlib.sha256(json.dumps(
        decision, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    cohort["deterministic_decision_receipt"] = decision
    return cohort, locators


def _tracker_receipts(cohort):
    receipts = []
    for vial in cohort["vials"]:
        stages = []
        cumulative = 0.0
        for index, (recipe_stage, vial_stage) in enumerate(
            zip(cohort["stages"], vial["stages"], strict=True), 1
        ):
            cumulative += recipe_stage["programmed_ml"]
            tank_number = int(recipe_stage["tank_id"].removeprefix("t"))
            stage = {
                "fill_step": index,
                "tank_id": recipe_stage["tank_id"],
                "tank_number": tank_number,
                "stream_step_index": index,
                "expected_dispense_ml": recipe_stage["programmed_ml"],
                "expected_cumulative_ml": cumulative,
                "twin_geometry_equivalent_volume_ml": cumulative,
                "measured_volume_ml": cumulative,
                "agreement": True,
                "tank_cluster_consistent": True,
                "stream_lab": recipe_stage["stream"]["median_lab"],
                "stream_spread_lab": recipe_stage["stream"]["spread_lab"],
                "images": {row["phase"]: row["image_id"] for row in vial_stage["frames"]},
            }
            stages.append(stage)
        receipt = {
            "schema_version": "wp7-camera-fill-step-receipt-v1",
            "artifact_type": "wp7_camera_fill_step_receipt",
            "cohort_id": cohort["cohort_id"],
            "vial_id": vial["vial_id"],
            "claim_boundary": "camera_twin_geometry_equivalent_not_physical_ground_truth",
            "stages": stages,
            "calibration_sha256": "a" * 64,
            "recipe_artifact_sha256": "b" * 64,
            "source_artifact_sha256": "c" * 64,
        }
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        receipts.append(receipt)
    return receipts


def _result(disposition="keep", score=0.2):
    return {
        "disposition": disposition,
        "disposition_confidence": score,
        "explanation": "Target evidence is close to both references.",
        "evidence_citations": ["EV-C1-V1-S1", "EV-C1-V2-S1"],
        "image_citations": ["IMG-C1-V1-S1-end", "IMG-C1-V2-S1-end"],
    }


def _attempt(repository, raw):
    from fluidmix.wp7_vlm_decision import MODELS

    revision = next(row["revision"] for row in MODELS if row["repository"] == repository)
    return {"repository": repository, "revision": revision, "raw_response": raw}


def test_bundle_is_hash_bound_path_free_and_has_all_early_mid_end_references(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    payload_text = json.dumps(built.prompt_payload, sort_keys=True)
    assert str(tmp_path) not in payload_text
    assert all(locator.path.is_file() for locator in built.image_locators.values())
    assert len(built.image_locators) == 18
    assert {row["phase"] for row in built.prompt_payload["images"]} == set(PHASES)
    assert {row["vial_position"] for row in built.prompt_payload["images"]} == {1, 2, 3}
    assert built.prompt_payload["target_vial_position"] == 1
    scored = [row for row in built.prompt_payload["deterministic_evidence"] if "deterministic_disposition" in row]
    assert {row["vial_position"] for row in scored} == {1}
    assert built.prompt_payload["tracker_binding"]["claim_boundary"] == "camera_twin_geometry_equivalent_not_physical_ground_truth"
    assert built.prompt_payload["deterministic_binding"]["receipt_sha256"] == cohort["deterministic_decision_receipt"]["receipt_sha256"]


def test_bundle_recomputes_same_stage_delta_e_and_requires_bound_decision_receipt(tmp_path):
    cohort, locators = _cohort(tmp_path)
    cohort["vials"][0]["stages"][0]["measurement"]["delta_e76_vial2"] = 999.0
    with pytest.raises(ContractError, match="deltaE"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))

    cohort, locators = _cohort(tmp_path / "decision")
    cohort["deterministic_decision_receipt"]["stages"][0]["disposition"] = "discard"
    with pytest.raises(ContractError, match="receipt SHA"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))


def test_bundle_requires_exact_hash_bound_tracker_receipts_and_agreement(tmp_path):
    cohort, locators = _cohort(tmp_path)
    receipts = _tracker_receipts(cohort)
    with pytest.raises(ContractError, match="three tracker"):
        build_vial1_bundle(cohort, locators, tracker_receipts=receipts[:2])
    drifted = copy.deepcopy(receipts)
    drifted[0]["stages"][0]["agreement"] = False
    drifted[0]["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in drifted[0].items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    with pytest.raises(ContractError, match="agreement"):
        validate_tracker_receipt(drifted[0])
    extra = copy.deepcopy(receipts[0])
    extra["unexpected"] = True
    with pytest.raises(ContractError, match="closed"):
        validate_tracker_receipt(extra)


def test_bundle_rejects_non_target_scoring_incomplete_or_mismatched_references(tmp_path):
    cohort, locators = _cohort(tmp_path)
    cohort["target_vial_position"] = 2
    with pytest.raises(ContractError, match="vial 1"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    cohort, locators = _cohort(tmp_path / "missing")
    cohort["vials"].pop()
    with pytest.raises(ContractError, match="three vials"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    cohort, locators = _cohort(tmp_path / "mismatch")
    cohort["vials"][1]["stages"][0]["stage_id"] = "wrong-stage"
    with pytest.raises(ContractError, match="stage"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))


def test_bundle_rejects_duplicate_phase_unbound_hash_drift_and_path_like_value(tmp_path):
    cohort, locators = _cohort(tmp_path)
    cohort["vials"][0]["stages"][0]["frames"][1]["phase"] = "early"
    with pytest.raises(ContractError, match="image"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    cohort, locators = _cohort(tmp_path / "unbound")
    del locators[cohort["vials"][0]["stages"][0]["frames"][0]["image_id"]]
    with pytest.raises(ContractError, match="unbound"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    cohort, locators = _cohort(tmp_path / "drift")
    next(iter(locators.values()))["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="hash"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    cohort, locators = _cohort(tmp_path / "path")
    cohort["recipe_id"] = "../secret.json"
    with pytest.raises(ContractError, match="path-like"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))


@pytest.mark.parametrize("field", [
    "simulator_residue_ml", "palette_label", "oracle_mask", "semantic_mask",
    "particles", "simulator_truth", "test_label", "expected_lab",
    "beer_lambert_absorbance",
])
def test_bundle_rejects_forbidden_simulator_label_and_beer_lambert_fields(tmp_path, field):
    cohort, locators = _cohort(tmp_path)
    cohort["stages"][0][field] = 1
    with pytest.raises(ContractError, match="forbidden"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))


@pytest.mark.parametrize("field,value", [
    ("programmed_ml", float("nan")),
    ("programmed_ml", -1.0),
    ("stage_timing_seconds", [2.0, 1.0]),
])
def test_bundle_rejects_nonfinite_and_range_failures(tmp_path, field, value):
    cohort, locators = _cohort(tmp_path)
    tracker_receipts = _tracker_receipts(cohort)
    cohort["stages"][0][field] = value
    with pytest.raises(ContractError):
        build_vial1_bundle(cohort, locators, tracker_receipts=tracker_receipts)


def test_optional_camera_metadata_is_explicit_but_production_requires_core_fields(tmp_path):
    cohort, locators = _cohort(tmp_path)
    cohort["camera_metadata"] = {}
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    assert set(built.prompt_payload["camera_metadata"].values()) == {"unavailable"}
    cohort, locators = _cohort(tmp_path / "prod", partition="production")
    cohort["camera_metadata"]["resolution"] = None
    with pytest.raises(ContractError, match="production camera metadata"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort), production=True)


def test_parser_accepts_strict_and_bounded_fenced_json_but_rejects_malformed():
    expected = _result()
    assert parse_model_response(json.dumps(expected)) == expected
    assert parse_model_response(f"```json\n{json.dumps(expected)}\n```") == expected
    for raw in ("not json", "```json\n{}", "prefix {} suffix", "````json\n{}\n````"):
        with pytest.raises(ContractError):
            parse_model_response(raw)


def test_guard_validates_keep_discard_insufficient_score_and_exact_citations(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    for disposition, score, vote in (
        ("keep", 0.9, 0.1), ("discard", 0.8, 0.8),
        ("insufficient_evidence", 0.5, None),
    ):
        raw = _result(disposition, score)
        guarded = guard_model_response(raw, built.prompt_payload)
        if vote is None:
            assert guarded["vote"] is None
            assert guarded["discard_score"] is None
        else:
            assert guarded["vote"] == pytest.approx(vote)
            assert guarded["discard_score"] == pytest.approx(vote)
    invalid = _result()
    invalid["image_citations"] = ["IMG-NOT-DELIVERED"]
    assert guard_model_response(invalid, built.prompt_payload)["vote"] is None
    invalid = _result()
    invalid["explanation"] = "/tmp/leak.png"
    assert guard_model_response(invalid, built.prompt_payload)["vote"] is None
    invalid = _result(score=float("inf"))
    assert guard_model_response(invalid, built.prompt_payload)["vote"] is None


def test_equal_weight_aggregation_parses_guards_and_binds_unique_pinned_sources(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    qwen = "Qwen/Qwen3-VL-8B-Instruct"
    receipt = aggregate_decision(built.prompt_payload, [
        _attempt(qwen, json.dumps(_result("discard", 1.0))),
    ])
    assert receipt["mean_discard_score"] == 0.5
    assert receipt["final_disposition"] == "discard"
    only_one = aggregate_decision(built.prompt_payload, [])
    assert only_one["mean_discard_score"] == 0.0
    assert only_one["final_disposition"] == "insufficient_evidence"
    unknown = {"repository": "not-a-model", "revision": "x", "raw_response": "{}"}
    with pytest.raises(ContractError, match="pinned"):
        aggregate_decision(built.prompt_payload, [unknown])
    attempt = _attempt(qwen, json.dumps(_result()))
    with pytest.raises(ContractError, match="duplicate"):
        aggregate_decision(built.prompt_payload, [attempt, attempt])
    forged = {**attempt, "guard_status": "admitted"}
    with pytest.raises(ContractError, match="fields"):
        aggregate_decision(built.prompt_payload, [forged])
    tampered = copy.deepcopy(built.prompt_payload)
    tampered["deterministic_binding"]["disposition"] = "discard"
    with pytest.raises(ContractError, match="deterministic"):
        aggregate_decision(tampered, [])


def _development_manifest(tmp_path: Path, version: str, partition: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / f"{version}-{partition}.json"
    artifact.write_text("{}")
    manifest = {
        "schema_version": "wp7-vlm-development-input-manifest-v1",
        "dataset_version": version,
        "partition": partition,
        "data_origin": "synthetic_twin",
        "artifacts": [{
            "artifact_id": f"{version}-{partition}",
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }
    path = tmp_path / f"manifest-{version}-{partition}.json"
    path.write_text(json.dumps(manifest))
    return path


def test_freeze_binds_sources_models_and_blocks_production_open_until_matching(tmp_path):
    cohort, locators = _cohort(tmp_path / "dev")
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    freeze_path = tmp_path / "freeze.json"
    freeze = create_development_freeze(
        freeze_path, manifests, tracker_receipts=_tracker_receipts(cohort)
    )
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    assert len(freeze["models"]) == 3
    production_path = tmp_path / "production-bundle.json"
    production_path.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v1"}))
    assert load_production_bundle(
        production_path, freeze_path, expected_freeze_sha256=freeze_sha
    )["schema_version"] == "wp7-vlm-prompt-bundle-v1"
    missing = tmp_path / "missing-freeze.json"
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, missing, expected_freeze_sha256=freeze_sha)
    production_path.unlink()
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, missing, expected_freeze_sha256=freeze_sha)
    production_path.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v1"}))
    drifted = copy.deepcopy(freeze)
    drifted["aggregation"]["threshold"] = 0.6
    drifted_path = tmp_path / "drifted-freeze.json"
    drifted_path.write_text(json.dumps(drifted))
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, drifted_path, expected_freeze_sha256=freeze_sha)


def test_freeze_rejects_real_evaluation_test_and_video5_manifests(tmp_path):
    cohort, _ = _cohort(tmp_path / "dev")
    for partition in ("evaluation", "test", "real", "video5"):
        manifest = _development_manifest(tmp_path / partition, "V3", partition)
        with pytest.raises(ContractError, match="development manifest"):
            create_development_freeze(
                tmp_path / f"freeze-{partition}.json", [manifest],
                tracker_receipts=_tracker_receipts(cohort),
            )


def test_sequential_runtime_delivers_all_images_to_one_exact_model_at_a_time(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    active = []
    calls = []

    class Runtime:
        pass

    def loader(repository, revision, snapshot):
        assert not active
        active.append(repository)
        calls.append(("load", repository, revision, snapshot))
        return Runtime()

    def invoker(runtime, prompt, images):
        assert len(active) == 1
        assert len(images) == 18
        assert [row.image_id for row in images] == [
            row["image_id"] for row in built.prompt_payload["images"]
        ]
        repository = active.pop()
        calls.append(("invoke", repository, prompt, len(images)))
        return {"status": "completed", "response_raw": json.dumps(_result("keep", 0.9))}

    snapshots = {row["repository"]: tmp_path / row["repository"].replace("/", "-") for row in __import__("fluidmix.wp7_vlm_decision", fromlist=["MODELS"]).MODELS}
    receipt = run_sequential_vlm_runtime(
        built, snapshots, runtime_loader=loader, runtime_invoker=invoker
    )
    assert receipt["valid_source_count"] == 4
    assert receipt["mean_discard_score"] == pytest.approx(0.075)
    assert [row[0] for row in calls] == ["load", "invoke"] * 3
