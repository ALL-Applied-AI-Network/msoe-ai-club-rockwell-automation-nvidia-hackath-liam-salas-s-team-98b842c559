"""Focused synthetic contract tests for the WP7 image-grounded decision step."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
import fluidmix.wp7_vlm_decision as wp7


PHASES = ("early", "mid", "end")


def _cohort(
    tmp_path: Path, *, partition: str = "development",
    cohort_id: str = "cohort-1", vial_ids: tuple[str, str, str] = ("V1", "V2", "V3"),
):
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
                    "source_full_frame_sha256": hashlib.sha256(
                        f"full-{image_id}".encode()
                    ).hexdigest(),
                    "crop_pixel_sha256": hashlib.sha256(
                        f"pixels-{image_id}".encode()
                    ).hexdigest(),
                    "crop_box_xyxy": [10, 20, 100, 200],
                    "source_native_dimensions": [1920, 1080],
                    "crop_native_dimensions": [90, 180],
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
        vials.append({
            "vial_position": vial_position,
            "vial_id": vial_ids[vial_position - 1],
            "stages": vial_stages,
        })
    cohort = {
        "schema_version": "wp7-vlm-cohort-v1",
        "cohort_id": cohort_id,
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
        "cohort_id": cohort_id,
        "target_vial_id": vial_ids[0],
        "disposition": "keep",
        "limit_codes": [],
        "config_sha256": wp7.DETERMINISTIC_CONFIG_SHA256,
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
            "source_artifact_sha256": f"{vial['vial_position']:064x}",
        }
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        receipts.append(receipt)
    return receipts


def _result(disposition="keep", score=0.8):
    comparisons = [
        {
            "stage_id": f"stage-{ordinal}",
            "phase_and_fill_match": "matched",
            "vial1_vs_vial2": "keep_range",
            "vial1_vs_vial3": "keep_range",
            "deterministic_stage_disposition": "keep",
            "stream_tank_consistency": "consistent",
            "previous_recipe_direction": "not_assessable",
            "reliability": "clear_consistent",
            "bundle_ids": [f"EV-C1-V1-S{ordinal}"],
        }
        for ordinal in (1, 2)
    ]
    return {
        "observations": [
            {
                "stage_id": row["stage_id"],
                "visual_assessment": row["reliability"],
            }
            for row in comparisons
        ],
        "per_stage_comparison": comparisons,
        "disposition": disposition,
        "disposition_confidence": score,
        "explanation": (
            "The supplied evidence is insufficient for a supported disposition."
            if disposition == "insufficient_evidence"
            else "Clear stage evidence supports the structured disposition."
        ),
        "evidence_citations": ["EV-C1-V1-S1", "EV-C1-V2-S1"],
        "image_citations": ["IMG-C1-V1-S1-end", "IMG-C1-V2-S1-end"],
    }


def _attempt(repository, raw):
    from fluidmix.wp7_vlm_decision import MODELS

    revision = next(row["revision"] for row in MODELS if row["repository"] == repository)
    return {"repository": repository, "revision": revision, "raw_response": raw}


def _decoder_builder(_runtime, *, repository, response_schema):
    assert response_schema == wp7.RESPONSE_SCHEMA
    return object(), {
        "backend": "xgrammar", "version": "0.2.8", "license": "Apache-2.0",
        "decoder_id": "xgrammar-0.2.8-wp7-structural-v1",
        "decoder_spec_sha256": "d" * 64,
        "reasoning_mode": wp7.MODEL_REASONING_MODES[repository],
        "constrained_region": (
            "whole_response_json" if repository == wp7.MODELS[0]["repository"]
            else "answer_json_only"
        ),
    }


def _rehash_receipt(receipt):
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def test_bundle_is_hash_bound_path_free_and_has_all_early_mid_end_references(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    assert built.prompt_payload["schema_version"] == "wp7-vlm-prompt-bundle-v5"
    payload_text = json.dumps(built.prompt_payload, sort_keys=True)
    assert str(tmp_path) not in payload_text
    assert all(locator.path.is_file() for locator in built.image_locators.values())
    assert len(built.image_locators) == 18
    assert {row["phase"] for row in built.prompt_payload["images"]} == set(PHASES)
    assert {row["vial_position"] for row in built.prompt_payload["images"]} == {1, 2, 3}
    assert all(row["crop_box_xyxy"] == [10, 20, 100, 200] for row in built.prompt_payload["images"])
    assert all(row["source_native_dimensions"] == [1920, 1080] for row in built.prompt_payload["images"])
    assert all(row["crop_native_dimensions"] == [90, 180] for row in built.prompt_payload["images"])
    assert built.prompt_payload["target_vial_position"] == 1
    scored = [row for row in built.prompt_payload["deterministic_evidence"] if "deterministic_disposition" in row]
    assert {row["vial_position"] for row in scored} == {1}
    assert all(
        row["vial1_vs_vial2"] == "keep_range"
        and row["vial1_vs_vial3"] == "keep_range"
        for row in scored
    )
    references = [row for row in built.prompt_payload["deterministic_evidence"] if row["vial_position"] != 1]
    assert all("delta_e76_vial2" not in row and "delta_e76_vial3" not in row for row in references)
    assert all(
        "vial1_vs_vial2" not in row and "vial1_vs_vial3" not in row
        for row in references
    )
    assert built.prompt_payload["tracker_binding"]["claim_boundary"] == "camera_twin_geometry_equivalent_not_physical_ground_truth"
    assert built.prompt_payload["stages"][0]["camera_fill_state"]["measured_volume_ml"] == 1.0
    assert all(
        stage["tank_cluster_consistent"] is True
        for stage in built.prompt_payload["stages"]
    )
    assert built.prompt_payload["previous_recipe_direction_support"] == {
        "available": False,
        "reason": "previous_recipe_color_reference_unavailable",
    }
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


@pytest.mark.parametrize("mutation", ["config", "rule_call"])
def test_bundle_rejects_deterministic_receipt_outside_frozen_policy(tmp_path, mutation):
    cohort, locators = _cohort(tmp_path / mutation)
    receipt = cohort["deterministic_decision_receipt"]
    if mutation == "config":
        receipt["config_sha256"] = "0" * 64
    else:
        distance = 5.359287150896417
        computed_distance = (50.0 + distance) - 50.0
        for vial in cohort["vials"][1:]:
            for stage in vial["stages"]:
                stage["measurement"]["median_lab"] = [
                    50.0 + distance, 2.0, 3.0,
                ]
        for stage, decision_stage in zip(
            cohort["vials"][0]["stages"], receipt["stages"], strict=True
        ):
            stage["measurement"]["delta_e76_vial2"] = computed_distance
            stage["measurement"]["delta_e76_vial3"] = computed_distance
            decision_stage["delta_e76_vial2"] = computed_distance
            decision_stage["delta_e76_vial3"] = computed_distance
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()

    with pytest.raises(ContractError, match="deterministic.*(config|rule|policy)"):
        build_vial1_bundle(
            cohort, locators, tracker_receipts=_tracker_receipts(cohort)
        )


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


def test_bundle_accepts_per_vial_sources_and_rejects_stale_duplicate_or_mixed_cohort(tmp_path):
    cohort, locators = _cohort(
        tmp_path,
        cohort_id="RUN-7f4b9a2d",
        vial_ids=("CASE-V3-001", "CASE-V3-002", "CASE-V3-003"),
    )
    receipts = _tracker_receipts(cohort)
    build_vial1_bundle(cohort, locators, tracker_receipts=receipts)

    stale = copy.deepcopy(receipts)
    stale[0]["source_artifact_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="hash mismatch"):
        build_vial1_bundle(cohort, locators, tracker_receipts=stale)

    duplicate = copy.deepcopy(receipts)
    duplicate[1]["source_artifact_sha256"] = duplicate[0]["source_artifact_sha256"]
    _rehash_receipt(duplicate[1])
    with pytest.raises(ContractError, match="distinct"):
        build_vial1_bundle(cohort, locators, tracker_receipts=duplicate)

    mixed = copy.deepcopy(receipts)
    mixed[2]["cohort_id"] = "RUN-a19e65c0"
    _rehash_receipt(mixed[2])
    with pytest.raises(ContractError, match="identity mismatch"):
        build_vial1_bundle(cohort, locators, tracker_receipts=mixed)


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
    with pytest.raises(ContractError, match="resolution and frozen vial ROI"):
        build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
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
    invalid["evidence_citations"] = []
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "evidence_citations_invalid" in guarded["guard_reasons"]
    invalid = _result()
    invalid["image_citations"] = []
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "image_citations_invalid" in guarded["guard_reasons"]
    invalid = _result()
    invalid["explanation"] = "/tmp/leak.png"
    assert guard_model_response(invalid, built.prompt_payload)["vote"] is None
    invalid = _result()
    invalid["explanation"] = (
        "Stage reliability limits confidence in the structured disposition."
    )
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "explanation_template_mismatch" in guarded["guard_reasons"]
    invalid = _result(score=float("inf"))
    assert guard_model_response(invalid, built.prompt_payload)["vote"] is None
    invalid = _result()
    invalid["per_stage_comparison"] = []
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "per_stage_comparison_coverage_invalid" in guarded["guard_reasons"]
    invalid = _result()
    invalid["per_stage_comparison"][0]["bundle_ids"] = ["EV-C1-V1-S2"]
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "per_stage_comparison_bundle_ids_invalid" in guarded["guard_reasons"]
    invalid = _result()
    invalid["observations"][0]["bundle_ids"] = ["EV-C1-V1-S1"]
    guarded = guard_model_response(invalid, built.prompt_payload)
    assert "observations_invalid" in guarded["guard_reasons"]


def test_guard_binds_tracker_consistency_and_fails_closed_without_previous_color_reference(
    tmp_path,
):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=_tracker_receipts(cohort)
    )

    wrong_consistency = _result()
    wrong_consistency["per_stage_comparison"][0][
        "stream_tank_consistency"
    ] = "inconsistent"
    guarded = guard_model_response(wrong_consistency, built.prompt_payload)
    assert "stream_tank_consistency_mismatch" in guarded["guard_reasons"]

    unsupported_direction = _result()
    unsupported_direction["per_stage_comparison"][0][
        "previous_recipe_direction"
    ] = "toward_previous_recipe"
    guarded = guard_model_response(unsupported_direction, built.prompt_payload)
    assert "previous_recipe_direction_unsupported" in guarded["guard_reasons"]

    stage_local_images = _result()
    stage_local_images["per_stage_comparison"][0]["bundle_ids"] = [
        "IMG-C1-V1-S1-end", "IMG-C1-V2-S1-end",
    ]
    guarded = guard_model_response(stage_local_images, built.prompt_payload)
    assert guarded["guard_status"] == "admitted"

    later_stage_image = _result()
    later_stage_image["per_stage_comparison"][0]["bundle_ids"] = [
        "IMG-C1-V1-S2-end"
    ]
    guarded = guard_model_response(later_stage_image, built.prompt_payload)
    assert "per_stage_comparison_bundle_ids_invalid" in guarded["guard_reasons"]

    earlier_stage_image = _result()
    earlier_stage_image["per_stage_comparison"][1]["bundle_ids"] = [
        "IMG-C1-V1-S1-end"
    ]
    guarded = guard_model_response(earlier_stage_image, built.prompt_payload)
    assert "per_stage_comparison_bundle_ids_invalid" in guarded["guard_reasons"]

    assessment_mismatch = _result()
    assessment_mismatch["observations"][0]["visual_assessment"] = "weak"
    guarded = guard_model_response(assessment_mismatch, built.prompt_payload)
    assert "observation_comparison_binding_mismatch" in guarded["guard_reasons"]

    invalid_phase = _result()
    invalid_phase["per_stage_comparison"][0]["phase_and_fill_match"] = "looks matched"
    guarded = guard_model_response(invalid_phase, built.prompt_payload)
    assert "phase_and_fill_match_invalid" in guarded["guard_reasons"]

    missing_support = copy.deepcopy(built.prompt_payload)
    del missing_support["previous_recipe_direction_support"]
    guarded = guard_model_response(_result(), missing_support)
    assert "previous_recipe_direction_support_invalid" in guarded["guard_reasons"]

    assert guard_model_response(_result(), built.prompt_payload)[
        "guard_status"
    ] == "admitted"


def test_guard_caps_any_non_clear_stage_at_point_8(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=_tracker_receipts(cohort))
    clear = guard_model_response(_result("discard", 0.9), built.prompt_payload)
    assert clear["guard_status"] == "admitted"
    for reliability in ("weak", "low", "conflicting"):
        at_cap = _result("discard", 0.8)
        at_cap["per_stage_comparison"][0]["reliability"] = reliability
        at_cap["observations"][0]["visual_assessment"] = reliability
        at_cap["explanation"] = (
            "Stage reliability limits confidence in the structured disposition."
        )
        assert guard_model_response(at_cap, built.prompt_payload)["guard_status"] == "admitted"
        above_cap = _result("discard", 0.81)
        above_cap["per_stage_comparison"][0]["reliability"] = reliability
        above_cap["observations"][0]["visual_assessment"] = reliability
        above_cap["explanation"] = (
            "Stage reliability limits confidence in the structured disposition."
        )
        guarded = guard_model_response(above_cap, built.prompt_payload)
        assert guarded["guard_status"] == "excluded"
        assert "non_clear_stage_confidence_above_0_8" in guarded["guard_reasons"]


def test_guard_rejects_attempt08_threshold_contradiction_and_admits_exact_outcomes(tmp_path):
    cohort, locators = _cohort(tmp_path)
    target_stage = cohort["vials"][0]["stages"][1]["measurement"]
    cohort["vials"][1]["stages"][1]["measurement"]["median_lab"] = [59.53, 2.0, 3.0]
    cohort["vials"][2]["stages"][1]["measurement"]["median_lab"] = [55.36, 2.0, 3.0]
    distance_vial2 = math.dist([50.0, 2.0, 3.0], [59.53, 2.0, 3.0])
    distance_vial3 = math.dist([50.0, 2.0, 3.0], [55.36, 2.0, 3.0])
    target_stage.update({
        "delta_e76_vial2": distance_vial2,
        "delta_e76_vial3": distance_vial3,
        "deterministic_disposition": "discard",
    })
    receipt = cohort["deterministic_decision_receipt"]
    receipt["disposition"] = "discard"
    receipt["stages"][1].update({
        "delta_e76_vial2": distance_vial2,
        "delta_e76_vial3": distance_vial3,
        "disposition": "discard",
    })
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=_tracker_receipts(cohort)
    )

    stage2_evidence = next(
        row for row in built.prompt_payload["deterministic_evidence"]
        if row["vial_position"] == 1 and row["stage_id"] == "stage-2"
    )
    assert stage2_evidence["delta_e76_vial2"] == pytest.approx(9.53)
    assert stage2_evidence["delta_e76_vial3"] == pytest.approx(5.36)
    assert stage2_evidence["vial1_vs_vial2"] == "discard_range"
    assert stage2_evidence["vial1_vs_vial3"] == "discard_range"

    contradictory = _result("discard", 0.8)
    contradictory["per_stage_comparison"][1].update({
        "vial1_vs_vial2": "deadband",
        "vial1_vs_vial3": "deadband",
        "deterministic_stage_disposition": "discard",
    })
    guarded = guard_model_response(contradictory, built.prompt_payload)
    assert guarded["guard_status"] == "excluded"
    assert "per_stage_reference_relation_mismatch" in guarded["guard_reasons"]

    correct = _result("discard", 0.8)
    correct["per_stage_comparison"][1].update({
        "vial1_vs_vial2": "discard_range",
        "vial1_vs_vial3": "discard_range",
        "deterministic_stage_disposition": "discard",
    })
    assert guard_model_response(correct, built.prompt_payload)["guard_status"] == "admitted"

    tampered = copy.deepcopy(built.prompt_payload)
    tampered_stage2 = next(
        row for row in tampered["deterministic_evidence"]
        if row["vial_position"] == 1 and row["stage_id"] == "stage-2"
    )
    tampered_stage2["vial1_vs_vial2"] = "deadband"
    guarded = guard_model_response(correct, tampered)
    assert "prompt_reference_relation_mismatch" in guarded["guard_reasons"]


def test_guard_rejects_attempt09_threshold_prose_with_no_other_guard_failure(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=_tracker_receipts(cohort)
    )
    result = _result("discard", 0.8)
    result["explanation"] = (
        "Stage 2 dE values are above the threshold and in discard_range."
    )
    guarded = guard_model_response(result, built.prompt_payload)
    assert guarded["guard_status"] == "excluded"
    assert guarded["guard_reasons"] == ["explanation_template_invalid"]


def test_guard_admits_attempt10_observations_without_redundant_citations(tmp_path):
    cohort, locators = _cohort(tmp_path)
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=_tracker_receipts(cohort)
    )
    result = _result("discard", 0.8)
    result["per_stage_comparison"][0]["bundle_ids"] = [
        "IMG-C1-V1-S1-end", "IMG-C1-V2-S1-end",
    ]
    assert all(tuple(row) == ("stage_id", "visual_assessment") for row in result["observations"])
    guarded = guard_model_response(result, built.prompt_payload)
    assert guarded["guard_status"] == "admitted"
    assert guarded["guard_reasons"] == []


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


def _materialization_receipt(tmp_path, manifests, tracker_receipts):
    value = {
        "schema_version": "wp7-vlm-development-bundle-materialization-v1",
        "status": "complete",
        "development_manifest_sha256": {
            json.loads(Path(path).read_text())["dataset_version"]:
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in manifests
        },
        "source_bindings": {
            "deterministic_config_sha256": wp7.DETERMINISTIC_CONFIG_SHA256,
            "tracker_receipt_sha256": [
                receipt["receipt_sha256"] for receipt in tracker_receipts
            ],
            "tracker_source_sha256": hashlib.sha256(
                Path(wp7.__file__).with_name("wp7_fill_step_tracker.py").read_bytes()
            ).hexdigest(),
        },
    }
    value["receipt_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    path = tmp_path / "materialization-receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_freeze_binds_sources_models_and_blocks_production_open_until_matching(tmp_path):
    cohort, locators = _cohort(tmp_path / "dev")
    tracker_receipts = _tracker_receipts(cohort)
    built = build_vial1_bundle(cohort, locators, tracker_receipts=tracker_receipts)
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    freeze_path = tmp_path / "freeze.json"
    materialization = _materialization_receipt(
        tmp_path, manifests, tracker_receipts
    )
    freeze = create_development_freeze(
        freeze_path, manifests, tracker_receipts=tracker_receipts,
        materialization_receipt=materialization,
    )
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    assert len(freeze["models"]) == 3
    assert freeze["schema_version"] == "wp7-vlm-development-freeze-v11"
    assert freeze["prompt_id"] == "wp7-grounded-sequential-mixture-v8"
    assert freeze["response_schema_id"] == "wp7-vlm-reasoned-response-v6"
    assert freeze["deterministic_policy"] == {
        "config_sha256": wp7.DETERMINISTIC_CONFIG_SHA256,
        **wp7.DETERMINISTIC_POLICY,
    }
    assert freeze["generation_config"] == wp7.GENERATION_CONFIG
    assert freeze["development_materialization_receipt_sha256"] == (
        hashlib.sha256(materialization.read_bytes()).hexdigest()
    )
    assert freeze["tracker_source_artifact_sha256"] == sorted(
        receipt["source_artifact_sha256"] for receipt in tracker_receipts
    )
    duplicate = copy.deepcopy(tracker_receipts)
    duplicate[1]["source_artifact_sha256"] = duplicate[0]["source_artifact_sha256"]
    _rehash_receipt(duplicate[1])
    with pytest.raises(ContractError, match="distinct"):
        create_development_freeze(
            tmp_path / "duplicate-source-freeze.json",
            manifests,
            tracker_receipts=duplicate,
            materialization_receipt=_materialization_receipt(
                tmp_path / "duplicate", manifests, duplicate
            ),
        )
    production_path = tmp_path / "production-bundle.json"
    production_path.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v5"}))
    assert load_production_bundle(
        production_path, freeze_path, expected_freeze_sha256=freeze_sha
    )["schema_version"] == "wp7-vlm-prompt-bundle-v5"
    missing = tmp_path / "missing-freeze.json"
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, missing, expected_freeze_sha256=freeze_sha)
    production_path.unlink()
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, missing, expected_freeze_sha256=freeze_sha)
    production_path.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v5"}))
    drifted = copy.deepcopy(freeze)
    drifted["aggregation"]["threshold"] = 0.6
    drifted_path = tmp_path / "drifted-freeze.json"
    drifted_path.write_text(json.dumps(drifted))
    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(production_path, drifted_path, expected_freeze_sha256=freeze_sha)


def test_freeze_rejects_materialization_receipt_provenance_drift(tmp_path):
    cohort, _ = _cohort(tmp_path / "dev")
    receipts = _tracker_receipts(cohort)
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    materialization = _materialization_receipt(
        tmp_path / "materialized", manifests, receipts
    )
    value = json.loads(materialization.read_text())
    value["source_bindings"]["deterministic_config_sha256"] = "0" * 64
    value["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: item for key, item in value.items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    materialization.write_text(json.dumps(value))

    with pytest.raises(ContractError, match="materialization receipt bindings"):
        create_development_freeze(
            tmp_path / "freeze.json", manifests, tracker_receipts=receipts,
            materialization_receipt=materialization,
        )


def test_freeze_binds_complete_real_shape_tracker_cohort(tmp_path):
    cohort, _ = _cohort(
        tmp_path / "dev",
        cohort_id="RUN-7f4b9a2d",
        vial_ids=("CASE-V3-001", "CASE-V3-002", "CASE-V3-003"),
    )
    receipts = _tracker_receipts(cohort)
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    freeze = create_development_freeze(
        tmp_path / "real-shape-freeze.json",
        manifests,
        tracker_receipts=receipts,
        materialization_receipt=_materialization_receipt(
            tmp_path / "real-shape", manifests, receipts
        ),
    )

    assert freeze["tracker_cohort_id"] == "RUN-7f4b9a2d"
    assert freeze["tracker_vial_ids"] == [
        "CASE-V3-001", "CASE-V3-002", "CASE-V3-003",
    ]
    assert freeze["tracker_calibration_sha256"] == ["a" * 64]
    assert freeze["tracker_recipe_artifact_sha256"] == ["b" * 64]
    assert freeze["tracker_claim_boundary"] == (
        "camera_twin_geometry_equivalent_not_physical_ground_truth"
    )
    assert freeze["tracker_recipe_stages"] == [
        {
            "fill_step": 1,
            "tank_id": "t100",
            "tank_number": 100,
            "stream_step_index": 1,
            "expected_dispense_ml": 1.0,
            "expected_cumulative_ml": 1.0,
        },
        {
            "fill_step": 2,
            "tank_id": "t200",
            "tank_number": 200,
            "stream_step_index": 2,
            "expected_dispense_ml": 2.0,
            "expected_cumulative_ml": 3.0,
        },
    ]
    assert freeze["tracker_receipt_sha256"] == [
        receipt["receipt_sha256"] for receipt in receipts
    ]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("duplicate_vial", "vial"),
        ("calibration", "calibration"),
        ("recipe", "recipe_artifact"),
        ("claim", "claim boundary"),
        ("stage", "recipe state"),
    ],
)
def test_freeze_rejects_mixed_tracker_cohort_invariants(tmp_path, case, expected):
    cohort, _ = _cohort(tmp_path / "dev")
    receipts = _tracker_receipts(cohort)
    if case == "duplicate_vial":
        receipts[1]["vial_id"] = receipts[0]["vial_id"]
    elif case == "calibration":
        receipts[1]["calibration_sha256"] = "9" * 64
    elif case == "recipe":
        receipts[1]["recipe_artifact_sha256"] = "8" * 64
    elif case == "claim":
        receipts[1]["claim_boundary"] = "different_claim_boundary"
    else:
        receipts[1]["stages"][0]["expected_dispense_ml"] = 1.5
        receipts[1]["stages"][0]["expected_cumulative_ml"] = 1.5
        receipts[1]["stages"][1]["expected_cumulative_ml"] = 3.5
    _rehash_receipt(receipts[1])
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]

    with pytest.raises(ContractError, match=expected):
        create_development_freeze(
            tmp_path / f"{case}-freeze.json",
            manifests,
            tracker_receipts=receipts,
            materialization_receipt=_materialization_receipt(
                tmp_path / case, manifests, receipts
            ),
        )


@pytest.mark.parametrize("tamper", [
    "extra_field", "cohort", "vials", "calibrations", "recipes", "claim",
    "recipe_stages", "sources",
])
def test_load_rejects_tampered_tracker_freeze_binding(tmp_path, tamper):
    cohort, _ = _cohort(tmp_path / "dev")
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    receipts = _tracker_receipts(cohort)
    freeze = create_development_freeze(
        tmp_path / "freeze.json",
        manifests,
        tracker_receipts=receipts,
        materialization_receipt=_materialization_receipt(
            tmp_path / "materialized", manifests, receipts
        ),
    )
    if tamper == "extra_field":
        freeze["unexpected"] = True
    elif tamper == "cohort":
        freeze["tracker_cohort_id"] = ""
    elif tamper == "vials":
        freeze["tracker_vial_ids"][1] = freeze["tracker_vial_ids"][0]
    elif tamper == "calibrations":
        freeze["tracker_calibration_sha256"].append("9" * 64)
    elif tamper == "recipes":
        freeze["tracker_recipe_artifact_sha256"].append("8" * 64)
    elif tamper == "claim":
        freeze["tracker_claim_boundary"] = "different_claim_boundary"
    elif tamper == "recipe_stages":
        freeze["tracker_recipe_stages"][0]["unexpected"] = True
    else:
        freeze["tracker_source_artifact_sha256"][1] = (
            freeze["tracker_source_artifact_sha256"][0]
        )
    tampered = tmp_path / f"tampered-{tamper}.json"
    tampered.write_text(json.dumps(freeze))
    expected_sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    production = tmp_path / "production.json"
    production.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v5"}))

    with pytest.raises(ContractError, match="freeze"):
        load_production_bundle(
            production,
            tampered,
            expected_freeze_sha256=expected_sha,
        )


def test_load_rejects_generation_config_drift(tmp_path):
    cohort, _ = _cohort(tmp_path / "dev")
    manifests = [
        _development_manifest(tmp_path, "V3", "twin_train"),
        _development_manifest(tmp_path, "V4", "twin_validation"),
    ]
    receipts = _tracker_receipts(cohort)
    freeze = create_development_freeze(
        tmp_path / "freeze.json",
        manifests,
        tracker_receipts=receipts,
        materialization_receipt=_materialization_receipt(
            tmp_path / "materialized", manifests, receipts
        ),
    )
    freeze["generation_config"]["max_new_tokens"] = 769
    drifted = tmp_path / "drifted-generation-freeze.json"
    drifted.write_text(json.dumps(freeze))
    expected_sha = hashlib.sha256(drifted.read_bytes()).hexdigest()
    production = tmp_path / "production.json"
    production.write_text(json.dumps({"schema_version": "wp7-vlm-prompt-bundle-v5"}))

    with pytest.raises(ContractError, match="freeze constants"):
        load_production_bundle(
            production,
            drifted,
            expected_freeze_sha256=expected_sha,
        )


def test_freeze_rejects_real_evaluation_test_and_video5_manifests(tmp_path):
    cohort, _ = _cohort(tmp_path / "dev")
    for partition in ("evaluation", "test", "real", "video5"):
        manifest = _development_manifest(tmp_path / partition, "V3", partition)
        receipts = _tracker_receipts(cohort)
        with pytest.raises(ContractError, match="development manifest"):
            create_development_freeze(
                tmp_path / f"freeze-{partition}.json", [manifest],
                tracker_receipts=receipts,
                materialization_receipt=_materialization_receipt(
                    tmp_path / partition / "materialized", [manifest], receipts
                ),
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

    def invoker(runtime, prompt, images, *, max_new_tokens, logits_processor, processor_pixels):
        assert len(active) == 1
        assert max_new_tokens == 10000
        assert logits_processor is not None
        assert len(images) == 18
        assert [row.image_id for row in images] == [
            row["image_id"] for row in built.prompt_payload["images"]
        ]
        repository = active.pop()
        calls.append(("invoke", repository, prompt, len(images)))
        raw = json.dumps(_result("keep", 0.9))
        if repository != wp7.MODELS[0]["repository"]:
            raw = f"<think>compare</think><answer>{raw}</answer>"
        return {"status": "completed", "response_raw": raw}

    snapshots = {row["repository"]: tmp_path / row["repository"].replace("/", "-") for row in __import__("fluidmix.wp7_vlm_decision", fromlist=["MODELS"]).MODELS}
    receipt = run_sequential_vlm_runtime(
        built, snapshots, runtime_loader=loader, runtime_invoker=invoker,
        constrained_decoder_builder=_decoder_builder,
    )
    assert receipt["valid_source_count"] == 4
    assert receipt["mean_discard_score"] == pytest.approx(0.075)
    assert [row[0] for row in calls] == ["load", "invoke"] * 3


def test_runtime_capabilities_are_required_before_bundle_or_file_access(tmp_path):
    with pytest.raises(TypeError, match="runtime_loader.*runtime_invoker"):
        run_sequential_vlm_runtime(None, {})

    output = tmp_path / "attempt.json"
    with pytest.raises(TypeError, match="runtime_loader.*runtime_invoker"):
        wp7.run_one_model_attempt_from_files(
            tmp_path / "missing-cohort.json",
            tmp_path / "missing-locators.json",
            [tmp_path / "missing-tracker.json"] * 3,
            freeze_path=tmp_path / "missing-freeze.json",
            expected_freeze_sha256="0" * 64,
            repository=wp7.MODELS[0]["repository"],
            revision=wp7.MODELS[0]["revision"],
            snapshot_path=tmp_path / "missing-snapshot",
            output_path=output,
        )
    assert not output.exists()


def test_wp7_model_pins_remain_exact_and_local():
    assert wp7.MODELS == (
        {
            "repository": "Qwen/Qwen3-VL-8B-Instruct",
            "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        },
        {
            "repository": "nvidia/Cosmos-Reason2-8B",
            "revision": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
        },
        {
            "repository": "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
            "revision": "8331bc79ecf01e3466f0e6a27cec8c77f182f430",
        },
    )


def _runtime_files(tmp_path):
    cohort, locators = _cohort(tmp_path / "images")
    trackers = _tracker_receipts(cohort)
    cohort_path = tmp_path / "cohort.json"
    locator_path = tmp_path / "locators.json"
    cohort_path.write_text(json.dumps(cohort))
    published_locators = {
        image_id: {
            **row,
            "path": str(Path(row["path"]).relative_to(tmp_path)),
        }
        for image_id, row in locators.items()
    }
    locator_path.write_text(json.dumps(published_locators))
    tracker_paths = []
    for index, receipt in enumerate(trackers, 1):
        path = tmp_path / f"tracker-{index}.json"
        path.write_text(json.dumps(receipt))
        tracker_paths.append(path)
    manifests = [
        _development_manifest(tmp_path / "freeze-inputs", "V3", "twin_train"),
        _development_manifest(tmp_path / "freeze-inputs", "V4", "twin_validation"),
    ]
    freeze_path = tmp_path / "freeze.json"
    create_development_freeze(
        freeze_path, manifests, tracker_receipts=trackers,
        materialization_receipt=_materialization_receipt(
            tmp_path / "freeze-materialized", manifests, trackers
        ),
    )
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    return cohort_path, locator_path, tracker_paths, freeze_path, freeze_sha


def test_one_model_runtime_publishes_exclusive_bound_raw_receipt(tmp_path):
    cohort, locators, trackers, freeze, freeze_sha = _runtime_files(tmp_path)
    spec = wp7.MODELS[0]
    calls = []

    class Runtime:
        inventory = {"files": [{"path": "config.json", "bytes": 2, "sha256": "a" * 64}], "sha256": "b" * 64}

    def loader(repository, revision, snapshot):
        calls.append(("load", repository, revision, Path(snapshot)))
        return Runtime()

    def invoker(runtime, prompt, images, *, max_new_tokens, logits_processor, processor_pixels):
        assert max_new_tokens == 10000
        assert logits_processor is not None
        assert processor_pixels == {"min_pixels": 1, "max_pixels": 16777216}
        calls.append(("invoke", hashlib.sha256(prompt.encode()).hexdigest(), len(images)))
        return {
            "status": "completed",
            "response_raw": json.dumps(_result("keep", 0.9)),
            "response_utf8_sha256": hashlib.sha256(json.dumps(_result("keep", 0.9)).encode()).hexdigest(),
            "generated_token_count": 31,
            "prompt_token_count": 2048,
            "model_context_limit": 262144,
            "elapsed_seconds": 1.25,
            "peak_cuda_memory_allocated_bytes": 100,
            "peak_cuda_memory_reserved_bytes": 200,
            "peak_memory_bytes": 100,
            "processor_telemetry": {"delivered_image_count": 18},
            "attempt_index": 0,
        }

    output = tmp_path / "qwen-attempt.json"
    receipt = wp7.run_one_model_attempt_from_files(
        cohort, locators, trackers,
        freeze_path=freeze, expected_freeze_sha256=freeze_sha,
        repository=spec["repository"], revision=spec["revision"],
        snapshot_path=tmp_path / "snapshot", output_path=output,
        runtime_loader=loader, runtime_invoker=invoker,
        constrained_decoder_builder=_decoder_builder,
    )

    assert output.is_file()
    assert receipt["schema_version"] == "wp7-vlm-model-attempt-receipt-v2"
    assert receipt["model"]["trust_remote_code"] is False
    assert receipt["model"]["snapshot_inventory_sha256"] == "b" * 64
    assert receipt["invoke"]["elapsed_seconds"] == 1.25
    assert receipt["invoke"]["peak_cuda_memory_reserved_bytes"] == 200
    assert receipt["invoke"]["prompt_token_count"] == 2048
    assert receipt["invoke"]["model_context_limit"] == 262144
    assert receipt["token_budget"] == {
        "requested_max_new_tokens": 10000,
        "effective_max_new_tokens": 10000,
        "fallback_used": False,
        "fallback_proof_receipt_sha256": None,
        "prompt_token_count": 2048,
        "model_context_limit": 262144,
        "compatibility": "prompt_plus_10000_lte_context",
    }
    assert receipt["constrained_decoding"]["reasoning_mode"] == "schema_json"
    assert receipt["determinism"]["seed"] == 17
    assert receipt["binding"]["generation_config"] == wp7.GENERATION_CONFIG
    assert "processor_telemetry" not in receipt["invoke"]
    assert len(receipt["images"]) == 18
    assert all(set(row) == {
        "image_id", "crop_sha256", "source_full_frame_sha256",
        "crop_pixel_sha256", "crop_box_xyxy", "source_native_dimensions",
        "crop_native_dimensions",
    } for row in receipt["images"])
    assert str(tmp_path) not in json.dumps(receipt["prompt"])
    assert [row[0] for row in calls] == ["load", "invoke"]
    with pytest.raises(ContractError, match="overwrite"):
        wp7.run_one_model_attempt_from_files(
            cohort, locators, trackers,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
            repository=spec["repository"], revision=spec["revision"],
            snapshot_path=tmp_path / "snapshot", output_path=output,
            runtime_loader=loader, runtime_invoker=invoker,
        )


def test_runtime_validates_reviewed_freeze_before_model_load(tmp_path):
    cohort, locators, trackers, freeze, _freeze_sha = _runtime_files(tmp_path)
    called = False

    def loader(*args):
        nonlocal called
        called = True
        raise AssertionError("must not load")

    with pytest.raises(ContractError, match="reviewed SHA"):
        wp7.run_one_model_attempt_from_files(
            cohort, locators, trackers,
            freeze_path=freeze, expected_freeze_sha256="0" * 64,
            repository=wp7.MODELS[0]["repository"], revision=wp7.MODELS[0]["revision"],
            snapshot_path=tmp_path / "snapshot", output_path=tmp_path / "attempt.json",
            runtime_loader=loader, runtime_invoker=lambda *_args, **_kwargs: None,
        )
    assert called is False


def test_non_qwen_runtime_requires_guarded_qwen_attempt_before_load(tmp_path):
    cohort, locators, trackers, freeze, freeze_sha = _runtime_files(tmp_path)
    called = False

    def loader(*args):
        nonlocal called
        called = True
        raise AssertionError("must not load")

    cosmos = wp7.MODELS[1]
    with pytest.raises(ContractError, match="Qwen attempt first"):
        wp7.run_one_model_attempt_from_files(
            cohort, locators, trackers,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
            repository=cosmos["repository"], revision=cosmos["revision"],
            snapshot_path=tmp_path / "snapshot", output_path=tmp_path / "cosmos.json",
            runtime_loader=loader, runtime_invoker=lambda *_args, **_kwargs: None,
        )
    assert called is False


def test_production_mode_binds_new_tracker_cohort_after_reviewed_policy_freeze(tmp_path):
    _dev_cohort, _dev_locators, _dev_trackers, freeze, freeze_sha = _runtime_files(
        tmp_path / "development"
    )
    cohort, locators = _cohort(
        tmp_path / "production-images", partition="production",
        cohort_id="PROD-C1", vial_ids=("PROD-V1", "PROD-V2", "PROD-V3"),
    )
    trackers = _tracker_receipts(cohort)
    cohort_path = tmp_path / "production-cohort.json"
    locator_path = tmp_path / "production-locators.json"
    cohort_path.write_text(json.dumps(cohort))
    locator_path.write_text(json.dumps(locators))
    tracker_paths = []
    for index, tracker in enumerate(trackers, 1):
        path = tmp_path / f"production-tracker-{index}.json"
        path.write_text(json.dumps(tracker))
        tracker_paths.append(path)

    development_value = copy.deepcopy(cohort)
    development_value["partition"] = "development"
    different_development_path = tmp_path / "different-development-cohort.json"
    different_development_path.write_text(json.dumps(development_value))
    with pytest.raises(ContractError, match="freeze does not bind"):
        wp7.load_runtime_context(
            different_development_path, locator_path, tracker_paths,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
        )

    class Runtime:
        inventory = {"files": [{"path": "config.json", "bytes": 2, "sha256": "a" * 64}], "sha256": "b" * 64}

    raw = json.dumps(_result("keep", 0.9))

    def invoker(_runtime, _prompt, _images, *, max_new_tokens, logits_processor, processor_pixels):
        assert max_new_tokens == 10000
        return {
            "status": "completed", "response_raw": raw,
            "response_utf8_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "generated_token_count": 10, "prompt_token_count": 2048,
            "model_context_limit": 262144, "elapsed_seconds": 0.5,
            "peak_cuda_memory_allocated_bytes": 1,
            "peak_cuda_memory_reserved_bytes": 2, "peak_memory_bytes": 1,
        }

    receipt = wp7.run_one_model_attempt_from_files(
        cohort_path, locator_path, tracker_paths,
        freeze_path=freeze, expected_freeze_sha256=freeze_sha,
        repository=wp7.MODELS[0]["repository"], revision=wp7.MODELS[0]["revision"],
        snapshot_path=tmp_path / "snapshot", output_path=tmp_path / "production-attempt.json",
        production=True, runtime_loader=lambda *args: Runtime(),
        runtime_invoker=invoker, constrained_decoder_builder=_decoder_builder,
    )
    assert receipt["cohort_id"] == "PROD-C1"
    assert receipt["binding"]["production"] is True
    assert receipt["binding"]["tracker_receipt_sha256"] == [
        row["receipt_sha256"] for row in trackers
    ]


@pytest.mark.parametrize("config_sha256", ["0" * 64, wp7.DETERMINISTIC_CONFIG_SHA256])
def test_production_runtime_rejects_self_consistent_wrong_deterministic_call(
    tmp_path, config_sha256
):
    _dev_cohort, _dev_locators, _dev_trackers, freeze, freeze_sha = _runtime_files(
        tmp_path / "development"
    )
    cohort, locators = _cohort(
        tmp_path / "production-images", partition="production",
        cohort_id="PROD-C-WRONG-CALL",
        vial_ids=("PROD-WRONG-V1", "PROD-WRONG-V2", "PROD-WRONG-V3"),
    )
    requested_distance = 5.359287150896417
    computed_distance = (50.0 + requested_distance) - 50.0
    for vial in cohort["vials"][1:]:
        for stage in vial["stages"]:
            stage["measurement"]["median_lab"] = [
                50.0 + requested_distance, 2.0, 3.0,
            ]
    decision = cohort["deterministic_decision_receipt"]
    decision["config_sha256"] = config_sha256
    for stage, decision_stage in zip(
        cohort["vials"][0]["stages"], decision["stages"], strict=True
    ):
        stage["measurement"]["delta_e76_vial2"] = computed_distance
        stage["measurement"]["delta_e76_vial3"] = computed_distance
        decision_stage["delta_e76_vial2"] = computed_distance
        decision_stage["delta_e76_vial3"] = computed_distance
    decision["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in decision.items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    cohort_path = tmp_path / "production-cohort.json"
    locator_path = tmp_path / "production-locators.json"
    cohort_path.write_text(json.dumps(cohort))
    locator_path.write_text(json.dumps(locators))
    tracker_paths = []
    for ordinal, tracker in enumerate(_tracker_receipts(cohort), 1):
        path = tmp_path / f"production-tracker-{ordinal}.json"
        path.write_text(json.dumps(tracker))
        tracker_paths.append(path)

    expected = "deterministic config" if config_sha256 == "0" * 64 else "deterministic rule"
    with pytest.raises(ContractError, match=expected):
        wp7.load_runtime_context(
            cohort_path, locator_path, tracker_paths,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
            production=True,
        )


def test_freeze_source_drift_fails_before_cohort_access(tmp_path):
    _cohort_path, _locators, _trackers, freeze, _freeze_sha = _runtime_files(tmp_path)
    drifted = json.loads(freeze.read_text())
    drifted["decision_source_sha256"] = "0" * 64
    drifted_path = tmp_path / "drifted-freeze.json"
    drifted_path.write_text(json.dumps(drifted))
    drifted_sha = hashlib.sha256(drifted_path.read_bytes()).hexdigest()

    with pytest.raises(ContractError, match="freeze source"):
        wp7.load_runtime_context(
            tmp_path / "missing-cohort.json",
            tmp_path / "missing-locators.json",
            [tmp_path / "missing-1.json", tmp_path / "missing-2.json", tmp_path / "missing-3.json"],
            freeze_path=drifted_path,
            expected_freeze_sha256=drifted_sha,
            production=True,
        )


def test_runtime_mode_must_match_production_partition(tmp_path):
    cohort, locators, trackers, freeze, freeze_sha = _runtime_files(tmp_path)
    with pytest.raises(ContractError, match="production partition"):
        wp7.load_runtime_context(
            cohort, locators, trackers,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
            production=True,
        )

    production_value = json.loads(cohort.read_text())
    production_value["partition"] = "production"
    production_path = tmp_path / "production-cohort.json"
    production_path.write_text(json.dumps(production_value))
    with pytest.raises(ContractError, match="production runtime mode"):
        wp7.load_runtime_context(
            production_path, locators, trackers,
            freeze_path=freeze, expected_freeze_sha256=freeze_sha,
        )


def test_cpu_aggregate_accepts_zero_to_three_bound_attempts_and_rejects_tamper(tmp_path):
    cohort, locators, trackers, freeze, freeze_sha = _runtime_files(tmp_path)
    empty_output = tmp_path / "empty-aggregation.json"
    empty = wp7.aggregate_model_attempts_from_files(
        cohort, locators, trackers, [], freeze_path=freeze,
        expected_freeze_sha256=freeze_sha, output_path=empty_output,
    )
    assert empty["decision"]["valid_source_count"] == 1
    assert empty["decision"]["final_disposition"] == "insufficient_evidence"

    spec = wp7.MODELS[0]

    class Runtime:
        inventory = {"files": [{"path": "config.json", "bytes": 2, "sha256": "a" * 64}], "sha256": "b" * 64}

    raw = json.dumps(_result("keep", 0.9))
    attempt_path = tmp_path / "attempt.json"

    def invoker(_runtime, _prompt, _images, *, max_new_tokens, logits_processor, processor_pixels):
        assert max_new_tokens == 10000
        return {
            "status": "completed", "response_raw": raw,
            "response_utf8_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "generated_token_count": 10, "prompt_token_count": 2048,
            "model_context_limit": 262144, "elapsed_seconds": 0.5,
            "peak_cuda_memory_allocated_bytes": 1,
            "peak_cuda_memory_reserved_bytes": 2, "peak_memory_bytes": 1,
        }

    wp7.run_one_model_attempt_from_files(
        cohort, locators, trackers,
        freeze_path=freeze, expected_freeze_sha256=freeze_sha,
        repository=spec["repository"], revision=spec["revision"],
        snapshot_path=tmp_path / "snapshot", output_path=attempt_path,
        runtime_loader=lambda *args: Runtime(),
        runtime_invoker=invoker, constrained_decoder_builder=_decoder_builder,
    )
    published = wp7.aggregate_model_attempts_from_files(
        cohort, locators, trackers, [attempt_path], freeze_path=freeze,
        expected_freeze_sha256=freeze_sha, output_path=tmp_path / "aggregation.json",
    )
    assert published["decision"]["valid_source_count"] == 2
    assert published["decision"]["final_disposition"] == "keep"

    tampered = json.loads(attempt_path.read_text())
    tampered["binding"]["bundle_sha256"] = "f" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ContractError, match="receipt SHA|binding"):
        wp7.aggregate_model_attempts_from_files(
            cohort, locators, trackers, [tampered_path], freeze_path=freeze,
            expected_freeze_sha256=freeze_sha, output_path=tmp_path / "bad.json",
        )
