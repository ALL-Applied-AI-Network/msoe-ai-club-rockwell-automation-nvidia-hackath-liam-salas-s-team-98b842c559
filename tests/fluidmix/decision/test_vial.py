"""Pure final WP7 decision and guarded explanation contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement_binding(tmp_path: Path, ordinal: int, *, valid: bool = True):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    image_id = f"TARGET-IMAGE-{ordinal}"
    rgb = tmp_path / "camera" / f"{image_id}-rgb.png"
    mask = tmp_path / "camera" / f"{image_id}-predicted-mask.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=(80 + ordinal, 100, 120)).save(rgb)
    pixels = np.zeros((3, 4), dtype=np.uint8)
    if valid:
        pixels[1:, 1:3] = 255
    Image.fromarray(pixels, mode="L").save(mask)
    artifact = {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": {
            "producer_id": "segmenter-1", "producer_kind": "rgb_only_liquid_segmenter",
            "model_id": "model-1", "model_sha256": "a" * 64,
            "config_id": "config-1", "config_sha256": "b" * 64,
            "code_commit": "c" * 40, "threshold_source": "twin_validation_frozen",
            "training_data_origin": "twin", "input_fields": ["rgb"],
        },
        "images": {image_id: {
            "rgb_path": str(rgb.resolve()), "rgb_sha256": _sha(rgb),
            "mask_path": str(mask.resolve()), "mask_sha256": _sha(mask),
        }},
    }
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    return measure_trusted_predicted_image(
        registry.entry(image_id), measurement_id=f"TARGET-MEASUREMENT-{ordinal}",
        evidence_id=f"TARGET-EVIDENCE-{ordinal}",
    )


def _target(tmp_path: Path, ordinal: int, *, valid: bool = True):
    from fluidmix.decision.vial import bind_runtime_camera_stage

    _, binding = _measurement_binding(tmp_path, ordinal, valid=valid)
    return bind_runtime_camera_stage({
        "vial_id": "VIAL-1", "recipe_id": "R-A",
        "stage_id": f"RUNTIME-STAGE-{ordinal}", "stage_ordinal": ordinal,
        "tank_id": {1: "t100", 2: "t200", 3: "t400"}[ordinal],
        "source": "runtime_camera",
    }, binding)


def _program(targets):
    return [{
        "recipe_id": target.recipe_id, "stage_id": target.stage_id,
        "stage_ordinal": target.stage_ordinal, "tank_id": target.tank_id,
    } for target in targets]


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _store(tmp_path: Path, targets, *, center_offsets=None, peer_delta=2.0,
           configured_threshold=None):
    from fluidmix.memory.history.development_reference import DevelopmentReferenceStore, _create_minimal_store

    center_offsets = center_offsets or {}
    records = []
    configurations = []
    cursor = 0
    threshold = peer_delta / 2.0 + 1.0 if configured_threshold is None else configured_threshold
    for batch, run, sign in (("INN67-DEV-RB-B01", "RUN-B01", -1.0),
                             ("INN67-DEV-RB-B02", "RUN-B02", 1.0)):
        for target in targets:
            ordinal = target.stage_ordinal
            center = list(target.median_lab or (50.0, float(ordinal), 0.0))
            center[0] += center_offsets.get(ordinal, 0.0)
            lab = center.copy()
            lab[1] += sign * peer_delta / 2.0
            evidence_id = f"DONOR-{batch}-{ordinal}"
            image_id = f"DONOR-IMAGE-{batch}-{ordinal}"
            rgb = tmp_path / "donors" / f"{image_id}-rgb.png"
            mask = tmp_path / "donors" / f"{image_id}-mask.png"
            records.append({
                "evidence_id": evidence_id, "event_cursor": cursor, "batch_id": batch,
                "run_id": run, "case_id": f"CASE-{batch}", "recipe_id": target.recipe_id,
                "stage_id": f"DEV-STAGE-{ordinal}", "stage_ordinal": ordinal,
                "tank_id": target.tank_id, "split_role": "development_reference",
                "image_id": image_id, "rgb_path": str(rgb),
                "rgb_sha256": _write(rgb, evidence_id.encode()), "mask_path": str(mask),
                "mask_sha256": _write(mask, image_id.encode()), "median_lab": lab,
                "source_render_receipt_sha256": "1" * 64,
                "source_validation_receipt_sha256": "2" * 64,
                "source_render_plan_sha256": "3" * 64, "known_good_profile_sha256": "4" * 64,
            })
            cursor += 1
    for target in targets:
        ordinal = target.stage_ordinal
        configurations.append({
            "recipe_id": target.recipe_id, "stage_id": f"DEV-STAGE-{ordinal}",
            "stage_ordinal": ordinal, "tank_id": target.tank_id,
            "peer_evidence_ids": [f"DONOR-INN67-DEV-RB-B01-{ordinal}",
                                  f"DONOR-INN67-DEV-RB-B02-{ordinal}"],
            "threshold_delta_e76": threshold,
            "configuration_target_evidence_id": f"CONFIG-{ordinal}",
        })
    database = tmp_path / f"development-reference-{len(list(tmp_path.glob('*.sqlite3')))}.sqlite3"
    _create_minimal_store(database, records=records, configurations=configurations, provenance={
        "development_reference_bank_sha256": "a" * 64, "delta_e_frozen_config_sha256": "b" * 64,
        "materialization_receipt_sha256": "c" * 64, "predicted_store_export_sha256": "d" * 64,
        "derived_source_manifest_sha256": "e" * 64,
        "predictor": {"producer_id": "accepted-test-predictor", "model_id": "test-model",
                      "model_sha256": "8" * 64, "config_sha256": "9" * 64,
                      "code_commit": "a" * 40},
        "predictor_sha256": "f" * 64,
        "accepted_store_tuple": {"database_path": str(tmp_path / "accepted.sqlite3"),
            "database_sha256": "6" * 64, "population_receipt_path": str(tmp_path / "population.json"),
            "population_receipt_sha256": "7" * 64}, "inn67_provenance": {"fixed": "test"},
    })
    return DevelopmentReferenceStore(database, expected_snapshot_sha256=_sha(database))


def _queries(store, targets):
    from fluidmix.memory.history.development_reference import (
        bind_runtime_target_evidence,
        development_reference_values,
    )
    result = {}
    for target in targets:
        selector = (target.recipe_id, target.stage_ordinal, target.tank_id)
        result[selector] = development_reference_values(
            store.reference_bank(bind_runtime_target_evidence({
                "evidence_id": target.evidence_id, "event_cursor": target.stage_ordinal,
                "recipe_id": target.recipe_id, "stage_id": target.stage_id,
                "stage_ordinal": target.stage_ordinal, "tank_id": target.tank_id,
                "split_role": "test", "mask_source": "predicted",
            }))
        )
    return result


def _decide(decision_id, targets, queries, *, program=None):
    from fluidmix.decision.vial import decide_wp7_final
    return decide_wp7_final(decision_id, targets, queries,
                            _program(targets) if program is None else program)


def test_memory_projects_validated_reference_values_for_decision(tmp_path):
    from fluidmix.memory.history.development_reference import (
        bind_runtime_target_evidence,
        development_reference_values,
    )

    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    target = targets[0]
    query = _store(tmp_path, targets).reference_bank(bind_runtime_target_evidence({
        "evidence_id": target.evidence_id, "event_cursor": target.stage_ordinal,
        "recipe_id": target.recipe_id, "stage_id": target.stage_id,
        "stage_ordinal": target.stage_ordinal, "tank_id": target.tank_id,
        "split_role": "test", "mask_source": "predicted",
    }))

    values = development_reference_values(query)

    assert values["query_kind"] == "development_reference_bank"
    assert values["query_sha256"] == query.query_sha256
    assert values["filters"] == query.filters
    assert len(values["members"]) == 2


def test_threshold_equality_is_keep(tmp_path):
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets, center_offsets={1: 2.0})
    result = _decide("DECISION-1", targets, _queries(store, targets))
    assert result["disposition"] == "keep"
    assert result["stages"][0]["target_delta_e76"] == pytest.approx(2.0)
    assert result["stages"][0]["stage_outlier"] is False


def test_epsilon_above_threshold_discards_and_explains_precisely(tmp_path):
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets, center_offsets={1: 2.000000001})
    result = _decide("DECISION-EPSILON", targets, _queries(store, targets))
    assert result["disposition"] == "discard"
    comparison = result["explanation"]["text"].split("camera-colour distance ", 1)[1]
    displayed_delta = float(comparison.split(" is above threshold ", 1)[0])
    displayed_threshold = float(comparison.split(" is above threshold ", 1)[1].split(";", 1)[0])
    assert displayed_delta > displayed_threshold


def test_three_programmed_stages_with_only_two_targets_is_fatal(tmp_path):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets)
    program = _program(targets) + [{"recipe_id": "R-A", "stage_id": "RUNTIME-STAGE-3",
                                    "stage_ordinal": 3, "tank_id": "t400"}]
    with pytest.raises(WP7FinalDecisionFatalError, match="programmed_stage_target_mismatch"):
        _decide("DECISION-PARTIAL", targets, _queries(store, targets), program=program)


def test_extra_target_beyond_expected_program_is_fatal(tmp_path):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2), _target(tmp_path, 3)]
    with pytest.raises(WP7FinalDecisionFatalError, match="programmed_stage_target_mismatch"):
        _decide("DECISION-EXTRA", targets, {}, program=_program(targets[:2]))


@pytest.mark.parametrize("change", ["wrong_stage", "reversed"])
def test_program_sequence_must_exactly_match_sorted_targets(tmp_path, change):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    program = _program(targets)
    if change == "wrong_stage":
        program[1] = {**program[1], "stage_id": "WRONG-STAGE"}
    else:
        program.reverse()
    with pytest.raises(WP7FinalDecisionFatalError):
        _decide("DECISION-WRONG-PROGRAM", targets, {}, program=program)


@pytest.mark.parametrize("program", [
    [{"recipe_id": "R-A", "stage_id": "S1", "stage_ordinal": 1, "tank_id": "t100"}],
    [{"recipe_id": "R-A", "stage_id": "S1", "stage_ordinal": 1, "tank_id": "t100"},
     {"recipe_id": "R-A", "stage_id": "S3", "stage_ordinal": 3, "tank_id": "t400"}],
    [{"recipe_id": "R-A", "stage_id": "S1", "stage_ordinal": 1, "tank_id": "t100"},
     {"recipe_id": "R-A", "stage_id": "S2", "stage_ordinal": 2, "tank_id": "t100"}],
])
def test_program_sequence_shape_is_fatal(program, tmp_path):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    with pytest.raises(WP7FinalDecisionFatalError):
        _decide("DECISION-PROGRAM", targets, {}, program=program)


def test_raw_self_asserted_measurement_mapping_is_rejected():
    from fluidmix.decision.vial import bind_runtime_camera_stage
    with pytest.raises(TypeError, match="trusted predicted measurement"):
        bind_runtime_camera_stage({"vial_id": "VIAL-1", "recipe_id": "R-A", "stage_id": "S1",
            "stage_ordinal": 1, "tank_id": "t100", "source": "runtime_camera"},
            {"median_lab": [50.0, 0.0, 0.0]})


def test_binding_recomputes_canonical_measurement_digest(tmp_path):
    from fluidmix.decision.vial import (
        WP7FinalDecisionFatalError,
        bind_runtime_camera_stage,
    )
    _, binding = _measurement_binding(tmp_path, 1)
    object.__setattr__(binding, "_measurement_sha256", "0" * 64)
    with pytest.raises(WP7FinalDecisionFatalError, match="trusted_measurement_sha256_invalid"):
        bind_runtime_camera_stage({
            "vial_id": "VIAL-1", "recipe_id": "R-A", "stage_id": "RUNTIME-STAGE-1",
            "stage_ordinal": 1, "tank_id": "t100", "source": "runtime_camera",
        }, binding)


def test_target_binding_carries_complete_measurement_provenance(tmp_path):
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets)
    binding = _decide("DECISION-PROVENANCE", targets, _queries(store, targets))["operator_record"]["source_bindings"]["targets"][0]
    hashes = ("rgb_sha256", "mask_sha256", "entry_sha256", "registry_sha256", "measurement_sha256")
    assert all(len(binding[field]) == 64 for field in hashes)
    assert binding["validity"] == "valid" and binding["limit_codes"] == []


def test_loose_peers_still_validate_configured_threshold(tmp_path):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets, peer_delta=4.0, configured_threshold=999.0)
    with pytest.raises(WP7FinalDecisionFatalError, match="configuration_threshold_binding_invalid"):
        _decide("DECISION-CONFIG", targets, _queries(store, targets))


def test_invalid_target_plus_unsupported_selector_preserves_all_limits(tmp_path):
    targets = [_target(tmp_path, 1, valid=False), _target(tmp_path, 2)]
    result = _decide("DECISION-INVALID-UNSUPPORTED", targets, {})
    expected = ["unsupported_development_selector", "target_measurement_invalid", "mask_empty"]
    assert result["stages"][0]["limit_codes"] == expected
    assert result["limit_codes"] == expected
    binding = result["operator_record"]["source_bindings"]["targets"][0]
    assert binding["validity"] == "invalid" and binding["limit_codes"] == ["mask_empty"]


def test_any_valid_outlier_discards_even_with_invalid_stage(tmp_path):
    targets = [_target(tmp_path, 1), _target(tmp_path, 2, valid=False), _target(tmp_path, 3)]
    store = _store(tmp_path, targets, center_offsets={1: 2.000001})
    result = _decide("DECISION-MIXED", targets, _queries(store, targets))
    assert result["disposition"] == "discard"
    assert result["denominators"] == {"programmed_stage_count": 3, "valid_stage_count": 2,
        "outlier_stage_count": 1, "bound_donor_count": 6, "required_donors_per_supported_stage": 2}


@pytest.mark.parametrize("field", ["query", "store", "config"])
def test_corrupt_query_store_or_config_hash_is_fatal(tmp_path, field):
    from fluidmix.decision.vial import WP7FinalDecisionFatalError
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets)
    queries = _queries(store, targets)
    query = queries[("R-A", 1, "t100")]
    if field == "query":
        query["query_sha256"] = "0" * 64
    else:
        query["filters"][
            "donor_store_snapshot_sha256" if field == "store" else "configuration_sha256"
        ] = "0" * 64
    with pytest.raises(WP7FinalDecisionFatalError):
        _decide("DECISION-CORRUPT", targets, queries)


def test_output_citations_hashes_and_guarded_text_are_complete(tmp_path):
    targets = [_target(tmp_path, 1), _target(tmp_path, 2)]
    store = _store(tmp_path, targets)
    result = _decide("DECISION-BOUND", targets, _queries(store, targets))
    assert result["schema_version"] == "wp7-final-decision-v1"
    assert result["model"] is None and result["operator_record"]["model"] is None
    assert len(result["explanation"]["citations"]) == 6
    text = result["explanation"]["text"].lower()
    for phrase in ("camera-colour versus synthetic-reference", "cause", "physical defect",
                   "carryover", "next-vial effect", "plant action", "unverified"):
        assert phrase in text
    encoded = json.dumps(result, sort_keys=True).lower()
    for forbidden in ("recipe_ml", "truth", "label", "model_answer", "part_a"):
        assert forbidden not in encoded
