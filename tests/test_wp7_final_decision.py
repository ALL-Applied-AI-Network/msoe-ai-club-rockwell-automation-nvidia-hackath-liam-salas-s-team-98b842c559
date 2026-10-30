"""Pure final WP7 decision and guarded explanation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _store(tmp_path: Path):
    from fluidmix.wp7_development_reference import (
        DevelopmentReferenceStore,
        _create_minimal_store,
    )

    records: list[dict[str, object]] = []
    configurations: list[dict[str, object]] = []
    cursor = 0
    for batch, run, offset in (
        ("INN67-DEV-RB-B01", "RUN-B01", -1.0),
        ("INN67-DEV-RB-B02", "RUN-B02", 1.0),
    ):
        for ordinal, tank in ((1, "t100"), (2, "t200"), (3, "t400")):
            evidence_id = f"DONOR-{batch}-{ordinal}"
            image_id = f"DONOR-IMAGE-{batch}-{ordinal}"
            rgb = tmp_path / "rgb" / f"{image_id}.png"
            mask = tmp_path / "mask" / f"{image_id}.png"
            records.append(
                {
                    "evidence_id": evidence_id,
                    "event_cursor": cursor,
                    "batch_id": batch,
                    "run_id": run,
                    "case_id": f"CASE-{batch}",
                    "recipe_id": "R-A",
                    "stage_id": f"DEV-STAGE-{ordinal}",
                    "stage_ordinal": ordinal,
                    "tank_id": tank,
                    "split_role": "development_reference",
                    "image_id": image_id,
                    "rgb_path": str(rgb),
                    "rgb_sha256": _write(rgb, evidence_id.encode()),
                    "mask_path": str(mask),
                    "mask_sha256": _write(mask, image_id.encode()),
                    "median_lab": [50.0 + offset, float(ordinal), 0.0],
                    "source_render_receipt_sha256": "1" * 64,
                    "source_validation_receipt_sha256": "2" * 64,
                    "source_render_plan_sha256": "3" * 64,
                    "known_good_profile_sha256": "4" * 64,
                }
            )
            cursor += 1
    for ordinal, tank in ((1, "t100"), (2, "t200"), (3, "t400")):
        configurations.append(
            {
                "recipe_id": "R-A",
                "stage_id": f"DEV-STAGE-{ordinal}",
                "stage_ordinal": ordinal,
                "tank_id": tank,
                "peer_evidence_ids": [
                    f"DONOR-INN67-DEV-RB-B01-{ordinal}",
                    f"DONOR-INN67-DEV-RB-B02-{ordinal}",
                ],
                "threshold_delta_e76": 2.0,
                "configuration_target_evidence_id": f"CONFIG-{ordinal}",
            }
        )
    database = tmp_path / "development-reference.sqlite3"
    _create_minimal_store(
        database,
        records=records,
        configurations=configurations,
        provenance={
            "development_reference_bank_sha256": "a" * 64,
            "delta_e_frozen_config_sha256": "b" * 64,
            "materialization_receipt_sha256": "c" * 64,
            "predicted_store_export_sha256": "d" * 64,
            "derived_source_manifest_sha256": "e" * 64,
            "predictor": {
                "producer_id": "accepted-test-predictor",
                "model_id": "test-model",
                "model_sha256": "8" * 64,
                "config_sha256": "9" * 64,
                "code_commit": "a" * 40,
            },
            "predictor_sha256": "f" * 64,
            "accepted_store_tuple": {
                "database_path": str(tmp_path / "accepted.sqlite3"),
                "database_sha256": "6" * 64,
                "population_receipt_path": str(tmp_path / "population.json"),
                "population_receipt_sha256": "7" * 64,
            },
            "inn67_provenance": {"fixed": "test"},
        },
    )
    snapshot = hashlib.sha256(database.read_bytes()).hexdigest()
    return DevelopmentReferenceStore(database, expected_snapshot_sha256=snapshot)


def _target(ordinal: int, lab: tuple[float, float, float] | None, **changes: object):
    from fluidmix.wp7_final_decision import bind_runtime_camera_stage

    value: dict[str, object] = {
        "evidence_id": f"TARGET-EVIDENCE-{ordinal}",
        "measurement_id": f"TARGET-MEASUREMENT-{ordinal}",
        "image_id": f"TARGET-IMAGE-{ordinal}",
        "vial_id": "VIAL-1",
        "recipe_id": "R-A",
        "stage_id": f"RUNTIME-STAGE-{ordinal}",
        "stage_ordinal": ordinal,
        "tank_id": {1: "t100", 2: "t200", 3: "t400"}[ordinal],
        "median_lab": lab,
        "validity": "valid" if lab is not None else "invalid",
        "limit_codes": [] if lab is not None else ["camera_measurement_missing"],
        "source": "runtime_camera",
    }
    value.update(changes)
    return bind_runtime_camera_stage(value)


def _queries(store, targets):
    from fluidmix.wp7_development_reference import bind_runtime_target_evidence

    result = {}
    for target in targets:
        selector = (target.recipe_id, target.stage_ordinal, target.tank_id)
        result[selector] = store.reference_bank(
            bind_runtime_target_evidence(
                {
                    "evidence_id": target.evidence_id,
                    "event_cursor": target.stage_ordinal,
                    "recipe_id": target.recipe_id,
                    "stage_id": target.stage_id,
                    "stage_ordinal": target.stage_ordinal,
                    "tank_id": target.tank_id,
                    "split_role": "test",
                    "mask_source": "predicted",
                }
            )
        )
    return result


def test_threshold_equality_is_keep(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [_target(1, (52.0, 1.0, 0.0))]
    result = decide_wp7_final("DECISION-1", targets, _queries(store, targets))

    assert result["disposition"] == "keep"
    assert result["stages"][0]["peer_delta_e76"] == pytest.approx(2.0)
    assert result["stages"][0]["threshold_delta_e76"] == pytest.approx(2.0)
    assert result["stages"][0]["target_delta_e76"] == pytest.approx(2.0)
    assert result["stages"][0]["stage_outlier"] is False


def test_epsilon_above_threshold_is_discard(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [_target(1, (52.0 + 1e-9, 1.0, 0.0))]
    result = decide_wp7_final("DECISION-EPSILON", targets, _queries(store, targets))

    assert result["disposition"] == "discard"
    assert result["stages"][0]["stage_outlier"] is True


def test_any_valid_outlier_discards_even_with_an_invalid_stage(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [
        _target(1, (52.000001, 1.0, 0.0)),
        _target(2, None),
        _target(3, (50.0, 3.0, 0.0)),
    ]
    result = decide_wp7_final("DECISION-MIXED", targets, _queries(store, targets))

    assert result["disposition"] == "discard"
    assert [stage["validity"] for stage in result["stages"]] == [
        "valid", "insufficient_evidence", "valid"
    ]
    assert result["denominators"] == {
        "programmed_stage_count": 3,
        "valid_stage_count": 2,
        "outlier_stage_count": 1,
        "bound_donor_count": 6,
        "required_donors_per_supported_stage": 2,
    }


def test_supported_stage_binds_exactly_two_independent_batches(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [_target(1, (50.0, 1.0, 0.0))]
    result = decide_wp7_final("DECISION-DONORS", targets, _queries(store, targets))
    stage = result["stages"][0]

    assert {donor["batch_id"] for donor in stage["donors"]} == {
        "INN67-DEV-RB-B01", "INN67-DEV-RB-B02"
    }
    assert len({donor["run_id"] for donor in stage["donors"]}) == 2
    assert len(stage["donors"]) == 2


def test_unsupported_selector_abstains(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    _store(tmp_path)
    targets = [_target(1, (50.0, 1.0, 0.0))]
    result = decide_wp7_final("DECISION-UNSUPPORTED", targets, {})

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == ["unsupported_development_selector"]


def test_invalid_target_abstains_and_preserves_measurement_limit(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [_target(1, None)]
    result = decide_wp7_final("DECISION-INVALID", targets, _queries(store, targets))

    assert result["disposition"] == "insufficient_evidence"
    assert result["stages"][0]["median_lab"] is None
    assert result["limit_codes"] == [
        "target_measurement_invalid", "camera_measurement_missing"
    ]


@pytest.mark.parametrize("field", ["query", "store", "config"])
def test_corrupt_query_store_or_config_hash_is_fatal(
    tmp_path: Path, field: str
) -> None:
    from fluidmix.wp7_final_decision import (
        WP7FinalDecisionFatalError,
        decide_wp7_final,
    )

    store = _store(tmp_path)
    targets = [_target(1, (50.0, 1.0, 0.0))]
    queries = _queries(store, targets)
    query = next(iter(queries.values()))
    if field == "query":
        object.__setattr__(query, "query_sha256", "0" * 64)
    else:
        filters = query.filters
        filters[
            "donor_store_snapshot_sha256" if field == "store" else "configuration_sha256"
        ] = "0" * 64
        object.__setattr__(
            query, "_filters_json", json.dumps(filters, sort_keys=True, separators=(",", ":"))
        )

    with pytest.raises(WP7FinalDecisionFatalError):
        decide_wp7_final("DECISION-CORRUPT", targets, queries)


def test_output_citations_hashes_and_guarded_text_are_complete(tmp_path: Path) -> None:
    from fluidmix.wp7_final_decision import decide_wp7_final

    store = _store(tmp_path)
    targets = [_target(1, (50.0, 1.0, 0.0)), _target(2, (50.0, 2.0, 0.0))]
    result = decide_wp7_final("DECISION-BOUND", targets, _queries(store, targets))

    assert result["schema_version"] == "wp7-final-decision-v1"
    assert result["model"] is None
    assert result["operator_record"]["model"] is None
    for stage in result["stages"]:
        assert stage["query_sha256"]
        assert stage["donor_store_snapshot_sha256"]
        assert stage["configuration_sha256"]
        assert stage["median_lab"] is not None
        assert stage["target_delta_e76"] is not None
        assert stage["threshold_delta_e76"] is not None
        assert len(stage["donors"]) == 2
    citations = result["explanation"]["citations"]
    assert len(citations) == 6
    assert {row["role"] for row in citations} == {
        "target", "synthetic_development_donor"
    }
    text = result["explanation"]["text"].lower()
    assert "camera-colour versus synthetic-reference" in text
    for phrase in (
        "cause", "physical defect", "carryover", "next-vial effect", "plant action",
        "unverified",
    ):
        assert phrase in text
    encoded = json.dumps(result, sort_keys=True).lower()
    for forbidden in ("recipe_ml", "truth", "label", "model_answer", "part_a"):
        assert forbidden not in encoded
    assert result["operator_record"]["source_bindings"]["targets"][0][
        "measurement_id"
    ] == "TARGET-MEASUREMENT-1"
