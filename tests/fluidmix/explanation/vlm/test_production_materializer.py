"""Focused source-only production WP7 VLM materializer tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from fluidmix.explanation.vlm.decision import build_vial1_bundle, load_runtime_context
from fluidmix.explanation.vlm.production_materializer import (
    ProductionMaterializerError,
    materialize_production_vlm_cohort,
)
from tests.fluidmix.explanation.vlm.test_decision import _runtime_files


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _sources(tmp_path: Path) -> dict[str, object]:
    cohort_id = "PROD-COHORT-001"
    camera = {
        "resolution": [1920, 1080],
        "frame_rate_hz": 30.0,
        "exposure": "manual-1000us",
        "white_balance": "manual-5000k",
        "viewpoint_id": "fill-camera-1",
        "vial_roi": [10, 20, 100, 200],
    }
    recipe = {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "test",
        "split_version": "production-source-v1",
        "runs": [{
            "run_id": cohort_id,
            "vials": [
                {
                    "case_id": f"PROD-VIAL-{position}",
                    "vial_ordinal": position,
                    "block_recipe_id": "BLOCK-RECIPE-7",
                    "block_position": position,
                    "recipe_id": "RECIPE-7",
                    "previous_recipe_id": "RECIPE-6" if position == 1 else "RECIPE-7",
                    "programmed_tank_ml": {"t100": 10.0, "t200": 20.0, "t400": 30.0},
                    "dispense_order": ["t100", "t200", "t400"],
                }
                for position in (1, 2, 3)
            ],
        }],
    }
    recipe_sha = _digest(recipe)
    trackers = []
    bindings: dict[str, dict[str, object]] = {}
    records = []
    for position in (1, 2, 3):
        vial_id = f"PROD-VIAL-{position}"
        stages = []
        cumulative = 0.0
        for stage_ordinal, (tank_id, amount) in enumerate(
            (("t100", 10.0), ("t200", 20.0), ("t400", 30.0)), 1,
        ):
            cumulative += amount
            images = {}
            for phase_index, phase in enumerate(("early", "mid", "end")):
                image_id = f"IMG-{vial_id}-S{stage_ordinal}-{phase}"
                image_path = tmp_path / "crops" / f"{image_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(bytes((position, stage_ordinal, phase_index)))
                images[phase] = image_id
                bindings[image_id] = {
                    "path": str(image_path),
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "source_full_frame_sha256": hashlib.sha256(
                        f"full-{image_id}".encode()
                    ).hexdigest(),
                    "crop_pixel_sha256": hashlib.sha256(
                        f"pixels-{image_id}".encode()
                    ).hexdigest(),
                    "crop_box_xyxy": [10, 20, 100, 200],
                    "source_native_dimensions": [1920, 1080],
                    "crop_native_dimensions": [90, 180],
                    "timestamp_seconds": float(position * 100 + stage_ordinal * 10 + phase_index),
                }
            stages.append({
                "fill_step": stage_ordinal,
                "tank_id": tank_id,
                "tank_number": int(tank_id[1:]),
                "stream_step_index": stage_ordinal,
                "expected_dispense_ml": amount,
                "expected_cumulative_ml": cumulative,
                "twin_geometry_equivalent_volume_ml": cumulative,
                "measured_volume_ml": cumulative,
                "agreement": True,
                "tank_cluster_consistent": True,
                "stream_lab": [45.0, float(stage_ordinal), 2.0],
                "stream_spread_lab": [0.1, 0.2, 0.3],
                "images": images,
            })
            records.append({
                "case_id": vial_id,
                "vial_position": position,
                "stage_ordinal": stage_ordinal,
                "stage_id": f"STAGE-{cohort_id}-{stage_ordinal}",
                "evidence_id": f"EVID-{vial_id}-{stage_ordinal}",
                "image_ids": [images[phase] for phase in ("early", "mid", "end")],
                "median_lab": [50.0, float(stage_ordinal), 3.0],
                "mad_lab": [0.1, 0.1, 0.1],
                "iqr_lab": [0.2, 0.2, 0.2],
            })
        tracker = {
            "schema_version": "wp7-camera-fill-step-receipt-v1",
            "artifact_type": "wp7_camera_fill_step_receipt",
            "cohort_id": cohort_id,
            "vial_id": vial_id,
            "claim_boundary": "camera_twin_geometry_equivalent_not_physical_ground_truth",
            "stages": stages,
            "calibration_sha256": "a" * 64,
            "recipe_artifact_sha256": recipe_sha,
            "source_artifact_sha256": f"{position:064x}",
        }
        tracker["receipt_sha256"] = _digest(tracker)
        trackers.append(tracker)
    measurements = {
        "schema_version": "wp7-production-camera-measurements-v1",
        "artifact_type": "wp7_production_camera_measurements",
        "cohort_id": cohort_id,
        "tracker_receipt_sha256": [row["receipt_sha256"] for row in trackers],
        "records": records,
    }
    measurements["receipt_sha256"] = _digest(measurements)
    return {
        "recipe_metadata": recipe,
        "recipe_artifact_sha256": recipe_sha,
        "tracker_receipts": trackers,
        "image_bindings": bindings,
        "measurement_receipt": measurements,
        "camera_metadata": camera,
    }


def _materialize(tmp_path: Path, **changes: object):
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")
    arguments = {
        **_sources(tmp_path),
        "output_root": tmp_path / "published-production-cohort",
        "freeze_path": freeze_path,
        "expected_freeze_sha256": freeze_sha,
    }
    arguments.update(changes)
    return materialize_production_vlm_cohort(**arguments), arguments


def _rebind_recipe_and_trackers(sources: dict[str, object]) -> None:
    recipe_sha = _digest(sources["recipe_metadata"])
    sources["recipe_artifact_sha256"] = recipe_sha
    for tracker in sources["tracker_receipts"]:
        tracker["recipe_artifact_sha256"] = recipe_sha
        tracker["receipt_sha256"] = _digest({
            key: value for key, value in tracker.items() if key != "receipt_sha256"
        })
    measurements = sources["measurement_receipt"]
    measurements["tracker_receipt_sha256"] = [
        tracker["receipt_sha256"] for tracker in sources["tracker_receipts"]
    ]
    measurements["receipt_sha256"] = _digest({
        key: value for key, value in measurements.items() if key != "receipt_sha256"
    })


def _two_stage_sources(tmp_path: Path) -> dict[str, object]:
    sources = _sources(tmp_path)
    recipe = sources["recipe_metadata"]
    for vial in recipe["runs"][0]["vials"]:
        vial["programmed_tank_ml"]["t400"] = 0.0
        vial["dispense_order"] = ["t100", "t200"]
    removed_ids = set()
    for tracker in sources["tracker_receipts"]:
        removed_ids.update(tracker["stages"][2]["images"].values())
        tracker["stages"] = tracker["stages"][:2]
    for image_id in removed_ids:
        sources["image_bindings"].pop(image_id)
    measurements = sources["measurement_receipt"]
    measurements["records"] = [
        record for record in measurements["records"]
        if record["stage_ordinal"] <= 2
    ]
    _rebind_recipe_and_trackers(sources)
    return sources


def test_materializes_exact_production_runtime_inputs_and_reloads_reviewed_freeze(
    tmp_path: Path,
) -> None:
    receipt, arguments = _materialize(tmp_path)
    output = arguments["output_root"]

    assert receipt["schema_version"] == "wp7-vlm-production-materialization-receipt-v1"
    assert receipt["status"] == "complete"
    assert receipt["image_count"] == 27
    assert receipt["runtime_validation"] == {
        "production": True,
        "ordered_image_count": 27,
        "status": "pass",
    }
    cohort = json.loads((output / "cohort.json").read_text())
    locators = json.loads((output / "image-locators.json").read_text())
    trackers = [
        json.loads((output / "tracker-receipts" / f"vial{position}.json").read_text())
        for position in (1, 2, 3)
    ]
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=trackers, production=True,
    )
    assert cohort["partition"] == "production"
    assert len(built.prompt_payload["images"]) == 27
    assert all("path" not in row for row in built.prompt_payload["images"])
    assert str(tmp_path) not in json.dumps(built.prompt_payload)
    context = load_runtime_context(
        output / "cohort.json", output / "image-locators.json",
        [output / "tracker-receipts" / f"vial{position}.json" for position in (1, 2, 3)],
        freeze_path=arguments["freeze_path"],
        expected_freeze_sha256=arguments["expected_freeze_sha256"], production=True,
    )
    assert context.production is True
    assert context.binding["bundle_sha256"] == receipt["bundle_sha256"]
    assert (output / "manifest.json").is_file()
    assert (output / "materialization-receipt.json").is_file()


@pytest.mark.parametrize("stage_count,expected_images", [(2, 18), (3, 27)])
def test_missing_phase_image_fails_closed_with_derived_denominator(
    tmp_path: Path, stage_count: int, expected_images: int,
) -> None:
    sources = (
        _two_stage_sources(tmp_path) if stage_count == 2 else _sources(tmp_path)
    )
    sources["image_bindings"].pop(next(iter(sources["image_bindings"])))
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")
    output = tmp_path / "output"
    with pytest.raises(
        ProductionMaterializerError, match=f"exactly {expected_images}"
    ):
        materialize_production_vlm_cohort(
            **sources, output_root=output, freeze_path=freeze_path,
            expected_freeze_sha256=freeze_sha,
        )
    assert not output.exists()


def test_two_stage_recipe_derives_eighteen_image_denominator(tmp_path: Path) -> None:
    sources = _two_stage_sources(tmp_path)
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")
    output = tmp_path / "two-stage-output"

    receipt = materialize_production_vlm_cohort(
        **sources, output_root=output, freeze_path=freeze_path,
        expected_freeze_sha256=freeze_sha,
    )

    assert receipt["image_count"] == 18
    assert receipt["runtime_validation"]["ordered_image_count"] == 18
    cohort = json.loads((output / "cohort.json").read_text())
    assert [stage["tank_id"] for stage in cohort["stages"]] == ["t100", "t200"]


def test_later_block_accepts_contiguous_absolute_vial_ordinals(tmp_path: Path) -> None:
    sources = _two_stage_sources(tmp_path)
    for ordinal, vial in enumerate(
        sources["recipe_metadata"]["runs"][0]["vials"], start=4,
    ):
        vial["vial_ordinal"] = ordinal
    _rebind_recipe_and_trackers(sources)
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")

    receipt = materialize_production_vlm_cohort(
        **sources, output_root=tmp_path / "later-block", freeze_path=freeze_path,
        expected_freeze_sha256=freeze_sha,
    )

    assert receipt["status"] == "complete"
    assert receipt["image_count"] == 18


def test_tracker_stage_denominator_must_match_exact_dispense_order(tmp_path: Path) -> None:
    sources = _two_stage_sources(tmp_path)
    tracker = sources["tracker_receipts"][0]
    tracker["stages"][1]["tank_id"] = "t400"
    tracker["stages"][1]["tank_number"] = 400
    tracker["receipt_sha256"] = _digest({
        key: value for key, value in tracker.items() if key != "receipt_sha256"
    })
    measurements = sources["measurement_receipt"]
    measurements["tracker_receipt_sha256"][0] = tracker["receipt_sha256"]
    measurements["receipt_sha256"] = _digest({
        key: value for key, value in measurements.items() if key != "receipt_sha256"
    })
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")

    with pytest.raises(ProductionMaterializerError, match="tracker stage disagrees"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / "order-drift", freeze_path=freeze_path,
            expected_freeze_sha256=freeze_sha,
        )


def test_tracker_recipe_and_measurement_hash_disagreement_fail_closed(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    sources["recipe_artifact_sha256"] = "f" * 64
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")
    with pytest.raises(ProductionMaterializerError, match="recipe artifact SHA"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / "recipe-drift", freeze_path=freeze_path,
            expected_freeze_sha256=freeze_sha,
        )

    sources = _sources(tmp_path / "recipe-body-drift")
    sources["recipe_metadata"]["runs"][0]["vials"][0]["recipe_id"] = "RECIPE-8"
    with pytest.raises(ProductionMaterializerError, match="recipe artifact SHA"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / "recipe-body-drift-output",
            freeze_path=freeze_path, expected_freeze_sha256=freeze_sha,
        )

    sources = _sources(tmp_path / "measurement-drift")
    sources["measurement_receipt"]["records"][0]["median_lab"] = [1.0, 2.0, 3.0]
    with pytest.raises(ProductionMaterializerError, match="measurement receipt hash"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / "measurement-drift-output",
            freeze_path=freeze_path, expected_freeze_sha256=freeze_sha,
        )


@pytest.mark.parametrize("field", ["exposure", "white_balance", "viewpoint_id"])
def test_empty_production_camera_metadata_fails_closed(
    tmp_path: Path, field: str,
) -> None:
    sources = _sources(tmp_path)
    sources["camera_metadata"][field] = ""
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")

    with pytest.raises(ProductionMaterializerError, match="camera metadata is incomplete"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / f"empty-{field}",
            freeze_path=freeze_path, expected_freeze_sha256=freeze_sha,
        )


def test_phase_bindings_must_be_chronological_across_stages(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    second_stage = sources["tracker_receipts"][0]["stages"][1]
    for timestamp, phase in enumerate(("early", "mid", "end"), start=1):
        image_id = second_stage["images"][phase]
        sources["image_bindings"][image_id]["timestamp_seconds"] = float(timestamp)
    _, _, _, freeze_path, freeze_sha = _runtime_files(tmp_path / "freeze-source")

    with pytest.raises(ProductionMaterializerError, match="stage timestamps"):
        materialize_production_vlm_cohort(
            **sources, output_root=tmp_path / "stage-time-drift",
            freeze_path=freeze_path, expected_freeze_sha256=freeze_sha,
        )


def test_materializer_is_no_clobber(tmp_path: Path) -> None:
    _, arguments = _materialize(tmp_path)
    sentinel = arguments["output_root"] / "cohort.json"
    original = sentinel.read_bytes()
    with pytest.raises(ProductionMaterializerError, match="already exists"):
        materialize_production_vlm_cohort(**{
            key: value for key, value in arguments.items()
            if key not in {"output_root"}
        }, output_root=arguments["output_root"])
    assert sentinel.read_bytes() == original
