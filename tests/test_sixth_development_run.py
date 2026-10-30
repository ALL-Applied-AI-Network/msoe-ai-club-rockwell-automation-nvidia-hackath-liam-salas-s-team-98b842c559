"""Tests for the no-oracle sixth development-run extension boundary."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PREDICTOR = {
    "producer_id": "frozen-wp5-six-channel-cnn",
    "producer_kind": "rgb_only_liquid_segmenter",
    "model_id": "resnet18-unet-current-empty-rgb",
    "model_sha256": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    "config_id": "wp5-cnn-six-channel-frozen-v1",
    "config_sha256": "7be72579f0cb0dc94f6bbe2cf894f46e63f24b9254f131106a0ed7805e96c64f",
    "code_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
    "threshold_source": "twin_validation_frozen",
    "input_fields": ["rgb"],
    "training_data_origin": "twin",
}


def test_base_store_tuple_is_the_accepted_cnn_publication() -> None:
    from fluidmix.sixth_development_run import (
        BASE_LOCATORS_PATH,
        BASE_LOCATORS_SHA256,
        BASE_RECEIPT_PATH,
        BASE_RECEIPT_SHA256,
        BASE_STORE_PATH,
        BASE_STORE_SHA256,
    )

    assert "wp5-cnn-swap-store-fc694db-01a18128-attempt01" in BASE_STORE_PATH
    assert BASE_STORE_SHA256 == (
        "f41c29ef150f32ee88a3fa9376c0b6e26111654d2e14b3e663b6c2ee29f03dcd"
    )
    assert "wp5-cnn-swap-store-fc694db-01a18128-attempt01" in BASE_RECEIPT_PATH
    assert BASE_RECEIPT_SHA256 == (
        "4752439c2d7a7a212188928b072fc625de2bd02ba33d588df78ce74bc426c636"
    )
    assert (
        "wp5-cnn-swap-materialization-d89a3f9-01a18128-attempt02"
        in BASE_LOCATORS_PATH
    )
    assert BASE_LOCATORS_SHA256 == (
        "578ab949803214b12cd4942cff7adeda3e89610eff92523505cf7c0436e5e269"
    )


def _lineage() -> dict[str, object]:
    return {
        "run_id": "RUN-V4-TRAIN-01",
        "batch_id": "BATCH_90290ed52b6d0e17",
        "source_plan_sha256": "a" * 64,
        "vials": {
            "V1": {
                "case_id": "CASE-1",
                "previous_recipe_id": "14",
                "seed": 101,
                "source_receipt_sha256": "1" * 64,
                "validation_receipt_sha256": "4" * 64,
            },
            "V2": {
                "case_id": "CASE-2",
                "previous_recipe_id": "5",
                "seed": 102,
                "source_receipt_sha256": "2" * 64,
                "validation_receipt_sha256": "5" * 64,
            },
            "V3": {
                "case_id": "CASE-3",
                "previous_recipe_id": "5",
                "seed": 103,
                "source_receipt_sha256": "3" * 64,
                "validation_receipt_sha256": "6" * 64,
            },
        },
    }


def _manifest() -> dict[str, object]:
    lineage = _lineage()
    tanks = ("t200", "t400", "t100")
    vials = []
    for ordinal in range(1, 4):
        vial_id = f"V{ordinal}"
        expected = lineage["vials"][vial_id]
        vials.append(
            {
                "vial_id": vial_id,
                "vial_ordinal": ordinal,
                "case_id": expected["case_id"],
                "previous_recipe_id": expected["previous_recipe_id"],
                "seed": expected["seed"],
                "source_receipt_path": f"/source/{vial_id}/receipt.json",
                "source_receipt_sha256": expected["source_receipt_sha256"],
                "validation_receipt_path": f"/validation/{vial_id}.json",
                "validation_receipt_sha256": expected["validation_receipt_sha256"],
                "prefill_rgb_path": f"/source/{vial_id}/pre-pour-parity/f2-prefill-source.png",
                "prefill_rgb_sha256": f"{ordinal + 6}" * 64,
                "fill_clip_path": f"/source/{vial_id}/three-pour.avi",
                "fill_clip_sha256": f"{ordinal + 3:x}" * 64,
                "stages": [
                    {
                        "image_id": f"BATCH_90290ed52b6d0e17-{vial_id}-dispense-{stage_ordinal}-{tank}",
                        "stage_id": f"dispense-{stage_ordinal}-{tank}",
                        "stage_ordinal": stage_ordinal,
                        "tank_id": tank,
                        "rgb_path": f"/source/{vial_id}/frames/frame-{stage_ordinal:04d}-{tank}-stage.png",
                        "rgb_sha256": f"{ordinal + stage_ordinal + 8:x}"[-1] * 64,
                    }
                    for stage_ordinal, tank in enumerate(tanks, 1)
                ],
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "wp5_sixth_development_run_source",
        "status": "accepted",
        "data_origin": "twin",
        "run_id": lineage["run_id"],
        "batch_id": lineage["batch_id"],
        "split_role": "train",
        "split_version": "whole-run-varied-v4",
        "source_plan_path": "/source/render-schedule-plan.json",
        "source_plan_sha256": lineage["source_plan_sha256"],
        "predictor": deepcopy(PREDICTOR),
        "recipe_context": {
            "recipe_id": "5",
            "recipe_ml": {"t100": 10.0, "t200": 10.0, "t400": 10.0},
            "dispense_order": ["t200", "t400", "t100"],
        },
        "vials": vials,
    }


def test_accepts_exact_three_vial_train_run_contract() -> None:
    from fluidmix.sixth_development_run import validate_sixth_run_manifest

    checked = validate_sixth_run_manifest(
        _manifest(),
        expected_lineage=_lineage(),
        existing_run_ids={
            "RUN-TRAIN-01",
            "RUN-TRAIN-02",
            "RUN-TRAIN-03",
            "RUN-TRAIN-04",
            "RUN-VALIDATION-01",
        },
    )

    assert checked["run_id"] == "RUN-V4-TRAIN-01"
    assert len(checked["vials"]) == 3
    assert sum(len(vial["stages"]) for vial in checked["vials"]) == 9


def test_rejects_stale_predictor_lineage_before_any_file_read() -> None:
    from fluidmix.sixth_development_run import (
        SixthDevelopmentRunError,
        validate_sixth_run_manifest,
    )

    manifest = _manifest()
    manifest["predictor"]["code_commit"] = "0" * 40

    with pytest.raises(SixthDevelopmentRunError, match="predictor_lineage_invalid"):
        validate_sixth_run_manifest(
            manifest,
            expected_lineage=_lineage(),
            existing_run_ids=set(),
        )


def test_rejects_mixed_cross_vial_lineage() -> None:
    from fluidmix.sixth_development_run import (
        SixthDevelopmentRunError,
        validate_sixth_run_manifest,
    )

    manifest = _manifest()
    manifest["vials"][0]["validation_receipt_sha256"] = manifest["vials"][1][
        "validation_receipt_sha256"
    ]

    with pytest.raises(SixthDevelopmentRunError, match="vial_lineage_invalid"):
        validate_sixth_run_manifest(
            manifest,
            expected_lineage=_lineage(),
            existing_run_ids=set(),
        )


def test_rejects_wrong_previous_recipe_before_any_file_read() -> None:
    from fluidmix.sixth_development_run import (
        SixthDevelopmentRunError,
        validate_sixth_run_manifest,
    )

    manifest = _manifest()
    manifest["vials"][1]["previous_recipe_id"] = "14"

    with pytest.raises(SixthDevelopmentRunError, match="vial_lineage_invalid"):
        validate_sixth_run_manifest(
            manifest,
            expected_lineage=_lineage(),
            existing_run_ids=set(),
        )


def test_rejects_oracle_field_and_existing_run_identity() -> None:
    from fluidmix.sixth_development_run import (
        SixthDevelopmentRunError,
        validate_sixth_run_manifest,
    )

    manifest = _manifest()
    manifest["vials"][0]["oracle_mask_path"] = "/forbidden/oracle.png"
    with pytest.raises(SixthDevelopmentRunError, match="vial_fields_invalid"):
        validate_sixth_run_manifest(
            manifest,
            expected_lineage=_lineage(),
            existing_run_ids=set(),
        )

    manifest = _manifest()
    with pytest.raises(SixthDevelopmentRunError, match="run_id_not_disjoint"):
        validate_sixth_run_manifest(
            manifest,
            expected_lineage=_lineage(),
            existing_run_ids={"RUN-V4-TRAIN-01"},
        )


def test_default_manifest_is_exact_accepted_v4_train_run() -> None:
    from fluidmix.sixth_development_run import (
        DEFAULT_LINEAGE,
        default_source_manifest,
        validate_sixth_run_manifest,
    )

    manifest = default_source_manifest()
    checked = validate_sixth_run_manifest(
        manifest,
        expected_lineage=DEFAULT_LINEAGE,
        existing_run_ids={
            "RUN-TRAIN-01",
            "RUN-TRAIN-02",
            "RUN-TRAIN-03",
            "RUN-TRAIN-04",
            "RUN-VALIDATION-01",
        },
    )

    assert [row["case_id"] for row in checked["vials"]] == [
        "CASE_97a8c6cb8161881c",
        "CASE_59038b977a73fdfb",
        "CASE_0b872555c4f08091",
    ]
    assert checked["source_plan_sha256"] == (
        "867fa9516d13c79cbe9fd2960a25477926d17dda4834ec81adc5df55e046d37f"
    )


def test_default_manifest_preserves_independent_previous_recipe_per_vial() -> None:
    from fluidmix.sixth_development_run import (
        DEFAULT_LINEAGE,
        default_source_manifest,
        validate_sixth_run_manifest,
    )

    expected_by_case = {
        "CASE_97a8c6cb8161881c": "14",
        "CASE_59038b977a73fdfb": "5",
        "CASE_0b872555c4f08091": "5",
    }
    checked = validate_sixth_run_manifest(
        default_source_manifest(),
        expected_lineage=DEFAULT_LINEAGE,
        existing_run_ids=set(),
    )

    assert "previous_recipe_id" not in checked["recipe_context"]
    assert {
        vial["case_id"]: vial["previous_recipe_id"] for vial in checked["vials"]
    } == expected_by_case


def test_default_sources_are_hash_bound_without_opening_plan_or_masks() -> None:
    from fluidmix.sixth_development_run import (
        DEFAULT_LINEAGE,
        default_source_manifest,
        validate_sixth_run_manifest,
        verify_source_files,
    )

    checked = validate_sixth_run_manifest(
        default_source_manifest(),
        expected_lineage=DEFAULT_LINEAGE,
        existing_run_ids=set(),
    )
    reads = verify_source_files(checked)

    assert len(reads) == 21
    assert "source_plan" not in {row["role"] for row in reads}
    assert not any(
        token in row["path"].lower()
        for row in reads
        for token in ("/liquid/", "/semantics/", "/oracle", "/labels/", "video5")
    )


def test_frozen_predictor_materializes_all_nine_v4_stage_masks_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fluidmix.sixth_development_run as module
    from fluidmix.sixth_development_run import (
        DEFAULT_LINEAGE,
        SixthDevelopmentRunError,
        default_source_manifest,
        materialize_extension_predictions,
        validate_sixth_run_manifest,
    )

    checked = validate_sixth_run_manifest(
        default_source_manifest(),
        expected_lineage=DEFAULT_LINEAGE,
        existing_run_ids=set(),
    )
    monkeypatch.setattr(
        module,
        "__file__",
        str(ROOT / "tests" / ".." / "src" / "fluidmix" / "sixth_development_run.py"),
    )
    monkeypatch.setattr(
        module,
        "_load_frozen_cnn",
        lambda **_: (object(), module.PREDICTOR, {}, {}),
        raising=False,
    )
    from fluidmix.empty_reference_rgb_predictor import fillcam_masks

    roi, clamp = fillcam_masks()
    predicted_mask = roi & ~clamp
    monkeypatch.setattr(
        module,
        "_predict_cnn_pair",
        lambda _model, _rgb, _reference: type(
            "Prediction",
            (),
            {
                "mask": predicted_mask,
                "status": "visible_measured",
                "surface_row": int(np.flatnonzero(np.any(predicted_mask, axis=1))[0]),
                "visible_height_px": int(np.count_nonzero(np.any(predicted_mask, axis=1))),
                "changed_pixel_count": int(np.count_nonzero(predicted_mask)),
            },
        )(),
        raising=False,
    )
    output = tmp_path / "predictions"
    result = materialize_extension_predictions(checked, output)

    assert result["predictor"]["producer_id"] == "frozen-wp5-six-channel-cnn"
    assert result["predictor"]["model_sha256"] == (
        "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba"
    )
    assert result["denominators"] == {"runs": 1, "vials": 3, "stages": 9}
    assert len(result["records"]) == 9
    assert {row["prediction_status"] for row in result["records"]} == {
        "visible_measured"
    }
    with pytest.raises(SixthDevelopmentRunError, match="prediction_output_not_fresh"):
        materialize_extension_predictions(checked, output)


def test_strict_json_loader_accepts_generated_absolute_path(tmp_path: Path) -> None:
    from fluidmix.sixth_development_run import _load_strict_json

    path = tmp_path / "generated.json"
    path.write_text(json.dumps({"images": {}}))

    assert _load_strict_json(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), "generated"
    ) == {"images": {}}
