"""Hash-bound locator publication for the corrected Varied V3 cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_f2_varied_stage_measurement import _manifest_path, _sha, _write_json


def _cohort(tmp_path: Path) -> tuple[list[Path], Path]:
    cohort = tmp_path / "corrected-v2"
    publications = cohort / "publications"
    publications.mkdir(parents=True)
    manifests: list[Path] = []
    manifest_hashes: dict[str, str] = {}
    recipes = ("5", "6", "7", "8", "13", "14")
    splits = ("train", "train", "train", "train", "validation", "evaluation")
    active_tanks = (
        ("t200", "t400", "t100"),
        ("t200", "t100"),
        ("t400", "t100"),
        ("t200", "t400", "t100"),
        ("t200", "t100"),
        ("t400", "t100"),
    )
    for index, (recipe, split, tanks) in enumerate(
        zip(recipes, splits, active_tanks, strict=True), start=1
    ):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        original = _manifest_path(source, active_tanks=tanks)
        batch_id = f"BATCH_{index:016x}"
        target = publications / batch_id
        shutil.move(original.parent, target)
        manifest_path = target / "f2-stage-capture-manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["batch_id"] = batch_id
        payload["run_id"] = f"RUN-{split.upper()}-{index:02d}"
        payload["recipe_id"] = recipe
        payload["split_role"] = split
        payload["runtime_receipt"]["path"] = str(target / "runtime-receipt.json")
        for row in payload["rows"]:
            for key in ("rgb", "liquid_mask"):
                old_path = Path(row[key]["path"])
                row[key]["path"] = str(target / old_path.name)
        _write_json(manifest_path, payload)
        manifests.append(manifest_path)
        manifest_hashes[batch_id] = _sha(manifest_path)
    receipt = cohort / "downstream-validation-receipt.json"
    _write_json(
        receipt,
        {
            "schema_version": 2,
            "artifact_type": "f2_varied_v3_downstream_validation_receipt",
            "status": "pass",
            "batch_count": 6,
            "vial_count": 18,
            "stage_pair_count": 42,
            "source_artifact_validation_count": 18,
            "split_batch_counts": {"train": 4, "validation": 1, "evaluation": 1},
            "height_warning_case_count": 15,
            "height_warning_count": 16,
            "warning_free_case_count": 3,
            "capture_manifest_sha256_by_batch": manifest_hashes,
            "claim_boundary": "synthetic artifact routing only",
        },
    )
    return manifests, receipt


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(key for child in value.values() for key in _all_keys(child))]
    if isinstance(value, list):
        return [key for child in value for key in _all_keys(child)]
    return []


def test_builds_exact_label_free_six_run_locator(tmp_path):
    from fluidmix.f2_varied_image_locator import build_varied_image_locators

    manifests, validation = _cohort(tmp_path)
    result = build_varied_image_locators(manifests, validation)

    assert result["artifact_type"] == "f2_varied_image_locators"
    assert result["status"] == "published"
    assert result["real_data_used"] is False
    assert result["source_validation_receipt"] == {
        "path": str(validation.resolve()),
        "sha256": _sha(validation),
    }
    assert result["denominators"] == {
        "source_manifests": 6,
        "runs": 6,
        "vials": 18,
        "active_stage_events": 14,
        "images": 42,
        "two_stage_runs": 4,
        "three_stage_runs": 2,
        "split_images": {"train": 30, "validation": 6, "evaluation": 6},
    }
    assert len(result["images"]) == 42
    assert len(result["source_manifests"]) == 6
    first = next(iter(result["images"].values()))
    assert Path(first["rgb_path"]).is_absolute()
    assert first["rgb_sha256"] == hashlib.sha256(
        Path(first["rgb_path"]).read_bytes()
    ).hexdigest()
    assert first["mask_sha256"] == hashlib.sha256(
        Path(first["mask_path"]).read_bytes()
    ).hexdigest()
    assert set(first) == {
        "batch_id",
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "vial_id",
        "stage_id",
        "stage_ordinal",
        "tank_id",
        "source_manifest_sha256",
        "rgb_path",
        "rgb_sha256",
        "mask_path",
        "mask_sha256",
    }
    assert not any(
        fragment in key.lower()
        for key in _all_keys(result)
        for fragment in (
            "label",
            "residual",
            "palette",
            "expected_disposition",
            "render_parameter",
        )
    )


def test_rejects_validation_manifest_hash_drift(tmp_path):
    from fluidmix.f2_varied_image_locator import (
        VariedImageLocatorError,
        build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    receipt = json.loads(validation.read_text(encoding="utf-8"))
    receipt["capture_manifest_sha256_by_batch"]["BATCH_0000000000000001"] = "0" * 64
    _write_json(validation, receipt)

    with pytest.raises(VariedImageLocatorError, match="validation_manifest_hash_mismatch"):
        build_varied_image_locators(manifests, validation)


def test_rejects_duplicate_or_wrong_denominator(tmp_path):
    from fluidmix.f2_varied_image_locator import (
        VariedImageLocatorError,
        build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    with pytest.raises(VariedImageLocatorError, match="manifest_denominator_invalid"):
        build_varied_image_locators(manifests[:-1], validation)
    with pytest.raises(VariedImageLocatorError, match="duplicate_manifest_path"):
        build_varied_image_locators([*manifests[:-1], manifests[0]], validation)


def test_artifact_drift_fails_in_manifest_binder(tmp_path):
    from fluidmix.f2_varied_image_locator import (
        VariedImageLocatorError,
        build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    Path(payload["rows"][0]["rgb"]["path"]).write_bytes(b"drift")

    with pytest.raises(VariedImageLocatorError, match="rgb_hash_mismatch"):
        build_varied_image_locators(manifests, validation)


def test_write_is_no_clobber(tmp_path):
    from fluidmix.f2_varied_image_locator import write_varied_image_locators

    manifests, validation = _cohort(tmp_path)
    output = tmp_path / "locator.json"
    result = write_varied_image_locators(manifests, validation, output)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        write_varied_image_locators(manifests, validation, output)
