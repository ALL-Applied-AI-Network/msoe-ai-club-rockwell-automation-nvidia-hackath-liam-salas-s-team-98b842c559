"""Hash-bound locator publication for the corrected Varied V3 cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tests.fluidmix.measurement.test_varied_stage_measurement import _manifest_path, _sha, _write_json


def _bind_capture_manifest(path: Path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
    )

    return bind_stage_capture_manifest(path, strict_published_artifacts=True)


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
    from twin.cohorts.varied_image_locator import _build_varied_image_locators

    manifests, validation = _cohort(tmp_path)
    result = _build_varied_image_locators(
        manifests,
        validation,
        bind_capture_manifest=_bind_capture_manifest,
    )

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
    from twin.cohorts.varied_image_locator import (
        VariedImageLocatorError,
        _build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    receipt = json.loads(validation.read_text(encoding="utf-8"))
    receipt["capture_manifest_sha256_by_batch"]["BATCH_0000000000000001"] = "0" * 64
    _write_json(validation, receipt)

    with pytest.raises(VariedImageLocatorError, match="validation_manifest_hash_mismatch"):
        _build_varied_image_locators(
            manifests,
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )


def test_rejects_duplicate_or_wrong_denominator(tmp_path):
    from twin.cohorts.varied_image_locator import (
        VariedImageLocatorError,
        _build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    with pytest.raises(VariedImageLocatorError, match="manifest_denominator_invalid"):
        _build_varied_image_locators(
            manifests[:-1],
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )
    with pytest.raises(VariedImageLocatorError, match="duplicate_manifest_path"):
        _build_varied_image_locators(
            [*manifests[:-1], manifests[0]],
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )


def test_artifact_drift_fails_in_manifest_binder(tmp_path):
    from twin.cohorts.varied_image_locator import (
        VariedImageLocatorError,
        _build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    Path(payload["rows"][0]["rgb"]["path"]).write_bytes(b"drift")

    with pytest.raises(VariedImageLocatorError, match="rgb_hash_mismatch"):
        _build_varied_image_locators(
            manifests,
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )


@pytest.mark.parametrize(
    ("artifact_key", "mode", "size", "reason"),
    [
        ("rgb", "RGB", (2, 2), "rgb_shape_invalid"),
        ("rgb", "L", (640, 480), "rgb_shape_invalid"),
        ("liquid_mask", "L", (2, 2), "mask_shape_invalid"),
        ("liquid_mask", "RGB", (640, 480), "mask_shape_invalid"),
    ],
)
def test_rejects_hash_bound_artifact_with_wrong_shape_or_mode(
    tmp_path, artifact_key, mode, size, reason
):
    from twin.cohorts.varied_image_locator import (
        VariedImageLocatorError,
        _build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    manifest_path = manifests[0]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(payload["rows"][0][artifact_key]["path"])
    Image.new(mode, size, color=0).save(artifact, format="PNG")
    payload["rows"][0][artifact_key]["sha256"] = _sha(artifact)
    _write_json(manifest_path, payload)

    receipt = json.loads(validation.read_text(encoding="utf-8"))
    receipt["capture_manifest_sha256_by_batch"][payload["batch_id"]] = _sha(
        manifest_path
    )
    _write_json(validation, receipt)

    with pytest.raises(VariedImageLocatorError, match=reason):
        _build_varied_image_locators(
            manifests,
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )


def test_private_builder_rejects_invalid_capability_before_io(tmp_path):
    import twin.cohorts.varied_image_locator as locator

    assert "_build_varied_image_locators" not in locator.__all__
    assert "_write_varied_image_locators" not in locator.__all__
    with pytest.raises(TypeError, match="bind_capture_manifest must be callable"):
        locator._build_varied_image_locators(
            [tmp_path / f"missing-{index}.json" for index in range(6)],
            tmp_path / "missing-receipt.json",
            bind_capture_manifest=None,  # type: ignore[arg-type]
        )


def test_non_png_f2_artifact_keeps_twin_missing_error(tmp_path):
    from twin.cohorts.varied_image_locator import (
        VariedImageLocatorError,
        _build_varied_image_locators,
    )

    manifests, validation = _cohort(tmp_path)
    manifest_path = manifests[0]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rgb = Path(payload["rows"][0]["rgb"]["path"])
    non_png = rgb.with_suffix(".jpg")
    non_png.write_bytes(rgb.read_bytes())
    payload["rows"][0]["rgb"] = {
        "path": str(non_png),
        "sha256": _sha(non_png),
    }
    _write_json(manifest_path, payload)

    receipt = json.loads(validation.read_text(encoding="utf-8"))
    receipt["capture_manifest_sha256_by_batch"][payload["batch_id"]] = _sha(
        manifest_path
    )
    _write_json(validation, receipt)

    with pytest.raises(VariedImageLocatorError, match="rgb_missing"):
        _build_varied_image_locators(
            manifests,
            validation,
            bind_capture_manifest=_bind_capture_manifest,
        )


def test_write_is_no_clobber(tmp_path):
    from twin.cohorts.varied_image_locator import _write_varied_image_locators

    manifests, validation = _cohort(tmp_path)
    output = tmp_path / "locator.json"
    result = _write_varied_image_locators(
        manifests,
        validation,
        output,
        bind_capture_manifest=_bind_capture_manifest,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        _write_varied_image_locators(
            manifests,
            validation,
            output,
            bind_capture_manifest=_bind_capture_manifest,
        )
