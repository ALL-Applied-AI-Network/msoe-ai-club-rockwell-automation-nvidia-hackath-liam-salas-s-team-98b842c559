from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import cv2
import pytest

from fluidmix.decision.vlm import _bind_materialization_receipt
from fluidmix.explanation.vlm.development_materializer import canonical_sha256
from fluidmix.explanation.vlm.native_roi_rematerializer import (
    NativeRoiRematerializerError,
    rematerialize_native_roi,
)


ATTEMPT09 = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp7-vlm-dev-bundle-materializer-01a18128-attempt09"
)


def _require_attempt09() -> Path:
    if not ATTEMPT09.is_dir():
        pytest.skip("accepted attempt09 development artifact is not mounted")
    return ATTEMPT09


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rematerializes_native_roi_and_publishes_freeze_compatible_receipt(tmp_path):
    source = _require_attempt09()
    output = tmp_path / "attempt10-not-published"

    receipt = rematerialize_native_roi(source, output)

    assert receipt["schema_version"] == "wp7-vlm-development-bundle-materialization-v1"
    assert receipt["status"] == "complete"
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["source_bindings"]["source_attempt09_receipt_file_sha256"] == _sha(
        source / "materialization-receipt.json"
    )
    source_receipt = json.loads((source / "materialization-receipt.json").read_bytes())
    assert receipt["source_bindings"]["source_attempt09_receipt_body_sha256"] == source_receipt[
        "receipt_sha256"
    ]
    locators = json.loads((output / "image-locators.json").read_bytes())
    inventory = json.loads((output / "image-inventory.json").read_bytes())
    assert len(locators) == len(inventory) == 27
    for image_id, locator in locators.items():
        image = cv2.imread(str(output / locator["path"]), cv2.IMREAD_COLOR)
        assert image.shape == (223, 55, 3)
        assert locator["crop_box_xyxy"] == [283, 194, 338, 417]
        assert locator["source_native_dimensions"] == [640, 480]
        assert locator["crop_native_dimensions"] == [55, 223]
        assert locator["sha256"] == _sha(output / locator["path"])
        assert inventory[image_id] == {
            key: value for key, value in locator.items() if key != "path"
        }
    assert (output / "v3-inventory.json").read_bytes() == (source / "v3-inventory.json").read_bytes()
    assert (output / "v3-development-manifest.json").read_bytes() == (
        source / "v3-development-manifest.json"
    ).read_bytes()
    assert (output / "cohort.json").read_bytes() == (source / "cohort.json").read_bytes()
    trackers = []
    for position in range(1, 4):
        relative = Path("tracker-receipts") / f"vial{position}-tracker-receipt.json"
        assert (output / relative).read_bytes() == (source / relative).read_bytes()
        trackers.append(json.loads((output / relative).read_bytes()))
    records = [
        {"dataset_version": version, "manifest_sha256": _sha(output / path)}
        for version, path in (
            ("V3", "v3-development-manifest.json"),
            ("V4", "v4-development-manifest.json"),
        )
    ]
    assert _bind_materialization_receipt(
        output / "materialization-receipt.json",
        development_records=records,
        trackers=trackers,
    ) == _sha(output / "materialization-receipt.json")

    repeated = tmp_path / "repeat"
    rematerialize_native_roi(source, repeated)
    assert (repeated / "image-inventory.json").read_bytes() == (
        output / "image-inventory.json"
    ).read_bytes()


def test_rejects_tampered_input_before_creating_output(tmp_path):
    source = tmp_path / "tampered-attempt09"
    shutil.copytree(_require_attempt09(), source)
    image = next((source / "images").iterdir())
    image.write_bytes(image.read_bytes() + b"tampered")
    output = tmp_path / "must-not-exist"

    with pytest.raises(NativeRoiRematerializerError, match="hash"):
        rematerialize_native_roi(source, output)

    assert not output.exists()


def test_refuses_existing_output_without_reading_source(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(NativeRoiRematerializerError, match="already exists"):
        rematerialize_native_roi(tmp_path / "missing-source", output)
