"""Upgrade accepted WP7 development full frames to native-resolution ROI crops."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from fluidmix.wp7_vlm_decision import build_vial1_bundle
from fluidmix.wp7_vlm_development_materializer import canonical_sha256, crop_vial_roi


class NativeRoiRematerializerError(ValueError):
    """Accepted input drifted or the no-clobber output cannot be published."""


ACCEPTED_ATTEMPT09_ROOT = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp7-vlm-dev-bundle-materializer-01a18128-attempt09"
)
ACCEPTED_ATTEMPT09_RECEIPT_FILE_SHA256 = "f22d060ee46d4a358efd53112626c2754926f1e239d3a6784f49f15de02b513c"
REMATERIALIZER_BASE_COMMIT = "b5e63cb5ceb54f0f3d5688b48613ce2e7362e620"
FROZEN_VIAL_ROI = [283, 194, 338, 417]
_SCHEMA = "wp7-vlm-development-bundle-materialization-v1"
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise NativeRoiRematerializerError("input artifact path is invalid")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise NativeRoiRematerializerError("input artifact path escapes source root") from None
    if path.is_symlink() or not path.is_file():
        raise NativeRoiRematerializerError(f"input artifact is absent or linked: {relative}")
    return path


def _load(path: Path, role: str) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeRoiRematerializerError(f"{role} is not valid JSON") from error


def _bound(path: Path, expected: object, role: str) -> None:
    if not isinstance(expected, str) or _sha(path) != expected:
        raise NativeRoiRematerializerError(f"{role} hash differs from accepted attempt09")


def _manifest(root: Path, path: Path, version: str) -> dict[str, object]:
    value = _load(path, f"{version} manifest")
    if (not isinstance(value, dict)
            or value.get("schema_version") != "wp7-vlm-development-input-manifest-v1"
            or value.get("dataset_version") != version
            or value.get("partition") != "twin_train"
            or value.get("data_origin") != "synthetic_twin"
            or not isinstance(value.get("artifacts"), list)):
        raise NativeRoiRematerializerError(f"{version} manifest schema differs")
    paths: set[str] = set()
    for row in value["artifacts"]:
        if not isinstance(row, dict) or set(row) != {"artifact_id", "path", "sha256"}:
            raise NativeRoiRematerializerError(f"{version} manifest artifact is invalid")
        if row["path"] in paths:
            raise NativeRoiRematerializerError(f"{version} manifest repeats an artifact")
        paths.add(row["path"])
        _bound(_file(root, row["path"]), row["sha256"], f"{version} {row['path']}")
    return value


def _validate_source(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise NativeRoiRematerializerError("accepted attempt09 source root is absent or linked")
    receipt_path = _file(root, "materialization-receipt.json")
    _bound(receipt_path, ACCEPTED_ATTEMPT09_RECEIPT_FILE_SHA256, "attempt09 receipt file")
    receipt = _load(receipt_path, "attempt09 receipt")
    if not isinstance(receipt, dict):
        raise NativeRoiRematerializerError("attempt09 receipt is invalid")
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if (receipt.get("schema_version") != _SCHEMA or receipt.get("status") != "complete"
            or claimed != canonical_sha256(body)):
        raise NativeRoiRematerializerError("attempt09 receipt schema/status/self-hash is invalid")

    direct = {"cohort.json": receipt.get("cohort_sha256"),
              "bundle.json": receipt.get("bundle_sha256"),
              "image-inventory.json": receipt.get("image_inventory_sha256"),
              "image-locators.json": receipt.get("image_locators_sha256")}
    for relative, expected in direct.items():
        _bound(_file(root, relative), expected, relative)
    tracker_paths = [f"tracker-receipts/vial{n}-tracker-receipt.json" for n in range(1, 4)]
    tracker_hashes = receipt.get("tracker_receipt_file_sha256")
    if not isinstance(tracker_hashes, list) or len(tracker_hashes) != 3:
        raise NativeRoiRematerializerError("attempt09 tracker hashes are invalid")
    for relative, expected in zip(tracker_paths, tracker_hashes, strict=True):
        _bound(_file(root, relative), expected, relative)

    manifest_hashes = receipt.get("development_manifest_sha256")
    if not isinstance(manifest_hashes, Mapping) or set(manifest_hashes) != {"V3", "V4"}:
        raise NativeRoiRematerializerError("attempt09 manifest hashes are invalid")
    manifests = {}
    for version in ("V3", "V4"):
        path = _file(root, f"{version.lower()}-development-manifest.json")
        _bound(path, manifest_hashes[version], f"{version} manifest")
        manifests[version] = _manifest(root, path, version)
    if ({row["path"] for row in manifests["V3"]["artifacts"]} != {"v3-inventory.json"}
            or {row["path"] for row in manifests["V4"]["artifacts"]}
            != set(direct) | set(tracker_paths)):
        raise NativeRoiRematerializerError("attempt09 manifest inventory is incomplete")

    locators = _load(_file(root, "image-locators.json"), "image locators")
    inventory = _load(_file(root, "image-inventory.json"), "image inventory")
    if (not isinstance(locators, dict) or not isinstance(inventory, dict)
            or set(locators) != set(inventory) or len(locators) != receipt.get("image_count")):
        raise NativeRoiRematerializerError("attempt09 image inventory/locators differ")
    frames = {}
    for image_id in sorted(locators):
        locator = locators[image_id]
        if (not isinstance(locator, dict) or set(locator) != {"path", "sha256"}
                or inventory[image_id] != locator["sha256"]):
            raise NativeRoiRematerializerError("attempt09 image locator binding is invalid")
        path = _file(root, locator["path"])
        _bound(path, locator["sha256"], f"full-frame image {image_id}")
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise NativeRoiRematerializerError(f"full-frame image {image_id} is not native 640x480 color")
        frames[image_id] = frame
    return {"receipt": receipt, "frames": frames, "tracker_paths": tracker_paths}


def _write(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _source_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise NativeRoiRematerializerError("repository source commit is invalid")
    return value


def rematerialize_native_roi(source_root: str | Path, output_root: str | Path) -> dict[str, object]:
    """Rematerialize accepted attempt09 full frames into a fresh ROI-crop root."""
    source, output = Path(source_root), Path(output_root)
    if output.exists() or output.is_symlink():
        raise NativeRoiRematerializerError("output already exists")
    checked = _validate_source(source)
    old_receipt = checked["receipt"]
    encoded, locators = {}, {}
    for image_id, frame in checked["frames"].items():
        crop, binding = crop_vial_roi(frame, FROZEN_VIAL_ROI)
        ok, png = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise NativeRoiRematerializerError(f"PNG encoding failed for {image_id}")
        raw = png.tobytes()
        encoded[image_id] = raw
        locators[image_id] = {"path": f"images/{image_id}.png",
                              "sha256": hashlib.sha256(raw).hexdigest(), **binding}

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise NativeRoiRematerializerError("output already exists") from None
    try:
        (output / "images").mkdir()
        for image_id, raw in encoded.items():
            (output / "images" / f"{image_id}.png").write_bytes(raw)
        preserved = ["cohort.json", "v3-inventory.json", "v3-development-manifest.json",
                     *checked["tracker_paths"]]
        for relative in preserved:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_file(source, relative).read_bytes())
        _write(output / "image-locators.json", locators)
        inventory = {key: {field: value for field, value in row.items() if field != "path"}
                     for key, row in locators.items()}
        _write(output / "image-inventory.json", inventory)
        cohort = _load(output / "cohort.json", "cohort")
        trackers = [_load(output / path, "tracker") for path in checked["tracker_paths"]]
        bound = {key: {**row, "path": str(output / row["path"])} for key, row in locators.items()}
        _write(output / "bundle.json", build_vial1_bundle(
            cohort, bound, tracker_receipts=trackers).prompt_payload)

        old_v4 = _load(_file(source, "v4-development-manifest.json"), "V4 manifest")
        _write(output / "v4-development-manifest.json", {
            **old_v4, "artifacts": [{**row, "sha256": _sha(output / row["path"])}
                                     for row in old_v4["artifacts"]]})
        source_bindings = {**old_receipt["source_bindings"],
            "source_attempt09_receipt_file_sha256": _sha(source / "materialization-receipt.json"),
            "source_attempt09_receipt_body_sha256": old_receipt["receipt_sha256"],
            "tracker_source_sha256": _sha(
                _ROOT / "src" / "fluidmix" / "wp7_fill_step_tracker.py"
            ),
            "rematerializer_cli_source_sha256": _sha(_ROOT / "scripts" / "rematerialize_wp7_vlm_native_roi.py"),
            "rematerializer_module_source_sha256": _sha(Path(__file__)),
            "bundle_builder_source_sha256": _sha(_ROOT / "src" / "fluidmix" / "wp7_vlm_decision.py"),
            "rematerializer_base_commit": REMATERIALIZER_BASE_COMMIT,
            "rematerializer_source_commit": _source_commit()}
        receipt = {**old_receipt, "cohort_sha256": _sha(output / "cohort.json"),
            "bundle_sha256": _sha(output / "bundle.json"),
            "image_inventory_sha256": _sha(output / "image-inventory.json"),
            "image_locators_sha256": _sha(output / "image-locators.json"),
            "tracker_receipt_file_sha256": [_sha(output / path) for path in checked["tracker_paths"]],
            "development_manifest_sha256": {
                "V3": _sha(output / "v3-development-manifest.json"),
                "V4": _sha(output / "v4-development-manifest.json")},
            "source_bindings": source_bindings}
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write(output / "materialization-receipt.json", receipt)
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise
