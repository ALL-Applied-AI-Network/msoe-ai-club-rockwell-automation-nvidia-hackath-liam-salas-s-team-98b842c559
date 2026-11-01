"""Source-only production materialization for the WP7 VLM runtime.

This module does not select frames, decode video, run a detector, or load model
weights.  It closes already accepted camera artifacts into the exact file set
consumed by :func:`fluidmix.wp7_vlm_decision.load_runtime_context`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import tempfile

from fluidmix.wp7_vlm_decision import (
    ContractError,
    DETERMINISTIC_CONFIG_SHA256,
    build_vial1_bundle,
    load_runtime_context,
    validate_tracker_receipt,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PHASES = ("early", "mid", "end")
_TANKS = ("t100", "t200", "t400")
_CAMERA_FIELDS = {
    "resolution", "frame_rate_hz", "exposure", "white_balance",
    "viewpoint_id", "vial_roi",
}
_RECIPE_FIELDS = {"schema_version", "split_role", "split_version", "runs"}
_RUN_FIELDS = {"run_id", "vials"}
_VIAL_FIELDS = {
    "case_id", "vial_ordinal", "block_recipe_id", "block_position",
    "recipe_id", "previous_recipe_id", "programmed_tank_ml", "dispense_order",
}
_MEASUREMENT_FIELDS = {
    "case_id", "vial_position", "stage_ordinal", "stage_id", "evidence_id",
    "image_ids", "median_lab", "mad_lab", "iqr_lab",
}
_LOCATOR_FIELDS = {
    "path", "sha256", "source_full_frame_sha256", "crop_pixel_sha256",
    "crop_box_xyxy", "source_native_dimensions", "crop_native_dimensions",
}


class ProductionMaterializerError(ValueError):
    """Production source artifacts do not satisfy the closed VLM contract."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProductionMaterializerError("source artifact is not finite JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionMaterializerError(f"{role} must be a lowercase SHA-256")
    return value


def _identifier(value: object, role: str) -> str:
    if (
        not isinstance(value, str) or not value
        or any(token in value for token in ("/", "\\", ".."))
    ):
        raise ProductionMaterializerError(f"{role} must be an opaque identifier")
    return value


def _number(value: object, role: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProductionMaterializerError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ProductionMaterializerError(f"{role} is outside its finite range")
    return result


def _lab(value: object, role: str, *, nonnegative: bool = False) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ProductionMaterializerError(f"{role} must have three channels")
    result = [_number(item, role) for item in value]
    if nonnegative and any(item < 0.0 for item in result):
        raise ProductionMaterializerError(f"{role} must be nonnegative")
    return result


def _validate_recipe(value: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_FIELDS:
        raise ProductionMaterializerError("recipe metadata fields are not closed")
    if value.get("schema_version") != "carryover-recipe-sequence-v1":
        raise ProductionMaterializerError("recipe metadata schema is invalid")
    _identifier(value.get("split_role"), "recipe split role")
    _identifier(value.get("split_version"), "recipe split version")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ProductionMaterializerError("one exact production run is required")
    run = runs[0]
    if not isinstance(run, Mapping) or set(run) != _RUN_FIELDS:
        raise ProductionMaterializerError("recipe run fields are not closed")
    _identifier(run.get("run_id"), "run ID")
    vials = run.get("vials")
    if not isinstance(vials, list) or len(vials) != 3:
        raise ProductionMaterializerError("one exact three-vial cohort is required")
    checked: list[dict[str, object]] = []
    for position, value_vial in enumerate(vials, 1):
        if not isinstance(value_vial, Mapping) or set(value_vial) != _VIAL_FIELDS:
            raise ProductionMaterializerError("recipe vial fields are not closed")
        vial = dict(value_vial)
        if (
            type(vial.get("vial_ordinal")) is not int
            or int(vial["vial_ordinal"]) <= 0
            or vial.get("block_position") != position
        ):
            raise ProductionMaterializerError("recipe vial/block order is invalid")
        for field in ("case_id", "block_recipe_id", "recipe_id"):
            _identifier(vial.get(field), field)
        if vial.get("previous_recipe_id") is not None:
            _identifier(vial.get("previous_recipe_id"), "previous recipe ID")
        amounts, order = vial.get("programmed_tank_ml"), vial.get("dispense_order")
        if not isinstance(amounts, Mapping) or set(amounts) != set(_TANKS):
            raise ProductionMaterializerError("programmed tank mL fields are not closed")
        normalized = {
            tank: _number(amounts[tank], f"{tank} programmed mL") for tank in _TANKS
        }
        if any(amount < 0.0 for amount in normalized.values()):
            raise ProductionMaterializerError("programmed tank mL must be nonnegative")
        active = [tank for tank in _TANKS if normalized[tank] > 0.0]
        if (
            not isinstance(order, list)
            or len(order) not in {2, 3}
            or len(set(order)) != len(order)
            or set(order) != set(active)
        ):
            raise ProductionMaterializerError(
                "production cohort requires two or three exact active recipe stages"
            )
        indices = [_TANKS.index(tank) for tank in order]
        if sum((right - left) % 3 for left, right in zip(indices, indices[1:])) > 2:
            raise ProductionMaterializerError("dispense order is not a cyclic tank subsequence")
        vial["programmed_tank_ml"] = normalized
        checked.append(vial)
    if len({vial["case_id"] for vial in checked}) != 3:
        raise ProductionMaterializerError("recipe vial identities must be unique")
    first_ordinal = int(checked[0]["vial_ordinal"])
    if [vial["vial_ordinal"] for vial in checked] != list(
        range(first_ordinal, first_ordinal + 3)
    ):
        raise ProductionMaterializerError("absolute vial ordinals must be contiguous")
    if len({vial["recipe_id"] for vial in checked}) != 1:
        raise ProductionMaterializerError("all three vials must use one recipe")
    recipe_shape = [
        (vial["dispense_order"], vial["programmed_tank_ml"])
        for vial in checked
    ]
    if any(shape != recipe_shape[0] for shape in recipe_shape[1:]):
        raise ProductionMaterializerError("same-recipe vial amounts/order disagree")
    return dict(run), checked


def _validate_trackers(
    value: object, *, run: Mapping[str, object], vials: Sequence[Mapping[str, object]],
    recipe_sha256: str,
) -> list[dict[str, object]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ProductionMaterializerError("exactly three tracker receipts are required")
    try:
        trackers = [validate_tracker_receipt(item) for item in value]
    except (ContractError, TypeError, ValueError) as error:
        raise ProductionMaterializerError(f"tracker receipt rejected: {error}") from error
    if [item["vial_id"] for item in trackers] != [vial["case_id"] for vial in vials]:
        raise ProductionMaterializerError("tracker vial order/identity disagrees with recipe")
    if any(item["cohort_id"] != run["run_id"] for item in trackers):
        raise ProductionMaterializerError("tracker cohort disagrees with recipe run")
    if any(item["recipe_artifact_sha256"] != recipe_sha256 for item in trackers):
        raise ProductionMaterializerError("tracker recipe artifact SHA differs")
    if len({item["calibration_sha256"] for item in trackers}) != 1:
        raise ProductionMaterializerError("tracker calibration differs within cohort")
    if len({item["source_artifact_sha256"] for item in trackers}) != 3:
        raise ProductionMaterializerError("tracker source artifacts must be distinct")
    active_order = list(vials[0]["dispense_order"])
    for vial, tracker in zip(vials, trackers, strict=True):
        stages = tracker["stages"]
        if not isinstance(stages, list) or len(stages) != len(active_order):
            raise ProductionMaterializerError("tracker stage denominator differs from recipe")
        cumulative = 0.0
        amounts = vial["programmed_tank_ml"]
        for ordinal, (tank_id, stage) in enumerate(zip(active_order, stages, strict=True), 1):
            cumulative += float(amounts[tank_id])  # type: ignore[index]
            if (
                stage["fill_step"] != ordinal
                or stage["stream_step_index"] != ordinal
                or stage["tank_id"] != tank_id
                or float(stage["expected_dispense_ml"]) != float(amounts[tank_id])  # type: ignore[index]
                or float(stage["expected_cumulative_ml"]) != cumulative
                or stage["agreement"] is not True
                or stage["tank_cluster_consistent"] is not True
            ):
                raise ProductionMaterializerError("tracker stage disagrees with recipe")
    return trackers


def _validate_measurements(
    value: object, trackers: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {
        "schema_version", "artifact_type", "cohort_id",
        "tracker_receipt_sha256", "records", "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProductionMaterializerError("measurement receipt fields are not closed")
    if (
        value.get("schema_version") != "wp7-production-camera-measurements-v1"
        or value.get("artifact_type") != "wp7_production_camera_measurements"
        or value.get("cohort_id") != trackers[0]["cohort_id"]
    ):
        raise ProductionMaterializerError("measurement receipt identity is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != _digest(body):
        raise ProductionMaterializerError("measurement receipt hash mismatch")
    if value.get("tracker_receipt_sha256") != [item["receipt_sha256"] for item in trackers]:
        raise ProductionMaterializerError("measurement tracker binding disagrees")
    records = value.get("records")
    stage_count = len(trackers[0]["stages"])  # type: ignore[arg-type]
    expected_record_count = 3 * stage_count
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise ProductionMaterializerError(
            f"exactly {expected_record_count} camera measurements are required"
        )
    checked: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _MEASUREMENT_FIELDS:
            raise ProductionMaterializerError("camera measurement fields are not closed")
        row = dict(record)
        for field in ("case_id", "stage_id", "evidence_id"):
            _identifier(row.get(field), f"measurement {field}")
        if type(row.get("vial_position")) is not int or type(row.get("stage_ordinal")) is not int:
            raise ProductionMaterializerError("measurement ordinals must be integers")
        image_ids = row.get("image_ids")
        if (
            not isinstance(image_ids, list) or len(image_ids) != 3
            or len(set(image_ids)) != 3
            or any(not isinstance(item, str) or not item for item in image_ids)
        ):
            raise ProductionMaterializerError("measurement image IDs are invalid")
        row["median_lab"] = _lab(row.get("median_lab"), "median Lab")
        row["mad_lab"] = _lab(row.get("mad_lab"), "MAD Lab", nonnegative=True)
        row["iqr_lab"] = _lab(row.get("iqr_lab"), "IQR Lab", nonnegative=True)
        checked.append(row)
    checked.sort(key=lambda row: (row["vial_position"], row["stage_ordinal"]))
    expected_pairs = [
        (position, ordinal)
        for position in (1, 2, 3)
        for ordinal in range(1, stage_count + 1)
    ]
    if [(row["vial_position"], row["stage_ordinal"]) for row in checked] != expected_pairs:
        raise ProductionMaterializerError("camera measurement denominator differs")
    stage_ids = [checked[index]["stage_id"] for index in range(stage_count)]
    if any(
        row["stage_id"] != stage_ids[int(row["stage_ordinal"]) - 1]
        for row in checked
    ):
        raise ProductionMaterializerError("camera stage identities differ across vials")
    for row in checked:
        tracker = trackers[int(row["vial_position"]) - 1]
        stage = tracker["stages"][int(row["stage_ordinal"]) - 1]  # type: ignore[index]
        expected_images = [stage["images"][phase] for phase in _PHASES]  # type: ignore[index]
        if row["case_id"] != tracker["vial_id"] or row["image_ids"] != expected_images:
            raise ProductionMaterializerError("measurement source/tracker binding differs")
    return checked


def _validate_camera(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CAMERA_FIELDS:
        raise ProductionMaterializerError("complete camera metadata fields are required")
    if any(
        value[field] is None
        or value[field] == "unavailable"
        or (isinstance(value[field], (str, list, tuple, dict)) and not value[field])
        for field in _CAMERA_FIELDS
    ):
        raise ProductionMaterializerError("production camera metadata is incomplete")
    _number(value["frame_rate_hz"], "camera frame rate", positive=True)
    return dict(value)


def _validate_bindings(
    value: object, trackers: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    expected_count = 3 * len(trackers[0]["stages"]) * len(_PHASES)  # type: ignore[arg-type]
    if not isinstance(value, Mapping) or len(value) != expected_count:
        raise ProductionMaterializerError(
            f"exactly {expected_count} trusted phase image bindings are required"
        )
    expected_ids = [
        stage["images"][phase]  # type: ignore[index]
        for tracker in trackers
        for stage in tracker["stages"]  # type: ignore[union-attr]
        for phase in _PHASES
    ]
    if len(set(expected_ids)) != expected_count or set(value) != set(expected_ids):
        raise ProductionMaterializerError(
            f"exactly {expected_count} tracker-bound phase images are required"
        )
    result: dict[str, dict[str, object]] = {}
    for tracker in trackers:
        previous_stage_end: float | None = None
        for stage in tracker["stages"]:  # type: ignore[union-attr]
            times: list[float] = []
            for phase in _PHASES:
                image_id = stage["images"][phase]  # type: ignore[index]
                row = value[image_id]
                if not isinstance(row, Mapping) or set(row) != _LOCATOR_FIELDS | {"timestamp_seconds"}:
                    raise ProductionMaterializerError("trusted image binding fields are not closed")
                timestamp = _number(row["timestamp_seconds"], "image timestamp")
                times.append(timestamp)
                path = Path(row["path"])
                result[image_id] = {
                    key: (str(path.resolve()) if key == "path" else item)
                    for key, item in row.items()
                    if key != "timestamp_seconds"
                } | {"timestamp_seconds": timestamp}
            if not times[0] < times[1] < times[2]:
                raise ProductionMaterializerError("early/mid/end timestamps are not ordered")
            if previous_stage_end is not None and times[0] <= previous_stage_end:
                raise ProductionMaterializerError("tracker stage timestamps are not ordered")
            previous_stage_end = times[2]
    return result


def _build_decision(
    *, cohort_id: str, target_vial_id: str,
    records: Sequence[Mapping[str, object]], source_artifact_sha256: str,
) -> dict[str, object]:
    stages = []
    stage_count = len(records) // 3
    for ordinal in range(1, stage_count + 1):
        rows = [row for row in records if row["stage_ordinal"] == ordinal]
        rows.sort(key=lambda row: row["vial_position"])
        distances = [
            math.dist(rows[0]["median_lab"], rows[position]["median_lab"])  # type: ignore[arg-type]
            for position in (1, 2)
        ]
        if all(distance <= 1.0 for distance in distances):
            disposition, limits = "keep", []
        elif all(distance >= 2.0 for distance in distances):
            disposition, limits = "discard", []
        else:
            disposition, limits = "insufficient_evidence", ["same_cohort_threshold_deadband"]
        stages.append({
            "stage_id": rows[0]["stage_id"],
            "evidence_id": rows[0]["evidence_id"],
            "disposition": disposition,
            "limit_codes": limits,
            "delta_e76_vial2": distances[0],
            "delta_e76_vial3": distances[1],
        })
    dispositions = [stage["disposition"] for stage in stages]
    overall = (
        "discard" if "discard" in dispositions
        else "keep" if set(dispositions) == {"keep"}
        else "insufficient_evidence"
    )
    receipt: dict[str, object] = {
        "schema_version": "wp7-deterministic-decision-receipt-v1",
        "artifact_type": "wp7_deterministic_decision_receipt",
        "cohort_id": cohort_id,
        "target_vial_id": target_vial_id,
        "disposition": overall,
        "limit_codes": list(dict.fromkeys(
            code for stage in stages for code in stage["limit_codes"]
        )),
        "config_sha256": DETERMINISTIC_CONFIG_SHA256,
        "source_artifact_sha256": source_artifact_sha256,
        "stages": stages,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def _write_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def materialize_production_vlm_cohort(
    *,
    recipe_metadata: object,
    recipe_artifact_sha256: str,
    tracker_receipts: object,
    image_bindings: object,
    measurement_receipt: object,
    camera_metadata: object,
    output_root: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
) -> dict[str, object]:
    """Publish and runtime-validate one exact three-vial production cohort.

    ``image_bindings`` must point at already materialized native-ROI crops.  The
    function verifies but never copies or transforms those images.  The caller
    can invoke this seam immediately after publishing the three tracker receipts.
    """

    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise ProductionMaterializerError(f"output already exists: {output}")
    recipe_sha = _sha256(recipe_artifact_sha256, "recipe artifact SHA-256")
    if recipe_sha != _digest(recipe_metadata):
        raise ProductionMaterializerError("recipe artifact SHA-256 mismatch")
    _sha256(expected_freeze_sha256, "reviewed freeze SHA-256")
    run, vials = _validate_recipe(recipe_metadata)
    trackers = _validate_trackers(
        tracker_receipts, run=run, vials=vials, recipe_sha256=recipe_sha,
    )
    measurements = _validate_measurements(measurement_receipt, trackers)
    camera = _validate_camera(camera_metadata)
    bindings = _validate_bindings(image_bindings, trackers)
    measurement_sha = str(measurement_receipt["receipt_sha256"])  # type: ignore[index]
    decision = _build_decision(
        cohort_id=str(run["run_id"]), target_vial_id=str(vials[0]["case_id"]),
        records=measurements, source_artifact_sha256=measurement_sha,
    )
    stage_count = len(trackers[0]["stages"])
    expected_image_count = 3 * stage_count * len(_PHASES)
    stage_ids = [measurements[index]["stage_id"] for index in range(stage_count)]
    target_tracker = trackers[0]
    cohort_stages = []
    for ordinal, (stage_id, tracker_stage) in enumerate(
        zip(stage_ids, target_tracker["stages"], strict=True), 1,
    ):
        times = [bindings[tracker_stage["images"][phase]]["timestamp_seconds"] for phase in _PHASES]
        cohort_stages.append({
            "stage_id": stage_id,
            "stage_ordinal": ordinal,
            "tank_id": tracker_stage["tank_id"],
            "programmed_ml": tracker_stage["expected_dispense_ml"],
            "dispense_order": ordinal,
            "stage_timing_seconds": [times[0], times[-1]],
            "stream": {
                "evidence_id": f"STREAM-{run['run_id']}-{ordinal}",
                "median_lab": tracker_stage["stream_lab"],
                "spread_lab": tracker_stage["stream_spread_lab"],
            },
        })
    cohort_vials = []
    for position, (vial, tracker) in enumerate(zip(vials, trackers, strict=True), 1):
        vial_stages = []
        vial_rows = [row for row in measurements if row["vial_position"] == position]
        for row, tracker_stage, decision_stage in zip(
            vial_rows, tracker["stages"], decision["stages"], strict=True,
        ):
            measurement = {
                "evidence_id": row["evidence_id"],
                "median_lab": row["median_lab"],
                "mad_lab": row["mad_lab"],
                "iqr_lab": row["iqr_lab"],
                "limit_codes": decision_stage["limit_codes"] if position == 1 else [],
            }
            if position == 1:
                measurement.update({
                    "delta_e76_vial2": decision_stage["delta_e76_vial2"],
                    "delta_e76_vial3": decision_stage["delta_e76_vial3"],
                    "deterministic_disposition": decision_stage["disposition"],
                })
            vial_stages.append({
                "stage_id": row["stage_id"],
                "frames": [{
                    "image_id": tracker_stage["images"][phase],
                    "phase": phase,
                    "timestamp_seconds": bindings[tracker_stage["images"][phase]]["timestamp_seconds"],
                } for phase in _PHASES],
                "measurement": measurement,
            })
        cohort_vials.append({
            "vial_position": position,
            "vial_id": vial["case_id"],
            "stages": vial_stages,
        })
    cohort = {
        "schema_version": "wp7-vlm-cohort-v1",
        "cohort_id": run["run_id"],
        "partition": "production",
        "recipe_id": vials[0]["recipe_id"],
        "previous_recipe_id": vials[0]["previous_recipe_id"],
        "camera_metadata": camera,
        "stages": cohort_stages,
        "vials": cohort_vials,
        "deterministic_decision_receipt": decision,
    }
    locators = {
        image_id: {key: item for key, item in row.items() if key != "timestamp_seconds"}
        for image_id, row in bindings.items()
    }
    try:
        built = build_vial1_bundle(
            cohort, locators, tracker_receipts=trackers, production=True,
        )
    except ContractError as error:
        raise ProductionMaterializerError(f"production bundle rejected: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        hashes = {
            "cohort.json": _write_json(staging / "cohort.json", cohort),
            "image-locators.json": _write_json(staging / "image-locators.json", locators),
        }
        tracker_paths = []
        for position, tracker in enumerate(trackers, 1):
            relative = f"tracker-receipts/vial{position}.json"
            path = staging / relative
            hashes[relative] = _write_json(path, tracker)
            tracker_paths.append(path)
        manifest: dict[str, object] = {
            "schema_version": "wp7-vlm-production-input-manifest-v1",
            "artifact_type": "wp7_vlm_production_runtime_inputs",
            "partition": "production",
            "cohort_id": run["run_id"],
            "artifacts": [
                {"path": relative, "sha256": digest}
                for relative, digest in hashes.items()
            ],
        }
        manifest["manifest_sha256"] = _digest(manifest)
        manifest_file_sha = _write_json(staging / "manifest.json", manifest)
        try:
            context = load_runtime_context(
                staging / "cohort.json", staging / "image-locators.json", tracker_paths,
                freeze_path=freeze_path,
                expected_freeze_sha256=expected_freeze_sha256,
                production=True,
            )
        except ContractError as error:
            raise ProductionMaterializerError(
                f"reviewed production runtime rejected materialization: {error}"
            ) from error
        receipt: dict[str, object] = {
            "schema_version": "wp7-vlm-production-materialization-receipt-v1",
            "artifact_type": "wp7_vlm_production_materialization_receipt",
            "status": "complete",
            "partition": "production",
            "cohort_id": run["run_id"],
            "image_count": expected_image_count,
            "bundle_sha256": context.binding["bundle_sha256"],
            "manifest_file_sha256": manifest_file_sha,
            "artifact_sha256": hashes,
            "source_bindings": {
                "recipe_artifact_sha256": recipe_sha,
                "tracker_receipt_sha256": [item["receipt_sha256"] for item in trackers],
                "tracker_source_artifact_sha256": [item["source_artifact_sha256"] for item in trackers],
                "measurement_receipt_sha256": measurement_sha,
                "deterministic_config_sha256": DETERMINISTIC_CONFIG_SHA256,
                "deterministic_receipt_sha256": decision["receipt_sha256"],
                "reviewed_development_freeze_sha256": expected_freeze_sha256,
                "materializer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "runtime_validation": {
                "production": True,
                "ordered_image_count": len(built.prompt_payload["images"]),
                "status": "pass",
            },
        }
        receipt["receipt_sha256"] = _digest(receipt)
        _write_json(staging / "materialization-receipt.json", receipt)
        if output.exists() or output.is_symlink():
            raise ProductionMaterializerError(f"output already exists: {output}")
        staging.rename(output)
        return receipt
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise



__all__ = ["ProductionMaterializerError", "materialize_production_vlm_cohort"]
