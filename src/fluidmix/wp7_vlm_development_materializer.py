"""Path-free WP7 VLM bundles from accepted development evidence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
import re
import numpy as np

from fluidmix.wp7_fill_step_tracker import validate_fill_step_receipt


class DevelopmentMaterializerError(ValueError):
    """Development evidence is absent, stale, or outside the frozen boundary."""


ATTEMPT04_SCHEMA = "wp7-fill-step-development-materialization-v1"
ATTEMPT04_SOURCE_COMMIT = "4ab6d9b198e6d079165b9e40c21361e3cd745065"
TRACKER_SYNC_COMMIT = "354278056f1ab3bc262f8dbbc18faa9adbdc9c25"
TRACKER_SOURCE_SHA256 = "a3f21046df7e3b0c4b68d82854ab0a3f8116b6eca68594a69ce3939b07386b0d"
DECISION_CONFIG_SHA256 = "6145a4da5c452ff4c8e8d4c6018ffeb5009410501f2a7438efd867317962057e"
COMPLETE_RUNS = {"RUN-V4-TRAIN-01": "train", "RUN-V4-VALIDATION-02": "validation", "RUN-V4-VALIDATION-03": "validation"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def crop_vial_roi(
    frame: object, roi: Sequence[object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Crop one native-resolution RGB/BGR frame and bind its source pixels."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise DevelopmentMaterializerError("ROI source frame must be uint8 three-channel")
    if (
        isinstance(roi, (str, bytes)) or not isinstance(roi, Sequence)
        or len(roi) != 4 or any(type(item) is not int for item in roi)
    ):
        raise DevelopmentMaterializerError("ROI coordinates are invalid")
    x1, y1, x2, y2 = (int(item) for item in roi)
    height, width = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise DevelopmentMaterializerError("ROI lies outside the native frame")
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    return crop, {
        "crop_box_xyxy": [x1, y1, x2, y2],
        "source_native_dimensions": [width, height],
        "crop_native_dimensions": [x2 - x1, y2 - y1],
        "source_full_frame_sha256": hashlib.sha256(
            np.ascontiguousarray(image).tobytes()
        ).hexdigest(),
        "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
    }


def canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise DevelopmentMaterializerError("nonfinite or non-JSON evidence") from error
    return hashlib.sha256(raw).hexdigest()


def _hash(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DevelopmentMaterializerError(f"{role} SHA-256 is invalid")
    return value


def _lab(value: object, role: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise DevelopmentMaterializerError(f"{role} Lab is invalid")
    out = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, Real) or not math.isfinite(float(channel)):
            raise DevelopmentMaterializerError(f"{role} Lab is invalid")
        out.append(float(channel))
    return out[0], out[1], out[2]


def select_complete_cohort(inventory: object, plan: object, *, run_id: str, inventory_file_sha256: str, plan_file_sha256: str) -> dict[str, object]:
    """Select one accepted complete cohort and bind recipe/tracker topology."""
    if not isinstance(inventory, Mapping):
        raise DevelopmentMaterializerError("attempt04 inventory must be an object")
    body = dict(inventory); claimed = body.pop("receipt_sha256", None)
    if (inventory.get("schema_version") != ATTEMPT04_SCHEMA or inventory.get("artifact_type") != "development_only_v3_v4_fill_step_comparison" or inventory.get("source_commit") != ATTEMPT04_SOURCE_COMMIT or inventory.get("device") != "cpu" or inventory.get("excluded_partitions") != ["evaluation", "test", "video5"] or claimed != canonical_sha256(body)):
        raise DevelopmentMaterializerError("attempt04 inventory identity/self-hash is invalid")
    if run_id not in COMPLETE_RUNS:
        raise DevelopmentMaterializerError("run is not an accepted complete development cohort")
    inventory_sha, plan_sha = _hash(inventory_file_sha256, "inventory file"), _hash(plan_file_sha256, "plan file")
    try:
        if inventory["source_bindings"]["plans"]["v4"]["sha256"] != plan_sha:  # type: ignore[index]
            raise DevelopmentMaterializerError("V4 plan hash differs from attempt04")
    except (KeyError, TypeError):
        raise DevelopmentMaterializerError("attempt04 V4 plan binding is absent") from None
    rows_value, receipts = inventory.get("per_vial"), inventory.get("supported_fill_step_receipts")
    if not isinstance(rows_value, list) or not isinstance(receipts, Mapping):
        raise DevelopmentMaterializerError("attempt04 supported-vial inventory is absent")
    rows = [row for row in rows_value if isinstance(row, Mapping) and row.get("run_id") == run_id and row.get("new_supported") is True]
    if len(rows) != 3 or len({row.get("vial_id") for row in rows}) != 3:
        raise DevelopmentMaterializerError("complete accepted three-vial cohort is required")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("execution_schedules"), list):
        raise DevelopmentMaterializerError("V4 plan schedules are absent")
    schedules = [row for row in plan["execution_schedules"] if isinstance(row, Mapping) and isinstance(row.get("case_binding"), Mapping) and row["case_binding"].get("run_id") == run_id]
    schedules.sort(key=lambda row: row["case_binding"].get("vial_ordinal"))
    if len(schedules) != 3:
        raise DevelopmentMaterializerError("complete three-vial plan cohort is required")
    vials, trackers, cohort_ids = [], [], set()
    for position, schedule in enumerate(schedules, 1):
        binding, model_input = schedule["case_binding"], schedule.get("model_input_record")
        if not isinstance(model_input, Mapping) or binding.get("vial_ordinal") != position:
            raise DevelopmentMaterializerError("plan vial order/model input is invalid")
        case_id, key = binding.get("case_id"), f"v4:{binding.get('case_id')}"
        matching = [row for row in rows if row.get("vial_id") == key]
        if len(matching) != 1 or key not in receipts:
            raise DevelopmentMaterializerError("complete accepted three-vial cohort is required")
        try:
            receipt = validate_fill_step_receipt(receipts[key])
        except (TypeError, ValueError) as error:
            raise DevelopmentMaterializerError(f"accepted tracker receipt invalid: {error}") from error
        if (receipt["vial_id"] != case_id or receipt["cohort_id"] != matching[0].get("receipt_cohort_id") or receipt["source_artifact_sha256"] != matching[0].get("video_sha256") or binding.get("batch_id") != matching[0].get("batch_id")):
            raise DevelopmentMaterializerError("tracker identity/source differs from attempt04")
        order, amounts = model_input.get("dispense_order"), model_input.get("programmed_ml")
        if not isinstance(order, list) or not isinstance(amounts, Mapping):
            raise DevelopmentMaterializerError("recipe order/amounts are absent")
        observed = [(tank, float(amounts[tank])) for tank in order if float(amounts[tank]) > 0]
        expected = [(stage["tank_id"], float(stage["expected_dispense_ml"])) for stage in receipt["stages"]]
        if observed != expected:
            raise DevelopmentMaterializerError("tracker stages differ from plan recipe/order")
        cohort_ids.add(receipt["cohort_id"]); trackers.append(receipt)
        vials.append({"vial_position": position, "case_id": case_id, "vial_id": case_id, "current_recipe_id": binding.get("current_recipe_id"), "previous_recipe_id": binding.get("previous_recipe_id"), "batch_id": binding.get("batch_id"), "split_role": binding.get("split_role"), "video_sha256": matching[0].get("video_sha256")})
    if len(cohort_ids) != 1 or len({row["current_recipe_id"] for row in vials}) != 1 or any(row["split_role"] != COMPLETE_RUNS[run_id] for row in vials):
        raise DevelopmentMaterializerError("cohort tracker/recipe/partition differs")
    return {"run_id": run_id, "cohort_id": next(iter(cohort_ids)), "batch_id": vials[0]["batch_id"], "split_role": COMPLETE_RUNS[run_id], "recipe_id": vials[0]["current_recipe_id"], "target_previous_recipe_id": vials[0]["previous_recipe_id"], "vials": vials, "tracker_receipts": trackers, "development_binding": {"dataset_version": "V4", "run_id": run_id, "batch_id": vials[0]["batch_id"], "source_materialization_receipt_sha256": inventory_sha, "source_materialization_receipt_body_sha256": inventory["receipt_sha256"], "source_commit": ATTEMPT04_SOURCE_COMMIT, "plan_sha256": plan_sha, "tracker_sync_commit": TRACKER_SYNC_COMMIT, "tracker_source_sha256": TRACKER_SOURCE_SHA256}}


def aggregate_masked_lab(frames: Sequence[object], masks: Sequence[object]) -> dict[str, object]:
    """Return Lab statistics over exactly three CNN-masked frames."""
    if len(frames) != 3 or len(masks) != 3:
        raise DevelopmentMaterializerError("exactly early/mid/end frames and masks are required")
    from skimage.color import rgb2lab
    selected = []
    for frame_raw, mask_raw in zip(frames, masks, strict=True):
        frame, mask = np.asarray(frame_raw), np.asarray(mask_raw)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise DevelopmentMaterializerError("frame must be uint8 RGB")
        if mask.shape != frame.shape[:2] or mask.dtype != np.bool_ or not np.any(mask):
            raise DevelopmentMaterializerError("CNN mask is empty or mismatched")
        selected.append(rgb2lab(frame.astype(np.float64) / 255.0)[mask])
    values = np.concatenate(selected, axis=0); median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0); q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
    return {"frame_count": 3, "liquid_pixel_count": int(values.shape[0]), "median_lab": [float(x) for x in median], "mad_lab": [float(x) for x in mad], "iqr_lab": [float(x) for x in q75 - q25]}


def build_deterministic_decision(*, cohort_id: str, target_vial_id: str, stage_measurements: Sequence[Sequence[Mapping[str, object]]], source_artifact_sha256: str) -> dict[str, object]:
    """Apply frozen same-cohort keep<=1/discard>=2 dE76 rule."""
    stages = []
    for ordinal, rows in enumerate(stage_measurements, 1):
        if len(rows) != 3 or len({row.get("stage_id") for row in rows}) != 1:
            raise DevelopmentMaterializerError("each stage requires aligned target and references")
        labs = [_lab(row.get("median_lab"), f"stage {ordinal}") for row in rows]
        d2, d3 = math.dist(labs[0], labs[1]), math.dist(labs[0], labs[2])
        if d2 <= 1.0 and d3 <= 1.0: disposition, limits = "keep", []
        elif d2 >= 2.0 and d3 >= 2.0: disposition, limits = "discard", []
        else: disposition, limits = "insufficient_evidence", ["same_cohort_threshold_deadband"]
        stages.append({"stage_id": rows[0].get("stage_id"), "evidence_id": rows[0].get("evidence_id"), "disposition": disposition, "limit_codes": limits, "delta_e76_vial2": d2, "delta_e76_vial3": d3})
    dispositions = [row["disposition"] for row in stages]
    overall = "discard" if "discard" in dispositions else ("keep" if dispositions and set(dispositions) == {"keep"} else "insufficient_evidence")
    receipt: dict[str, object] = {"schema_version": "wp7-deterministic-decision-receipt-v1", "artifact_type": "wp7_deterministic_decision_receipt", "cohort_id": cohort_id, "target_vial_id": target_vial_id, "disposition": overall, "limit_codes": list(dict.fromkeys(code for row in stages for code in row["limit_codes"])), "config_sha256": DECISION_CONFIG_SHA256, "source_artifact_sha256": _hash(source_artifact_sha256, "source artifact"), "stages": stages}
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt
