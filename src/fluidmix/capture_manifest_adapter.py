"""Bind verified twin stage-capture manifests to measured evidence."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

from fluidmix.render_process import load_strict_json_snapshot
from fluidmix.stage_decision import StageStatistic
from fluidmix.vision_measurement import evaluate_model_plan, measure_image
from fluidmix.deploy_input_policy import validate_deploy_input_fields


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"CASE_[0-9a-f]{16}\Z")
_REQUIRED_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "data_origin",
    "family_id",
    "batch_id",
    "recipe_id",
    "split_role",
    "split_version",
    "view_id",
    "stage_slots",
    "rows",
}
_STAGE_KEYS = {"stage_id", "stage_ordinal", "time_code"}
_ROW_KEYS = {
    "vial_id",
    "stage_id",
    "stage_ordinal",
    "time_code",
    "view_id",
    "verified",
    "rgb",
    "liquid_mask",
}
_ARTIFACT_KEYS = {"path", "sha256"}
_VIALS = {"V1", "V2", "V3"}
_SPLIT_ROLES = {"train", "val", "test", "buffer"}
_SPLIT_VERSION = "inn-7-splits-v2"
_F2_REQUIRED_FIELDS = {
    "schema_version", "artifact_type", "status", "data_origin", "family_id",
    "batch_id", "run_id", "recipe_id", "split_role", "split_version", "view_id",
    "runtime_binding", "runtime_receipt", "stage_events", "rows",
}
_F2_VARIED_REQUIRED_FIELDS = _F2_REQUIRED_FIELDS - {"family_id"}
_F2_VARIED_SPLIT_ROLES = {"train", "validation", "evaluation"}
_F2_VARIED_SPLIT_VERSIONS = frozenset(
    ("whole-run-varied-v3", "whole-run-varied-v4")
)
_F2_RUNTIME_KEYS = {
    "provider", "runtime_name", "isaac_sim_version", "runtime_id", "source_sha256",
}
_F2_RECEIPT_LOCATOR_KEYS = {"path", "sha256"}
_F2_RECEIPT_KEYS = {
    "schema_version", "artifact_type", "status", "provider", "runtime_id",
    "isaac_sim_version", "source_sha256", "gates",
}
_F2_VARIED_RECEIPT_KEYS = {
    "schema_version", "artifact_type", "status", "provider", "runtime_id",
    "isaac_sim_version", "source_sha256", "hard_gates",
    "particle_height_diagnostic",
}
_F2_ACCEPTANCE_GATES = {
    "scene_parity",
    "finite_particle_state",
    "containment",
    "zero_leakage",
    "volume_height",
    "stream_continuity",
    "source_color",
    "numeric_mixing",
    "visual_mixing",
    "semantics_alignment",
    "timing",
}
_F2_VARIED_HARD_GATES = (_F2_ACCEPTANCE_GATES - {"volume_height"}) | {
    "rendered_body_volume_height"
}
_F2_PARTICLE_HEIGHT_KEYS = {
    "hard_gate", "physics_pass_claimed", "status", "bounds_passed",
    "warning_count", "warned_vial_count", "vial_count", "claim", "vials",
}
_F2_PARTICLE_HEIGHT_VIAL_KEYS = {
    "case_id", "vial_id", "status", "bounds_passed", "warning_count",
}
_F2_EVENT_KEYS = {
    "event_id", "stage_ordinal", "tank_id", "tank_order", "physics_step",
    "frame_id", "event_relative_timestamp_s", "final_observation_offset_s",
}
_F2_ROW_KEYS = {
    "vial_id", "event_id", "stage_ordinal", "physics_step", "frame_id",
    "event_relative_timestamp_s", "final_observation_offset_s", "view_id",
    "verified", "rgb", "liquid_mask",
}


class CaptureManifestError(ValueError):
    """A capture manifest failed a named fail-closed boundary."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


def bind_stage_capture_manifest(
    manifest: object,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, object]:
    """Bind one future verified twin 3-vial/3-stage capture manifest."""

    payload, source_sha, source_base = _load_manifest(manifest, base_dir)
    try:
        validate_deploy_input_fields(payload, require_allowlist=False)
    except ValueError as exc:
        raise CaptureManifestError("forbidden_oracle_field", str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise CaptureManifestError("manifest_malformed", "manifest must be an object")
    if payload.get("artifact_type") == "f2_stage_capture_manifest":
        return _bind_f2_capture(
            payload,
            source_sha=source_sha,
            source_base=source_base,
            varied=payload.get("schema_version") == 3,
        )
    missing = sorted(_REQUIRED_FIELDS.difference(payload))
    if missing:
        raise CaptureManifestError("manifest_fields_missing", repr(missing))
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != (
        "twin_stage_capture_manifest"
    ):
        raise CaptureManifestError("manifest_schema_invalid")
    if payload.get("status") != "verified":
        raise CaptureManifestError("manifest_unverified")
    if payload.get("data_origin") != "twin":
        raise CaptureManifestError("data_origin_invalid")
    batch_id = _text(payload.get("batch_id"), "batch_id")
    family_id = _text(payload.get("family_id"), "family_id")
    recipe_id = _text(payload.get("recipe_id"), "recipe_id")
    split_role = _text(payload.get("split_role"), "split_role")
    split_version = _text(payload.get("split_version"), "split_version")
    if split_role not in _SPLIT_ROLES:
        raise CaptureManifestError("split_role_invalid", split_role)
    if split_version != _SPLIT_VERSION:
        raise CaptureManifestError("split_version_invalid", split_version)
    view_id = _text(payload.get("view_id"), "view_id")
    if view_id != "vial_fill":
        raise CaptureManifestError("wrong_view_id", view_id)

    stages = _validate_stage_slots(payload.get("stage_slots"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise CaptureManifestError("rows_malformed")
    expected_keys = {(vial, stage["stage_id"]) for vial in _VIALS for stage in stages}
    seen: set[tuple[str, str]] = set()
    checked_rows: list[dict[str, object]] = []
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping) or set(raw) != _ROW_KEYS:
            raise CaptureManifestError("row_malformed", f"row {row_number}")
        if raw.get("verified") is not True:
            raise CaptureManifestError("row_unverified", f"row {row_number}")
        if raw.get("view_id") != view_id:
            raise CaptureManifestError("wrong_view_id", f"row {row_number}")
        vial_id = _text(raw.get("vial_id"), "vial_id")
        if vial_id not in _VIALS:
            raise CaptureManifestError("vial_id_invalid", vial_id)
        stage_id = _text(raw.get("stage_id"), "stage_id")
        key = (vial_id, stage_id)
        if key in seen:
            raise CaptureManifestError("duplicate_stage_row", f"row {row_number}: {key}")
        seen.add(key)
        stage = next((item for item in stages if item["stage_id"] == stage_id), None)
        if stage is None:
            raise CaptureManifestError("stage_id_invalid", stage_id)
        if raw.get("stage_ordinal") != stage["stage_ordinal"]:
            raise CaptureManifestError("wrong_stage_ordinal", f"row {row_number}")
        if raw.get("time_code") != stage["time_code"]:
            raise CaptureManifestError("wrong_time_code", f"row {row_number}")
        checked_rows.append(dict(raw))
    missing_rows = sorted(expected_keys.difference(seen))
    if missing_rows:
        raise CaptureManifestError("missing_stage_row", repr(missing_rows))
    if len(rows) != 9:
        raise CaptureManifestError("row_count_invalid", f"expected 9, got {len(rows)}")

    locators: dict[str, dict[str, str]] = {}
    measurements: list[dict[str, object]] = []
    statistic_records: list[dict[str, object]] = []
    model_inputs: list[dict[str, object]] = []
    ordered_rows = sorted(
        checked_rows,
        key=lambda item: (str(item["vial_id"]), int(item["stage_ordinal"])),
    )
    for row in ordered_rows:
        image_id = f"{batch_id}-{row['vial_id']}-{row['stage_id']}"
        rgb_path, rgb_sha = _bind_artifact(row["rgb"], "rgb", source_base)
        mask_path, mask_sha = _bind_artifact(row["liquid_mask"], "mask", source_base)
        _require_fillcam_resolution(rgb_path, "rgb")
        _require_fillcam_resolution(mask_path, "mask")
        locator = {
            "rgb_path": str(rgb_path),
            "rgb_sha256": rgb_sha,
            "mask_path": str(mask_path),
            "mask_sha256": mask_sha,
        }
        measured = measure_image(image_id, locator)
        if measured["validity"] != "valid":
            code = str(measured["limit_codes"][0])
            if code == "mask_empty":
                code = "empty_mask"
            raise CaptureManifestError(code, image_id)
        locators[image_id] = locator
        measurement = {
            **measured,
            "family_id": family_id,
            "batch_id": batch_id,
            "vial_id": row["vial_id"],
            "recipe_id": recipe_id,
            "split_role": split_role,
            "split_version": split_version,
            "stage_id": row["stage_id"],
            "stage_ordinal": row["stage_ordinal"],
            "time_code": row["time_code"],
            "view_id": view_id,
        }
        measurements.append(measurement)
        evidence_id = f"EVID-{image_id}"
        measurement_id = f"MEAS-{image_id}"
        stat = StageStatistic(
            evidence_id=evidence_id,
            measurement_id=measurement_id,
            image_id=image_id,
            batch_id=batch_id,
            vial_id=str(row["vial_id"]),
            recipe_id=recipe_id,
            data_origin="twin",
            split_role=split_role,
            split_version=split_version,
            stage_id=str(row["stage_id"]),
            stage_ordinal=int(row["stage_ordinal"]),
            lab=tuple(float(value) for value in measured["lab_raw"]),
        )
        stat_record = {
            "evidence_id": evidence_id,
            "measurement_id": measurement_id,
            "image_id": image_id,
            "batch_id": batch_id,
            "family_id": family_id,
            "vial_id": row["vial_id"],
            "recipe_id": recipe_id,
            "data_origin": "twin",
            "split_role": split_role,
            "split_version": split_version,
            "stage_id": row["stage_id"],
            "stage_ordinal": row["stage_ordinal"],
            "time_code": row["time_code"],
            "lab": list(stat.lab),
            "validity": "valid",
            "limit_codes": [],
        }
        statistic_records.append(stat_record)
        model_inputs.append(
            {
                "evidence_id": evidence_id,
                "measurement_id": measurement_id,
                "image_id": image_id,
                "batch_id": batch_id,
                "family_id": family_id,
                "vial_id": row["vial_id"],
                "recipe_id": recipe_id,
                "data_origin": "twin",
                "split_role": split_role,
                "split_version": split_version,
                "stage_id": row["stage_id"],
                "stage_ordinal": row["stage_ordinal"],
                "lab": list(stat.lab),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "stage_capture_measurements",
        "data_origin": "twin",
        "real_data_used": False,
        "family_id": family_id,
        "batch_id": batch_id,
        "recipe_id": recipe_id,
        "split_role": split_role,
        "split_version": split_version,
        "view_id": view_id,
        "source_manifest_sha256": source_sha,
        "image_locators": {
            "schema_version": 1,
            "data_origin": "twin",
            "images": locators,
        },
        "image_measurements": measurements,
        "stage_statistics": statistic_records,
        "stage_statistics_records": statistic_records,
        "model_inputs": model_inputs,
        "denominators": {"image_measurements": 9, "stage_statistics": 9},
        "capture_contract_version": "historical-isaac601-time-code-v1",
    }


def _bind_f2_capture(
    payload: Mapping[str, object],
    *,
    source_sha: str,
    source_base: Path,
    varied: bool = False,
) -> dict[str, object]:
    """Bind the event-based accepted Brev Isaac-6.2 F2 capture schema."""

    required_fields = _F2_VARIED_REQUIRED_FIELDS if varied else _F2_REQUIRED_FIELDS
    schema_version = 3 if varied else 2
    split_version = (
        _text(payload.get("split_version"), "split_version")
        if varied
        else _SPLIT_VERSION
    )
    split_roles = _F2_VARIED_SPLIT_ROLES if varied else _SPLIT_ROLES
    if set(payload) != required_fields:
        raise CaptureManifestError("manifest_schema_invalid")
    if (
        payload.get("schema_version") != schema_version
        or payload.get("artifact_type") != "f2_stage_capture_manifest"
        or payload.get("status") != "accepted"
    ):
        raise CaptureManifestError("manifest_schema_invalid")
    if payload.get("data_origin") != "twin":
        raise CaptureManifestError("data_origin_invalid")
    family_id = None if varied else _text(payload.get("family_id"), "family_id")
    batch_id = _text(payload.get("batch_id"), "batch_id")
    run_id = _text(payload.get("run_id"), "run_id")
    recipe_id = _text(payload.get("recipe_id"), "recipe_id")
    split_role = _text(payload.get("split_role"), "split_role")
    if split_role not in split_roles:
        raise CaptureManifestError("split_role_invalid", split_role)
    if (
        split_version not in _F2_VARIED_SPLIT_VERSIONS
        if varied
        else payload.get("split_version") != split_version
    ):
        raise CaptureManifestError("split_version_invalid")
    view_id = _text(payload.get("view_id"), "view_id")
    if view_id != "vial_fill":
        raise CaptureManifestError("wrong_view_id", view_id)
    runtime = _validate_f2_runtime(payload.get("runtime_binding"))
    receipt_binding = _bind_f2_runtime_receipt(
        payload.get("runtime_receipt"),
        runtime=runtime,
        base_dir=source_base,
        varied=varied,
    )
    events = _validate_f2_events(payload.get("stage_events"), varied=varied)
    event_by_id = {str(event["event_id"]): event for event in events}
    rows = payload.get("rows")
    expected_row_count = len(_VIALS) * len(events)
    if not isinstance(rows, list) or len(rows) != expected_row_count:
        raise CaptureManifestError("row_count_invalid")
    seen: set[tuple[str, str]] = set()
    locators: dict[str, dict[str, str]] = {}
    measurements: list[dict[str, object]] = []
    statistic_records: list[dict[str, object]] = []
    model_inputs: list[dict[str, object]] = []
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping) or set(raw) != _F2_ROW_KEYS:
            raise CaptureManifestError("row_malformed", f"row {row_number}")
        if raw.get("verified") is not True or raw.get("view_id") != view_id:
            raise CaptureManifestError("row_unverified", f"row {row_number}")
        vial_id = _text(raw.get("vial_id"), "vial_id")
        if vial_id not in _VIALS:
            raise CaptureManifestError("vial_id_invalid", vial_id)
        event_id = _text(raw.get("event_id"), "event_id")
        _text(raw.get("frame_id"), "frame_id")
        event = event_by_id.get(event_id)
        if event is None:
            raise CaptureManifestError("stage_event_mismatch", event_id)
        if any(
            raw.get(field) != event[field]
            for field in _F2_EVENT_KEYS - {"tank_id", "tank_order", "frame_id"}
        ):
            raise CaptureManifestError("stage_event_mismatch", f"row {row_number}")
        key = (vial_id, event_id)
        if key in seen:
            raise CaptureManifestError("duplicate_stage_row", repr(key))
        seen.add(key)
        image_id = f"{batch_id}-{vial_id}-{event_id}"
        rgb_path, rgb_sha = _bind_artifact(raw["rgb"], "rgb", source_base)
        mask_path, mask_sha = _bind_artifact(raw["liquid_mask"], "mask", source_base)
        _require_fillcam_resolution(rgb_path, "rgb")
        _require_decodable_png(mask_path, "mask")
        locator = {
            "rgb_path": str(rgb_path), "rgb_sha256": rgb_sha,
            "mask_path": str(mask_path), "mask_sha256": mask_sha,
        }
        measured = measure_image(image_id, locator)
        locators[image_id] = locator
        stage_event = dict(event)
        measurement = {
            **measured, "batch_id": batch_id,
            "run_id": run_id, "vial_id": vial_id, "recipe_id": recipe_id, "split_role": split_role,
            "split_version": split_version, "stage_id": event_id,
            "stage_ordinal": event["stage_ordinal"], "stage_event": stage_event,
            "view_id": view_id,
        }
        if family_id is not None:
            measurement["family_id"] = family_id
        measurements.append(measurement)
        stat_record = {
            "evidence_id": f"EVID-{image_id}", "measurement_id": f"MEAS-{image_id}",
            "image_id": image_id, "batch_id": batch_id,
            "run_id": run_id, "vial_id": vial_id, "recipe_id": recipe_id, "data_origin": "twin",
            "split_role": split_role, "split_version": split_version,
            "stage_id": event_id, "stage_ordinal": event["stage_ordinal"],
            "stage_event": stage_event, "lab": measured["lab_raw"],
            "validity": measured["validity"], "limit_codes": measured["limit_codes"],
        }
        if family_id is not None:
            stat_record["family_id"] = family_id
        statistic_records.append(stat_record)
        model_input_keys = [
            "evidence_id", "measurement_id", "image_id", "batch_id", "run_id",
            "vial_id", "recipe_id", "data_origin", "split_role", "split_version",
            "stage_id", "stage_ordinal", "stage_event", "lab", "validity", "limit_codes",
        ]
        if family_id is not None:
            model_input_keys.insert(5, "family_id")
        model_inputs.append({key: stat_record[key] for key in model_input_keys})
    expected = {(vial, str(event["event_id"])) for vial in _VIALS for event in events}
    if seen != expected:
        raise CaptureManifestError("missing_stage_row", repr(sorted(expected - seen)))
    public_runtime = {
        "runtime_id": runtime["runtime_id"],
        "runtime_receipt_sha256": receipt_binding["runtime_receipt_sha256"],
    }
    runtime_sha = hashlib.sha256(
        json.dumps(
            public_runtime, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    result = {
        "schema_version": schema_version,
        "artifact_type": "stage_capture_measurements",
        "capture_contract_version": (
            "f2-brev-isaac62-varied-event-v3"
            if varied
            else "f2-brev-isaac62-event-v2"
        ),
        "data_origin": "twin", "real_data_used": False,
        "batch_id": batch_id, "run_id": run_id, "recipe_id": recipe_id,
        "split_role": split_role, "split_version": split_version, "view_id": view_id,
        "runtime_binding": public_runtime, "runtime_binding_sha256": runtime_sha,
        "stage_events": events, "source_manifest_sha256": source_sha,
        "image_locators": {"schema_version": 1, "data_origin": "twin", "images": locators},
        "image_measurements": measurements,
        "stage_statistics": statistic_records,
        "stage_statistics_records": statistic_records,
        "model_inputs": model_inputs,
        "denominators": {
            "image_measurements": expected_row_count,
            "stage_statistics": expected_row_count,
        },
    }
    if family_id is not None:
        result["family_id"] = family_id
    return result


def _validate_f2_runtime(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _F2_RUNTIME_KEYS:
        raise CaptureManifestError("f2_runtime_binding_invalid")
    if (
        value.get("provider") != "brev"
        or value.get("runtime_name") != "NVIDIA Isaac Sim"
        or value.get("isaac_sim_version") != "6.2.0.0"
    ):
        raise CaptureManifestError("f2_runtime_binding_invalid")
    if (
        not isinstance(value.get("runtime_id"), str)
        or not value["runtime_id"]
        or not isinstance(value.get("source_sha256"), str)
        or _SHA256.fullmatch(str(value["source_sha256"])) is None
    ):
        raise CaptureManifestError("f2_runtime_binding_invalid")
    return dict(value)


def _bind_f2_runtime_receipt(
    value: object,
    *,
    runtime: Mapping[str, object],
    base_dir: Path,
    varied: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _F2_RECEIPT_LOCATOR_KEYS:
        raise CaptureManifestError("runtime_receipt_artifact_invalid")
    path_text = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        raise CaptureManifestError("runtime_receipt_path_invalid")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise CaptureManifestError("runtime_receipt_hash_invalid")
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve(strict=True)
        if resolved != path or not path.is_file():
            raise OSError("runtime receipt is not a direct regular file")
    except OSError as exc:
        raise CaptureManifestError("runtime_receipt_missing", str(path)) from exc
    try:
        receipt, snapshot = load_strict_json_snapshot(path)
    except (OSError, ValueError) as exc:
        raise CaptureManifestError("runtime_receipt_unreadable", str(path)) from exc
    actual = str(snapshot["sha256"])
    if actual != expected:
        raise CaptureManifestError("runtime_receipt_hash_mismatch", str(path))
    if varied:
        _validate_f2_varied_runtime_receipt(receipt, runtime=runtime)
    else:
        if not isinstance(receipt, Mapping) or set(receipt) != _F2_RECEIPT_KEYS:
            raise CaptureManifestError("runtime_receipt_invalid")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("artifact_type") != "f2_runtime_acceptance_receipt"
            or receipt.get("status") != "accepted"
            or receipt.get("provider") != "brev"
            or receipt.get("provider") != runtime.get("provider")
            or receipt.get("runtime_id") != runtime.get("runtime_id")
            or receipt.get("isaac_sim_version") != "6.2.0.0"
            or receipt.get("isaac_sim_version") != runtime.get("isaac_sim_version")
            or receipt.get("source_sha256") != runtime.get("source_sha256")
        ):
            raise CaptureManifestError("runtime_receipt_invalid")
        gates = receipt.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != _F2_ACCEPTANCE_GATES:
            raise CaptureManifestError("runtime_receipt_invalid")
        failed = sorted(key for key, passed in gates.items() if passed is not True)
        if failed:
            raise CaptureManifestError("runtime_receipt_gate_failed", repr(failed))
    return {
        "runtime_id": str(receipt["runtime_id"]),
        "runtime_receipt_sha256": actual,
    }


def _validate_f2_varied_runtime_receipt(
    receipt: object, *, runtime: Mapping[str, object]
) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != _F2_VARIED_RECEIPT_KEYS:
        raise CaptureManifestError("runtime_receipt_invalid")
    if (
        receipt.get("schema_version") != 3
        or receipt.get("artifact_type") != "f2_varied_runtime_acceptance_receipt"
        or receipt.get("status") != "accepted"
        or receipt.get("provider") != "brev"
        or receipt.get("provider") != runtime.get("provider")
        or receipt.get("runtime_id") != runtime.get("runtime_id")
        or receipt.get("isaac_sim_version") != "6.2.0.0"
        or receipt.get("isaac_sim_version") != runtime.get("isaac_sim_version")
        or receipt.get("source_sha256") != runtime.get("source_sha256")
    ):
        raise CaptureManifestError("runtime_receipt_invalid")
    hard_gates = receipt.get("hard_gates")
    if (
        not isinstance(hard_gates, Mapping)
        or set(hard_gates) != _F2_VARIED_HARD_GATES
        or any(value is not True for value in hard_gates.values())
    ):
        raise CaptureManifestError("runtime_receipt_gate_failed")
    diagnostic = receipt.get("particle_height_diagnostic")
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != _F2_PARTICLE_HEIGHT_KEYS:
        raise CaptureManifestError("runtime_receipt_invalid")
    warning_count = diagnostic.get("warning_count")
    warned_vial_count = diagnostic.get("warned_vial_count")
    vial_count = diagnostic.get("vial_count")
    vials = diagnostic.get("vials")
    if (
        diagnostic.get("hard_gate") is not False
        or diagnostic.get("physics_pass_claimed") is not False
        or isinstance(warning_count, bool)
        or not isinstance(warning_count, int)
        or warning_count < 0
        or isinstance(warned_vial_count, bool)
        or not isinstance(warned_vial_count, int)
        or warned_vial_count < 0
        or vial_count != 3
        or not isinstance(vials, list)
        or len(vials) != 3
    ):
        raise CaptureManifestError("runtime_receipt_invalid")
    normalized: list[tuple[str, bool, int]] = []
    for row in vials:
        if not isinstance(row, Mapping) or set(row) != _F2_PARTICLE_HEIGHT_VIAL_KEYS:
            raise CaptureManifestError("runtime_receipt_invalid")
        case_id = row.get("case_id")
        vial_id = row.get("vial_id")
        count = row.get("warning_count")
        passed = row.get("bounds_passed")
        if (
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            or vial_id not in _VIALS
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(passed, bool)
            or passed != (count == 0)
            or row.get("status") != ("within_bounds" if passed else "warning")
        ):
            raise CaptureManifestError("runtime_receipt_invalid")
        normalized.append((str(vial_id), passed, count))
    normalized.sort()
    if [row[0] for row in normalized] != sorted(_VIALS):
        raise CaptureManifestError("runtime_receipt_invalid")
    computed_warning_count = sum(row[2] for row in normalized)
    computed_warned_vials = sum(row[1] is False for row in normalized)
    bounds_passed = computed_warning_count == 0
    expected_status = "within_bounds" if bounds_passed else "warning"
    expected_claim = (
        "particle_height_within_bounds_diagnostic_only"
        if bounds_passed
        else "diagnostic_warning_not_a_physics_pass"
    )
    if (
        warning_count != computed_warning_count
        or warned_vial_count != computed_warned_vials
        or diagnostic.get("bounds_passed") is not bounds_passed
        or diagnostic.get("status") != expected_status
        or diagnostic.get("claim") != expected_claim
    ):
        raise CaptureManifestError("runtime_receipt_invalid")


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) >= 0.0
        and float(value) < float("inf")
    )


def _validate_f2_events(
    value: object, *, varied: bool = False
) -> list[dict[str, object]]:
    allowed_counts = {2, 3} if varied else {3}
    if not isinstance(value, list) or len(value) not in allowed_counts:
        raise CaptureManifestError("stage_events_invalid")
    events: list[dict[str, object]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _F2_EVENT_KEYS:
            raise CaptureManifestError("stage_events_invalid")
        event = dict(raw)
        ordinal = event.get("stage_ordinal")
        if (
            not isinstance(event.get("event_id"), str) or not event["event_id"]
            or not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal not in range(1, len(value) + 1)
            or event.get("tank_id") not in {"t100", "t200", "t400"}
            or event.get("tank_order") != ordinal
            or (event.get("physics_step") is None and event.get("frame_id") is None)
            or not _finite_nonnegative(event.get("event_relative_timestamp_s"))
            or not _finite_nonnegative(event.get("final_observation_offset_s"))
        ):
            raise CaptureManifestError("stage_events_invalid")
        if event.get("physics_step") is not None and (
            isinstance(event["physics_step"], bool) or not isinstance(event["physics_step"], int)
            or int(event["physics_step"]) < 0
        ):
            raise CaptureManifestError("stage_events_invalid")
        if event.get("frame_id") is not None and (
            not isinstance(event["frame_id"], str) or not event["frame_id"]
        ):
            raise CaptureManifestError("stage_events_invalid")
        events.append(event)
    ordered = sorted(events, key=lambda event: int(event["stage_ordinal"]))
    expected_ordinals = list(range(1, len(ordered) + 1))
    if (
        [event["stage_ordinal"] for event in ordered] != expected_ordinals
        or len({event["event_id"] for event in ordered}) != len(ordered)
        or len({event["tank_id"] for event in ordered}) != len(ordered)
    ):
        raise CaptureManifestError("stage_events_invalid")
    return ordered


def _load_manifest(manifest: object, base_dir: str | Path | None) -> tuple[object, str, Path]:
    if isinstance(manifest, Mapping):
        payload = manifest
        source = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return payload, hashlib.sha256(source).hexdigest(), Path(base_dir or Path.cwd()).resolve()
    if not isinstance(manifest, (str, Path)):
        raise CaptureManifestError("manifest_malformed")
    path = Path(manifest)
    try:
        payload, snapshot = load_strict_json_snapshot(path)
    except ValueError as exc:
        raise CaptureManifestError("manifest_missing", str(path)) from exc
    return payload, str(snapshot["sha256"]), Path(base_dir or path.parent).resolve()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CaptureManifestError(f"{field}_invalid")
    return value


def _validate_stage_slots(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 3:
        raise CaptureManifestError("stage_slots_invalid")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _STAGE_KEYS:
            raise CaptureManifestError("stage_slots_invalid")
        stage = dict(item)
        if not isinstance(stage["stage_id"], str) or not stage["stage_id"]:
            raise CaptureManifestError("stage_slots_invalid")
        if (
            isinstance(stage["stage_ordinal"], bool)
            or not isinstance(stage["stage_ordinal"], int)
            or stage["stage_ordinal"] not in {1, 2, 3}
            or isinstance(stage["time_code"], bool)
            or not isinstance(stage["time_code"], int)
            or stage["time_code"] <= 0
        ):
            raise CaptureManifestError("stage_slots_invalid")
        result.append(stage)
    if {int(item["stage_ordinal"]) for item in result} != {1, 2, 3}:
        raise CaptureManifestError("stage_slots_invalid")
    if len({str(item["stage_id"]) for item in result}) != 3:
        raise CaptureManifestError("stage_slots_invalid")
    return sorted(result, key=lambda item: int(item["stage_ordinal"]))


def _bind_artifact(value: object, role: str, base_dir: Path) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_KEYS:
        raise CaptureManifestError(f"{role}_artifact_invalid")
    path_text = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        raise CaptureManifestError(f"{role}_path_invalid")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise CaptureManifestError(f"{role}_hash_invalid")
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    if path.suffix != ".png":
        raise CaptureManifestError(f"{role}_not_png", str(path))
    try:
        resolved = path.resolve(strict=True)
        if resolved != path or not path.is_file():
            raise OSError("not a direct regular file")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CaptureManifestError(f"{role}_missing", str(path)) from exc
    if actual != expected:
        raise CaptureManifestError(f"{role}_hash_mismatch", str(path))
    return path, actual


def _require_fillcam_resolution(path: Path, role: str) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            if image.size != (640, 480):
                raise CaptureManifestError(f"{role}_resolution_invalid", str(image.size))
    except (OSError, UnidentifiedImageError) as exc:
        raise CaptureManifestError(f"{role}_png_decode_failed", str(path)) from exc


def _require_decodable_png(path: Path, role: str) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise CaptureManifestError(f"{role}_not_png", str(path))
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise CaptureManifestError(f"{role}_png_decode_failed", str(path)) from exc


# Short public spelling for callers that do not need to mention stage scope.
bind_capture_manifest = bind_stage_capture_manifest


def build_measured_evidence(
    model_input_plan: object,
    captures: object,
) -> dict[str, object]:
    """Assemble bound captures through the canonical vision evidence boundary."""

    if not isinstance(model_input_plan, Mapping):
        raise CaptureManifestError("model_input_plan_malformed")
    if (
        model_input_plan.get("artifact_type") != "model_input_plan"
        or model_input_plan.get("real_data_used") is not False
    ):
        raise CaptureManifestError("model_input_plan_invalid")
    decisions = model_input_plan.get("decision_records")
    if not isinstance(decisions, list) or not decisions:
        raise CaptureManifestError("model_input_decisions_missing")
    if not isinstance(captures, list) or not captures:
        raise CaptureManifestError("capture_batches_missing")

    referenced = _referenced_image_ids(decisions)
    expected_provenance = _plan_image_provenance(decisions)
    batch_ids: set[str] = set()
    locator_images: dict[str, object] = {}
    for index, capture in enumerate(captures):
        bound = _bound_capture(capture)
        if bound.get("artifact_type") != "stage_capture_measurements":
            raise CaptureManifestError("capture_result_invalid", str(index))
        batch_id = _text(bound.get("batch_id"), "capture batch_id")
        if batch_id in batch_ids:
            raise CaptureManifestError("duplicate_capture_batch", batch_id)
        batch_ids.add(batch_id)
        if bound.get("split_version") != _SPLIT_VERSION:
            raise CaptureManifestError("capture_provenance_mismatch", batch_id)
        is_f2 = bound.get("capture_contract_version") == "f2-brev-isaac62-event-v2"
        if is_f2 and bound.get("runtime_binding_sha256") != model_input_plan.get(
            "runtime_binding_sha256"
        ):
            raise CaptureManifestError("f2_runtime_binding_mismatch", batch_id)
        locators = bound.get("image_locators")
        if (
            not isinstance(locators, Mapping)
            or locators.get("schema_version") != 1
            or locators.get("data_origin") != "twin"
            or not isinstance(locators.get("images"), Mapping)
        ):
            raise CaptureManifestError("capture_locators_invalid", batch_id)
        images = locators["images"]
        assert isinstance(images, Mapping)
        measurements = bound.get("image_measurements")
        if not isinstance(measurements, list) or len(measurements) != len(images):
            raise CaptureManifestError("capture_measurements_invalid", batch_id)
        measured_ids: set[str] = set()
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                raise CaptureManifestError("capture_measurement_invalid", batch_id)
            image_id = measurement.get("image_id")
            if not isinstance(image_id, str) or image_id in measured_ids:
                raise CaptureManifestError("duplicate_capture_image", str(image_id))
            measured_ids.add(image_id)
            expected = expected_provenance.get(image_id)
            if (
                measurement.get("batch_id") != batch_id
                or measurement.get("split_version") != _SPLIT_VERSION
                or (not is_f2 and measurement.get("validity") != "valid")
                or (
                    expected is not None
                    and any(
                        measurement.get(field) != expected[field]
                        for field in expected
                    )
                )
            ):
                raise CaptureManifestError("capture_measurement_provenance_mismatch", image_id)
        if measured_ids != set(images):
            raise CaptureManifestError("capture_measurement_ids_mismatch", batch_id)
        for image_id, locator in images.items():
            if image_id in locator_images:
                raise CaptureManifestError("duplicate_capture_image", str(image_id))
            locator_images[image_id] = locator

    if set(locator_images) != referenced:
        missing = sorted(referenced.difference(locator_images))
        extra = sorted(set(locator_images).difference(referenced))
        raise CaptureManifestError(
            "capture_image_ids_mismatch", f"missing={missing!r}, extra={extra!r}"
        )
    return evaluate_model_plan(
        model_input_plan,
        {"schema_version": 1, "data_origin": "twin", "images": locator_images},
    )


def _bound_capture(value: object) -> Mapping[str, object]:
    if isinstance(value, (str, Path)):
        return bind_stage_capture_manifest(value)
    if isinstance(value, Mapping):
        artifact_type = value.get("artifact_type")
        if artifact_type in {"twin_stage_capture_manifest", "f2_stage_capture_manifest"}:
            return bind_stage_capture_manifest(value)
        if artifact_type == "stage_capture_measurements":
            return value
    raise CaptureManifestError("capture_input_invalid")


def _plan_image_provenance(decisions: list[object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise CaptureManifestError("model_input_decision_invalid")
        family_id = _text(decision.get("family_id"), "model family_id")
        recipe_id = _text(decision.get("current_sim_recipe_id"), "model recipe_id")
        split_role = _text(decision.get("partition"), "model partition")
        if split_role not in _SPLIT_ROLES:
            raise CaptureManifestError("model_partition_invalid", split_role)
        batch_id = _text(decision.get("batch_id"), "model batch_id")
        decision_run_id = decision.get("run_id")
        if decision_run_id is not None:
            decision_run_id = _text(decision_run_id, "model run_id")
        vial_id = _text(decision.get("vial_id"), "model vial_id")
        stages = decision.get("stages")
        if not isinstance(stages, list):
            raise CaptureManifestError("model_input_stages_invalid")
        event_based = any(
            isinstance(stage, Mapping) and "stage_event" in stage for stage in stages
        )
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise CaptureManifestError("model_input_stage_invalid")
            stage_id = _text(stage.get("stage_id"), "model stage_id")
            ordinal = stage.get("stage_ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise CaptureManifestError("model_stage_ordinal_invalid")
            context = {
                "family_id": family_id,
                "recipe_id": recipe_id,
                "split_role": split_role,
                "batch_id": batch_id,
                "vial_id": vial_id,
                "stage_id": stage_id,
                "stage_ordinal": ordinal,
            }
            if event_based and decision_run_id is not None:
                context["run_id"] = decision_run_id
            image_contexts = [(stage.get("target_image_id"), context)]
            for case in stage.get("reference_cases", []):
                if not isinstance(case, Mapping) or not isinstance(
                    case.get("candidate_peers"), list
                ):
                    raise CaptureManifestError("model_input_reference_case_invalid")
                for peer in case["candidate_peers"]:
                    if not isinstance(peer, Mapping):
                        raise CaptureManifestError("model_input_peer_invalid")
                    peer_image_id = peer.get("peer_image_id")
                    peer_batch_id = _text(peer.get("batch_id"), "peer batch_id")
                    peer_context = dict(context)
                    peer_context["batch_id"] = peer_batch_id
                    if event_based and peer.get("run_id") is not None:
                        peer_context["run_id"] = _text(peer.get("run_id"), "peer run_id")
                    peer_context.pop("vial_id", None)
                    if peer.get("vial_id") is not None:
                        peer_context["vial_id"] = _text(peer.get("vial_id"), "peer vial_id")
                    for peer_field, context_field, role in (
                        ("current_sim_recipe_id", "recipe_id", "peer recipe_id"),
                        ("partition", "split_role", "peer partition"),
                        ("family_id", "family_id", "peer family_id"),
                    ):
                        if peer.get(peer_field) is not None:
                            declared = _text(peer.get(peer_field), role)
                            if declared != context[context_field]:
                                raise CaptureManifestError(
                                    "model_input_peer_provenance_mismatch",
                                    str(peer_image_id),
                                )
                    image_contexts.append((peer_image_id, peer_context))
            for image_id, image_context in image_contexts:
                if not isinstance(image_id, str) or not image_id:
                    raise CaptureManifestError("model_input_image_references_invalid")
                prior = result.get(image_id)
                if prior is not None and prior != image_context:
                    raise CaptureManifestError("model_input_image_provenance_mismatch")
                result[image_id] = image_context
    return result


def _referenced_image_ids(decisions: list[object]) -> set[str]:
    referenced: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("stages"), list):
            raise CaptureManifestError("model_input_stages_invalid")
        for stage in decision["stages"]:
            if not isinstance(stage, Mapping):
                raise CaptureManifestError("model_input_stage_invalid")
            target = stage.get("target_image_id")
            cases = stage.get("reference_cases")
            if not isinstance(target, str) or not target or not isinstance(cases, list):
                raise CaptureManifestError("model_input_image_references_invalid")
            referenced.add(target)
            for case in cases:
                if not isinstance(case, Mapping) or not isinstance(
                    case.get("candidate_peers"), list
                ):
                    raise CaptureManifestError("model_input_reference_case_invalid")
                for peer in case["candidate_peers"]:
                    if not isinstance(peer, Mapping) or not isinstance(
                        peer.get("peer_image_id"), str
                    ):
                        raise CaptureManifestError("model_input_peer_invalid")
                    referenced.add(peer["peer_image_id"])
    return referenced


assemble_measured_evidence = build_measured_evidence


def stage_statistics_from_result(result: object) -> list[StageStatistic]:
    """Convert serializable adapter records for the existing decision boundary."""

    if (
        not isinstance(result, Mapping)
        or result.get("artifact_type") != "stage_capture_measurements"
    ):
        raise ValueError("capture result is not measured evidence")
    rows = result.get("stage_statistics")
    if not isinstance(rows, list):
        raise ValueError("capture result stage statistics are missing")
    required = {
        "evidence_id", "measurement_id", "image_id", "batch_id", "vial_id",
        "family_id", "recipe_id", "data_origin", "split_role", "split_version", "stage_id",
        "stage_ordinal", "time_code", "lab", "validity", "limit_codes",
    }
    converted: list[StageStatistic] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("capture stage statistic fields are malformed")
        lab = row["lab"]
        if not isinstance(lab, list) or len(lab) != 3:
            raise ValueError("capture stage statistic Lab is malformed")
        converted.append(
            StageStatistic(
                evidence_id=row["evidence_id"], measurement_id=row["measurement_id"],
                image_id=row["image_id"], batch_id=row["batch_id"], vial_id=row["vial_id"],
                recipe_id=row["recipe_id"], data_origin=row["data_origin"],
                split_role=row["split_role"], split_version=row["split_version"],
                stage_id=row["stage_id"], stage_ordinal=row["stage_ordinal"],
                lab=tuple(float(value) for value in lab), validity=row["validity"],
                limit_codes=tuple(row["limit_codes"]),
            )
        )
    return converted
