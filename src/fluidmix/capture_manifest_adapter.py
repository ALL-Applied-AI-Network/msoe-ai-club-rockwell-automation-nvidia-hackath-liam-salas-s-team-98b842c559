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


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
_FORBIDDEN_KEYS = {
    "label",
    "labels",
    "latent_parameters",
    "semantic_aov",
    "oracle",
    "oracle_fields",
    "expected_disposition",
    "render_seed",
    "synthetic_profile_id",
    "appearance_blend_alpha",
    "appearance_blend_alpha_by_vial",
    "synthetic_onset_id",
    "clean_stage_linear_rgb",
    "observed_stage_linear_rgb",
    "profile_id",
    "residual_particle_count",
    "residual_particle_count_by_vial",
    "residual_particle_plan",
    "residual_source_component_counts",
    "synthetic_oracle_record",
    "label_source",
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
    _reject_forbidden(payload)
    if not isinstance(payload, Mapping):
        raise CaptureManifestError("manifest_malformed", "manifest must be an object")
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
    }


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


def _reject_forbidden(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS:
                raise CaptureManifestError("forbidden_oracle_field", key)
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


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
                or measurement.get("validity") != "valid"
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
        if artifact_type == "twin_stage_capture_manifest":
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
        vial_id = _text(decision.get("vial_id"), "model vial_id")
        stages = decision.get("stages")
        if not isinstance(stages, list):
            raise CaptureManifestError("model_input_stages_invalid")
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
