"""Fail-closed package contracts for whole-run synthetic carryover videos."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .deploy_input_policy import validate_deploy_input_fields


INDEX_FIELDS = (
    "run_id",
    "run_number",
    "vial_ordinal",
    "case_id",
    "split",
    "recipe_id",
    "previous_recipe_id",
    "prescribed_residue_volume_ml",
    "residue_source_tank_id",
    "residue_composition_t200_t400_t100",
    "disposition",
    "palette_t200_rgb",
    "palette_t400_rgb",
    "palette_t100_rgb",
    "palette_sha256",
    "height_warning_count",
    "height_warnings_json",
    "per_vial_video_path",
    "run_video_path",
    "stills_path",
    "masks_path",
    "semantic_masks_path",
    "receipt_path",
    "transport_receipt_path",
    "metadata_excluded_from_model_inputs",
    "simulator_metadata_excluded_from_model_inputs",
)

MODEL_INPUT_FORBIDDEN_KEY_FRAGMENTS = (
    "label",
    "residual",
    "palette",
    "particle",
    "disposition",
    "warning",
    "oracle",
    "keep",
    "discard",
)

MODEL_INPUT_SOURCE_FIELDS = (
    "case_id",
    "batch_id",
    "run_id",
    "vial_position",
    "data_origin",
    "split_role",
    "split_version",
    "previous_recipe_id",
    "current_recipe_id",
    "previous_programmed_ml",
    "current_programmed_ml",
    "programmed_dispense_order",
    "stage_image_evidence_ids",
    "quantitative_evidence_ids",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rgb_hex(value: object) -> str:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(channel, bool) or not isinstance(channel, (int, float)) for channel in value)
        or any(float(channel) < 0.0 or float(channel) > 1.0 for channel in value)
    ):
        raise ValueError("palette RGB must contain three values in [0, 1]")
    return "".join(f"{round(float(channel) * 255):02X}" for channel in value)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _run_number(run_id: str, split: str) -> str:
    prefixes = (f"RUN-{split.upper()}-", f"RUN-V4-{split.upper()}-")
    prefix = next((value for value in prefixes if run_id.startswith(value)), None)
    if prefix is None:
        raise ValueError(f"run {run_id} does not match split {split}")
    suffix = run_id.removeprefix(prefix)
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError(f"run {run_id} lacks a two-digit split ordinal")
    return suffix


def _run_video_path(run: Mapping[str, object], recipe_sequence: Iterable[str]) -> str:
    run_id = str(run["run_id"])
    split = str(run["split_role"])
    run_number = _run_number(run_id, split)
    recipe_ids = _ordered_unique(str(value) for value in recipe_sequence)
    if not recipe_ids:
        raise ValueError(f"run {run_id} has no scheduled recipes")
    palette = run.get("palette")
    if not isinstance(palette, dict):
        raise ValueError(f"run {run_id} lacks a palette")
    transmission = palette.get("transmission_1cm")
    if not isinstance(transmission, dict):
        raise ValueError(f"run {run_id} lacks transmission colors")
    filename = (
        f"{split}-run{run_number}-{'-'.join(recipe_ids)}-"
        f"T200{_rgb_hex(transmission.get('t200'))}-"
        f"T400{_rgb_hex(transmission.get('t400'))}.mp4"
    )
    return f"videos/runs/{filename}"


def _residue_composition(schedule: Mapping[str, object]) -> tuple[str, list[float]]:
    pours = schedule.get("pours")
    if not isinstance(pours, list) or not pours or not isinstance(pours[0], dict):
        raise ValueError("execution schedule lacks a first pour")
    first_pour = pours[0]
    segments = first_pour.get("emission_segments")
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        raise ValueError("execution schedule lacks a leading-residual segment")
    leading = segments[0]
    if leading.get("segment_role") != "leading_residual":
        raise ValueError("first emission segment is not leading residual")
    source = leading.get("source_tank_id")
    composition = leading.get("one_hot_composition")
    if not isinstance(source, str):
        raise ValueError("leading residual source tank is invalid")
    if (
        not isinstance(composition, list)
        or len(composition) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in composition)
    ):
        raise ValueError("leading residual composition is invalid")
    return source, [float(value) for value in composition]


def build_dataset_index_rows(
    artifacts: Mapping[str, object],
    *,
    height_warnings_by_case: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> list[dict[str, str]]:
    """Build the audit/handoff index; its latent fields are never model input."""

    render = artifacts.get("render-schedule-plan.json")
    labels = artifacts.get("label-provenance.json")
    if not isinstance(render, dict) or not isinstance(labels, dict):
        raise ValueError("render and label plans are required")
    runs = render.get("runs")
    schedules = render.get("execution_schedules")
    label_rows = labels.get("vial_labels")
    if not all(isinstance(value, list) for value in (runs, schedules, label_rows)):
        raise ValueError("render and label plans are malformed")
    warnings_by_case = height_warnings_by_case or {}
    run_by_id = {
        str(run["run_id"]): run
        for run in runs
        if isinstance(run, dict) and isinstance(run.get("run_id"), str)
    }
    schedule_by_case = {
        str(schedule["case_binding"]["case_id"]): schedule
        for schedule in schedules
        if isinstance(schedule, dict)
        and isinstance(schedule.get("case_binding"), dict)
        and isinstance(schedule["case_binding"].get("case_id"), str)
    }
    label_by_case = {
        str(row["case_id"]): row
        for row in label_rows
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    if not (set(schedule_by_case) == set(label_by_case)):
        raise ValueError("render schedules and labels have different case sets")
    if set(warnings_by_case) - set(schedule_by_case):
        raise ValueError("height warnings reference an unknown case")
    recipes_by_run: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for schedule in schedules:
        binding = schedule.get("case_binding") if isinstance(schedule, dict) else None
        if not isinstance(binding, dict):
            raise ValueError("execution schedule lacks case binding")
        recipes_by_run[str(binding["run_id"])].append(
            (int(binding["vial_ordinal"]), str(binding["current_recipe_id"]))
        )

    result: list[dict[str, str]] = []
    for schedule in schedules:
        if not isinstance(schedule, dict):
            raise ValueError("execution schedule must be an object")
        binding = schedule.get("case_binding")
        if not isinstance(binding, dict):
            raise ValueError("execution schedule lacks case binding")
        case_id = str(binding["case_id"])
        run_id = str(binding["run_id"])
        split = str(binding["split_role"])
        ordinal = int(binding["vial_ordinal"])
        run = run_by_id.get(run_id)
        label = label_by_case[case_id]
        if not isinstance(run, dict):
            raise ValueError(f"case {case_id} references an unknown run")
        if run.get("split_role") != split or label.get("split_role") != split:
            raise ValueError(f"case {case_id} crosses split provenance")
        expected = str(label.get("expected_disposition", "")).upper()
        if expected not in {"KEEP", "DISCARD"}:
            raise ValueError(f"case {case_id} has an invalid disposition")
        residual_volume = float(label["prescribed_leading_residual_volume_ml"])
        if expected != ("KEEP" if residual_volume == 0.0 else "DISCARD"):
            raise ValueError(f"case {case_id} violates the synthetic label rule")
        palette = run.get("palette")
        if not isinstance(palette, dict) or not isinstance(
            palette.get("transmission_1cm"), dict
        ):
            raise ValueError(f"run {run_id} has invalid palette provenance")
        transmission = palette["transmission_1cm"]
        residue_source, residue_composition = _residue_composition(schedule)
        warnings = [dict(row) for row in warnings_by_case.get(case_id, [])]
        run_number = _run_number(run_id, split)
        per_vial_name = (
            f"{split}-run{run_number}-vial{ordinal:02d}-{expected}-{case_id}.mp4"
        )
        result.append(
            {
                "run_id": run_id,
                "run_number": run_number,
                "vial_ordinal": str(ordinal),
                "case_id": case_id,
                "split": split,
                "recipe_id": str(binding["current_recipe_id"]),
                "previous_recipe_id": str(binding["previous_recipe_id"]),
                "prescribed_residue_volume_ml": format(residual_volume, ".15g"),
                "residue_source_tank_id": residue_source,
                "residue_composition_t200_t400_t100": _canonical_json(
                    residue_composition
                ),
                "disposition": expected,
                "palette_t200_rgb": _canonical_json(transmission["t200"]),
                "palette_t400_rgb": _canonical_json(transmission["t400"]),
                "palette_t100_rgb": _canonical_json(transmission["t100"]),
                "palette_sha256": str(palette["palette_sha256"]),
                "height_warning_count": str(len(warnings)),
                "height_warnings_json": _canonical_json(warnings),
                "per_vial_video_path": f"videos/vials/{per_vial_name}",
                "run_video_path": _run_video_path(
                    run,
                    (
                        recipe_id
                        for _, recipe_id in sorted(recipes_by_run.get(run_id, []))
                    ),
                ),
                "stills_path": f"cases/{case_id}/frames",
                "masks_path": f"cases/{case_id}/liquid",
                "semantic_masks_path": f"cases/{case_id}/semantic",
                "receipt_path": f"cases/{case_id}/receipt.json",
                "transport_receipt_path": (
                    f"videos/vials/{Path(per_vial_name).stem}.transport.json"
                ),
                "metadata_excluded_from_model_inputs": "true",
                "simulator_metadata_excluded_from_model_inputs": "true",
            }
        )
    validate_dataset_index(result)
    return result


def build_model_input_rows(
    artifacts: Mapping[str, object], *, index_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    """Return label-free rows whose paths are opaque to disposition."""

    model = artifacts.get("model-input-plan.json")
    if not isinstance(model, dict) or not isinstance(model.get("vial_records"), list):
        raise ValueError("model input plan is required")
    indexed_cases = {str(row.get("case_id")) for row in index_rows}
    rows: list[dict[str, object]] = []
    for source in model["vial_records"]:
        if not isinstance(source, dict) or not isinstance(source.get("case_id"), str):
            raise ValueError("model input plan has an invalid row")
        validate_deploy_input_fields(source, require_allowlist=True)
        unknown_fields = set(source) - set(MODEL_INPUT_SOURCE_FIELDS)
        missing_fields = set(MODEL_INPUT_SOURCE_FIELDS) - set(source)
        if unknown_fields:
            raise ValueError(
                f"forbidden_deploy_input_field: {sorted(unknown_fields)[0]}"
            )
        if missing_fields:
            raise ValueError(
                f"model input plan row lacks required field: {sorted(missing_fields)[0]}"
            )
        case_id = source["case_id"]
        if case_id not in indexed_cases:
            raise ValueError("model input plan and dataset index differ")
        row = {key: deepcopy(source[key]) for key in MODEL_INPUT_SOURCE_FIELDS}
        row["video_path"] = f"model-inputs/videos/{case_id}.mp4"
        rows.append(row)
    if {str(row["case_id"]) for row in rows} != indexed_cases:
        raise ValueError("model input plan and dataset index differ")
    for row in rows:
        video_path = str(row["video_path"])
        if "KEEP" in video_path or "DISCARD" in video_path:
            raise ValueError("model video path exposes disposition")
    return rows


_TRANSPORT_POINTER_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "run_id",
    "case_id",
    "run_receipt",
    "entry_segment",
    "outgoing_segment",
    "qualified_fill_receipt_changed",
}
_TRANSPORT_SEGMENT_FIELDS = {
    "render_dir",
    "receipt_path",
    "receipt_sha256",
    "mode",
}


def _resolved_bound_path(raw: object, *, relative_to: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _validate_transport_segment_record(
    value: object, *, expected_mode: str, relative_to: Path
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _TRANSPORT_SEGMENT_FIELDS:
        raise ValueError("transport segment record differs from its frozen contract")
    if value.get("mode") != expected_mode:
        raise ValueError(f"transport segment mode is not {expected_mode}")
    render_dir = _resolved_bound_path(
        value.get("render_dir"), relative_to=relative_to, label="transport render"
    )
    receipt_path = _resolved_bound_path(
        value.get("receipt_path"),
        relative_to=relative_to,
        label="transport receipt",
    )
    if receipt_path.parent != render_dir or not receipt_path.is_file():
        raise ValueError("transport segment receipt location is invalid")
    if value.get("receipt_sha256") != _sha256(receipt_path):
        raise ValueError("transport segment receipt hash differs")
    receipt = _read_object(receipt_path, "transport render receipt")
    checker = receipt.get("checker_evaluation")
    visual_review = receipt.get("pilot_visual_review")
    gates = (
        checker.get("automatic_acceptance_gates")
        if isinstance(checker, dict)
        else None
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "fluidmix_transport_stage_render_receipt_v1"
        or receipt.get("status") != "candidate"
        or receipt.get("mode") != expected_mode
        or receipt.get("state_indices") != list(range(13))
        or receipt.get("rendered_states_byte_unique") is not True
        or not isinstance(checker, dict)
        or checker.get("schema_version") != 1
        or checker.get("artifact_type") != "fluidmix_transport_stage_checker_v1"
        or checker.get("status") != "pass"
        or checker.get("pilot") is not False
        or checker.get("manual_visual_review_required") is not False
        or checker.get("pilot_visual_review_status") != "accepted"
        or not isinstance(gates, dict)
        or not gates
        or any(value is not True for value in gates.values())
        or gates.get("pilot_visual_review_accepted") is not True
        or not isinstance(visual_review, dict)
        or visual_review.get("status") != "accepted"
    ):
        raise ValueError("transport segment is not a full accepted candidate")
    return {key: str(value[key]) for key in sorted(_TRANSPORT_SEGMENT_FIELDS)}


def validate_transport_pointer(
    pointer_path: str | Path,
    *,
    expected_run_id: str,
    expected_case_ids: Sequence[str],
    expected_ordinal: int,
    expected_run_video: str | Path,
) -> dict[str, object]:
    """Validate one vial pointer through the immutable accepted run receipt."""

    path = Path(pointer_path).resolve()
    pointer = _read_object(path, "transport vial pointer")
    case_ids = [str(case_id) for case_id in expected_case_ids]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("expected transport cases are invalid")
    if type(expected_ordinal) is not int or not 1 <= expected_ordinal <= len(case_ids):
        raise ValueError("expected transport ordinal is invalid")
    if set(pointer) != _TRANSPORT_POINTER_FIELDS:
        raise ValueError("transport vial pointer differs from its frozen contract")
    if (
        pointer.get("schema_version") != 2
        or pointer.get("artifact_type") != "rendered_transport_vial_pointer_v2"
        or pointer.get("status") != "pass"
        or pointer.get("run_id") != expected_run_id
        or pointer.get("case_id") != case_ids[expected_ordinal - 1]
        or pointer.get("qualified_fill_receipt_changed") is not False
    ):
        raise ValueError("transport vial pointer identity or boundary is invalid")

    run_binding = pointer.get("run_receipt")
    if not isinstance(run_binding, dict) or set(run_binding) != {"path", "sha256"}:
        raise ValueError("transport pointer run receipt binding is invalid")
    run_receipt_path = _resolved_bound_path(
        run_binding.get("path"), relative_to=path.parent, label="run receipt"
    )
    expected_video_path = Path(expected_run_video).resolve()
    if run_receipt_path != expected_video_path.with_suffix(".run.json"):
        raise ValueError("transport pointer binds the wrong run receipt path")
    if (
        not run_receipt_path.is_file()
        or run_binding.get("sha256") != _sha256(run_receipt_path)
    ):
        raise ValueError("transport pointer run receipt hash differs")
    run_receipt = _read_object(run_receipt_path, "transport run receipt")
    segments = run_receipt.get("transport_segments")
    if (
        run_receipt.get("schema_version") != 2
        or run_receipt.get("artifact_type") != "rendered_transport_run_sequence_v2"
        or run_receipt.get("status") != "pass"
        or run_receipt.get("run_id") != expected_run_id
        or run_receipt.get("case_ids") != case_ids
        or run_receipt.get("qualified_fill_receipts_changed") is not False
        or run_receipt.get("camera_and_lighting_changed") is not False
        or not isinstance(segments, list)
        or len(segments) != len(case_ids) + 1
        or run_receipt.get("transport_segment_count") != len(segments)
    ):
        raise ValueError("transport run receipt identity or boundary is invalid")
    output_video = _resolved_bound_path(
        run_receipt.get("output_video"),
        relative_to=run_receipt_path.parent,
        label="transport run video",
    )
    if (
        output_video != expected_video_path
        or not output_video.is_file()
        or run_receipt.get("output_video_sha256") != _sha256(output_video)
    ):
        raise ValueError("transport run video hash differs from its receipt")

    modes = ["entry", *(["between"] * (len(case_ids) - 1)), "exit"]
    encoded = run_receipt.get("encoded_transition_evidence")
    encoded_fields = {
        "segment_index",
        "mode",
        "start_frame_index",
        "stop_frame_index",
        "frame_count",
        "decoded_bgr_sha256",
        "byte_unique",
    }
    if (
        run_receipt.get("encoded_transition_sequences_byte_unique") is not True
        or not isinstance(encoded, list)
        or len(encoded) != len(modes)
    ):
        raise ValueError("transport run receipt lacks encoded transition evidence")
    for segment_index, (row, mode) in enumerate(zip(encoded, modes, strict=True)):
        hashes = row.get("decoded_bgr_sha256") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != encoded_fields
            or row.get("segment_index") != segment_index
            or row.get("mode") != mode
            or type(row.get("start_frame_index")) is not int
            or type(row.get("stop_frame_index")) is not int
            or row.get("stop_frame_index") - row.get("start_frame_index") != 13
            or row.get("frame_count") != 13
            or row.get("byte_unique") is not True
            or not isinstance(hashes, list)
            or len(hashes) != 13
            or len(set(hashes)) != 13
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
        ):
            raise ValueError("transport encoded transition evidence is malformed")
    seam_rows = run_receipt.get("encoded_seam_evidence")
    seam_fields = {
        "segment_index",
        "mode",
        "start_seam_mad",
        "start_seam_threshold_p99",
        "start_seam_passed",
        "end_seam_mad",
        "end_seam_threshold_p99",
        "end_seam_passed",
    }
    if (
        run_receipt.get("encoded_transition_seams_passed") is not True
        or not isinstance(seam_rows, list)
        or len(seam_rows) != len(modes)
    ):
        raise ValueError("transport run receipt lacks encoded seam evidence")
    for segment_index, (row, mode) in enumerate(zip(seam_rows, modes, strict=True)):
        if (
            not isinstance(row, Mapping)
            or set(row) != seam_fields
            or row.get("segment_index") != segment_index
            or row.get("mode") != mode
            or row.get("start_seam_passed") is not True
            or row.get("end_seam_passed") is not True
        ):
            raise ValueError("transport encoded seam evidence is malformed")
        for side, required in (
            ("start", mode != "entry"),
            ("end", mode != "exit"),
        ):
            mad = row.get(f"{side}_seam_mad")
            threshold = row.get(f"{side}_seam_threshold_p99")
            if required:
                if (
                    isinstance(mad, bool)
                    or not isinstance(mad, (int, float))
                    or isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(float(mad))
                    or not math.isfinite(float(threshold))
                    or float(mad) > float(threshold)
                ):
                    raise ValueError("transport encoded seam evidence is malformed")
            elif mad is not None or threshold is not None:
                raise ValueError("transport encoded seam evidence is malformed")
    for segment, mode in zip(segments, modes, strict=True):
        _validate_transport_segment_record(
            segment, expected_mode=mode, relative_to=run_receipt_path.parent
        )
    expected_entry = segments[0] if expected_ordinal == 1 else None
    if pointer.get("entry_segment") != expected_entry:
        raise ValueError("transport pointer entry segment differs from the run receipt")
    expected_outgoing = segments[expected_ordinal]
    if pointer.get("outgoing_segment") != expected_outgoing:
        raise ValueError("transport pointer outgoing segment differs from the run receipt")
    return {
        "status": "pass",
        "run_id": expected_run_id,
        "case_id": case_ids[expected_ordinal - 1],
        "outgoing_mode": modes[expected_ordinal],
        "run_receipt_sha256": _sha256(run_receipt_path),
    }


def validate_dataset_index(
    rows: Sequence[Mapping[str, str]], *, package_root: str | Path | None = None
) -> dict[str, object]:
    """Validate structural run isolation, naming, and optionally package paths."""

    if not rows:
        raise ValueError("dataset index must contain at least one vial")
    if any(set(row) != set(INDEX_FIELDS) for row in rows):
        raise ValueError("dataset index fields differ from the frozen contract")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("dataset index contains duplicate cases")
    disposition_counts = Counter(str(row["disposition"]) for row in rows)
    if set(disposition_counts) - {"KEEP", "DISCARD"}:
        raise ValueError("dataset index contains an invalid disposition")
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_run: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        split_counts[str(row["split"])][str(row["disposition"])] += 1
        by_run[str(row["run_id"])].append(row)
    for run_id, run_rows in by_run.items():
        if len({row["split"] for row in run_rows}) != 1:
            raise ValueError(f"run {run_id} crosses a split boundary")
        if sorted(int(row["vial_ordinal"]) for row in run_rows) != list(
            range(1, len(run_rows) + 1)
        ):
            raise ValueError(f"run {run_id} vial ordinals are incomplete")
        if len({row["run_video_path"] for row in run_rows}) != 1:
            raise ValueError(f"run {run_id} has inconsistent run video paths")
    if package_root is not None:
        root = Path(package_root)
        path_fields = (
            "per_vial_video_path",
            "run_video_path",
            "stills_path",
            "masks_path",
            "semantic_masks_path",
            "receipt_path",
            "transport_receipt_path",
        )
        for row in rows:
            for field in path_fields:
                path = root / row[field]
                if not path.exists():
                    raise ValueError(f"missing package path for {field}: {path}")
        for run_rows in by_run.values():
            ordered = sorted(run_rows, key=lambda row: int(row["vial_ordinal"]))
            expected_case_ids = [str(row["case_id"]) for row in ordered]
            for ordinal, row in enumerate(ordered, start=1):
                validate_transport_pointer(
                    root / str(row["transport_receipt_path"]),
                    expected_run_id=str(row["run_id"]),
                    expected_case_ids=expected_case_ids,
                    expected_ordinal=ordinal,
                    expected_run_video=root / str(row["run_video_path"]),
                )
    return {
        "status": "pass",
        "case_count": len(rows),
        "run_count": len(by_run),
        "disposition_counts": dict(disposition_counts),
        "split_disposition_counts": {
            key: dict(value) for key, value in split_counts.items()
        },
    }


def _validated_height_warnings(
    value: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    warnings = [dict(row) for row in (value or [])]
    for warning in warnings:
        if (
            warning.get("warning_type")
            != "particle_height_outside_geometry_volume_bounds"
            or warning.get("hard_gate") is not False
            or warning.get("claim") != "diagnostic_warning_not_a_physics_pass"
        ):
            raise ValueError("height warning provenance is invalid")
    return warnings


def _check_bound_file(root: Path, relative: object, expected_hash: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} path is invalid")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{label} hash differs from its receipt")
    return path


def validate_case_artifacts(
    case_root: str | Path,
    *,
    height_warnings: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate immutable case evidence before any post-fill composition."""

    root = Path(case_root)
    case_id = root.name
    result = _read_object(root / "case-result.json", "case result")
    receipt_root = root / "app-output"
    receipt = _read_object(receipt_root / "receipt.json", "renderer receipt")
    warnings = _validated_height_warnings(height_warnings)
    if result.get("case_id") != case_id:
        raise ValueError("case result identity differs from its directory")
    evidence = result.get("evidence")
    artifacts = evidence.get("artifact_validation") if isinstance(evidence, dict) else None
    ordinary_pass = (
        result.get("outcome") == "passed"
        and result.get("return_code") == 0
        and result.get("timed_out") is False
        and isinstance(evidence, dict)
        and evidence.get("failures") == []
        and evidence.get("launch_failures") == []
        and evidence.get("receipt_failures") == []
        and isinstance(artifacts, dict)
        and artifacts.get("status") == "pass"
        and artifacts.get("failures") == []
    )
    legacy_height_warning = (
        bool(warnings)
        and result.get("outcome") == "failed"
        and result.get("return_code") == 0
        and result.get("timed_out") is False
        and isinstance(evidence, dict)
        and evidence.get("failures") == ["receipt:realtime_volume_height_gate"]
        and evidence.get("launch_failures") == []
        and evidence.get("receipt_failures") == []
        and isinstance(artifacts, dict)
        and artifacts.get("status") == "insufficient_evidence"
        and artifacts.get("failures") == ["receipt:realtime_volume_height_gate"]
    )
    if not ordinary_pass and not legacy_height_warning:
        raise ValueError("case result is not a completed pass")
    if receipt.get("status") != "pass" or receipt.get("errors") != []:
        raise ValueError("renderer receipt hard gates did not pass")
    render = receipt.get("render")
    semantics = receipt.get("semantics")
    if not isinstance(render, dict) or not isinstance(semantics, dict):
        raise ValueError("renderer receipt lacks render or semantics evidence")
    if render.get("video_decodable") is not True:
        raise ValueError("renderer source video did not pass decode validation")
    if semantics.get("semantic_gate_passed") is not True:
        raise ValueError("renderer semantic hard gate did not pass")
    source_video = _check_bound_file(
        receipt_root,
        render.get("video_path"),
        render.get("video_sha256"),
        "source video",
    )

    stage_paths = render.get("stage_frame_paths")
    stage_hashes = render.get("stage_frame_sha256")
    if (
        not isinstance(stage_paths, list)
        or not stage_paths
        or not isinstance(stage_hashes, list)
        or len(stage_paths) != len(stage_hashes)
    ):
        raise ValueError("renderer receipt has invalid stage RGB evidence")
    if len(set(stage_hashes)) != len(stage_hashes):
        raise ValueError("stage RGB captures are not byte-unique")
    for index, (relative, expected_hash) in enumerate(
        zip(stage_paths, stage_hashes, strict=True), start=1
    ):
        _check_bound_file(receipt_root, relative, expected_hash, f"stage RGB {index}")

    pairs = semantics.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("renderer receipt has no paired RGB/mask evidence")
    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            raise ValueError("renderer RGB/mask pair is invalid")
        _check_bound_file(
            receipt_root,
            pair.get("rgb_path"),
            pair.get("rgb_sha256"),
            f"paired RGB {index}",
        )
        _check_bound_file(
            receipt_root,
            pair.get("mask_path"),
            pair.get("mask_sha256"),
            f"paired mask {index}",
        )
    receipt_warnings = receipt.get("warnings")
    if not isinstance(receipt_warnings, list):
        raise ValueError("renderer warnings must be a list")
    return {
        "case_id": case_id,
        "hard_gates_passed": True,
        "source_video": str(source_video),
        "source_video_sha256": _sha256(source_video),
        "stage_rgb_count": len(stage_paths),
        "stage_rgb_byte_unique": True,
        "paired_rgb_mask_count": len(pairs),
        "renderer_warnings": deepcopy(receipt_warnings),
        "height_warnings": warnings,
        "legacy_height_warning_revalidated": legacy_height_warning,
    }


__all__ = [
    "INDEX_FIELDS",
    "MODEL_INPUT_FORBIDDEN_KEY_FRAGMENTS",
    "MODEL_INPUT_SOURCE_FIELDS",
    "build_dataset_index_rows",
    "build_model_input_rows",
    "validate_case_artifacts",
    "validate_dataset_index",
    "validate_transport_pointer",
]
