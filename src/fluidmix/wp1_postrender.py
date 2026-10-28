"""Fail-closed CPU post-render decisions, acceptance, and packaging for WP1.

This module does not render or synthesize transport.  It only consumes frozen source
plans, immutable renderer artifacts, and separately reviewed transport products.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import cv2
import numpy as np

from .full_transition_manifest import validate_full_transition_artifacts
from .physx_run_variation import validate_tank_colour_plan_overlay
from .rigid_vial_transport import (
    INCOMING_PRIM_PATHS,
    INCOMING_VIAL_PATH,
    OUTGOING_PRIM_PATHS,
    QUEUED_PRIM_PATHS,
)
from .run_sequence_dataset import validate_case_artifacts
from .vision_measurement import measure_image


_SHA256_CHARS = frozenset("0123456789abcdef")
_VISIBILITY_TARGET = "CASE_d52026e7df33b0cb"
_VISIBILITY_REFERENCE = "CASE_ab6449c995c4b07d"
_STAGES = ("t200", "t400", "t100")
_C20_PROFILE_ID = "c20_2p5mm"
_C20_EFFECTIVE_PARTICLE_VOLUME_ML = 0.011272866937444935
_C20_SOURCE_SAMPLER = "c20_3x3_lattice"
_WP1_EXECUTION_PROFILE = {
    "gpu_model": "NVIDIA L40S",
    "isaac_sim_version": "6.2.0.0",
    "kit_version": "110.4.0+feature.386251.187cf21b.gl",
    "implementation": {
        "source_commit": "16fbb85582935ce7fa3ceeffcfa6ddffb5af5e83",
        "supervisor_sha256": "d27a610fe307f1c4e23986fef56caac547747a57804e59a87e7f3635f92da528",
        "renderer_sha256": "0a397e3de8046a3ad4754b0924e4068459939b17fe49073f916f278b30c6548c",
        "offline_validator_sha256": "f7d5014b5fdccb744ebc4e666fee8ab680b3e9f9c65d5f550c38e283716ed1b0",
        "occlusion_profile_sha256": "b056579aec18b2d2ace726d7640289e0401f6239fa4303057646040e880504f7",
        "fill_station_sha256": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
        "geometry_sha256": "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db",
    },
}
_TRANSPORT_GROUP_PREFIXES = {
    "outgoing": tuple(sorted(OUTGOING_PRIM_PATHS)),
    "incoming": tuple(sorted(INCOMING_PRIM_PATHS)),
    "queued": tuple(sorted(QUEUED_PRIM_PATHS)),
    "incoming_labels": (
        "/FillStation/FillStation_LocalRoot/FluidmixTransport/IncomingLabels",
    ),
    "queue_tail": (
        "/FillStation/FillStation_LocalRoot/FluidmixTransport/QueueTail",
    ),
}
_TRANSPORT_BACKGROUND_SENTINELS = (
    (25, 25),
    (100, 80),
    (210, 90),
    (430, 80),
    (610, 25),
    (30, 450),
    (610, 450),
)
_TRANSPORT_FILL_ANCHOR_X_PX = 304.5


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON semantics, which is not a file-byte hash."""

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _read_object(path: Path, role: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{role} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    """Atomically publish a new JSON file without a clobber race."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"output parent must be one real directory: {path.parent}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _artifact_root(path: str | Path, role: str) -> Path:
    lexical = Path(path)
    if lexical.is_symlink():
        raise ValueError(f"{role} root must not be a symlink")
    resolved = lexical.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{role} root is missing: {resolved}")
    return resolved


def _bound_regular_file(root: Path, raw: object, role: str) -> Path:
    """Resolve one relative locator beneath a real artifact root."""

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{role} locator is invalid")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} locator traversal/escape is forbidden")
    candidate = root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} locator escapes its artifact root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{role} locator crosses a symlink")
    if not resolved.is_file():
        raise ValueError(f"{role} is missing: {resolved}")
    return resolved


def _bound_absolute_file(root: Path, raw: object, role: str) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ValueError(f"{role} must use one absolute path")
    path = Path(raw)
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} path escapes its accepted artifact root") from error
    return _bound_regular_file(root, relative.as_posix(), role)


def _absolute_regular_file(raw: object, role: str) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ValueError(f"{role} must use one absolute path")
    lexical = Path(raw)
    if lexical.is_symlink():
        raise ValueError(f"{role} must not be a symlink")
    resolved = lexical.resolve()
    if not resolved.is_file():
        raise ValueError(f"{role} is missing: {resolved}")
    return resolved


def _load_frozen_bundle(root: Path, repo_root: Path) -> dict[str, object]:
    root = _artifact_root(root, "manifest")
    source_path = _bound_regular_file(
        root, "full-transition-source-spec.json", "WP1 source spec"
    )
    collection_path = _bound_regular_file(
        root, "full-transition-manifest.json", "WP1 collection"
    )
    source = _read_object(source_path, "WP1 source spec")
    collection = _read_object(collection_path, "WP1 collection")
    artifacts: dict[str, object] = {
        "full-transition-source-spec.json": source,
        "full-transition-manifest.json": collection,
    }
    leakage_path = _bound_regular_file(
        root, collection.get("leakage_manifest_path"), "leakage manifest"
    )
    artifacts["leakage-manifest.csv"] = leakage_path.read_text(encoding="utf-8")
    bindings = collection.get("run_bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("collection run bindings are not the frozen two-run set")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("collection run binding is malformed")
        for field, label in (
            ("render_plan_path", "render plan"),
            ("model_input_plan_path", "model input plan"),
            ("label_provenance_path", "label provenance"),
        ):
            relative = binding.get(field)
            child_path = _bound_regular_file(root, relative, label)
            value = _read_object(child_path, label)
            expected = binding.get(field.replace("_path", "_canonical_sha256"))
            if canonical_sha256(value) != expected:
                code = "label" if field == "label_provenance_path" else field
                raise ValueError(f"{code}_canonical_sha256 mismatch")
            artifacts[relative] = value
    return validate_full_transition_artifacts(artifacts, repo_root=repo_root)


def _plan_maps(
    artifacts: Mapping[str, object], binding: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    plan = artifacts[str(binding["render_plan_path"])]
    model = artifacts[str(binding["model_input_plan_path"])]
    if not isinstance(plan, dict) or not isinstance(model, dict):
        raise ValueError("bound child manifests are malformed")
    schedules = plan.get("execution_schedules")
    rows = model.get("vial_records")
    if not isinstance(schedules, list) or not isinstance(rows, list):
        raise ValueError("child manifest rows are malformed")
    schedule_by_case = {
        str(row["case_binding"]["case_id"]): row
        for row in schedules
        if isinstance(row, dict) and isinstance(row.get("case_binding"), dict)
    }
    model_by_case = {
        str(row["case_id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    return plan, schedule_by_case, model_by_case


def _stage_pairs(receipt: Mapping[str, object], stages: Sequence[str]) -> list[dict[str, object]]:
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("receipt-bound stage pairs are missing")
    result: list[dict[str, object]] = []
    positions: list[int] = []
    for stage in stages:
        matches = [
            (index, pair) for index, pair in enumerate(pairs)
            if isinstance(pair, dict) and pair.get("phase") == f"{stage}-stage"
        ]
        if len(matches) != 1:
            raise ValueError(f"receipt must contain exactly one {stage} stage pair")
        index, pair = matches[0]
        positions.append(index)
        if (
            pair.get("paired_capture_state_passed") is not True
            or pair.get("visible_stream_tanks") != []
            or pair.get("liquid_representation")
            != "count_driven_twin_liquid_mesh"
        ):
            raise ValueError(
                f"{stage} stage pair lacks the frozen count_driven paired-state guards"
            )
        result.append(pair)
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("receipt stage phases are not in planned order")
    if len({pair.get("rgb_sha256") for pair in result}) != len(result):
        raise ValueError("ordered stage RGBs are not byte-unique")
    return result


def _absolute_locator(app_root: Path, pair: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in ("rgb", "mask"):
        relative = pair.get(f"{role}_path")
        digest = pair.get(f"{role}_sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or not _is_sha256(digest)
        ):
            raise ValueError(f"{role} locator is malformed")
        result[f"{role}_path"] = str(
            _bound_regular_file(app_root, relative, f"paired {role}")
        )
        result[f"{role}_sha256"] = digest
    return result


def _validate_count_driven_receipt(
    receipt: Mapping[str, object], plan: Mapping[str, object]
) -> None:
    profile = plan.get("particle_resolution_profile")
    observed_profile = receipt.get("particle_resolution_profile")
    arguments = receipt.get("arguments")
    body = receipt.get("count_driven_liquid_body")
    semantics = receipt.get("semantics")
    particles = receipt.get("particle_system")
    runtime = (
        particles.get("runtime_configuration")
        if isinstance(particles, Mapping)
        else None
    )
    if not isinstance(profile, Mapping) or not isinstance(observed_profile, Mapping):
        raise ValueError("count_driven particle profile is missing")
    if set(observed_profile) != set(profile) | {"source_sampler"} or any(
        observed_profile.get(key) != value
        for key, value in profile.items()
        if key != "effective_particle_volume_ml"
    ):
        raise ValueError("count_driven particle profile differs from the frozen plan")
    if (
        profile.get("profile_id") != _C20_PROFILE_ID
        or profile.get("effective_particle_volume_ml") is not None
        or observed_profile.get("effective_particle_volume_ml")
        != _C20_EFFECTIVE_PARTICLE_VOLUME_ML
        or observed_profile.get("source_sampler") != _C20_SOURCE_SAMPLER
        or receipt.get("schema_version") != 20
        or receipt.get("render_path") != "count_driven_twin_liquid_mesh"
        or not isinstance(arguments, Mapping)
        or arguments.get("count_driven_twin_liquid_mesh") is not True
        or arguments.get("particle_resolution_profile") != _C20_PROFILE_ID
        or arguments.get("recipe8_effective_particle_volume_ml")
        != _C20_EFFECTIVE_PARTICLE_VOLUME_ML
        or not isinstance(body, Mapping)
        or body.get("representation") != "count_driven_twin_liquid_mesh"
        or not isinstance(semantics, Mapping)
        or semantics.get("count_driven_body_semantic_bound") is not True
        or semantics.get("mask_source") != "exact_renderer_semantic_class_ids"
        or not isinstance(runtime, Mapping)
        or runtime.get("spawn_spacing_m") != profile.get("spawn_spacing_m")
        or runtime.get("fluid_rest_offset_m") != profile.get("fluid_rest_offset_m")
        or runtime.get("particle_contact_offset_m")
        != profile.get("particle_contact_offset_m")
        or runtime.get("solver_position_iterations") != 16
    ):
        raise ValueError("count_driven renderer/particle/profile contract mismatch")


def _validate_case_locator_closure(case_root: Path) -> tuple[Path, dict[str, object]]:
    case_root = _artifact_root(case_root, "case")
    for relative, role in (
        ("case-result.json", "case result"),
        ("launch-receipt.json", "launch receipt"),
        ("exit-code.txt", "case exit code"),
        ("app-output/receipt.json", "renderer receipt"),
    ):
        _bound_regular_file(case_root, relative, role)
    app_root = _artifact_root(case_root / "app-output", "renderer output")
    receipt_path = _bound_regular_file(app_root, "receipt.json", "renderer receipt")
    receipt = _read_object(receipt_path, "renderer receipt")
    render = receipt.get("render")
    semantics = receipt.get("semantics")
    if not isinstance(render, Mapping) or not isinstance(semantics, Mapping):
        raise ValueError("renderer receipt locator closure is malformed")
    _bound_regular_file(app_root, render.get("video_path"), "source fill video")
    stage_paths = render.get("stage_frame_paths")
    pairs = semantics.get("pairs")
    if not isinstance(stage_paths, list) or not isinstance(pairs, list):
        raise ValueError("renderer receipt locator closure is incomplete")
    for index, relative in enumerate(stage_paths, start=1):
        _bound_regular_file(app_root, relative, f"stage RGB {index}")
    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, Mapping):
            raise ValueError("renderer receipt pair locator is malformed")
        _bound_regular_file(app_root, pair.get("rgb_path"), f"paired RGB {index}")
        _bound_regular_file(app_root, pair.get("mask_path"), f"paired mask {index}")
    return receipt_path, receipt


def _validate_accepted_fill_video(
    receipt: Mapping[str, object], path: Path
) -> dict[str, object]:
    """Require the accepted MJPG/AVI bytes; no implicit extension transcode."""

    render = receipt.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("accepted fill render record is missing")
    with path.open("rb") as stream:
        header = stream.read(12)
    if (
        path.suffix.lower() != ".avi"
        or render.get("video_container") != "avi"
        or render.get("video_codec") != "MJPG"
        or len(header) != 12
        or header[:4] != b"RIFF"
        or header[8:12] != b"AVI "
    ):
        raise ValueError(
            "accepted fill must preserve its RIFF/AVI suffix and container without an "
            "explicit hash-bound transcode"
        )
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("accepted AVI fill is not decodable")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (480, 640, 3):
                raise ValueError("accepted AVI fill resolution differs from 640x480")
            count += 1
    finally:
        capture.release()
    if (
        fps != 10.0
        or count <= 0
        or render.get("video_playback_fps") != 10.0
        or render.get("video_decoded_frame_count") != count
    ):
        raise ValueError("accepted AVI fill fps/frame-count evidence differs")
    return {"container": "avi", "codec": "MJPG", "fps": fps, "frame_count": count}


def _lab_distance(left: Sequence[object], right: Sequence[object]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("CIE Lab descriptor is malformed")
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    if not math.isfinite(value):
        raise ValueError("CIE76 is not finite")
    return value


def _visibility_payload(
    *, manifest_root: Path, run_root: Path, repo_root: Path
) -> dict[str, object]:
    manifest_root = _artifact_root(manifest_root, "manifest")
    run_root = _artifact_root(run_root, "run")
    artifacts = _load_frozen_bundle(manifest_root, repo_root)
    collection = artifacts["full-transition-manifest.json"]
    binding = collection["run_bindings"][0]
    plan, schedules, _ = _plan_maps(artifacts, binding)
    if list(binding["case_ids"])[4:6] != [_VISIBILITY_TARGET, _VISIBILITY_REFERENCE]:
        raise ValueError("visibility cases differ from the pre-registration")
    target_schedule = schedules[_VISIBILITY_TARGET]
    reference_schedule = schedules[_VISIBILITY_REFERENCE]
    target_binding = target_schedule["case_binding"]
    reference_binding = reference_schedule["case_binding"]
    if (
        target_binding.get("run_id") != "WP1-TEST-RUN-01"
        or target_binding.get("vial_ordinal") != 5
        or target_binding.get("current_recipe_id") != "S0440"
        or float(target_binding.get("residual_volume_ml", -1)) != 1.5
        or reference_binding.get("vial_ordinal") != 6
        or reference_binding.get("current_recipe_id") != "S0440"
        or float(reference_binding.get("residual_volume_ml", -1)) != 0.0
    ):
        raise ValueError("visibility case provenance differs from the pre-registration")
    stages = target_schedule["physical_sequence"]["dispense_order"]
    if stages != list(_STAGES) or reference_schedule["physical_sequence"]["dispense_order"] != stages:
        raise ValueError("visibility cases do not have the frozen three-stage order")

    warnings_by_case: dict[str, list[dict[str, object]]] = {}
    if (run_root / "shard-merge.json").is_file():
        warnings_by_case = _merge_warnings_and_validate(
            run_root, list(binding["case_ids"]), canonical_sha256(plan)
        )
    case_records: dict[str, dict[str, object]] = {}
    measured: dict[str, list[dict[str, object]]] = {}
    for role, case_id in (("target", _VISIBILITY_TARGET), ("reference", _VISIBILITY_REFERENCE)):
        case_root = run_root / "cases" / case_id
        receipt_path, receipt = _validate_case_locator_closure(case_root)
        evidence = validate_case_artifacts(
            case_root, height_warnings=warnings_by_case.get(case_id)
        )
        _validate_count_driven_receipt(receipt, plan)
        pairs = _stage_pairs(receipt, stages)
        values = []
        for stage, pair in zip(stages, pairs, strict=True):
            measurement = measure_image(
                f"{case_id}:{stage}", _absolute_locator(receipt_path.parent, pair)
            )
            if measurement.get("validity") != "valid":
                codes = measurement.get("limit_codes")
                raise ValueError(f"{role} {stage} mask evidence invalid: {codes}")
            values.append(measurement)
        measured[role] = values
        case_records[role] = {
            "case_id": case_id,
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "artifact_validation": evidence,
            "rgb_sha256": [row["rgb_sha256"] for row in values],
            "mask_sha256": [row["mask_sha256"] for row in values],
        }
    if case_records["target"]["mask_sha256"] != case_records["reference"]["mask_sha256"]:
        raise ValueError("corresponding visibility masks have mismatched hashes")

    stage_rows = []
    for stage, left, right in zip(
        stages, measured["target"], measured["reference"], strict=True
    ):
        stage_rows.append(
            {
                "stage": stage,
                "target_lab_6dp": left["lab_6dp"],
                "reference_lab_6dp": right["lab_6dp"],
                "delta_e76": round(_lab_distance(left["lab_raw"], right["lab_raw"]), 6),
                "mask_sha256": left["mask_sha256"],
            }
        )
    maximum = max(row["delta_e76"] for row in stage_rows)
    passed = maximum >= 2.0
    payload = {
        "schema_version": 1,
        "artifact_type": "fluidmix-wp1-visibility-decision-v1",
        "status": "pass" if passed else "fallback_required",
        "decision": "original_1_5ml_visible" if passed else "add_fresh_3_0ml_fallback_cases",
        "threshold_delta_e76": 2.0,
        "aggregation": "maximum_same_stage_delta_e76",
        "maximum_delta_e76": maximum,
        "stage_measurements": stage_rows,
        "plan_binding": {
            "path": str((manifest_root / binding["render_plan_path"]).resolve()),
            "canonical_sha256": canonical_sha256(plan),
            "file_sha256": file_sha256(manifest_root / binding["render_plan_path"]),
        },
        "cases": case_records,
        "automatic_manifest_mutation": False,
        "excluded_from_model_inputs": True,
        "claim_boundary": "bounded synthetic pixel visibility only; not physical or real-camera measurability",
    }
    payload["decision_sha256"] = canonical_sha256(payload)
    return payload


def decide_wp1_visibility(
    *,
    manifest_root: str | Path,
    run_root: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Measure the frozen run-1 vial-5/vial-6 DP3 pair and publish one decision."""

    root = Path(manifest_root)
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"visibility output already exists: {destination}")
    try:
        payload = _visibility_payload(
            manifest_root=root, run_root=Path(run_root), repo_root=repository
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        payload = {
            "schema_version": 1,
            "artifact_type": "fluidmix-wp1-visibility-decision-v1",
            "status": "insufficient_evidence",
            "decision": "none",
            "threshold_delta_e76": 2.0,
            "aggregation": "maximum_same_stage_delta_e76",
            "reasons": [str(error)],
            "automatic_manifest_mutation": False,
            "excluded_from_model_inputs": True,
            "claim_boundary": (
                "bounded synthetic pixel visibility only; invalid evidence makes no "
                "visibility, physical, or real-camera claim"
            ),
        }
        payload["decision_sha256"] = canonical_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def _merge_warnings_and_validate(
    run_root: Path, expected_case_ids: Sequence[str], plan_sha256: str
) -> dict[str, list[dict[str, object]]]:
    merge = _read_object(run_root / "shard-merge.json", "shard merge receipt")
    resume = merge.get("full_resume_summary")
    rows = resume.get("case_results") if isinstance(resume, dict) else None
    if (
        merge.get("schema_version") != 1
        or merge.get("artifact_type") != "physx_train_residual_shard_merge"
        or merge.get("status") != "pass"
        or merge.get("plan_sha256") != plan_sha256
        or merge.get("merged_case_count") != 9
        or merge.get("case_ids") != list(expected_case_ids)
        or not isinstance(resume, dict)
        or resume.get("status") != "pass"
        or resume.get("planned_case_count") != 9
        or resume.get("validated_completed_case_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
    ):
        raise ValueError("merged nine-case root receipt is incomplete or mismatched")
    warnings: dict[str, list[dict[str, object]]] = {}
    observed = []
    for row in rows:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        validation = evidence.get("artifact_validation") if isinstance(evidence, dict) else None
        case_id = row.get("case_id") if isinstance(row, dict) else None
        raw_warnings = validation.get("warnings") if isinstance(validation, dict) else None
        if (
            not isinstance(case_id, str)
            or row.get("outcome") not in {"passed", "recognized_completed"}
            or not isinstance(validation, dict)
            or validation.get("status") != "pass"
            or validation.get("failures") != []
            or not isinstance(raw_warnings, list)
            or any(not isinstance(warning, dict) for warning in raw_warnings)
        ):
            raise ValueError("merged case hard gates or warning provenance failed")
        observed.append(case_id)
        warnings[case_id] = deepcopy(raw_warnings)
    if observed != list(expected_case_ids):
        raise ValueError("merged case order differs from the frozen plan")
    return warnings


def _validate_fill_readiness(
    run_root: Path, *, case_ids: Sequence[str]
) -> tuple[dict[str, object], str]:
    """Bind the actual L40S profile to readiness, shard, launch, and case evidence."""

    readiness_path = _bound_regular_file(
        run_root, "wp1-fill-readiness.json", "WP1 fill readiness"
    )
    readiness = _read_object(readiness_path, "WP1 fill readiness")
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "owner_thread_id",
        "execution_profile",
        "proof_case_id",
        "supervisor_case_result",
        "offline_validator_result",
        "host_profile",
    }
    if (
        set(readiness) != required
        or readiness.get("schema_version") != 1
        or readiness.get("artifact_type") != "fluidmix_wp1_fill_readiness_v1"
        or readiness.get("status") != "pass"
        or readiness.get("proof_case_id") != "CASE_91365fbcc3369a1c"
        or readiness.get("execution_profile") != _WP1_EXECUTION_PROFILE
    ):
        raise ValueError("runtime_profile_mismatch: WP1 fill readiness is invalid")
    objects: dict[str, dict[str, object]] = {}
    for key in (
        "supervisor_case_result",
        "offline_validator_result",
        "host_profile",
    ):
        binding = readiness.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError("runtime_profile_mismatch: readiness binding is malformed")
        path = _bound_absolute_file(run_root, binding.get("path"), f"readiness {key}")
        if file_sha256(path) != binding.get("sha256"):
            raise ValueError("runtime_profile_mismatch: readiness evidence hash differs")
        objects[key] = _read_object(path, f"readiness {key}")
    host_profile = objects["host_profile"]
    if (
        set(host_profile)
        != {
            "schema_version",
            "artifact_type",
            "status",
            "gpu_model",
            "isaac_sim_version",
            "kit_version",
            "implementation",
            "process_conflicts",
        }
        or host_profile.get("schema_version") != 1
        or host_profile.get("artifact_type")
        != "fluidmix_wp1_l40s_host_profile_v1"
        or host_profile.get("status") != "pass"
        or host_profile.get("process_conflicts") != []
        or {
            key: host_profile[key]
            for key in (
                "gpu_model",
                "isaac_sim_version",
                "kit_version",
                "implementation",
            )
        }
        != _WP1_EXECUTION_PROFILE
    ):
        raise ValueError("runtime_profile_mismatch: actual L40S host profile is invalid")
    proof = objects["supervisor_case_result"]
    proof_evidence = proof.get("evidence")
    proof_validation = (
        proof_evidence.get("artifact_validation")
        if isinstance(proof_evidence, Mapping)
        else None
    )
    if (
        proof.get("case_id") != "CASE_91365fbcc3369a1c"
        or proof.get("outcome") != "passed"
        or proof.get("return_code") != 0
        or proof.get("timed_out") is not False
        or not isinstance(proof_evidence, Mapping)
        or proof_evidence.get("failures") != []
        or proof_evidence.get("launch_failures") != []
        or proof_evidence.get("receipt_failures") != []
        or not isinstance(proof_validation, Mapping)
        or proof_validation.get("status") != "pass"
        or proof_validation.get("failures") != []
    ):
        raise ValueError("runtime_profile_mismatch: bound proof case did not pass")
    offline = objects["offline_validator_result"]
    if (
        offline.get("status") != "pass"
        or offline.get("failures") != []
        or not isinstance(offline.get("offline_revalidations"), list)
        or not offline["offline_revalidations"]
        or any(
            not isinstance(row, Mapping) or row.get("status") != "pass"
            for row in offline["offline_revalidations"]
        )
    ):
        raise ValueError("runtime_profile_mismatch: offline revalidation failed")

    merge_path = _bound_regular_file(run_root, "shard-merge.json", "shard merge")
    merge = _read_object(merge_path, "shard merge")
    shards = merge.get("source_shards")
    hosts = {
        row.get("execution_host") for row in shards if isinstance(row, Mapping)
    } if isinstance(shards, list) else set()
    if len(hosts) != 1 or not all(isinstance(host, str) and host for host in hosts):
        raise ValueError("runtime_profile_mismatch: shard host evidence is invalid")
    execution_host = next(iter(hosts))
    for case_id in case_ids:
        case_root = _artifact_root(run_root / "cases" / case_id, "case")
        launch_path = _bound_regular_file(case_root, "launch-receipt.json", "launch receipt")
        launch = _read_object(launch_path, "launch receipt")
        receipt_path = _bound_regular_file(
            case_root, "app-output/receipt.json", "renderer receipt"
        )
        receipt = _read_object(receipt_path, "renderer receipt")
        versions = receipt.get("versions")
        provenance = receipt.get("provenance")
        if (
            launch.get("case_id") != case_id
            or launch.get("execution_host") != execution_host
            or not isinstance(versions, Mapping)
            or versions.get("isaac") != _WP1_EXECUTION_PROFILE["isaac_sim_version"]
            or versions.get("kit") != _WP1_EXECUTION_PROFILE["kit_version"]
            or not isinstance(provenance, Mapping)
            or provenance.get("script_sha256")
            != _WP1_EXECUTION_PROFILE["implementation"]["renderer_sha256"]
            or provenance.get("requested_stage_sha256")
            != _WP1_EXECUTION_PROFILE["implementation"]["fill_station_sha256"]
        ):
            raise ValueError(
                "runtime_profile_mismatch: case/shard/host-profile evidence differs"
            )
    return {
        "receipt_path": str(readiness_path),
        "receipt_sha256": file_sha256(readiness_path),
        "execution_profile": deepcopy(_WP1_EXECUTION_PROFILE),
        "host_profile": deepcopy(host_profile),
    }, execution_host


def _decode_video(path: Path, role: str) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        raise ValueError(f"{role} is not decodable")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (480, 640, 3):
                raise ValueError(f"{role} contains a non-640x480 frame")
            frames.append(frame)
    finally:
        capture.release()
    if not frames or not math.isclose(fps, 10.0, abs_tol=1.0e-6):
        raise ValueError(f"{role} must be a non-empty 10 fps 640x480 video")
    return frames, fps


def _frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _frame_mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _matched_p99(frames: Sequence[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(
        np.percentile(
            [_frame_mad(left, right) for left, right in zip(frames, frames[1:])],
            99,
        )
    )


def _transport_state_indices(mode: str) -> dict[str, object]:
    if mode == "entry":
        return {
            "accepted_start_endpoint": None,
            "rendered_state_indices": list(range(12)),
            "accepted_end_endpoint": "next_fill_first_pre_pour",
        }
    if mode == "between":
        return {
            "accepted_start_endpoint": "outgoing_fill_final",
            "rendered_state_indices": list(range(1, 12)),
            "accepted_end_endpoint": "next_fill_first_pre_pour",
        }
    if mode == "exit":
        return {
            "accepted_start_endpoint": "outgoing_fill_final",
            "rendered_state_indices": list(range(1, 13)),
            "accepted_end_endpoint": None,
        }
    raise ValueError("transport mode is invalid")


def _transport_layout(case_ids: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "segment_index": 0,
            "mode": "entry",
            "outgoing_case_id": None,
            "next_case_id": case_ids[0],
        }
    ]
    for index in range(len(case_ids) - 1):
        rows.append(
            {
                "segment_index": index + 1,
                "mode": "between",
                "outgoing_case_id": case_ids[index],
                "next_case_id": case_ids[index + 1],
            }
        )
    rows.append(
        {
            "segment_index": len(case_ids),
            "mode": "exit",
            "outgoing_case_id": case_ids[-1],
            "next_case_id": None,
        }
    )
    return rows


def _transport_capture(
    render_root: Path, row: Mapping[str, object], expected_index: int
) -> dict[str, object]:
    if row.get("state_index") != expected_index:
        raise ValueError("transport capture state order differs")
    rgb_path = _bound_regular_file(
        render_root, row.get("rgb_path"), f"transport state {expected_index} RGB"
    )
    if file_sha256(rgb_path) != row.get("rgb_sha256"):
        raise ValueError("transport RGB differs from its receipt")
    rgb = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape != (480, 640, 3):
        raise ValueError("transport RGB is not decodable 640x480")
    instance_path = _bound_regular_file(
        render_root,
        row.get("instance_path"),
        f"transport state {expected_index} instance raster",
    )
    mapping_path = _bound_regular_file(
        render_root,
        row.get("mapping_path"),
        f"transport state {expected_index} mapping",
    )
    if (
        file_sha256(instance_path) != row.get("instance_sha256")
        or file_sha256(mapping_path) != row.get("mapping_sha256")
    ):
        raise ValueError("transport instance evidence differs from its receipt")
    instance = np.load(instance_path, allow_pickle=False)
    mapping = _read_object(mapping_path, f"transport state {expected_index} mapping")
    if (
        instance.shape != (480, 640)
        or instance.dtype != np.uint32
        or any(not key.isdigit() or not isinstance(value, str) for key, value in mapping.items())
    ):
        raise ValueError("transport instance evidence is malformed")
    observed_ids = set(int(value) for value in np.unique(instance)) - {0}
    if not observed_ids.issubset({int(key) for key in mapping}):
        raise ValueError("transport instance raster contains an unmapped ID")
    mask_rows = row.get("group_masks")
    if not isinstance(mask_rows, Mapping) or set(mask_rows) != set(_TRANSPORT_GROUP_PREFIXES):
        raise ValueError("transport group-mask membership differs")
    counts: dict[str, int] = {}
    for name, prefixes in _TRANSPORT_GROUP_PREFIXES.items():
        mask_row = mask_rows.get(name)
        if not isinstance(mask_row, Mapping):
            raise ValueError("transport group-mask row is malformed")
        mask_path = _bound_regular_file(
            render_root,
            mask_row.get("path"),
            f"transport state {expected_index} {name} mask",
        )
        if file_sha256(mask_path) != mask_row.get("sha256"):
            raise ValueError("transport group mask differs from its receipt")
        mask = cv2.imread(os.fspath(mask_path), cv2.IMREAD_GRAYSCALE)
        ids = [
            int(instance_id)
            for instance_id, prim_path in mapping.items()
            if any(
                prim_path == prefix or prim_path.startswith(prefix + "/")
                for prefix in prefixes
            )
        ]
        expected = np.isin(instance, ids).astype(np.uint8) * 255
        if (
            mask is None
            or mask.shape != (480, 640)
            or not set(np.unique(mask)).issubset({0, 255})
            or not np.array_equal(mask, expected)
            or int(np.count_nonzero(mask)) != mask_row.get("pixel_count")
        ):
            raise ValueError("transport group mask differs from instance evidence")
        counts[name] = int(np.count_nonzero(mask))
    incoming_ids = [
        int(instance_id)
        for instance_id, prim_path in mapping.items()
        if prim_path == INCOMING_VIAL_PATH
        or prim_path.startswith(INCOMING_VIAL_PATH + "/")
    ]
    incoming_pixels = np.argwhere(np.isin(instance, incoming_ids))
    center = None if not len(incoming_pixels) else float(np.mean(incoming_pixels[:, 1]))
    attestation = row.get("attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("state_index") != expected_index
        or attestation.get("source_time_code") != 1
        or attestation.get("physics_steps") != 0
        or attestation.get("physics_callbacks") != 0
        or attestation.get("timeline_playing") is not False
    ):
        raise ValueError("transport capture runtime attestation is invalid")
    return {
        "rgb": rgb,
        "rgb_sha256": row["rgb_sha256"],
        "counts": counts,
        "incoming_center_x_px": center,
    }


def _validate_pilot_review_binding(
    value: object, *, implementation_sha256: object
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("artifact_type")
        != "fluidmix_transport_pilot_visual_review_binding_v1"
        or value.get("status") != "accepted"
        or value.get("transport_implementation_sha256") != implementation_sha256
        or value.get("independent_from_implementation") is not True
        or not isinstance(value.get("checks"), Mapping)
        or not value["checks"]
        or any(item is not True for item in value["checks"].values())
    ):
        raise ValueError("transport pilot visual review binding is invalid")
    for path_key, hash_key in (
        ("review_path", "review_sha256"),
        ("pilot_render_receipt_path", "pilot_render_receipt_sha256"),
        ("contact_sheet_path", "contact_sheet_sha256"),
    ):
        artifact = _absolute_regular_file(
            value.get(path_key), f"transport pilot {path_key}"
        )
        if file_sha256(artifact) != value.get(hash_key):
            raise ValueError("transport pilot visual review artifact differs")


def _validate_segment(
    row: Mapping[str, object],
    *,
    layout: Mapping[str, object],
    fill_records: Mapping[str, Mapping[str, object]],
    readiness: Mapping[str, object],
) -> dict[str, object]:
    mode = str(layout["mode"])
    render_root = _artifact_root(
        Path(str(row.get("render_dir", ""))), "transport render directory"
    )
    receipt_path = _absolute_regular_file(
        row.get("receipt_path"), "transport render receipt"
    )
    if receipt_path != render_root / "transport-render-receipt.json":
        raise ValueError("transport render receipt is outside its render directory")
    if file_sha256(receipt_path) != row.get("receipt_sha256") or row.get("mode") != mode:
        raise ValueError("transport segment composer binding differs")
    receipt = _read_object(receipt_path, "transport render receipt")
    segment_root = _artifact_root(
        Path(str(receipt.get("segment_dir", ""))), "transport build directory"
    )
    build_path = _bound_regular_file(
        segment_root, "transport-build-receipt.json", "transport build receipt"
    )
    build = _read_object(build_path, "transport build receipt")
    implementation = build.get("transport_implementation")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "fluidmix_transport_stage_render_receipt_v1"
        or receipt.get("status") != "candidate"
        or receipt.get("mode") != mode
        or receipt.get("state_indices") != list(range(13))
        or receipt.get("source_time_code") != 1
        or receipt.get("physics_steps") != 0
        or receipt.get("physics_callbacks") != 0
        or receipt.get("rendered_states_byte_unique") is not True
        or receipt.get("build_receipt_sha256") != file_sha256(build_path)
        or build.get("schema_version") != 1
        or build.get("artifact_type")
        != "fluidmix_transport_stage_build_receipt_v1"
        or build.get("status") != "pass"
        or build.get("mode") != mode
        or build.get("source_time_code") != 1
        or build.get("physics_steps") != 0
        or build.get("camera_and_light_specs_authored") != []
        or build.get("owner_thread_id") != receipt.get("owner_thread_id")
        or not isinstance(implementation, Mapping)
        or not _is_sha256(implementation.get("sha256"))
    ):
        raise ValueError("transport segment render/build hard contract failed")
    readiness_source = _read_object(Path(str(readiness["receipt_path"])), "WP1 fill readiness")
    expected_readiness = {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_fill_readiness_binding_v1",
        "status": "pass",
        "owner_thread_id": readiness_source["owner_thread_id"],
        "proof_case_id": readiness_source["proof_case_id"],
        "execution_profile": readiness_source["execution_profile"],
        "receipt_path": readiness["receipt_path"],
        "receipt_sha256": readiness["receipt_sha256"],
        "supervisor_case_result": readiness_source["supervisor_case_result"],
        "offline_validator_result": readiness_source["offline_validator_result"],
        "host_profile": readiness_source["host_profile"],
    }
    if receipt.get("wp1_fill_readiness") != expected_readiness:
        raise ValueError("transport WP1 fill readiness binding differs")
    _validate_pilot_review_binding(
        receipt.get("pilot_visual_review"),
        implementation_sha256=implementation.get("sha256"),
    )
    endpoints = build.get("adjacent_endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("transport build endpoint evidence is missing")
    outgoing_id = layout["outgoing_case_id"]
    next_id = layout["next_case_id"]
    outgoing = None if outgoing_id is None else fill_records[str(outgoing_id)]
    incoming = None if next_id is None else fill_records[str(next_id)]
    for field, case_id, fill in (
        ("outgoing_case_output", outgoing_id, outgoing),
        ("next_case_output", next_id, incoming),
    ):
        raw = endpoints.get(field)
        if fill is None:
            if raw is not None:
                raise ValueError("transport endpoint unexpectedly binds a case")
        elif Path(str(raw)).resolve() != Path(str(fill["path"])).parent:
            raise ValueError(f"transport endpoint case identity differs: {case_id}")

    def endpoint(name: str, fill: Mapping[str, object] | None, frame_index: int) -> None:
        value = endpoints.get(name)
        if fill is None:
            if value is not None:
                raise ValueError("transport mode-aware endpoint must be null")
            return
        expected_fields = (
            {"path", "sha256", "video_path", "video_sha256", "frame_index"}
            if name == "accepted_start"
            else {"video_path", "video_sha256", "frame_index"}
        )
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ValueError("transport endpoint binding is missing")
        video = Path(str(fill["path"]))
        if (
            value.get("frame_index") != frame_index
            or value.get("video_path") != str(video)
            or value.get("video_sha256") != fill["sha256"]
        ):
            raise ValueError("transport endpoint fill-video binding differs")
        if name == "accepted_start":
            if (
                value.get("path") != fill.get("final_rgb_path")
                or value.get("sha256") != fill.get("final_rgb_sha256")
            ):
                raise ValueError("transport accepted-start still binding differs")
            endpoint_path = _bound_absolute_file(
                video.parent, value.get("path"), "transport accepted-start still"
            )
            endpoint_rgb = cv2.imread(os.fspath(endpoint_path), cv2.IMREAD_COLOR)
            if (
                endpoint_rgb is None
                or endpoint_rgb.shape != (480, 640, 3)
                or file_sha256(endpoint_path) != value.get("sha256")
            ):
                raise ValueError("transport accepted-start still is invalid")

    endpoint("accepted_start", outgoing, -1)
    endpoint("accepted_end", incoming, 0)
    capture_rows = receipt.get("captures")
    if not isinstance(capture_rows, list) or len(capture_rows) != 13:
        raise ValueError("transport segment lacks 13 ordered captures")
    captures = [
        _transport_capture(render_root, capture, index)
        for index, capture in enumerate(capture_rows)
        if isinstance(capture, Mapping)
    ]
    if len(captures) != 13 or len({item["rgb_sha256"] for item in captures}) != 13:
        raise ValueError("transport segment captures are missing or not unique")
    baseline = captures[0]["rgb"]
    background_delta = max(
        int(
            np.max(
                np.abs(
                    item["rgb"][y, x].astype(np.int16)
                    - baseline[y, x].astype(np.int16)
                )
            )
        )
        for item in captures
        for x, y in _TRANSPORT_BACKGROUND_SENTINELS
    )
    late = captures[11:13]
    if mode == "entry":
        occupancy_passed = all(item["counts"]["outgoing"] == 0 for item in captures)
    else:
        occupancy_passed = (
            captures[0]["counts"]["outgoing"] > 0
            and all(item["counts"]["outgoing"] == 0 for item in late)
        )
    if mode in {"entry", "between"}:
        occupancy_passed = occupancy_passed and all(
            item["counts"]["incoming"] > 0
            and item["incoming_center_x_px"] is not None
            and abs(float(item["incoming_center_x_px"]) - _TRANSPORT_FILL_ANCHOR_X_PX)
            <= 3.0
            for item in late
        )
    else:
        occupancy_passed = occupancy_passed and all(
            item["counts"]["incoming_labels"] == 0
            and item["counts"]["queue_tail"] == 0
            for item in captures
        )
    start_frames = None if outgoing is None else outgoing["frames"]
    end_frames = None if incoming is None else incoming["frames"]
    start_mad = None if start_frames is None else _frame_mad(start_frames[-1], captures[1]["rgb"])
    start_threshold = None if start_frames is None else _matched_p99(start_frames)
    end_mad = None if end_frames is None else _frame_mad(captures[11]["rgb"], end_frames[0])
    end_threshold = None if end_frames is None else _matched_p99(end_frames)
    substitution = _transport_state_indices(mode)
    packaged = []
    if start_frames is not None:
        packaged.append(start_frames[-1])
    packaged.extend(captures[index]["rgb"] for index in substitution["rendered_state_indices"])
    if end_frames is not None:
        packaged.append(end_frames[0])
    independent = {
        "background_sentinels_within_2": background_delta <= 2,
        "mode_occupancy_valid": occupancy_passed,
        "start_seam_passed": start_mad is None or start_mad <= float(start_threshold),
        "end_seam_passed": end_mad is None or end_mad <= float(end_threshold),
        "packaged_sequence_byte_unique": len(packaged) == 13
        and len({_frame_sha256(frame) for frame in packaged}) == 13,
    }
    if any(value is not True for value in independent.values()):
        raise ValueError("transport independently recomputed seam/background/centering gate failed")
    checker = receipt.get("checker_evaluation")
    gates = checker.get("automatic_acceptance_gates") if isinstance(checker, Mapping) else None
    metrics = checker.get("metrics") if isinstance(checker, Mapping) else None
    if (
        not isinstance(checker, Mapping)
        or checker.get("schema_version") != 1
        or checker.get("artifact_type") != "fluidmix_transport_stage_checker_v1"
        or checker.get("status") != "pass"
        or checker.get("pilot") is not False
        or checker.get("manual_visual_review_required") is not False
        or checker.get("pilot_visual_review_status") != "accepted"
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(metrics, Mapping)
        or metrics.get("background_sentinel_max_channel_delta") != background_delta
        or metrics.get("background_sentinels_within_2")
        != independent["background_sentinels_within_2"]
        or metrics.get("mode_occupancy_valid") != independent["mode_occupancy_valid"]
        or metrics.get("start_seam_passed") != independent["start_seam_passed"]
        or metrics.get("end_seam_passed") != independent["end_seam_passed"]
        or metrics.get("packaged_sequence_byte_unique")
        != independent["packaged_sequence_byte_unique"]
    ):
        raise ValueError("transport checker evidence differs from independent recomputation")
    return {
        "segment_index": layout["segment_index"],
        "mode": mode,
        "render_dir": str(render_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "independent_gates": independent,
    }


def _validate_transport(
    path: Path,
    *,
    binding: Mapping[str, object],
    source_fills: Mapping[str, Mapping[str, object]],
    readiness: Mapping[str, object],
) -> dict[str, object]:
    transport_root = _artifact_root(path.parent, "transport")
    receipt_path = _bound_regular_file(
        transport_root, path.name, "rendered transport run receipt"
    )
    receipt = _read_object(receipt_path, "rendered transport run receipt")
    case_ids = list(binding["case_ids"])
    layout = _transport_layout(case_ids)
    if (
        receipt.get("schema_version") != 2
        or receipt.get("artifact_type") != "rendered_transport_run_sequence_v2"
        or receipt.get("status") != "pass"
        or receipt.get("run_id") != binding.get("run_id")
        or receipt.get("case_ids") != case_ids
        or receipt.get("transport_segment_count") != 10
        or receipt.get("entry_segment_count") != 1
        or receipt.get("between_segment_count") != 8
        or receipt.get("exit_segment_count") != 1
        or receipt.get("qualified_fill_receipts_changed") is not False
        or receipt.get("camera_and_lighting_changed") is not False
        or receipt.get("endpoint_substitution")
        != {mode: _transport_state_indices(mode) for mode in ("entry", "between", "exit")}
    ):
        raise ValueError("transport composer identity or hard boundary failed")
    fill_rows = receipt.get("fill_clips")
    if not isinstance(fill_rows, list) or len(fill_rows) != 9:
        raise ValueError("transport composer fill set is incomplete")
    decoded_fills: dict[str, dict[str, object]] = {}
    for case_id, row in zip(case_ids, fill_rows, strict=True):
        source = source_fills[case_id]
        if (
            not isinstance(row, Mapping)
            or row.get("case_id") != case_id
            or row.get("path") != source["path"]
            or row.get("sha256") != source["sha256"]
        ):
            raise ValueError(f"transport source_fill differs: {case_id}")
        video = Path(str(source["path"]))
        frames, fps = _decode_video(video, f"transport accepted fill {case_id}")
        if row.get("frame_count") != len(frames) or row.get("fps") != fps:
            raise ValueError(f"transport fill decode evidence differs: {case_id}")
        decoded_fills[case_id] = {**source, "frames": frames, "fps": fps}
    binding_rows = receipt.get("transport_fill_video_bindings")
    segment_rows = receipt.get("transport_segments")
    if (
        not isinstance(binding_rows, list)
        or not isinstance(segment_rows, list)
        or len(binding_rows) != 10
        or len(segment_rows) != 10
    ):
        raise ValueError("transport composer lacks its exact ten segment bindings")
    normalized_segments = []
    for expected, fill_binding, segment in zip(
        layout, binding_rows, segment_rows, strict=True
    ):
        if not isinstance(fill_binding, Mapping) or not isinstance(segment, Mapping):
            raise ValueError("transport composer segment row is malformed")

        def expected_fill(case_id: object) -> dict[str, str] | None:
            if case_id is None:
                return None
            source = source_fills[str(case_id)]
            return {
                "case_id": str(case_id),
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
            }

        if dict(fill_binding) != {
            "segment_index": expected["segment_index"],
            "mode": expected["mode"],
            "accepted_start_fill": expected_fill(expected["outgoing_case_id"]),
            "accepted_end_fill": expected_fill(expected["next_case_id"]),
        }:
            raise ValueError("transport mode-aware fill binding differs")
        normalized_segments.append(
            _validate_segment(
                segment,
                layout=expected,
                fill_records=decoded_fills,
                readiness=readiness,
            )
        )
    output = _bound_absolute_file(
        transport_root, receipt.get("output_video"), "rendered transport run video"
    )
    if file_sha256(output) != receipt.get("output_video_sha256"):
        raise ValueError("transport run video hash differs")
    frames, fps = _decode_video(output, "rendered transport run video")
    expected_count = sum(len(decoded_fills[case_id]["frames"]) for case_id in case_ids) + 112
    if (
        receipt.get("output_frame_count") != expected_count
        or len(frames) != expected_count
        or receipt.get("fps") != fps
        or receipt.get("resolution") != [640, 480]
    ):
        raise ValueError("transport run decoded fps/frame-count contract differs")
    ranges: list[tuple[int, int]] = [(0, 13)]
    cursor = 12 + len(decoded_fills[case_ids[0]]["frames"])
    for case_id in case_ids[1:]:
        ranges.append((cursor - 1, cursor + 12))
        cursor += 11 + len(decoded_fills[case_id]["frames"])
    ranges.append((cursor - 1, cursor + 12))
    transition_rows = receipt.get("encoded_transition_evidence")
    seam_rows = receipt.get("encoded_seam_evidence")
    if (
        not isinstance(transition_rows, list)
        or not isinstance(seam_rows, list)
        or len(transition_rows) != 10
        or len(seam_rows) != 10
        or receipt.get("encoded_transition_sequences_byte_unique") is not True
        or receipt.get("encoded_transition_seams_passed") is not True
    ):
        raise ValueError("transport encoded transition evidence is incomplete")
    for expected, bounds, transition, seam in zip(
        layout, ranges, transition_rows, seam_rows, strict=True
    ):
        if not isinstance(transition, Mapping) or not isinstance(seam, Mapping):
            raise ValueError("transport encoded transition row is malformed")
        start, stop = bounds
        window = frames[start:stop]
        hashes = [_frame_sha256(frame) for frame in window]
        outgoing = expected["outgoing_case_id"]
        incoming = expected["next_case_id"]
        start_mad = None if outgoing is None else _frame_mad(window[0], window[1])
        start_threshold = (
            None if outgoing is None else _matched_p99(decoded_fills[str(outgoing)]["frames"])
        )
        end_mad = None if incoming is None else _frame_mad(window[11], window[12])
        end_threshold = (
            None if incoming is None else _matched_p99(decoded_fills[str(incoming)]["frames"])
        )
        expected_transition = {
            "segment_index": expected["segment_index"],
            "mode": expected["mode"],
            "start_frame_index": start,
            "stop_frame_index": stop,
            "frame_count": 13,
            "decoded_bgr_sha256": hashes,
            "byte_unique": len(set(hashes)) == 13,
        }
        if dict(transition) != expected_transition or expected_transition["byte_unique"] is not True:
            raise ValueError("transport encoded transition hashes differ")
        numeric = {
            "segment_index": expected["segment_index"],
            "mode": expected["mode"],
            "start_seam_mad": start_mad,
            "start_seam_threshold_p99": start_threshold,
            "start_seam_passed": start_mad is None or start_mad <= float(start_threshold),
            "end_seam_mad": end_mad,
            "end_seam_threshold_p99": end_threshold,
            "end_seam_passed": end_mad is None or end_mad <= float(end_threshold),
        }
        if dict(seam) != numeric or numeric["start_seam_passed"] is not True or numeric["end_seam_passed"] is not True:
            raise ValueError("transport encoded seam evidence differs")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "segments": normalized_segments,
        "fill_clips": [deepcopy(dict(row)) for row in fill_rows],
        "run_product": {
            "path": str(output),
            "sha256": file_sha256(output),
            "frame_count": len(frames),
            "fps": fps,
        },
    }


def _validate_auxiliary_receipts(
    manifest_root: Path,
    collection: Mapping[str, object],
) -> dict[str, object]:
    case_ids = [case for binding in collection["run_bindings"] for case in binding["case_ids"]]
    recipes = [recipe for binding in collection["run_bindings"] for recipe in binding["recipe_ids"]]
    leakage_path = _bound_regular_file(
        manifest_root, "leakage-check-receipt.json", "leakage receipt"
    )
    leakage_manifest_path = _bound_regular_file(
        manifest_root, "leakage-manifest.csv", "leakage manifest"
    )
    leakage = _read_object(leakage_path, "leakage receipt")
    if (
        leakage.get("artifact_type") != "fluidmix-wp1-leakage-check-v1"
        or leakage.get("status") != "pass"
        or leakage.get("split_version") != "inn-7-splits-v3"
        or leakage.get("case_ids") != case_ids
        or leakage.get("recipe_ids") != recipes
        or leakage.get("near_video5_count") != 0
        or leakage.get("leakage_failure_count") != 0
        or leakage.get("leakage_manifest_sha256")
        != file_sha256(leakage_manifest_path)
    ):
        raise ValueError("leakage receipt failed or differs from the frozen collection")
    boundary_path = _bound_regular_file(
        manifest_root, "input-boundary-receipt.json", "input boundary receipt"
    )
    boundary = _read_object(boundary_path, "input boundary receipt")
    expected_model_hashes = [
        binding["model_input_plan_canonical_sha256"]
        for binding in collection["run_bindings"]
    ]
    if (
        boundary.get("artifact_type") != "fluidmix-wp1-input-boundary-check-v1"
        or boundary.get("status") != "pass"
        or boundary.get("forbidden_field_count") != 0
        or boundary.get("model_plan_canonical_sha256") != expected_model_hashes
    ):
        raise ValueError("input boundary receipt failed or was relabelled")
    return {
        "leakage_receipt_path": str(leakage_path),
        "leakage_receipt_sha256": file_sha256(leakage_path),
        "boundary_receipt_path": str(boundary_path),
        "boundary_receipt_sha256": file_sha256(boundary_path),
    }


def _validate_review(path: Path, run_shas: Sequence[str]) -> dict[str, object]:
    review_root = _artifact_root(path.parent, "lead review")
    receipt_path = _bound_regular_file(review_root, path.name, "lead review receipt")
    receipt = _read_object(receipt_path, "lead review receipt")
    video = _bound_absolute_file(
        review_root, receipt.get("review_video_path"), "lead review video"
    )
    if (
        receipt.get("artifact_type") != "fluidmix-wp1-review-acceptance-v1"
        or receipt.get("status") != "accepted"
        or receipt.get("reviewer_role") != "lead"
        or receipt.get("all_runs_viewed") is not True
        or receipt.get("visually_distinct_presets") is not True
        or receipt.get("run_product_sha256") != list(run_shas)
        or file_sha256(video) != receipt.get("review_video_sha256")
    ):
        raise ValueError("review artifact is absent, unaccepted, or hash-mismatched")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "video_path": str(video),
        "video_sha256": file_sha256(video),
    }


def validate_wp1_acceptance(
    *,
    manifest_root: str | Path,
    run_roots: Sequence[str | Path],
    transport_receipts: Sequence[str | Path],
    review_receipt: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate all mechanically checkable WP1 claims and always publish a result."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"acceptance output already exists: {destination}")
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fluidmix-wp1-acceptance-v1",
        "status": "failed",
        "failures": [],
        "claim_boundary": "synthetic renderer and reviewed CPU transport evidence only; no plant-validity claim",
    }
    try:
        root = _artifact_root(Path(manifest_root), "manifest")
        artifacts = _load_frozen_bundle(root, repository)
        collection = artifacts["full-transition-manifest.json"]
        bindings = collection["run_bindings"]
        if len(run_roots) != 2:
            raise ValueError("case_set requires exactly two merged nine-case roots")
        if len(transport_receipts) != 2:
            raise ValueError("transport receipt set requires exactly two runs")
        auxiliary = _validate_auxiliary_receipts(root, collection)
        visibility = _visibility_payload(
            manifest_root=root,
            run_root=Path(run_roots[0]),
            repo_root=repository,
        )
        if visibility["status"] != "pass":
            raise ValueError("visibility DP3 requires unrendered fallback cases")

        all_cases: list[dict[str, object]] = []
        readiness_records: list[dict[str, object]] = []
        execution_hosts: list[str] = []
        source_fills: dict[str, dict[str, object]] = {}
        label_counts: Counter[str] = Counter()
        recipe_ids: list[str] = []
        run_records: list[dict[str, object]] = []
        zero_measurements: dict[tuple[str, str], list[tuple[str, list[list[float]]]]] = defaultdict(list)
        for run_index, (raw_run_root, binding) in enumerate(
            zip(run_roots, bindings, strict=True), start=1
        ):
            run_root = _artifact_root(Path(raw_run_root), "merged run")
            plan_path = _bound_regular_file(
                root, binding["render_plan_path"], "render plan"
            )
            plan, schedules, model_rows = _plan_maps(artifacts, binding)
            validate_tank_colour_plan_overlay(plan)
            expected_cases = list(binding["case_ids"])
            actual_cases = sorted(
                path.name for path in (run_root / "cases").glob("CASE_*") if path.is_dir()
            )
            if sorted(expected_cases) != actual_cases:
                raise ValueError("case_set differs from the frozen nine-case plan")
            warnings = _merge_warnings_and_validate(
                run_root, expected_cases, canonical_sha256(plan)
            )
            readiness, execution_host = _validate_fill_readiness(
                run_root, case_ids=expected_cases
            )
            readiness_records.append(readiness)
            execution_hosts.append(execution_host)
            labels = artifacts[binding["label_provenance_path"]]["vial_labels"]
            label_by_case = {row["case_id"]: row for row in labels}
            ordered_recipes = []
            run_cases = []
            for ordinal, case_id in enumerate(expected_cases, start=1):
                schedule = schedules[case_id]
                model_row = model_rows[case_id]
                case_binding = schedule["case_binding"]
                recipe = case_binding["current_recipe_id"]
                ordered_recipes.append(recipe)
                label = label_by_case.get(case_id)
                residue = float(case_binding["residual_volume_ml"])
                expected_label = "keep" if residue == 0.0 else "discard"
                if (
                    not isinstance(label, dict)
                    or label.get("vial_ordinal") != ordinal
                    or label.get("expected_disposition") != expected_label
                    or label.get("prescribed_leading_residual_volume_ml") != residue
                ):
                    raise ValueError(f"label provenance was relabelled: {case_id}")
                label_counts[expected_label] += 1
                case_root = run_root / "cases" / case_id
                receipt_path, receipt = _validate_case_locator_closure(case_root)
                evidence = validate_case_artifacts(
                    case_root, height_warnings=warnings.get(case_id)
                )
                _validate_count_driven_receipt(receipt, plan)
                execution = receipt.get("execution_schedule")
                arguments = receipt.get("arguments")
                resolution = receipt.get("tank_colour_resolution")
                render = receipt.get("render")
                stages = schedule["physical_sequence"]["dispense_order"]
                pairs = _stage_pairs(receipt, stages)
                expected_stage_steps = [
                    pour["stage_capture_step"] for pour in schedule["pours"]
                ]
                if (
                    not isinstance(execution, dict)
                    or execution.get("case_id") != case_id
                    or execution.get("cohort_plan_sha256") != canonical_sha256(plan)
                    or execution.get("schedule_file_sha256") != file_sha256(plan_path)
                    or execution.get("source_run_spec_sha256") != plan["source_run_spec_sha256"]
                    or execution.get("schedule_sha256") != plan["execution_schedule_sha256_by_case"][case_id]
                    or execution.get("vial_record_sha256") != canonical_sha256(model_row)
                    or not isinstance(arguments, dict)
                    or arguments.get("video_fps") != 10.0
                    or resolution != plan["tank_colour_resolution"]
                    or not isinstance(render, dict)
                    or render.get("video_renderer") != "RaytracedLighting"
                    or render.get("stage_still_renderer") != "PathTracing"
                    or render.get("stage_frames_are_video_subsets") is not False
                    or render.get("stage_frame_sha256")
                    != [pair["rgb_sha256"] for pair in pairs]
                    or render.get("stage_frame_physics_steps") != expected_stage_steps
                    or [pair["physics_step"] for pair in pairs]
                    != expected_stage_steps
                ):
                    raise ValueError(f"canonical plan/receipt/stage/profile mismatch: {case_id}")
                fill_path = Path(str(evidence["source_video"]))
                fill_sha = str(evidence["source_video_sha256"])
                fill_video_evidence = _validate_accepted_fill_video(
                    receipt, fill_path
                )
                final_pair = max(pairs, key=lambda pair: int(pair["physics_step"]))
                final_rgb_path = _bound_regular_file(
                    receipt_path.parent,
                    final_pair.get("rgb_path"),
                    "accepted final transport RGB endpoint",
                )
                if file_sha256(final_rgb_path) != final_pair.get("rgb_sha256"):
                    raise ValueError(
                        f"accepted final transport RGB endpoint differs: {case_id}"
                    )
                source_fills[case_id] = {
                    "path": str(fill_path.resolve()),
                    "sha256": fill_sha,
                    "final_rgb_path": str(final_rgb_path),
                    "final_rgb_sha256": final_pair["rgb_sha256"],
                }
                if residue == 0.0:
                    labs = []
                    app_root = receipt_path.parent
                    for stage, pair in zip(stages, pairs, strict=True):
                        measurement = measure_image(
                            f"{case_id}:{stage}", _absolute_locator(app_root, pair)
                        )
                        if measurement.get("validity") != "valid":
                            raise ValueError(f"stage mask measurement invalid: {case_id}:{stage}")
                        labs.append(measurement["lab_raw"])
                    zero_measurements[(binding["run_id"], recipe)].append((case_id, labs))
                case_record = {
                    "run_number": run_index,
                    "run_id": binding["run_id"],
                    "vial_ordinal": ordinal,
                    "case_id": case_id,
                    "recipe_id": recipe,
                    "disposition": expected_label.upper(),
                    "residue_ml": residue,
                    "source_fill_path": str(fill_path.resolve()),
                    "source_fill_sha256": fill_sha,
                    "source_fill_video": fill_video_evidence,
                    "receipt_path": str(receipt_path.resolve()),
                    "receipt_sha256": file_sha256(receipt_path),
                    "warnings": evidence["height_warnings"],
                    "renderer_warnings": evidence["renderer_warnings"],
                }
                all_cases.append(case_record)
                run_cases.append(case_record)
            expected_recipe_sequence = [recipe for recipe in binding["recipe_ids"] for _ in range(3)]
            if ordered_recipes != expected_recipe_sequence:
                raise ValueError("recipe case order is not exactly AAA-BBB-CCC")
            recipe_ids.extend(binding["recipe_ids"])
            run_records.append(
                {
                    "run_number": run_index,
                    "run_id": binding["run_id"],
                    "recipe_ids": binding["recipe_ids"],
                    "preset_name": binding["resolved_preset_name"],
                    "preset_hash": binding["resolved_preset_hash"],
                    "plan_path": str(plan_path.resolve()),
                    "plan_file_sha256": file_sha256(plan_path),
                    "plan_canonical_sha256": canonical_sha256(plan),
                    "cases": run_cases,
                }
            )
        if any(
            record["execution_profile"] != _WP1_EXECUTION_PROFILE
            for record in readiness_records
        ):
            raise ValueError("runtime_profile_mismatch across the two host names")
        if len(set(execution_hosts)) != 2:
            raise ValueError("runtime_profile_mismatch: two distinct execution host names are required")
        if len(set(recipe_ids)) != 6 or label_counts != {"keep": 9, "discard": 9}:
            raise ValueError("counts, recipe isolation, or 9/9 label balance failed")
        sanity = []
        for (run_id, recipe), rows in zero_measurements.items():
            if len(rows) < 2:
                continue
            reference_id, reference_labs = rows[0]
            for case_id, labs in rows[1:]:
                deltas = [_lab_distance(left, right) for left, right in zip(reference_labs, labs, strict=True)]
                if max(deltas) > 2.0:
                    raise ValueError("within-block zero-residue stage sanity exceeded 2.0 DeltaE76")
                sanity.append({"run_id": run_id, "recipe_id": recipe, "left": reference_id, "right": case_id, "stage_delta_e76": [round(value, 6) for value in deltas]})

        transports = [
            _validate_transport(
                Path(path),
                binding=binding,
                source_fills=source_fills,
                readiness=readiness,
            )
            for path, binding, readiness in zip(
                transport_receipts, bindings, readiness_records, strict=True
            )
        ]
        for run, transport in zip(run_records, transports, strict=True):
            run["transport"] = transport
        run_shas = [row["run_product"]["sha256"] for row in transports]
        review = _validate_review(Path(review_receipt), run_shas)

        result.update(
            status="pass",
            failures=[],
            manifest_root=str(root),
            collection_manifest_sha256=file_sha256(root / "full-transition-manifest.json"),
            counts={"runs": 2, "cases": 18, "recipes": 6, "keep": 9, "discard": 9},
            runs=run_records,
            runtime={
                **deepcopy(_WP1_EXECUTION_PROFILE),
                "hosts": execution_hosts,
                "fill_readiness": readiness_records,
            },
            auxiliary_receipts=auxiliary,
            visibility=visibility,
            within_block_sanity=sanity,
            transport={"segment_count": 20, "receipts": transports},
            review=review,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        result["failures"] = [str(error)]
    result["acceptance_sha256"] = canonical_sha256(result)
    _write_json_new(destination, result)
    return result


def _copy_bound(source: Path, destination: Path, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file() or file_sha256(source) != expected_sha256:
        raise ValueError(f"package source differs from acceptance binding: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(destination) != expected_sha256:
        raise RuntimeError(f"package copy verification failed: {destination}")


def package_wp1_test_videos(
    *, acceptance_path: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Copy accepted AVI fills and reviewed transport products without transcoding."""

    acceptance_source = Path(acceptance_path).resolve()
    acceptance = _read_object(acceptance_source, "WP1 acceptance")
    recorded_hash = acceptance.get("acceptance_sha256")
    without_hash = {key: value for key, value in acceptance.items() if key != "acceptance_sha256"}
    if acceptance.get("status") != "pass" or recorded_hash != canonical_sha256(without_hash):
        raise ValueError("WP1 package requires an intact passing acceptance receipt")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"package output already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("package output parent must exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        test_root = staging / "videos" / "test"
        test_root.mkdir(parents=True)
        index_rows = []
        for run in acceptance["runs"]:
            number = int(run["run_number"])
            recipes = "-".join(run["recipe_ids"])
            run_name = f"test-run{number:02d}-{recipes}-{run['preset_name']}.mp4"
            run_product = run["transport"]["run_product"]
            _copy_bound(Path(run_product["path"]), test_root / run_name, run_product["sha256"])
            _copy_bound(
                Path(run["transport"]["receipt_path"]),
                test_root
                / "review-inputs"
                / f"run-{number:02d}"
                / "rendered-run-sequence.json",
                run["transport"]["receipt_sha256"],
            )
            for segment in run["transport"]["segments"]:
                segment_name = (
                    f"segment-{int(segment['segment_index']):02d}-{segment['mode']}"
                    "/transport-render-receipt.json"
                )
                _copy_bound(
                    Path(segment["receipt_path"]),
                    test_root
                    / "review-inputs"
                    / f"run-{number:02d}"
                    / segment_name,
                    segment["receipt_sha256"],
                )
            for case in run["cases"]:
                suffix = Path(case["source_fill_path"]).suffix.lower()
                if suffix != ".avi":
                    raise ValueError("accepted fill package source must preserve AVI")
                vial_name = (
                    f"test-run{number:02d}-vial{int(case['vial_ordinal']):02d}-"
                    f"{case['disposition']}-{case['case_id']}{suffix}"
                )
                _copy_bound(
                    Path(case["source_fill_path"]),
                    test_root / vial_name,
                    case["source_fill_sha256"],
                )
                fill_relative = f"review-inputs/source-fills/{case['case_id']}{suffix}"
                _copy_bound(
                    Path(case["source_fill_path"]),
                    test_root / fill_relative,
                    case["source_fill_sha256"],
                )
                index_rows.append(
                    {
                        "run_id": run["run_id"],
                        "run_number": f"{number:02d}",
                        "vial_ordinal": f"{int(case['vial_ordinal']):02d}",
                        "case_id": case["case_id"],
                        "recipe_id": case["recipe_id"],
                        "disposition_metadata": case["disposition"],
                        "preset_metadata": run["preset_name"],
                        "vial_video": vial_name,
                        "run_video": run_name,
                        "accepted_fill_review_input": fill_relative,
                        "accepted_fill_sha256": case["source_fill_sha256"],
                        "transport_run_sha256": run_product["sha256"],
                    }
                )
        review = acceptance["review"]
        _copy_bound(
            Path(review["video_path"]), test_root / "wp1-review.mp4", review["video_sha256"]
        )
        _copy_bound(
            Path(review["receipt_path"]),
            test_root / "review-inputs" / "lead-review-receipt.json",
            review["receipt_sha256"],
        )
        _copy_bound(
            acceptance_source,
            test_root / "review-inputs" / "wp1-acceptance.json",
            file_sha256(acceptance_source),
        )
        with (test_root / "INDEX.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
        (test_root / "README.md").write_text(
            "# WP1 synthetic test videos\n\n"
            "This directory contains 2 reviewed synthetic runs and 18 accepted AVI fill clips. "
            "Disposition and preset columns in `INDEX.csv` are metadata, not model inputs. "
            "No implicit transcode is performed. `review-inputs/` preserves hash-bound accepted "
            "fill clips, alternative-4 segment receipts, run composer receipts, "
            "and review receipts. The package makes no physical or plant-validity claim.\n",
            encoding="utf-8",
        )
        paths = sorted(path for path in staging.rglob("*") if path.is_file())
        sums = staging / "SHA256SUMS"
        with sums.open("x", encoding="utf-8") as stream:
            for path in paths:
                stream.write(f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}\n")
        receipt = {
            "schema_version": 1,
            "artifact_type": "fluidmix-wp1-test-video-package-v1",
            "status": "pass",
            "acceptance_path": str(acceptance_source),
            "acceptance_file_sha256": file_sha256(acceptance_source),
            "run_video_count": 2,
            "vial_video_count": 18,
            "transport_segment_count": 20,
            "accepted_fill_input_count": 18,
            "sha256sums_sha256": file_sha256(sums),
            "transport_gates_weakened": False,
            "qualified_fill_receipts_changed": False,
        }
        _write_json_new(staging / "package-receipt.json", receipt)
        os.rename(staging, destination)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "canonical_sha256",
    "decide_wp1_visibility",
    "file_sha256",
    "package_wp1_test_videos",
    "validate_wp1_acceptance",
]
