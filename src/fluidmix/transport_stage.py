"""Contracts for the reviewed, authored-layer vial transport sequence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from .rigid_vial_transport import (
    INCOMING_PRIM_PATHS,
    OUTGOING_PRIM_PATHS,
    QUEUED_PRIM_PATHS,
    SOURCE_LAYER_HASHES,
)


ENTRY_MODE = "entry"
BETWEEN_MODE = "between"
EXIT_MODE = "exit"
TRANSPORT_MODES = (ENTRY_MODE, BETWEEN_MODE, EXIT_MODE)
TRANSPORT_FPS = 10
STATE_COUNT = 13
ALL_STATE_INDICES = tuple(range(STATE_COUNT))
TRANSPORT_IMPLEMENTATION_PATHS = (
    "src/fluidmix/transport_stage.py",
    "src/fluidmix/rigid_vial_transport.py",
    "src/fluidmix/render_capture.py",
    "src/fluidmix/nuisance_authoring.py",
    "scripts/physx_three_pour_sequence.py",
    "scripts/build_transport_stage.py",
    "scripts/render_transport_stage.py",
    "scripts/check_transport_stage.py",
)
SOURCE_CURVE_APPROVED_SHA256 = (
    "c340e573d8b225326176a2cd3523ea2b0c19895c7d1187e0a67954a8a485a317"
)
SOURCE_CURVE_DELTA_37_SHA256 = (
    "621410b9ece7875fc0a0c7e5ef6feb3e7a34d521a6f6490bbf454f578702e528"
)
SOURCE_CURVE_CANONICAL_37_SHA256 = (
    "58ac18e079d9c6f99280338046584b21772eea475eac976d9a83d75d15597625"
)
SOURCE_CURVE_CANONICAL_13_SHA256 = (
    "aed5b56bcd9443684f376db70272a25940b4715051d047b0a98cf64e06b2a613"
)


@dataclass(frozen=True)
class CurveSample:
    time_code: int
    local_y: float
    progress: float


SOURCE_CURVE_SAMPLES = (
    CurveSample(180, 10.680419921875, 0.0),
    CurveSample(183, 10.601715087890625, 0.01967620849609375),
    CurveSample(186, 10.384124755859375, 0.07407379150390625),
    CurveSample(189, 10.055419921875, 0.15625),
    CurveSample(192, 9.643386840820312, 0.2592582702636719),
    CurveSample(195, 9.175796508789062, 0.3761558532714844),
    CurveSample(198, 8.680419921875, 0.5),
    CurveSample(201, 8.185050964355469, 0.6238422393798828),
    CurveSample(204, 7.717460632324219, 0.7407398223876953),
    CurveSample(207, 7.305419921875, 0.84375),
    CurveSample(210, 6.976722717285156, 0.9259243011474609),
    CurveSample(213, 6.759117126464844, 0.9803256988525391),
    CurveSample(216, 6.680419921875, 1.0),
)

ROOT = "/FillStation/FillStation_LocalRoot"
INCOMING_LABEL_SOURCE_PATHS = tuple(
    sorted(
        {
            f"{ROOT}/Fill_barcode_label",
            f"{ROOT}/Fill_label_top_rule",
            *(f"{ROOT}/Fill_barcode_{index:02d}" for index in range(27)),
        }
    )
)
QUEUE_TAIL_SOURCE_PATHS = tuple(sorted(QUEUED_PRIM_PATHS))
LEGACY_ANIMATED_PATHS = (
    f"{ROOT}/Fill_liquid",
    f"{ROOT}/Fill_dispensing_stream",
)
CAMERA_AND_LIGHT_PATHS = (
    f"{ROOT}/FillCam",
    f"{ROOT}/FillCam/FillCam",
    f"{ROOT}/Fill_front_softbox",
    f"{ROOT}/Fill_front_softbox/Fill_front_softbox",
    f"{ROOT}/Fill_overhead_panel",
    f"{ROOT}/Fill_overhead_panel/Fill_overhead_panel",
    f"{ROOT}/Fill_right_reflection",
    f"{ROOT}/Fill_right_reflection/Fill_right_reflection",
    f"{ROOT}/V2_base_reflection",
    f"{ROOT}/V2_base_reflection/V2_base_reflection",
    f"{ROOT}/V2_guard_bounce",
    f"{ROOT}/V2_guard_bounce/V2_guard_bounce",
    "/FillStation/env_light",
)

_ROOT_WORLD_Y_M_PER_LOCAL_Y = -0.02539999969303608
_DISTANCE_BY_GROUP_M = {
    "outgoing": 0.19,
    "incoming": 0.0848413319824885,
    "queued": 0.186441330754633,
    "queue_tail": 0.1016,
}
_TRANSPORT_NAMESPACE = f"{ROOT}/FluidmixTransport"
_INCOMING_LABEL_ROOT = f"{_TRANSPORT_NAMESPACE}/IncomingLabels"
_QUEUE_TAIL_ROOT = f"{_TRANSPORT_NAMESPACE}/QueueTail"
_TRANSPORT_XFORM_OP = "xformOp:translate:fluidmixTransport"
_OWNER_THREAD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVIEWED_UTC_PATTERN = re.compile(
    r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z"
)
PILOT_VISUAL_REVIEW_CHECKS = (
    "rigid_assembly_motion",
    "ghosting_and_collision_absent",
    "labels_remain_attached",
    "liquid_remains_in_outgoing_vial",
    "legacy_stream_and_liquid_absent",
    "background_and_lighting_unchanged",
)
WP1_FILL_PROOF_CASE_ID = "CASE_91365fbcc3369a1c"
WP1_FILL_EXECUTION_PROFILE = {
    "gpu_model": "NVIDIA L40S",
    "isaac_sim_version": "6.2.0.0",
    "kit_version": "110.4.0+feature.386251.187cf21b.gl",
    "implementation": {
        "source_commit": "24889cad3cf9d7302622a4f844fb4b2a4977d4c9",
        "supervisor_sha256": (
            "d34c8b5aaf94c90f69dcfe5a4e234d42be2d808609342355ed00c2f9f4cba01e"
        ),
        "renderer_sha256": (
            "c8be7ec97131d2e9de3cb1656ffd1f23fbae80b03fc27864158db929c29f8c24"
        ),
        "offline_validator_sha256": (
            "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315"
        ),
        "occlusion_profile_sha256": (
            "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e"
        ),
        "fill_station_sha256": SOURCE_LAYER_HASHES["fill_station.usda"],
        "geometry_sha256": SOURCE_LAYER_HASHES["geometry/fill_station.usdc"],
    },
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_transport_implementation(repository_root: Path) -> dict[str, object]:
    """Hash the reviewed runtime code so one pilot cannot unlock changed code."""

    root = Path(repository_root).resolve()
    records: list[dict[str, object]] = []
    for relative in TRANSPORT_IMPLEMENTATION_PATHS:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("transport implementation path escapes its repository") from error
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"transport implementation file is missing: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "files": records,
        "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
    }


def _validate_mode(mode: str) -> str:
    if type(mode) is not str or mode not in TRANSPORT_MODES:
        raise ValueError(f"transport mode must be one of {TRANSPORT_MODES}")
    return mode


def validate_owner_thread_id(owner_thread_id: str) -> str:
    """Return a USD/JSON-safe owner tag or fail closed."""

    if type(owner_thread_id) is not str or not _OWNER_THREAD_ID_PATTERN.fullmatch(
        owner_thread_id
    ):
        raise ValueError("owner thread id is not a bounded USD-safe tag")
    return owner_thread_id


def validate_wp1_fill_readiness(
    receipt_path: Path, *, declared_sha256: str, owner_thread_id: str
) -> dict[str, object]:
    """Validate the hash-bound WP1 fill readiness required before transport."""

    lexical_path = Path(receipt_path)
    path = lexical_path.resolve()
    owner = validate_owner_thread_id(owner_thread_id)
    if (
        type(declared_sha256) is not str
        or not _SHA256_PATTERN.fullmatch(declared_sha256)
        or not path.is_file()
        or lexical_path.is_symlink()
        or _sha256(path) != declared_sha256
    ):
        raise ValueError("WP1 fill readiness declared hash or path is invalid")
    receipt = _strict_json_object(path, "WP1 fill readiness receipt")
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
    if set(receipt) != required:
        raise ValueError("WP1 fill readiness fields differ from the frozen contract")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "fluidmix_wp1_fill_readiness_v1"
        or receipt.get("status") != "pass"
        or receipt.get("owner_thread_id") != owner
    ):
        raise ValueError("WP1 fill readiness identity or status is invalid")
    if receipt.get("execution_profile") != WP1_FILL_EXECUTION_PROFILE:
        raise ValueError("WP1 fill readiness execution profile is invalid")
    if receipt.get("proof_case_id") != WP1_FILL_PROOF_CASE_ID:
        raise ValueError("WP1 fill readiness proof case is invalid")

    objects: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for key, label in (
        ("supervisor_case_result", "supervisor case result"),
        ("offline_validator_result", "offline validator result"),
        ("host_profile", "host profile"),
    ):
        binding = receipt.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"WP1 {label} binding is malformed")
        raw_path = binding.get("path")
        expected_hash = binding.get("sha256")
        if (
            type(raw_path) is not str
            or not Path(raw_path).is_absolute()
            or not _SHA256_PATTERN.fullmatch(str(expected_hash))
        ):
            raise ValueError(f"WP1 {label} binding is malformed")
        lexical_bound_path = Path(raw_path)
        bound_path = lexical_bound_path.resolve()
        if (
            not bound_path.is_file()
            or lexical_bound_path.is_symlink()
            or _sha256(bound_path) != expected_hash
        ):
            raise ValueError(f"WP1 {label} differs from its binding")
        paths[key] = bound_path
        objects[key] = _strict_json_object(bound_path, f"WP1 {label}")

    case = objects["supervisor_case_result"]
    evidence = case.get("evidence")
    validation = evidence.get("artifact_validation") if isinstance(evidence, Mapping) else None
    counts = evidence.get("observed_counts") if isinstance(evidence, Mapping) else None
    if (
        case.get("case_id") != WP1_FILL_PROOF_CASE_ID
        or type(case.get("ordinal")) is not int
        or case.get("ordinal") != 4
        or case.get("outcome") != "passed"
        or type(case.get("return_code")) is not int
        or case.get("return_code") != 0
        or case.get("timed_out") is not False
        or case.get("launch_error") is not None
        or not isinstance(evidence, Mapping)
        or type(evidence.get("exit_code")) is not int
        or evidence.get("exit_code") != 0
        or evidence.get("failures") != []
        or evidence.get("launch_failures") != []
        or evidence.get("receipt_failures") != []
        or counts != {"decoded_video_frame_count": 433, "source_capture_count": 3}
        or not isinstance(validation, Mapping)
        or validation.get("status") != "pass"
        or validation.get("failures") != []
        or type(validation.get("verified_artifact_count")) is not int
        or validation.get("verified_artifact_count", 0) <= 0
    ):
        raise ValueError("WP1 supervisor case result did not pass its bound proof")

    offline = objects["offline_validator_result"]
    revalidations = offline.get("offline_revalidations")
    if (
        offline.get("status") != "pass"
        or offline.get("failures") != []
        or type(offline.get("verified_artifact_count")) is not int
        or offline.get("verified_artifact_count", 0) <= 0
        or not isinstance(revalidations, list)
        or not revalidations
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in revalidations
        )
    ):
        raise ValueError("WP1 offline validator result did not pass fresh revalidation")

    host = objects["host_profile"]
    if (
        set(host)
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
        or type(host.get("schema_version")) is not int
        or host.get("schema_version") != 1
        or host.get("artifact_type") != "fluidmix_wp1_l40s_host_profile_v1"
        or host.get("status") != "pass"
        or host.get("gpu_model") != WP1_FILL_EXECUTION_PROFILE["gpu_model"]
        or host.get("isaac_sim_version")
        != WP1_FILL_EXECUTION_PROFILE["isaac_sim_version"]
        or host.get("kit_version") != WP1_FILL_EXECUTION_PROFILE["kit_version"]
        or host.get("implementation") != WP1_FILL_EXECUTION_PROFILE["implementation"]
        or host.get("process_conflicts") != []
    ):
        raise ValueError("WP1 host profile did not prove the frozen L40S runtime")

    return {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_fill_readiness_binding_v1",
        "status": "pass",
        "owner_thread_id": owner,
        "proof_case_id": WP1_FILL_PROOF_CASE_ID,
        "execution_profile": json.loads(_canonical_json(WP1_FILL_EXECUTION_PROFILE)),
        "receipt_path": os.fspath(path),
        "receipt_sha256": declared_sha256,
        "supervisor_case_result": dict(receipt["supervisor_case_result"]),
        "offline_validator_result": dict(receipt["offline_validator_result"]),
        "host_profile": dict(receipt["host_profile"]),
    }


def _strict_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError(f"{label} must be one bounded regular file")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _review_artifact_path(raw: object, *, relative_to: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not one regular file")
    return path


def validate_pilot_visual_review(
    review_path: Path,
    *,
    transport_implementation_sha256: str,
    owner_thread_id: str,
) -> dict[str, object]:
    """Validate a human pilot review bound to the exact three-state render."""

    review_file = Path(review_path).resolve()
    owner = validate_owner_thread_id(owner_thread_id)
    if (
        type(transport_implementation_sha256) is not str
        or not _SHA256_PATTERN.fullmatch(transport_implementation_sha256)
    ):
        raise ValueError("pilot visual review implementation digest is invalid")
    review = _strict_json_object(review_file, "pilot visual review")
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "owner_thread_id",
        "transport_implementation_sha256",
        "reviewer_id",
        "independent_from_implementation",
        "reviewed_utc",
        "decision_reference",
        "pilot_render_receipt",
        "contact_sheet",
        "checks",
    }
    if set(review) != required:
        raise ValueError("pilot visual review fields differ from the frozen contract")
    reviewer_id = review.get("reviewer_id")
    reviewed_utc = review.get("reviewed_utc")
    decision_reference = review.get("decision_reference")
    checks = review.get("checks")
    if (
        review.get("schema_version") != 1
        or review.get("artifact_type")
        != "fluidmix_transport_pilot_visual_review_v1"
        or review.get("status") != "accepted"
        or review.get("owner_thread_id") != owner
        or review.get("transport_implementation_sha256")
        != transport_implementation_sha256
        or type(reviewer_id) is not str
        or not _OWNER_THREAD_ID_PATTERN.fullmatch(reviewer_id)
        or reviewer_id == owner
        or review.get("independent_from_implementation") is not True
        or type(reviewed_utc) is not str
        or not _REVIEWED_UTC_PATTERN.fullmatch(reviewed_utc)
        or type(decision_reference) is not str
        or not 1 <= len(decision_reference) <= 256
        or not isinstance(checks, Mapping)
        or set(checks) != set(PILOT_VISUAL_REVIEW_CHECKS)
        or any(checks.get(key) is not True for key in PILOT_VISUAL_REVIEW_CHECKS)
    ):
        raise ValueError("pilot visual review identity, independence, or checks are invalid")

    pilot_binding = review.get("pilot_render_receipt")
    contact_binding = review.get("contact_sheet")
    if (
        not isinstance(pilot_binding, Mapping)
        or set(pilot_binding) != {"path", "sha256"}
        or not isinstance(contact_binding, Mapping)
        or set(contact_binding) != {"path", "sha256"}
    ):
        raise ValueError("pilot visual review artifact bindings are malformed")
    pilot_path = _review_artifact_path(
        pilot_binding.get("path"),
        relative_to=review_file.parent,
        label="pilot render receipt",
    )
    if (
        not _SHA256_PATTERN.fullmatch(str(pilot_binding.get("sha256")))
        or _sha256(pilot_path) != pilot_binding.get("sha256")
    ):
        raise ValueError("pilot render receipt differs from its visual review binding")
    pilot = _strict_json_object(pilot_path, "pilot render receipt")
    pilot_segment = Path(str(pilot.get("segment_dir"))).resolve()
    pilot_build_path = pilot_segment / "transport-build-receipt.json"
    pilot_build = _strict_json_object(pilot_build_path, "pilot build receipt")
    pilot_implementation = pilot_build.get("transport_implementation")
    checker = pilot.get("checker_evaluation")
    gates = checker.get("automatic_acceptance_gates") if isinstance(checker, Mapping) else None
    required_true_gates = {
        "evidence_closure_exact",
        "runtime_partition_exact",
        "render_mode_transitions_exact",
        "camera_and_lighting_unchanged",
        "source_time_one_and_zero_physics",
        "instance_mappings_and_masks_valid",
        "background_sentinels_within_2",
        "mode_occupancy_valid",
    }
    pending_full_gates = {
        "start_seam_passed",
        "end_seam_passed",
        "packaged_sequence_byte_unique",
        "pilot_visual_review_accepted",
    }
    if (
        pilot.get("schema_version") != 1
        or pilot.get("artifact_type")
        != "fluidmix_transport_stage_render_receipt_v1"
        or pilot.get("status") != "pilot"
        or pilot.get("mode") != BETWEEN_MODE
        or pilot.get("owner_thread_id") != owner
        or pilot.get("source_time_code") != 1
        or pilot.get("physics_steps") != 0
        or pilot.get("physics_callbacks") != 0
        or pilot.get("rendered_states_byte_unique") is not True
        or pilot.get("build_receipt_sha256") != _sha256(pilot_build_path)
        or pilot_build.get("schema_version") != 1
        or pilot_build.get("artifact_type")
        != "fluidmix_transport_stage_build_receipt_v1"
        or pilot_build.get("status") != "pass"
        or pilot_build.get("mode") != BETWEEN_MODE
        or pilot_build.get("owner_thread_id") != owner
        or not isinstance(pilot_implementation, Mapping)
        or pilot_implementation.get("sha256")
        != transport_implementation_sha256
        or pilot.get("state_indices") != [0, 6, 12]
        or pilot.get("pilot_visual_review") is not None
        or not isinstance(checker, Mapping)
        or checker.get("schema_version") != 1
        or checker.get("artifact_type") != "fluidmix_transport_stage_checker_v1"
        or checker.get("status") != "pass"
        or checker.get("pilot") is not True
        or checker.get("manual_visual_review_required") is not True
        or checker.get("pilot_visual_review_status") != "required"
        or not isinstance(gates, Mapping)
        or set(gates) != required_true_gates | pending_full_gates
        or any(gates.get(key) is not True for key in required_true_gates)
        or any(gates.get(key) is not None for key in pending_full_gates)
    ):
        raise ValueError("pilot render receipt has not passed its mechanical gates")

    contact_path = _review_artifact_path(
        contact_binding.get("path"),
        relative_to=review_file.parent,
        label="pilot contact sheet",
    )
    pilot_root = pilot_path.parent.resolve()
    relative_contact = pilot.get("contact_sheet_path")
    if (
        not isinstance(relative_contact, str)
        or not relative_contact
        or Path(relative_contact).is_absolute()
        or ".." in Path(relative_contact).parts
        or (pilot_root / relative_contact).resolve() != contact_path
        or not _SHA256_PATTERN.fullmatch(str(contact_binding.get("sha256")))
        or contact_binding.get("sha256") != pilot.get("contact_sheet_sha256")
        or _sha256(contact_path) != contact_binding.get("sha256")
    ):
        raise ValueError("pilot contact sheet differs from its visual review binding")
    return {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_pilot_visual_review_binding_v1",
        "status": "accepted",
        "owner_thread_id": owner,
        "transport_implementation_sha256": transport_implementation_sha256,
        "reviewer_id": reviewer_id,
        "independent_from_implementation": True,
        "reviewed_utc": reviewed_utc,
        "decision_reference": decision_reference,
        "review_path": os.fspath(review_file),
        "review_sha256": _sha256(review_file),
        "pilot_render_receipt_path": os.fspath(pilot_path),
        "pilot_render_receipt_sha256": _sha256(pilot_path),
        "contact_sheet_path": os.fspath(contact_path),
        "contact_sheet_sha256": _sha256(contact_path),
        "checks": dict(checks),
    }


def _local_y_for_world_delta(world_delta_y_m: float) -> float:
    return world_delta_y_m / _ROOT_WORLD_Y_M_PER_LOCAL_Y


def build_state_plan(mode: str) -> tuple[dict[str, object], ...]:
    """Return the exact 13 state records for one approved transport mode."""

    selected = _validate_mode(mode)
    groups = {
        ENTRY_MODE: ("incoming", "queued", "queue_tail"),
        BETWEEN_MODE: ("outgoing", "incoming", "queued", "queue_tail"),
        EXIT_MODE: ("outgoing",),
    }[selected]
    rows: list[dict[str, object]] = []
    for state_index, sample in enumerate(SOURCE_CURVE_SAMPLES):
        progress = sample.progress
        world_deltas = {
            group: _DISTANCE_BY_GROUP_M[group] * progress for group in groups
        }
        rows.append(
            {
                "state_index": state_index,
                "mode": selected,
                "source_time_code": 1,
                "curve_time_code": sample.time_code,
                "curve_local_y": sample.local_y,
                "progress": progress,
                "physics_steps": 0,
                "world_delta_y_m_by_group": world_deltas,
                "local_delta_y_by_group": {
                    group: _local_y_for_world_delta(value)
                    for group, value in world_deltas.items()
                    if group != "queue_tail"
                },
                "incoming_label_root_local_y": (
                    _local_y_for_world_delta(
                        -_DISTANCE_BY_GROUP_M["incoming"] * (1.0 - progress)
                    )
                    if selected != EXIT_MODE
                    else None
                ),
                "queue_tail_root_local_y": (
                    4.0 * (1.0 - progress) if selected != EXIT_MODE else None
                ),
                "outgoing_visibility": (
                    "invisible" if selected == ENTRY_MODE else "inherited"
                ),
                "queued_vial_visibility": (
                    "invisible"
                    if selected != EXIT_MODE and state_index == 0
                    else "inherited"
                ),
            }
        )
    return tuple(rows)


def package_state_indices(mode: str) -> dict[str, object]:
    selected = _validate_mode(mode)
    if selected == ENTRY_MODE:
        return {
            "accepted_start_endpoint": None,
            "rendered_state_indices": list(range(12)),
            "accepted_end_endpoint": "next_fill_first_pre_pour",
        }
    if selected == BETWEEN_MODE:
        return {
            "accepted_start_endpoint": "outgoing_fill_final",
            "rendered_state_indices": list(range(1, 12)),
            "accepted_end_endpoint": "next_fill_first_pre_pour",
        }
    return {
        "accepted_start_endpoint": "outgoing_fill_final",
        "rendered_state_indices": list(range(1, 13)),
        "accepted_end_endpoint": None,
    }


def build_run_transport_layout(case_ids: Sequence[str]) -> list[dict[str, object]]:
    if (
        not isinstance(case_ids, Sequence)
        or isinstance(case_ids, (str, bytes))
        or not 1 <= len(case_ids) <= 100
        or any(type(case_id) is not str or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("a transport layout requires 1..100 unique accepted cases")
    rows: list[dict[str, object]] = [
        {
            "segment_index": 0,
            "mode": ENTRY_MODE,
            "outgoing_case_id": None,
            "next_case_id": case_ids[0],
            "endpoint_substitution": package_state_indices(ENTRY_MODE),
        }
    ]
    for index in range(len(case_ids) - 1):
        rows.append(
            {
                "segment_index": index + 1,
                "mode": BETWEEN_MODE,
                "outgoing_case_id": case_ids[index],
                "next_case_id": case_ids[index + 1],
                "endpoint_substitution": package_state_indices(BETWEEN_MODE),
            }
        )
    rows.append(
        {
            "segment_index": len(case_ids),
            "mode": EXIT_MODE,
            "outgoing_case_id": case_ids[-1],
            "next_case_id": None,
            "endpoint_substitution": package_state_indices(EXIT_MODE),
        }
    )
    return rows


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("USD numeric values must be finite")
    return format(value, ".17g")


def _reference_children(paths: Sequence[str], source_stage: Path) -> str:
    lines: list[str] = []
    for source_path in paths:
        name = source_path.rsplit("/", 1)[-1]
        lines.extend(
            (
                f'            def Xform "{name}" (',
                f"                references = @{source_stage.as_posix()}@<{source_path}>",
                "            ) {}",
            )
        )
    return "\n".join(lines)


def _mover_specs(paths: Iterable[str], local_y: float, visibility: str | None = None) -> str:
    rows: list[str] = []
    for path in sorted(paths):
        name = path.rsplit("/", 1)[-1]
        rows.append(f'        over "{name}" {{')
        rows.append(
            "            double3 xformOp:translate:fluidmixTransport = "
            f"(0, {_format_number(local_y)}, 0)"
        )
        existing_order = (
            ("xformOp:transform",)
            if path == f"{ROOT}/FluidmixF2CountDrivenLiquid"
            else ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
        )
        order = (_TRANSPORT_XFORM_OP, *existing_order)
        rows.append(
            "            uniform token[] xformOpOrder = ["
            + ", ".join(f'"{item}"' for item in order)
            + "]"
        )
        if visibility is not None:
            rows.append(f'            token visibility = "{visibility}"')
        rows.append("        }")
    return "\n".join(rows)


# These line-oriented emitters deliberately avoid compact one-line prim specs.  USD's
# text grammar accepts the same semantics only when a property declaration and closing
# brace are on separate statements, and keeping every line explicit makes review of the
# authored boundary straightforward.
def _common_layer_text(
    mode: str,
    source_stage: Path,
    owner_thread_id: str,
    transport_implementation_sha256: str,
) -> str:
    lines = [
        "#usda 1.0",
        "(",
        "    startTimeCode = 1",
        "    endTimeCode = 686",
        "    framesPerSecond = 30",
        "    timeCodesPerSecond = 30",
        "    subLayers = [",
        "        @transport-liquid.usda@,",
        f"        @{source_stage.as_posix()}@",
        "    ]",
        "    customLayerData = {",
        f'        string fluidmixTransportMode = "{mode}"',
        f'        string ownerThreadId = "{owner_thread_id}"',
        f'        string transportImplementationSha256 = "{transport_implementation_sha256}"',
        "        int sourceTimeCode = 1",
        f'        string sourceCurveSha256 = "{SOURCE_CURVE_APPROVED_SHA256}"',
        "    }",
        ")",
        "",
        'over "FillStation" {',
        '    over "FillStation_LocalRoot" {',
        '        over "Fill_liquid" {',
        '            token visibility = "invisible"',
        "        }",
        '        over "Fill_dispensing_stream" {',
        '            token visibility = "invisible"',
        "        }",
        '        def Xform "FluidmixTransport" (',
        "            customData = {",
        f'                string mode = "{mode}"',
        "                int sourceTimeCode = 1",
        f'                string sourceCurveSha256 = "{SOURCE_CURVE_APPROVED_SHA256}"',
        "            }",
        "        ) {",
    ]
    if mode != EXIT_MODE:
        lines.extend(('            def Xform "IncomingLabels" {',))
        lines.extend(_reference_children(INCOMING_LABEL_SOURCE_PATHS, source_stage).splitlines())
        lines.extend(("            }", '            def Xform "QueueTail" {'))
        lines.extend(_reference_children(QUEUE_TAIL_SOURCE_PATHS, source_stage).splitlines())
        lines.append("            }")
    lines.extend(("        }", "    }", "}", ""))
    return "\n".join(lines)


def _visibility_specs(paths: Iterable[str], visibility: str) -> str:
    lines: list[str] = []
    for path in sorted(paths):
        name = path.rsplit("/", 1)[-1]
        lines.extend(
            (
                f'        over "{name}" {{',
                f'            token visibility = "{visibility}"',
                "        }",
            )
        )
    return "\n".join(lines)


def _state_layer_text(
    mode: str,
    row: Mapping[str, object],
    owner_thread_id: str,
    transport_implementation_sha256: str,
) -> str:
    local = row["local_delta_y_by_group"]
    if not isinstance(local, Mapping):
        raise ValueError("state plan local deltas are malformed")
    blocks: list[str] = []
    if mode == ENTRY_MODE:
        blocks.append(_visibility_specs(OUTGOING_PRIM_PATHS, "invisible"))
    if "outgoing" in local:
        blocks.append(_mover_specs(OUTGOING_PRIM_PATHS, float(local["outgoing"])))
    if "incoming" in local:
        blocks.append(_mover_specs(INCOMING_PRIM_PATHS, float(local["incoming"])))
    if "queued" in local:
        other_queued = set(QUEUED_PRIM_PATHS) - {f"{ROOT}/Fill_vial_neighbour_2"}
        blocks.append(_mover_specs(other_queued, float(local["queued"])))
        blocks.append(
            _mover_specs(
                (f"{ROOT}/Fill_vial_neighbour_2",),
                float(local["queued"]),
                str(row["queued_vial_visibility"]),
            )
        )
    lines = [
        "#usda 1.0",
        "(",
        "    startTimeCode = 1",
        "    endTimeCode = 686",
        "    framesPerSecond = 30",
        "    timeCodesPerSecond = 30",
        "    subLayers = [@../transport-stage.usda@]",
        "    customLayerData = {",
        f'        int curveTimeCode = {int(row["curve_time_code"])}',
        f'        string ownerThreadId = "{owner_thread_id}"',
        f'        string transportImplementationSha256 = "{transport_implementation_sha256}"',
        "        int sourceTimeCode = 1",
        f'        int stateIndex = {int(row["state_index"])}',
        f'        string transportMode = "{mode}"',
        "    }",
        ")",
        "",
        'over "FillStation" {',
        '    over "FillStation_LocalRoot" {',
        *[line for block in blocks if block for line in block.splitlines()],
    ]
    if mode != EXIT_MODE:
        lines.extend(
            (
                '        over "FluidmixTransport" {',
                '            over "IncomingLabels" {',
                "                double3 xformOp:translate:fluidmixTransport = "
                f'(0, {_format_number(float(row["incoming_label_root_local_y"]))}, 0)',
                "                uniform token[] xformOpOrder = "
                '["xformOp:translate:fluidmixTransport"]',
                "            }",
                '            over "QueueTail" {',
                "                double3 xformOp:translate:fluidmixTransport = "
                f'(0, {_format_number(float(row["queue_tail_root_local_y"]))}, 0)',
                "                uniform token[] xformOpOrder = "
                '["xformOp:translate:fluidmixTransport"]',
                "            }",
                "        }",
            )
        )
    lines.extend(("    }", "}", ""))
    return "\n".join(lines)


def build_transport_layers(
    *,
    mode: str,
    owner_thread_id: str,
    transport_implementation_sha256: str,
    source_stage: Path,
    source_geometry: Path,
    liquid_layer: Path,
    output_dir: Path,
    expected_source_hashes: Mapping[str, str] = SOURCE_LAYER_HASHES,
    liquid_evidence_sha256: str,
) -> dict[str, object]:
    """Materialize one immutable 13-state authored overlay set atomically."""

    selected = _validate_mode(mode)
    owner = validate_owner_thread_id(owner_thread_id)
    if (
        type(transport_implementation_sha256) is not str
        or not _SHA256_PATTERN.fullmatch(transport_implementation_sha256)
    ):
        raise ValueError("transport implementation digest is invalid")
    stage = Path(source_stage).resolve()
    geometry = Path(source_geometry).resolve()
    liquid = Path(liquid_layer).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("transport output directory must not already exist")
    if set(expected_source_hashes) != set(SOURCE_LAYER_HASHES):
        raise ValueError("source layer hash membership is not exact")
    for key, path in (
        ("fill_station.usda", stage),
        ("geometry/fill_station.usdc", geometry),
    ):
        if not path.is_file() or _sha256(path) != expected_source_hashes[key]:
            raise ValueError(f"source layer differs from its bound hash: {key}")
    if not liquid.is_file() or _sha256(liquid) != liquid_evidence_sha256:
        raise ValueError("transport liquid layer differs from its bound evidence")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        copied_liquid = temporary / "transport-liquid.usda"
        shutil.copyfile(liquid, copied_liquid)
        common = temporary / "transport-stage.usda"
        common.write_text(
            _common_layer_text(
                selected, stage, owner, transport_implementation_sha256
            ),
            encoding="utf-8",
        )
        states = temporary / "states"
        states.mkdir()
        state_records: list[dict[str, object]] = []
        for row in build_state_plan(selected):
            state_index = int(row["state_index"])
            state = states / f"state-{state_index:02d}.usda"
            state.write_text(
                _state_layer_text(
                    selected, row, owner, transport_implementation_sha256
                ),
                encoding="utf-8",
            )
            state_records.append(
                {
                    "state_index": state_index,
                    "path": state.relative_to(temporary).as_posix(),
                    "sha256": _sha256(state),
                    "curve_time_code": row["curve_time_code"],
                    "progress": row["progress"],
                }
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_layers_v1",
            "mode": selected,
            "owner_thread_id": owner,
            "transport_implementation_sha256": transport_implementation_sha256,
            "source_time_code": 1,
            "fps": TRANSPORT_FPS,
            "state_count": STATE_COUNT,
            "source_curve_approved_sha256": SOURCE_CURVE_APPROVED_SHA256,
            "source_curve_delta_37_sha256": SOURCE_CURVE_DELTA_37_SHA256,
            "source_curve_canonical_37_sha256": SOURCE_CURVE_CANONICAL_37_SHA256,
            "source_curve_canonical_13_sha256": SOURCE_CURVE_CANONICAL_13_SHA256,
            "source_layers": {
                "fill_station.usda": {
                    "path": os.fspath(stage),
                    "sha256": expected_source_hashes["fill_station.usda"],
                },
                "geometry/fill_station.usdc": {
                    "path": os.fspath(geometry),
                    "sha256": expected_source_hashes["geometry/fill_station.usdc"],
                },
            },
            "common_layer": {
                "path": common.name,
                "sha256": _sha256(common),
            },
            "liquid_layer": {
                "path": copied_liquid.name,
                "sha256": _sha256(copied_liquid),
            },
            "state_layers": state_records,
            "mover_path_counts": {"outgoing": 57, "incoming": 27, "queued": 27},
            "camera_and_light_specs_authored": [],
            "physics_steps": 0,
            "endpoint_substitution": package_state_indices(selected),
        }
        manifest["manifest_content_sha256"] = hashlib.sha256(
            _canonical_json(manifest)
        ).hexdigest()
        (temporary / "transport-manifest.json").write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        temporary.rename(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _file_records(paths: Iterable[Path], label: str) -> list[dict[str, object]]:
    resolved = [Path(path).resolve() for path in paths]
    if any(not path.is_file() for path in resolved):
        raise ValueError(f"{label} evidence contains a missing file")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} evidence contains a duplicate path")
    return [
        {"path": os.fspath(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(resolved)
    ]


def capture_file_records(
    paths: Iterable[Path], label: str = "file"
) -> list[dict[str, object]]:
    """Freeze one exact, duplicate-free file set for a later equality check."""

    return _file_records(paths, label)


def _validated_counters(counters: Mapping[str, object]) -> dict[str, object]:
    required = {
        "physics_steps": 0,
        "physics_callbacks": 0,
        "timeline_playing": False,
        "source_time_code": 1,
    }
    if not isinstance(counters, Mapping) or any(
        counters.get(key) != expected or type(counters.get(key)) is not type(expected)
        for key, expected in required.items()
    ):
        raise ValueError("transport counters do not prove frozen time and zero physics")
    return dict(counters)


def capture_evidence_closure(
    *,
    source_files: Iterable[Path],
    snapshot_arrays: Iterable[Path],
    state_layers: Iterable[Path],
    dependency_files: Iterable[Path],
    counters: Mapping[str, object],
) -> dict[str, object]:
    """Freeze every byte consumed by a transport render before output exists."""

    closure: dict[str, object] = {
        "source_files": _file_records(source_files, "source file"),
        "snapshot_arrays": _file_records(snapshot_arrays, "snapshot array"),
        "state_layers": _file_records(state_layers, "state layer"),
        "dependency_files": _file_records(dependency_files, "dependency file"),
        "counters": _validated_counters(counters),
    }
    if len(closure["source_files"]) != 2:  # type: ignore[arg-type]
        raise ValueError("source evidence must contain exactly both reviewed layers")
    if len(closure["snapshot_arrays"]) != 13:  # type: ignore[arg-type]
        raise ValueError("snapshot evidence must bind exactly 13 arrays")
    if len(closure["state_layers"]) != 13:  # type: ignore[arg-type]
        raise ValueError("state evidence must bind exactly 13 layers")
    if not closure["dependency_files"]:
        raise ValueError("dependency evidence must not be empty")
    closure["sha256"] = hashlib.sha256(_canonical_json(closure)).hexdigest()
    return closure


def validate_evidence_closure(
    frozen: Mapping[str, object],
    *,
    source_files: Iterable[Path],
    snapshot_arrays: Iterable[Path],
    state_layers: Iterable[Path],
    dependency_files: Iterable[Path],
    counters: Mapping[str, object],
) -> None:
    """Re-read the complete closure after the final independent checker."""

    if not isinstance(frozen, Mapping) or set(frozen) != {
        "source_files",
        "snapshot_arrays",
        "state_layers",
        "dependency_files",
        "counters",
        "sha256",
    }:
        raise ValueError("frozen transport evidence closure is malformed")
    checks = (
        ("source_files", _file_records(source_files, "source file"), "source file"),
        (
            "snapshot_arrays",
            _file_records(snapshot_arrays, "snapshot array"),
            "snapshot array",
        ),
        ("state_layers", _file_records(state_layers, "state layer"), "state layer"),
        (
            "dependency_files",
            _file_records(dependency_files, "dependency file"),
            "dependency file",
        ),
    )
    for key, current, label in checks:
        if current != frozen.get(key):
            raise ValueError(f"{label} evidence drifted")
    if _validated_counters(counters) != frozen.get("counters"):
        raise ValueError("transport counter evidence drifted")
    payload = {key: frozen[key] for key in frozen if key != "sha256"}
    if frozen.get("sha256") != hashlib.sha256(_canonical_json(payload)).hexdigest():
        raise ValueError("frozen transport evidence closure digest is invalid")


def freeze_runtime_partition(
    prim_paths: Iterable[str], property_paths: Iterable[str]
) -> dict[str, object]:
    raw_prims = list(prim_paths)
    raw_properties = list(property_paths)
    if (
        not raw_prims
        or any(type(path) is not str for path in raw_prims)
        or any(type(path) is not str for path in raw_properties)
    ):
        raise ValueError("post-annotator runtime render partition is malformed")
    prims = sorted(raw_prims)
    properties = sorted(raw_properties)
    def is_render_property(path: str) -> bool:
        return path.startswith("/Render/") or path.startswith("/Render.")

    if (
        len(set(prims)) != len(prims)
        or any(
            path != "/Render" and not path.startswith("/Render/") for path in prims
        )
        or not {"/Render", "/Render/Vars", "/Render/PostProcess/SDGPipeline"}.issubset(
            prims
        )
        or not properties
        or len(set(properties)) != len(properties)
        or any(not is_render_property(path) for path in properties)
    ):
        raise ValueError("post-annotator runtime render partition is malformed")
    payload: dict[str, object] = {
        "prim_paths": prims,
        "property_paths": properties,
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def validate_runtime_partition(
    frozen: Mapping[str, object], prim_paths: Iterable[str], property_paths: Iterable[str]
) -> None:
    try:
        current = freeze_runtime_partition(prim_paths, property_paths)
    except ValueError as error:
        raise ValueError("runtime render partition drifted") from error
    if dict(frozen) != current:
        raise ValueError("runtime render partition drifted")


__all__ = [
    "ALL_STATE_INDICES",
    "BETWEEN_MODE",
    "CAMERA_AND_LIGHT_PATHS",
    "CurveSample",
    "ENTRY_MODE",
    "EXIT_MODE",
    "INCOMING_LABEL_SOURCE_PATHS",
    "INCOMING_PRIM_PATHS",
    "LEGACY_ANIMATED_PATHS",
    "OUTGOING_PRIM_PATHS",
    "PILOT_VISUAL_REVIEW_CHECKS",
    "QUEUED_PRIM_PATHS",
    "QUEUE_TAIL_SOURCE_PATHS",
    "SOURCE_CURVE_APPROVED_SHA256",
    "SOURCE_CURVE_CANONICAL_13_SHA256",
    "SOURCE_CURVE_CANONICAL_37_SHA256",
    "SOURCE_CURVE_DELTA_37_SHA256",
    "SOURCE_CURVE_SAMPLES",
    "SOURCE_LAYER_HASHES",
    "STATE_COUNT",
    "TRANSPORT_FPS",
    "TRANSPORT_IMPLEMENTATION_PATHS",
    "WP1_FILL_EXECUTION_PROFILE",
    "WP1_FILL_PROOF_CASE_ID",
    "build_run_transport_layout",
    "build_state_plan",
    "build_transport_layers",
    "capture_transport_implementation",
    "capture_evidence_closure",
    "capture_file_records",
    "freeze_runtime_partition",
    "package_state_indices",
    "validate_evidence_closure",
    "validate_owner_thread_id",
    "validate_pilot_visual_review",
    "validate_runtime_partition",
    "validate_wp1_fill_readiness",
]
