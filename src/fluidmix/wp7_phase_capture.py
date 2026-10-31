"""Hash-bound WP7 phase capture plans and legacy capture-receipt publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from .wp7_rgb_runtime import RGBRuntimeError, compose_fill_windows


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_FIELDS = {
    "schema_version", "artifact_type", "phase", "capture_origin",
    "video_sha256", "fill_clips", "encoded_transition_evidence",
    "source_provenance",
}
_TRANSITION_FIELDS = {
    "segment_index", "start_frame_index", "stop_frame_index", "frame_count",
    "decoded_bgr_sha256",
}
_FILL_FIELDS = {"case_id", "frame_count"}
_SOURCE_FIELDS = {"role", "path", "sha256"}
_SOURCE_ROLES_BY_PHASE = {
    "run-test": frozenset({
        "ordered_wp1_composition_receipt",
        "ordered_wp1_review_receipt",
        "ordered_wp1_run_receipt",
    }),
    "run-real-check": frozenset({
        "video5_dataset_manifest",
        "video5_recipe_metadata",
        "video5_timing_metadata",
        "video5_transition_window_metadata",
    }),
}
_SOURCE_ROLE_ORDER_BY_PHASE = {
    "run-test": (
        "ordered_wp1_composition_receipt",
        "ordered_wp1_review_receipt",
        "ordered_wp1_run_receipt",
    ),
    "run-real-check": (
        "video5_dataset_manifest",
        "video5_recipe_metadata",
        "video5_timing_metadata",
        "video5_transition_window_metadata",
    ),
}
_UPSTREAM_ARTIFACT_TYPES = {
    "ordered_wp1_composition_receipt": "wp7_ordered_test_video_v1",
    "ordered_wp1_review_receipt": (
        "fluidmix_wp7_ordered_test_video_final_output_review_v1"
    ),
    "ordered_wp1_run_receipt": "wp7_ordered_test_run_receipt_v1",
    "video5_dataset_manifest": "video5_dataset_manifest_v1",
    "video5_recipe_metadata": "video5_recipe_metadata_v1",
    "video5_timing_metadata": "video5_timing_metadata_v1",
    "video5_transition_window_metadata": "video5_transition_window_metadata_v1",
}


class PhaseCaptureError(ValueError):
    """A phase-capture plan or publication failed closed."""


@dataclass(frozen=True)
class PhaseCapturePublication:
    capture_receipt: dict[str, object]
    provenance_receipt: dict[str, object]
    decoded_rgb_frames: tuple[np.ndarray, ...]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhaseCaptureError("phase capture value is not canonical JSON") from error


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PhaseCaptureError(f"{role} must be a lowercase SHA-256")
    return value


def _integer(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        raise PhaseCaptureError(f"{role} must be a non-negative integer")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PhaseCaptureError(f"phase capture JSON has duplicate field: {key}")
        value[key] = item
    return value


def _read_bound_file(path_value: Path, expected_sha256: object, role: str) -> tuple[Path, bytes]:
    path = Path(path_value).expanduser().absolute()
    expected = _sha(expected_sha256, f"{role} SHA-256")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PhaseCaptureError(f"{role} must be a regular non-symlink file") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PhaseCaptureError(f"{role} must be a regular non-symlink file")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise PhaseCaptureError(f"{role} must be a regular non-symlink file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PhaseCaptureError(f"{role} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PhaseCaptureError(f"{role} raw-file SHA-256 mismatch")
    return path, raw


def _fresh(path_value: Path, role: str) -> Path:
    path = Path(path_value).expanduser().absolute()
    if os.path.lexists(path):
        raise PhaseCaptureError(f"{role} already exists")
    return path


def _publish_pair(
    first_path: Path, first: object, second_path: Path, second: object
) -> None:
    descriptors: list[int] = []
    created: list[Path] = []
    try:
        for path in (first_path, second_path):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
            )
            descriptors.append(descriptor)
            created.append(path)
        for value in (first, second):
            descriptor = descriptors.pop(0)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_bytes(value) + b"\n")
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _decoded_frames(value: object) -> tuple[np.ndarray, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise PhaseCaptureError("decoded RGB frames must be a nonempty sequence")
    frames = tuple(np.asarray(frame) for frame in value)
    if any(frame.dtype != np.uint8 or frame.shape != (480, 640, 3) for frame in frames):
        raise PhaseCaptureError("decoded RGB frames must be 480x640 uint8 RGB")
    return frames


def _validate_source_metadata(
    role: str,
    value: object,
    *,
    phase: str,
    video_sha256: str,
    recipe_sequence_sha256: str,
    case_ids: list[str],
    capture_timing_sha256: str,
    decoded_frame_count: int,
    composition_receipt_sha256: str | None,
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "artifact_type", "phase", "role", "video_sha256",
        "recipe_sequence_sha256", "ordered_case_ids", "capture_timing_sha256",
        "upstream_artifact_type", "upstream_path", "upstream_sha256", "status",
    }:
        raise PhaseCaptureError("capture source metadata identity is invalid")
    if (
        value["schema_version"] != "wp7-capture-source-adapter-v1"
        or value["artifact_type"] != "wp7_capture_source_adapter"
        or value["phase"] != phase
        or value["role"] != role
        or value["video_sha256"] != video_sha256
        or value["recipe_sequence_sha256"] != recipe_sequence_sha256
        or value["ordered_case_ids"] != case_ids
        or value["capture_timing_sha256"] != capture_timing_sha256
        or value["upstream_artifact_type"] != _UPSTREAM_ARTIFACT_TYPES[role]
        or value["status"] != "pass"
    ):
        raise PhaseCaptureError("capture source metadata identity is invalid")
    upstream_value = value["upstream_path"]
    if (
        not isinstance(upstream_value, str)
        or not Path(upstream_value).is_absolute()
        or Path(upstream_value).suffix.lower() != ".json"
    ):
        raise PhaseCaptureError("capture upstream must be absolute JSON metadata")
    upstream_path, upstream_raw = _read_bound_file(
        Path(upstream_value), value["upstream_sha256"],
        f"capture upstream {role}",
    )
    try:
        upstream = json.loads(
            upstream_raw.decode("utf-8"), object_pairs_hook=_strict_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCaptureError("capture upstream must be strict JSON metadata") from error
    if not isinstance(upstream, dict) or upstream.get(
        "artifact_type"
    ) != value["upstream_artifact_type"] or upstream.get("status") != "pass":
        raise PhaseCaptureError("capture upstream artifact identity is invalid")
    if role == "ordered_wp1_composition_receipt":
        output = upstream.get("output")
        runs = upstream.get("ordered_runs")
        verification = upstream.get("verification")
        if (
            not isinstance(output, Mapping)
            or output.get("sha256") != video_sha256
            or output.get("frame_count") != decoded_frame_count
            or not isinstance(runs, list)
            or len(runs) != 2
            or any(not isinstance(run, Mapping) for run in runs)
            or [case for run in runs if isinstance(run, Mapping)
                for case in run.get("case_ids", [])] != case_ids
            or not isinstance(verification, Mapping)
            or verification.get("case_count") != len(case_ids)
            or verification.get("run_count") != 2
            or verification.get("source_order")
            != [run.get("run_id") for run in runs]
        ):
            raise PhaseCaptureError("ordered WP1 composition identity is invalid")
    elif role == "ordered_wp1_review_receipt":
        identity = upstream.get("identity")
        contract = upstream.get("case_contract")
        media = upstream.get("output_media")
        if (
            upstream.get("verdict") != "pass"
            or not isinstance(identity, Mapping)
            or identity.get("output_mp4_sha256") != video_sha256
            or identity.get("output_receipt_sha256")
            != composition_receipt_sha256
            or not isinstance(contract, Mapping)
            or contract.get("count") != len(case_ids)
            or contract.get("unique") is not True
            or not isinstance(media, Mapping)
            or media.get("sha256") != video_sha256
            or media.get("frame_count") != decoded_frame_count
        ):
            raise PhaseCaptureError("ordered WP1 review identity is invalid")
    elif role == "ordered_wp1_run_receipt":
        if set(upstream) != {
            "schema_version", "artifact_type", "status", "run_receipts"
        }:
            raise PhaseCaptureError("ordered WP1 run receipt manifest is invalid")
        run_receipts = upstream["run_receipts"]
        if not isinstance(run_receipts, list) or len(run_receipts) != 2:
            raise PhaseCaptureError("ordered WP1 run receipt manifest is invalid")
        run_structures: list[dict[str, object]] = []
        for index, binding in enumerate(run_receipts):
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {
                    "path", "sha256", "artifact_type", "run_id", "case_ids",
                    "output_video_sha256", "output_frame_count",
                }
                or binding["artifact_type"] != "rendered_transport_run_sequence_v4"
                or not isinstance(binding["path"], str)
                or not Path(binding["path"]).is_absolute()
                or Path(binding["path"]).suffix.lower() != ".json"
            ):
                raise PhaseCaptureError("ordered WP1 run receipt manifest is invalid")
            _, run_raw = _read_bound_file(
                Path(str(binding["path"])), binding["sha256"],
                f"ordered WP1 source run receipt {index + 1}",
            )
            try:
                run_receipt = json.loads(
                    run_raw.decode("utf-8"), object_pairs_hook=_strict_object
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PhaseCaptureError(
                    "ordered WP1 source run receipt must be strict JSON"
                ) from error
            if (
                not isinstance(run_receipt, dict)
                or run_receipt.get("artifact_type") != binding["artifact_type"]
                or run_receipt.get("status") != "pass"
                or run_receipt.get("run_id") != binding["run_id"]
                or run_receipt.get("case_ids") != binding["case_ids"]
                or run_receipt.get("output_video_sha256")
                != binding["output_video_sha256"]
                or run_receipt.get("output_frame_count")
                != binding["output_frame_count"]
            ):
                raise PhaseCaptureError(
                    "ordered WP1 source run receipt identity is invalid"
                )
            transitions = run_receipt.get("encoded_transition_evidence")
            fills = run_receipt.get("fill_clips")
            if (
                not isinstance(transitions, list)
                or not transitions
                or not isinstance(fills, list)
                or any(not isinstance(row, Mapping) for row in (*transitions, *fills))
            ):
                raise PhaseCaptureError(
                    "ordered WP1 source run timing identity is invalid"
                )
            run_structures.append({
                "frame_count": binding["output_frame_count"],
                "transitions": [
                    {
                        key: row.get(key)
                        for key in (
                            "segment_index", "start_frame_index",
                            "stop_frame_index", "frame_count",
                        )
                    }
                    for row in transitions
                ],
                "fills": [
                    {"case_id": row.get("case_id"), "frame_count": row.get("frame_count")}
                    for row in fills
                ],
            })
        if (
            [case for binding in run_receipts for case in binding["case_ids"]]
            != case_ids
            or sum(int(binding["output_frame_count"]) for binding in run_receipts)
            != decoded_frame_count
        ):
            raise PhaseCaptureError("ordered WP1 run receipt manifest is invalid")
        combined_transitions: list[dict[str, object]] = []
        combined_fills: list[dict[str, object]] = []
        frame_offset = 0
        for run_index, structure in enumerate(run_structures):
            transitions = structure["transitions"]
            fills = structure["fills"]
            assert isinstance(transitions, list) and isinstance(fills, list)
            combined_fills.extend(fills)
            for transition in transitions[0 if run_index == 0 else 1:]:
                assert isinstance(transition, Mapping)
                combined_transitions.append({
                    "segment_index": len(combined_transitions),
                    "start_frame_index": int(transition["start_frame_index"]) + frame_offset,
                    "stop_frame_index": int(transition["stop_frame_index"]) + frame_offset,
                    "frame_count": int(transition["frame_count"]),
                    "decoded_bgr_sha256": ["0" * 64] * int(transition["frame_count"]),
                })
            frame_offset += int(structure["frame_count"])
        try:
            combined_windows = compose_fill_windows(
                combined_transitions, combined_fills
            )
        except RGBRuntimeError as error:
            raise PhaseCaptureError(
                "ordered WP1 source run timing identity is invalid"
            ) from error
        reconstructed_timing_sha256 = _canonical_hash({
            "fill_clips": combined_fills,
            "transition_ranges": [
                {
                    key: row[key]
                    for key in (
                        "segment_index", "start_frame_index", "stop_frame_index",
                        "frame_count",
                    )
                }
                for row in combined_transitions
            ],
            "fill_windows": [
                {
                    "fill_index": window.fill_index,
                    "case_id": window.case_id,
                    "start_frame_index": window.start_frame_index,
                    "stop_frame_index": window.stop_frame_index,
                    "frame_count": window.frame_count,
                }
                for window in combined_windows
            ],
        })
        if reconstructed_timing_sha256 != capture_timing_sha256:
            raise PhaseCaptureError(
                "ordered WP1 source run timing differs from capture plan"
            )
    elif role == "video5_dataset_manifest":
        expected = {
            "schema_version": 1, "artifact_type": _UPSTREAM_ARTIFACT_TYPES[role],
            "status": "pass", "video_sha256": video_sha256,
            "case_ids": case_ids, "case_count": len(case_ids),
        }
        if upstream != expected:
            raise PhaseCaptureError("video5 dataset manifest identity is invalid")
    elif role == "video5_recipe_metadata":
        expected = {
            "schema_version": 1, "artifact_type": _UPSTREAM_ARTIFACT_TYPES[role],
            "status": "pass", "video_sha256": video_sha256,
            "recipe_sequence_sha256": recipe_sequence_sha256,
            "case_ids": case_ids,
        }
        if upstream != expected:
            raise PhaseCaptureError("video5 recipe metadata identity is invalid")
    elif role == "video5_timing_metadata":
        expected = {
            "schema_version": 1, "artifact_type": _UPSTREAM_ARTIFACT_TYPES[role],
            "status": "pass", "video_sha256": video_sha256,
            "capture_timing_sha256": capture_timing_sha256,
            "decoded_frame_count": decoded_frame_count,
        }
        if upstream != expected:
            raise PhaseCaptureError("video5 timing metadata identity is invalid")
    elif role == "video5_transition_window_metadata":
        expected = {
            "schema_version": 1, "artifact_type": _UPSTREAM_ARTIFACT_TYPES[role],
            "status": "pass", "video_sha256": video_sha256,
            "capture_timing_sha256": capture_timing_sha256,
            "case_ids": case_ids,
        }
        if upstream != expected:
            raise PhaseCaptureError("video5 transition-window identity is invalid")
    return upstream_path


def _recipe_case_ids(sequence: object) -> list[str]:
    if not isinstance(sequence, Mapping) or not isinstance(sequence.get("runs"), list):
        raise PhaseCaptureError("recipe sequence is invalid")
    case_ids: list[str] = []
    for run in sequence["runs"]:
        if not isinstance(run, Mapping) or not isinstance(run.get("vials"), list):
            raise PhaseCaptureError("recipe sequence is invalid")
        for vial in run["vials"]:
            case_id = vial.get("case_id") if isinstance(vial, Mapping) else None
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise PhaseCaptureError("recipe case identity/order is invalid")
            case_ids.append(case_id)
    if not case_ids:
        raise PhaseCaptureError("recipe sequence has no cases")
    return case_ids


def _load_plan(path: Path, expected_sha256: str) -> tuple[Path, bytes, dict[str, object]]:
    plan_path, raw = _read_bound_file(path, expected_sha256, "capture plan")
    try:
        plan = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCaptureError("capture plan must be strict UTF-8 JSON") from error
    if not isinstance(plan, dict) or set(plan) != _PLAN_FIELDS:
        raise PhaseCaptureError("capture plan fields are invalid")
    return plan_path, raw, plan


def publish_phase_capture(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    expected_phase: str,
    expected_capture_origin: str | None,
    video_bytes: bytes,
    recipe_sequence: Mapping[str, object],
    decoded_rgb_frames: Sequence[np.ndarray],
    capture_receipt_out: Path,
    provenance_out: Path,
) -> PhaseCapturePublication:
    """Validate one already-claimed phase capture and publish two fresh receipts."""

    capture_path = _fresh(capture_receipt_out, "capture receipt output")
    provenance_path = _fresh(provenance_out, "capture provenance output")
    plan_candidate = Path(plan_path).expanduser().absolute()
    identities = {
        capture_path.resolve(strict=False), provenance_path.resolve(strict=False),
        plan_candidate.resolve(strict=False),
    }
    if len(identities) != 3:
        raise PhaseCaptureError("capture plan and output paths must be distinct")
    plan_file, plan_raw, plan = _load_plan(plan_candidate, expected_plan_sha256)
    capture_origin = plan.get("capture_origin")
    if (
        plan.get("schema_version") != "wp7-phase-capture-plan-v1"
        or plan.get("artifact_type") != "wp7_phase_capture_plan"
        or plan.get("phase") != expected_phase
        or expected_phase not in {"run-test", "run-real-check"}
        or capture_origin not in {"synthetic_camera", "real_camera"}
        or (
            expected_capture_origin is not None
            and capture_origin != expected_capture_origin
        )
    ):
        raise PhaseCaptureError("capture plan phase/origin identity is invalid")
    video_sha256 = hashlib.sha256(video_bytes).hexdigest()
    if _sha(plan.get("video_sha256"), "capture plan video") != video_sha256:
        raise PhaseCaptureError("capture plan video SHA-256 mismatch")
    frames = _decoded_frames(decoded_rgb_frames)

    fills = plan.get("fill_clips")
    transitions = plan.get("encoded_transition_evidence")
    if not isinstance(fills, list) or not isinstance(transitions, list):
        raise PhaseCaptureError("capture timing rows must be lists")
    if len(transitions) != len(fills) + 1:
        raise PhaseCaptureError("capture transition/fill count must be N+1/N")
    case_ids = _recipe_case_ids(recipe_sequence)
    normalized_fills: list[dict[str, object]] = []
    for fill in fills:
        if not isinstance(fill, Mapping) or set(fill) != _FILL_FIELDS:
            raise PhaseCaptureError("capture fill fields are invalid")
        case_id = fill.get("case_id")
        count = _integer(fill.get("frame_count"), "capture fill frame count")
        if not isinstance(case_id, str) or not case_id or count <= 0:
            raise PhaseCaptureError("capture fill identity/count is invalid")
        normalized_fills.append({"case_id": case_id, "frame_count": count})
    if [row["case_id"] for row in normalized_fills] != case_ids:
        raise PhaseCaptureError("capture fill case order differs from recipe order")

    normalized_transitions: list[dict[str, object]] = []
    prior_stop = -1
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or set(transition) != _TRANSITION_FIELDS:
            raise PhaseCaptureError("capture transition fields are invalid")
        start = _integer(transition.get("start_frame_index"), "transition start")
        stop = _integer(transition.get("stop_frame_index"), "transition stop")
        count = _integer(transition.get("frame_count"), "transition frame count")
        if (
            transition.get("segment_index") != index
            or stop <= start
            or stop - start != count
            or stop > len(frames)
            or start < prior_stop
        ):
            raise PhaseCaptureError("capture transition range is invalid")
        computed = [
            hashlib.sha256(frame[..., ::-1].copy().tobytes()).hexdigest()
            for frame in frames[start:stop]
        ]
        attested = transition.get("decoded_bgr_sha256")
        if attested is not None:
            if (
                not isinstance(attested, list)
                or any(_sha(item, "decoded BGR frame") != item for item in attested)
                or attested != computed
            ):
                raise PhaseCaptureError("pre-attested decoded BGR hashes drifted")
        normalized_transitions.append({
            "segment_index": index,
            "start_frame_index": start,
            "stop_frame_index": stop,
            "frame_count": count,
            "decoded_bgr_sha256": computed,
        })
        prior_stop = stop
    if (
        normalized_transitions[0]["start_frame_index"] != 0
        or normalized_transitions[-1]["stop_frame_index"] != len(frames)
    ):
        raise PhaseCaptureError(
            "capture transitions must cover the full decoded sequence"
        )
    try:
        windows = compose_fill_windows(normalized_transitions, normalized_fills)
    except RGBRuntimeError as error:
        raise PhaseCaptureError(str(error)) from error
    if len(windows) != len(normalized_fills):
        raise PhaseCaptureError("capture fill-window denominator is invalid")

    sources = plan.get("source_provenance")
    if not isinstance(sources, list) or not sources:
        raise PhaseCaptureError("capture source provenance must be nonempty")
    supplied_roles = [
        source.get("role") if isinstance(source, Mapping) else None
        for source in sources
    ]
    if (
        tuple(supplied_roles) != _SOURCE_ROLE_ORDER_BY_PHASE[expected_phase]
    ):
        raise PhaseCaptureError(
            "capture source provenance must contain the exact phase role set"
        )
    normalized_sources: list[dict[str, str]] = []
    output_identities = {
        capture_path.resolve(strict=False), provenance_path.resolve(strict=False),
        plan_file.resolve(strict=True),
    }
    roles: set[str] = set()
    upstream_sha_by_role: dict[str, str] = {}
    recipe_sequence_sha256 = _canonical_hash(recipe_sequence)
    capture_timing_sha256 = _canonical_hash({
        "fill_clips": normalized_fills,
        "transition_ranges": [
            {
                key: transition[key]
                for key in (
                    "segment_index", "start_frame_index", "stop_frame_index",
                    "frame_count",
                )
            }
            for transition in normalized_transitions
        ],
        "fill_windows": [
            {
                "fill_index": window.fill_index,
                "case_id": window.case_id,
                "start_frame_index": window.start_frame_index,
                "stop_frame_index": window.stop_frame_index,
                "frame_count": window.frame_count,
            }
            for window in windows
        ],
    })
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
            raise PhaseCaptureError("capture source provenance fields are invalid")
        role = source.get("role")
        if (
            not isinstance(role, str)
            or role not in _SOURCE_ROLES_BY_PHASE[expected_phase]
            or role in roles
        ):
            raise PhaseCaptureError("capture source provenance role is invalid")
        source_value = source.get("path")
        if (
            not isinstance(source_value, str)
            or not Path(source_value).is_absolute()
            or Path(source_value).suffix.lower() != ".json"
        ):
            raise PhaseCaptureError(
                "capture source provenance path must be absolute JSON metadata"
            )
        source_path = Path(source_value)
        if source_path.resolve(strict=False) in output_identities:
            raise PhaseCaptureError("capture source and output paths must be distinct")
        bound_path, source_raw = _read_bound_file(
            source_path, source.get("sha256"), f"capture source {role}"
        )
        try:
            metadata = json.loads(
                source_raw.decode("utf-8"), object_pairs_hook=_strict_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PhaseCaptureError(
                "capture source provenance must be strict JSON metadata"
            ) from error
        if not isinstance(metadata, dict):
            raise PhaseCaptureError(
                "capture source provenance must be JSON object metadata"
            )
        upstream_path = _validate_source_metadata(
            role,
            metadata,
            phase=expected_phase,
            video_sha256=video_sha256,
            recipe_sequence_sha256=recipe_sequence_sha256,
            case_ids=case_ids,
            capture_timing_sha256=capture_timing_sha256,
            decoded_frame_count=len(frames),
            composition_receipt_sha256=upstream_sha_by_role.get(
                "ordered_wp1_composition_receipt"
            ),
        )
        if upstream_path.resolve(strict=True) in output_identities | {
            bound_path.resolve(strict=True)
        }:
            raise PhaseCaptureError(
                "capture source, upstream and output paths must be distinct"
            )
        normalized_sources.append({
            "role": role, "path": str(bound_path), "sha256": str(source["sha256"]),
        })
        roles.add(role)
        upstream_sha_by_role[role] = str(metadata["upstream_sha256"])

    capture: dict[str, object] = {
        "schema_version": "wp7-camera-capture-v1",
        "artifact_type": "wp7_camera_capture",
        "capture_origin": capture_origin,
        "video_sha256": video_sha256,
        "encoded_transition_evidence": normalized_transitions,
        "fill_clips": normalized_fills,
    }
    capture["receipt_sha256"] = _canonical_hash(capture)
    capture_raw_sha256 = hashlib.sha256(_canonical_bytes(capture) + b"\n").hexdigest()
    provenance: dict[str, object] = {
        "schema_version": "wp7-phase-capture-publication-v1",
        "artifact_type": "wp7_phase_capture_publication",
        "status": "published",
        "phase": expected_phase,
        "capture_origin": capture_origin,
        "video_sha256": video_sha256,
        "plan_binding": {
            "path": str(plan_file),
            "raw_file_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "canonical_plan_sha256": _canonical_hash(plan),
        },
        "capture_receipt_binding": {
            "path": str(capture_path),
            "raw_file_sha256": capture_raw_sha256,
            "receipt_sha256": capture["receipt_sha256"],
        },
        "source_provenance": normalized_sources,
    }
    _publish_pair(capture_path, capture, provenance_path, provenance)
    return PhaseCapturePublication(capture, provenance, frames)


def decode_video_rgb_frames(video_bytes: bytes) -> tuple[np.ndarray, ...]:
    """Decode already-claimed video bytes into exact RGB frames."""

    from io import BytesIO
    import av

    frames: list[np.ndarray] = []
    with av.open(BytesIO(video_bytes), mode="r") as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return _decoded_frames(frames)


__all__ = [
    "PhaseCaptureError", "PhaseCapturePublication", "decode_video_rgb_frames",
    "publish_phase_capture",
]
