"""Bind one completed synthetic validation cohort to measurement-only inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
from pathlib import PurePosixPath
import re

from fluidmix.core.io import (
    atomic_write_json,
    hash_regular_file,
    load_strict_json_snapshot,
)


SOURCE_RENDER_PLAN_SHA256 = (
    "98e73854c4817bd3e2ef2dfa0c21bd6c63a6f2d0ac83021365a0ccd665d5296b"
)
VIDEO_TIME_CODES = tuple(sorted((*range(31, 687, 5), 342)))
REQUIRED_STAGE_CODES = (171, 342, 686)
SOURCE_MODEL_INPUT_SHA256 = (
    "8dbba6e29f3d63c9c6270a6804f79203b5da40b4955fc12f1f5d1ec3c88da01a"
)
VALIDATION_FAMILY_ID = "VA"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BATCH_ID = re.compile(r"BATCH_[0-9A-F]{12}\Z")
_STAGE_TIME_CODES = {"STAGE1": 171, "STAGE2": 342, "STAGE3": 686}
_EXPECTED_RUNTIME = {
    "provider": "Brev",
    "instance_name": "claude-isaac-smoke-1024",
    "instance_id": "m1vgz4tbs",
    "machine_type": "g6e.xlarge",
    "gpu": "L40S",
    "machine": "x86_64",
    "runtime_name": "NVIDIA Isaac Sim",
    "isaac_sim_version": "6.0.1.0",
    "environment_profile": "generic",
}
_EXPECTED_DENOMINATORS = {
    "batches": 8,
    "vials_and_render_launches": 24,
    "frames_per_launch": 133,
    "total_frames": 3192,
    "stage_frames_per_launch": 3,
    "stage_frames_total": 72,
    "videos": 24,
    "maximum_process_attempts_per_launch": 1,
}


def _referenced_image_ids(decisions: list[object]) -> set[str]:
    referenced: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(
            decision.get("stages"), list
        ):
            raise ValueError("model-input decision is malformed")
        for stage in decision["stages"]:
            if not isinstance(stage, Mapping):
                raise ValueError("model-input stage is malformed")
            target = stage.get("target_image_id")
            cases = stage.get("reference_cases")
            if not isinstance(target, str) or not isinstance(cases, list):
                raise ValueError("model-input stage image references are malformed")
            referenced.add(target)
            for case in cases:
                if not isinstance(case, Mapping) or not isinstance(
                    case.get("candidate_peers"), list
                ):
                    raise ValueError("model-input reference case is malformed")
                for peer in case["candidate_peers"]:
                    if not isinstance(peer, Mapping) or not isinstance(
                        peer.get("peer_image_id"), str
                    ):
                        raise ValueError("model-input peer image reference is malformed")
                    referenced.add(peer["peer_image_id"])
    return referenced


def write_validation_model_slice(
    source_path: str | os.PathLike[str], output_path: str | os.PathLike[str]
) -> dict[str, object]:
    """Write the allowlisted VA-only model-input slice."""

    payload, snapshot = load_strict_json_snapshot(source_path)
    if snapshot["sha256"] != SOURCE_MODEL_INPUT_SHA256:
        raise ValueError("source model-input plan SHA-256 drifted")
    if not isinstance(payload, Mapping):
        raise ValueError("source model-input plan must be an object")
    if payload.get("schema_version") != 3 or payload.get("artifact_type") != "model_input_plan":
        raise ValueError("source model-input plan schema or artifact type drifted")
    if payload.get("real_data_used") is not False:
        raise ValueError("validation model slice must remain synthetic-only")

    decisions_raw = payload.get("decision_records")
    images_raw = payload.get("image_placeholders")
    attempts_raw = payload.get("cosmos_attempts")
    references_raw = payload.get("reference_batches")
    targets_raw = payload.get("target_batches")
    if not all(
        isinstance(value, list)
        for value in (
            decisions_raw,
            images_raw,
            attempts_raw,
            references_raw,
            targets_raw,
        )
    ):
        raise ValueError("source model-input plan list fields are malformed")
    decisions = [
        dict(row)
        for row in decisions_raw
        if isinstance(row, Mapping) and row.get("family_id") == VALIDATION_FAMILY_ID
    ]
    if len(decisions) != 12:
        raise ValueError("validation slice must contain exactly 12 decisions")
    decision_ids = {row.get("decision_id") for row in decisions}
    if None in decision_ids or len(decision_ids) != 12:
        raise ValueError("validation decision identities are malformed")
    referenced = _referenced_image_ids(decisions)
    images = [
        dict(row)
        for row in images_raw
        if isinstance(row, Mapping) and row.get("image_id") in referenced
    ]
    if len(images) != 72 or {row.get("image_id") for row in images} != referenced:
        raise ValueError("validation slice must contain exactly its 72 referenced images")
    if any(row.get("family_id") != VALIDATION_FAMILY_ID for row in images):
        raise ValueError("validation image reference crossed a recipe family")
    attempts = [
        dict(row)
        for row in attempts_raw
        if isinstance(row, Mapping) and row.get("decision_id") in decision_ids
    ]
    references = [
        dict(row)
        for row in references_raw
        if isinstance(row, Mapping) and row.get("family_id") == VALIDATION_FAMILY_ID
    ]
    targets = [
        dict(row)
        for row in targets_raw
        if isinstance(row, Mapping) and row.get("family_id") == VALIDATION_FAMILY_ID
    ]
    if len(attempts) != 48 or len(references) != 4 or len(targets) != 4:
        raise ValueError("validation slice attempt or batch denominators drifted")
    model_attempts = sum(row.get("kind") == "model" for row in attempts)
    controls = sum(row.get("kind") == "control" for row in attempts)
    if model_attempts != 36 or controls != 12:
        raise ValueError("validation Cosmos/control denominators drifted")

    result = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "decision_records",
            "image_placeholders",
            "cosmos_attempts",
            "reference_batches",
            "target_batches",
            "denominators",
        }
    }
    result.update(
        {
            "slice_family_id": VALIDATION_FAMILY_ID,
            "source_model_input_sha256": snapshot["sha256"],
            "source_model_input_bytes": snapshot["bytes"],
            "decision_records": decisions,
            "image_placeholders": images,
            "cosmos_attempts": attempts,
            "reference_batches": references,
            "target_batches": targets,
            "denominators": {
                "cosmos_model_calls_scored": model_attempts,
                "multi_image_smoke_unscored": 1,
                "reference_availability_target_stage_records": sum(
                    len(stage["reference_cases"])
                    for decision in decisions
                    for stage in decision["stages"]
                ),
                "scored_attempts_total": len(attempts),
                "scored_target_decisions": len(decisions),
                "template_controls_scored": controls,
            },
        }
    )
    output = Path(output_path)
    atomic_write_json(output.parent, output.name, result)
    return result


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _list(value: object, role: str, length: int | None = None) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" with exactly {length} rows"
        raise ValueError(f"{role} must be a list{suffix}")
    return value


def _check_sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return value


def _check_snapshot(path: Path, expected_sha: object, role: str) -> dict[str, object]:
    expected = _check_sha(expected_sha, f"{role} hash")
    snapshot = hash_regular_file(path)
    if snapshot["sha256"] != expected:
        raise ValueError(f"{role} hash drifted")
    return snapshot


def _relative_artifact(
    capture: Path,
    record: object,
    *,
    kind: str,
    time_code: int,
) -> tuple[Path, dict[str, object]]:
    row = _mapping(record, f"capture {kind} artifact")
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"capture {kind} artifact fields are malformed")
    relative_text = row.get("path")
    if not isinstance(relative_text, str) or "\\" in relative_text:
        raise ValueError(f"capture {kind} artifact path is malformed")
    relative = PurePosixPath(relative_text)
    expected = PurePosixPath(kind, f"{time_code:04d}.png")
    if relative != expected:
        raise ValueError(f"capture {kind} artifact path drifted")
    path = capture.joinpath(*relative.parts)
    snapshot = _check_snapshot(path, row.get("sha256"), f"capture {kind} artifact")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_relative_to(capture):
        raise ValueError(f"capture {kind} artifact escaped through a symlink")
    return resolved, snapshot


def _validate_model_slice(
    payload: object, snapshot: Mapping[str, object]
) -> list[dict[str, object]]:
    plan = _mapping(payload, "validation model-input slice")
    if plan.get("schema_version") != 3 or plan.get("artifact_type") != "model_input_plan":
        raise ValueError("validation model-input slice schema or type drifted")
    if plan.get("slice_family_id") != VALIDATION_FAMILY_ID:
        raise ValueError("validation model-input slice family drifted")
    if plan.get("source_model_input_sha256") != SOURCE_MODEL_INPUT_SHA256:
        raise ValueError("validation model-input slice source binding drifted")
    if plan.get("real_data_used") is not False:
        raise ValueError("validation image binding is synthetic-only")
    decisions = _list(plan.get("decision_records"), "VA decisions", 12)
    referenced = _referenced_image_ids(decisions)
    images_raw = _list(plan.get("image_placeholders"), "VA image placeholders", 72)
    images = [dict(_mapping(row, "VA image placeholder")) for row in images_raw]
    image_ids = [row.get("image_id") for row in images]
    if any(not isinstance(image_id, str) for image_id in image_ids):
        raise ValueError("VA image placeholder IDs must be strings")
    if len(set(image_ids)) != 72 or set(image_ids) != referenced:
        raise ValueError("VA image placeholders differ from exact decision references")
    for row in images:
        image_id = row.get("image_id")
        batch_id = row.get("batch_id")
        vial_id = row.get("vial_id")
        stage_id = row.get("stage_id")
        if (
            row.get("family_id") != VALIDATION_FAMILY_ID
            or row.get("partition") != "val"
            or row.get("render_status") != "unrendered"
            or row.get("image_placeholder") != f"unrendered:{image_id}"
            or stage_id not in _STAGE_TIME_CODES
            or row.get("time_code") != _STAGE_TIME_CODES.get(stage_id)
            or not isinstance(batch_id, str)
            or _BATCH_ID.fullmatch(batch_id) is None
            or vial_id not in {"V1", "V2", "V3"}
            or image_id != f"{batch_id}-{vial_id}-{stage_id}"
        ):
            raise ValueError("VA image placeholder identity or stage schedule drifted")
    _check_sha(snapshot.get("sha256"), "validation model-input slice snapshot")
    return images


def _validate_graph(
    payload: object,
    *,
    root: Path,
    launch_ids: list[str],
) -> tuple[list[Mapping[str, object]], Path]:
    graph = _mapping(payload, "cohort graph")
    if (
        graph.get("schema_version") != 1
        or graph.get("artifact_type") != "brev_validation_video_cohort"
        or graph.get("status") != "materialized_unrendered"
    ):
        raise ValueError("cohort graph schema, type, or materialized status drifted")
    if graph.get("source_render_plan_sha256") != SOURCE_RENDER_PLAN_SHA256:
        raise ValueError("cohort source render-plan binding drifted")
    runtime = _mapping(graph.get("render_runtime"), "cohort render runtime")
    if any(runtime.get(key) != value for key, value in _EXPECTED_RUNTIME.items()):
        raise ValueError("cohort is not the frozen Brev Isaac 6.0.1 graph")
    isaac_python = runtime.get("isaac_python")
    if not isinstance(isaac_python, str) or not Path(isaac_python).is_absolute():
        raise ValueError("cohort Isaac Python binding is malformed")
    if graph.get("runtime_mixing_policy") != (
        "whole cohort on one Brev host and Isaac 6.0.1; never pool RGB/Lab "
        "with Isaac 6.2 cohorts"
    ):
        raise ValueError("cohort runtime-mixing policy drifted")
    if graph.get("denominators") != _EXPECTED_DENOMINATORS:
        raise ValueError("cohort render denominators drifted")
    family = _mapping(graph.get("family"), "cohort family")
    if family != {
        "family_id": "VA",
        "partition": "val",
        "previous_sim_recipe_id": "S0160",
        "current_sim_recipe_id": "S0380",
        "synthetic_onset_id": "after_t400",
    }:
        raise ValueError("cohort VA family identity drifted")
    controls = _mapping(graph.get("render_controls"), "cohort render controls")
    if (
        controls.get("camera") != "FillCam"
        or controls.get("view_id") != "vial_fill"
        or controls.get("renderer") != "PathTracing"
        or controls.get("resolution") != [640, 480]
        or controls.get("time_codes") != list(VIDEO_TIME_CODES)
        or controls.get("frame_spec")
        != ",".join(str(code) for code in VIDEO_TIME_CODES)
        or controls.get("stage_time_codes") != list(REQUIRED_STAGE_CODES)
        or controls.get("subframes") != 64
        or controls.get("warmup") != 60
        or controls.get("max_tries") != 20
        or controls.get("timeout_seconds") != 1800
        or controls.get("require_liquid_pixels") is not True
    ):
        raise ValueError("cohort render controls drifted")
    encoding = _mapping(graph.get("video_encoding"), "cohort video encoding")
    if (
        encoding.get("codec_request") != "mp4v"
        or encoding.get("container") != "mp4"
        or encoding.get("fps") != 6.0
        or encoding.get("expected_decoded_frames") != 133
        or encoding.get("use") != "lossy_review_visualization_only"
        or encoding.get("measurement_source")
        != "hash_bound_capture_png_and_liquid_mask_only"
    ):
        raise ValueError("cohort review-video policy drifted")
    if graph.get("process_policy") != {
        "max_process_attempts_per_launch": 1,
        "replacement_attempts": 0,
        "cohort_stop": "first_failure",
    }:
        raise ValueError("cohort process policy drifted")
    launches = [
        _mapping(row, "cohort launch")
        for row in _list(graph.get("launches"), "cohort launches", 24)
    ]
    if [row.get("ordinal") for row in launches] != list(range(1, 25)):
        raise ValueError("cohort launch ordinals drifted")
    if [row.get("launch_id") for row in launches] != launch_ids:
        raise ValueError("cohort launches differ from the VA image placeholders")
    first_capture = launches[0].get("capture_root")
    if not isinstance(first_capture, str) or not Path(first_capture).is_absolute():
        raise ValueError("cohort capture root binding is malformed")
    original_root = Path(first_capture).parent.parent
    for row in launches:
        launch_id = str(row["launch_id"])
        capture_text = row.get("capture_root")
        video_text = row.get("video_output_dir")
        if (
            capture_text != str(original_root / launch_id / "capture")
            or video_text != str(original_root / "videos" / launch_id)
            or row.get("family_id") != "VA"
            or row.get("partition") != "val"
            or launch_id != f"{row.get('batch_id')}-{row.get('vial_id')}"
            or row.get("max_process_attempts") != 1
            or row.get("replacement_attempts") != 0
        ):
            raise ValueError("cohort launch artifact layout drifted")
        copied_capture = root / launch_id / "capture"
        copied_video = root / "videos" / launch_id
        if not copied_capture.is_dir() or copied_capture.resolve(strict=True) != copied_capture:
            raise ValueError("copied cohort capture directory is missing")
        if not copied_video.is_dir() or copied_video.resolve(strict=True) != copied_video:
            raise ValueError("copied cohort video directory is missing")
    return launches, original_root


def _validate_completed_run(
    payload: object,
    *,
    cohort_snapshot: Mapping[str, object],
    original_root: Path,
    launch_ids: list[str],
) -> list[Mapping[str, object]]:
    run = _mapping(payload, "cohort run")
    if (
        run.get("schema_version") != 1
        or run.get("status") != "completed"
        or run.get("expected_launches") != 24
        or run.get("attempted_launches") != 24
        or run.get("completed_launches") != 24
        or run.get("failed_launch_id") is not None
        or run.get("replacement_attempts") != 0
        or run.get("stop_policy") != "first_failure"
    ):
        raise ValueError("cohort run is not a completed 24/24 zero-replacement run")
    cohort = _mapping(run.get("cohort"), "cohort-run graph binding")
    if (
        cohort.get("path") != str(original_root / "cohort.json")
        or cohort.get("sha256") != cohort_snapshot.get("sha256")
        or cohort.get("bytes") != cohort_snapshot.get("bytes")
    ):
        raise ValueError("cohort-run graph hash binding drifted")
    rows = [
        _mapping(row, "cohort-run row")
        for row in _list(run.get("rows"), "cohort-run rows", 24)
    ]
    if [row.get("ordinal") for row in rows] != list(range(1, 25)):
        raise ValueError("cohort-run row ordinals drifted")
    if [row.get("launch_id") for row in rows] != launch_ids:
        raise ValueError("cohort-run launches differ from the VA cohort")
    for row in rows:
        render = _mapping(row.get("render"), "cohort-run render gate")
        encoder = _mapping(row.get("encoder"), "cohort-run encoder gate")
        if (
            row.get("outcome") != "passed"
            or row.get("gate_error") not in {None, ""}
            or render.get("returncode") != 0
            or encoder.get("returncode") != 0
        ):
            raise ValueError("a cohort attempt, capture, or video gate did not pass")
    return rows


def _validate_capture(
    *,
    root: Path,
    original_root: Path,
    launch: Mapping[str, object],
    run_row: Mapping[str, object],
    images: list[dict[str, object]],
    capture_summary_verifier: Callable[[object, object], Mapping[str, object]],
) -> tuple[dict[str, dict[str, str]], list[tuple[Path, dict[str, object]]], str]:
    launch_id = str(launch["launch_id"])
    capture = root / launch_id / "capture"
    attempt_path = root / launch_id / "attempt.json"
    attempt, attempt_snapshot = load_strict_json_snapshot(attempt_path)
    attempt = _mapping(attempt, "render attempt")
    if (
        attempt.get("schema_version") != 1
        or attempt.get("run_id") != launch_id
        or attempt.get("status") != "mechanical_gates_passed"
        or attempt.get("failure_reasons") != []
        or attempt.get("visual_check_status") != "pending"
    ):
        raise ValueError("render attempt gate did not pass")
    capture_record = _mapping(attempt.get("capture"), "render attempt capture gate")
    if (
        capture_record.get("root") != str(original_root / launch_id / "capture")
        or capture_record.get("verified") is not True
        or capture_record.get("appearance_provenance_verified") is not True
    ):
        raise ValueError("render attempt capture gate did not pass")
    if (
        "attempt_sha256" in run_row
        and run_row.get("attempt_sha256") != attempt_snapshot["sha256"]
    ):
        raise ValueError("render attempt hash drifted")

    summary_path = capture / "summary.json"
    summary, summary_snapshot = load_strict_json_snapshot(summary_path)
    summary = _mapping(summary, "capture summary")
    verified_capture = capture_summary_verifier(capture, None)
    if (
        verified_capture.get("summary") != dict(summary)
        or verified_capture.get("summary_sha256") != summary_snapshot["sha256"]
    ):
        raise ValueError("capture changed between strict load and full-tree verification")
    if capture_record.get("inner_summary_sha256") != summary_snapshot["sha256"]:
        raise ValueError("capture summary hash drifted from the verified attempt")
    settings = _mapping(summary.get("render_settings"), "capture render settings")
    frame_spec = ",".join(str(code) for code in VIDEO_TIME_CODES)
    if (
        summary.get("schema_version") != 1
        or summary.get("mechanical_gates_passed") is not True
        or summary.get("visual_check_status") != "pending"
        or summary.get("frame_count") != 133
        or settings.get("resolution") != [640, 480]
        or settings.get("time_codes") != list(VIDEO_TIME_CODES)
        or settings.get("frame_spec") != frame_spec
    ):
        raise ValueError("capture summary gate or frozen frame schedule drifted")
    frames = [
        _mapping(row, "capture frame")
        for row in _list(summary.get("frames"), "capture frames", 133)
    ]
    if [row.get("capture_ordinal") for row in frames] != list(range(1, 134)):
        raise ValueError("capture frame ordinals drifted")
    if [row.get("time_code") for row in frames] != list(VIDEO_TIME_CODES):
        raise ValueError("capture frame time-code order drifted")

    by_time = {int(row["time_code"]): row for row in frames}
    locators: dict[str, dict[str, str]] = {}
    snapshots: list[tuple[Path, dict[str, object]]] = [
        (attempt_path, attempt_snapshot),
        (summary_path, summary_snapshot),
    ]
    for image in images:
        time_code = int(image["time_code"])
        artifacts = _mapping(by_time[time_code].get("artifacts"), "stage frame artifacts")
        rgb_path, rgb_snapshot = _relative_artifact(
            capture, artifacts.get("rgb"), kind="rgb", time_code=time_code
        )
        mask_path, mask_snapshot = _relative_artifact(
            capture, artifacts.get("liquid"), kind="liquid", time_code=time_code
        )
        image_id = str(image["image_id"])
        locators[image_id] = {
            "rgb_path": str(rgb_path),
            "rgb_sha256": str(rgb_snapshot["sha256"]),
            "mask_path": str(mask_path),
            "mask_sha256": str(mask_snapshot["sha256"]),
        }
        snapshots.extend(((rgb_path, rgb_snapshot), (mask_path, mask_snapshot)))
    return locators, snapshots, str(summary_snapshot["sha256"])


def _validate_video(
    *,
    root: Path,
    original_root: Path,
    launch_id: str,
    run_row: Mapping[str, object],
    summary_sha256: str,
    stage_rgb_hashes: Mapping[int, str],
) -> list[tuple[Path, dict[str, object]]]:
    video_dir = root / "videos" / launch_id
    record_path = video_dir / "video.json"
    record, record_snapshot = load_strict_json_snapshot(record_path)
    record = _mapping(record, "video record")
    if record.get("schema_version") != 1 or record.get("status") != "encoded_verified":
        raise ValueError("video gate did not pass")
    capture = _mapping(record.get("capture"), "video capture binding")
    if (
        capture.get("root") != str(original_root / launch_id / "capture")
        or capture.get("summary_sha256") != summary_sha256
    ):
        raise ValueError("video is not bound to the same verified capture")
    video = _mapping(record.get("video"), "video artifact")
    if video.get("path") != "sequence.mp4":
        raise ValueError("video artifact path drifted")
    video_path = video_dir / "sequence.mp4"
    video_snapshot = _check_snapshot(video_path, video.get("sha256"), "video artifact")
    if video.get("bytes") != video_snapshot["bytes"]:
        raise ValueError("video artifact byte count drifted")
    encoding = _mapping(record.get("encoding"), "video encoding")
    if (
        encoding.get("codec_requested") != "mp4v"
        or encoding.get("container") != "mp4"
        or encoding.get("fps") != 6.0
        or encoding.get("width") != 640
        or encoding.get("height") != 480
    ):
        raise ValueError("video encoding gate drifted")
    verification = _mapping(record.get("verification"), "video verification gate")
    reported_fps = verification.get("reported_fps")
    if (
        verification.get("decoded_frame_count") != 133
        or verification.get("reported_fourcc") not in {"FMP4", "mp4v", "MP4V"}
        or isinstance(reported_fps, bool)
        or not isinstance(reported_fps, (int, float))
        or abs(float(reported_fps) - 6.0) > 0.01
        or verification.get("reported_width") != 640
        or verification.get("reported_height") != 480
    ):
        raise ValueError("video verification gate drifted")
    limits = _mapping(record.get("limits"), "video claim limits")
    if (
        limits.get("measurement_source") != "hash-bound capture PNGs only"
        or limits.get("video_role") != "lossy review visualization only"
    ):
        raise ValueError("video measurement/review role drifted")
    if record.get("time_codes") != list(VIDEO_TIME_CODES):
        raise ValueError("video time-code schedule drifted")
    stages = [
        _mapping(row, "video stage frame")
        for row in _list(record.get("stage_frames"), "video stage frames", 3)
    ]
    expected_stages = [
        {
            "capture_ordinal": list(VIDEO_TIME_CODES).index(code) + 1,
            "time_code": code,
            "rgb_sha256": stage_rgb_hashes[code],
        }
        for code in REQUIRED_STAGE_CODES
    ]
    if stages != expected_stages:
        raise ValueError("video stage frames are not the same capture's bound RGB frames")
    if (
        "video_record_sha256" in run_row
        and run_row.get("video_record_sha256") != record_snapshot["sha256"]
    ):
        raise ValueError("video record hash drifted")
    return [(record_path, record_snapshot), (video_path, video_snapshot)]


def _write_validation_image_locators_with_twin_verifier(
    sliced_model_path: str | os.PathLike[str],
    cohort_root: str | os.PathLike[str],
    locator_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    capture_summary_verifier: Callable[[object, object], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Write SHA-bound VA RGB/liquid-mask locators and a separate receipt.

    MP4 bytes are rehashed only to require the already-recorded review-video
    gate. They are never decoded or exposed as quantitative measurement input.
    """

    if not callable(capture_summary_verifier):
        raise TypeError("capture_summary_verifier must be callable")
    locator_output = Path(locator_path)
    receipt_output = Path(receipt_path)
    if locator_output.absolute() == receipt_output.absolute():
        raise ValueError("locator and receipt outputs must be different files")
    for output in (locator_output, receipt_output):
        if os.path.lexists(output):
            raise FileExistsError(f"validation binding output already exists: {output}")

    root = Path(cohort_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("cohort root must be a directory")
    model, model_snapshot = load_strict_json_snapshot(sliced_model_path)
    images = _validate_model_slice(model, model_snapshot)
    launch_ids = list(
        dict.fromkeys(f"{row['batch_id']}-{row['vial_id']}" for row in images)
    )
    if len(launch_ids) != 24:
        raise ValueError("VA image placeholders must map to exactly 24 captures")

    cohort_path = root / "cohort.json"
    run_path = root / "cohort-run.json"
    graph, graph_snapshot = load_strict_json_snapshot(cohort_path)
    launches, original_root = _validate_graph(
        graph, root=root, launch_ids=launch_ids
    )
    run, run_snapshot = load_strict_json_snapshot(run_path)
    run_rows = _validate_completed_run(
        run,
        cohort_snapshot=graph_snapshot,
        original_root=original_root,
        launch_ids=launch_ids,
    )

    images_by_launch: dict[str, list[dict[str, object]]] = {
        launch_id: [] for launch_id in launch_ids
    }
    for row in images:
        images_by_launch[f"{row['batch_id']}-{row['vial_id']}"].append(row)
    all_locators: dict[str, dict[str, str]] = {}
    input_snapshots: list[tuple[Path, Mapping[str, object]]] = [
        (Path(sliced_model_path), model_snapshot),
        (cohort_path, graph_snapshot),
        (run_path, run_snapshot),
    ]
    for launch, run_row in zip(launches, run_rows, strict=True):
        launch_id = str(launch["launch_id"])
        launch_images = images_by_launch[launch_id]
        if sorted(row["time_code"] for row in launch_images) != list(REQUIRED_STAGE_CODES):
            raise ValueError("each VA capture must supply the three frozen stage images")
        locators, capture_snapshots, summary_sha = _validate_capture(
            root=root,
            original_root=original_root,
            launch=launch,
            run_row=run_row,
            images=launch_images,
            capture_summary_verifier=capture_summary_verifier,
        )
        all_locators.update(locators)
        input_snapshots.extend(capture_snapshots)
        input_snapshots.extend(
            _validate_video(
                root=root,
                original_root=original_root,
                launch_id=launch_id,
                run_row=run_row,
                summary_sha256=summary_sha,
                stage_rgb_hashes={
                    int(row["time_code"]): locators[str(row["image_id"])][
                        "rgb_sha256"
                    ]
                    for row in launch_images
                },
            )
        )
    if set(all_locators) != {str(row["image_id"]) for row in images}:
        raise ValueError("bound image locators differ from exact VA image placeholders")

    for path, expected in input_snapshots:
        if hash_regular_file(path) != dict(expected):
            raise RuntimeError(f"validation binding input changed during verification: {path}")

    locator: dict[str, object] = {
        "schema_version": 1,
        "data_origin": "twin",
        "images": {key: all_locators[key] for key in sorted(all_locators)},
    }
    atomic_write_json(locator_output.parent, locator_output.name, locator)
    locator_snapshot = hash_regular_file(locator_output)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "validation_image_binding_receipt",
        "status": "bound",
        "family_id": VALIDATION_FAMILY_ID,
        "partition": "val",
        "data_origin": "synthetic_twin",
        "real_data_used": False,
        "source_sliced_model_input_sha256": model_snapshot["sha256"],
        "source_cohort_graph_sha256": graph_snapshot["sha256"],
        "source_cohort_run_sha256": run_snapshot["sha256"],
        "image_locator_sha256": locator_snapshot["sha256"],
        "measurement_source": "hash_bound_capture_rgb_and_liquid_mask_pngs",
        "capture_gate_verification": "full_summary_and_artifact_tree_rehashed",
        "video_gate_verification": (
            "recorded_decode_gate_and_mp4_hash_reverified_without_measurement_decode"
        ),
        "review_video_role": "verified_lossy_visualization_not_measurement",
        "stage_time_codes": list(REQUIRED_STAGE_CODES),
        "render_runtime": dict(_mapping(graph, "cohort graph")["render_runtime"]),
        "denominators": {
            "batches": 8,
            "captures": 24,
            "stage_images": 72,
            "verified_videos_not_used_for_measurement": 24,
        },
        "claim_boundary": (
            "Synthetic twin pixels only; no real data, decoded-MP4 measurement, "
            "physical carryover, defect, plant-action, or sim-to-real claim."
        ),
    }
    atomic_write_json(receipt_output.parent, receipt_output.name, receipt)
    return {"locator": locator, "receipt": receipt}


__all__ = [
    "SOURCE_MODEL_INPUT_SHA256",
    "write_validation_model_slice",
]
