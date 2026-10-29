"""Safe evaluation-only projection of accepted V3 fill-clip inputs.

Accepted renderer receipts remain evaluation provenance. This module checks their
publication graph and emits only fields needed by the Part-B clip operator. It
never serializes source-step arrays or mask paths.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

import cv2


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BATCH = re.compile(r"BATCH_[0-9a-f]{16}\Z")
_CASE = re.compile(r"CASE_[0-9a-f]{16}\Z")
_FORBIDDEN_KEY_FRAGMENTS = (
    "composition", "label", "mask", "oracle", "palette", "particle",
    "preset", "residu", "semantic",
)
_RECEIPT_REQUIRED = {
    "schema_version": 2,
    "artifact_type": "f2_varied_v3_downstream_validation_receipt",
    "status": "pass",
    "batch_count": 6,
    "vial_count": 18,
    "stage_pair_count": 42,
    "source_artifact_validation_count": 18,
}
_SPLIT_BATCH_COUNTS = {"train": 4, "validation": 1, "evaluation": 1}
_SPLIT_VIAL_COUNTS = {"train": 12, "validation": 3, "evaluation": 3}


class F2FillClipInputError(ValueError):
    """The accepted-receipt projection failed closed."""


def _fail(code: str, error: BaseException | None = None) -> F2FillClipInputError:
    failure = F2FillClipInputError(code)
    if error is not None:
        failure.__cause__ = error
    return failure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, code: str) -> tuple[Mapping[str, object], str]:
    lexical = path.expanduser().absolute()
    try:
        resolved = lexical.resolve(strict=True)
        if resolved != lexical or not lexical.is_file() or lexical.is_symlink():
            raise ValueError("path is not an exact regular file")
        raw = lexical.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _fail(code, error)
    if not isinstance(value, Mapping):
        raise _fail(code)
    return value, hashlib.sha256(raw).hexdigest()


def _bound_file(
    root: Path,
    relative: object,
    expected_sha: object,
    expected_bytes: object | None,
    code: str,
) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_sha, str)
        or _SHA256.fullmatch(expected_sha) is None
    ):
        raise _fail(code)
    lexical = (root / relative).absolute()
    try:
        resolved_root = root.resolve(strict=True)
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise _fail(code, error)
    if (
        resolved != lexical
        or not resolved.is_relative_to(resolved_root)
        or not lexical.is_file()
        or lexical.is_symlink()
        or (expected_bytes is not None and lexical.stat().st_size != expected_bytes)
        or _sha256(lexical) != expected_sha
    ):
        raise _fail(code)
    return lexical


def _absolute_bound_file(path_value: object, expected_sha: object, code: str) -> Path:
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(expected_sha, str)
        or _SHA256.fullmatch(expected_sha) is None
    ):
        raise _fail(code)
    path = Path(path_value).absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _fail(code, error)
    if resolved != path or not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha:
        raise _fail(code)
    return path


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise _fail(code)
    return value


def _official_artifact_validation(app_output: Path) -> None:
    """Invoke the accepted public validator CLI without importing Part-A modules."""

    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_physx_f2_artifacts.py"
    try:
        completed = subprocess.run(
            [sys.executable, os.fspath(script), os.fspath(app_output)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise _fail("artifact_validator_unavailable", error)
    if (
        completed.returncode != 0
        or not isinstance(result, Mapping)
        or result.get("status") != "pass"
        or result.get("failures") != []
    ):
        raise _fail("artifact_validation_failed")


def _validate_video_metadata(path: Path, render: Mapping[str, object]) -> None:
    expected_count = render.get("video_decoded_frame_count")
    if (
        render.get("resolution") != [640, 480]
        or render.get("video_playback_fps") != 10.0
        or render.get("video_codec") != "MJPG"
        or render.get("video_container") != "avi"
        or render.get("video_decodable") is not True
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count <= 0
    ):
        raise _fail("video_contract_invalid")
    capture = cv2.VideoCapture(os.fspath(path))
    try:
        if not capture.isOpened():
            raise _fail("video_decode_invalid")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        decoded_count = 0
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            if frame is None or frame.shape != (480, 640, 3) or frame.dtype.name != "uint8":
                raise _fail("video_decode_mismatch")
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count != expected_count:
        raise _fail("video_decode_mismatch")
    if not math.isclose(fps, 10.0, rel_tol=0.0, abs_tol=1e-9) or fourcc != "MJPG":
        raise _fail("video_metadata_mismatch")


def _active_fill_range(render: Mapping[str, object], timing: Mapping[str, object]) -> tuple[int, int]:
    count = render.get("video_decoded_frame_count")
    steps = render.get("video_timing_source_frame_steps")
    indices = render.get("video_frame_source_capture_indices")
    repeats = render.get("video_frame_repeat_counts")
    fill_end = timing.get("fill_end_step")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or not isinstance(fill_end, int)
        or isinstance(fill_end, bool)
        or not isinstance(steps, list)
        or len(steps) != count
        or any(not isinstance(step, int) or isinstance(step, bool) for step in steps)
        or any(left >= right for left, right in zip(steps, steps[1:]))
        or indices != list(range(count))
        or repeats != [1] * count
    ):
        raise _fail("video_timing_invalid")
    end = bisect_right(steps, fill_end)
    if (
        not 0 < end < count
        or any(step > fill_end for step in steps[:end])
        or any(step <= fill_end for step in steps[end:])
    ):
        raise _fail("active_fill_range_invalid")
    return end, count


def _assert_safe_projection(value: object) -> None:
    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise _fail("output_field_invalid")
                normalized = key.lower()
                if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                    raise _fail(f"forbidden_output_field:{key}")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def _publish_json_no_clobber(output: Path, payload: object) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"fill-clip input output already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise _fail("output_parent_invalid")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output


def _validate_requested_model_record(
    request: Mapping[str, object], vial: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    plan = _mapping(receipt.get("plan"), "receipt_plan_invalid")
    schedule = _mapping(plan.get("execution_schedule"), "execution_schedule_invalid")
    programmed = _mapping(request.get("programmed_recipe"), "programmed_recipe_invalid")
    expected = {
        "batch_id": request.get("batch_id"),
        "case_id": vial.get("case_id"),
        "current_programmed_ml": programmed.get("current_ml_by_tank"),
        "current_recipe_id": request.get("current_recipe_id"),
        "data_origin": request.get("data_origin"),
        "previous_programmed_ml": vial.get("previous_ml_by_tank"),
        "previous_recipe_id": vial.get("previous_recipe_id"),
        "programmed_dispense_order": programmed.get("dispense_order"),
        "quantitative_evidence_ids": [],
        "run_id": request.get("run_id"),
        "split_role": request.get("split_role"),
        "split_version": request.get("split_version"),
        "stage_image_evidence_ids": [],
        "vial_position": vial.get("vial_position"),
    }
    if schedule.get("model_input_record") != expected:
        raise _fail("model_input_binding_mismatch")


def _validate_pours(
    request: Mapping[str, object], receipt: Mapping[str, object], events: Sequence[object]
) -> None:
    plan = _mapping(receipt.get("plan"), "receipt_plan_invalid")
    pours = _sequence(plan.get("pours"), "pours_invalid")
    timing = _mapping(receipt.get("timing"), "timing_invalid")
    programmed = _mapping(request.get("programmed_recipe"), "programmed_recipe_invalid")
    order = programmed.get("dispense_order")
    if not isinstance(order, list) or len(order) not in {2, 3} or len(pours) != len(order):
        raise _fail("pour_order_invalid")
    ends: list[int] = []
    previous_end = 0
    for ordinal, (value, tank_id, event_value) in enumerate(
        zip(pours, order, events, strict=True), start=1
    ):
        pour = _mapping(value, "pour_invalid")
        event = _mapping(event_value, "stage_event_invalid")
        start = pour.get("start_step")
        end = pour.get("end_step")
        if (
            pour.get("ordinal") != ordinal
            or pour.get("tank_id") != tank_id
            or start != previous_end + 1
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or isinstance(start, bool)
            or end < start
            or event.get("stage_ordinal") != ordinal
            or event.get("tank_order") != ordinal
            or event.get("tank_id") != tank_id
            or event.get("event_id") != f"dispense-{ordinal}-{tank_id}"
            or event.get("physics_step") != pour.get("stage_capture_step")
            or event.get("frame_id") is not None
        ):
            raise _fail("pour_stage_binding_mismatch")
        ends.append(end)
        previous_end = end
    if timing.get("pour_end_steps") != ends or timing.get("fill_end_step") != ends[-1]:
        raise _fail("fill_end_binding_mismatch")


def _stage_records(
    request: Mapping[str, object],
    vial: Mapping[str, object],
    capture: Mapping[str, object],
    capture_sha256: str,
    locators: Mapping[str, object],
) -> list[dict[str, object]]:
    events = _sequence(capture.get("stage_events"), "stage_events_invalid")
    rows = _sequence(capture.get("rows"), "capture_rows_invalid")
    vial_id = vial.get("vial_id")
    vial_rows = [row for row in rows if isinstance(row, Mapping) and row.get("vial_id") == vial_id]
    if len(vial_rows) != len(events):
        raise _fail("capture_stage_denominator_invalid")
    by_event = {row.get("event_id"): row for row in vial_rows}
    result: list[dict[str, object]] = []
    for event_value in events:
        event = _mapping(event_value, "stage_event_invalid")
        stage_id = event.get("event_id")
        capture_row = _mapping(by_event.get(stage_id), "capture_stage_missing")
        image_id = f"{request['batch_id']}-{vial_id}-{stage_id}"
        locator = _mapping(locators.get(image_id), "image_locator_missing")
        expected_locator = {
            "batch_id": request.get("batch_id"),
            "run_id": request.get("run_id"),
            "recipe_id": request.get("current_recipe_id"),
            "split_role": request.get("split_role"),
            "split_version": request.get("split_version"),
            "vial_id": vial_id,
            "stage_id": stage_id,
            "stage_ordinal": event.get("stage_ordinal"),
            "tank_id": event.get("tank_id"),
            "source_manifest_sha256": capture_sha256,
        }
        if any(locator.get(key) != value for key, value in expected_locator.items()):
            raise _fail("image_locator_binding_mismatch")
        rgb = _mapping(capture_row.get("rgb"), "capture_rgb_invalid")
        mask = _mapping(capture_row.get("liquid_mask"), "capture_mask_invalid")
        if (
            capture_row.get("verified") is not True
            or capture_row.get("physics_step") != event.get("physics_step")
            or locator.get("rgb_path") != rgb.get("path")
            or locator.get("rgb_sha256") != rgb.get("sha256")
            or locator.get("mask_path") != mask.get("path")
            or locator.get("mask_sha256") != mask.get("sha256")
        ):
            raise _fail("capture_locator_mismatch")
        _absolute_bound_file(locator.get("rgb_path"), locator.get("rgb_sha256"), "rgb_invalid")
        _absolute_bound_file(locator.get("mask_path"), locator.get("mask_sha256"), "mask_invalid")
        result.append(
            {
                "image_id": image_id,
                "stage_id": stage_id,
                "stage_ordinal": event.get("stage_ordinal"),
                "tank_id": event.get("tank_id"),
                "physics_step": event.get("physics_step"),
            }
        )
    return result


def build_f2_fill_clip_input_manifest(
    *,
    downstream_validation_receipt: str | Path,
    image_locators: str | Path,
    output_path: str | Path,
) -> Path:
    """Project one safe active-fill input from every accepted V3 case."""

    receipt_path = Path(downstream_validation_receipt).expanduser().absolute()
    downstream, downstream_sha = _load_json(receipt_path, "downstream_receipt_invalid")
    if any(downstream.get(key) != value for key, value in _RECEIPT_REQUIRED.items()):
        raise _fail("downstream_receipt_invalid")
    if downstream.get("split_batch_counts") != _SPLIT_BATCH_COUNTS:
        raise _fail("downstream_split_counts_invalid")
    root = receipt_path.parent
    listed = _sequence(downstream.get("files"), "downstream_file_list_invalid")
    if len(listed) != 31:
        raise _fail("downstream_file_denominator_invalid")
    listed_paths: dict[str, Mapping[str, object]] = {}
    for item in listed:
        row = _mapping(item, "downstream_file_invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in listed_paths:
            raise _fail("downstream_file_duplicate")
        _bound_file(root, relative, row.get("sha256"), row.get("bytes"), "downstream_file_invalid")
        listed_paths[relative] = row
    for map_key, file_name in (
        ("capture_manifest_sha256_by_batch", "f2-stage-capture-manifest.json"),
        ("publication_receipt_sha256_by_batch", "publication-receipt.json"),
        ("batch_context_sha256_by_batch", "stage-batch-context.json"),
        ("runtime_receipt_sha256_by_batch", "f2-runtime-acceptance-receipt.json"),
    ):
        hashes = _mapping(downstream.get(map_key), "downstream_hash_map_invalid")
        if len(hashes) != 6:
            raise _fail("downstream_hash_map_invalid")
        for batch_id, expected_sha in hashes.items():
            row = listed_paths.get(f"publications/{batch_id}/{file_name}")
            if row is None or row.get("sha256") != expected_sha:
                raise _fail("downstream_hash_map_invalid")

    materialization_binding = _mapping(
        downstream.get("materialization_receipt"), "materialization_binding_invalid"
    )
    materialization_path = _absolute_bound_file(
        materialization_binding.get("path"),
        materialization_binding.get("sha256"),
        "materialization_receipt_invalid",
    )
    if materialization_path.stat().st_size != materialization_binding.get("bytes"):
        raise _fail("materialization_receipt_invalid")
    materialization, materialization_sha = _load_json(
        materialization_path, "materialization_receipt_invalid"
    )
    expected_materialization = {
        "schema_version": 3,
        "artifact_type": "f2_varied_stage_publish_request_bundle",
        "status": "materialized",
        "runtime_id": "brev-isaac62-varied-v3",
        "batch_count": 6,
        "vial_receipt_count": 18,
        "stage_pair_count": 42,
        "split_vial_counts": _SPLIT_VIAL_COUNTS,
    }
    if any(materialization.get(key) != value for key, value in expected_materialization.items()):
        raise _fail("materialization_receipt_invalid")

    locator_path = Path(image_locators).expanduser().absolute()
    locator_artifact, locator_sha = _load_json(locator_path, "image_locators_invalid")
    expected_locator_header = {
        "schema_version": 1,
        "artifact_type": "f2_varied_image_locators",
        "status": "published",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
    }
    if any(locator_artifact.get(key) != value for key, value in expected_locator_header.items()):
        raise _fail("image_locators_invalid")
    source_validation = _mapping(
        locator_artifact.get("source_validation_receipt"), "locator_validation_binding_invalid"
    )
    if source_validation != {"path": str(receipt_path), "sha256": downstream_sha}:
        raise _fail("locator_validation_binding_invalid")
    locator_rows = _mapping(locator_artifact.get("images"), "image_locators_invalid")
    if len(locator_rows) != 42:
        raise _fail("image_locator_denominator_invalid")
    if locator_artifact.get("denominators") != {
        "source_manifests": 6,
        "runs": 6,
        "vials": 18,
        "active_stage_events": 14,
        "images": 42,
        "two_stage_runs": 4,
        "three_stage_runs": 2,
        "split_images": {"train": 30, "validation": 6, "evaluation": 6},
    }:
        raise _fail("image_locator_denominator_invalid")

    request_bindings = _sequence(materialization.get("requests"), "request_bindings_invalid")
    if len(request_bindings) != 6:
        raise _fail("request_denominator_invalid")
    records: list[dict[str, object]] = []
    case_ids: set[object] = set()
    receipt_hashes: set[object] = set()
    video_paths: set[Path] = set()
    video_hashes: set[object] = set()
    run_ids: set[object] = set()
    capture_hashes = _mapping(downstream.get("capture_manifest_sha256_by_batch"), "capture_hashes_invalid")
    for binding_value in request_bindings:
        binding = _mapping(binding_value, "request_binding_invalid")
        relative = binding.get("path")
        if not isinstance(relative, str):
            raise _fail("request_binding_invalid")
        request_path = _bound_file(
            materialization_path.parent, relative, binding.get("sha256"),
            binding.get("bytes"), "request_invalid",
        )
        downstream_relative = request_path.relative_to(root).as_posix()
        downstream_row = listed_paths.get(downstream_relative)
        if downstream_row is None or downstream_row.get("sha256") != binding.get("sha256"):
            raise _fail("request_downstream_binding_mismatch")
        request, _ = _load_json(request_path, "request_invalid")
        if (
            request.get("schema_version") != 3
            or request.get("artifact_type") != "f2_varied_stage_batch_publish_request"
            or request.get("runtime_id") != "brev-isaac62-varied-v3"
            or request.get("data_origin") != "twin"
            or request.get("split_version") != "whole-run-varied-v3"
            or request.get("batch_id") != binding.get("batch_id")
            or request.get("run_id") != binding.get("run_id")
            or request.get("split_role") != binding.get("split_role")
        ):
            raise _fail("request_header_invalid")
        batch_id = request.get("batch_id")
        if not isinstance(batch_id, str) or _BATCH.fullmatch(batch_id) is None:
            raise _fail("batch_id_invalid")
        run_ids.add(request.get("run_id"))
        capture_relative = f"publications/{batch_id}/f2-stage-capture-manifest.json"
        capture_binding = listed_paths.get(capture_relative)
        if capture_binding is None:
            raise _fail("capture_manifest_missing")
        capture_path = _bound_file(
            root, capture_relative, capture_binding.get("sha256"),
            capture_binding.get("bytes"), "capture_manifest_invalid",
        )
        capture, capture_sha = _load_json(capture_path, "capture_manifest_invalid")
        expected_capture = {
            "schema_version": 3,
            "artifact_type": "f2_stage_capture_manifest",
            "status": "accepted",
            "data_origin": "twin",
            "batch_id": batch_id,
            "run_id": request.get("run_id"),
            "recipe_id": request.get("current_recipe_id"),
            "split_role": request.get("split_role"),
            "split_version": request.get("split_version"),
        }
        if any(capture.get(key) != value for key, value in expected_capture.items()):
            raise _fail("capture_manifest_invalid")
        if capture_hashes.get(batch_id) != capture_sha:
            raise _fail("capture_manifest_hash_mismatch")
        events = _sequence(capture.get("stage_events"), "stage_events_invalid")
        vials = _sequence(request.get("vials"), "request_vials_invalid")
        if len(vials) != 3:
            raise _fail("request_vial_denominator_invalid")
        for vial_value in vials:
            vial = _mapping(vial_value, "request_vial_invalid")
            case_id = vial.get("case_id")
            if (
                not isinstance(case_id, str)
                or _CASE.fullmatch(case_id) is None
                or case_id in case_ids
                or vial.get("vial_id") not in {"V1", "V2", "V3"}
                or vial.get("vial_position") not in {1, 2, 3}
            ):
                raise _fail("case_vial_identity_invalid")
            case_ids.add(case_id)
            receipt_sha = vial.get("receipt_sha256")
            receipt_file = _absolute_bound_file(
                vial.get("receipt_path"), receipt_sha, "source_receipt_invalid"
            )
            if receipt_file.name != "receipt.json" or receipt_sha in receipt_hashes:
                raise _fail("source_receipt_identity_invalid")
            receipt_hashes.add(receipt_sha)
            receipt, _ = _load_json(receipt_file, "source_receipt_invalid")
            if (
                receipt.get("schema_version") != 20
                or receipt.get("status") != "pass"
                or receipt.get("errors") != []
                or receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter"
                or _mapping(receipt.get("arguments"), "receipt_arguments_invalid").get("realtime_video") is not True
            ):
                raise _fail("source_receipt_contract_invalid")
            _official_artifact_validation(receipt_file.parent)
            _validate_requested_model_record(request, vial, receipt)
            _validate_pours(request, receipt, events)
            render = _mapping(receipt.get("render"), "receipt_render_invalid")
            timing = _mapping(receipt.get("timing"), "receipt_timing_invalid")
            if render.get("stage_frames_are_video_subsets") is not False:
                raise _fail("stage_video_subset_invalid")
            video_path = _bound_file(
                receipt_file.parent, render.get("video_path"), render.get("video_sha256"),
                None, "video_invalid",
            )
            if video_path in video_paths or render.get("video_sha256") in video_hashes:
                raise _fail("video_identity_duplicate")
            video_paths.add(video_path)
            video_hashes.add(render.get("video_sha256"))
            _validate_video_metadata(video_path, render)
            fill_end_frame, decoded_count = _active_fill_range(render, timing)
            stages = _stage_records(request, vial, capture, capture_sha, locator_rows)
            programmed = _mapping(request.get("programmed_recipe"), "programmed_recipe_invalid")
            steps = render.get("video_timing_source_frame_steps")
            records.append(
                {
                    "batch_id": batch_id,
                    "run_id": request.get("run_id"),
                    "case_id": case_id,
                    "vial_id": vial.get("vial_id"),
                    "vial_position": vial.get("vial_position"),
                    "recipe_id": request.get("current_recipe_id"),
                    "previous_recipe_id": vial.get("previous_recipe_id"),
                    "split_role": request.get("split_role"),
                    "split_version": request.get("split_version"),
                    "recipe_ml": programmed.get("current_ml_by_tank"),
                    "dispense_order": programmed.get("dispense_order"),
                    "source_receipt_sha256": receipt_sha,
                    "video_path": str(video_path),
                    "video_sha256": render.get("video_sha256"),
                    "video_decoded_frame_count": decoded_count,
                    "video_playback_fps": render.get("video_playback_fps"),
                    "video_codec": render.get("video_codec"),
                    "video_container": render.get("video_container"),
                    "fill_frame_range": [0, fill_end_frame],
                    "excluded_post_fill_frame_range": [fill_end_frame, decoded_count],
                    "fill_end_physics_step": timing.get("fill_end_step"),
                    "interval_rule": "bisect_right_source_steps_at_fill_end",
                    "video_timing_source_steps_sha256": _canonical_sha256(steps),
                    "stages": stages,
                }
            )
    stage_ids = {stage["image_id"] for row in records for stage in row["stages"]}
    source_validation_rows = _sequence(
        downstream.get("source_validation"), "source_validation_invalid"
    )
    validated_cases = {
        row.get("case_id")
        for row in source_validation_rows
        if isinstance(row, Mapping) and row.get("status") == "pass"
    }
    if (
        len(case_ids) != 18 or len(run_ids) != 6 or len(records) != 18
        or sum(len(row["stages"]) for row in records) != 42
        or set(locator_rows) != stage_ids
        or validated_cases != case_ids
        or len(source_validation_rows) != 18
    ):
        raise _fail("cohort_denominator_invalid")
    result = {
        "schema_version": 1,
        "artifact_type": "f2_fill_clip_input_manifest",
        "status": "published",
        "data_origin": "twin",
        "split_version": "whole-run-varied-v3",
        "source_downstream_validation_receipt_sha256": downstream_sha,
        "source_request_materialization_receipt_sha256": materialization_sha,
        "source_image_locators_sha256": locator_sha,
        "denominators": {"batches": 6, "runs": 6, "vials": 18, "stages": 42},
        "records": sorted(
            records, key=lambda row: (str(row["batch_id"]), int(row["vial_position"]))
        ),
        "claim_boundary": (
            "Synthetic-twin clip routing only; source receipts and exact frame-step arrays "
            "remain evaluation provenance, and no oracle, label, model, physical-accuracy, "
            "or sim-to-real claim is made."
        ),
    }
    _assert_safe_projection(result)
    return _publish_json_no_clobber(Path(output_path), result)


__all__ = ["F2FillClipInputError", "build_f2_fill_clip_input_manifest"]
