"""CPU contracts for the NVIDIA Paint Ball Emitter F2 render route.

The Isaac-side runner owns the PointInstancer and PhysX attributes.  This
module keeps the append, prototype, containment and receipt rules testable
without importing Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from .physx_volume_gate import (
    expected_cylindrical_height_m,
    find_dense_surface_row,
    height_within_tolerance,
    particle_bulk_height_from_quantiles_m,
    projected_top_dodecagon_min_y_px,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)


PAINT_BALL_RECEIPT_SCHEMA_VERSION = 18
MAX_VELOCITY_M_S = 2.5  # 250 cm/s in ParticlePostProcessingDemo.py at 0.01 m/unit.
EMITTER_RENDER_PATH = "point_instancer_sphere_emitter"
TRANSPARENT_ANNOTATION_DOCUMENTATION_URL = (
    "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/"
    "annotations_with_transparency.html"
)
PAIRED_CAPTURE_CAMERA_PATH = (
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
)
PAIRED_CAPTURE_STATE_FIELDS = (
    "callback_count",
    "timeline_time_s",
    "camera_path",
    "positions",
    "simulation_points",
    "velocities",
    "ids",
    "proto_indices",
)
RENDER_LIMITATIONS = {
    "continuous_surface": False,
    "palette_uncalibrated": True,
    "pbd_not_navier_stokes": True,
}
_ARRAY_FIELDS = ("positions", "velocities", "orientations", "scales")
FULL_RUN_VOLUME_ROWS = ((1412, 10.0, 370), (2622, 20.0, 740), (4004, 60.0, 2221))
FULL_RUN_FRAME_STEPS = tuple(
    sorted({*range(12, 4005, 12), *(row[0] for row in FULL_RUN_VOLUME_ROWS)})
)
FULL_RUN_FRAME_COUNT = 336
FULL_RUN_STAGE_INDICES = tuple(
    FULL_RUN_FRAME_STEPS.index(row[0]) for row in FULL_RUN_VOLUME_ROWS
)
FULL_RUN_STREAM_ROWS = (
    (1, "t200", 1, 1412),
    (2, "t400", 1413, 2622),
    (3, "t100", 2623, 3968),
)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_manifest_valid(
    paths: object,
    hashes: object,
    *,
    count: int,
) -> bool:
    return bool(
        isinstance(paths, list)
        and isinstance(hashes, list)
        and len(paths) == len(hashes) == count
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == count
        and all(_valid_sha256(value) for value in hashes)
    )


def _active_stream_for_step(step: int) -> list[str]:
    for _ordinal, tank_id, start_step, end_step in FULL_RUN_STREAM_ROWS:
        if start_step <= step <= end_step:
            return [tank_id]
    return []


def _validate_full_run_contract(receipt: Mapping[str, object]) -> tuple[str, ...]:
    """Validate the frozen A30 full-run denominator and artifact contract."""

    failures: list[str] = []
    if receipt.get("status") != "pass":
        failures.append("full_run_status_contract")
    captures = receipt.get("captures")
    required = (
        [
            item
            for item in captures
            if isinstance(item, Mapping)
            and isinstance(item.get("volume_gate"), Mapping)
            and item["volume_gate"].get("required") is True
        ]
        if isinstance(captures, list)
        else []
    )
    observed_volume_rows: list[tuple[int, float, int, int, int]] = []
    for item in required:
        gate = item["volume_gate"]
        try:
            observed_volume_rows.append(
                (
                    int(item["physics_step"]),
                    float(gate["programmed_cumulative_volume_ml"]),
                    int(gate["expected_cumulative_particle_count"]),
                    int(item["emitted_particle_count"]),
                    int(item["active_particle_count"]),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            observed_volume_rows.append((-1, math.nan, -1, -1, -1))
    expected_volume_rows = [
        (step, volume_ml, count, count, count)
        for step, volume_ml, count in FULL_RUN_VOLUME_ROWS
    ]
    if (
        observed_volume_rows != expected_volume_rows
        or any(
            item["volume_gate"].get("state_passed") is not True
            or item["volume_gate"].get("visible_passed") is not True
            or item["volume_gate"].get("passed") is not True
            or item["volume_gate"].get("failure_reasons") != []
            for item in required
        )
    ):
        failures.append("full_run_volume_contract")

    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    frame_paths = render.get("frame_paths")
    frame_hashes = render.get("frame_sha256")
    frame_steps = render.get("frame_physics_steps")
    if (
        render.get("renderer") != "PathTracing"
        or render.get("resolution") != [640, 480]
        or render.get("rt_subframes") != 64
        or render.get("warmup_updates") != 60
        or render.get("frame_count") != FULL_RUN_FRAME_COUNT
        or not _artifact_manifest_valid(
            frame_paths, frame_hashes, count=FULL_RUN_FRAME_COUNT
        )
        or frame_steps != list(FULL_RUN_FRAME_STEPS)
    ):
        failures.append("full_run_rgb_contract")

    stage_indices = list(FULL_RUN_STAGE_INDICES)
    expected_stage_paths = (
        [frame_paths[index] for index in stage_indices]
        if isinstance(frame_paths, list) and len(frame_paths) == FULL_RUN_FRAME_COUNT
        else None
    )
    expected_stage_hashes = (
        [frame_hashes[index] for index in stage_indices]
        if isinstance(frame_hashes, list) and len(frame_hashes) == FULL_RUN_FRAME_COUNT
        else None
    )
    if (
        render.get("stage_frame_indices") != stage_indices
        or render.get("stage_frame_physics_steps")
        != [row[0] for row in FULL_RUN_VOLUME_ROWS]
        or render.get("stage_frame_paths") != expected_stage_paths
        or render.get("stage_frame_sha256") != expected_stage_hashes
        or render.get("primary_final_frame_sha256")
        != (expected_stage_hashes[-1] if expected_stage_hashes else None)
        or render.get("stream_only_frame_count") != 0
        or render.get("vial_proxy_frame_count") != FULL_RUN_FRAME_COUNT
        or render.get("vial_proxy_transition_step") != FULL_RUN_FRAME_STEPS[0]
        or render.get("liquid_representation_monotonic") is not True
    ):
        failures.append("full_run_stage_contract")

    video_path = render.get("video_path")
    if (
        not isinstance(video_path, str)
        or not video_path
        or not _valid_sha256(render.get("video_sha256"))
        or not isinstance(render.get("video_codec"), str)
        or not render.get("video_codec")
        or not isinstance(render.get("video_container"), str)
        or not render.get("video_container")
        or not video_path.endswith(f".{render.get('video_container')}")
        or type(render.get("video_writer_fallback_used")) is not bool
        or render.get("video_decodable") is not True
        or render.get("video_decoded_frame_count") != FULL_RUN_FRAME_COUNT
        or render.get("stage_frames_are_video_subsets") is not True
    ):
        failures.append("full_run_video_contract")

    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    timing = receipt.get("timing")
    timing = timing if isinstance(timing, Mapping) else {}
    if (
        arguments.get("resolution") != "640x480"
        or arguments.get("capture_stride") != 12
        or arguments.get("video_fps") != 10.0
        or timing.get("physics_hz") != 120
        or timing.get("capture_hz") != 10
        or timing.get("pour_duration_steps") != [1412, 1210, 1346]
        or timing.get("pour_end_steps") != [1412, 2622, 3968]
        or timing.get("no_inter_pour_settle") is not True
        or timing.get("fill_end_step") != 3968
        or timing.get("primary_final_capture_step") != 4004
        or not _finite_number(timing.get("primary_final_offset_s"))
        or not math.isclose(
            float(timing.get("primary_final_offset_s", math.nan)),
            0.3,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or timing.get("executed_physics_steps") != 4004
        or timing.get("callbacks_expected") != 4004
        or timing.get("callbacks_observed") != 4004
        or timing.get("one_callback_per_app_update") is not True
        or timing.get("capture_callbacks_advanced") != 0
        or timing.get("capture_timeline_time_advanced") is not False
        or timing.get("capture_playback_preserved") is not True
    ):
        failures.append("full_run_timing_contract")

    scene = receipt.get("scene_contract")
    scene = scene if isinstance(scene, Mapping) else {}
    hidden = scene.get("hidden_source_visibility")
    expected_hidden_paths = {
        "/FillStation/FillStation_LocalRoot/Fill_liquid",
        "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream",
    }
    hidden_valid = bool(
        isinstance(hidden, Mapping)
        and set(hidden) == expected_hidden_paths
        and all(
            isinstance(rows, Mapping)
            and rows
            and all(value == "invisible" for value in rows.values())
            for rows in hidden.values()
        )
    )
    source_layers = scene.get("source_layer_records")
    source_layers_valid = bool(
        isinstance(source_layers, list)
        and source_layers
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("resolved_path"), str)
            and row.get("resolved_path")
            and _valid_sha256(row.get("sha256"))
            for row in source_layers
        )
    )
    camera = receipt.get("camera")
    camera = camera if isinstance(camera, Mapping) else {}
    clip = camera.get("clipping_range_effective")
    clip_valid = bool(
        isinstance(clip, list)
        and len(clip) == 2
        and all(_finite_number(value) for value in clip)
        and math.isclose(
            float(clip[0]), 0.0007619999814778566, rel_tol=0.0, abs_tol=1.0e-9
        )
        and math.isclose(
            float(clip[1]), 25.399999618530273, rel_tol=0.0, abs_tol=1.0e-9
        )
    )
    parity = receipt.get("pre_pour_parity")
    parity = parity if isinstance(parity, Mapping) else {}
    parity_valid = bool(
        parity.get("passed") is True
        and all(
            isinstance(parity.get(path_key), str)
            and parity.get(path_key)
            and _valid_sha256(parity.get(hash_key))
            for path_key, hash_key in (
                ("reference_path", "reference_sha256"),
                ("f2_prefill_path", "f2_prefill_sha256"),
                ("absdiff_path", "absdiff_sha256"),
            )
        )
    )
    if (
        scene.get("default_prim") != "/FillStation"
        or scene.get("meters_per_unit") != 1.0
        or scene.get("up_axis") != "Z"
        or scene.get("source_layers_unchanged") is not True
        or not source_layers_valid
        or not hidden_valid
        or scene.get("forbidden_authored_paths") != []
        or scene.get("custom_camera_count") != 0
        or scene.get("custom_light_count") != 0
        or scene.get("custom_backdrop_count") != 0
        or camera.get("path") != PAIRED_CAPTURE_CAMERA_PATH
        or camera.get("source_has_authored_clipping_range") is not True
        or camera.get("clipping_range_authored_by_runner") is not False
        or not clip_valid
        or camera.get("resolution") != [640, 480]
        or not parity_valid
    ):
        failures.append("full_run_scene_contract")

    stream = receipt.get("stream_render")
    stream = stream if isinstance(stream, Mapping) else {}
    records = stream.get("records")
    stream_rows = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                stream_rows.append(None)
                continue
            stream_rows.append(
                (
                    record.get("ordinal"),
                    record.get("tank_id"),
                    record.get("start_step"),
                    record.get("end_step"),
                )
            )
    if (
        stream.get("representation")
        != "render_only_mesh_current_tank_material"
        or stream_rows != list(FULL_RUN_STREAM_ROWS)
        or not isinstance(records, list)
        or any(
            not isinstance(record, Mapping)
            or record.get("active_only_during_own_pour") is not True
            or not isinstance(record.get("state_artifact_path"), str)
            or not record.get("state_artifact_path")
            or not _valid_sha256(record.get("state_artifact_sha256"))
            for record in records
        )
    ):
        failures.append("full_run_stream_contract")

    semantics = receipt.get("semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    pairs = semantics.get("pairs")
    mask_paths = [pair.get("mask_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    mask_hashes = [pair.get("mask_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    semantic_paths = [pair.get("semantic_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    semantic_hashes = [pair.get("semantic_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    pairs_valid = bool(
        isinstance(pairs, list)
        and len(pairs) == FULL_RUN_FRAME_COUNT
        and isinstance(frame_paths, list)
        and len(frame_paths) == FULL_RUN_FRAME_COUNT
        and isinstance(frame_hashes, list)
        and len(frame_hashes) == FULL_RUN_FRAME_COUNT
        and _artifact_manifest_valid(mask_paths, mask_hashes, count=FULL_RUN_FRAME_COUNT)
        and _artifact_manifest_valid(
            semantic_paths, semantic_hashes, count=FULL_RUN_FRAME_COUNT
        )
        and all(
            pair.get("physics_step") == FULL_RUN_FRAME_STEPS[index]
            and pair.get("rgb_path") == frame_paths[index]
            and pair.get("rgb_sha256") == frame_hashes[index]
            and pair.get("liquid_representation") == "point_instancer"
            and pair.get("shared_rgb_semantic_capture") is False
            and pair.get("paired_same_physics_state_rgb_semantic_capture") is True
            and pair.get("paired_capture_state_passed") is True
            and pair.get("mask_readable") is True
            and pair.get("mask_resolution") == [640, 480]
            and isinstance(pair.get("liquid_pixel_count"), int)
            and not isinstance(pair.get("liquid_pixel_count"), bool)
            and 0 < pair.get("liquid_pixel_count") < 640 * 480
            and pair.get("visible_stream_tanks")
            == (
                []
                if index in FULL_RUN_STAGE_INDICES
                else _active_stream_for_step(FULL_RUN_FRAME_STEPS[index])
            )
            for index, pair in enumerate(pairs)
        )
    )
    paired = semantics.get("paired_capture_contract")
    state_records = paired.get("state_records") if isinstance(paired, Mapping) else None
    annotation = semantics.get("transparent_annotation_override")
    transitions = annotation.get("transitions") if isinstance(annotation, Mapping) else None
    guards = timing.get("capture_guard_records")
    if (
        semantics.get("semantic_label") != "liquid"
        or semantics.get("semantic_type") != "class"
        or semantics.get("annotator") != "semantic_segmentation"
        or semantics.get("mask_source") != "exact_renderer_semantic_class_ids"
        or semantics.get("resolution") != [640, 480]
        or semantics.get("point_instancer_semantic_bound") is not True
        or semantics.get("stream_semantic_bound") is not True
        or semantics.get("capture_volumes_enabled_readback") is not True
        or type(semantics.get("capture_volumes_restored_readback")) is not bool
        or semantics.get("semantic_gate_passed") is not True
        or not pairs_valid
        or semantics.get("frame_semantic_paths") != semantic_paths
        or semantics.get("frame_semantic_sha256") != semantic_hashes
        or semantics.get("stage_mask_sha256")
        != ([mask_hashes[index] for index in FULL_RUN_STAGE_INDICES] if mask_hashes else None)
        or not isinstance(state_records, list)
        or len(state_records) < FULL_RUN_FRAME_COUNT
        or not isinstance(guards, list)
        or len(guards) < 2 * FULL_RUN_FRAME_COUNT
        or not isinstance(transitions, list)
        or len(transitions) < 3 * FULL_RUN_FRAME_COUNT
    ):
        failures.append("full_run_semantic_contract")
    if semantics.get("training_eligible") is not False:
        failures.append("training_eligibility_contract")
    return tuple(dict.fromkeys(failures))


def _capture_guard_preserves_state(record: object, expected_pass: str) -> bool:
    if not isinstance(record, Mapping) or record.get("capture_pass") != expected_pass:
        return False
    callbacks_before = record.get("callbacks_before")
    callbacks_after = record.get("callbacks_after")
    time_before = record.get("timeline_time_before_s")
    time_after = record.get("timeline_time_after_s")
    return bool(
        isinstance(callbacks_before, int)
        and not isinstance(callbacks_before, bool)
        and callbacks_before == callbacks_after
        and _finite_number(time_before)
        and _finite_number(time_after)
        and math.isclose(
            float(time_before), float(time_after), rel_tol=0.0, abs_tol=1.0e-12
        )
        and type(record.get("timeline_playing_before")) is bool
        and record.get("timeline_playing_before")
        is record.get("timeline_playing_after")
    )


def _paired_state_record_valid(
    record: object, capture_guards: list[object]
) -> bool:
    if not isinstance(record, Mapping):
        return False
    if record.get("passed") is not True or record.get("failed_predicates") != []:
        return False
    row_count = record.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        return False
    physics_step = record.get("physics_step")
    phase = record.get("phase")
    attempt = record.get("attempt")
    if (
        isinstance(physics_step, bool)
        or not isinstance(physics_step, int)
        or physics_step < 0
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt <= 0
    ):
        return False
    for field in ("callback_count", "timeline_time_s", "camera_path"):
        before = record.get(f"{field}_before")
        after = record.get(f"{field}_after")
        if before != after:
            return False
    if (
        not isinstance(record.get("callback_count_before"), int)
        or isinstance(record.get("callback_count_before"), bool)
        or not _finite_number(record.get("timeline_time_s_before"))
        or not isinstance(record.get("camera_path_before"), str)
        or record.get("camera_path_before") != PAIRED_CAPTURE_CAMERA_PATH
    ):
        return False
    for field in ("positions", "simulation_points", "velocities", "ids", "proto_indices"):
        before = record.get(f"{field}_sha256_before")
        after = record.get(f"{field}_sha256_after")
        if not _valid_sha256(before) or before != after:
            return False
    rgb_index = record.get("rgb_guard_record_index")
    semantic_index = record.get("semantic_guard_record_index")
    if (
        isinstance(rgb_index, bool)
        or not isinstance(rgb_index, int)
        or isinstance(semantic_index, bool)
        or not isinstance(semantic_index, int)
        or rgb_index == semantic_index
        or not (0 <= rgb_index < len(capture_guards))
        or not (0 <= semantic_index < len(capture_guards))
    ):
        return False
    rgb_guard = capture_guards[rgb_index]
    semantic_guard = capture_guards[semantic_index]
    if not _capture_guard_preserves_state(
        rgb_guard, "kept_rgb_source_equivalent"
    ) or not _capture_guard_preserves_state(
        semantic_guard, "semantic_transparent_annotation"
    ):
        return False
    if (
        rgb_guard.get("callbacks_after")
        != semantic_guard.get("callbacks_before")
        or record.get("callback_count_before")
        != semantic_guard.get("callbacks_before")
        or record.get("callback_count_after")
        != semantic_guard.get("callbacks_after")
        or not math.isclose(
            float(rgb_guard.get("timeline_time_after_s")),
            float(semantic_guard.get("timeline_time_before_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(record.get("timeline_time_s_before")),
            float(semantic_guard.get("timeline_time_before_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(record.get("timeline_time_s_after")),
            float(semantic_guard.get("timeline_time_after_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        return False
    identity_fields = ("physics_step", "phase", "attempt")
    return all(
        record.get(field) == rgb_guard.get(field) == semantic_guard.get(field)
        for field in identity_fields
    )


def _copy_rows(value: object, name: str, width: int) -> list[list[float]]:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (width,) or not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} must be finite with shape (N, {width})")
    return rows.tolist()


def append_point_instancer_state(
    previous: Mapping[str, object],
    *,
    positions: object,
    velocities: object,
    prototype_index: int,
    particle_ids: object,
) -> dict[str, object]:
    """Append one emission while preserving all PointInstancer row identity."""

    if isinstance(prototype_index, bool) or not isinstance(prototype_index, int) or prototype_index < 0:
        raise ValueError("prototype_index must be a non-negative integer")
    position_rows = _copy_rows(positions, "positions", 3)
    velocity_rows = _copy_rows(velocities, "velocities", 3)
    if len(position_rows) != len(velocity_rows) or not position_rows:
        raise ValueError("positions and velocities length mismatch or empty emission")
    ids = [int(value) for value in particle_ids]
    if len(ids) != len(position_rows) or len(set(ids)) != len(ids):
        raise ValueError("particle_ids length or uniqueness mismatch")
    existing_ids = [int(value) for value in previous.get("ids", [])]
    if len(set(existing_ids)) != len(existing_ids):
        raise ValueError("existing particle IDs are not stable")
    if existing_ids and ids[0] <= max(existing_ids):
        raise ValueError("new particle IDs must extend stable row identity")
    row_count = len(existing_ids)
    copied: dict[str, object] = {}
    widths = {"positions": 3, "velocities": 3, "orientations": 4, "scales": 3}
    for field in _ARRAY_FIELDS:
        rows = _copy_rows(previous.get(field, []), field, widths[field])
        if len(rows) != row_count:
            raise ValueError(f"{field} length drifted from stable IDs")
        copied[field] = rows
    simulation_rows = _copy_rows(
        previous.get("simulation_points", previous.get("positions", [])),
        "simulation_points",
        3,
    )
    if len(simulation_rows) != row_count:
        raise ValueError("simulation_points length drifted from stable IDs")
    copied["simulation_points"] = simulation_rows
    existing_proto = [int(value) for value in previous.get("proto_indices", [])]
    if len(existing_proto) != row_count:
        raise ValueError("proto_indices length drifted from stable IDs")
    copied["positions"].extend(position_rows)
    copied["simulation_points"].extend(position_rows)
    copied["velocities"].extend(velocity_rows)
    copied["orientations"].extend([[1.0, 0.0, 0.0, 0.0] for _ in position_rows])
    copied["scales"].extend([[1.0, 1.0, 1.0] for _ in position_rows])
    copied["ids"] = existing_ids + ids
    copied["proto_indices"] = existing_proto + [prototype_index] * len(position_rows)
    return copied


def build_prototype_specs(palette: Mapping[str, Sequence[float]]) -> tuple[dict[str, object], ...]:
    """Describe one semantic liquid sphere prototype per fixed colour."""

    specs: list[dict[str, object]] = []
    for index, (tank_id, color) in enumerate(palette.items()):
        values = [float(value) for value in color]
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("prototype colour must contain three finite values")
        specs.append(
            {
                "prototype_index": index,
                "tank_id": str(tank_id),
                "shape": "Sphere",
                "semantic_label": "liquid",
                "color": values,
            }
        )
    if not specs:
        raise ValueError("palette must not be empty")
    return tuple(specs)


def build_palette_material_path(namespace: str, prototype_index: int) -> str:
    """Return an absolute USD material path with a valid named prim leaf."""

    if not isinstance(namespace, str) or not namespace.startswith("/") or namespace.endswith("/"):
        raise ValueError("namespace must be a non-root absolute USD prim path")
    if isinstance(prototype_index, bool) or not isinstance(prototype_index, int) or prototype_index < 0:
        raise ValueError("prototype_index must be a non-negative integer")
    return f"{namespace}/PaletteMaterials/Palette_{prototype_index:02d}"


def containment_evidence(
    positions: object,
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    expected_in_flight_mask: object | None = None,
    cylindrical_center_xy: Sequence[float] | None = None,
    inner_radius_m: float | None = None,
    polygon_center_xy: Sequence[float] | None = None,
    polygon_apothem_m: float | None = None,
    polygon_wall_count: int | None = None,
    numerical_tolerance_m: float = 1.0e-6,
) -> dict[str, object]:
    """Count particles inside the fitted vial interior and expose leakage."""

    rows = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or not np.all(np.isfinite(rows)):
        raise ValueError("positions must be finite with shape (N, 3)")
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must contain three finite values")
    if np.any(upper <= lower):
        raise ValueError("upper bounds must exceed lower bounds")
    if numerical_tolerance_m < 0.0 or not math.isfinite(float(numerical_tolerance_m)):
        raise ValueError("numerical_tolerance_m must be finite and non-negative")
    inside_z = np.all(
        (rows[:, 2:] >= lower[None, 2:] - numerical_tolerance_m)
        & (rows[:, 2:] <= upper[None, 2:] + numerical_tolerance_m),
        axis=1,
    )
    conservative_inside: np.ndarray | None = None
    if cylindrical_center_xy is not None or inner_radius_m is not None:
        if cylindrical_center_xy is None or inner_radius_m is None:
            raise ValueError("cylindrical center and radius must be supplied together")
        center = np.asarray(cylindrical_center_xy, dtype=np.float64)
        radius = float(inner_radius_m)
        if center.shape != (2,) or not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("cylindrical center/radius must be finite and valid")
        conservative_inside = (
            np.sum((rows[:, :2] - center[None, :]) ** 2, axis=1)
            <= (radius + numerical_tolerance_m) ** 2
        ) & inside_z
    else:
        center = None
        radius = None
    polygon_values = (polygon_center_xy, polygon_apothem_m, polygon_wall_count)
    if any(value is not None for value in polygon_values):
        if any(value is None for value in polygon_values):
            raise ValueError("polygon center, apothem and wall count must be supplied together")
        polygon_center = np.asarray(polygon_center_xy, dtype=np.float64)
        apothem = float(polygon_apothem_m)
        if (
            polygon_center.shape != (2,)
            or not np.all(np.isfinite(polygon_center))
            or not math.isfinite(apothem)
            or apothem <= 0.0
            or isinstance(polygon_wall_count, bool)
            or not isinstance(polygon_wall_count, int)
            or polygon_wall_count < 3
        ):
            raise ValueError("polygon center/apothem/wall count must be finite and valid")
        angles = 2.0 * math.pi * np.arange(polygon_wall_count) / polygon_wall_count
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        centered = rows[:, :2] - polygon_center[None, :]
        inside_xy = np.all(
            centered @ normals.T <= apothem + numerical_tolerance_m,
            axis=1,
        )
        inside = inside_xy & inside_z
    elif conservative_inside is not None:
        polygon_center = None
        apothem = None
        inside = conservative_inside
    else:
        polygon_center = None
        apothem = None
        inside = np.all((rows >= lower[None, :]) & (rows <= upper[None, :]), axis=1)
    if expected_in_flight_mask is None:
        in_flight = np.zeros(len(rows), dtype=bool)
    else:
        in_flight = np.asarray(expected_in_flight_mask, dtype=bool)
        if in_flight.shape != (len(rows),):
            raise ValueError("expected_in_flight_mask length must match positions")
        if np.any(in_flight & inside):
            raise ValueError("in-flight particles cannot also be inside the vial")
    outside = ~inside
    leaked = outside & ~in_flight
    max_abs = np.max(np.abs(rows), axis=0) if len(rows) else np.zeros(3)
    contained_z = rows[inside, 2]
    if len(contained_z):
        z_q05, z_q95 = np.quantile(contained_z, (0.05, 0.95), method="linear")
        z_min: float | None = float(np.min(contained_z))
        z_max: float | None = float(np.max(contained_z))
        z_q05_value: float | None = float(z_q05)
        z_q95_value: float | None = float(z_q95)
    else:
        z_min = z_max = z_q05_value = z_q95_value = None
    return {
        "active_particle_count": int(len(rows)),
        "inside_vial_interior_count": int(np.count_nonzero(inside)),
        "outside_vial_interior_count": int(np.count_nonzero(outside)),
        "expected_in_flight_stream_count": int(np.count_nonzero(in_flight)),
        "leaked_particle_count": int(np.count_nonzero(leaked)),
        "max_abs_bounds_m": [float(value) for value in max_abs],
        "lower_bounds_m": lower.tolist(),
        "upper_bounds_m": upper.tolist(),
        "cylindrical_center_xy_m": None if cylindrical_center_xy is None else [float(value) for value in cylindrical_center_xy],
        "inner_radius_m": None if inner_radius_m is None else float(inner_radius_m),
        "inside_conservative_circle_count": (
            None
            if conservative_inside is None
            else int(np.count_nonzero(conservative_inside))
        ),
        "exact_polygon_containment": polygon_center_xy is not None,
        "polygon_center_xy_m": (
            None if polygon_center is None else [float(value) for value in polygon_center]
        ),
        "polygon_apothem_m": None if apothem is None else float(apothem),
        "polygon_wall_count": polygon_wall_count,
        "finite_state": True,
        "contained_z_min_m": z_min,
        "contained_z_q05_m": z_q05_value,
        "contained_z_q95_m": z_q95_value,
        "contained_z_max_m": z_max,
        "numerical_tolerance_m": float(numerical_tolerance_m),
        "leakage_detected": bool(np.any(leaked)),
    }


def build_capture_record(
    *,
    physics_step: int,
    phase: str,
    containment: Mapping[str, object],
    prototype_count: int,
    instancer_path: str,
    emitted_particle_count: int | None = None,
    programmed_cumulative_volume_ml: float | None = None,
    expected_cumulative_particle_count: int | None = None,
    cross_section_area_m2: float | None = None,
    height_tolerance_m: float | None = None,
    render_radius_m: float | None = None,
    point_instancer_scales: object | None = None,
    smoothed_render_containment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(physics_step, bool) or not isinstance(physics_step, int) or physics_step < 0:
        raise ValueError("physics_step must be a non-negative integer")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be non-empty")
    if isinstance(prototype_count, bool) or not isinstance(prototype_count, int) or prototype_count <= 0:
        raise ValueError("prototype_count must be positive")
    if not isinstance(instancer_path, str) or not instancer_path:
        raise ValueError("instancer_path must be non-empty")
    required = {
        "active_particle_count",
        "inside_vial_interior_count",
        "outside_vial_interior_count",
        "expected_in_flight_stream_count",
        "leaked_particle_count",
        "max_abs_bounds_m",
        "leakage_detected",
        "inside_conservative_circle_count",
        "exact_polygon_containment",
        "finite_state",
        "contained_z_min_m",
        "contained_z_q05_m",
        "contained_z_q95_m",
        "contained_z_max_m",
    }
    if not required.issubset(containment):
        raise ValueError("containment evidence is incomplete")
    active_count = int(containment["active_particle_count"])
    emitted_count = active_count if emitted_particle_count is None else emitted_particle_count
    if (
        isinstance(emitted_count, bool)
        or not isinstance(emitted_count, int)
        or emitted_count < 0
    ):
        raise ValueError("emitted_particle_count must be a non-negative integer")
    stage_values = (
        programmed_cumulative_volume_ml,
        expected_cumulative_particle_count,
        cross_section_area_m2,
        height_tolerance_m,
        render_radius_m,
        point_instancer_scales,
        smoothed_render_containment,
    )
    volume_gate_required = any(value is not None for value in stage_values)
    if volume_gate_required and any(value is None for value in stage_values):
        raise ValueError("stage volume evidence inputs must be supplied together")
    scale_min: list[float] | None = None
    scale_max: list[float] | None = None
    volume_gate: dict[str, object] = {
        "required": volume_gate_required,
        "passed": None,
        "failure_reasons": [],
    }
    if volume_gate_required:
        if (
            isinstance(expected_cumulative_particle_count, bool)
            or not isinstance(expected_cumulative_particle_count, int)
            or expected_cumulative_particle_count <= 0
        ):
            raise ValueError("expected_cumulative_particle_count must be positive")
        if not isinstance(smoothed_render_containment, Mapping):
            raise TypeError("smoothed_render_containment must be a mapping")
        scales = np.asarray(point_instancer_scales)
        if scales.dtype.kind not in "fiu":
            raise TypeError("point_instancer_scales must contain real numbers")
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (active_count, 3) or not np.all(np.isfinite(scales)):
            raise ValueError("point_instancer_scales must be finite with shape (N, 3)")
        scale_min = [float(value) for value in np.min(scales, axis=0)]
        scale_max = [float(value) for value in np.max(scales, axis=0)]
        expected_height = expected_cylindrical_height_m(
            programmed_cumulative_volume_ml,
            cross_section_area_m2,
        )
        q05 = containment["contained_z_q05_m"]
        q95 = containment["contained_z_q95_m"]
        particle_height = (
            particle_bulk_height_from_quantiles_m(q05, q95, render_radius_m)
            if q05 is not None and q95 is not None
            else None
        )
        render_q05 = smoothed_render_containment.get("contained_z_q05_m")
        render_q95 = smoothed_render_containment.get("contained_z_q95_m")
        smoothed_render_height = (
            particle_bulk_height_from_quantiles_m(
                render_q05,
                render_q95,
                render_radius_m,
            )
            if render_q05 is not None and render_q95 is not None
            else None
        )
        failures: list[str] = []
        if containment["exact_polygon_containment"] is not True:
            failures.append("exact_dodecagon_containment_missing")
        if containment["finite_state"] is not True:
            failures.append("particle_state_nonfinite")
        if emitted_count != expected_cumulative_particle_count:
            failures.append("emitted_count_mismatch")
        if active_count != emitted_count:
            failures.append("active_count_mismatch")
        if int(containment["inside_vial_interior_count"]) != emitted_count:
            failures.append("inside_count_mismatch")
        if int(containment["expected_in_flight_stream_count"]) != 0:
            failures.append("in_flight_particles_at_stage")
        if int(containment["leaked_particle_count"]) != 0:
            failures.append("outside_or_escaped_particles")
        if particle_height is None or not height_within_tolerance(
            particle_height,
            expected_height,
            height_tolerance_m,
        ):
            failures.append("particle_height_outside_tolerance")
        volume_gate = {
            "required": True,
            "state_passed": not failures,
            "visible_passed": None,
            "passed": False,
            "failure_reasons": failures,
            "programmed_cumulative_volume_ml": float(programmed_cumulative_volume_ml),
            "expected_cumulative_particle_count": expected_cumulative_particle_count,
            "cross_section_area_m2": float(cross_section_area_m2),
            "expected_height_m": expected_height,
            "particle_bulk_height_m": particle_height,
            "height_tolerance_m": float(height_tolerance_m),
            "render_radius_m": float(render_radius_m),
            "nominal_particle_volume_ml": float(
                8.0 * float(render_radius_m) ** 3 * 1_000_000.0
            ),
            "particle_count_volume_formula": "(2 * fluid_rest_offset_m)^3",
            "particle_count_volume_source": (
                "NVIDIA FluidBallEmitterDemo.py: particleSpacing = "
                "2 * fluidRestOffset"
            ),
            "expected_count_nominal_volume_ml": float(
                expected_cumulative_particle_count
                * 8.0
                * float(render_radius_m) ** 3
                * 1_000_000.0
            ),
            "smoothed_render_state": {
                "inside_vial_interior_count": smoothed_render_containment.get(
                    "inside_vial_interior_count"
                ),
                "contained_z_min_m": smoothed_render_containment.get(
                    "contained_z_min_m"
                ),
                "contained_z_q05_m": render_q05,
                "contained_z_q95_m": render_q95,
                "contained_z_max_m": smoothed_render_containment.get(
                    "contained_z_max_m"
                ),
                "particle_bulk_height_m": smoothed_render_height,
            },
        }
    return {
        "physics_step": physics_step,
        "phase": phase,
        "emitted_particle_count": emitted_count,
        "active_particle_count": containment["active_particle_count"],
        "inside_vial_interior_count": containment["inside_vial_interior_count"],
        "outside_vial_interior_count": containment["outside_vial_interior_count"],
        "expected_in_flight_stream_count": containment["expected_in_flight_stream_count"],
        "in_flight_above_mouth_count": containment["expected_in_flight_stream_count"],
        "leaked_particle_count": containment["leaked_particle_count"],
        "outside_or_escaped_particle_count": containment["leaked_particle_count"],
        "inside_conservative_circle_count": containment["inside_conservative_circle_count"],
        "exact_polygon_containment": containment["exact_polygon_containment"],
        "finite_state": containment["finite_state"],
        "contained_z_min_m": containment["contained_z_min_m"],
        "contained_z_q05_m": containment["contained_z_q05_m"],
        "contained_z_q95_m": containment["contained_z_q95_m"],
        "contained_z_max_m": containment["contained_z_max_m"],
        "point_instancer_scale_component_min": scale_min,
        "point_instancer_scale_component_max": scale_max,
        "volume_gate": volume_gate,
        "max_abs_bounds_m": list(containment["max_abs_bounds_m"]),
        "leakage_detected": containment["leakage_detected"],
        "prototype_count": prototype_count,
        "instancer_path": instancer_path,
        "max_velocity_m_s": MAX_VELOCITY_M_S,
        "render_limitations": dict(RENDER_LIMITATIONS),
    }


def attach_visible_volume_evidence(
    capture_record: Mapping[str, object],
    liquid_mask: object,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    mouth_z_m: float,
    circumradius_m: float,
) -> dict[str, object]:
    """Attach the frozen semantic-mask height gate to one stage capture."""

    if not isinstance(capture_record, Mapping):
        raise TypeError("capture_record must be a mapping")
    result = dict(capture_record)
    existing_gate = capture_record.get("volume_gate")
    if not isinstance(existing_gate, Mapping):
        raise ValueError("capture_record has no volume gate")
    gate = dict(existing_gate)
    result["volume_gate"] = gate
    if gate.get("required") is not True:
        gate.update(
            {
                "visible_passed": None,
                "visible_mask_evidence": {"required": False},
                "passed": gate.get("state_passed") is True,
            }
        )
        return result
    mask = np.asarray(liquid_mask)
    if mask.ndim != 2:
        raise ValueError("liquid_mask must be two-dimensional")
    roi = projected_vial_geometry_roi_mask(
        mask.shape,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        mouth_z_m=mouth_z_m,
        circumradius_m=circumradius_m,
    )
    failures = list(gate.get("failure_reasons", []))
    try:
        surface = find_dense_surface_row(mask, geometry_roi_mask=roi)
    except (TypeError, ValueError) as error:
        visible_evidence: dict[str, object] = {
            "required": True,
            "passed": False,
            "status": "missing_dense_surface",
            "error": str(error),
        }
        visible_passed = False
    else:
        visible_height = visible_height_from_surface_row_m(
            surface["surface_row_px"],
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            maximum_height_m=float(mouth_z_m) - float(bottom_z_m),
        )
        visible_passed = height_within_tolerance(
            visible_height,
            gate["expected_height_m"],
            gate["height_tolerance_m"],
        )
        visible_evidence = {
            "required": True,
            "passed": visible_passed,
            "status": "measured",
            **surface,
            "observed_height_m": visible_height,
            "absolute_error_m": abs(
                visible_height - float(gate["expected_height_m"])
            ),
        }
    if not visible_passed and "visible_height_outside_tolerance" not in failures:
        failures.append("visible_height_outside_tolerance")
    gate.update(
        {
            "visible_passed": visible_passed,
            "visible_mask_evidence": visible_evidence,
            "failure_reasons": failures,
            "passed": gate.get("state_passed") is True and visible_passed,
        }
    )
    return result


def validate_paint_ball_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(receipt, Mapping):
        return ("receipt_missing_or_unreadable",)
    failures: list[str] = []
    if receipt.get("schema_version") != PAINT_BALL_RECEIPT_SCHEMA_VERSION:
        failures.append("receipt_schema")
    if receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter":
        failures.append("receipt_identity")
    if receipt.get("render_path") != EMITTER_RENDER_PATH:
        failures.append("render_path")
    if receipt.get("status") not in {"pass", "insufficient_evidence"}:
        failures.append("run_status")
    system = receipt.get("particle_system")
    if (
        not isinstance(system, Mapping)
        or system.get("max_velocity_m_s") != MAX_VELOCITY_M_S
        or system.get("smoothing") is not True
        or system.get("smoothing_enabled") is not True
        or system.get("smoothing_strength") != 1.0
        or system.get("anisotropy") is not True
        or system.get("anisotropy_enabled") is not True
        or system.get("anisotropy_scale") != 1.0
        or system.get("ccd") is not True
        or any(
            not _finite_number(system.get(key))
            for key in (
                "particle_contact_offset_m",
                "fluid_rest_offset_m",
                "contact_offset_m",
                "rest_offset_m",
                "solid_rest_offset_m",
                "solver_position_iteration_count",
                "max_neighborhood",
            )
        )
        or not isinstance(system.get("pbd_material_readback"), Mapping)
        or any(
            not _finite_number(system["pbd_material_readback"].get(key))
            for key in (
                "density_kg_m3",
                "friction",
                "cohesion",
                "damping",
                "surface_tension",
                "viscosity",
                "vorticity_confinement",
            )
        )
    ):
        failures.append("particle_postprocessing_contract")
    limitations = receipt.get("render_limitations")
    if not isinstance(limitations, Mapping) or any(limitations.get(key) is not value for key, value in RENDER_LIMITATIONS.items()):
        failures.append("render_limitations")
    instancer = receipt.get("point_instancer")
    if (
        not isinstance(instancer, Mapping)
        or not isinstance(instancer.get("path"), str)
        or not instancer.get("path", "").endswith("/FluidParticles/Shared")
        or instancer.get("max_particles") != 2221
        or instancer.get("ids_stable") is not True
        or instancer.get("simulation_points_authored") is not True
    ):
        failures.append("point_instancer_contract")
    mass_readback = (
        instancer.get("mass_density_readback") if isinstance(instancer, Mapping) else None
    )
    if (
        not isinstance(mass_readback, Mapping)
        or mass_readback.get("mass_api_present") is not True
        or not _finite_number(mass_readback.get("total_mass_attr_kg"))
        or not _finite_number(mass_readback.get("density_attr_kg_m3"))
        or not isinstance(mass_readback.get("precedence"), str)
        or not isinstance(mass_readback.get("source"), str)
    ):
        failures.append("particle_mass_density_evidence")
    captures = receipt.get("captures")
    if not isinstance(captures, list):
        failures.append("captures")
    elif any(
        not isinstance(item, Mapping)
        or int(item.get("emitted_particle_count", -1))
        != int(item.get("active_particle_count", -2))
        or int(item.get("active_particle_count", -1))
        != int(item.get("inside_vial_interior_count", -1))
        + int(item.get("in_flight_above_mouth_count", -1))
        + int(item.get("outside_or_escaped_particle_count", -1))
        or int(item.get("outside_vial_interior_count", -1))
        != int(item.get("expected_in_flight_stream_count", -1))
        + int(item.get("leaked_particle_count", -1))
        for item in captures
    ):
        failures.append("containment_partition")
    elif any(item.get("leakage_detected") is True for item in captures if isinstance(item, Mapping)):
        failures.append("particle_leakage")
    if isinstance(captures, list):
        required_volume_records = [
            item
            for item in captures
            if isinstance(item, Mapping)
            and isinstance(item.get("volume_gate"), Mapping)
            and item["volume_gate"].get("required") is True
        ]
        state_gate_failed = not required_volume_records or any(
            item.get("exact_polygon_containment") is not True
            or item.get("finite_state") is not True
            or item.get("contained_z_min_m") is None
            or item.get("contained_z_q05_m") is None
            or item.get("contained_z_q95_m") is None
            or item.get("contained_z_max_m") is None
            or not isinstance(item.get("point_instancer_scale_component_min"), list)
            or len(item.get("point_instancer_scale_component_min", [])) != 3
            or not isinstance(item.get("point_instancer_scale_component_max"), list)
            or len(item.get("point_instancer_scale_component_max", [])) != 3
            or item["volume_gate"].get("state_passed") is not True
            or not isinstance(
                item["volume_gate"].get("smoothed_render_state"), Mapping
            )
            or item["volume_gate"]["smoothed_render_state"].get(
                "particle_bulk_height_m"
            )
            is None
            for item in required_volume_records
        )
        if state_gate_failed:
            failures.append("volume_state_gate")
        visible_gate_failed = not required_volume_records or any(
            not isinstance(
                item["volume_gate"].get("visible_mask_evidence"), Mapping
            )
            or item["volume_gate"].get("visible_passed") is not True
            for item in required_volume_records
        )
        if visible_gate_failed:
            failures.append("volume_visible_gate")
        if required_volume_records and any(
            item["volume_gate"].get("passed")
            is not (
                item["volume_gate"].get("state_passed") is True
                and item["volume_gate"].get("visible_passed") is True
            )
            or (
                item["volume_gate"].get("passed") is True
                and item["volume_gate"].get("failure_reasons") != []
            )
            for item in required_volume_records
        ):
            failures.append("volume_gate_contract")

    qualification = receipt.get("qualification")
    if (
        isinstance(qualification, Mapping)
        and qualification.get("first_stage_only") is True
        and (
            qualification.get("volume_state_gate_passed") is not True
            or qualification.get("volume_visible_gate_passed") is not True
            or qualification.get("volume_gate_passed") is not True
        )
    ):
        failures.append("qualification_volume_gate")

    semantics = receipt.get("semantics")
    paired = semantics.get("paired_capture_contract") if isinstance(semantics, Mapping) else None
    if (
        not isinstance(paired, Mapping)
        or paired.get("shared_single_render_pass") is not False
        or paired.get("paired_same_physics_state") is not True
        or paired.get("kept_rgb_uses_source_equivalent_glass") is not True
        or paired.get("semantic_pass_uses_transparent_annotation_flag") is not True
        or paired.get("flag_restored_false_after_semantic_pass") is not True
        or paired.get("state_fields_required_unchanged")
        != list(PAIRED_CAPTURE_STATE_FIELDS)
    ):
        failures.append("paired_capture_contract")

    annotation = (
        semantics.get("transparent_annotation_override")
        if isinstance(semantics, Mapping)
        else None
    )
    source = annotation.get("source") if isinstance(annotation, Mapping) else None
    initial = (
        annotation.get("initial_rgb_state") if isinstance(annotation, Mapping) else None
    )
    if (
        not isinstance(annotation, Mapping)
        or annotation.get("capture_mode")
        != "paired_same_physics_state_rgb_then_semantic"
        or annotation.get("documentation_url")
        != TRANSPARENT_ANNOTATION_DOCUMENTATION_URL
        or annotation.get("semantic_pass_value") is not True
        or annotation.get("rgb_and_restore_value") is not False
        or not isinstance(source, Mapping)
        or (
            source.get("value") is not None
            and source.get("value") is not False
        )
        or not isinstance(initial, Mapping)
        or initial.get("composed_value") is not False
        or initial.get("property_stack_strongest_is_session_layer") is not True
        or initial.get("source_equivalent") is not True
        or annotation.get("property_stack_strongest_is_session_layer") is not True
        or annotation.get("authored_after_untouched_reference_capture") is not True
        or annotation.get("source_layer_unchanged_expected") is not True
    ):
        failures.append("transparent_annotation_contract")
    transitions = annotation.get("transitions") if isinstance(annotation, Mapping) else None
    if not isinstance(transitions, list) or not transitions:
        failures.append("transparent_annotation_restore")
    else:
        transition_values = []
        transitions_valid = True
        for transition in transitions:
            if not isinstance(transition, Mapping):
                transitions_valid = False
                continue
            capture_pass = transition.get("capture_pass")
            composed = transition.get("composed_value")
            expected = {
                "kept_rgb_source_equivalent": False,
                "semantic_transparent_annotation": True,
                "restore_after_semantic": False,
            }.get(capture_pass)
            if (
                expected is None
                or transition.get("requested_value") is not expected
                or transition.get("composed_type_name") != "bool"
                or composed is not expected
                or transition.get("property_stack_strongest_is_session_layer")
                is not True
            ):
                transitions_valid = False
            transition_values.append((capture_pass, composed))
        if (
            not transitions_valid
            or not any(
                capture_pass == "semantic_transparent_annotation"
                and composed is True
                for capture_pass, composed in transition_values
            )
            or transition_values[-1] != ("restore_after_semantic", False)
            or sum(
                capture_pass == "semantic_transparent_annotation"
                for capture_pass, _ in transition_values
            )
            != sum(
                capture_pass == "restore_after_semantic"
                for capture_pass, _ in transition_values
            )
        ):
            failures.append("transparent_annotation_restore")

    timing = receipt.get("timing")
    capture_guards = (
        timing.get("capture_guard_records") if isinstance(timing, Mapping) else None
    )
    state_records = paired.get("state_records") if isinstance(paired, Mapping) else None
    if not isinstance(capture_guards, list) or not capture_guards:
        failures.append("paired_capture_guards")
    if not isinstance(state_records, list) or not state_records:
        failures.append("paired_capture_state_drift")
    elif isinstance(capture_guards, list) and any(
        not _paired_state_record_valid(record, capture_guards)
        for record in state_records
    ):
        failures.append("paired_capture_state_drift")
    if isinstance(state_records, list) and isinstance(transitions, list) and any(
        not all(
            any(
                isinstance(transition, Mapping)
                and transition.get("capture_pass") == capture_pass
                and transition.get("physics_step") == record.get("physics_step")
                and transition.get("phase") == record.get("phase")
                and transition.get("attempt") == record.get("attempt")
                for transition in transitions
            )
            for capture_pass in (
                "kept_rgb_source_equivalent",
                "semantic_transparent_annotation",
                "restore_after_semantic",
            )
        )
        for record in state_records
        if isinstance(record, Mapping)
    ):
        failures.append("transparent_annotation_restore")

    pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
    if not isinstance(pairs, list) or not pairs or any(
        not isinstance(pair, Mapping)
        or pair.get("shared_rgb_semantic_capture") is True
        or pair.get("paired_same_physics_state_rgb_semantic_capture") is not True
        or pair.get("paired_capture_state_passed") is not True
        for pair in pairs
    ):
        failures.append("paired_capture_pairs")
    # Isosurface and canonical-body paths must not be part of this receipt.
    if receipt.get("isosurface") is not None or receipt.get("mesh_column") is not None:
        failures.append("retired_render_path_present")
    if (
        isinstance(qualification, Mapping)
        and qualification.get("first_stage_only") is False
    ):
        failures.extend(_validate_full_run_contract(receipt))
    return tuple(dict.fromkeys(failures))


__all__ = [
    "MAX_VELOCITY_M_S",
    "PAINT_BALL_RECEIPT_SCHEMA_VERSION",
    "PAIRED_CAPTURE_STATE_FIELDS",
    "TRANSPARENT_ANNOTATION_DOCUMENTATION_URL",
    "EMITTER_RENDER_PATH",
    "append_point_instancer_state",
    "attach_visible_volume_evidence",
    "build_palette_material_path",
    "build_prototype_specs",
    "containment_evidence",
    "build_capture_record",
    "validate_paint_ball_receipt",
]
