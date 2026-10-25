#!/usr/bin/env python3
"""Run one bounded GPU PhysX three-pour fallback and preserve visual evidence.

This is a synthetic simulator experiment.  Programmed millilitres select a
documented nominal particle count; they are not measured displaced volume.
PhysX supplies one-system fluid motion and an isosurface. Tracked per-particle
compositions move with that motion and exchange with nearby particles, but the
spatial composition is not rendered. Source-coloured render-only streams and a
global mean-composition vial colour are explicit fallback visualization rules;
no absorbance fitting occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.physx_three_pour import (
    CAMERA_PATH,
    CAMERA_CLIPPING_RANGE,
    COMPOSITION_COMPONENTS,
    FLUIDMIX_NAMESPACE,
    GENERATED_ISOSURFACE_PATH,
    LIQUID_PROXY_PATH,
    NOMINAL_TANK_TRANSMISSION_COLORS,
    NOZZLE_EXIT_CENTER_M,
    O1B_MATERIAL_PARAMETERS,
    PARITY_THRESHOLDS,
    SOURCE_LIQUID_BOTTOM_M,
    SOURCE_LIQUID_PATH,
    SOURCE_LIQUID_RADIUS_M,
    SOURCE_NOZZLE_PATH,
    SOURCE_STREAM_PATH,
    SOURCE_VIAL_PATH,
    STAGE_GPU_MEMORY_GROWTH_STOP_MIB,
    TWIN_REFERENCE_TIME_CODE,
    VIAL_CENTER_XY_M,
    VIAL_MOUTH_M,
    VISIBILITY_CHECK_TIME_CODES,
    build_twin_collider_specs,
    build_three_pour_plan,
    distribute_particle_count,
    validate_three_pour_receipt,
)
from fluidmix.particle_diffusion import (
    DIFFUSION_DT_S,
    DIFFUSION_NEIGHBOR_RADIUS_M,
    EFFECTIVE_DIFFUSIVITY_M2_S,
    diffuse_particle_compositions_with_receipt,
)
from fluidmix.particle_palette import build_particle_palette, mix_linear_rgb
from fluidmix.render_capture import exact_semantic_class_mask
from fluidmix.isosurface_mesh_proxy import (
    PROXY_RGB_PARITY_THRESHOLDS,
    build_liquid_proxy_intent,
    copy_exact_mesh_proxy,
    evaluate_proxy_rgb_parity_for_source_hide,
    refresh_exact_mesh_proxy,
)
from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry

# Reuse the F1 routines that were exercised on both GB10 and Brev.  This script
# must be copied beside physx_particle_compat_probe.py on the remote host.
from physx_particle_compat_probe import (
    _finite_positions,
    _movement,
    _positions,
    _safe_call,
    _version_info,
)


WALL_LIMIT_SECONDS = 5400
ISOSURFACE_LIMIT_SECONDS = 2700
GPU_MEMORY_LIMIT_MIB = 40000
PHYSICS_HZ = 60
SUPPORTED_COLLIDER_TYPE = "Cube"
# Isaac Sim 6.0.1 documents skipped Replicator frames when a stopped timeline
# lets isaacsim.core.throttling enable asynchronous rendering.  Keep F2 on the
# same launch contract as the already exercised FillCam renderer.
REPLICATOR_ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
REPLICATOR_EXTRA_ARGS = (f"--{REPLICATOR_ASYNC_SETTING}=false",)
SYNTHETICDATA_CAPTURE_VOLUMES_SETTING = "/syntheticdata/sensors/captureVolumes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--renderer", choices=("PathTracing",), default="PathTracing")
    parser.add_argument("--rt-subframes", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument(
        "--max-capture-tries",
        type=int,
        default=20,
        help="maximum same-state renderer attempts for one kept RGB frame",
    )
    parser.add_argument("--capture-stride", type=int, default=6)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, int]:
    try:
        width_text, height_text = args.resolution.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("resolution must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    if (width, height) != (640, 480):
        raise ValueError("twin F2 resolution must be 640x480")
    if not args.stage.is_absolute() or not args.stage.is_file():
        raise ValueError("stage must be an existing absolute file")
    for name in (
        "rt_subframes",
        "warmup",
        "max_capture_tries",
        "capture_stride",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("video_fps",):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("seed must be an integer")
    if args.rt_subframes != 64 or args.warmup != 60:
        raise ValueError("twin F2 requires rt_subframes=64 and warmup=60")
    if args.capture_stride != 6:
        raise ValueError("twin F2 capture_stride must be 6 for the frozen 10 Hz timeline")
    if not math.isclose(float(args.video_fps), 10.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("twin F2 video_fps must be 10 Hz")
    return width, height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _build_emission_event_index(plan: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Index the exact frozen emission events by physics step."""

    indexed: dict[int, list[dict[str, Any]]] = {}
    for pour in plan["pours"]:
        for event in pour["emission_schedule"]:
            step = int(event["physics_step"])
            indexed.setdefault(step, []).append(
                {
                    "tank_id": pour["tank_id"],
                    "pour_ordinal": pour["ordinal"],
                    **dict(event),
                }
            )
    return indexed


def _active_tank_for_step(plan: dict[str, Any], step: int) -> str | None:
    """Return the programmed source tank for a back-to-back pour step."""

    for pour in plan["pours"]:
        if int(pour["start_step"]) <= step <= int(pour["end_step"]):
            return str(pour["tank_id"])
    return None


def _build_capture_steps(plan: dict[str, Any], *, stride: int) -> tuple[int, ...]:
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("capture stride must be a positive integer")
    final_step = int(plan["timing"]["primary_final_capture_step"])
    stage_steps = {int(pour["stage_capture_step"]) for pour in plan["pours"]}
    return tuple(sorted({*range(stride, final_step + 1, stride), *stage_steps}))


def _update_global_component_counts(
    counts: Any, tank_id: str, particle_count: int
) -> Any:
    """Add one source event without mutating the cumulative component counts."""

    import numpy as np

    values = np.asarray(counts)
    if values.dtype.kind not in "fiu" or values.shape != (len(COMPOSITION_COMPONENTS),):
        raise TypeError("component counts must be a numeric length-three vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("component counts must be finite and non-negative")
    if tank_id not in COMPOSITION_COMPONENTS:
        raise ValueError(f"unknown tank {tank_id!r}")
    if isinstance(particle_count, bool) or not isinstance(particle_count, int):
        raise TypeError("particle_count must be an integer")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    result = np.asarray(values, dtype=np.float64).copy()
    result[COMPOSITION_COMPONENTS.index(tank_id)] += particle_count
    return result


def _global_transmission_color(component_counts: Any) -> Any:
    """Linearly blend uncalibrated tank transmission parameters by count."""

    import numpy as np

    counts = np.asarray(component_counts, dtype=np.float64)
    if counts.shape != (len(COMPOSITION_COMPONENTS),) or not np.all(np.isfinite(counts)):
        raise ValueError("component counts must be a finite length-three vector")
    total = float(np.sum(counts))
    if total <= 0.0 or np.any(counts < 0.0):
        raise ValueError("component counts must have a positive non-negative total")
    colors = np.asarray(
        [NOMINAL_TANK_TRANSMISSION_COLORS[tank] for tank in COMPOSITION_COMPONENTS],
        dtype=np.float64,
    )
    return mix_linear_rgb((counts / total)[None, :], colors)[0]


def _gpu_used_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _gb10_uma_memory_expected() -> bool:
    """Return true only for NVIDIA GB10's documented unsupported FB-memory query."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(
        "NVIDIA GB10" in line and line.count("[N/A]") >= 2
        for line in completed.stdout.splitlines()
    )


def _parse_meminfo_used_mib(text: str) -> int:
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"MemTotal:", "MemAvailable:"}:
            values[fields[0]] = int(fields[1])
    if "MemTotal:" not in values or "MemAvailable:" not in values:
        raise ValueError("MemTotal and MemAvailable are required")
    used_kib = values["MemTotal:"] - values["MemAvailable:"]
    if used_kib < 0:
        raise ValueError("MemAvailable exceeds MemTotal")
    return used_kib // 1024


def _system_used_memory_mib() -> int | None:
    try:
        return _parse_meminfo_used_mib(Path("/proc/meminfo").read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError):
        return None


def _open_video_writer(
    cv2: Any,
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> tuple[Any, Path, str, bool]:
    """Open the preferred MP4 writer or the measured GB10-compatible AVI fallback."""

    candidates = (
        ("three-pour.mp4", "mp4v", False),
        ("three-pour.avi", "MJPG", True),
    )
    for filename, codec, fallback in candidates:
        path = output / filename
        writer = cv2.VideoWriter(
            os.fspath(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer, path, codec, fallback
        writer.release()
    raise RuntimeError("OpenCV mp4v or MJPG video writer did not open")


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_three_pour_plan()
    script_path = Path(__file__).resolve()
    contract_path = ROOT / "src" / "fluidmix" / "physx_three_pour.py"
    return {
        "schema_version": plan["schema_version"],
        "experiment": "physx_three_pour_f2",
        "status": "insufficient_evidence",
        "plan": plan,
        "arguments": {
            "stage": str(args.stage),
            "seed": args.seed,
            "resolution": args.resolution,
            "renderer": args.renderer,
            "rt_subframes": args.rt_subframes,
            "warmup": args.warmup,
            "max_capture_tries": args.max_capture_tries,
            "capture_stride": args.capture_stride,
            "video_fps": args.video_fps,
        },
        "provenance": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "script_path": str(script_path),
            "script_sha256": _sha256(script_path),
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": _sha256(contract_path),
            "requested_stage_path": str(args.stage.resolve()),
            "requested_stage_sha256": _sha256(args.stage),
        },
        "versions": {"isaac": None, "kit": None, "physx": None},
        "gpu_simulation": {
            "cuda_device_available": False,
            "scene_enable_gpu_dynamics": False,
            "broadphase": None,
            "velocity_writeback_initial_readback": None,
            "velocity_writeback_enabled_readback": False,
            "velocity_writeback_restored_readback": None,
        },
        "particle_system": {
            "count": 0,
            "exactly_one": False,
            "particle_set_count": 0,
            "mass_api_count": None,
            "all_particle_sets_target_one_system": False,
            "collider_types": [],
            "collider_paths": [],
            "all_colliders_invisible": False,
            "decorative_vial_collision_api": None,
            "pbd_material_binding": False,
            "render_material_binding": False,
            "ccd_enabled": False,
            "finite_motion_observed": False,
            "maximum_particle_movement_m": 0.0,
            "runtime_particle_ids_authored": False,
            "runtime_particle_ids_exact_each_callback": False,
            "runtime_particle_identity_check_count": 0,
            "runtime_particle_id_mismatch_count": 0,
            "runtime_particle_ids_final_sha256": None,
        },
        "scene_contract": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "frozen_time_code": TWIN_REFERENCE_TIME_CODE,
            "physics_namespace": FLUIDMIX_NAMESPACE,
            "source_layer_records": [],
            "source_layers_unchanged": False,
            "hidden_source_visibility": {},
            "forbidden_authored_paths": [],
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
        },
        "camera": {
            "path": CAMERA_PATH,
            "source_has_authored_clipping_range": False,
            "clipping_range_authored_by_runner": False,
            "clipping_range_effective": None,
            "resolution": [640, 480],
        },
        "material": {
            "pbd_path": f"{FLUIDMIX_NAMESPACE}/PBDMaterial",
            "global_liquid_path": f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
            "vial_representation": "exact_standard_mesh_proxy_one_global_material",
            "stream_representation": "render_only_mesh_current_tank_material",
            "omnisurface_parameter_family": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in O1B_MATERIAL_PARAMETERS.items()
                if key != "specular_transmission_color"
            },
            "tank_transmission_colors": {
                tank: list(color)
                for tank, color in NOMINAL_TANK_TRANSMISSION_COLORS.items()
            },
            "transmission_colors_uncalibrated": True,
            "global_blend_rule": "cumulative_particle_weighted_linear_transmission_parameter",
            "isosurface_global_material_binding": False,
        },
        "emitter": {
            "source_nozzle_path": SOURCE_NOZZLE_PATH,
            "source_nozzle_valid": False,
            "origin_m": list(NOZZLE_EXIT_CENTER_M),
            "velocity_m_s": [0.0, 0.0, -0.1],
        },
        "pre_pour_parity": {
            "thresholds": dict(PARITY_THRESHOLDS),
            "reference_path": None,
            "reference_sha256": None,
            "f2_prefill_path": None,
            "f2_prefill_sha256": None,
            "absdiff_path": None,
            "absdiff_sha256": None,
            "mean_abs_rgb": None,
            "p95_abs_rgb": None,
            "fraction_abs_rgb_gt_16": None,
            "passed": False,
        },
        "pours": [],
        "fallback_evidence": dict(plan["fallback_trigger"]),
        "stream_render": {
            "representation": "render_only_mesh_current_tank_material",
            "records": [],
        },
        "global_blend": {
            "model": plan["mixing_model"],
            "component_order": list(COMPOSITION_COMPONENTS),
            "global_blend_rule": plan["fallback_render_model"]["global_blend_rule"],
            "blend_update_timing": "every_emission_event",
            "blend_update_count": 0,
            "updated_on_every_emission_event": False,
            "all_updates_usd_readback_verified": False,
            "particle_composition_tracked": True,
            "diffusion_simulated": True,
            "composition_state": "internal_unrendered",
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "stage_states": [],
            "emission_trace_event_count": 0,
            "emission_trace_path": None,
            "emission_trace_sha256": None,
        },
        "composition_evidence": {
            "schema_version": 1,
            "component_order": list(COMPOSITION_COMPONENTS),
            "internal_unrendered": True,
            "particle_id_scheme": "zero_based_emission_order_stable_row_identity",
            "diffusion": {
                "coefficient_m2_s": EFFECTIVE_DIFFUSIVITY_M2_S,
                "neighbor_radius_m": DIFFUSION_NEIGHBOR_RADIUS_M,
                "dt_s": DIFFUSION_DT_S,
            },
            "diffusion_update_count": 0,
            "diffusion_neighbor_count_total": 0,
            "diffusion_max_degree": 0,
            "diffusion_pair_weight_max": 0.0,
            "stage_count": 0,
            "stages": [],
        },
        "totals": {
            "programmed_volume_ml": 60,
            "emitted_particle_count": 0,
            "final_particle_count": 0,
            "nominal_mass_convention_equivalent_ml": plan["totals"]["nominal_mass_convention_equivalent_ml"],
            "nominal_error_ml": plan["totals"]["nominal_error_ml"],
        },
        "isosurface": {
            "enabled": False,
            "visible_for_global_blend": False,
            "mesh_point_count": 0,
            "mesh_distinct_point_count": 0,
            "mesh_face_count": 0,
            "mesh_corner_count": 0,
            "mesh_finite_points": False,
            "mesh_topology_consistent": False,
            "mesh_indices_valid": False,
            "mesh_nonzero_area_face_count": 0,
            "mesh_all_faces_nonzero_area": False,
            "mesh_extent_m": [None, None, None],
            "mesh_extent_nonempty": False,
            "mesh_geometry_valid": False,
            "mesh_geometry_failures": ["not_observed"],
        },
        "mesh_proxy": {
            "source_path": GENERATED_ISOSURFACE_PATH,
            "proxy_path": LIQUID_PROXY_PATH,
            "material_path": f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
            "created": False,
            "creation_step": None,
            "creation_evidence": None,
            "creation_artifact_path": None,
            "creation_artifact_sha256": None,
            "rgb_parity": {
                "status": "not_run",
                "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
                "hide_source_allowed": False,
            },
            "refresh_records": [],
            "source_hidden_at_all_vial_proxy_captures": False,
            "proxy_visible_at_all_vial_proxy_captures": False,
            "one_shared_rgb_semantic_capture_per_vial_proxy_frame": False,
            "claim_limits": {
                "exact_render_proxy_only": True,
                "hand_authored_polygon": False,
                "physical_validation": False,
                "training_eligible": False,
            },
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "annotator": "semantic_segmentation",
            "annotator_init_params": {
                "colorize": False,
                "semanticTypes": ["class"],
            },
            "mask_source": "exact_renderer_semantic_class_ids",
            "resolution": [640, 480],
            "system_binding_path": f"{FLUIDMIX_NAMESPACE}/ParticleSystem",
            "stream_binding_paths": [],
            "material_binding_authored": False,
            "material_binding": None,
            "generated_isosurface_path": GENERATED_ISOSURFACE_PATH,
            "isosurface_path": LIQUID_PROXY_PATH,
            "required_isosurface_path": LIQUID_PROXY_PATH,
            "allowed_stream_paths": [
                f"{FLUIDMIX_NAMESPACE}/RenderStreams/{tank_id.upper()}"
                for tank_id in COMPOSITION_COMPONENTS
            ],
            "isosurface_direct_binding_authored": False,
            "isosurface_direct_binding_step": None,
            "isosurface_direct_binding": None,
            "isosurface_semantic_bound": False,
            "stream_semantic_bound": False,
            "annotator_attached_after_stream_labels": False,
            "pairs": [],
            "stage_mask_sha256": [],
            "semantic_gate_passed": False,
            "frame_semantic_paths": [],
            "frame_semantic_sha256": [],
            "training_eligible": False,
            "ineligibility_reason": "qualification_run_not_case_dataset",
            "capture_volumes_setting_path": SYNTHETICDATA_CAPTURE_VOLUMES_SETTING,
            "capture_volumes_initial_readback": None,
            "capture_volumes_enabled_readback": False,
            "capture_volumes_restored_readback": None,
        },
        "render": {
            "renderer": args.renderer,
            "resolution": [640, 480],
            "rt_subframes": args.rt_subframes,
            "warmup_updates": args.warmup,
            "frame_count": 0,
            "frame_paths": [],
            "frame_sha256": [],
            "frame_physics_steps": [],
            "stage_frame_paths": [],
            "stage_frame_sha256": [],
            "stage_frame_indices": [],
            "stage_frame_physics_steps": [],
            "primary_final_frame_sha256": None,
            "video_path": None,
            "video_sha256": None,
            "video_codec": None,
            "video_container": None,
            "video_writer_fallback_used": None,
            "video_decodable": False,
            "video_decoded_frame_count": 0,
            "stage_frames_are_video_subsets": False,
            "stream_only_frame_count": 0,
            "vial_proxy_frame_count": 0,
            "vial_proxy_transition_step": None,
            "liquid_representation_monotonic": False,
            "max_capture_tries": args.max_capture_tries,
            "capture_attempt_count": 0,
            "capture_attempts": [],
            "capture_on_play": False,
            "wait_for_render": True,
            "same_state_delta_time": 0.0,
            "async_throttling_enabled": None,
            "rejected_rgb_diagnostics": [],
            "rejected_semantic_diagnostics": [],
        },
        "memory": {
            "hard_limit_mib": GPU_MEMORY_LIMIT_MIB,
            "peak_used_mib": None,
            "hard_limit_exceeded": False,
            "sample_source": None,
            "sample_scope": None,
            "stage_growth_stop_mib": STAGE_GPU_MEMORY_GROWTH_STOP_MIB,
            "stage_growth_events": [],
            "samples_mib": [],
            "isosurface_enabled_seconds": None,
            "isosurface_time_limit_seconds": ISOSURFACE_LIMIT_SECONDS,
        },
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "capture_hz": plan["timing"]["capture_hz"],
            "pour_duration_steps": [pour["duration_steps"] for pour in plan["pours"]],
            "pour_end_steps": [pour["end_step"] for pour in plan["pours"]],
            "no_inter_pour_settle": True,
            "fill_end_step": plan["timing"]["fill_end_step"],
            "primary_final_capture_step": plan["timing"]["primary_final_capture_step"],
            "primary_final_offset_s": plan["timing"]["primary_final_offset_s"],
            "executed_physics_steps": 0,
            "simulation_owner": "timeline",
            "callbacks_expected": plan["timing"]["primary_final_capture_step"],
            "callbacks_observed": 0,
            "callback_indices": [],
            "callback_dts_s": [],
            "callback_times_s": [],
            "callback_app_update_indices": [],
            "one_callback_per_app_update": False,
            "emissions_authored_on_main_thread": False,
            "emission_boundary_verified": False,
            "emission_boundaries": [],
            "points_session_layer_only": False,
            "velocities_session_layer_only": False,
            "widths_session_layer_only": False,
            "ids_session_layer_only": False,
            "capture_callbacks_advanced": None,
            "capture_timeline_time_advanced": None,
            "capture_playback_preserved": None,
            "capture_guard_records": [],
            "direct_physx_api_used": False,
            "source": plan["timing"]["source"],
            "source_fit_r2": plan["timing"]["source_fit_r2"],
            "source_fit_rmse_s": plan["timing"]["source_fit_rmse_s"],
            "source_fit_loocv_rmse_s": plan["timing"]["source_fit_loocv_rmse_s"],
            "interpretation": plan["timing"]["interpretation"],
            "wall_seconds": None,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
        },
        "mixing_assumption": plan["mixing_assumption"],
        "claims": dict(plan["claims"]),
        "errors": [],
        "warnings": [],
    }


def _memory_sample(receipt: dict[str, Any]) -> int:
    value = _gpu_used_memory_mib()
    source = "nvidia_smi_gpu_memory_used"
    scope = "gpu_framebuffer"
    if value is None and _gb10_uma_memory_expected():
        value = _system_used_memory_mib()
        source = "proc_meminfo_system_used_uma"
        scope = "system_wide_unified_memory"
        if receipt["memory"]["sample_source"] is None:
            receipt["warnings"].append(
                "NVIDIA GB10 reports nvidia-smi FB memory as unsupported; "
                "the 40000 MiB safety stop uses conservative system-wide "
                "MemTotal-MemAvailable because CPU and GPU share UMA"
            )
    if value is None:
        raise RuntimeError("nvidia-smi did not return a GPU memory sample")
    prior_source = receipt["memory"]["sample_source"]
    prior_scope = receipt["memory"]["sample_scope"]
    if prior_source not in (None, source) or prior_scope not in (None, scope):
        raise RuntimeError("memory sample source changed within one application")
    receipt["memory"]["sample_source"] = source
    receipt["memory"]["sample_scope"] = scope
    receipt["memory"]["samples_mib"].append(value)
    peak = receipt["memory"]["peak_used_mib"]
    receipt["memory"]["peak_used_mib"] = value if peak is None else max(peak, value)
    if value >= GPU_MEMORY_LIMIT_MIB:
        receipt["memory"]["hard_limit_exceeded"] = True
        raise RuntimeError(f"GPU memory safety stop reached {value} MiB")
    return value


def _author_omnisurface(
    stage: Any,
    material: Any,
    color: tuple[float, float, float],
    Sdf: Any,
    Gf: Any,
    UsdShade: Any,
) -> Any:
    shader = UsdShade.Shader.Define(
        stage, f"{material.GetPath()}/OmniSurface"
    )
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_transmission_color": (
            Sdf.ValueTypeNames.Color3f,
            Gf.Vec3f(*color),
        ),
        "specular_transmission_scattering_depth": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_transmission_scattering_depth_m"],
        ),
        "thin_walled": (Sdf.ValueTypeNames.Bool, False),
        "diffuse_reflection_weight": (Sdf.ValueTypeNames.Float, 0.0),
        "specular_reflection_ior": (Sdf.ValueTypeNames.Float, 1.333),
        "specular_reflection_roughness": (Sdf.ValueTypeNames.Float, 0.08),
        "enable_opacity": (Sdf.ValueTypeNames.Bool, False),
        "geometry_opacity": (Sdf.ValueTypeNames.Float, 1.0),
        "geometry_opacity_threshold": (Sdf.ValueTypeNames.Float, 0.0),
    }
    for name, (type_name, value) in values.items():
        shader.CreateInput(name, type_name).Set(value)
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)
    material.CreateVolumeOutput("mdl").ConnectToSource(shader_output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader_output)
    return shader


def _set_global_blend_color(
    shader: Any, Gf: Any, color: Any
) -> tuple[float, float, float]:
    material_input = shader.GetInput("specular_transmission_color")
    requested = tuple(float(value) for value in color)
    if not material_input.Set(Gf.Vec3f(*requested)):
        raise RuntimeError("could not update the explicit global vial blend color")
    effective = material_input.Get()
    if effective is None:
        raise RuntimeError("global vial blend color has no live USD readback")
    observed = tuple(float(effective[index]) for index in range(3))
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected in zip(observed, requested)
    ):
        raise RuntimeError(
            f"global vial blend USD readback mismatch: {observed!r} != {requested!r}"
        )
    return observed


def _with_session_edit_context(
    Usd: Any,
    stage: Any,
    session_layer: Any,
    action: Any,
) -> Any:
    """Run one live USD mutation against the experiment's strongest layer."""

    with Usd.EditContext(stage, session_layer):
        return action()


class _UsdMeshProxyAdapter:
    """Kit adapter for the pure exact-mesh proxy contract."""

    def __init__(
        self,
        *,
        stage: Any,
        session_layer: Any,
        Usd: Any,
        UsdGeom: Any,
        UsdShade: Any,
        UsdSemantics: Any,
        Gf: Any,
        Vt: Any,
    ) -> None:
        self.stage = stage
        self.session_layer = session_layer
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.UsdShade = UsdShade
        self.UsdSemantics = UsdSemantics
        self.Gf = Gf
        self.Vt = Vt

    def read_mesh(self, path: str) -> dict[str, object]:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(self.UsdGeom.Mesh):
            return {"schema_type": str(prim.GetTypeName()) if prim.IsValid() else None}
        mesh = self.UsdGeom.Mesh(prim)
        matrix = self.UsdGeom.Xformable(prim).GetLocalTransformation()
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        return {
            "schema_type": "Mesh",
            "points": [tuple(float(value) for value in row) for row in (mesh.GetPointsAttr().Get() or [])],
            "face_vertex_counts": [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])],
            "face_vertex_indices": [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])],
            "local_transform": [
                [float(matrix[row][column]) for column in range(4)]
                for row in range(4)
            ],
        }

    def prim_exists(self, path: str) -> bool:
        return bool(self.stage.GetPrimAtPath(path).IsValid())

    def define_standard_mesh(self, path: str) -> None:
        with self.Usd.EditContext(self.stage, self.session_layer):
            self.UsdGeom.Mesh.Define(self.stage, path)

    def write_mesh(self, path: str, snapshot: Any) -> None:
        with self.Usd.EditContext(self.stage, self.session_layer):
            mesh = self.UsdGeom.Mesh(self.stage.GetPrimAtPath(path))
            if not mesh or not mesh.GetPrim().IsValid():
                raise RuntimeError("standard mesh proxy is missing during write")
            mesh.GetPointsAttr().Set(
                self.Vt.Vec3fArray(
                    [self.Gf.Vec3f(*point) for point in snapshot.points]
                )
            )
            mesh.GetFaceVertexCountsAttr().Set(
                self.Vt.IntArray(snapshot.face_vertex_counts)
            )
            mesh.GetFaceVertexIndicesAttr().Set(
                self.Vt.IntArray(snapshot.face_vertex_indices)
            )
            matrix = self.Gf.Matrix4d(*snapshot.local_transform)
            self.UsdGeom.Xformable(mesh.GetPrim()).MakeMatrixXform().Set(matrix)
            point_rows = list(snapshot.points)
            minimum = [min(row[index] for row in point_rows) for index in range(3)]
            maximum = [max(row[index] for row in point_rows) for index in range(3)]
            mesh.CreateExtentAttr().Set(
                [self.Gf.Vec3f(*minimum), self.Gf.Vec3f(*maximum)]
            )

    def bind_material(self, path: str, material_path: str) -> None:
        with self.Usd.EditContext(self.stage, self.session_layer):
            prim = self.stage.GetPrimAtPath(path)
            material = self.UsdShade.Material(
                self.stage.GetPrimAtPath(material_path)
            )
            if not prim.IsValid() or not material or not material.GetPrim().IsValid():
                raise RuntimeError("proxy material binding target is missing")
            self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def bind_semantics(
        self, path: str, semantic_type: str, labels: tuple[str, ...]
    ) -> None:
        with self.Usd.EditContext(self.stage, self.session_layer):
            prim = self.stage.GetPrimAtPath(path)
            api = self.UsdSemantics.LabelsAPI.Apply(prim, semantic_type)
            if not api or not api.CreateLabelsAttr().Set(list(labels)):
                raise RuntimeError("could not author proxy semantic labels")

    def read_material_binding(self, path: str) -> object:
        bound, _relationship = self.UsdShade.MaterialBindingAPI(
            self.stage.GetPrimAtPath(path)
        ).ComputeBoundMaterial()
        return str(bound.GetPath()) if bound and bound.GetPrim().IsValid() else None

    def read_semantics(self, path: str, semantic_type: str) -> object:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        return tuple(
            sorted(
                str(value)
                for value in self.UsdSemantics.LabelsQuery(
                    semantic_type, self.Usd.TimeCode.Default()
                ).ComputeUniqueInheritedLabels(prim)
            )
        )


def _look_at_matrix(Gf: Any, eye: tuple[float, float, float], target: tuple[float, float, float]):
    # Camera local -Z points toward the target; inverse view gives the USD transform.
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 1.0, 0.0)).GetInverse()


def _author_and_attest_camera_clipping(
    camera: Any, Gf: Any, receipt: dict[str, Any]
) -> None:
    requested = list(CAMERA_CLIPPING_RANGE)
    receipt["camera"]["clipping_range_requested"] = requested
    attribute = camera.CreateClippingRangeAttr()
    set_ok = attribute.Set(Gf.Vec2f(*CAMERA_CLIPPING_RANGE))
    effective_value = camera.GetClippingRangeAttr().Get()
    effective = (
        [float(effective_value[0]), float(effective_value[1])]
        if effective_value is not None
        else None
    )
    authored = effective if attribute.HasAuthoredValueOpinion() else None
    receipt["camera"]["clipping_range_authored"] = authored
    receipt["camera"]["clipping_range_effective"] = effective
    if not set_ok or authored != requested or effective != requested:
        raise RuntimeError(
            "camera clipping range did not retain the requested live USD value"
        )


def _chunk_positions(Gf: Any, count: int, rng: random.Random) -> list[Any]:
    values = []
    spacing = 0.003
    for index in range(count):
        ix = index % 3
        iz = (index // 3) % 3
        iy = index // 9
        jitter_x = (rng.random() - 0.5) * 0.0003
        jitter_z = (rng.random() - 0.5) * 0.0003
        values.append(
            Gf.Vec3f(
                (ix - 1) * spacing + jitter_x,
                0.145 + iy * spacing,
                (iz - 1) * spacing + jitter_z,
            )
        )
    return values


def _substantive_isosurface_mesh_evidence(
    stage: Any, path: Any, UsdGeom: Any
) -> dict[str, Any]:
    """Inspect generated mesh bytes and reject PhysX's degenerate placeholder."""

    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    points = (mesh.GetPointsAttr().Get() or []) if mesh else []
    face_counts = (mesh.GetFaceVertexCountsAttr().Get() or []) if mesh else []
    face_indices = (mesh.GetFaceVertexIndicesAttr().Get() or []) if mesh else []
    evidence = {
        "prim_present": bool(prim),
        "mesh_is_mesh": bool(mesh),
    }
    evidence.update(
        analyze_triangle_mesh_geometry(points, face_counts, face_indices)
    )
    return evidence


def _diffuse_active_particle_rows(
    active_records: list[dict[str, Any]],
    compositions_by_id: dict[int, Any],
    positions_m: Any,
    observed_particle_ids: list[int],
) -> tuple[Any, dict[str, Any]]:
    """Diffuse rows in active-prim/row order and update the ID-keyed state.

    The runner deliberately treats a USD position readback as an ordered row
    contract.  A count, duplicate/missing ID, nonfinite position, or violated
    diffusion invariant stops the run rather than silently reassigning liquid
    composition to another particle.
    """

    import numpy as np

    expected_particle_ids: list[int] = []
    for record in active_records:
        ids = record.get("particle_ids")
        count = record.get("particle_count")
        if (
            not isinstance(ids, list)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or len(ids) != count
            or any(type(value) is not int for value in ids)
        ):
            raise RuntimeError("particle count/order contract failed")
        expected_particle_ids.extend(ids)
    if len(set(expected_particle_ids)) != len(expected_particle_ids):
        raise RuntimeError("particle count/order contract failed: duplicate particle ID")
    if (
        not isinstance(observed_particle_ids, list)
        or any(type(value) is not int for value in observed_particle_ids)
        or observed_particle_ids != expected_particle_ids
    ):
        raise RuntimeError("particle count/order contract failed: observed USD ids")
    particle_ids = list(observed_particle_ids)
    positions = np.asarray(positions_m, dtype=np.float64)
    if not particle_ids and positions.size == 0:
        positions = np.empty((0, 3), dtype=np.float64)
    if positions.shape != (len(particle_ids), 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("particle count/order contract failed: positions readback")
    if not particle_ids:
        before = np.empty((0, 3), dtype=np.float64)
    else:
        try:
            before = np.asarray(
                [compositions_by_id[particle_id] for particle_id in particle_ids],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("particle count/order contract failed: composition row missing") from exc
    if before.shape != (len(particle_ids), 3) or not np.all(np.isfinite(before)):
        raise RuntimeError("particle count/order contract failed: composition rows")
    if not particle_ids:
        particle_ids = np.asarray([], dtype=np.int64)
    try:
        after, diagnostics = diffuse_particle_compositions_with_receipt(
            particle_ids, positions, before
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("particle diffusion failed closed") from exc
    if (
        diagnostics.get("particle_count") != len(particle_ids)
        or diagnostics.get("particle_row_order_preserved") is not True
        or diagnostics.get("component_conservation_passed") is not True
        or diagnostics.get("row_sum_passed") is not True
        or diagnostics.get("nonnegative_passed") is not True
        or not isinstance(diagnostics.get("conservation_error"), (int, float))
        or not math.isfinite(float(diagnostics.get("conservation_error")))
        or float(diagnostics.get("conservation_error")) > 1.0e-12
        or not np.all(np.isfinite(after))
        or np.any(after < 0.0)
        or not np.allclose(after.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("particle diffusion violated count/order or conservation contract")
    for particle_id, row in zip(particle_ids, after):
        compositions_by_id[particle_id] = np.asarray(row, dtype=np.float64).copy()
    return after, diagnostics


def _float_array_sha256(values: Any) -> str:
    import numpy as np

    return hashlib.sha256(
        np.asarray(values, dtype="<f8", order="C").tobytes()
    ).hexdigest()


def _particle_ids_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = f"builtins.int:{value!r}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_runtime_particle_identity(
    particle_points: Any | None, expected_ids: list[int]
) -> list[int]:
    """Read and require exact USD row identity for the live particle set."""

    if particle_points is None:
        if expected_ids:
            raise RuntimeError("particle identity contract failed: particle set missing")
        return []
    observed_ids = [int(value) for value in (particle_points.GetIdsAttr().Get() or [])]
    row_counts = {
        "ids": len(observed_ids),
        "points": len(_positions(particle_points.GetPrim())),
        "velocities": len(particle_points.GetVelocitiesAttr().Get() or []),
        "widths": len(particle_points.GetWidthsAttr().Get() or []),
    }
    expected_count = len(expected_ids)
    if any(value != expected_count for value in row_counts.values()):
        raise RuntimeError(
            "particle identity contract failed: row-count mismatch "
            f"expected={expected_count} observed={row_counts}"
        )
    if observed_ids != expected_ids:
        raise RuntimeError("particle identity contract failed: USD ids changed order or value")
    return observed_ids


def _build_composition_stage_evidence(
    *,
    ordinal: int,
    tank_id: str,
    physics_step: int,
    particle_ids: list[int],
    positions_m: Any,
    compositions: Any,
    source_component_totals: Any,
    diffusion_update_count: int,
) -> dict[str, Any]:
    """Summarize one stage without serializing per-particle arrays."""

    import numpy as np

    positions = np.asarray(positions_m, dtype=np.float64)
    rows = np.asarray(compositions, dtype=np.float64)
    source_totals = np.asarray(source_component_totals, dtype=np.float64)
    if (
        positions.shape != (len(particle_ids), 3)
        or rows.shape != (len(particle_ids), 3)
        or source_totals.shape != (3,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rows))
        or not np.all(np.isfinite(source_totals))
        or len(set(particle_ids)) != len(particle_ids)
    ):
        raise RuntimeError("composition stage evidence failed finite/count validation")
    palette_colors, proto_indices, palette_receipt = build_particle_palette(rows)
    component_totals = rows.sum(axis=0)
    row_sum_error = float(np.max(np.abs(rows.sum(axis=1) - 1.0), initial=0.0))
    conservation_error = float(
        np.max(np.abs(component_totals - source_totals), initial=0.0)
    )
    if (
        row_sum_error > 1.0e-12
        or conservation_error > 1.0e-9
        or np.any(rows < 0.0)
        or not np.all(np.isfinite(palette_colors))
        or not np.all(np.isfinite(proto_indices))
    ):
        raise RuntimeError("composition stage evidence failed conservation validation")
    order = np.argsort(positions[:, 2], kind="mergesort")
    vertical_summary: list[dict[str, Any]] = []
    for bucket in np.array_split(order, min(4, len(order))):
        if bucket.size == 0:
            continue
        vertical_summary.append(
            {
                "particle_count": int(bucket.size),
                "z_min_m": float(np.min(positions[bucket, 2])),
                "z_max_m": float(np.max(positions[bucket, 2])),
                "composition_mean": rows[bucket].mean(axis=0).tolist(),
            }
        )
    return {
        "ordinal": int(ordinal),
        "tank_id": str(tank_id),
        "physics_step": int(physics_step),
        "particle_count": int(len(particle_ids)),
        "cumulative_particle_count": int(len(particle_ids)),
        "particle_ids_sha256": _particle_ids_sha256(particle_ids),
        "positions_sha256": _float_array_sha256(positions),
        "compositions_sha256": _float_array_sha256(rows),
        "component_totals": [float(value) for value in component_totals],
        "source_component_totals": [float(value) for value in source_totals],
        "component_fractions": (
            component_totals / float(np.sum(component_totals))
        ).tolist(),
        "row_sum_error": row_sum_error,
        "conservation_error": conservation_error,
        "nonnegative_passed": True,
        "conservation_passed": True,
        "particle_row_order_preserved": True,
        "diffusion_update_count": int(diffusion_update_count),
        "vertical_composition_summary": vertical_summary,
        "palette_resolution": int(palette_receipt["resolution"]),
        "palette_prototype_count": int(palette_receipt["prototype_count"]),
        "palette_colors_sha256": palette_receipt["palette_colors_sha256"],
        "palette_indices_sha256": palette_receipt["proto_indices_sha256"],
    }


def _author_direct_liquid_semantics(prim: Any, UsdSemantics: Any) -> dict[str, Any]:
    """Apply and verify a direct class label on generated renderable geometry."""

    if not prim.IsValid():
        raise RuntimeError("cannot label an invalid target prim")
    labels_api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    if not labels_api or not labels_api.CreateLabelsAttr().Set(["liquid"]):
        raise RuntimeError("could not author liquid semantics on target prim")
    raw_labels = labels_api.GetLabelsAttr().Get()
    labels = [] if raw_labels is None else [str(value) for value in raw_labels]
    schema = "SemanticsLabelsAPI:class"
    if schema not in {str(value) for value in prim.GetAppliedSchemas()} or labels != [
        "liquid"
    ]:
        raise RuntimeError("target prim direct semantic readback failed")
    return {
        "path": str(prim.GetPath()),
        "schema": schema,
        "labels": labels,
        "direct_binding_verified": True,
    }


def _escaped_count(positions: list[list[float]]) -> int:
    return sum(
        1
        for x, y, z in positions
        if not (-0.015 <= x <= 0.015 and -0.0005 <= y <= 0.145 and -0.015 <= z <= 0.015)
    )


def _decode_video(cv2: Any, path: Path) -> tuple[bool, int]:
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        return False, 0
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                return False, count
            count += 1
    finally:
        capture.release()
    return count > 0, count


def _write_rejected_rgb_diagnostic(
    cv2: Any,
    np: Any,
    output: Path,
    rgb: Any,
    *,
    step: int,
    phase: str,
) -> dict[str, Any]:
    """Persist a rejected full-shape RGB without making it a sequence frame."""

    safe_phase = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in phase
    )
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    path = diagnostics / f"rejected-rgb-step-{step:05d}-{safe_phase}.png"
    if not cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write rejected RGB diagnostic {path}")
    channel_mean = np.mean(rgb, axis=(0, 1))
    channel_std = np.std(rgb, axis=(0, 1))
    return {
        "path": str(path.relative_to(output)),
        "sha256": _sha256(path),
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "channel_mean": [float(value) for value in channel_mean],
        "channel_std": [float(value) for value in channel_std],
        "spatial_rgb_std": float(np.std(rgb)),
        "minimum": int(np.min(rgb)),
        "maximum": int(np.max(rgb)),
        "spatially_constant": not bool(np.any(rgb != rgb[0, 0])),
        "included_in_frame_count": False,
        "included_in_video": False,
        "included_as_stage": False,
    }


def _start_or_reuse_kit() -> tuple[Any, Any]:
    try:
        import omni.kit.app

        current = omni.kit.app.get_app()
        if current is not None and bool(_safe_call(current, "is_running", False)):
            return current, None
    except Exception:
        pass
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "extra_args": list(REPLICATOR_EXTRA_ARGS),
        }
    )
    import omni.kit.app

    return omni.kit.app.get_app(), simulation_app


def _configure_replicator_capture(rep: Any) -> None:
    rep.orchestrator.set_capture_on_play(False)


def _enable_syntheticdata_volume_capture(
    settings: Any, receipt: dict[str, Any]
) -> bool:
    """Opt volume prims into synthetic-data rays and preserve the readback.

    omni.syntheticdata 0.6.17 documents this opt-in for OMPE-96991: volume
    prims otherwise remain outside synthetic-data primary rays even when their
    semantic schema is valid. Whether that documented mechanism restores this
    generated PhysX isosurface to the semantic-ID pass is measured by the run.
    """

    initial = bool(settings.get_as_bool(SYNTHETICDATA_CAPTURE_VOLUMES_SETTING))
    settings.set_bool(SYNTHETICDATA_CAPTURE_VOLUMES_SETTING, True)
    enabled = bool(settings.get_as_bool(SYNTHETICDATA_CAPTURE_VOLUMES_SETTING))
    receipt["semantics"].update(
        {
            "capture_volumes_setting_path": SYNTHETICDATA_CAPTURE_VOLUMES_SETTING,
            "capture_volumes_initial_readback": initial,
            "capture_volumes_enabled_readback": enabled,
        }
    )
    if not enabled:
        raise RuntimeError("synthetic-data volume capture setting did not read back true")
    return initial


def _replicator_same_state_step(rep: Any, *, rt_subframes: int) -> None:
    rep.orchestrator.step(
        delta_time=0.0,
        pause_timeline=False,
        rt_subframes=rt_subframes,
        wait_for_render=True,
    )


def _replicator_same_state_step_checked(
    rep: Any, *, rt_subframes: int, callback_count: Any, timeline: Any
) -> dict[str, object]:
    """Capture render output without permitting the playing timeline to step."""

    before = int(callback_count())
    before_time = float(timeline.get_current_time())
    was_playing = bool(timeline.is_playing())
    _replicator_same_state_step(rep, rt_subframes=rt_subframes)
    after = int(callback_count())
    if after != before:
        raise RuntimeError(
            "same-state Replicator capture advanced physics "
            f"from callback {before} to {after}"
        )
    if not math.isclose(
        float(timeline.get_current_time()), before_time, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError("same-state Replicator capture changed timeline time")
    if bool(timeline.is_playing()) != was_playing:
        raise RuntimeError("same-state Replicator capture changed timeline playback")
    return {
        "callbacks_before": before,
        "callbacks_after": after,
        "timeline_time_before_s": before_time,
        "timeline_time_after_s": float(timeline.get_current_time()),
        "timeline_playing_before": was_playing,
        "timeline_playing_after": bool(timeline.is_playing()),
    }


def _app_update_in_session(stage: Any, session_layer: Any, app: Any, Usd: Any) -> None:
    """Run Kit work in the session target so PhysX USD writeback cannot hit root."""

    with Usd.EditContext(stage, session_layer):
        app.update()


def _source_layer_records(stage: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if not layer.realPath:
            continue
        path = Path(layer.realPath).resolve(strict=True)
        records.append(
            {
                "identifier": str(layer.identifier),
                "resolved_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise RuntimeError("opened twin stage has no hashable non-session layers")
    return records


def _twin_chunk_positions(Gf: Any, count: int, rng: random.Random) -> list[Any]:
    values = []
    spacing = 0.003
    for index in range(count):
        ix = index % 3
        iy = (index // 3) % 3
        iz = index // 9
        values.append(
            Gf.Vec3f(
                NOZZLE_EXIT_CENTER_M[0]
                + (ix - 1) * spacing
                + (rng.random() - 0.5) * 0.0003,
                NOZZLE_EXIT_CENTER_M[1]
                + (iy - 1) * spacing
                + (rng.random() - 0.5) * 0.0003,
                min(NOZZLE_EXIT_CENTER_M[2] - 0.003, VIAL_MOUTH_M - 0.001)
                - iz * spacing,
            )
        )
    return values


def _twin_escaped_count(positions: list[list[float]]) -> int:
    return sum(
        1
        for x, y, z in positions
        if (
            math.hypot(x - VIAL_CENTER_XY_M[0], y - VIAL_CENTER_XY_M[1])
            > SOURCE_LIQUID_RADIUS_M
            or z < SOURCE_LIQUID_BOTTOM_M - 0.001
            or z > VIAL_MOUTH_M + 0.001
        )
    )


def _parity_metrics(np: Any, reference: Any, current: Any) -> dict[str, Any]:
    if reference.shape != current.shape:
        raise RuntimeError("pre-pour parity images have different shapes")
    difference = np.abs(reference.astype(np.int16) - current.astype(np.int16))
    mean_abs = float(np.mean(difference))
    p95_abs = float(np.percentile(difference, 95))
    fraction_gt_16 = float(np.mean(difference > 16))
    return {
        "mean_abs_rgb": mean_abs,
        "p95_abs_rgb": p95_abs,
        "fraction_abs_rgb_gt_16": fraction_gt_16,
        "passed": bool(
            mean_abs <= PARITY_THRESHOLDS["mean_abs_rgb_max"]
            and p95_abs <= PARITY_THRESHOLDS["p95_abs_rgb_max"]
            and fraction_gt_16
            <= PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"]
        ),
        "difference": difference,
    }


def run_sequence(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    """Run F2 on the existing fill-station twin; only liquid physics is added."""

    width, height = _validate_args(args)
    receipt = _empty_receipt(args)
    started = time.monotonic()
    deadline = started + WALL_LIMIT_SECONDS
    isosurface_started: float | None = None
    timeline = None
    physics_step_subscription = None
    parity_capture_subscription = None
    settings = None
    capture_volumes_initial: bool | None = None
    velocity_writeback_setting: str | None = None
    velocity_writeback_initial: bool | None = None
    render_product = None
    rgb_annotator = None
    semantic_annotator = None
    writer = None
    frame_paths: list[Path] = []
    frame_steps: list[int] = []
    semantic_paths: list[Path] = []
    liquid_mask_paths: list[Path] = []
    liquid_pixel_counts: list[int] = []
    semantic_mask_records: list[dict[str, Any]] = []
    stage_paths: list[Path] = []
    stage_frame_indices: list[int] = []
    stage_state_paths: list[Path] = []
    observed_stage_states: list[dict[str, Any]] = []
    mesh_proxy_adapter = None
    mesh_proxy_intent = None
    mesh_proxy_created = False
    mesh_proxy_parity_passed = False
    mesh_proxy_refresh_records: list[dict[str, Any]] = []
    callback_records: list[dict[str, Any]] = []
    capture_guard_records: list[dict[str, object]] = []
    timeline_app_update_index = 0
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, Vt

        settings = carb.settings.get_settings()
        capture_volumes_initial = _enable_syntheticdata_volume_capture(
            settings, receipt
        )
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            REPLICATOR_ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is still enabled")
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        manager = app.get_extension_manager()
        receipt["versions"] = _version_info(app)
        if not manager.is_extension_enabled("omni.physx"):
            raise RuntimeError("omni.physx extension is not enabled")
        foundation = get_physx_foundation_interface()
        receipt["gpu_simulation"]["cuda_device_available"] = bool(
            foundation.cuda_device_check()
        )
        if not receipt["gpu_simulation"]["cuda_device_available"]:
            raise RuntimeError("GPU particles require a CUDA device")

        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError(f"could not open twin stage: {args.stage}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened twin stage")
        layer_records = _source_layer_records(stage)
        receipt["scene_contract"]["source_layer_records"] = layer_records
        default_prim = stage.GetDefaultPrim()
        default_path = str(default_prim.GetPath()) if default_prim.IsValid() else None
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        receipt["scene_contract"].update(
            {
                "default_prim": default_path,
                "meters_per_unit": meters_per_unit,
                "up_axis": up_axis,
            }
        )
        if default_path != "/FillStation" or meters_per_unit != 1.0 or up_axis != "Z":
            raise RuntimeError("opened stage is not the reviewed metre-scale Z-up FillStation twin")

        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0:
            raise RuntimeError("twin timeCodesPerSecond is invalid")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(TWIN_REFERENCE_TIME_CODE / time_codes_per_second)
        app.update()

        # A real, stopped subscription makes the parity capture guard
        # meaningful without allowing a pre-run callback into the F2 trace.
        def on_parity_physics_step(callback_dt: float) -> None:
            callback_records.append({"dt_s": float(callback_dt)})
        parity_capture_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_parity_physics_step
        )

        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"reviewed FillCam missing: {CAMERA_PATH}")
        camera = UsdGeom.Camera(camera_prim)
        clip_attr = camera.GetClippingRangeAttr()
        clip_value = clip_attr.Get(Usd.TimeCode(TWIN_REFERENCE_TIME_CODE))
        effective_clip = [float(clip_value[0]), float(clip_value[1])]
        receipt["camera"].update(
            {
                "source_has_authored_clipping_range": bool(
                    clip_attr.HasAuthoredValueOpinion()
                ),
                "clipping_range_effective": effective_clip,
            }
        )
        if not clip_attr.HasAuthoredValueOpinion() or any(
            not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
            for value, expected in zip(effective_clip, CAMERA_CLIPPING_RANGE)
        ):
            raise RuntimeError("FillCam authored clipping range differs from source contract")

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        _configure_replicator_capture(rep)

        parity_dir = output / "pre-pour-parity"
        frames_dir = output / "frames"
        stages_dir = output / "stages"
        semantic_dir = output / "semantic"
        liquid_masks_dir = output / "liquid"
        stage_states_dir = output / "stage-states"
        stream_states_dir = output / "stream-states"
        trace_dir = output / "blend"
        mesh_proxy_dir = output / "mesh-proxy"
        mesh_proxy_refresh_dir = mesh_proxy_dir / "refresh"
        parity_dir.mkdir()
        frames_dir.mkdir()
        stages_dir.mkdir()
        semantic_dir.mkdir()
        liquid_masks_dir.mkdir()
        stage_states_dir.mkdir()
        stream_states_dir.mkdir()
        trace_dir.mkdir()
        mesh_proxy_dir.mkdir()
        mesh_proxy_refresh_dir.mkdir()
        for _ in range(args.warmup):
            app.update()

        def check_time_limits() -> None:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("F2 wall-time safety stop reached")
            if isosurface_started is not None and now - isosurface_started > ISOSURFACE_LIMIT_SECONDS:
                raise RuntimeError("F2 isosurface-enabled time safety stop reached")

        def acquire_rgb(step: int, phase: str, *, included_in_video: bool) -> Any:
            invalid_reasons: list[str] = []
            rgb = None
            last_rejected_rgb = None
            for _attempt in range(1, args.max_capture_tries + 1):
                check_time_limits()
                guard_record = _replicator_same_state_step_checked(
                        rep,
                        rt_subframes=args.rt_subframes,
                        callback_count=lambda: len(callback_records),
                        timeline=timeline,
                    )
                guard_record.update(
                    {"physics_step": step, "phase": phase, "attempt": _attempt}
                )
                capture_guard_records.append(guard_record)
                candidate = np.asarray(rgb_annotator.get_data())
                if candidate.ndim != 3 or candidate.shape[:2] != (height, width) or candidate.shape[2] < 3:
                    invalid_reasons.append(f"shape={candidate.shape}")
                    continue
                candidate = np.asarray(candidate[..., :3], dtype=np.uint8)
                if not np.any(candidate != candidate[0, 0]):
                    invalid_reasons.append("spatially_constant")
                    last_rejected_rgb = candidate.copy()
                    continue
                rgb = candidate
                break
            tries = len(invalid_reasons) + (1 if rgb is not None else 0)
            receipt["render"]["capture_attempt_count"] += tries
            attempt_record = {
                "frame_index": len(frame_paths) + 1 if included_in_video else None,
                "physics_step": step,
                "phase": phase,
                "tries": tries,
                "invalid_reasons": invalid_reasons,
                "included_in_video": included_in_video,
                "rejected_rgb_diagnostic": None,
            }
            receipt["render"]["capture_attempts"].append(attempt_record)
            if rgb is None:
                if last_rejected_rgb is not None:
                    diagnostic = _write_rejected_rgb_diagnostic(
                        cv2, np, output, last_rejected_rgb, step=step, phase=phase
                    )
                    attempt_record["rejected_rgb_diagnostic"] = diagnostic
                    receipt["render"]["rejected_rgb_diagnostics"].append(diagnostic)
                raise RuntimeError(
                    f"RGB capture failed after {args.max_capture_tries} tries: "
                    f"{invalid_reasons[-1] if invalid_reasons else 'unknown'}"
                )
            return rgb

        def acquire_shared_frame(
            step: int, phase: str
        ) -> tuple[Any, Any, Any, int, dict[str, Any]]:
            invalid_reasons: list[str] = []
            last_rgb = None
            last_id_to_labels: object = None
            last_semantic_unique_ids: list[int] = []
            attempts = 0
            for _attempt in range(1, args.max_capture_tries + 1):
                attempts = _attempt
                check_time_limits()
                guard_record = _replicator_same_state_step_checked(
                        rep,
                        rt_subframes=args.rt_subframes,
                        callback_count=lambda: len(callback_records),
                        timeline=timeline,
                    )
                guard_record.update(
                    {"physics_step": step, "phase": phase, "attempt": _attempt}
                )
                capture_guard_records.append(guard_record)
                rgb = np.asarray(rgb_annotator.get_data())
                raw_semantic = semantic_annotator.get_data()
                if rgb.ndim != 3 or rgb.shape[:2] != (height, width) or rgb.shape[2] < 3:
                    invalid_reasons.append(f"rgb_shape={rgb.shape}")
                    continue
                rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
                last_rgb = rgb
                if not np.any(rgb != rgb[0, 0]):
                    invalid_reasons.append("rgb_spatially_constant")
                    continue
                if not isinstance(raw_semantic, dict):
                    invalid_reasons.append("semantic_envelope_not_mapping")
                    continue
                info = raw_semantic.get("info")
                if not isinstance(info, dict) or "idToLabels" not in info:
                    invalid_reasons.append("semantic_mapping_missing")
                    continue
                last_id_to_labels = info["idToLabels"]
                semantic = np.asarray(raw_semantic.get("data"))
                if semantic.ndim == 3 and semantic.shape[2] == 1:
                    semantic = semantic[..., 0]
                if (
                    semantic.shape != (height, width)
                    or np.issubdtype(semantic.dtype, np.bool_)
                    or not np.issubdtype(semantic.dtype, np.integer)
                ):
                    invalid_reasons.append(
                        f"semantic_shape_or_dtype={semantic.shape}/{semantic.dtype}"
                    )
                    continue
                last_semantic_unique_ids = [
                    int(value) for value in np.unique(semantic).tolist()
                ]
                try:
                    exact = exact_semantic_class_mask(
                        semantic,
                        info["idToLabels"],
                        class_name="liquid",
                    )
                except Exception as exc:
                    reason = getattr(exc, "reason_code", type(exc).__name__)
                    invalid_reasons.append(f"semantic_class_mask_invalid={reason}")
                    continue
                mask_record = {
                    "semantic_unique_ids": list(exact.visible_semantic_ids),
                    "liquid_semantic_ids": list(exact.semantic_ids),
                    "semantic_id_to_class_labels": {
                        str(semantic_id): list(labels)
                        for semantic_id, labels in exact.id_to_labels
                    },
                    "liquid_pixel_counts_by_id": {
                        str(semantic_id): int(pixel_count)
                        for semantic_id, pixel_count in exact.pixel_counts_by_id
                    },
                }
                receipt["render"]["capture_attempt_count"] += attempts
                receipt["render"]["capture_attempts"].append(
                    {
                        "frame_index": len(frame_paths) + 1,
                        "physics_step": step,
                        "phase": phase,
                        "tries": attempts,
                        "invalid_reasons": invalid_reasons,
                        "included_in_video": True,
                        "semantic_class_mask_required": True,
                    }
                )
                return rgb, semantic, exact.mask, exact.pixel_count, mask_record
            if last_rgb is not None:
                diagnostic = _write_rejected_rgb_diagnostic(
                    cv2, np, output, last_rgb, step=step, phase=phase
                )
                receipt["render"]["rejected_rgb_diagnostics"].append(diagnostic)
            receipt["render"]["capture_attempt_count"] += attempts
            receipt["render"]["capture_attempts"].append(
                {
                    "frame_index": len(frame_paths) + 1,
                    "physics_step": step,
                    "phase": phase,
                    "tries": attempts,
                    "invalid_reasons": invalid_reasons,
                    "included_in_video": False,
                    "semantic_class_mask_required": True,
                    "failed": True,
                }
            )
            semantic_diagnostic_path = (
                output
                / "diagnostics"
                / f"rejected-semantic-step-{step:05d}-{phase}.json"
            )
            semantic_diagnostic_path.parent.mkdir(exist_ok=True)
            label_evidence: dict[str, object] = {}
            for prim_path in (
                f"{FLUIDMIX_NAMESPACE}/ParticleSystem",
                str(receipt["semantics"]["required_isosurface_path"]),
                *receipt["semantics"]["stream_binding_paths"],
            ):
                prim = stage.GetPrimAtPath(prim_path)
                label_evidence[prim_path] = {
                    "valid": bool(prim.IsValid()),
                    "applied_schemas": (
                        [str(value) for value in prim.GetAppliedSchemas()]
                        if prim.IsValid()
                        else []
                    ),
                    "inherited_class_labels": (
                        sorted(
                            str(value)
                            for value in UsdSemantics.LabelsQuery(
                                "class", Usd.TimeCode.Default()
                            ).ComputeUniqueInheritedLabels(prim)
                        )
                        if prim.IsValid()
                        else []
                    ),
                }
            _write_json(
                semantic_diagnostic_path,
                {
                    "physics_step": step,
                    "phase": phase,
                    "attempts": attempts,
                    "invalid_reasons": invalid_reasons,
                    "id_to_labels_repr": repr(last_id_to_labels),
                    "semantic_unique_ids": last_semantic_unique_ids,
                    "prim_label_evidence": label_evidence,
                },
            )
            receipt["render"]["rejected_semantic_diagnostics"].append(
                {
                    "path": str(semantic_diagnostic_path.relative_to(output)),
                    "sha256": _sha256(semantic_diagnostic_path),
                }
            )
            raise RuntimeError(
                f"shared RGB/semantic capture failed after {args.max_capture_tries} tries: "
                f"{invalid_reasons[-1] if invalid_reasons else 'unknown'}"
            )

        reference_rgb = acquire_rgb(0, "twin-reference-time-0001", included_in_video=False)
        reference_path = parity_dir / "twin-reference-time-0001.png"
        if not cv2.imwrite(os.fspath(reference_path), cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write twin reference image")

        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_setting = getattr(physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None)
        if particles_setting is None:
            raise RuntimeError("PhysX lacks SETTING_UPDATE_PARTICLES_TO_USD")
        settings.set(particles_setting, True)
        velocity_writeback_setting = getattr(
            physx_bindings, "SETTING_UPDATE_VELOCITIES_TO_USD", None
        )
        if velocity_writeback_setting is None:
            raise RuntimeError("PhysX lacks SETTING_UPDATE_VELOCITIES_TO_USD")
        velocity_writeback_initial = bool(
            settings.get_as_bool(velocity_writeback_setting)
        )
        settings.set_bool(velocity_writeback_setting, True)
        velocity_writeback_enabled = bool(
            settings.get_as_bool(velocity_writeback_setting)
        )
        receipt["gpu_simulation"].update(
            {
                "velocity_writeback_initial_readback": velocity_writeback_initial,
                "velocity_writeback_enabled_readback": velocity_writeback_enabled,
            }
        )
        if not velocity_writeback_enabled:
            raise RuntimeError("PhysX velocity USD writeback did not enable")
        session_layer = stage.GetSessionLayer()
        authored_paths: list[str] = []
        with Usd.EditContext(stage, session_layer):
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                prim = stage.GetPrimAtPath(hidden_path)
                if not prim.IsValid():
                    raise RuntimeError(f"source liquid prim missing: {hidden_path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                authored_paths.append(hidden_path)

            UsdGeom.Xform.Define(stage, FLUIDMIX_NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{FLUIDMIX_NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            scene_api.CreateSolverTypeAttr().Set("TGS")
            receipt["gpu_simulation"].update(
                {
                    "scene_enable_gpu_dynamics": bool(scene_api.GetEnableGPUDynamicsAttr().Get()),
                    "broadphase": str(scene_api.GetBroadphaseTypeAttr().Get()),
                }
            )

            system_path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateEnableCCDAttr().Set(True)
            receipt["particle_system"]["ccd_enabled"] = bool(
                system.GetEnableCCDAttr().Get()
            )
            if receipt["particle_system"]["ccd_enabled"] is not True:
                raise RuntimeError("particle-system CCD did not retain the requested value")
            system.CreateParticleContactOffsetAttr().Set(0.003)
            system.CreateFluidRestOffsetAttr().Set(0.0015)
            system.CreateContactOffsetAttr().Set(0.003)
            system.CreateRestOffsetAttr().Set(0.0029)
            system.CreateSolidRestOffsetAttr().Set(0.0029)

            physics_material = UsdShade.Material.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial"
            )
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(physics_material.GetPrim())
            pbd.CreateDensityAttr().Set(1000.0)
            pbd.CreateFrictionAttr().Set(0.1)
            binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
            binding.Bind(
                physics_material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            global_material = UsdShade.Material.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial"
            )
            global_shader = _author_omnisurface(
                stage,
                global_material,
                NOMINAL_TANK_TRANSMISSION_COLORS[COMPOSITION_COMPONENTS[0]],
                Sdf,
                Gf,
                UsdShade,
            )
            receipt["semantics"].update(
                {
                    "material_binding_authored": True,
                    "material_binding": _author_direct_liquid_semantics(
                        global_material.GetPrim(), UsdSemantics
                    ),
                }
            )
            binding.Bind(global_material)
            receipt["particle_system"].update(
                {
                    "pbd_material_binding": bool(binding.GetDirectBinding("physics").GetMaterial()),
                    "render_material_binding": bool(binding.GetDirectBinding().GetMaterial()),
                }
            )
            receipt["material"]["isosurface_global_material_binding"] = bool(
                binding.GetDirectBinding().GetMaterial()
            )
            system_semantics = UsdSemantics.LabelsAPI.Apply(system.GetPrim(), "class")
            if not system_semantics or not system_semantics.CreateLabelsAttr().Set(["liquid"]):
                raise RuntimeError("could not author liquid semantics on the particle system")
            receipt["semantics"]["system_binding_authored"] = True

            stream_records: dict[str, dict[str, Any]] = {}
            stream_paths: list[str] = []
            full_stream_height = NOZZLE_EXIT_CENTER_M[2] - SOURCE_LIQUID_BOTTOM_M
            for tank_id in COMPOSITION_COMPONENTS:
                stream_path = f"{FLUIDMIX_NAMESPACE}/RenderStreams/{tank_id.upper()}"
                stream = UsdGeom.Cylinder.Define(stage, stream_path)
                stream.CreateAxisAttr().Set(UsdGeom.Tokens.z)
                stream.CreateRadiusAttr().Set(0.0015)
                stream.CreateHeightAttr().Set(full_stream_height)
                translate = stream.AddTranslateOp()
                translate.Set(
                    Gf.Vec3d(
                        NOZZLE_EXIT_CENTER_M[0],
                        NOZZLE_EXIT_CENTER_M[1],
                        SOURCE_LIQUID_BOTTOM_M + full_stream_height / 2.0,
                    )
                )
                stream.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                stream_material = UsdShade.Material.Define(
                    stage, f"{FLUIDMIX_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
                )
                _author_omnisurface(
                    stage,
                    stream_material,
                    NOMINAL_TANK_TRANSMISSION_COLORS[tank_id],
                    Sdf,
                    Gf,
                    UsdShade,
                )
                UsdShade.MaterialBindingAPI.Apply(stream.GetPrim()).Bind(stream_material)
                stream_semantics = UsdSemantics.LabelsAPI.Apply(stream.GetPrim(), "class")
                if not stream_semantics or not stream_semantics.CreateLabelsAttr().Set(["liquid"]):
                    raise RuntimeError(f"could not author liquid semantics on {stream_path}")
                stream_records[tank_id] = {
                    "prim": stream.GetPrim(),
                    "height": stream.GetHeightAttr(),
                    "translate": translate,
                }
                stream_paths.append(stream_path)
                authored_paths.append(stream_path)
            receipt["semantics"].update(
                {
                    "stream_binding_paths": stream_paths,
                    "stream_semantic_bound": True,
                }
            )

            collider_prims = []
            for spec in build_twin_collider_specs():
                cube = UsdGeom.Cube.Define(stage, Sdf.Path(str(spec["path"])))
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())
                authored_paths.append(str(spec["path"]))
            receipt["particle_system"].update(
                {
                    "collider_types": ["Cube"] * len(collider_prims),
                    "collider_paths": [str(prim.GetPath()) for prim in collider_prims],
                    "all_colliders_invisible": all(
                        UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
                        for prim in collider_prims
                    ),
                    "decorative_vial_collision_api": any(
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        for prim in Usd.PrimRange(stage.GetPrimAtPath(SOURCE_VIAL_PATH))
                    ),
                }
            )

            iso_api = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system.GetPrim())
            iso_api.CreateIsosurfaceEnabledAttr().Set(True)
            iso_api.CreateGridSpacingAttr().Set(0.002)
            iso_api.CreateMaxVerticesAttr().Set(250000)
            iso_api.CreateMaxTrianglesAttr().Set(500000)
            receipt["isosurface"]["enabled"] = bool(iso_api.GetIsosurfaceEnabledAttr().Get())

            # The Points set is deliberately not authored here.  A18 established
            # that PhysX must first see a non-empty set on the scheduled main
            # thread boundary immediately before its owning timeline callback.

        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        semantic_annotator.attach([render_product])
        receipt["semantics"]["annotator_attached_after_stream_labels"] = True

        mesh_proxy_intent = build_liquid_proxy_intent(
            source_path=GENERATED_ISOSURFACE_PATH,
            proxy_path=LIQUID_PROXY_PATH,
            material_path=f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
        )
        mesh_proxy_adapter = _UsdMeshProxyAdapter(
            stage=stage,
            session_layer=session_layer,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            UsdSemantics=UsdSemantics,
            Gf=Gf,
            Vt=Vt,
        )

        receipt["emitter"]["source_nozzle_valid"] = stage.GetPrimAtPath(SOURCE_NOZZLE_PATH).IsValid()
        visibility_record: dict[str, dict[str, str]] = {}
        for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(hidden_path))
            visibility_record[hidden_path] = {
                str(code): str(imageable.GetVisibilityAttr().Get(Usd.TimeCode(code)))
                for code in VISIBILITY_CHECK_TIME_CODES
            }
        receipt["scene_contract"]["hidden_source_visibility"] = visibility_record
        if any(value != "invisible" for row in visibility_record.values() for value in row.values()):
            raise RuntimeError("source liquid visibility override did not compose at all critical times")
        camera_property = Sdf.Path(CAMERA_PATH).AppendProperty("clippingRange")
        receipt["camera"]["clipping_range_authored_by_runner"] = bool(
            session_layer.GetPropertyAtPath(camera_property)
        )
        if receipt["camera"]["clipping_range_authored_by_runner"]:
            raise RuntimeError("runner unexpectedly authored FillCam clippingRange")
        forbidden = []
        if session_layer.GetPrimAtPath("/World"):
            forbidden.append("/World")
        namespace_prims = [
            prim for prim in stage.Traverse() if str(prim.GetPath()).startswith(FLUIDMIX_NAMESPACE)
        ]
        receipt["scene_contract"].update(
            {
                "forbidden_authored_paths": forbidden,
                "custom_camera_count": sum(prim.IsA(UsdGeom.Camera) for prim in namespace_prims),
                "custom_light_count": sum(prim.GetTypeName().endswith("Light") for prim in namespace_prims),
                "custom_backdrop_count": sum("backdrop" in str(prim.GetPath()).lower() for prim in namespace_prims),
                "authored_paths": authored_paths,
            }
        )
        if forbidden or any(receipt["scene_contract"][key] for key in ("custom_camera_count", "custom_light_count", "custom_backdrop_count")):
            raise RuntimeError("F2 authored a forbidden replacement-scene prim")

        isosurface_started = time.monotonic()
        for _ in range(2):
            _app_update_in_session(stage, session_layer, app, Usd)
        _memory_sample(receipt)

        writer, video_path, video_codec, video_fallback = _open_video_writer(
            cv2,
            output,
            width=width,
            height=height,
            fps=float(args.video_fps),
        )

        def set_stream_state(active_tank: str | None, fill_fraction: float) -> tuple[str, ...]:
            surface_z = SOURCE_LIQUID_BOTTOM_M + min(max(fill_fraction, 0.0), 1.0) * (
                VIAL_MOUTH_M - SOURCE_LIQUID_BOTTOM_M
            )
            height = max(NOZZLE_EXIT_CENTER_M[2] - surface_z, 0.0005)
            center_z = surface_z + height / 2.0
            def author_and_read() -> tuple[str, ...]:
                for tank_id, record in stream_records.items():
                    record["height"].Set(height)
                    record["translate"].Set(
                        Gf.Vec3d(
                            NOZZLE_EXIT_CENTER_M[0],
                            NOZZLE_EXIT_CENTER_M[1],
                            center_z,
                        )
                    )
                    UsdGeom.Imageable(record["prim"]).GetVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited
                        if tank_id == active_tank
                        else UsdGeom.Tokens.invisible
                    )
                return tuple(
                    tank_id
                    for tank_id, record in stream_records.items()
                    if UsdGeom.Imageable(record["prim"]).ComputeVisibility()
                    != UsdGeom.Tokens.invisible
                )

            visible = _with_session_edit_context(
                Usd,
                stage,
                session_layer,
                author_and_read,
            )
            expected = () if active_tank is None else (active_tank,)
            if visible != expected:
                raise RuntimeError(
                    f"render-only stream visibility mismatch: expected {expected}, got {visible}"
                )
            return visible

        def set_proxy_visibility(*, source_visible: bool, proxy_visible: bool) -> tuple[str, str]:
            if not mesh_proxy_created:
                raise RuntimeError("mesh proxy visibility requested before creation")

            def author_and_read() -> tuple[str, str]:
                source = UsdGeom.Imageable(stage.GetPrimAtPath(GENERATED_ISOSURFACE_PATH))
                proxy = UsdGeom.Imageable(stage.GetPrimAtPath(LIQUID_PROXY_PATH))
                source.GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited if source_visible else UsdGeom.Tokens.invisible
                )
                proxy.GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited if proxy_visible else UsdGeom.Tokens.invisible
                )
                return (
                    str(source.ComputeVisibility()),
                    str(proxy.ComputeVisibility()),
                )

            observed = _with_session_edit_context(
                Usd, stage, session_layer, author_and_read
            )
            expected = (
                "inherited" if source_visible else "invisible",
                "inherited" if proxy_visible else "invisible",
            )
            if observed != expected:
                raise RuntimeError(
                    f"mesh proxy visibility mismatch: expected {expected}, got {observed}"
                )
            return observed

        def set_stream_only_liquid_visibility() -> tuple[str, str]:
            """Hide any non-substantive particle surface before stream-only capture."""

            def author_and_read() -> tuple[str, str]:
                values: list[str] = []
                for path in (GENERATED_ISOSURFACE_PATH, LIQUID_PROXY_PATH):
                    prim = stage.GetPrimAtPath(path)
                    if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
                        values.append("absent")
                        continue
                    imageable = UsdGeom.Imageable(prim)
                    imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                    values.append(str(imageable.ComputeVisibility()))
                return values[0], values[1]

            observed = _with_session_edit_context(
                Usd, stage, session_layer, author_and_read
            )
            if any(value not in {"absent", "invisible"} for value in observed):
                raise RuntimeError(
                    f"stream-only capture left particle geometry visible: {observed}"
                )
            return observed

        def ensure_mesh_proxy(step: int) -> None:
            nonlocal mesh_proxy_created
            if mesh_proxy_created:
                return
            source_prim = stage.GetPrimAtPath(GENERATED_ISOSURFACE_PATH)
            if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Mesh):
                return
            source_evidence = _substantive_isosurface_mesh_evidence(
                stage, Sdf.Path(GENERATED_ISOSURFACE_PATH), UsdGeom
            )
            if source_evidence["mesh_geometry_valid"] is not True:
                return
            if mesh_proxy_adapter is None or mesh_proxy_intent is None:
                raise RuntimeError("mesh proxy adapter was not initialized")
            creation = copy_exact_mesh_proxy(
                mesh_proxy_adapter, mesh_proxy_intent
            )
            mesh_proxy_created = True
            authored_paths.append(LIQUID_PROXY_PATH)
            set_proxy_visibility(source_visible=True, proxy_visible=False)
            direct_binding = _with_session_edit_context(
                Usd,
                stage,
                session_layer,
                lambda: _author_direct_liquid_semantics(
                    stage.GetPrimAtPath(LIQUID_PROXY_PATH), UsdSemantics
                ),
            )
            creation_path = mesh_proxy_dir / f"creation-step-{step:05d}.json"
            _write_json(
                creation_path,
                {
                    "physics_step": step,
                    "evidence": creation,
                    "source_visibility": "inherited",
                    "proxy_visibility": "invisible",
                },
            )
            receipt["mesh_proxy"].update(
                {
                    "created": True,
                    "creation_step": step,
                    "creation_evidence": creation,
                    "creation_artifact_path": str(creation_path.relative_to(output)),
                    "creation_artifact_sha256": _sha256(creation_path),
                }
            )
            receipt["semantics"].update(
                {
                    "isosurface_direct_binding_authored": True,
                    "isosurface_direct_binding_step": step,
                    "isosurface_direct_binding": direct_binding,
                    "isosurface_semantic_bound": True,
                }
            )

        def prepare_mesh_proxy_capture(step: int, phase: str) -> dict[str, Any]:
            nonlocal mesh_proxy_parity_passed
            if not mesh_proxy_created or mesh_proxy_adapter is None or mesh_proxy_intent is None:
                raise RuntimeError("nonempty exact mesh proxy is required before capture")
            evidence = refresh_exact_mesh_proxy(
                mesh_proxy_adapter, mesh_proxy_intent
            )
            if not mesh_proxy_parity_passed:
                parity_state = set_proxy_visibility(
                    source_visible=True, proxy_visible=False
                )
                original_rgb = acquire_rgb(
                    step,
                    f"mesh-proxy-parity-original-{phase}",
                    included_in_video=False,
                )
                original_path = mesh_proxy_dir / "parity-original-source.png"
                if not cv2.imwrite(
                    os.fspath(original_path),
                    cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError("could not write mesh proxy source parity image")
                proxy_capture_state = set_proxy_visibility(
                    source_visible=False, proxy_visible=True
                )
                proxy_rgb = acquire_rgb(
                    step,
                    f"mesh-proxy-parity-proxy-{phase}",
                    included_in_video=False,
                )
                proxy_path = mesh_proxy_dir / "parity-standard-proxy.png"
                if not cv2.imwrite(
                    os.fspath(proxy_path),
                    cv2.cvtColor(proxy_rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError("could not write mesh proxy parity image")
                restored = set_proxy_visibility(
                    source_visible=True, proxy_visible=False
                )
                measured = _parity_metrics(np, original_rgb, proxy_rgb)
                difference = np.asarray(
                    measured.pop("difference"), dtype=np.uint8
                )
                difference_path = mesh_proxy_dir / "parity-absdiff.png"
                if not cv2.imwrite(
                    os.fspath(difference_path),
                    cv2.cvtColor(difference, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError("could not write mesh proxy parity difference")
                parity_input = {
                    "same_physics_step": True,
                    "same_renderer_settings": True,
                    "parity_capture_isolated": True,
                    "source_restored_visible": restored
                    == ("inherited", "invisible"),
                    "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
                    "original_rgb_sha256": _sha256(original_path),
                    "proxy_rgb_sha256": _sha256(proxy_path),
                    "absdiff_sha256": _sha256(difference_path),
                    "mean_abs_rgb": measured["mean_abs_rgb"],
                    "p95_abs_rgb": measured["p95_abs_rgb"],
                    "fraction_abs_rgb_gt_16": measured[
                        "fraction_abs_rgb_gt_16"
                    ],
                }
                decision = evaluate_proxy_rgb_parity_for_source_hide(
                    evidence, parity_input
                )
                receipt["mesh_proxy"]["rgb_parity"] = {
                    **decision,
                    "physics_step": step,
                    "phase": phase,
                    "parity_geometry_mesh_sha256": evidence["mesh_sha256"],
                    "parity_geometry_points_sha256": evidence["points_sha256"],
                    "original_capture_visibility": list(parity_state),
                    "proxy_capture_visibility": list(proxy_capture_state),
                    "restored_visibility": list(restored),
                    **{
                        key: parity_input[key]
                        for key in (
                            "same_physics_step",
                            "same_renderer_settings",
                            "parity_capture_isolated",
                            "source_restored_visible",
                            "original_rgb_sha256",
                            "proxy_rgb_sha256",
                            "absdiff_sha256",
                        )
                    },
                    "original_rgb_path": str(original_path.relative_to(output)),
                    "proxy_rgb_path": str(proxy_path.relative_to(output)),
                    "absdiff_path": str(difference_path.relative_to(output)),
                }
                if decision["hide_source_allowed"] is not True:
                    raise RuntimeError("exact mesh proxy RGB parity gate failed")
                mesh_proxy_parity_passed = True
            observed_visibility = set_proxy_visibility(
                source_visible=False, proxy_visible=True
            )
            return {
                "physics_step": step,
                "phase": phase,
                "evidence": evidence,
                "source_visibility": observed_visibility[0],
                "proxy_visibility": observed_visibility[1],
                "shared_rgb_semantic_capture": False,
            }

        def capture(
            step: int,
            phase: str,
            visible_stream_tanks: tuple[str, ...],
        ) -> tuple[Path, list[list[float]], Any]:
            if mesh_proxy_created:
                liquid_representation = "vial_proxy"
                proxy_refresh = prepare_mesh_proxy_capture(step, phase)
                visibility = (
                    proxy_refresh["source_visibility"],
                    proxy_refresh["proxy_visibility"],
                )
            else:
                liquid_representation = "stream_only"
                proxy_refresh = None
                if (
                    len(visible_stream_tanks) != 1
                    or not phase.endswith("-pour")
                ):
                    raise RuntimeError(
                        "stream-only representation is allowed only during one active pour"
                    )
                visibility = set_stream_only_liquid_visibility()
            rgb, semantic, liquid_mask, liquid_pixels, mask_record = acquire_shared_frame(
                step, phase
            )
            proxy_refresh_record = None
            if proxy_refresh is not None:
                proxy_refresh["shared_rgb_semantic_capture"] = True
                proxy_refresh_path = mesh_proxy_refresh_dir / (
                    f"frame-{len(frame_paths) + 1:04d}-step-{step:05d}.json"
                )
                _write_json(proxy_refresh_path, proxy_refresh)
                proxy_refresh_record = {
                    **proxy_refresh,
                    "artifact_path": str(proxy_refresh_path.relative_to(output)),
                    "artifact_sha256": _sha256(proxy_refresh_path),
                }
                mesh_proxy_refresh_records.append(proxy_refresh_record)
            index = len(frame_paths) + 1
            path = frames_dir / f"frame-{index:04d}-step-{step:05d}-{phase}.png"
            semantic_path = semantic_dir / f"frame-{index:04d}-step-{step:05d}.npy"
            liquid_path = liquid_masks_dir / f"frame-{index:04d}-step-{step:05d}.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(os.fspath(path), bgr):
                raise RuntimeError(f"could not write RGB frame {path}")
            np.save(semantic_path, semantic, allow_pickle=False)
            if not cv2.imwrite(os.fspath(liquid_path), liquid_mask):
                raise RuntimeError(f"could not write liquid semantic mask {liquid_path}")
            reloaded_semantic = np.load(semantic_path, allow_pickle=False)
            reloaded_mask = cv2.imread(os.fspath(liquid_path), cv2.IMREAD_UNCHANGED)
            saved_mapping = {
                semantic_id: {"class": ", ".join(labels)}
                for semantic_id, labels in mask_record[
                    "semantic_id_to_class_labels"
                ].items()
            }
            saved_exact = exact_semantic_class_mask(
                reloaded_semantic,
                saved_mapping,
                class_name="liquid",
            )
            saved_record = {
                "semantic_unique_ids": list(saved_exact.visible_semantic_ids),
                "liquid_semantic_ids": list(saved_exact.semantic_ids),
                "semantic_id_to_class_labels": {
                    str(semantic_id): list(labels)
                    for semantic_id, labels in saved_exact.id_to_labels
                },
                "liquid_pixel_counts_by_id": {
                    str(semantic_id): int(pixel_count)
                    for semantic_id, pixel_count in saved_exact.pixel_counts_by_id
                },
            }
            if (
                reloaded_semantic.shape != (height, width)
                or reloaded_mask is None
                or reloaded_mask.shape != (height, width)
                or not np.array_equal(reloaded_semantic, semantic)
                or not np.array_equal(reloaded_mask, liquid_mask)
                or not np.array_equal(reloaded_mask, saved_exact.mask)
                or saved_exact.pixel_count != liquid_pixels
                or saved_record != mask_record
            ):
                raise RuntimeError("saved semantic evidence is unreadable or misaligned")
            writer.write(bgr)
            frame_paths.append(path)
            frame_steps.append(step)
            semantic_paths.append(semantic_path)
            liquid_mask_paths.append(liquid_path)
            liquid_pixel_counts.append(liquid_pixels)
            semantic_mask_records.append(
                {
                    "phase": phase,
                    "visible_stream_tanks": list(visible_stream_tanks),
                    "liquid_representation": liquid_representation,
                    "mesh_proxy_refresh_artifact_path": (
                        proxy_refresh_record["artifact_path"]
                        if proxy_refresh_record is not None
                        else None
                    ),
                    "mesh_proxy_refresh_artifact_sha256": (
                        proxy_refresh_record["artifact_sha256"]
                        if proxy_refresh_record is not None
                        else None
                    ),
                    "source_isosurface_visibility": visibility[0],
                    "proxy_visibility": visibility[1],
                    "shared_rgb_semantic_capture": True,
                    **mask_record,
                }
            )
            return (
                path,
                _positions(shared_particle_points.GetPrim())
                if shared_particle_points is not None
                else [],
                rgb,
            )

        prefill_rgb = acquire_rgb(0, "f2-prefill-time-0001", included_in_video=False)
        prefill_path = parity_dir / "f2-prefill-source.png"
        if not cv2.imwrite(
            os.fspath(prefill_path), cv2.cvtColor(prefill_rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError("could not write F2 prefill parity image")
        parity_prefill_path = parity_dir / "f2-prefill-time-0001.png"
        shutil.copyfile(prefill_path, parity_prefill_path)
        parity = _parity_metrics(np, reference_rgb, prefill_rgb)
        difference_image = np.clip(parity.pop("difference") * 8, 0, 255).astype(np.uint8)
        difference_path = parity_dir / "absdiff-times-8.png"
        if not cv2.imwrite(os.fspath(difference_path), cv2.cvtColor(difference_image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write parity absolute-difference image")
        receipt["pre_pour_parity"].update(
            {
                "reference_path": str(reference_path.relative_to(output)),
                "reference_sha256": _sha256(reference_path),
                "f2_prefill_path": str(parity_prefill_path.relative_to(output)),
                "f2_prefill_sha256": _sha256(parity_prefill_path),
                "absdiff_path": str(difference_path.relative_to(output)),
                "absdiff_sha256": _sha256(difference_path),
                **parity,
            }
        )
        if not parity["passed"]:
            raise RuntimeError("F2 pre-pour frame does not match the source twin within frozen noise limits")

        if callback_records:
            raise RuntimeError("stopped pre-run capture unexpectedly advanced physics")
        parity_capture_subscription = None
        plan = receipt["plan"]
        event_index = _build_emission_event_index(plan)
        capture_steps = set(_build_capture_steps(plan, stride=args.capture_stride))
        stage_by_step = {
            int(pour["stage_capture_step"]): (index, pour)
            for index, pour in enumerate(plan["pours"], start=1)
        }
        actual_events: dict[int, list[dict[str, int]]] = {
            int(pour["ordinal"]): [] for pour in plan["pours"]
        }
        emission_trace: list[dict[str, Any]] = []
        stream_active_steps: dict[str, list[int]] = {
            tank_id: [] for tank_id in COMPOSITION_COMPONENTS
        }
        stage_memory_before = {
            int(pour["ordinal"]): None for pour in plan["pours"]
        }
        component_counts = np.zeros(len(COMPOSITION_COMPONENTS), dtype=np.float64)
        compositions_by_id: dict[int, Any] = {}
        active_row_metadata: list[dict[str, Any]] = []
        shared_particle_points = None
        shared_initial_positions: list[list[float]] = []
        shared_particle_ids: list[int] = []
        next_particle_id = 0
        rng = random.Random(args.seed)
        diffusion_update_count = 0
        diffusion_neighbor_count = 0
        diffusion_max_degree = 0
        diffusion_pair_weight_max = 0.0
        global_blend_update_count = 0
        last_global_color_readback: tuple[float, float, float] | None = None
        cumulative = 0
        particle_identity_check_count = 0
        particle_identity_mismatch_count = 0

        scene_api = PhysxSchema.PhysxSceneAPI(scene.GetPrim())
        with Usd.EditContext(stage, session_layer):
            scene_api.GetTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
        if scene_api.GetTimeStepsPerSecondAttr().Get() != PHYSICS_HZ:
            raise RuntimeError("PhysicsScene did not retain the 60 Hz timeline rate")
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(physx_bindings.SETTING_MIN_FRAME_RATE, PHYSICS_HZ)
        timeline = omni.timeline.get_timeline_interface()
        with Usd.EditContext(stage, session_layer):
            timeline.stop()
            timeline.set_current_time(0.0)
            timeline.set_time_codes_per_second(float(PHYSICS_HZ))
            timeline.set_ticks_per_frame(1)
            timeline.set_target_framerate(float(PHYSICS_HZ))
            timeline.set_play_every_frame(True)
            timeline.commit()
        _app_update_in_session(stage, session_layer, app, Usd)

        with Usd.EditContext(stage, session_layer):
            timeline.play()
            timeline.commit()

        def on_physics_step(callback_dt: float) -> None:
            callback_records.append(
                {
                    "callback_index": len(callback_records) + 1,
                    "dt_s": float(callback_dt),
                    "timeline_time_s": float(timeline.get_current_time()),
                    "app_update_index": timeline_app_update_index,
                }
            )

        physics_step_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_physics_step
        )

        while len(callback_records) < int(plan["timing"]["primary_final_capture_step"]):
            global_step = len(callback_records) + 1
            check_time_limits()
            active_tank = _active_tank_for_step(plan, global_step)
            if active_tank is not None:
                active_pour = next(
                    pour for pour in plan["pours"] if pour["tank_id"] == active_tank
                )
                ordinal = int(active_pour["ordinal"])
                if stage_memory_before[ordinal] is None:
                    stage_memory_before[ordinal] = _memory_sample(receipt)
            for event in event_index.get(global_step, []):
                count = int(event["particle_count"])
                particle_ids = list(
                    range(next_particle_id, next_particle_id + count)
                )
                next_component_counts = _update_global_component_counts(
                    component_counts, str(event["tank_id"]), count
                )
                color = _global_transmission_color(next_component_counts)

                def append_and_update_color() -> tuple[float, float, float]:
                    nonlocal shared_particle_points, next_particle_id
                    positions = _twin_chunk_positions(Gf, count, rng)
                    if shared_particle_points is None:
                        shared_particle_points = UsdGeom.Points.Define(
                            stage, f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared"
                        )
                        set_api = PhysxSchema.PhysxParticleSetAPI.Apply(
                            shared_particle_points.GetPrim()
                        )
                        particle_api = PhysxSchema.PhysxParticleAPI(set_api)
                        particle_api.CreateParticleSystemRel().SetTargets([system_path])
                        particle_api.CreateSelfCollisionAttr().Set(True)
                        particle_api.CreateParticleEnabledAttr().Set(True)
                        particle_api.CreateParticleGroupAttr().Set(0)
                        set_api.CreateFluidAttr().Set(True)
                        shared_particle_points.GetPrim().CreateAttribute(
                            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
                        ).Set(int(plan["totals"]["particle_count"]))
                        existing_positions: list[Any] = []
                        existing_velocities: list[Any] = []
                        existing_widths: list[float] = []
                        existing_ids: list[int] = []
                    else:
                        existing_positions = _positions(shared_particle_points.GetPrim())
                        existing_velocities = list(
                            shared_particle_points.GetVelocitiesAttr().Get() or []
                        )
                        existing_widths = list(
                            shared_particle_points.GetWidthsAttr().Get() or []
                        )
                        existing_ids = [
                            int(value)
                            for value in (
                                shared_particle_points.GetIdsAttr().Get() or []
                            )
                        ]
                    if (
                        len(existing_positions) != len(existing_velocities)
                        or len(existing_positions) != len(existing_widths)
                        or len(existing_positions) != len(existing_ids)
                        or existing_ids != shared_particle_ids
                    ):
                        raise RuntimeError("shared particle row attributes lost identity")
                    combined_positions = [*existing_positions, *positions]
                    shared_particle_points.GetPointsAttr().Set(
                        Vt.Vec3fArray(combined_positions)
                    )
                    shared_particle_points.GetVelocitiesAttr().Set(
                        Vt.Vec3fArray(
                            [
                                *existing_velocities,
                                *[Gf.Vec3f(0.0, 0.0, -0.10)] * count,
                            ]
                        )
                    )
                    shared_particle_points.GetWidthsAttr().Set(
                        Vt.FloatArray([*existing_widths, *[0.003] * count])
                    )
                    shared_particle_points.CreateIdsAttr().Set(
                        Vt.Int64Array([*existing_ids, *particle_ids])
                    )
                    return _set_global_blend_color(global_shader, Gf, color)

                last_global_color_readback = _with_session_edit_context(
                    Usd,
                    stage,
                    session_layer,
                    append_and_update_color,
                )
                source_compositions = np.zeros(
                    (count, len(COMPOSITION_COMPONENTS)), dtype=np.float64
                )
                source_compositions[:, COMPOSITION_COMPONENTS.index(event["tank_id"])] = 1.0
                initial_positions = [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in _positions(shared_particle_points.GetPrim())[-count:]
                ]
                active_row_metadata.append(
                    {
                        "particle_ids": particle_ids,
                        "particle_count": count,
                        "source_compositions": source_compositions,
                        "tank_id": event["tank_id"],
                        "pour_ordinal": event["pour_ordinal"],
                        "event": dict(event),
                        "initial_positions": initial_positions,
                    }
                )
                shared_particle_ids.extend(particle_ids)
                shared_initial_positions.extend(initial_positions)
                next_particle_id += count
                for particle_id, composition in zip(particle_ids, source_compositions):
                    compositions_by_id[int(particle_id)] = np.asarray(
                        composition, dtype=np.float64
                    ).copy()
                component_counts = next_component_counts
                cumulative += count
                receipt["totals"]["emitted_particle_count"] = cumulative
                global_blend_update_count += 1
                emission_trace.append(
                    {
                        "update_ordinal": global_blend_update_count,
                        "pour_ordinal": int(event["pour_ordinal"]),
                        "tank_id": str(event["tank_id"]),
                        "emission_ordinal": int(event["ordinal"]),
                        "physics_step": int(event["physics_step"]),
                        "authored_after_callback_count": global_step - 1,
                        "scheduled_before_callback_index": global_step,
                        "authored_on_main_thread": True,
                        "particle_count": count,
                        "cumulative_particle_counts": [
                            int(value) for value in component_counts
                        ],
                        "cumulative_particle_count": int(np.sum(component_counts)),
                        "component_fractions": (
                            component_counts / float(np.sum(component_counts))
                        ).tolist(),
                        "transmission_color": color.tolist(),
                        "usd_readback_transmission_color": list(
                            last_global_color_readback
                        ),
                    }
                )
                actual_events[int(event["pour_ordinal"])].append(
                    {
                        key: int(event[key])
                        for key in ("ordinal", "physics_step", "particle_count")
                    }
                )
            visible_streams = set_stream_state(
                active_tank, cumulative / float(plan["totals"]["particle_count"])
            )
            for tank_id in visible_streams:
                stream_active_steps[tank_id].append(global_step)
            before_callback_count = len(callback_records)
            timeline_app_update_index += 1
            _app_update_in_session(stage, session_layer, app, Usd)
            if len(callback_records) != before_callback_count + 1:
                raise RuntimeError("timeline produced not exactly one callback per app update")
            callback = callback_records[-1]
            if (
                not math.isclose(
                    float(callback["dt_s"]),
                    1.0 / PHYSICS_HZ,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
                or not math.isclose(
                    float(callback["timeline_time_s"]),
                    global_step / float(PHYSICS_HZ),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
                or callback["app_update_index"] != timeline_app_update_index
                or callback["callback_index"] != global_step
            ):
                raise RuntimeError("timeline callback trace differs from the A18 60 Hz contract")
            try:
                observed_particle_ids = _read_runtime_particle_identity(
                    shared_particle_points, shared_particle_ids
                )
            except RuntimeError:
                particle_identity_mismatch_count += 1
                receipt["particle_system"].update(
                    {
                        "runtime_particle_identity_check_count": particle_identity_check_count,
                        "runtime_particle_id_mismatch_count": particle_identity_mismatch_count,
                    }
                )
                raise
            particle_identity_check_count += 1
            receipt["particle_system"].update(
                {
                    "runtime_particle_ids_authored": shared_particle_points is not None,
                    "runtime_particle_ids_exact_each_callback": True,
                    "runtime_particle_identity_check_count": particle_identity_check_count,
                    "runtime_particle_id_mismatch_count": particle_identity_mismatch_count,
                    "runtime_particle_ids_final_sha256": _particle_ids_sha256(
                        observed_particle_ids
                    ),
                }
            )
            ensure_mesh_proxy(global_step)
            positions_after_physics = np.asarray(
                _positions(shared_particle_points.GetPrim())
                if shared_particle_points is not None
                else [],
                dtype=np.float64,
            )
            _updated_compositions, diffusion_diagnostics = _diffuse_active_particle_rows(
                active_row_metadata,
                compositions_by_id,
                positions_after_physics,
                observed_particle_ids,
            )
            if cumulative > 0:
                mean_composition = _updated_compositions.mean(axis=0)
                mean_color = mix_linear_rgb(
                    mean_composition[None, :],
                    np.asarray(
                        [
                            NOMINAL_TANK_TRANSMISSION_COLORS[tank]
                            for tank in COMPOSITION_COMPONENTS
                        ],
                        dtype=np.float64,
                    ),
                )[0]
                if not np.allclose(
                    mean_color,
                    _global_transmission_color(component_counts),
                    rtol=0.0,
                    atol=1.0e-10,
                ):
                    raise RuntimeError(
                        "mean composition drifted from the conserved global render colour"
                    )
            diffusion_update_count += 1
            diffusion_neighbor_count += int(diffusion_diagnostics["neighbor_count"])
            diffusion_max_degree = max(
                diffusion_max_degree,
                int(diffusion_diagnostics.get("max_degree", 0)),
            )
            diffusion_pair_weight_max = max(
                diffusion_pair_weight_max,
                float(diffusion_diagnostics.get("pair_weight", 0.0)),
            )
            receipt["timing"]["executed_physics_steps"] = global_step

            if global_step not in capture_steps:
                continue
            receipt["totals"]["final_particle_count"] = len(
                positions_after_physics
            )
            stage_record = stage_by_step.get(global_step)
            if stage_record is not None:
                visible_streams = set_stream_state(
                    None,
                    cumulative / float(plan["totals"]["particle_count"]),
                )
                phase = f"{stage_record[1]['tank_id']}-stage"
            elif active_tank is not None:
                phase = f"{active_tank}-pour"
            else:
                phase = "post-fill"
            frame_path, positions, _ = capture(
                global_step,
                phase,
                visible_streams,
            )

            if stage_record is None:
                continue
            stage_index, pour = stage_record
            destination = stages_dir / f"stage-{stage_index}-{pour['tank_id']}.png"
            shutil.copyfile(frame_path, destination)
            if _sha256(destination) != _sha256(frame_path):
                raise RuntimeError("stage-frame copy differs from source video frame")
            stage_paths.append(destination)
            stage_frame_indices.append(len(frame_paths) - 1)
            stage_particle_ids = _read_runtime_particle_identity(
                shared_particle_points, shared_particle_ids
            )
            stage_rows = np.asarray(
                [compositions_by_id[particle_id] for particle_id in stage_particle_ids],
                dtype=np.float64,
            )
            stage_evidence = _build_composition_stage_evidence(
                ordinal=stage_index,
                tank_id=str(pour["tank_id"]),
                physics_step=global_step,
                particle_ids=stage_particle_ids,
                positions_m=positions,
                compositions=stage_rows,
                source_component_totals=component_counts,
                diffusion_update_count=diffusion_update_count,
            )
            stage_evidence.update(
                {
                    "diffusion_neighbor_count_total": diffusion_neighbor_count,
                    "diffusion_max_degree": diffusion_max_degree,
                    "diffusion_pair_weight_max": diffusion_pair_weight_max,
                }
            )
            composition_evidence_dir = output / "composition-evidence"
            composition_evidence_dir.mkdir(exist_ok=True)
            composition_evidence_path = composition_evidence_dir / (
                f"stage-{stage_index}-step-{global_step:05d}.json"
            )
            _write_json(composition_evidence_path, stage_evidence)
            receipt["composition_evidence"]["stages"].append(
                {
                    **stage_evidence,
                    "artifact_path": str(composition_evidence_path.relative_to(output)),
                    "artifact_sha256": _sha256(composition_evidence_path),
                }
            )
            fractions = component_counts / float(np.sum(component_counts))
            expected_state = dict(
                plan["fallback_render_model"]["stage_states"][stage_index - 1]
            )
            observed_counts = [int(value) for value in component_counts]
            observed_color = _global_transmission_color(component_counts).tolist()
            if last_global_color_readback is None or not np.allclose(
                last_global_color_readback,
                observed_color,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("stage global blend lacks matching live USD readback")
            if (
                observed_counts != expected_state["cumulative_particle_counts"]
                or cumulative != expected_state["cumulative_particle_count"]
                or not np.allclose(
                    fractions,
                    expected_state["component_fractions"],
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    observed_color,
                    expected_state["transmission_color"],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise RuntimeError("observed global blend stage state drifted from frozen plan")
            state_path = stage_states_dir / f"stage-{stage_index}-step-{global_step:05d}.json"
            _write_json(
                state_path,
                {
                    **expected_state,
                    "component_order": list(COMPOSITION_COMPONENTS),
                    "observed_after_physics_step": True,
                    "usd_readback_transmission_color": list(
                        last_global_color_readback
                    ),
                    "spatial_mixing_color_rendered": False,
                    "spatial_layering_color_rendered": False,
                },
            )
            stage_state_paths.append(state_path)
            observed_stage_states.append(
                {
                    **expected_state,
                    "usd_readback_transmission_color": list(
                        last_global_color_readback
                    ),
                    "state_artifact_path": str(state_path.relative_to(output)),
                    "state_artifact_sha256": _sha256(state_path),
                }
            )
            iso_path = system_path.AppendChild("Isosurface")
            iso_evidence = _substantive_isosurface_mesh_evidence(
                stage, iso_path, UsdGeom
            )
            iso_prim = stage.GetPrimAtPath(iso_path)
            proxy_prim = stage.GetPrimAtPath(LIQUID_PROXY_PATH)
            if not iso_prim.IsValid():
                raise RuntimeError("generated PhysX isosurface prim is missing")
            if not proxy_prim.IsValid() or not proxy_prim.IsA(UsdGeom.Mesh):
                raise RuntimeError("standard liquid semantic proxy is missing")
            labels = {
                str(value)
                for value in UsdSemantics.LabelsQuery(
                    "class", Usd.TimeCode.Default()
                ).ComputeUniqueInheritedLabels(proxy_prim)
            }
            if labels != {"liquid"}:
                raise RuntimeError(
                    f"liquid mesh proxy lacks the liquid semantic label: {sorted(labels)!r}"
                )
            receipt["semantics"]["isosurface_semantic_bound"] = True
            memory_after = _memory_sample(receipt)
            memory_before = stage_memory_before[stage_index]
            if (
                memory_before is not None
                and memory_after - memory_before
                >= STAGE_GPU_MEMORY_GROWTH_STOP_MIB
            ):
                receipt["memory"]["stage_growth_events"].append(
                    {
                        "ordinal": stage_index,
                        "tank_id": pour["tank_id"],
                        "used_gpu_memory_mib_before": memory_before,
                        "used_gpu_memory_mib_after": memory_after,
                        "growth_mib": memory_after - memory_before,
                    }
                )
                raise RuntimeError(
                    "stage GPU memory growth safety stop reached: "
                    f"{memory_after - memory_before} MiB"
                )
            pour_particle_ids = [
                particle_id
                for record in active_row_metadata
                if int(record["pour_ordinal"]) == stage_index
                for particle_id in record["particle_ids"]
            ]
            if not pour_particle_ids:
                raise RuntimeError(f"{pour['tank_id']} particle movement evidence is missing")
            row_by_id = {particle_id: index for index, particle_id in enumerate(shared_particle_ids)}
            initial_by_id = {
                particle_id: shared_initial_positions[index]
                for index, particle_id in enumerate(shared_particle_ids)
            }
            pour_movement = _movement(
                [initial_by_id[particle_id] for particle_id in pour_particle_ids],
                [positions[row_by_id[particle_id]] for particle_id in pour_particle_ids],
            )
            if pour_movement is None:
                raise RuntimeError(f"{pour['tank_id']} particle movement evidence is missing")
            maximum_movement = float(pour_movement["max_m"])
            row = {
                "ordinal": stage_index,
                "tank_id": pour["tank_id"],
                "programmed_volume_ml": pour["programmed_volume_ml"],
                "emitted_particle_count": pour["particle_count"],
                "cumulative_particle_count": cumulative,
                "start_step": pour["start_step"],
                "duration_steps": pour["duration_steps"],
                "end_step": pour["end_step"],
                "stage_capture_step": pour["stage_capture_step"],
                "emission_chunk_count": pour["emission_chunk_count"],
                "emission_events": actual_events[stage_index],
                "finite_positions": _finite_positions(positions),
                "moved_under_physics": maximum_movement > 0.0,
                "maximum_particle_movement_m": maximum_movement,
                "escaped_particle_count": _twin_escaped_count(positions),
                "isosurface_nonempty": bool(
                    iso_evidence["mesh_geometry_valid"] is True
                ),
                "used_gpu_memory_mib_before": stage_memory_before[stage_index],
                "used_gpu_memory_mib_after": memory_after,
                "global_component_counts_particles": component_counts.tolist(),
                "global_composition_fractions": fractions.tolist(),
                "global_transmission_color": _global_transmission_color(
                    component_counts
                ).tolist(),
            }
            receipt["pours"].append(row)
            if row["escaped_particle_count"]:
                raise RuntimeError(f"{pour['tank_id']} has escaped particles")
            if not row["isosurface_nonempty"]:
                raise RuntimeError(f"{pour['tank_id']} has no nonempty isosurface")

        expected_update_count = sum(
            int(pour["emission_chunk_count"]) for pour in plan["pours"]
        )
        if global_blend_update_count != expected_update_count:
            raise RuntimeError(
                f"global blend updated {global_blend_update_count} times; "
                f"expected {expected_update_count}"
            )
        if any(
            actual_events[int(pour["ordinal"])] != pour["emission_schedule"]
            for pour in plan["pours"]
        ):
            raise RuntimeError("executed emission schedule differs from the frozen plan")
        all_blend_readbacks_verified = all(
            np.allclose(
                event["transmission_color"],
                event["usd_readback_transmission_color"],
                rtol=0.0,
                atol=1e-7,
            )
            for event in emission_trace
        )
        if not all_blend_readbacks_verified:
            raise RuntimeError("one or more global blend USD readbacks drifted")
        trace_path = trace_dir / "emission-updates.json"
        _write_json(
            trace_path,
            {
                "model": plan["mixing_model"],
                "component_order": list(COMPOSITION_COMPONENTS),
                "event_count": len(emission_trace),
                "events": emission_trace,
            },
        )
        receipt["global_blend"].update(
            {
                "blend_update_count": global_blend_update_count,
                "updated_on_every_emission_event": True,
                "all_updates_usd_readback_verified": all_blend_readbacks_verified,
                "stage_states": observed_stage_states,
                "emission_trace_event_count": len(emission_trace),
                "emission_trace_path": str(trace_path.relative_to(output)),
                "emission_trace_sha256": _sha256(trace_path),
            }
        )
        if diffusion_update_count != int(plan["timing"]["primary_final_capture_step"]):
            raise RuntimeError("diffusion update count does not cover every physics step")
        receipt["composition_evidence"].update(
            {
                "diffusion_update_count": diffusion_update_count,
                "diffusion_neighbor_count_total": diffusion_neighbor_count,
                "diffusion_max_degree": diffusion_max_degree,
                "diffusion_pair_weight_max": diffusion_pair_weight_max,
                "stage_count": len(receipt["composition_evidence"]["stages"]),
            }
        )

        stream_evidence_records = []
        for pour in plan["pours"]:
            ordinal = int(pour["ordinal"])
            tank_id = str(pour["tank_id"])
            expected_active_steps = list(
                range(int(pour["start_step"]), int(pour["end_step"]) + 1)
            )
            observed_active_steps = stream_active_steps[tank_id]
            if observed_active_steps != expected_active_steps:
                raise RuntimeError(f"{tank_id} render-only stream timing drifted")
            state_path = stream_states_dir / f"stream-{ordinal}-{tank_id}.json"
            state = {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "start_step": int(pour["start_step"]),
                "end_step": int(pour["end_step"]),
                "transmission_color": list(
                    NOMINAL_TANK_TRANSMISSION_COLORS[tank_id]
                ),
                "active_only_during_own_pour": True,
                "observed_active_step_count": len(observed_active_steps),
                "observed_active_first_step": observed_active_steps[0],
                "observed_active_last_step": observed_active_steps[-1],
                "other_streams_hidden_each_step": True,
                "render_only_no_collision": True,
            }
            _write_json(state_path, state)
            stream_evidence_records.append(
                {
                    **{
                        key: state[key]
                        for key in (
                            "ordinal",
                            "tank_id",
                            "start_step",
                            "end_step",
                            "transmission_color",
                            "active_only_during_own_pour",
                        )
                    },
                    "state_artifact_path": str(state_path.relative_to(output)),
                    "state_artifact_sha256": _sha256(state_path),
                }
            )
        receipt["stream_render"]["records"] = stream_evidence_records
        receipt["timing"]["executed_physics_steps"] = int(
            plan["timing"]["primary_final_capture_step"]
        )

        if shared_particle_points is None:
            raise RuntimeError("timeline completed without a runtime particle set")
        final_particle_ids = _read_runtime_particle_identity(
            shared_particle_points, shared_particle_ids
        )
        if (
            particle_identity_check_count
            != int(plan["timing"]["primary_final_capture_step"])
            or particle_identity_mismatch_count != 0
        ):
            raise RuntimeError("particle identity was not observed at every callback")
        runtime_property_specs: dict[str, bool] = {}
        shared_points_path = shared_particle_points.GetPrim().GetPath()
        for property_name in ("points", "velocities", "widths", "ids"):
            property_path = Sdf.Path(shared_points_path).AppendProperty(property_name)
            runtime_property_specs[f"{property_name}_session_layer_spec"] = (
                session_layer.GetPropertyAtPath(property_path) is not None
            )
            runtime_property_specs[f"{property_name}_root_layer_spec"] = (
                stage.GetRootLayer().GetPropertyAtPath(property_path) is not None
            )
        if any(
            not runtime_property_specs[f"{property_name}_session_layer_spec"]
            or runtime_property_specs[f"{property_name}_root_layer_spec"]
            for property_name in ("points", "velocities", "widths", "ids")
        ):
            raise RuntimeError("runtime particle rows escaped the session edit target")
        expected_particle_count = int(plan["totals"]["particle_count"])
        if (
            len(shared_particle_ids) != expected_particle_count
            or shared_particle_ids != list(range(expected_particle_count))
            or len(_positions(shared_particle_points.GetPrim())) != expected_particle_count
            or len(active_row_metadata)
            != sum(int(pour["emission_chunk_count"]) for pour in plan["pours"])
        ):
            raise RuntimeError("shared runtime particle identity/count contract failed")
        callback_indices = [int(row["callback_index"]) for row in callback_records]
        callback_dts = [float(row["dt_s"]) for row in callback_records]
        callback_times = [float(row["timeline_time_s"]) for row in callback_records]
        callback_app_update_indices = [
            int(row["app_update_index"]) for row in callback_records
        ]
        capture_callbacks_advanced = sum(
            int(row["callbacks_after"]) - int(row["callbacks_before"])
            for row in capture_guard_records
        )
        capture_time_advanced = any(
            not math.isclose(
                float(row["timeline_time_before_s"]),
                float(row["timeline_time_after_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for row in capture_guard_records
        )
        capture_playback_preserved = all(
            bool(row["timeline_playing_before"])
            == bool(row["timeline_playing_after"])
            for row in capture_guard_records
        )
        receipt["timing"].update(
            {
                "callbacks_observed": len(callback_records),
                "callback_indices": callback_indices,
                "callback_dts_s": callback_dts,
                "callback_times_s": callback_times,
                "callback_app_update_indices": callback_app_update_indices,
                "one_callback_per_app_update": len(callback_records)
                == int(plan["timing"]["primary_final_capture_step"]),
                "emissions_authored_on_main_thread": all(
                    row["authored_on_main_thread"] is True
                    for row in emission_trace
                ),
                "emission_boundary_verified": all(
                    row["authored_after_callback_count"]
                    == row["scheduled_before_callback_index"] - 1
                    == row["physics_step"] - 1
                    for row in emission_trace
                ),
                "emission_boundaries": [
                    {
                        key: row[key]
                        for key in (
                            "physics_step",
                            "authored_after_callback_count",
                            "scheduled_before_callback_index",
                            "authored_on_main_thread",
                        )
                    }
                    for row in emission_trace
                ],
                "points_session_layer_only": runtime_property_specs[
                    "points_session_layer_spec"
                ] and not runtime_property_specs["points_root_layer_spec"],
                "velocities_session_layer_only": runtime_property_specs[
                    "velocities_session_layer_spec"
                ] and not runtime_property_specs["velocities_root_layer_spec"],
                "widths_session_layer_only": runtime_property_specs[
                    "widths_session_layer_spec"
                ] and not runtime_property_specs["widths_root_layer_spec"],
                "ids_session_layer_only": runtime_property_specs[
                    "ids_session_layer_spec"
                ] and not runtime_property_specs["ids_root_layer_spec"],
                "capture_callbacks_advanced": capture_callbacks_advanced,
                "capture_timeline_time_advanced": capture_time_advanced,
                "capture_playback_preserved": capture_playback_preserved,
                "capture_guard_records": capture_guard_records,
            }
        )

        systems = [prim for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)]
        particle_prims = [prim for prim in stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxParticleSetAPI)]
        receipt["particle_system"].update(
            {
                "count": len(systems),
                "exactly_one": len(systems) == 1,
                "particle_set_count": len(particle_prims),
                "mass_api_count": sum(1 for prim in particle_prims if prim.HasAPI(UsdPhysics.MassAPI)),
                "all_particle_sets_target_one_system": all(
                    [str(path) for path in PhysxSchema.PhysxParticleAPI(PhysxSchema.PhysxParticleSetAPI(prim)).GetParticleSystemRel().GetTargets()] == [str(system_path)]
                    for prim in particle_prims
                ),
                "runtime_points_path": str(shared_points_path),
                "runtime_property_specs": runtime_property_specs,
                "runtime_particle_row_count": expected_particle_count,
                "runtime_particle_ids_exact": final_particle_ids
                == list(range(expected_particle_count)),
                "runtime_particle_ids_authored": True,
                "runtime_particle_ids_exact_each_callback": True,
                "runtime_particle_identity_check_count": particle_identity_check_count,
                "runtime_particle_id_mismatch_count": particle_identity_mismatch_count,
                "runtime_particle_ids_final_sha256": _particle_ids_sha256(
                    final_particle_ids
                ),
                "runtime_emission_event_count": len(active_row_metadata),
            }
        )
        final_movement = _movement(
            shared_initial_positions, _positions(shared_particle_points.GetPrim())
        )
        if final_movement is None:
            raise RuntimeError("final particle movement evidence is missing")
        maximum_particle_movement = float(final_movement["max_m"])
        receipt["particle_system"].update(
            {
                "finite_motion_observed": maximum_particle_movement > 0.0,
                "maximum_particle_movement_m": maximum_particle_movement,
            }
        )
        final_positions = _positions(shared_particle_points.GetPrim())
        receipt["totals"].update({"emitted_particle_count": cumulative, "final_particle_count": len(final_positions)})
        receipt["isosurface"].update(
            _substantive_isosurface_mesh_evidence(
                stage, system_path.AppendChild("Isosurface"), UsdGeom
            )
        )
        receipt["isosurface"]["visible_for_global_blend"] = bool(
            receipt["isosurface"].get("mesh_geometry_valid") is True
            and receipt["material"]["isosurface_global_material_binding"]
            and mesh_proxy_created
            and mesh_proxy_parity_passed
        )
        liquid_representations = [
            str(row["liquid_representation"])
            for row in semantic_mask_records
        ]
        vial_proxy_indices = [
            index
            for index, value in enumerate(liquid_representations)
            if value == "vial_proxy"
        ]
        if not vial_proxy_indices:
            raise RuntimeError("no substantive vial-proxy capture was observed")
        first_vial_proxy_index = vial_proxy_indices[0]
        representation_monotonic = (
            liquid_representations[:first_vial_proxy_index]
            == ["stream_only"] * first_vial_proxy_index
            and liquid_representations[first_vial_proxy_index:]
            == ["vial_proxy"]
            * (len(liquid_representations) - first_vial_proxy_index)
        )
        if not representation_monotonic:
            raise RuntimeError("liquid representation downgraded after vial proxy")
        if any(index < first_vial_proxy_index for index in stage_frame_indices):
            raise RuntimeError("a stage frame used stream-only representation")
        receipt["mesh_proxy"].update(
            {
                "refresh_records": mesh_proxy_refresh_records,
                "source_hidden_at_all_vial_proxy_captures": bool(
                    mesh_proxy_refresh_records
                    and all(
                        row["source_visibility"] == "invisible"
                        for row in mesh_proxy_refresh_records
                    )
                ),
                "proxy_visible_at_all_vial_proxy_captures": bool(
                    mesh_proxy_refresh_records
                    and all(
                        row["proxy_visibility"] == "inherited"
                        for row in mesh_proxy_refresh_records
                    )
                ),
                "one_shared_rgb_semantic_capture_per_vial_proxy_frame": bool(
                    len(mesh_proxy_refresh_records) == len(vial_proxy_indices)
                    and all(
                        row["shared_rgb_semantic_capture"] is True
                        for row in mesh_proxy_refresh_records
                    )
                ),
            }
        )
        receipt["memory"]["isosurface_enabled_seconds"] = time.monotonic() - isosurface_started
        writer.release()
        writer = None
        decodable, decoded_count = _decode_video(cv2, video_path)
        frame_hashes = [_sha256(path) for path in frame_paths]
        stage_hashes = [_sha256(path) for path in stage_paths]
        receipt["render"].update(
            {
                "frame_count": len(frame_paths),
                "frame_paths": [str(path.relative_to(output)) for path in frame_paths],
                "frame_sha256": frame_hashes,
                "frame_physics_steps": frame_steps,
                "stage_frame_paths": [str(path.relative_to(output)) for path in stage_paths],
                "stage_frame_sha256": stage_hashes,
                "stage_frame_indices": stage_frame_indices,
                "stage_frame_physics_steps": [
                    int(pour["stage_capture_step"]) for pour in plan["pours"]
                ],
                "primary_final_frame_sha256": (
                    stage_hashes[-1] if stage_hashes else None
                ),
                "video_path": str(video_path.relative_to(output)),
                "video_sha256": _sha256(video_path),
                "video_codec": video_codec,
                "video_container": video_path.suffix.removeprefix("."),
                "video_writer_fallback_used": video_fallback,
                "video_decodable": decodable,
                "video_decoded_frame_count": decoded_count,
                "stage_frames_are_video_subsets": bool(len(stage_hashes) == 3 and set(stage_hashes).issubset(set(frame_hashes))),
                "stream_only_frame_count": first_vial_proxy_index,
                "vial_proxy_frame_count": len(vial_proxy_indices),
                "vial_proxy_transition_step": frame_steps[
                    first_vial_proxy_index
                ],
                "liquid_representation_monotonic": representation_monotonic,
            }
        )
        expected_frame_steps = list(
            _build_capture_steps(plan, stride=args.capture_stride)
        )
        if frame_steps != expected_frame_steps:
            raise RuntimeError("kept RGB timeline differs from the frozen 10 Hz sequence")
        if len(set(frame_hashes)) != len(frame_hashes):
            raise RuntimeError("kept RGB sequence contains byte-identical frames")
        mask_hashes = [_sha256(path) for path in liquid_mask_paths]
        semantic_hashes = [_sha256(path) for path in semantic_paths]
        semantic_pairs = [
            {
                "physics_step": frame_steps[index],
                "rgb_path": str(frame_paths[index].relative_to(output)),
                "rgb_sha256": frame_hashes[index],
                "mask_path": str(liquid_mask_paths[index].relative_to(output)),
                "mask_sha256": mask_hashes[index],
                "semantic_path": str(semantic_paths[index].relative_to(output)),
                "semantic_sha256": semantic_hashes[index],
                **semantic_mask_records[index],
                "mask_readable": True,
                "mask_resolution": [width, height],
                "liquid_pixel_count": liquid_pixel_counts[index],
            }
            for index in range(len(frame_paths))
        ]
        receipt["semantics"].update(
            {
                "pairs": semantic_pairs,
                "stage_mask_sha256": [
                    mask_hashes[index] for index in stage_frame_indices
                ],
                "semantic_gate_passed": bool(
                    receipt["semantics"]["isosurface_semantic_bound"]
                    and receipt["semantics"]["stream_semantic_bound"]
                    and receipt["semantics"]["capture_volumes_enabled_readback"]
                    is True
                    and len(semantic_pairs) == len(frame_paths)
                    and all(
                        0 < pair["liquid_pixel_count"] < width * height
                        for pair in semantic_pairs
                    )
                    and all(
                        semantic_pairs[index]["visible_stream_tanks"] == []
                        for index in stage_frame_indices
                    )
                ),
                "frame_semantic_paths": [
                    str(path.relative_to(output)) for path in semantic_paths
                ],
                "frame_semantic_sha256": semantic_hashes,
            }
        )
        receipt["scene_contract"]["source_layers_unchanged"] = all(
            Path(row["resolved_path"]).is_file()
            and _sha256(Path(row["resolved_path"])) == row["sha256"]
            for row in layer_records
        )
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(f"F2 execution failed: {type(exc).__name__}: {exc}")
    finally:
        if writer is not None:
            writer.release()
        if rgb_annotator is not None and render_product is not None:
            try:
                rgb_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"RGB annotator detach failed: {exc}")
        if semantic_annotator is not None and render_product is not None:
            try:
                semantic_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"semantic annotator detach failed: {exc}")
        if timeline is not None:
            try:
                timeline.pause()
                timeline.commit()
            except Exception as exc:
                receipt["warnings"].append(f"timeline pause failed: {exc}")
        physics_step_subscription = None
        if (
            settings is not None
            and velocity_writeback_setting is not None
            and velocity_writeback_initial is not None
        ):
            try:
                settings.set_bool(
                    velocity_writeback_setting, velocity_writeback_initial
                )
                receipt["gpu_simulation"][
                    "velocity_writeback_restored_readback"
                ] = bool(settings.get_as_bool(velocity_writeback_setting))
            except Exception as exc:
                receipt["warnings"].append(
                    f"velocity writeback setting restore failed: {exc}"
                )
        if settings is not None and capture_volumes_initial is not None:
            try:
                settings.set_bool(
                    SYNTHETICDATA_CAPTURE_VOLUMES_SETTING,
                    capture_volumes_initial,
                )
                receipt["semantics"]["capture_volumes_restored_readback"] = bool(
                    settings.get_as_bool(SYNTHETICDATA_CAPTURE_VOLUMES_SETTING)
                )
            except Exception as exc:
                receipt["warnings"].append(
                    f"volume capture setting restore failed: {exc}"
                )
        if isosurface_started is not None and receipt["memory"]["isosurface_enabled_seconds"] is None:
            receipt["memory"]["isosurface_enabled_seconds"] = time.monotonic() - isosurface_started
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
    if receipt["status"] == "pass":
        failures = validate_three_pour_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        if args.out.exists():
            raise FileExistsError(f"output path already exists: {args.out}")
        args.out.mkdir(parents=True)
        _write_json(args.out / "plan.json", build_three_pour_plan())
        return 0
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    owned_app = None
    exit_code = 2
    try:
        app, owned_app = _start_or_reuse_kit()
        receipt = run_sequence(args, app, args.out)
    except Exception as exc:
        receipt = _empty_receipt(args)
        receipt["errors"].append(
            f"Isaac application startup failed: {type(exc).__name__}: {exc}"
        )
    try:
        receipt_path = args.out / "receipt.json"
        _write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pass" else 2
    finally:
        if owned_app is not None:
            owned_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
