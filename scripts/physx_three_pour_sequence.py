#!/usr/bin/env python3
"""Run one bounded GPU PhysX three-pour fallback and preserve visual evidence.

This is a synthetic simulator experiment.  Programmed millilitres select a
documented nominal particle count; they are not measured displaced volume.
PhysX supplies one-system fluid motion and an isosurface.  Source-coloured
render-only streams and a global volume-weighted vial colour are explicit
fallback visualization rules; no spatial dye mixing or absorbance fitting occurs.
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
from fluidmix.render_capture import resolve_class_ids

# Reuse the F1 routines that were exercised on both GB10 and Brev.  This script
# must be copied beside physx_particle_compat_probe.py on the remote host.
from physx_particle_compat_probe import (
    _finite_positions,
    _isosurface_mesh_evidence,
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
    return (counts / total) @ colors


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


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_three_pour_plan()
    script_path = Path(__file__).resolve()
    contract_path = ROOT / "src" / "fluidmix" / "physx_three_pour.py"
    return {
        "schema_version": 4,
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
            "finite_motion_observed": False,
            "maximum_particle_movement_m": 0.0,
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
            "vial_representation": "visible_isosurface_one_global_material",
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
            "particle_composition_tracked": False,
            "diffusion_simulated": False,
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "stage_states": [],
            "emission_trace_event_count": 0,
            "emission_trace_path": None,
            "emission_trace_sha256": None,
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
            "mesh_face_count": 0,
            "mesh_finite_points": False,
            "mesh_extent_nonempty": False,
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "annotator": "semantic_segmentation",
            "resolution": [640, 480],
            "system_binding_path": f"{FLUIDMIX_NAMESPACE}/ParticleSystem",
            "stream_binding_paths": [],
            "isosurface_path": f"{FLUIDMIX_NAMESPACE}/ParticleSystem/Isosurface",
            "isosurface_semantic_bound": False,
            "stream_semantic_bound": False,
            "annotator_attached_after_labels_and_stage_attach": False,
            "pairs": [],
            "stage_mask_sha256": [],
            "semantic_gate_passed": False,
            "frame_semantic_paths": [],
            "frame_semantic_sha256": [],
            "training_eligible": False,
            "ineligibility_reason": "qualification_run_not_case_dataset",
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
            "video_decodable": False,
            "video_decoded_frame_count": 0,
            "stage_frames_are_video_subsets": False,
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


def _all_positions(active_prims: list[Any]) -> list[list[float]]:
    combined: list[list[float]] = []
    for prim in active_prims:
        combined.extend(_positions(prim))
    return combined


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


def _replicator_same_state_step(rep: Any, *, rt_subframes: int) -> None:
    rep.orchestrator.step(
        delta_time=0.0,
        pause_timeline=True,
        rt_subframes=rt_subframes,
        wait_for_render=True,
    )


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
    simulation = None
    stage_attached = False
    render_product = None
    rgb_annotator = None
    semantic_annotator = None
    writer = None
    frame_paths: list[Path] = []
    frame_steps: list[int] = []
    semantic_paths: list[Path] = []
    liquid_mask_paths: list[Path] = []
    liquid_pixel_counts: list[int] = []
    stage_paths: list[Path] = []
    stage_frame_indices: list[int] = []
    stage_state_paths: list[Path] = []
    observed_stage_states: list[dict[str, Any]] = []
    active_prims: list[Any] = []
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, UsdUtils, Vt

        settings = carb.settings.get_settings()
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
        simulation = omni.physx.get_physx_simulation_interface()
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
        parity_dir.mkdir()
        frames_dir.mkdir()
        stages_dir.mkdir()
        semantic_dir.mkdir()
        liquid_masks_dir.mkdir()
        stage_states_dir.mkdir()
        stream_states_dir.mkdir()
        trace_dir.mkdir()
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
                _replicator_same_state_step(rep, rt_subframes=args.rt_subframes)
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

        def acquire_shared_frame(step: int, phase: str) -> tuple[Any, Any, Any, int]:
            invalid_reasons: list[str] = []
            last_rgb = None
            last_id_to_labels: object = None
            last_semantic_unique_ids: list[int] = []
            attempts = 0
            for _attempt in range(1, args.max_capture_tries + 1):
                attempts = _attempt
                check_time_limits()
                _replicator_same_state_step(rep, rt_subframes=args.rt_subframes)
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
                if semantic.shape != (height, width) or not np.issubdtype(
                    semantic.dtype, np.integer
                ):
                    invalid_reasons.append(f"semantic_shape_or_dtype={semantic.shape}/{semantic.dtype}")
                    continue
                last_semantic_unique_ids = [
                    int(value) for value in np.unique(semantic).tolist()
                ]
                try:
                    mapping = resolve_class_ids(info["idToLabels"])
                except Exception as exc:
                    invalid_reasons.append(f"semantic_mapping_invalid={type(exc).__name__}")
                    continue
                liquid_ids = mapping.liquid_ids
                if not liquid_ids:
                    invalid_reasons.append("liquid_semantic_id_missing")
                    continue
                mask = np.isin(semantic, liquid_ids).astype(np.uint8) * 255
                liquid_pixels = int(np.count_nonzero(mask))
                if liquid_pixels <= 0:
                    invalid_reasons.append("liquid_semantic_pixels_empty")
                    continue
                receipt["render"]["capture_attempt_count"] += len(invalid_reasons) + 1
                receipt["render"]["capture_attempts"].append(
                    {
                        "frame_index": len(frame_paths) + 1,
                        "physics_step": step,
                        "phase": phase,
                        "tries": len(invalid_reasons) + 1,
                        "invalid_reasons": invalid_reasons,
                        "included_in_video": True,
                        "semantic_required": True,
                    }
                )
                return rgb, semantic, mask, liquid_pixels
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
                    "semantic_required": True,
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
                f"{FLUIDMIX_NAMESPACE}/ParticleSystem/Isosurface",
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

            plan = receipt["plan"]
            rng = random.Random(args.seed)
            chunk_records_by_step: dict[int, list[dict[str, Any]]] = {}
            for pour in plan["pours"]:
                for event in pour["emission_schedule"]:
                    count = int(event["particle_count"])
                    chunk_ordinal = int(event["ordinal"])
                    path = Sdf.Path(
                        f"{FLUIDMIX_NAMESPACE}/FluidParticles/"
                        f"{pour['tank_id'].upper()}/Chunk{chunk_ordinal:03d}"
                    )
                    points = UsdGeom.Points.Define(stage, path)
                    positions = _twin_chunk_positions(Gf, count, rng)
                    points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
                    points.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, -0.10)] * count))
                    points.GetWidthsAttr().Set(Vt.FloatArray([0.003] * count))
                    set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
                    particle_api = PhysxSchema.PhysxParticleAPI(set_api)
                    particle_api.CreateParticleSystemRel().SetTargets([system_path])
                    particle_api.CreateSelfCollisionAttr().Set(True)
                    particle_api.CreateParticleEnabledAttr().Set(False)
                    particle_api.CreateParticleGroupAttr().Set(0)
                    set_api.CreateFluidAttr().Set(True)
                    chunk_records_by_step.setdefault(
                        int(event["physics_step"]), []
                    ).append(
                        {
                            "prim": points.GetPrim(),
                            "particle_api": particle_api,
                            "particle_count": count,
                            "tank_id": pour["tank_id"],
                            "pour_ordinal": pour["ordinal"],
                            "event": dict(event),
                            "initial_positions": [
                                [float(value[0]), float(value[1]), float(value[2])]
                                for value in positions
                            ],
                        }
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

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("twin F2 stage has no StageCache ID")
        simulation.attach_stage(stage_id)
        stage_attached = True
        isosurface_started = time.monotonic()
        for _ in range(2):
            app.update()
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        semantic_annotator.attach([render_product])
        receipt["semantics"][
            "annotator_attached_after_labels_and_stage_attach"
        ] = True
        for _ in range(2):
            app.update()
        _memory_sample(receipt)

        video_path = output / "three-pour.mp4"
        writer = cv2.VideoWriter(os.fspath(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.video_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v writer did not open")

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

        def capture(step: int, phase: str) -> tuple[Path, list[list[float]], Any]:
            rgb, semantic, liquid_mask, liquid_pixels = acquire_shared_frame(step, phase)
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
            if (
                reloaded_semantic.shape != (height, width)
                or reloaded_mask is None
                or reloaded_mask.shape != (height, width)
                or int(np.count_nonzero(reloaded_mask)) != liquid_pixels
            ):
                raise RuntimeError("saved semantic evidence is unreadable or misaligned")
            writer.write(bgr)
            frame_paths.append(path)
            frame_steps.append(step)
            semantic_paths.append(semantic_path)
            liquid_mask_paths.append(liquid_path)
            liquid_pixel_counts.append(liquid_pixels)
            return path, _all_positions(active_prims), rgb

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

        dt = 1.0 / PHYSICS_HZ
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
        global_blend_update_count = 0
        last_global_color_readback: tuple[float, float, float] | None = None
        cumulative = 0
        for global_step in range(1, int(plan["timing"]["primary_final_capture_step"]) + 1):
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
                records = chunk_records_by_step.get(global_step, [])
                matching = [
                    record
                    for record in records
                    if record["tank_id"] == event["tank_id"]
                    and record["event"] == {
                        key: event[key]
                        for key in ("ordinal", "physics_step", "particle_count")
                    }
                ]
                if len(matching) != 1:
                    raise RuntimeError("pre-authored emission event drifted")
                record = matching[0]
                count = int(event["particle_count"])
                next_component_counts = _update_global_component_counts(
                    component_counts, str(event["tank_id"]), count
                )
                color = _global_transmission_color(next_component_counts)

                def activate_and_update_color() -> tuple[float, float, float]:
                    if not record["particle_api"].GetParticleEnabledAttr().Set(True):
                        raise RuntimeError("could not enable a particle emission chunk")
                    return _set_global_blend_color(global_shader, Gf, color)

                last_global_color_readback = _with_session_edit_context(
                    Usd,
                    stage,
                    session_layer,
                    activate_and_update_color,
                )
                active_prims.append(record["prim"])
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

            def advance_physics() -> None:
                simulation.simulate(dt, global_step * dt)
                simulation.fetch_results()
                app.update()

            _with_session_edit_context(
                Usd,
                stage,
                session_layer,
                advance_physics,
            )
            receipt["timing"]["executed_physics_steps"] = global_step

            if global_step not in capture_steps:
                continue
            receipt["totals"]["final_particle_count"] = len(
                _all_positions(active_prims)
            )
            stage_record = stage_by_step.get(global_step)
            if stage_record is not None:
                set_stream_state(None, cumulative / float(plan["totals"]["particle_count"]))
                app.update()
                phase = f"{stage_record[1]['tank_id']}-stage"
            elif active_tank is not None:
                phase = f"{active_tank}-pour"
            else:
                phase = "post-fill"
            frame_path, positions, _ = capture(global_step, phase)

            if stage_record is None:
                continue
            stage_index, pour = stage_record
            destination = stages_dir / f"stage-{stage_index}-{pour['tank_id']}.png"
            shutil.copyfile(frame_path, destination)
            if _sha256(destination) != _sha256(frame_path):
                raise RuntimeError("stage-frame copy differs from source video frame")
            stage_paths.append(destination)
            stage_frame_indices.append(len(frame_paths) - 1)
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
            iso_evidence = _isosurface_mesh_evidence(stage, iso_path)
            iso_prim = stage.GetPrimAtPath(iso_path)
            if not iso_prim.IsValid():
                raise RuntimeError("generated PhysX isosurface prim is missing")
            labels = {
                str(value)
                for value in UsdSemantics.LabelsQuery(
                    "class", Usd.TimeCode.Default()
                ).ComputeUniqueInheritedLabels(iso_prim)
            }
            if labels != {"liquid"}:
                raise RuntimeError(
                    f"generated isosurface lacks the liquid semantic label: {sorted(labels)!r}"
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
            pour_chunk_records = [
                record
                for records in chunk_records_by_step.values()
                for record in records
                if int(record["pour_ordinal"]) == stage_index
            ]
            pour_movements = [
                _movement(record["initial_positions"], _positions(record["prim"]))
                for record in pour_chunk_records
            ]
            if not pour_movements or any(value is None for value in pour_movements):
                raise RuntimeError(f"{pour['tank_id']} particle movement evidence is missing")
            maximum_movement = max(
                float(value["max_m"])
                for value in pour_movements
                if value is not None
            )
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
                    iso_evidence["mesh_point_count"] > 0
                    and iso_evidence["mesh_face_count"] > 0
                    and iso_evidence["mesh_finite_points"]
                    and iso_evidence["mesh_extent_nonempty"]
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
            }
        )
        all_chunk_records = [
            record
            for records in chunk_records_by_step.values()
            for record in records
        ]
        final_movements = [
            _movement(record["initial_positions"], _positions(record["prim"]))
            for record in all_chunk_records
        ]
        if not final_movements or any(value is None for value in final_movements):
            raise RuntimeError("final particle movement evidence is missing")
        maximum_particle_movement = max(
            float(value["max_m"])
            for value in final_movements
            if value is not None
        )
        receipt["particle_system"].update(
            {
                "finite_motion_observed": maximum_particle_movement > 0.0,
                "maximum_particle_movement_m": maximum_particle_movement,
            }
        )
        final_positions = _all_positions(active_prims)
        receipt["totals"].update({"emitted_particle_count": cumulative, "final_particle_count": len(final_positions)})
        receipt["isosurface"].update(
            _isosurface_mesh_evidence(stage, system_path.AppendChild("Isosurface"))
        )
        receipt["isosurface"]["visible_for_global_blend"] = bool(
            receipt["isosurface"]["mesh_point_count"] > 0
            and receipt["material"]["isosurface_global_material_binding"]
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
                "video_decodable": decodable,
                "video_decoded_frame_count": decoded_count,
                "stage_frames_are_video_subsets": bool(len(stage_hashes) == 3 and set(stage_hashes).issubset(set(frame_hashes))),
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
                    and len(semantic_pairs) == len(frame_paths)
                    and all(pair["liquid_pixel_count"] > 0 for pair in semantic_pairs)
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
        failures = validate_three_pour_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
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
        if stage_attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception as exc:
                receipt["warnings"].append(f"PhysX stage detach failed: {exc}")
        if isosurface_started is not None and receipt["memory"]["isosurface_enabled_seconds"] is None:
            receipt["memory"]["isosurface_enabled_seconds"] = time.monotonic() - isosurface_started
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
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
