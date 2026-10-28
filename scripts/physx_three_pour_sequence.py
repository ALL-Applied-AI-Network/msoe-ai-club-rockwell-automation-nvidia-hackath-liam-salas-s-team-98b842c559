#!/usr/bin/env python3
"""Run one bounded GPU PhysX three-pour fallback and preserve visual evidence.

This is a synthetic simulator experiment. Programmed millilitres select a
documented nominal particle count; they are not measured displaced volume.
PhysX supplies one-system fluid motion.  The default route renders sphere
instances; the explicit count-driven fallback instead hides those instances
and drives a continuous copy of the twin liquid mesh from live contained
particle count and one global uncalibrated blend.  Tracked per-particle
compositions move with the PBD state and exchange with nearby particles; no
absorbance fitting occurs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
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
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.physx_three_pour import (
    CAMERA_PATH,
    CAMERA_CLIPPING_RANGE,
    COMPOSITION_COMPONENTS,
    FLUID_REST_OFFSET_M,
    FLUIDMIX_NAMESPACE,
    FLOOR_CONTACT_OFFSET_M,
    FLOOR_REST_OFFSET_M,
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
    build_paint_ball_three_pour_plan,
    build_three_pour_plan,
    build_composition_palette,
    build_palette_transmission_colors,
    distribute_particle_count,
    map_compositions_to_palette,
    validate_three_pour_receipt,
)
from fluidmix.physx_composition_cohort import (
    build_composition_cohort_plan,
    validate_composition_execution_schedule,
)
from fluidmix.physx_run_variation import (
    BASE_TANK_TRANSMISSION_1CM,
    build_tank_colour_resolution,
    build_run_palette,
    run_palette_from_tank_colour_resolution,
    validate_tank_colour_plan_overlay,
    validate_tank_colour_resolution,
)
from fluidmix.spatial_surface_color import (
    build_height_banded_transmission_colors,
)
from fluidmix.physx_dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    build_recipe8_runtime_leading_residual_schedule,
)
from fluidmix.physx_train_residual_manifest_v2 import (
    build_train_residual_artifacts,
)
from fluidmix.physx_varied_cohort_manifest_v3 import (
    build_varied_cohort_artifacts,
)
from fluidmix.physx_varied_cohort_manifest_v4 import (
    build_varied_cohort_artifacts as build_varied_cohort_artifacts_v4,
)
from fluidmix.physx_particle_contract import (
    C21_EFFECTIVE_PARTICLE_VOLUME_ML,
    C21_RESOLUTION_PROFILE_ID,
    LEGACY_RESOLUTION_PROFILE_ID,
    NOMINAL_PARTICLE_VOLUME_ML,
    particle_resolution_profile,
)
from fluidmix.particle_diffusion import (
    DIFFUSION_DT_S,
    DIFFUSION_NEIGHBOR_RADIUS_M,
    EFFECTIVE_DIFFUSIVITY_M2_S,
    diffuse_particle_compositions_with_receipt,
)
from fluidmix.particle_palette import (
    build_particle_palette,
    mix_extinction_transmission,
    mix_linear_rgb,
)
from fluidmix.volume_height_curve import (
    INTERIOR_WALL_COUNT,
    height_bounds_for_volume_ml,
    height_for_volume_ml,
    volume_height_curve_digest,
)
from fluidmix.physx_volume_gate import particle_bulk_height_from_quantiles_m
from fluidmix.paint_ball_emitter import (
    F2_MOTION_FALSIFIER_STEP,
    F2_SAFE_CAPTURE_STRIDE_STEPS,
    F2_VIDEO_PLAYBACK_FPS,
    F2_WALL_LIMIT_SECONDS,
    COUNT_DRIVEN_DODECAGON_AREA_M2,
    COUNT_DRIVEN_PARTICLE_VOLUME_M3,
    COUNT_DRIVEN_RENDER_LIMITATIONS,
    COUNT_DRIVEN_RENDER_PATH,
    COUNT_DRIVEN_SOURCE_GEOMETRY_SHA256,
    COUNT_DRIVEN_SOURCE_TOPOLOGY_SHA256,
    EMITTER_RENDER_PATH,
    FLOOR_COLLISION_DOCUMENTATION_URL,
    FLOOR_TUNNELING_FAILURE_RECEIPT_SHA256,
    MAX_VELOCITY_M_S as PAINT_BALL_MAX_VELOCITY_M_S,
    PAIRED_CAPTURE_STATE_FIELDS,
    PATHTRACING_SUBFRAME_DOCUMENTATION_URL,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    PERSISTENT_RENDER_STREAM_PATH,
    RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL,
    RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
    RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES,
    RENDER_LIMITATIONS,
    SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL,
    SUPPORTED_TWIN_RT_SUBFRAMES,
    SUPPORTED_TWIN_PATHTRACING_TOTAL_SPP,
    TRANSPARENT_ANNOTATION_DOCUMENTATION_URL,
    WALL_CONTACT_OFFSET_M,
    WALL_REST_OFFSET_M,
    WALL_TUNNELING_FAILURE_RECEIPT_SHA256,
    append_point_instancer_state,
    attach_visible_volume_evidence,
    apply_nuisance_session_layer,
    bind_external_nuisance_receipt,
    build_count_driven_liquid_body_geometry,
    build_capture_record,
    build_palette_material_path,
    build_prototype_specs,
    build_render_state_duplicate_evidence,
    build_runtime_capture_schedule,
    build_time_preserving_video_repeat_plan,
    build_runtime_particle_configuration,
    build_resolution_aware_source_sample,
    build_source_state,
    containment_evidence,
    copy_semantic_effective_render_state,
    persist_runtime_particle_snapshot,
    read_existing_light_inventory,
    robust_contained_surface_z_m,
    signed_mesh_volume_m3,
    validate_paint_ball_receipt,
)
from fluidmix.render_capture import (
    TRANSPARENT_ANNOTATION_OVERRIDE,
    TransparentAnnotationSourceFact,
    exact_instance_semantic_overlap,
    exact_semantic_class_mask,
    plan_transparent_annotation_override,
    resolve_class_ids,
    resolve_instance_paths,
)
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
ISOSURFACE_LIMIT_SECONDS = 2700  # compatibility metadata only; no live lifecycle
GPU_MEMORY_LIMIT_MIB = 40000
# Recipe, emission, capture, diffusion, and video time are all expressed on this
# outer report clock.  ``PHYSICS_HZ`` remains as a compatibility alias for
# older probes; the C20 runner derives the independent PhysX rate explicitly.
REPORT_HZ = 120
PHYSICS_HZ = REPORT_HZ
SUPPORTED_PHYSICS_SUBSTEPS = (1, 2)
RUNTIME_PARTICLE_CONFIGURATION = build_runtime_particle_configuration()
RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML = (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
)
TIMELINE_END_MARGIN_CALLBACKS = 1
SUPPORTED_COLLIDER_TYPE = "Cube"
# Isaac Sim 6.0.1 documents skipped Replicator frames when a stopped timeline
# lets isaacsim.core.throttling enable asynchronous rendering.  Keep F2 on the
# same launch contract as the already exercised FillCam renderer.
REPLICATOR_ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
RTX_RENDER_MODE_SETTING = "/rtx/rendermode"
SPATIAL_OMNISURFACE_MDL = ROOT / "assets" / "mdl" / "FluidmixSpatialOmniSurface.mdl"
SPATIAL_OMNISURFACE_SUBIDENTIFIER = "FluidmixSpatialOmniSurface"
HEIGHT_BAND_COUNT = 8
REPLICATOR_EXTRA_ARGS = (
    f"--{REPLICATOR_ASYNC_SETTING}=false",
    "--/persistent/rtx/modes/rt/enabled=true",
    "--/persistent/rtx/modes/pt/enabled=true",
)
RENDER_TIMING_PHASES = (
    "warmup_app_update",
    "physics_direct_step",
    "diffusion",
    "palette_mapping",
    "prototype_authoring",
    "capture_rgb_orchestrator",
    "capture_semantic_orchestrator",
    "capture_artifact_io",
)
TIMELINE_RENDER_TIMING_PHASES = (
    "warmup_app_update",
    "physics_app_update",
    "diffusion",
    "palette_mapping",
    "prototype_authoring",
    "capture_rgb_orchestrator",
    "capture_semantic_orchestrator",
    "capture_artifact_io",
)
SIMULATION_MANAGER_STEP_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/"
    "isaacsim.core.simulation_manager/docs/index.html#"
    "isaacsim.core.simulation_manager.SimulationManager.step"
)
TRAIN_RESIDUAL_CONTRACT_KINDS = frozenset(
    {"train_residual_v2", "varied_run_residual_v3", "varied_run_residual_v4"}
)
VARIED_PLAN_ARTIFACT_TYPE_V3 = "physx_f2_varied_run_residual_render_plan_v3"
VARIED_PLAN_ARTIFACT_TYPE_V4 = "physx_f2_varied_run_residual_render_plan_v4"
# Compatibility name retained for callers that only know the accepted V3 plan.
VARIED_PLAN_ARTIFACT_TYPE = VARIED_PLAN_ARTIFACT_TYPE_V3


def _build_realtime_video_capture_steps(
    *, final_step: int, report_hz: int, video_fps: float
) -> tuple[int, ...]:
    """Return one simulation state for every fixed-rate encoded video frame."""

    if (
        isinstance(final_step, bool)
        or not isinstance(final_step, int)
        or final_step <= 0
    ):
        raise ValueError("final_step must be a positive integer")
    if (
        isinstance(report_hz, bool)
        or not isinstance(report_hz, int)
        or report_hz <= 0
    ):
        raise ValueError("report_hz must be a positive integer")
    if (
        isinstance(video_fps, bool)
        or not isinstance(video_fps, (int, float))
        or not math.isfinite(float(video_fps))
        or float(video_fps) <= 0.0
    ):
        raise ValueError("video_fps must be a finite positive number")
    interval = report_hz / float(video_fps)
    rounded = round(interval)
    if not math.isclose(interval, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("video_fps must produce an integer report-step interval")
    stride = int(rounded)
    return tuple(range(stride, final_step + 1, stride))


def _set_rtx_render_mode(
    settings: Any,
    receipt: dict[str, Any],
    *,
    mode: str,
    phase: str,
) -> str:
    """Switch RTX mode through the documented setting and require readback."""

    if mode not in {"rt", "pt"}:
        raise ValueError("RTX render mode must be 'rt' or 'pt'")
    if not isinstance(phase, str) or not phase:
        raise ValueError("render-mode phase must be a nonempty string")
    before = str(settings.get_as_string(RTX_RENDER_MODE_SETTING))
    # Isaac 6.2's live setting accepts the renderer names used by
    # SimulationApp.  The command-line enable flags use the shorter rt/pt
    # tokens, but writing those tokens here is not equivalent: measured on the
    # GB10, writing "pt" read back as "RealTimePathTracing".
    setting_value = {
        "rt": "RaytracedLighting",
        "pt": "PathTracing",
    }[mode]
    settings.set_string(RTX_RENDER_MODE_SETTING, setting_value)
    readback = str(settings.get_as_string(RTX_RENDER_MODE_SETTING))
    passed = readback == setting_value
    receipt["render"].setdefault("mode_transitions", []).append(
        {
            "phase": phase,
            "requested": mode,
            "requested_setting_value": setting_value,
            "before": before,
            "readback": readback,
            "passed": passed,
        }
    )
    if not passed:
        raise RuntimeError(
            "RTX render-mode readback mismatch: "
            f"{readback!r} != {setting_value!r}"
        )
    return readback


def _pin_tone_exposure_state(
    settings: Any,
    camera_prim: Any,
    Sdf: Any,
) -> dict[str, Any]:
    """Pin and read back the image-plane settings that otherwise drift by host."""

    tone_path = "/rtx/post/tonemap/op"
    auto_path = "/rtx/post/histogram/enabled"
    initial = {
        "tone_mapping_operator": int(settings.get_as_int(tone_path)),
        "auto_exposure_enabled": bool(settings.get_as_bool(auto_path)),
    }
    settings.set_int(tone_path, 4)  # NVIDIA-documented ACES selector.
    settings.set_bool(auto_path, False)
    exposure_attr = camera_prim.CreateAttribute(
        "exposure", Sdf.ValueTypeNames.Float
    )
    exposure_attr.Set(0.0)
    auto_attr = camera_prim.CreateAttribute(
        "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
    )
    auto_attr.Set(False)
    readback = {
        "tone_mapping_operator": int(settings.get_as_int(tone_path)),
        "auto_exposure_enabled": bool(settings.get_as_bool(auto_path)),
        "camera_exposure_compensation": float(exposure_attr.Get()),
        "camera_auto_exposure_enabled": bool(auto_attr.Get()),
    }
    passed = readback == {
        "tone_mapping_operator": 4,
        "auto_exposure_enabled": False,
        "camera_exposure_compensation": 0.0,
        "camera_auto_exposure_enabled": False,
    }
    if not passed:
        raise RuntimeError("tone/exposure settings did not retain pinned values")
    return {
        "initial": initial,
        "requested": {
            "tone_mapping_operator": 4,
            "tone_mapping_name": "ACES",
            "auto_exposure_enabled": False,
            "camera_exposure_compensation": 0.0,
        },
        "readback": readback,
        "passed": True,
        "documented_source": (
            "https://docs.omniverse.nvidia.com/materials-and-rendering/latest/"
            "rtx_post-processing.html"
        ),
    }
TIMELINE_APP_UPDATE_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/"
    "replicator_tutorials/tutorial_replicator_getting_started.html#"
    "event-triggered-data-capture-timeline-and-simulation"
)
DIRECT_STEP_PROBE_STEPS = F2_MOTION_FALSIFIER_STEP
MOTION_FALSIFIER_STEP = F2_MOTION_FALSIFIER_STEP
PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES = 48
C6_ROTATION_FAILURE_RECEIPT_SHA256 = (
    "bba03fe424408bf90e4585b92bc128e376aa66f8443780a4a76bdd0166793a4c"
)
SYNTHETICDATA_CAPTURE_VOLUMES_SETTING = "/syntheticdata/sensors/captureVolumes"
FIRST_VIAL_BOTTOM_ROI_XYXY_EXCLUSIVE = (280, 300, 340, 430)
FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS = 16
VOLUME_HEIGHT_TOLERANCE_M = 0.006
RENDER_STREAM_RADIUS_M = 0.0015
VIAL_COLLIDER_WALL_COUNT = INTERIOR_WALL_COUNT
COUNT_DRIVEN_BODY_XFORM_PATH = (
    "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid"
)
COUNT_DRIVEN_BODY_MESH_PATH = f"{COUNT_DRIVEN_BODY_XFORM_PATH}/Fill_liquid"
SOURCE_LIQUID_MESH_PATH = f"{SOURCE_LIQUID_PATH}/Fill_liquid"
COUNT_DRIVEN_SOURCE_WORLD_BOTTOM_Z_M = 0.9762065645
COUNT_DRIVEN_SOURCE_TIME_CODE = 686
COUNT_DRIVEN_SOURCE_WORLD_HEIGHT_M = 0.122894103
COUNT_DRIVEN_SOURCE_SIGNED_VOLUME_M3 = 66.762882e-6
COUNT_DRIVEN_SOURCE_EFFECTIVE_AREA_M2 = 543.255377e-6
COUNT_DRIVEN_BODY_CAPTURE_FIELDS = (
    "representation",
    "contained_particle_count",
    "particle_volume_m3",
    "cross_section_area_m2",
    "target_volume_m3",
    "target_height_m",
    "points_sha256",
    "extent_sha256",
    "world_points_sha256",
    "world_bottom_z_m",
    "world_height_m",
    "world_extent_m",
    "post_transform_signed_volume_m3",
    "post_transform_volume_residual_m3",
    "volume_proof_coordinate_path",
    "body_visibility",
    "body_material_color",
    "body_material_color_sha256",
)
EXECUTION_MODEL_INPUT_ALLOWLIST = frozenset(
    {
        "vial_id",
        "batch_id",
        "vial_position",
        "data_origin",
        "split_role",
        "split_version",
        "previous_sim_recipe_id",
        "current_sim_recipe_id",
        "previous_programmed_ml",
        "current_programmed_ml",
        "programmed_dispense_order",
        "stage_image_evidence_ids",
        "quantitative_evidence_ids",
    }
)
EXECUTION_MODEL_INPUT_FORBIDDEN_KEY_FRAGMENTS = (
    "alpha",
    "profile",
    "label",
    "oracle",
    "disposition",
)
WATER_PBD_PRESET_SOURCE_SHA256 = (
    "85da0b0a39931eff3e54cea5ca8987c07d1c903557d87cb609b205ab85f9b3df"
)
# Documented: NVIDIA uses primvars:doNotCastShadows for annotations behind
# transparent meshes.  Observed in A26: leaving it true changed all three
# frozen RGB parity metrics, so A27 scopes true to the semantic-only pass.


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
    parser.add_argument(
        "--realtime-video",
        action="store_true",
        help="capture every 10 Hz video frame with RTX real-time while retaining path-traced stage stills",
    )
    parser.add_argument(
        "--height-banded-body",
        action="store_true",
        help="render the continuous vial body from tracked per-particle composition in height bands",
    )
    parser.add_argument(
        "--run-palette-id",
        help="synthetic run identity; T200/T400 appearance remains fixed for every vial sharing it",
    )
    parser.add_argument(
        "--run-palette-seed",
        type=int,
        help="run-level palette seed, intentionally separate from each vial render seed",
    )
    parser.add_argument(
        "--tank-colours",
        default=argparse.SUPPRESS,
        help="named synthetic tank-colour preset for a new render plan",
    )
    for tank in ("t100", "t200", "t400"):
        parser.add_argument(
            f"--{tank}",
            type=_parse_tank_transmission,
            default=argparse.SUPPRESS,
            metavar="R,G,B",
            help=f"explicit {tank.upper()} 1-cm linear RGB transmission override",
        )
    parser.add_argument(
        "--dye-strength-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="extinction-space dye strength scale for --tank-colours",
    )
    parser.add_argument(
        "--unlock-t100",
        action="store_true",
        default=argparse.SUPPRESS,
        help="allow T100 to differ from clear for --tank-colours",
    )
    parser.add_argument(
        "--randomize-around-preset",
        action="store_true",
        default=argparse.SUPPRESS,
        help="enable deterministic bounded colour randomization",
    )
    parser.add_argument(
        "--tank-colour-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="required seed for --randomize-around-preset",
    )
    parser.add_argument(
        "--tank-colour-label",
        default=argparse.SUPPRESS,
        help="Part-A provenance label for the resolved tank colours",
    )
    parser.add_argument("--rt-subframes", type=int, default=64)
    parser.add_argument(
        "--pathtracing-total-spp",
        type=int,
        default=64,
        help=(
            "PathTracing total samples per pixel and samples per iteration; "
            "kept equal to avoid Replicator subframe oversampling"
        ),
    )
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument(
        "--max-capture-tries",
        type=int,
        default=20,
        help="maximum same-state renderer attempts for one kept RGB frame",
    )
    parser.add_argument(
        "--capture-stride", type=int, default=F2_SAFE_CAPTURE_STRIDE_STEPS
    )
    parser.add_argument(
        "--wall-limit-seconds",
        type=int,
        default=WALL_LIMIT_SECONDS,
        help="supervisor wall-time bound recorded in the run receipt",
    )
    parser.add_argument(
        "--qualification-mode",
        "--qualification-first-stage-only",
        dest="qualification_mode",
        action="store_true",
        help="run the 120 Hz production graph only through the first stream-hidden stage",
    )
    parser.add_argument(
        "--q2-no-render",
        "--q2-no-render-qualification",
        "--qualification-q2-no-render",
        dest="q2_no_render",
        action="store_true",
        help=(
            "run the dedicated Q2 640-particle stream-hidden checkpoint without "
            "Replicator, rendering, or capture"
        ),
    )
    parser.add_argument(
        "--q2-post-emission-settle-steps",
        type=int,
        default=0,
        help=(
            "Q2-only diagnostic callbacks after the 10 mL emission ends; "
            "later pours remain suppressed"
        ),
    )
    parser.add_argument(
        "--q2-solver-position-iterations",
        type=int,
        default=4,
        help="Q2-only diagnostic PhysX particle solver position iterations",
    )
    parser.add_argument(
        "--q2-particle-count",
        type=int,
        default=640,
        help=(
            "Q2-only first-pour particle count for one packing-calibration "
            "check; programmed target remains 10 mL"
        ),
    )
    parser.add_argument(
        "--solver-position-iterations",
        type=int,
        default=4,
        help=(
            "full-run PhysX particle solver position iterations; the default "
            "preserves historical runs"
        ),
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=1,
        help=(
            "PhysX callbacks inside each unchanged 120 Hz report interval; "
            "2 is the isolated C20 zero-residual Recipe-8 ablation"
        ),
    )
    parser.add_argument(
        "--recipe8-residual-volume-ml",
        type=float,
        default=None,
        help=(
            "run the real recipe-8 20/20/20 mL qualification schedule with "
            "either 0 or 1 mL leading residue per dispense"
        ),
    )
    parser.add_argument(
        "--recipe8-effective-particle-volume-ml",
        type=float,
        default=RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        help=(
            "measured packing correction for recipe-8 particle counts; this "
            "does not alter the density-derived nominal particle volume"
        ),
    )
    parser.add_argument(
        "--particle-resolution-profile",
        choices=(LEGACY_RESOLUTION_PROFILE_ID, C21_RESOLUTION_PROFILE_ID),
        default=LEGACY_RESOLUTION_PROFILE_ID,
        help=(
            "explicit particle-resolution profile; C21 is a bounded fine-"
            "resolution qualification and keeps the legacy default unchanged"
        ),
    )
    parser.add_argument(
        "--water-pbd-preset",
        action="store_true",
        help="apply NVIDIA's named water PBD material preset",
    )
    parser.add_argument(
        "--count-driven-twin-liquid-mesh",
        action="store_true",
        help=(
            "hide sphere prototypes from rendering and render a count-driven "
            "session-layer copy of the twin liquid mesh"
        ),
    )
    parser.add_argument(
        "--qualification-water-pbd-preset",
        action="store_true",
        help="deprecated qualification-mode alias for --water-pbd-preset",
    )
    parser.add_argument("--video-fps", type=float, default=F2_VIDEO_PLAYBACK_FPS)
    parser.add_argument(
        "--execution-schedule",
        type=Path,
        help="absolute path to one frozen composition-substitution schedule",
    )
    parser.add_argument(
        "--execution-schedule-sha256",
        help="canonical lowercase SHA-256 declared by the cohort render plan",
    )
    parser.add_argument(
        "--train-render-plan",
        type=Path,
        help="absolute source-bound train residual render-plan JSON",
    )
    parser.add_argument(
        "--train-render-plan-sha256",
        help="canonical lowercase SHA-256 of --train-render-plan",
    )
    parser.add_argument(
        "--train-case-id",
        help="opaque case ID selected from the source-bound train render plan",
    )
    parser.add_argument(
        "--accepted-qualification-receipt",
        type=Path,
        help="absolute accepted Recipe-8 full-run receipt JSON",
    )
    parser.add_argument(
        "--accepted-qualification-receipt-sha256",
        help="byte SHA-256 of --accepted-qualification-receipt",
    )
    parser.add_argument(
        "--nuisance-receipt",
        type=Path,
        help="absolute path to the measured real-photometric nuisance receipt",
    )
    parser.add_argument(
        "--nuisance-receipt-sha256",
        help="byte SHA-256 of --nuisance-receipt",
    )
    parser.add_argument(
        "--nuisance-case-id",
        help="opaque case identifier recorded as nuisance provenance only",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _parse_tank_transmission(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("tank transmission must be exactly R,G,B")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tank transmission channels must be numbers"
        ) from error


def _bind_direct_tank_colour_resolution(args: argparse.Namespace) -> None:
    if hasattr(args, "_tank_colour_resolution"):
        return
    preset_name = getattr(args, "tank_colours", None)
    advanced_names = (
        "t100",
        "t200",
        "t400",
        "dye_strength_scale",
        "unlock_t100",
        "randomize_around_preset",
        "tank_colour_seed",
        "tank_colour_label",
    )
    if preset_name is None:
        if any(hasattr(args, name) for name in advanced_names):
            raise ValueError("tank-colour advanced options require --tank-colours")
        return
    if getattr(args, "run_palette_id", None) is not None or getattr(
        args, "run_palette_seed", None
    ) is not None:
        raise ValueError("tank-colours and legacy run palette options are mutually exclusive")
    randomize = bool(getattr(args, "randomize_around_preset", False))
    seed_present = hasattr(args, "tank_colour_seed")
    if randomize != seed_present:
        raise ValueError(
            "--randomize-around-preset and --tank-colour-seed are required together"
        )
    overrides = {
        tank: getattr(args, tank)
        for tank in ("t100", "t200", "t400")
        if hasattr(args, tank)
    }
    record = build_tank_colour_resolution(
        preset_name,
        transmission_overrides=overrides or None,
        dye_strength_scale=getattr(args, "dye_strength_scale", 1.0),
        t100_clear_lock=not bool(getattr(args, "unlock_t100", False)),
        randomize_around_preset=randomize,
        randomization_seed=getattr(args, "tank_colour_seed", None),
        free_text_label=getattr(args, "tank_colour_label", None),
    )
    setattr(args, "_tank_colour_resolution", record)


def _build_physics_clock_contract(
    *, physics_substeps: int, report_interval_count: int
) -> dict[str, Any]:
    """Derive the independent report and PhysX clocks for one bounded run."""

    if (
        isinstance(physics_substeps, bool)
        or not isinstance(physics_substeps, int)
        or physics_substeps not in SUPPORTED_PHYSICS_SUBSTEPS
    ):
        raise ValueError(
            "physics_substeps must be one of: "
            + ", ".join(str(value) for value in SUPPORTED_PHYSICS_SUBSTEPS)
        )
    if (
        isinstance(report_interval_count, bool)
        or not isinstance(report_interval_count, int)
        or report_interval_count <= 0
    ):
        raise ValueError("report_interval_count must be a positive integer")
    physics_steps_per_second = REPORT_HZ * physics_substeps
    physics_callback_count = report_interval_count * physics_substeps
    physical_duration_s = report_interval_count / float(REPORT_HZ)
    return {
        "report_hz": REPORT_HZ,
        "report_dt_s": 1.0 / float(REPORT_HZ),
        "report_interval_count": report_interval_count,
        "physics_steps_per_second": physics_steps_per_second,
        "physics_substeps_per_report_interval": physics_substeps,
        "physics_dt_s": 1.0 / float(physics_steps_per_second),
        "physics_callback_count": physics_callback_count,
        "physical_duration_s": physical_duration_s,
        "min_frame_rate": REPORT_HZ,
        "timeline_ticks_per_frame": 1,
        "timeline_end_s": (
            report_interval_count + TIMELINE_END_MARGIN_CALLBACKS
        )
        / float(REPORT_HZ),
        "diffusion_updates_per_report_interval": 1,
        "diffusion_update_count": report_interval_count,
        "diffusion_dt_s": 1.0 / float(REPORT_HZ),
        "diffusion_integrated_time_s": physical_duration_s,
    }


def _emission_callback_boundary(
    *, report_step: int, physics_substeps: int
) -> dict[str, Any]:
    """Map one unchanged emission report step to its following callbacks."""

    _build_physics_clock_contract(
        physics_substeps=physics_substeps,
        report_interval_count=report_step,
    )
    first_callback = (report_step - 1) * physics_substeps + 1
    return {
        "report_step": report_step,
        "authored_after_callback_count": first_callback - 1,
        "scheduled_before_callback_index": first_callback,
        "scheduled_before_callback_indices": list(
            range(first_callback, first_callback + physics_substeps)
        ),
    }


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
        "wall_limit_seconds",
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
    if (args.run_palette_id is None) != (args.run_palette_seed is None):
        raise ValueError(
            "run_palette_id and run_palette_seed must be provided together"
        )
    if args.run_palette_id is not None and not args.run_palette_id.strip():
        raise ValueError("run_palette_id must be nonempty")
    _bind_direct_tank_colour_resolution(args)
    if args.height_banded_body and not args.count_driven_twin_liquid_mesh:
        raise ValueError(
            "height_banded_body requires count_driven_twin_liquid_mesh"
        )
    if args.realtime_video and not args.height_banded_body:
        raise ValueError(
            "realtime_video requires height_banded_body for the new cohort contract"
        )
    if (args.realtime_video or args.height_banded_body) and not SPATIAL_OMNISURFACE_MDL.is_file():
        raise ValueError("reviewed spatial OmniSurface MDL asset is missing")
    if (
        isinstance(args.q2_post_emission_settle_steps, bool)
        or not isinstance(args.q2_post_emission_settle_steps, int)
        or args.q2_post_emission_settle_steps < 0
    ):
        raise ValueError("q2_post_emission_settle_steps must be a non-negative integer")
    if (
        isinstance(args.q2_solver_position_iterations, bool)
        or not isinstance(args.q2_solver_position_iterations, int)
        or not 1 <= args.q2_solver_position_iterations <= 255
    ):
        raise ValueError("q2_solver_position_iterations must be an integer from 1 to 255")
    if (
        isinstance(args.q2_particle_count, bool)
        or not isinstance(args.q2_particle_count, int)
        or not 1 <= args.q2_particle_count <= 4096
    ):
        raise ValueError("q2_particle_count must be an integer from 1 to 4096")
    if (
        isinstance(args.solver_position_iterations, bool)
        or not isinstance(args.solver_position_iterations, int)
        or not 1 <= args.solver_position_iterations <= 255
    ):
        raise ValueError("solver_position_iterations must be an integer from 1 to 255")
    if (
        isinstance(args.physics_substeps, bool)
        or not isinstance(args.physics_substeps, int)
        or args.physics_substeps not in SUPPORTED_PHYSICS_SUBSTEPS
    ):
        raise ValueError(
            "physics_substeps must be one of: "
            + ", ".join(str(value) for value in SUPPORTED_PHYSICS_SUBSTEPS)
        )
    if not getattr(args, "q2_no_render", False) and (
        args.q2_post_emission_settle_steps != 0
        or args.q2_solver_position_iterations != 4
        or args.q2_particle_count != 640
    ):
        raise ValueError("Q2 ablation options require q2_no_render")
    if args.rt_subframes not in SUPPORTED_TWIN_RT_SUBFRAMES:
        supported = ", ".join(str(value) for value in SUPPORTED_TWIN_RT_SUBFRAMES)
        raise ValueError(f"rt_subframes must be one of: {supported}")
    if args.pathtracing_total_spp not in SUPPORTED_TWIN_PATHTRACING_TOTAL_SPP:
        supported = ", ".join(
            str(value) for value in SUPPORTED_TWIN_PATHTRACING_TOTAL_SPP
        )
        raise ValueError(f"pathtracing_total_spp must be one of: {supported}")
    if args.warmup != 60:
        raise ValueError("twin F2 requires warmup=60")
    if args.capture_stride != F2_SAFE_CAPTURE_STRIDE_STEPS:
        raise ValueError(
            "Paint Ball F2 capture_stride must be 312 for the frozen safe "
            "review schedule at 120 Hz"
        )
    if not math.isclose(
        float(args.video_fps), F2_VIDEO_PLAYBACK_FPS, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "twin F2 video_fps must be 10 Hz playback; sparse render capture "
            "cadence remains independently frozen at 120/312 Hz"
        )
    if args.qualification_water_pbd_preset and not args.qualification_mode:
        raise ValueError(
            "qualification_water_pbd_preset requires qualification_mode"
        )
    train_binding_values = (
        args.train_render_plan,
        args.train_render_plan_sha256,
        args.train_case_id,
        args.accepted_qualification_receipt,
        args.accepted_qualification_receipt_sha256,
    )
    any_train_binding = any(value is not None for value in train_binding_values)
    if args.physics_substeps == 2:
        if args.solver_position_iterations != 16:
            raise ValueError("physics_substeps=2 requires 16 solver iterations")
        if not args.water_pbd_preset:
            raise ValueError("physics_substeps=2 requires the water PBD preset")
        if not args.count_driven_twin_liquid_mesh:
            raise ValueError("physics_substeps=2 requires the count-driven render path")
        if args.qualification_mode or getattr(args, "q2_no_render", False):
            raise ValueError("physics_substeps=2 requires the full Recipe-8 run")
        if any_train_binding:
            if args.rt_subframes != 64 or args.pathtracing_total_spp != 64:
                raise ValueError(
                    "two-substep train cases require rt_subframes=64 and "
                    "pathtracing_total_spp=64"
                )
        elif (
            args.seed != 20261025
            or args.rt_subframes != 64
            or args.pathtracing_total_spp != 64
        ):
            raise ValueError(
                "frozen C20 requires seed=20261025, rt_subframes=64, "
                "and pathtracing_total_spp=64"
            )
        if any(
            value is not None
            for value in (
                args.nuisance_receipt,
                args.nuisance_receipt_sha256,
                args.nuisance_case_id,
            )
        ):
            raise ValueError("frozen C20 requires nuisance inputs to be absent")
        if any_train_binding:
            if args.recipe8_residual_volume_ml is not None:
                raise ValueError(
                    "frozen C20 qualification cannot be used for training or "
                    "cohort; two-substep train cases cannot also use Recipe-8 "
                    "residual inputs"
                )
            if (
                args.execution_schedule is not None
                or args.execution_schedule_sha256 is not None
            ):
                raise ValueError(
                    "two-substep train cases cannot use legacy execution schedules"
                )
            train_contract = _execution_schedule_contract(args)
            if train_contract["qualification"].get("physics_substeps") != 2:
                raise ValueError(
                    "two-substep train cases require an accepted two-substep qualification"
                )
        else:
            if args.recipe8_residual_volume_ml is None:
                raise ValueError(
                    "physics_substeps=2 without train bindings is confined to the "
                    "frozen Recipe-8 qualification"
                )
            if float(args.recipe8_residual_volume_ml) != 0.0:
                raise ValueError(
                    "physics_substeps=2 requires the zero-residual qualification"
                )
            if (
                args.execution_schedule is not None
                or args.execution_schedule_sha256 is not None
            ):
                raise ValueError(
                    "physics_substeps=2 qualification cannot use an execution schedule"
                )
    if getattr(args, "q2_no_render", False):
        if args.qualification_mode:
            raise ValueError("q2_no_render is already a dedicated qualification mode")
        if args.count_driven_twin_liquid_mesh:
            raise ValueError("q2_no_render cannot use the render-only count-driven mesh")
        if args.execution_schedule is not None or args.execution_schedule_sha256 is not None:
            raise ValueError("q2_no_render cannot use a full-run execution schedule")
        if any_train_binding:
            raise ValueError("q2_no_render cannot use a train render plan")
        if args.nuisance_receipt is not None or args.nuisance_receipt_sha256 is not None:
            raise ValueError("q2_no_render does not bind a render nuisance receipt")
        if args.solver_position_iterations != 4:
            raise ValueError(
                "q2_no_render uses q2_solver_position_iterations, not "
                "solver_position_iterations"
            )
    recipe8_residual = getattr(args, "recipe8_residual_volume_ml", None)
    effective_particle_volume_ml = float(
        getattr(
            args,
            "recipe8_effective_particle_volume_ml",
            RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        )
    )
    resolution_profile_id = getattr(
        args, "particle_resolution_profile", LEGACY_RESOLUTION_PROFILE_ID
    )
    expected_effective_volume_ml = (
        C21_EFFECTIVE_PARTICLE_VOLUME_ML
        if resolution_profile_id == C21_RESOLUTION_PROFILE_ID
        else RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
    )
    if not math.isfinite(effective_particle_volume_ml) or not math.isclose(
        effective_particle_volume_ml,
        expected_effective_volume_ml,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        if resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
            raise ValueError(
                "C21 effective particle volume must match the frozen cubic-scaling hypothesis"
            )
        raise ValueError(
            "recipe8_effective_particle_volume_ml must match the measured "
            "packing correction"
        )
    if resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
        if args.physics_substeps != 2:
            raise ValueError(
                "C21 particle resolution requires the two-substep runtime"
            )
        if any_train_binding:
            if recipe8_residual is not None:
                raise ValueError("C21 train cases require Recipe-8 residual input absent")
            qualification = _execution_schedule_contract(args)["qualification"]
            if qualification.get("particle_resolution_profile_id") != (
                C21_RESOLUTION_PROFILE_ID
            ):
                raise ValueError(
                    "C21 train cases require a C21-qualified receipt binding"
                )
        elif recipe8_residual != 0.0:
            raise ValueError(
                "C21 qualification requires explicit zero Recipe-8 residual"
            )
    if recipe8_residual is not None:
        if (
            not math.isfinite(float(recipe8_residual))
            or float(recipe8_residual) not in (0.0, 1.0)
        ):
            raise ValueError("recipe8_residual_volume_ml must be exactly 0 or 1")
        if args.qualification_mode or getattr(args, "q2_no_render", False):
            raise ValueError("recipe8 residual runs require the complete three-pour path")
        if args.execution_schedule is not None or args.execution_schedule_sha256 is not None:
            raise ValueError(
                "recipe8 residual runs cannot also use a composition execution schedule"
            )
        if any_train_binding:
            raise ValueError(
                "recipe8 residual runs cannot also use a train render plan"
            )
    schedule_path = getattr(args, "execution_schedule", None)
    schedule_sha256 = getattr(args, "execution_schedule_sha256", None)
    if (schedule_path is not None or schedule_sha256 is not None) and any_train_binding:
        raise ValueError(
            "legacy execution schedule and train render plan are mutually exclusive"
        )
    if (schedule_path is None) != (schedule_sha256 is None):
        raise ValueError(
            "execution_schedule and execution_schedule_sha256 are required together"
        )
    if schedule_path is not None:
        contract = _load_composition_execution_schedule(
            schedule_path, schedule_sha256
        )
        setattr(args, "_execution_schedule_contract", contract)
        if args.seed != int(contract["schedule"]["vial_binding"]["render_seed"]):
            raise ValueError("seed must match the execution schedule render seed")
        if args.qualification_mode:
            raise ValueError("execution schedules require all three stages")
        if not args.water_pbd_preset or not args.count_driven_twin_liquid_mesh:
            raise ValueError(
                "execution schedules require water_pbd_preset and "
                "count_driven_twin_liquid_mesh"
            )
        if not args.plan_only:
            raise ValueError(
                "execution schedule rendering remains blocked pending an accepted "
                "full three-stage F2 receipt"
            )
    if any_train_binding:
        if not all(value is not None for value in train_binding_values):
            raise ValueError("train render bindings are required together")
        if args.qualification_mode:
            raise ValueError("train render plans require all three stages")
        if args.rt_subframes != 64 or args.pathtracing_total_spp != 64:
            raise ValueError(
                "train cases require rt_subframes=64 and pathtracing_total_spp=64"
            )
        if not args.water_pbd_preset:
            raise ValueError("train render plans require water_pbd_preset")
        if not args.count_driven_twin_liquid_mesh:
            raise ValueError("train render plans require the count-driven render path")
        if args.solver_position_iterations != 16:
            raise ValueError(
                "train render plans require the qualification-bound 16 solver iterations"
            )
        train_contract = _execution_schedule_contract(args)
        if (
            train_contract["qualification"].get("physics_substeps")
            != args.physics_substeps
        ):
            raise ValueError(
                "accepted qualification physics substeps do not match the train case"
            )
        if args.seed != int(train_contract["render_seed"]):
            raise ValueError("seed must match the train case render seed")
    nuisance_path = getattr(args, "nuisance_receipt", None)
    nuisance_sha256 = getattr(args, "nuisance_receipt_sha256", None)
    if (nuisance_path is None) != (nuisance_sha256 is None):
        raise ValueError(
            "nuisance_receipt and nuisance_receipt_sha256 are required together"
        )
    if nuisance_path is not None:
        case_id = getattr(args, "nuisance_case_id", None)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "nuisance-case-id is required when nuisance-receipt is provided"
            )
        setattr(
            args,
            "_nuisance_receipt_binding",
            bind_external_nuisance_receipt(nuisance_path, nuisance_sha256),
        )
    return width, height


def _water_pbd_preset_treatment() -> dict[str, Any]:
    """Return the frozen NVIDIA water-preset treatment for A29."""

    return {
        "name": "nvidia_add_pbd_material_water",
        "qualification_only": False,
        "supported_modes": ["full_run", "first_stage_qualification"],
        "api": "omni.physx.scripts.particleUtils.AddPBDMaterialWater",
        "source_path": "omni/physx/scripts/particleUtils.py:191-197",
        "source_sha256": WATER_PBD_PRESET_SOURCE_SHA256,
        "expected_meters_per_unit": 1.0,
        "expected_authored_values": {
            "cohesion": 0.01,
            "damping": 0.0,
            "friction": 0.1,
            "surface_tension": 0.0074,
            "viscosity": 1.7e-6,
            "vorticity_confinement": 0.0,
        },
        "density_kg_m3_authored_after_preset": 1000.0,
    }


def _requested_solver_position_iterations(args: argparse.Namespace) -> int:
    """Return the independently recorded full-run solver setting."""

    return int(getattr(args, "solver_position_iterations", 4))


def _adapt_runtime_plan_for_recipe8_residual(
    base_plan: Mapping[str, Any],
    *,
    residual_volume_ml: float,
    effective_particle_volume_ml: float,
    resolution_profile_id: str = LEGACY_RESOLUTION_PROFILE_ID,
) -> dict[str, Any]:
    """Bind the source-pure real-recipe-8 schedule to the existing F2 graph."""

    schedule = build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=residual_volume_ml,
        effective_particle_volume_ml=effective_particle_volume_ml,
        resolution_profile_id=resolution_profile_id,
    )
    profile = particle_resolution_profile(resolution_profile_id)
    nominal_particle_volume_ml = float(profile["nominal_particle_volume_ml"])
    if resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
        reference_schedule = build_recipe8_runtime_leading_residual_schedule(
            residual_volume_ml=residual_volume_ml,
            effective_particle_volume_ml=RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
            resolution_profile_id=LEGACY_RESOLUTION_PROFILE_ID,
        )
        for pour, reference_pour in zip(
            schedule["pours"], reference_schedule["pours"], strict=True
        ):
            reference_by_step = {
                int(event["physics_step"]): int(event["particle_count"])
                for event in reference_pour["emission_schedule"]
            }
            for event in pour["emission_schedule"]:
                reference_count = reference_by_step.get(int(event["physics_step"]))
                if reference_count is None:
                    reference_count = max(
                        1,
                        int(
                            math.floor(
                                int(event["particle_count"])
                                * effective_particle_volume_ml
                                / RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
                                + 0.5
                            )
                        ),
                    )
                event["source_reference_particle_count"] = reference_count
    plan = deepcopy(dict(base_plan))
    cumulative = 0
    component_totals = [0.0] * len(COMPOSITION_COMPONENTS)
    pours: list[dict[str, Any]] = []
    stage_states: list[dict[str, Any]] = []
    for scheduled_pour in schedule["pours"]:
        pour = deepcopy(dict(scheduled_pour))
        particle_count = int(pour["programmed_particle_count"])
        cumulative += particle_count
        pour.update(
            {
                "particle_count": particle_count,
                "cumulative_particle_count": cumulative,
                "nominal_mass_convention_equivalent_ml": particle_count
                * nominal_particle_volume_ml,
                "packing_equivalent_volume_ml": particle_count
                * effective_particle_volume_ml,
                "nominal_error_ml": particle_count * nominal_particle_volume_ml
                - float(pour["programmed_volume_ml"]),
            }
        )
        # There is deliberately no pour-wide composition: each source-pure
        # event owns its residue/current identity.
        pour.pop("emitted_particle_composition", None)
        for event in pour["emission_schedule"]:
            for index, fraction in enumerate(event["particle_composition"]):
                component_totals[index] += int(event["particle_count"]) * float(
                    fraction
                )
        fractions = [value / cumulative for value in component_totals]
        stage_states.append(
            {
                "ordinal": int(pour["ordinal"]),
                "tank_id": str(pour["tank_id"]),
                "physics_step": int(pour["stage_capture_step"]),
                "cumulative_particle_counts": list(component_totals),
                "cumulative_particle_count": cumulative,
                "component_fractions": fractions,
                "transmission_color": mix_extinction_transmission(
                    [fractions]
                )[0].tolist(),
            }
        )
        pours.append(pour)
    if cumulative != int(schedule["totals"]["programmed_particle_count"]):
        raise RuntimeError("recipe-8 schedule did not conserve its particle total")
    plan["pours"] = pours
    plan["totals"] = {
        "programmed_volume_ml": float(schedule["totals"]["programmed_volume_ml"]),
        "particle_count": cumulative,
        "nominal_mass_convention_equivalent_ml": cumulative
        * nominal_particle_volume_ml,
        "packing_equivalent_volume_ml": cumulative
        * effective_particle_volume_ml,
        "nominal_error_ml": cumulative * nominal_particle_volume_ml
        - float(schedule["totals"]["programmed_volume_ml"]),
    }
    plan["timing"].update(
        {
            "physics_hz": int(schedule["timing"]["physics_hz"]),
            "capture_hz": 10,
            "fill_end_step": int(schedule["timing"]["fill_end_step"]),
            "primary_final_capture_step": int(
                schedule["timing"]["fill_plus_0p3_step"]
            ),
            "primary_final_offset_s": 0.3,
            "no_inter_pour_settle": True,
        }
    )
    plan["paint_ball_emitter"]["qualification_first_stage_step"] = int(
        pours[0]["stage_capture_step"]
    )
    plan["fallback_render_model"]["stage_states"] = stage_states
    plan["recipe8_leading_residual"] = {
        "artifact_type": schedule["artifact_type"],
        "physical_sequence": deepcopy(schedule["physical_sequence"]),
        "timing": deepcopy(schedule["timing"]),
        "runtime_event_contract": deepcopy(schedule["runtime_event_contract"]),
        "label_provenance": deepcopy(schedule["label_provenance"]),
        "model_input_record": deepcopy(schedule["model_input_record"]),
        "packing_calibration": deepcopy(schedule["particle_count_calibration"]),
        "particle_count_conserved": bool(
            schedule["totals"]["particle_count_conserved"]
        ),
    }
    plan["count_driven_particle_volume_m3"] = (
        effective_particle_volume_ml * 1.0e-6
    )
    plan["particle_resolution_profile"] = deepcopy(profile)
    plan["particle_resolution_profile"]["effective_particle_volume_ml"] = (
        effective_particle_volume_ml
    )
    plan["particle_resolution_profile"]["source_sampler"] = (
        "c20_3x3_lattice"
        if resolution_profile_id == LEGACY_RESOLUTION_PROFILE_ID
        else "resolution_aware_c20_boundary"
    )
    return plan


def _adapt_runtime_plan_for_train_residual(
    base_plan: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one validated train-only source-pure schedule to the F2 graph."""

    schedule = contract["schedule"]
    effective_particle_volume_ml = float(
        schedule["particle_count_calibration"]["effective_particle_volume_ml"]
    )
    resolution_profile = schedule.get("particle_resolution_profile")
    if not math.isclose(
        effective_particle_volume_ml,
        RECIPE8_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ) or resolution_profile != particle_resolution_profile(
        LEGACY_RESOLUTION_PROFILE_ID
    ):
        raise ValueError("train schedule legacy resolution profile drifted")
    nominal_particle_volume_ml = float(
        resolution_profile["nominal_particle_volume_ml"]
    )
    plan = deepcopy(dict(base_plan))
    cumulative = 0
    component_totals = [0.0] * len(COMPOSITION_COMPONENTS)
    pours: list[dict[str, Any]] = []
    stage_states: list[dict[str, Any]] = []
    for scheduled_pour in schedule["pours"]:
        pour = deepcopy(dict(scheduled_pour))
        particle_count = int(pour["programmed_particle_count"])
        cumulative += particle_count
        pour.update(
            {
                "particle_count": particle_count,
                "cumulative_particle_count": cumulative,
                "nominal_mass_convention_equivalent_ml": (
                    particle_count * nominal_particle_volume_ml
                ),
                "packing_equivalent_volume_ml": (
                    particle_count * effective_particle_volume_ml
                ),
                "nominal_error_ml": (
                    particle_count * nominal_particle_volume_ml
                    - float(pour["programmed_volume_ml"])
                ),
            }
        )
        pour.pop("emitted_particle_composition", None)
        for event in pour["emission_schedule"]:
            for index, fraction in enumerate(event["particle_composition"]):
                component_totals[index] += int(event["particle_count"]) * float(
                    fraction
                )
        fractions = [value / cumulative for value in component_totals]
        stage_states.append(
            {
                "ordinal": int(pour["ordinal"]),
                "tank_id": str(pour["tank_id"]),
                "physics_step": int(pour["stage_capture_step"]),
                "cumulative_particle_counts": list(component_totals),
                "cumulative_particle_count": cumulative,
                "component_fractions": fractions,
                "transmission_color": mix_extinction_transmission(
                    [fractions]
                )[0].tolist(),
            }
        )
        pours.append(pour)
    if cumulative != int(schedule["totals"]["programmed_particle_count"]):
        raise ValueError("train schedule did not conserve its particle total")
    plan["pours"] = pours
    programmed_volume_ml = float(schedule["totals"]["programmed_volume_ml"])
    plan["totals"] = {
        "programmed_volume_ml": programmed_volume_ml,
        "particle_count": cumulative,
        "nominal_mass_convention_equivalent_ml": (
            cumulative * nominal_particle_volume_ml
        ),
        "packing_equivalent_volume_ml": (
            cumulative * effective_particle_volume_ml
        ),
        "nominal_error_ml": (
            cumulative * nominal_particle_volume_ml - programmed_volume_ml
        ),
    }
    plan["timing"].update(
        {
            "physics_hz": int(schedule["timing"]["physics_hz"]),
            "capture_hz": 10,
            "fill_end_step": int(schedule["timing"]["fill_end_step"]),
            "primary_final_capture_step": int(
                schedule["timing"]["fill_plus_0p3_step"]
            ),
            "primary_final_offset_s": 0.3,
            "no_inter_pour_settle": True,
        }
    )
    plan["paint_ball_emitter"]["qualification_first_stage_step"] = int(
        pours[0]["stage_capture_step"]
    )
    plan["fallback_render_model"]["stage_states"] = stage_states
    plan["execution_schedule"] = {
        "adapter_schema_version": {
            "varied_run_residual_v3": 3,
            "varied_run_residual_v4": 4,
            "train_residual_v2": 2,
        }[contract["contract_kind"]],
        "contract_kind": contract["contract_kind"],
        "artifact_type": schedule["artifact_type"],
        "schedule_path": contract["schedule_path"],
        "schedule_sha256": contract["schedule_sha256"],
        "schedule_file_sha256": contract["schedule_file_sha256"],
        "case_id": contract["case_id"],
        "batch_id": contract["batch_id"],
        "renderer_case_binding": deepcopy(schedule["case_binding"]),
        "cohort_plan_sha256": contract["cohort_plan_sha256"],
        "vial_record_sha256": contract["vial_record_sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "source_plan_hashes_match": contract["source_plan_hashes_match"],
        "model_input_allowlist": list(contract["model_input_allowlist"]),
        "model_input_allowlist_passed": contract[
            "model_input_allowlist_passed"
        ],
        "model_input_record": deepcopy(contract["model_input_record"]),
        "render_seed": int(contract["render_seed"]),
        "composition_substitution_only": False,
        "source_pure_segment_schedule": True,
        "initial_extra_particle_count": 0,
        "fixed_particle_count_vs_clean": True,
        "required_max_particles": cumulative,
        "qualification_receipt_sha256": contract["qualification"][
            "receipt_sha256"
        ],
        "qualification_binding": deepcopy(contract["qualification"]),
        "stream_color_policy": "source_pure_emission_segment_material",
        "render_authorized": True,
        "training_eligible": False,
        "block_reason": "render_receipt_validation_and_visual_gate_pending",
    }
    plan["count_driven_particle_volume_m3"] = (
        effective_particle_volume_ml * 1.0e-6
    )
    plan["particle_resolution_profile"] = deepcopy(resolution_profile)
    plan["particle_resolution_profile"]["effective_particle_volume_ml"] = (
        effective_particle_volume_ml
    )
    plan["particle_resolution_profile"]["source_sampler"] = "c20_3x3_lattice"
    return plan


def _runtime_paint_ball_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Return the base runtime plan plus an explicitly requested treatment."""

    plan = build_paint_ball_three_pour_plan()
    recipe8_residual = getattr(args, "recipe8_residual_volume_ml", None)
    if recipe8_residual is not None:
        plan = _adapt_runtime_plan_for_recipe8_residual(
            plan,
            residual_volume_ml=float(recipe8_residual),
            effective_particle_volume_ml=float(
                args.recipe8_effective_particle_volume_ml
            ),
            resolution_profile_id=getattr(
                args,
                "particle_resolution_profile",
                LEGACY_RESOLUTION_PROFILE_ID,
            ),
        )
    execution_contract = _execution_schedule_contract(args)
    if execution_contract is not None:
        if execution_contract.get("contract_kind") in TRAIN_RESIDUAL_CONTRACT_KINDS:
            plan = _adapt_runtime_plan_for_train_residual(
                plan, execution_contract
            )
        else:
            plan = _adapt_runtime_plan_for_execution_schedule(
                plan, execution_contract
            )
    plan = deepcopy(plan)
    tank_colour_resolution = getattr(args, "_tank_colour_resolution", None)
    if tank_colour_resolution is None and execution_contract is not None:
        tank_colour_resolution = execution_contract.get("tank_colour_resolution")
    if tank_colour_resolution is not None:
        tank_colour_resolution = validate_tank_colour_resolution(
            tank_colour_resolution
        )
    new_appearance_contract = bool(
        args.run_palette_id is not None
        or tank_colour_resolution is not None
        or getattr(args, "height_banded_body", False)
        or getattr(args, "realtime_video", False)
    )
    if new_appearance_contract:
        if tank_colour_resolution is not None:
            run_palette = run_palette_from_tank_colour_resolution(
                tank_colour_resolution
            )
            plan["tank_colour_resolution"] = deepcopy(tank_colour_resolution)
        elif args.run_palette_id is None:
            transmission = {
                tank: [
                    float(value)
                    for value in BASE_TANK_TRANSMISSION_1CM[tank]
                ]
                for tank in COMPOSITION_COMPONENTS
            }
            run_palette = {
                "schema_version": "fluidmix-run-palette-v1-base",
                "run_id": "base-run-palette",
                "run_seed": None,
                "reference_depth_m": 0.01,
                "transmission_1cm": transmission,
                "variation_by_tank": {},
                "palette_sha256": _canonical_json_sha256(transmission),
                "interpretation": (
                    "uncalibrated_synthetic_appearance_parameters_not_physical_calibration"
                ),
            }
        else:
            run_palette = build_run_palette(
                run_id=args.run_palette_id,
                run_seed=args.run_palette_seed,
            )
        plan["run_palette"] = deepcopy(run_palette)
        run_extinctions = _source_extinctions_from_transmissions(
            run_palette["transmission_1cm"]
        )
        for state in plan["fallback_render_model"].get("stage_states", []):
            fractions = state.get("component_fractions")
            if fractions is not None:
                state["transmission_color"] = mix_extinction_transmission(
                    [fractions], run_extinctions
                )[0].tolist()
    nuisance_binding = getattr(args, "_nuisance_receipt_binding", None)
    if nuisance_binding is not None:
        plan["nuisance_profile_binding"] = deepcopy(nuisance_binding)
    if _water_pbd_preset_requested(args):
        plan["pbd_material_treatment"] = _water_pbd_preset_treatment()
    if getattr(args, "count_driven_twin_liquid_mesh", False):
        height_banded = bool(getattr(args, "height_banded_body", False))
        particle_volume_m3 = float(
            plan.get(
                "count_driven_particle_volume_m3",
                COUNT_DRIVEN_PARTICLE_VOLUME_M3,
            )
        )
        plan["claims"].update(
            {
                "particle_composition_rendered": height_banded,
                "spatial_mixing_color_rendered": height_banded,
                "spatial_layering_color_rendered": height_banded,
                "global_vial_blend_rendered": not height_banded,
            }
        )
        plan["fallback_render_model"].update(
            {
                "vial_representation": COUNT_DRIVEN_RENDER_PATH,
                "global_blend_rule": (
                    "height_banded_tracked_particle_beer_law_transmission_uncalibrated"
                    if height_banded
                    else "contained_particle_mean_beer_law_transmission_uncalibrated"
                ),
                "spatial_mixing_color_rendered": height_banded,
                "spatial_layering_color_rendered": height_banded,
                "composition_state": (
                    "tracked_pbd_state_height_banded_mesh_color"
                    if height_banded
                    else "tracked_pbd_state_global_mesh_blend"
                ),
                "representation_limit": (
                    "coarse_height_bands_not_continuous_concentration_field"
                    if height_banded
                    else "continuous_count_driven_body_has_no_spatial_or_layered_color"
                ),
            }
        )
        plan["render_treatment"] = {
            "name": COUNT_DRIVEN_RENDER_PATH,
            "supported_modes": ["full_run", "first_stage_qualification"],
            "particle_state": "one_system_pbd_runtime_emission_unchanged",
            "volume_driver": "live_contained_particle_count",
            "particle_volume_m3": particle_volume_m3,
            "cross_section_area_m2": COUNT_DRIVEN_DODECAGON_AREA_M2,
            "global_uncalibrated_blend": not height_banded,
            "spatial_mixing_color_rendered": height_banded,
            "spatial_layering_color_rendered": height_banded,
            "training_eligible": False,
        }
    if args.wall_limit_seconds != WALL_LIMIT_SECONDS:
        plan["execution_limits"] = {
            "wall_limit_seconds": args.wall_limit_seconds,
        }
    return plan


def _valid_lower_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_strict_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
        raise ValueError(f"{description} must be an existing absolute file")

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{description} contains non-finite JSON value {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return value


def _bind_accepted_qualification_receipt(
    path: Path,
    declared_sha256: str,
    *,
    stage_path: Path,
    rt_subframes: int,
    pathtracing_total_spp: int,
    count_driven_twin_liquid_mesh: bool = False,
    particle_resolution_profile_id: str = LEGACY_RESOLUTION_PROFILE_ID,
) -> dict[str, Any]:
    """Bind one substantive clean Recipe-8 full-run receipt before case rendering."""

    if not _valid_lower_sha256(declared_sha256):
        raise ValueError(
            "accepted qualification receipt SHA-256 must be 64 lowercase hex characters"
        )
    receipt = _read_strict_json_object(
        path, description="accepted qualification receipt"
    )
    observed_sha256 = _sha256(path)
    if observed_sha256 != declared_sha256:
        raise ValueError("accepted qualification receipt byte SHA-256 mismatch")
    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    qualification = receipt.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    versions = receipt.get("versions")
    versions = versions if isinstance(versions, Mapping) else {}
    provenance = receipt.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    plan = receipt.get("plan")
    runtime_capture_schedule = receipt.get("runtime_capture_schedule")
    expected_capture_schedule: dict[str, object] | None = None
    if isinstance(plan, Mapping):
        try:
            expected_capture_schedule = build_runtime_capture_schedule(
                plan, capture_stride_steps=F2_SAFE_CAPTURE_STRIDE_STEPS
            )
        except (TypeError, ValueError, KeyError, OverflowError):
            expected_capture_schedule = None
    expected_frame_count = (
        len(expected_capture_schedule["captures"])
        if expected_capture_schedule is not None
        else None
    )
    sparse_capture_contract = bool(
        arguments.get("capture_stride") == F2_SAFE_CAPTURE_STRIDE_STEPS
        and arguments.get("wall_limit_seconds") == F2_WALL_LIMIT_SECONDS
        and isinstance(arguments.get("video_fps"), (int, float))
        and not isinstance(arguments.get("video_fps"), bool)
        and math.isclose(
            float(arguments["video_fps"]),
            F2_VIDEO_PLAYBACK_FPS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and expected_capture_schedule is not None
        and runtime_capture_schedule == expected_capture_schedule
    )
    physics_substeps = arguments.get("physics_substeps", 1)
    timing = receipt.get("timing")
    timing = timing if isinstance(timing, Mapping) else {}
    expected_ablation = (
        {
            "id": "c21_2p0mm_resolution_and_source_boundary",
            "qualification_only": True,
            "training_or_cohort_use_authorized": False,
            "parent_qualification": "c20_two_physx_substeps_per_report_interval",
            "changed_variable": "particle_resolution_profile",
            "authorized_source_sampler_correction": True,
            "literal_one_variable_claim": False,
        }
        if particle_resolution_profile_id == C21_RESOLUTION_PROFILE_ID
        else {
            "id": "c20_two_physx_substeps_per_report_interval",
            "qualification_only": True,
            "training_or_cohort_use_authorized": False,
            "changed_variable": "physics_substeps_per_report_interval",
        }
    )
    expected_final_step = (
        expected_capture_schedule.get("final_plus_3_0_step")
        if expected_capture_schedule is not None
        else None
    )
    two_substep_contract = bool(
        physics_substeps == 2
        and isinstance(expected_final_step, int)
        and not isinstance(expected_final_step, bool)
        and expected_final_step == 4806
        and arguments.get("seed") == 20261025
        and arguments.get("rt_subframes") == 64
        and arguments.get("pathtracing_total_spp") == 64
        and arguments.get("nuisance_receipt") is None
        and arguments.get("nuisance_receipt_sha256") is None
        and arguments.get("nuisance_case_id") is None
        and receipt.get("nuisance_profile_binding") is None
        and receipt.get("nuisance_runtime") is None
        and receipt.get("ablation") == expected_ablation
        and timing.get("report_hz") == 120
        and timing.get("physics_steps_per_second") == 240
        and timing.get("physics_substeps_per_report_interval") == 2
        and timing.get("report_interval_count") == 4806
        and timing.get("executed_report_intervals") == 4806
        and timing.get("physics_callback_count") == 9612
        and timing.get("executed_physics_steps") == 9612
        and timing.get("callbacks_expected") == 9612
        and timing.get("callbacks_observed") == 9612
        and timing.get("diffusion_update_count") == 4806
        and timing.get("diffusion_updates_per_report_interval") == 1
        and timing.get("final_edit_target_is_session") is True
        and all(
            timing.get(name) is True
            for name in (
                "positions_session_layer_only",
                "velocities_session_layer_only",
                "orientations_session_layer_only",
                "scales_session_layer_only",
                "proto_indices_session_layer_only",
                "ids_session_layer_only",
            )
        )
        and timing.get("simulation_points_root_layer_absent") is True
        and timing.get("simulation_points_runtime_readback_passed") is True
    )
    if (
        receipt.get("status") != "pass"
        or qualification.get("first_stage_only") is not False
        or arguments.get("recipe8_residual_volume_ml") != 0.0
        or arguments.get("particle_resolution_profile")
        != particle_resolution_profile_id
        or (
            particle_resolution_profile_id == C21_RESOLUTION_PROFILE_ID
            and receipt.get("particle_resolution_profile")
            != {
                **particle_resolution_profile(particle_resolution_profile_id),
                "source_sampler": "resolution_aware_c20_boundary",
            }
        )
        or arguments.get("solver_position_iterations") != 16
        or arguments.get("count_driven_twin_liquid_mesh")
        is not count_driven_twin_liquid_mesh
        or receipt.get("render_path")
        != (
            COUNT_DRIVEN_RENDER_PATH
            if count_driven_twin_liquid_mesh
            else "point_instancer"
        )
        or versions.get("isaac") != "6.2.0.0"
        or provenance.get("requested_stage_sha256") != _sha256(stage_path)
        or render.get("renderer") != "PathTracing"
        or render.get("resolution") != [640, 480]
        or render.get("rt_subframes") != rt_subframes
        or render.get("pathtracing_total_spp_requested")
        != pathtracing_total_spp
        or render.get("pathtracing_total_spp_readback")
        != pathtracing_total_spp
        or render.get("pathtracing_spp_requested") != pathtracing_total_spp
        or render.get("pathtracing_spp_readback") != pathtracing_total_spp
        or render.get("warmup_updates") != 60
        or not sparse_capture_contract
        or render.get("frame_count") != expected_frame_count
        or type(physics_substeps) is not int
        or physics_substeps not in (1, 2)
        or (physics_substeps == 2 and not two_substep_contract)
    ):
        raise ValueError(
            "accepted qualification receipt is not the clean Recipe-8 Isaac-6.2 "
            "full-run contract or its renderer settings do not match"
        )
    failures = tuple(validate_paint_ball_receipt(receipt))
    if failures:
        raise ValueError(
            "accepted qualification receipt failed substantive validation: "
            + ",".join(failures)
        )
    script_sha256 = provenance.get("script_sha256")
    if not _valid_lower_sha256(script_sha256):
        raise ValueError("accepted qualification receipt lacks its runner hash")
    return {
        "receipt_path": os.fspath(path),
        "receipt_sha256": observed_sha256,
        "runner_script_sha256": script_sha256,
        "stage_sha256": provenance["requested_stage_sha256"],
        "isaac_version": versions["isaac"],
        "physx_version": versions.get("physx"),
        "rt_subframes": render["rt_subframes"],
        "pathtracing_total_spp": render["pathtracing_total_spp_readback"],
        "capture_stride": F2_SAFE_CAPTURE_STRIDE_STEPS,
        "video_fps": F2_VIDEO_PLAYBACK_FPS,
        "expected_frame_count": expected_frame_count,
        "physics_substeps": physics_substeps,
        "report_hz": 120,
        "physics_steps_per_second": 120 * physics_substeps,
        "particle_resolution_profile_id": particle_resolution_profile_id,
        "count_driven_twin_liquid_mesh": count_driven_twin_liquid_mesh,
        "substantive_validation_failures": [],
        "accepted": True,
    }


def _recursive_keys(value: object):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


def _load_train_residual_execution_contract(
    render_plan_path: Path,
    declared_plan_sha256: str,
    case_id: str,
    qualification_receipt_path: Path,
    qualification_receipt_sha256: str,
    *,
    stage_path: Path,
    rt_subframes: int,
    pathtracing_total_spp: int,
    count_driven_twin_liquid_mesh: bool,
    particle_resolution_profile_id: str,
) -> dict[str, Any]:
    """Select one source-rebuilt train schedule and bind its qualification."""

    if not _valid_lower_sha256(declared_plan_sha256):
        raise ValueError(
            "train render plan SHA-256 must be 64 lowercase hex characters"
        )
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("train case ID must be a non-empty string")
    observed_plan = _read_strict_json_object(
        render_plan_path, description="train render plan"
    )
    observed_plan_sha256 = _canonical_json_sha256(observed_plan)
    if observed_plan_sha256 != declared_plan_sha256:
        raise ValueError("train render plan canonical SHA-256 mismatch")
    artifact_type = observed_plan.get("artifact_type")
    if artifact_type == VARIED_PLAN_ARTIFACT_TYPE_V3:
        expected_artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
        contract_kind = "varied_run_residual_v3"
    elif artifact_type == VARIED_PLAN_ARTIFACT_TYPE_V4:
        expected_artifacts = build_varied_cohort_artifacts_v4(repo_root=ROOT)
        contract_kind = "varied_run_residual_v4"
    else:
        expected_artifacts = build_train_residual_artifacts(repo_root=ROOT)
        contract_kind = "train_residual_v2"
    tank_colour_resolution = validate_tank_colour_plan_overlay(observed_plan)
    expected_plan = expected_artifacts["render-schedule-plan.json"]
    if tank_colour_resolution is not None:
        expected_plan = deepcopy(expected_plan)
        expected_plan["tank_colour_resolution"] = deepcopy(
            tank_colour_resolution
        )
        expected_plan["run_palette"] = run_palette_from_tank_colour_resolution(
            tank_colour_resolution
        )
    if observed_plan != expected_plan:
        raise ValueError("train render plan differs from the source-bound train render plan")
    schedules = observed_plan.get("execution_schedules")
    matches = [
        schedule
        for schedule in schedules
        if isinstance(schedule, Mapping)
        and isinstance(schedule.get("case_binding"), Mapping)
        and schedule["case_binding"].get("case_id") == case_id
    ] if isinstance(schedules, list) else []
    if len(matches) != 1:
        raise ValueError("train case ID must select exactly one execution schedule")
    schedule = deepcopy(dict(matches[0]))
    batches = observed_plan.get("batches")
    batch_matches = [
        batch
        for batch in batches
        if isinstance(batch, Mapping)
        and isinstance(batch.get("case_ids"), list)
        and case_id in batch["case_ids"]
    ] if isinstance(batches, list) else []
    if len(batch_matches) != 1:
        raise ValueError("train case ID must select exactly one source-bound batch")
    batch = batch_matches[0]
    case_index = batch["case_ids"].index(case_id)
    seeds = batch.get("render_seed_by_vial")
    if not isinstance(seeds, list) or len(seeds) != len(batch["case_ids"]):
        raise ValueError("train batch render-seed binding is invalid")
    render_seed = seeds[case_index]
    if isinstance(render_seed, bool) or not isinstance(render_seed, int):
        raise ValueError("train render seed must be an integer")

    run_palette_binding = None
    if (
        tank_colour_resolution is None
        and contract_kind in {"varied_run_residual_v3", "varied_run_residual_v4"}
    ):
        case_binding = schedule.get("case_binding")
        run_id = (
            case_binding.get("run_id")
            if isinstance(case_binding, Mapping)
            else None
        )
        run_matches = [
            row
            for row in expected_plan.get("runs", [])
            if isinstance(row, Mapping) and row.get("run_id") == run_id
        ]
        if len(run_matches) != 1:
            raise ValueError("varied train case lacks one run palette binding")
        run_row = run_matches[0]
        palette = run_row.get("palette")
        if (
            not isinstance(run_row.get("run_seed"), int)
            or isinstance(run_row.get("run_seed"), bool)
            or not isinstance(palette, Mapping)
            or palette.get("palette_sha256")
            != case_binding.get("palette_sha256")
        ):
            raise ValueError("varied train case run palette binding is invalid")
        run_palette_binding = {
            "run_palette_id": run_id,
            "run_palette_seed": int(run_row["run_seed"]),
            "palette_sha256": palette["palette_sha256"],
        }

    model_plan = expected_artifacts["model-input-plan.json"]
    model_matches = [
        row
        for row in model_plan.get("vial_records", [])
        if isinstance(row, Mapping) and row.get("case_id") == case_id
    ]
    if len(model_matches) != 1:
        raise ValueError("train case lacks one source-bound model-input row")
    model_input = deepcopy(dict(model_matches[0]))
    forbidden = ("residual", "disposition", "label", "oracle", "severity", "profile")
    if any(
        fragment in key.lower()
        for key in _recursive_keys(model_input)
        for fragment in forbidden
    ):
        raise ValueError("train model-input row contains an oracle key")
    totals = schedule.get("totals")
    if (
        not isinstance(totals, Mapping)
        or totals.get("particle_count_conserved") is not True
        or isinstance(totals.get("programmed_particle_count"), bool)
        or not isinstance(totals.get("programmed_particle_count"), int)
        or totals["programmed_particle_count"] <= 0
        or totals.get("emitted_particle_count")
        != totals.get("programmed_particle_count")
    ):
        raise ValueError("train schedule does not conserve a positive particle count")
    qualification = _bind_accepted_qualification_receipt(
        qualification_receipt_path,
        qualification_receipt_sha256,
        stage_path=stage_path,
        rt_subframes=rt_subframes,
        pathtracing_total_spp=pathtracing_total_spp,
        count_driven_twin_liquid_mesh=count_driven_twin_liquid_mesh,
        particle_resolution_profile_id=particle_resolution_profile_id,
    )
    result = {
        "contract_kind": contract_kind,
        "schedule": schedule,
        "schedule_path": os.fspath(render_plan_path),
        "schedule_sha256": _canonical_json_sha256(schedule),
        "schedule_file_sha256": _sha256(render_plan_path),
        "cohort_plan_sha256": observed_plan_sha256,
        "vial_record_sha256": _canonical_json_sha256(model_input),
        "source_manifest_sha256": _canonical_json_sha256(
            observed_plan["source_sha256"]
        ),
        "source_plan_hashes_match": True,
        "model_input_allowlist": sorted(model_input),
        "model_input_allowlist_passed": True,
        "model_input_record": model_input,
        "required_max_particles": totals["programmed_particle_count"],
        "render_seed": render_seed,
        "case_id": case_id,
        "batch_id": schedule["case_binding"]["batch_id"],
        "qualification": qualification,
        "render_authorized": True,
        "training_eligible": False,
    }
    if run_palette_binding is not None:
        result["run_palette_binding"] = run_palette_binding
    if tank_colour_resolution is not None:
        result["tank_colour_resolution"] = deepcopy(tank_colour_resolution)
    return result


def _execution_schedule_contract(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    cached = getattr(args, "_execution_schedule_contract", None)
    if cached is not None:
        return cached
    path = getattr(args, "execution_schedule", None)
    declared = getattr(args, "execution_schedule_sha256", None)
    train_values = (
        getattr(args, "train_render_plan", None),
        getattr(args, "train_render_plan_sha256", None),
        getattr(args, "train_case_id", None),
        getattr(args, "accepted_qualification_receipt", None),
        getattr(args, "accepted_qualification_receipt_sha256", None),
    )
    if path is None and declared is None and not any(
        value is not None for value in train_values
    ):
        return None
    if path is not None or declared is not None:
        if any(value is not None for value in train_values):
            raise ValueError(
                "legacy execution schedule and train render plan are mutually exclusive"
            )
        if path is None or declared is None:
            raise ValueError(
                "execution_schedule and execution_schedule_sha256 are required together"
            )
        contract = _load_composition_execution_schedule(path, declared)
        contract["contract_kind"] = "legacy_composition_v1"
        setattr(args, "_execution_schedule_contract", contract)
        return contract
    if not all(value is not None for value in train_values):
        raise ValueError("train render bindings are required together")
    contract = _load_train_residual_execution_contract(
        train_values[0],
        train_values[1],
        train_values[2],
        train_values[3],
        train_values[4],
        stage_path=args.stage,
        rt_subframes=args.rt_subframes,
        pathtracing_total_spp=args.pathtracing_total_spp,
        count_driven_twin_liquid_mesh=bool(
            args.count_driven_twin_liquid_mesh
        ),
        particle_resolution_profile_id=getattr(
            args,
            "particle_resolution_profile",
            LEGACY_RESOLUTION_PROFILE_ID,
        ),
    )
    if contract.get("contract_kind") in {
        "varied_run_residual_v3",
        "varied_run_residual_v4",
    }:
        if "tank_colour_resolution" in contract:
            if getattr(args, "_tank_colour_resolution", None) is not None:
                raise ValueError(
                    "direct tank colours and a source-bound resolution are mutually exclusive"
                )
            if (
                getattr(args, "run_palette_id", None) is not None
                or getattr(args, "run_palette_seed", None) is not None
            ):
                raise ValueError(
                    "legacy run palette options and a source-bound resolution are mutually exclusive"
                )
        else:
            palette = contract["run_palette_binding"]
            if (
                getattr(args, "run_palette_id", None)
                != palette["run_palette_id"]
                or getattr(args, "run_palette_seed", None)
                != palette["run_palette_seed"]
            ):
                raise ValueError(
                    "varied train case CLI run palette differs from its source-bound run"
                )
    setattr(args, "_execution_schedule_contract", contract)
    return contract


def _adapt_runtime_plan_for_execution_schedule(
    base_plan: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Apply one validated schedule without changing its particle volume."""

    schedule = contract["schedule"]
    plan = deepcopy(base_plan)
    plan["pours"] = deepcopy(schedule["pours"])
    for pour in plan["pours"]:
        pour["duration_s"] = float(pour["duration_steps"]) / PHYSICS_HZ
        pour["nominal_mass_convention_equivalent_ml"] = (
            int(pour["particle_count"]) * COUNT_DRIVEN_PARTICLE_VOLUME_M3 * 1.0e6
        )
        pour["nominal_error_ml"] = (
            pour["nominal_mass_convention_equivalent_ml"]
            - float(pour["programmed_volume_ml"])
        )
    schedule_timing = schedule["timing"]
    plan["timing"].update(
        {
            "physics_hz": int(schedule_timing["physics_hz"]),
            "capture_hz": int(schedule_timing["capture_hz"]),
            "no_inter_pour_settle": bool(
                schedule_timing["no_inter_pour_settle"]
            ),
            "fill_end_step": int(schedule_timing["fill_end_step"]),
            "primary_final_capture_step": int(
                schedule_timing["stage_capture_steps"][-1]
            ),
            "primary_final_offset_s": float(
                schedule_timing["primary_final_offset_s"]
            ),
            "source": str(schedule_timing["timing_model_source"]),
            "interpretation": str(schedule_timing["timing_interpretation"]),
        }
    )
    maximum = int(schedule["totals"]["maximum_particle_count"])
    programmed = float(schedule["totals"]["programmed_total_volume_ml"])
    equivalent = float(schedule["totals"]["count_derived_total_volume_ml"])
    plan["totals"] = {
        "programmed_volume_ml": programmed,
        "particle_count": maximum,
        "nominal_mass_convention_equivalent_ml": equivalent,
        "nominal_error_ml": equivalent - programmed,
    }
    component_totals = [0.0] * len(COMPOSITION_COMPONENTS)
    cumulative_particle_count = 0
    stage_states = []
    for pour in plan["pours"]:
        particle_count = int(pour["particle_count"])
        composition = [
            float(value) for value in pour["emitted_particle_composition"]
        ]
        component_totals = [
            total + particle_count * fraction
            for total, fraction in zip(component_totals, composition)
        ]
        cumulative_particle_count += particle_count
        fractions = [
            total / cumulative_particle_count for total in component_totals
        ]
        # This is the archived count-driven fallback plan, not the accepted
        # particle palette.  Preserve its historical arithmetic for old
        # receipts; live particle prototypes use Beer-law transmission.
        transmission_color = mix_linear_rgb([fractions])[0].tolist()
        stage_states.append(
            {
                "ordinal": int(pour["ordinal"]),
                "tank_id": str(pour["tank_id"]),
                "physics_step": int(pour["stage_capture_step"]),
                "cumulative_particle_counts": list(component_totals),
                "cumulative_particle_count": cumulative_particle_count,
                "component_fractions": fractions,
                "transmission_color": transmission_color,
            }
        )
    if cumulative_particle_count != maximum:
        raise ValueError(
            "execution schedule particle total differs from its maximum capacity"
        )
    plan["fallback_render_model"]["stage_states"] = stage_states
    plan["paint_ball_emitter"]["qualification_first_stage_step"] = int(
        plan["pours"][0]["stage_capture_step"]
    )
    plan["execution_schedule"] = {
        "adapter_schema_version": 1,
        "artifact_type": schedule["artifact_type"],
        "schedule_path": contract["schedule_path"],
        "schedule_sha256": contract["schedule_sha256"],
        "schedule_file_sha256": contract["schedule_file_sha256"],
        "vial_binding": deepcopy(schedule["vial_binding"]),
        "cohort_plan_sha256": contract["cohort_plan_sha256"],
        "vial_record_sha256": contract["vial_record_sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "source_plan_hashes_match": contract["source_plan_hashes_match"],
        "model_input_allowlist": list(contract["model_input_allowlist"]),
        "model_input_allowlist_passed": contract[
            "model_input_allowlist_passed"
        ],
        "composition_substitution_only": True,
        "initial_extra_particle_count": int(
            schedule["totals"]["initial_extra_particle_count"]
        ),
        "fixed_particle_count_vs_clean": bool(
            schedule["totals"]["fixed_particle_count_vs_clean"]
        ),
        "required_max_particles": maximum,
        "render_authorized": False,
        "training_eligible": False,
        "block_reason": "accepted_full_three_stage_f2_receipt_required",
    }
    return plan


def _water_pbd_preset_requested(args: argparse.Namespace) -> bool:
    """Return whether the full-run flag or its qualification alias was used."""

    return bool(
        getattr(args, "water_pbd_preset", False)
        or getattr(args, "qualification_water_pbd_preset", False)
    )


def _water_pbd_preset_requested_via(args: argparse.Namespace) -> list[str]:
    """Return the exact CLI treatment switches represented by the parsed args."""

    return [
        name
        for name in (
            "water_pbd_preset",
            "qualification_water_pbd_preset",
        )
        if bool(getattr(args, name, False))
    ]


def _author_pbd_material_treatment(
    *,
    pbd: Any,
    material_prim: Any,
    particle_utils: Any,
    use_water_preset: bool,
) -> None:
    """Author either A29's treatment or A28's unchanged material settings."""

    if use_water_preset:
        particle_utils.AddPBDMaterialWater(material_prim)
        pbd.CreateDensityAttr().Set(1000.0)
    else:
        pbd.CreateDensityAttr().Set(1000.0)
        pbd.CreateFrictionAttr().Set(0.1)


def _water_pbd_preset_readback_matches_plan(
    readback: dict[str, Any],
    *,
    meters_per_unit: float,
) -> bool:
    """Return whether A29 actually authored every frozen water-preset value."""

    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        return False
    expected = {
        **_water_pbd_preset_treatment()["expected_authored_values"],
        "density_kg_m3": 1000.0,
    }
    for name, value in expected.items():
        observed = readback.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), float(value), rel_tol=1.0e-6, abs_tol=1.0e-12
            )
        ):
            return False
    return all(
        readback.get(f"{name}_authored") is True
        for name in (
            "density",
            "friction",
            "cohesion",
            "damping",
            "surface_tension",
            "viscosity",
            "vorticity_confinement",
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_evidence(exc: BaseException, *, phase: str) -> dict[str, str]:
    """Preserve the exact failing phase and active exception traceback."""

    return {
        "phase": str(phase),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _programmed_stage_volume_target(
    pours: list[Mapping[str, object]],
    *,
    step: int,
    visible_stream_tanks: tuple[str, ...],
) -> tuple[float, int] | None:
    """Return the cumulative target only at a programmed post-pour stage."""

    if visible_stream_tanks or step not in {
        int(pour["stage_capture_step"]) for pour in pours
    }:
        return None
    completed = [pour for pour in pours if step >= int(pour["end_step"])]
    if not completed:
        return None
    return (
        sum(float(pour["programmed_volume_ml"]) for pour in completed),
        sum(int(pour["particle_count"]) for pour in completed),
    )


def _first_vial_bottom_roi_evidence(mask: Any) -> dict[str, Any]:
    """Count exact liquid-mask pixels in the frozen FillCam vial-bottom ROI."""

    if getattr(mask, "shape", None) != (480, 640):
        raise ValueError("liquid mask must have shape (480, 640)")
    x0, y0, x1, y1 = FIRST_VIAL_BOTTOM_ROI_XYXY_EXCLUSIVE
    liquid_pixels = int((mask[y0:y1, x0:x1] != 0).sum())
    return {
        "roi_xyxy_exclusive": list(FIRST_VIAL_BOTTOM_ROI_XYXY_EXCLUSIVE),
        "liquid_pixel_count_in_roi": liquid_pixels,
        "minimum_liquid_pixels": FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS,
        "passed": liquid_pixels >= FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS,
    }


def _qualification_semantic_gate(
    *,
    records: list[dict[str, Any]],
    target_step: int,
    capture_steps: tuple[int, ...] | list[int] | set[int],
    frame_count: int,
    frame_pixel_count: int,
    point_instancer_semantic_bound: bool,
    stream_semantic_bound: bool,
) -> bool:
    """Fail closed unless the stream-hidden target labels in-vial liquid."""

    observed_steps = [int(record["physics_step"]) for record in records]
    target_records = [
        record for record in records if int(record["physics_step"]) == target_step
    ]
    return bool(
        point_instancer_semantic_bound is True
        and stream_semantic_bound is True
        and len(records) == frame_count
        and sorted(observed_steps) == sorted(int(step) for step in capture_steps)
        and len(target_records) == 1
        and all(
            0 < int(record["liquid_pixel_count"]) < frame_pixel_count
            for record in records
        )
        and all(
            record.get("paired_same_physics_state_rgb_semantic_capture") is True
            and record.get("paired_capture_state_passed") is True
            for record in records
        )
        and target_records[0].get("visible_stream_tanks") == []
        and int(
            target_records[0].get(
                "liquid_pixel_count_in_first_vial_bottom_roi", 0
            )
        )
        >= FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS
    )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_composition_execution_schedule(
    path: Path, declared_sha256: str
) -> dict[str, Any]:
    """Validate one frozen cohort schedule against the local source plan.

    The declared digest is the canonical JSON digest stored by the cohort
    builder, not the byte digest of a particular pretty-printed file.  The
    latter is retained separately in the receipt for file provenance.
    """

    if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
        raise ValueError("execution schedule must be an existing absolute file")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise ValueError("execution schedule SHA-256 must be 64 lowercase hex characters")

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"execution schedule contains non-finite JSON value {value}")

    try:
        schedule = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("execution schedule is unreadable JSON") from exc
    if not isinstance(schedule, dict):
        raise ValueError("execution schedule must contain one JSON object")
    observed_sha256 = _canonical_json_sha256(schedule)
    if observed_sha256 != declared_sha256:
        raise ValueError("execution schedule canonical SHA-256 mismatch")

    cohort_plan = build_composition_cohort_plan(repo_root=ROOT)
    vial_binding = schedule.get("vial_binding")
    if not isinstance(vial_binding, dict):
        raise ValueError("execution schedule vial binding must be an object")
    vial_id = vial_binding.get("vial_id")
    vial_record = next(
        (row for row in cohort_plan["vials"] if row["vial_id"] == vial_id), None
    )
    if vial_record is None:
        raise ValueError("execution schedule vial is not in the frozen cohort plan")
    validate_composition_execution_schedule(
        schedule, plan=cohort_plan, vial_record=vial_record
    )

    model_input = schedule.get("future_model_input_record")
    if not isinstance(model_input, dict) or set(model_input) != set(
        EXECUTION_MODEL_INPUT_ALLOWLIST
    ):
        raise ValueError("execution schedule model-input allowlist mismatch")

    def keys(value: object):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    if any(
        fragment in key.lower()
        for key in keys(model_input)
        for fragment in EXECUTION_MODEL_INPUT_FORBIDDEN_KEY_FRAGMENTS
    ):
        raise ValueError("execution schedule model input contains an oracle key")

    provenance = schedule["provenance_binding"]
    expected_provenance = {
        "cohort_plan_sha256": _canonical_json_sha256(cohort_plan),
        "vial_record_sha256": _canonical_json_sha256(vial_record),
        "source_manifest_sha256": _canonical_json_sha256(
            cohort_plan["source_sha256"]
        ),
    }
    if provenance != expected_provenance:
        raise ValueError("execution schedule source-plan provenance mismatch")
    required_max_particles = schedule["totals"]["maximum_particle_count"]
    if (
        isinstance(required_max_particles, bool)
        or not isinstance(required_max_particles, int)
        or required_max_particles < 2777
    ):
        raise ValueError("execution schedule particle capacity is below 2777")
    return {
        "schedule": schedule,
        "schedule_path": os.fspath(path),
        "schedule_sha256": observed_sha256,
        "schedule_file_sha256": _sha256(path),
        "vial_record": vial_record,
        **expected_provenance,
        "source_plan_hashes_match": True,
        "model_input_allowlist": sorted(EXECUTION_MODEL_INPUT_ALLOWLIST),
        "model_input_allowlist_passed": True,
        "required_max_particles": required_max_particles,
        "render_authorized": bool(cohort_plan["render_gate"]["render_authorized"]),
        "training_eligible": bool(cohort_plan["render_gate"]["training_eligible"]),
    }


def _author_and_attest_particle_capacity(
    prim: Any,
    *,
    value_type: Any,
    required_max_particles: int,
    required_value_source: str = (
        "validated_schedule.totals.maximum_particle_count"
    ),
) -> dict[str, object]:
    """Author particle capacity and fail unless the USD readback is sufficient."""

    if (
        isinstance(required_max_particles, bool)
        or not isinstance(required_max_particles, int)
        or required_max_particles <= 0
    ):
        raise ValueError("required_max_particles must be a positive integer")
    attribute = prim.CreateAttribute("physxParticle:maxParticles", value_type)
    if attribute.Set(required_max_particles) is not True:
        raise RuntimeError("particle capacity could not be authored")
    readback = attribute.Get()
    if (
        isinstance(readback, bool)
        or not isinstance(readback, int)
        or readback < required_max_particles
    ):
        raise RuntimeError("particle capacity readback is below the required value")
    return {
        "required_max_particles": required_max_particles,
        "required_value_source": required_value_source,
        "authored_max_particles": required_max_particles,
        "readback_max_particles": readback,
        "capacity_gate_passed": True,
    }


def _build_emission_event_index(plan: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Index the exact frozen emission events by physics step."""

    indexed: dict[int, list[dict[str, Any]]] = {}
    for pour in plan["pours"]:
        for event in pour["emission_schedule"]:
            step = int(event["physics_step"])
            indexed_event = {
                "tank_id": pour["tank_id"],
                "pour_ordinal": pour["ordinal"],
                **dict(event),
            }
            pour_composition = pour.get("emitted_particle_composition")
            if pour_composition is not None:
                event_composition = event.get("particle_composition")
                if (
                    event_composition is not None
                    and event_composition != pour_composition
                ):
                    raise ValueError(
                        "emission event composition differs from its pour"
                    )
                indexed_event["particle_composition"] = list(
                    pour_composition
                )
            indexed.setdefault(step, []).append(indexed_event)
    return indexed


def _q2_first_pour_event_index(
    plan: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Return only the first-pour events used by the Q2 settle diagnostic."""

    pours = plan.get("pours")
    if not isinstance(pours, list) or not pours:
        raise ValueError("Q2 plan must contain at least one pour")
    first_ordinal = int(pours[0]["ordinal"])
    indexed = _build_emission_event_index(plan)
    return {
        step: [
            dict(event)
            for event in rows
            if int(event["pour_ordinal"]) == first_ordinal
        ]
        for step, rows in indexed.items()
        if any(int(event["pour_ordinal"]) == first_ordinal for event in rows)
    }


def _q2_diffusion_neighbor_diagnostic(
    *, max_degree: int, physx_max_neighborhood: int
) -> dict[str, object]:
    """Label the diffusion graph separately from unavailable native neighbor state."""

    if (
        isinstance(max_degree, bool)
        or not isinstance(max_degree, int)
        or max_degree < 0
    ):
        raise ValueError("max_degree must be a non-negative integer")
    if (
        isinstance(physx_max_neighborhood, bool)
        or not isinstance(physx_max_neighborhood, int)
        or physx_max_neighborhood <= 0
    ):
        raise ValueError("physx_max_neighborhood must be a positive integer")
    return {
        "physx_max_neighborhood": physx_max_neighborhood,
        "max_observed_diffusion_degree": max_degree,
        "cap_saturated": None,
        "unsaturated": None,
        "measurement_source": "runtime_diffusion_graph_not_physx_neighbor_readback",
        "gate_required": False,
        "passed": None,
        "status": "diagnostic_only",
    }


def _build_pour_source_states(plan: Mapping[str, Any]) -> dict[int, Any]:
    """Create exactly one immutable stream/emission identity per active pour."""

    states: dict[int, Any] = {}
    for pour in plan["pours"]:
        ordinal = int(pour["ordinal"])
        tank_id = str(pour["tank_id"])
        composition_value = pour.get("emitted_particle_composition")
        if composition_value is None:
            composition = [0.0] * len(COMPOSITION_COMPONENTS)
            composition[COMPOSITION_COMPONENTS.index(tank_id)] = 1.0
        else:
            composition = [float(value) for value in composition_value]
        prototype_index = int(
            map_compositions_to_palette([composition])[0]
        )
        pure = sum(value == 1.0 for value in composition) == 1 and sum(
            value != 0.0 for value in composition
        ) == 1
        material_path = (
            f"{FLUIDMIX_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
            if pure
            else build_palette_material_path(FLUIDMIX_NAMESPACE, prototype_index)
        )
        states[ordinal] = build_source_state(
            scheduled_tank=tank_id,
            composition=composition,
            palette_prototype_index=prototype_index,
            material_path=material_path,
            pour_event_id=f"pour-{ordinal}-{tank_id}",
            birth_event_id=f"pour-{ordinal}-{tank_id}-births",
            birth_step=int(pour["start_step"]),
        )
    return states


def _source_state_for_event(event: Mapping[str, Any]) -> Any:
    """Build the immutable render/emission identity owned by one event."""

    tank_id = str(event.get("source_tank_id", event.get("tank_id")))
    if tank_id not in COMPOSITION_COMPONENTS:
        raise ValueError(f"unknown event source tank {tank_id!r}")
    composition_value = event.get("particle_composition")
    if composition_value is None:
        composition = [0.0] * len(COMPOSITION_COMPONENTS)
        composition[COMPOSITION_COMPONENTS.index(tank_id)] = 1.0
    else:
        composition = [float(value) for value in composition_value]
    prototype_index = int(map_compositions_to_palette([composition])[0])
    pure = sum(value == 1.0 for value in composition) == 1 and sum(
        value != 0.0 for value in composition
    ) == 1
    material_path = (
        f"{FLUIDMIX_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
        if pure
        else build_palette_material_path(FLUIDMIX_NAMESPACE, prototype_index)
    )
    pour_ordinal = int(event.get("pour_ordinal", 0))
    event_ordinal = int(event.get("ordinal", 0))
    event_id = str(
        event.get(
            "event_id",
            f"pour-{pour_ordinal}-{tank_id}-event-{event_ordinal}",
        )
    )
    return build_source_state(
        scheduled_tank=tank_id,
        composition=composition,
        palette_prototype_index=prototype_index,
        material_path=material_path,
        pour_event_id=str(
            event.get("pour_event_id", f"pour-{pour_ordinal}-{tank_id}")
        ),
        birth_event_id=str(event.get("birth_event_id", event_id)),
        birth_step=int(event["physics_step"]),
    )


def _active_stream_source_for_step(
    plan: Mapping[str, Any], step: int
) -> str | None:
    """Return residue/current source identity without changing pour identity."""

    for pour in plan["pours"]:
        segments = pour.get("emission_segments")
        if isinstance(segments, list):
            for segment in segments:
                start = segment.get("start_step")
                end = segment.get("end_step")
                if (
                    start is not None
                    and int(start) <= step <= int(end)
                    and int(segment.get("particle_count", 0)) > 0
                ):
                    return str(segment["source_tank_id"])
        if int(pour["start_step"]) <= step <= int(pour["end_step"]):
            return str(pour["tank_id"])
    return None


def _runtime_source_state_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe every nonempty residue/current stream segment in the receipt."""

    rows: list[dict[str, Any]] = []
    for pour in plan["pours"]:
        segments = pour.get("emission_segments")
        if not isinstance(segments, list):
            segments = [
                {
                    "segment_role": "current_tank",
                    "source_tank_id": pour["tank_id"],
                    "particle_count": pour["particle_count"],
                    "start_step": pour["start_step"],
                    "end_step": pour["end_step"],
                    "emission_schedule": pour["emission_schedule"],
                }
            ]
        for segment in segments:
            events = segment.get("emission_schedule")
            if not isinstance(events, list) or not events:
                continue
            state = _source_state_for_event(events[0])
            rows.append(
                {
                    "pour_ordinal": int(pour["ordinal"]),
                    "programmed_tank": str(pour["tank_id"]),
                    "segment_ordinal": int(segment.get("ordinal", 1)),
                    "segment_role": str(segment.get("segment_role", "current_tank")),
                    "start_step": int(segment["start_step"]),
                    "end_step": int(segment["end_step"]),
                    "scheduled_tank": state.scheduled_tank,
                    "composition": list(state.composition),
                    "stream_composition": list(state.stream_composition),
                    "birth_composition": list(state.birth_composition),
                    "palette_prototype_index": state.palette_prototype_index,
                    "material_path": state.material_path,
                    "pour_event_id": state.pour_event_id,
                    "birth_event_id": state.birth_event_id,
                    "birth_step": state.birth_step,
                }
            )
    return rows


def _prepare_emission_event(
    event: Mapping[str, Any], *, component_counts: Any
) -> dict[str, Any]:
    """Build the exact tracked rows, fractional totals and palette prototype."""

    import numpy as np

    if not isinstance(event, Mapping):
        raise TypeError("emission event must be a mapping")
    count = event.get("particle_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("emission event particle_count must be a positive integer")
    tank_id = event.get("tank_id")
    if tank_id not in COMPOSITION_COMPONENTS:
        raise ValueError(f"unknown tank {tank_id!r}")
    composition_value = event.get("particle_composition")
    if composition_value is None:
        composition = np.zeros(len(COMPOSITION_COMPONENTS), dtype=np.float64)
        composition[COMPOSITION_COMPONENTS.index(str(tank_id))] = 1.0
    else:
        raw_composition = np.asarray(composition_value)
        if raw_composition.dtype.kind not in "fiu":
            raise TypeError("particle composition must contain real numbers")
        composition = np.asarray(composition_value, dtype=np.float64)
        if (
            composition.shape != (len(COMPOSITION_COMPONENTS),)
            or not np.all(np.isfinite(composition))
            or np.any(composition < 0.0)
            or not math.isclose(
                float(composition.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError(
                "particle composition must be a finite non-negative "
                "length-three vector summing to one"
            )
    current_counts = np.asarray(component_counts)
    if (
        current_counts.dtype.kind not in "fiu"
        or current_counts.shape != (len(COMPOSITION_COMPONENTS),)
    ):
        raise TypeError("component counts must be a numeric length-three vector")
    current_counts = np.asarray(current_counts, dtype=np.float64)
    if not np.all(np.isfinite(current_counts)) or np.any(current_counts < 0.0):
        raise ValueError("component counts must be finite and non-negative")
    source_compositions = np.repeat(composition[None, :], count, axis=0)
    next_component_counts = current_counts + composition * count
    if not math.isclose(
        float(next_component_counts.sum()),
        float(current_counts.sum()) + count,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("emission composition changed the programmed particle count")
    return {
        "particle_count": count,
        "particle_composition": composition.tolist(),
        "source_compositions": source_compositions,
        "next_component_counts": next_component_counts,
        "prototype_index": int(
            map_compositions_to_palette(composition[None, :])[0]
        ),
        "composition_rows_match": bool(
            np.allclose(
                source_compositions,
                composition[None, :],
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
    }


def _active_tank_for_step(plan: dict[str, Any], step: int) -> str | None:
    """Return the programmed source tank for a back-to-back pour step."""

    for pour in plan["pours"]:
        if int(pour["start_step"]) <= step <= int(pour["end_step"]):
            return str(pour["tank_id"])
    return None


def _build_capture_steps(
    plan: dict[str, Any],
    *,
    stride: int,
    runtime_schedule: Mapping[str, object] | None = None,
) -> tuple[int, ...]:
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("capture stride must be a positive integer")
    schedule = runtime_schedule
    if isinstance(schedule, Mapping) and isinstance(schedule.get("captures"), list):
        return tuple(
            sorted({int(row["physics_step"]) for row in schedule["captures"]})
        )
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


def _source_extinctions_from_transmissions(
    transmission_by_tank: Mapping[str, Any],
) -> Any:
    """Return finite 1-cm extinction rows in component order."""

    import numpy as np

    try:
        colors = np.asarray(
            [transmission_by_tank[tank] for tank in COMPOSITION_COMPONENTS],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run palette must contain three numeric tank colors") from exc
    if colors.shape != (3, 3) or not np.all(np.isfinite(colors)):
        raise ValueError("run palette colors must be a finite 3x3 matrix")
    if np.any(colors <= 0.0) or np.any(colors > 1.0):
        raise ValueError("run palette colors must be in (0, 1]")
    return -np.log(colors) / 0.01


def _global_transmission_color(
    component_counts: Any,
    source_extinctions: Any = None,
) -> Any:
    """Apply the Task 1 Beer-law extinction transform to conserved totals."""

    import numpy as np

    counts = np.asarray(component_counts, dtype=np.float64)
    if counts.shape != (len(COMPOSITION_COMPONENTS),) or not np.all(np.isfinite(counts)):
        raise ValueError("component counts must be a finite length-three vector")
    total = float(np.sum(counts))
    if total <= 0.0 or np.any(counts < 0.0):
        raise ValueError("component counts must have a positive non-negative total")
    return mix_extinction_transmission(
        (counts / total)[None, :], source_extinctions
    )[0]


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


def _author_collider_collision_offsets(
    PhysxSchema: Any,
    prim: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Author and read back an explicitly requested collider contact envelope."""

    contact_offset = spec.get("contact_offset_m")
    rest_offset = spec.get("rest_offset_m")
    record = {
        "path": str(spec["path"]),
        "contact_offset_authored": contact_offset is not None,
        "rest_offset_authored": rest_offset is not None,
        "contact_offset_m": None,
        "rest_offset_m": None,
    }
    if contact_offset is None and rest_offset is None:
        return record
    if contact_offset is None or rest_offset is None:
        raise RuntimeError("collider contact and rest offsets must be authored together")
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(float(contact_offset))
    collision_api.CreateRestOffsetAttr().Set(float(rest_offset))
    contact_readback = float(collision_api.GetContactOffsetAttr().Get())
    rest_readback = float(collision_api.GetRestOffsetAttr().Get())
    if not math.isclose(
        contact_readback,
        float(contact_offset),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ) or not math.isclose(
        rest_readback,
        float(rest_offset),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("collider contact/rest offset readback differs from request")
    record.update(
        {
            "contact_offset_m": contact_readback,
            "rest_offset_m": rest_readback,
        }
    )
    return record


def _build_wall_tunneling_mitigation(
    collider_collision_readbacks: list[Mapping[str, Any]],
    *,
    physics_steps_per_second: int = PHYSICS_HZ,
) -> dict[str, Any]:
    """Bind the documented thin-collider envelope to every vial wall readback."""

    if (
        isinstance(physics_steps_per_second, bool)
        or not isinstance(physics_steps_per_second, int)
        or physics_steps_per_second <= 0
    ):
        raise ValueError("physics_steps_per_second must be a positive integer")
    wall_specs = build_twin_collider_specs()[1:]
    wall_readbacks = collider_collision_readbacks[1:]
    expected_paths = [str(spec["path"]) for spec in wall_specs]
    readback_passed = (
        len(wall_readbacks) == len(wall_specs)
        and all(isinstance(item, Mapping) for item in wall_readbacks)
        and [str(item.get("path")) for item in wall_readbacks] == expected_paths
        and all(
            item.get("contact_offset_authored") is True
            and item.get("rest_offset_authored") is True
            and math.isclose(
                float(item.get("contact_offset_m", math.nan)),
                WALL_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                float(item.get("rest_offset_m", math.nan)),
                WALL_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for item in wall_readbacks
        )
    )
    return {
        "documented_source": FLOOR_COLLISION_DOCUMENTATION_URL,
        "observed_failure_receipt_sha256": (
            WALL_TUNNELING_FAILURE_RECEIPT_SHA256
        ),
        "wall_count": len(wall_specs),
        "wall_contact_offset_m": WALL_CONTACT_OFFSET_M,
        "wall_rest_offset_m": WALL_REST_OFFSET_M,
        "maximum_capped_displacement_per_step_m": (
            PAINT_BALL_MAX_VELOCITY_M_S / physics_steps_per_second
        ),
        "contact_offset_exceeds_capped_step": (
            WALL_CONTACT_OFFSET_M
            > PAINT_BALL_MAX_VELOCITY_M_S / physics_steps_per_second
        ),
        "wall_thickness_unchanged_m": 0.001,
        "readback_passed": bool(readback_passed),
    }


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
    count_driven_twin_liquid_mesh = bool(
        getattr(args, "count_driven_twin_liquid_mesh", False)
    )
    receipt = {
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
            "pathtracing_total_spp": args.pathtracing_total_spp,
            "warmup": args.warmup,
            "max_capture_tries": args.max_capture_tries,
            "capture_stride": args.capture_stride,
            "video_fps": args.video_fps,
            "wall_limit_seconds": args.wall_limit_seconds,
            "count_driven_twin_liquid_mesh": count_driven_twin_liquid_mesh,
            "realtime_video": bool(getattr(args, "realtime_video", False)),
            "height_banded_body": bool(
                getattr(args, "height_banded_body", False)
            ),
            "run_palette_id": getattr(args, "run_palette_id", None),
            "run_palette_seed": getattr(args, "run_palette_seed", None),
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
            "position_writeback_initial_readback": None,
            "position_writeback_enabled_readback": False,
            "position_writeback_restored_readback": None,
            "velocity_writeback_initial_readback": None,
            "velocity_writeback_enabled_readback": False,
            "velocity_writeback_restored_readback": None,
            "fabric_initial_readback": None,
            "fabric_disabled_readback": False,
            "fabric_restored_readback": None,
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
            "global_blend_rule": plan["fallback_render_model"][
                "global_blend_rule"
            ],
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
            "stage_surface_sync_records": [],
            "stage_isosurface_lifecycle_refresh_records": [],
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
                PERSISTENT_RENDER_STREAM_PATH
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
            "pathtracing_total_spp_requested": args.pathtracing_total_spp,
            "pathtracing_total_spp_initial": None,
            "pathtracing_total_spp_readback": None,
            "pathtracing_spp_requested": args.pathtracing_total_spp,
            "pathtracing_spp_initial": None,
            "pathtracing_spp_readback": None,
            "pathtracing_sampling_documented_source": (
                PATHTRACING_SUBFRAME_DOCUMENTATION_URL
            ),
            "video_renderer": (
                "RaytracedLighting"
                if getattr(args, "realtime_video", False)
                else args.renderer
            ),
            "stage_still_renderer": "PathTracing",
            "mode_setting_path": RTX_RENDER_MODE_SETTING,
            "mode_transitions": [],
            "tone_exposure_state": None,
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
            "video_playback_fps": args.video_fps,
            "video_timing_algorithm": None,
            "video_encoded_frame_time_rule": None,
            "video_timing_source_frame_steps": [],
            "video_source_capture_onset_frame_indices": [],
            "video_source_capture_onset_times_s": [],
            "video_source_capture_onset_errors_s": [],
            "video_maximum_absolute_onset_error_s": None,
            "video_frame_repeat_counts": [],
            "video_frame_source_capture_indices": [],
            "video_frame_rgb_sha256": [],
            "video_distinct_rgb_sha256_count": 0,
            "video_source_duration_s": None,
            "video_playback_duration_s": None,
            "video_duration_error_s": None,
            "stage_frames_are_video_subsets": False,
            "stream_only_frame_count": 0,
            "vial_proxy_frame_count": 0,
            "vial_proxy_transition_step": None,
            "liquid_representation_monotonic": False,
            "max_capture_tries": args.max_capture_tries,
            "capture_attempt_count": 0,
            "paired_render_pass_count": 0,
            "capture_attempts": [],
            "capture_on_play": False,
            "wait_for_render": True,
            "same_state_delta_time": 0.0,
            "async_throttling_enabled": None,
            "rejected_rgb_diagnostics": [],
            "rejected_semantic_diagnostics": [],
            "render_product_update_gating": {
                "mode": "persistent_enabled_direct_physics",
                "documented_source": RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
                "physics_step_documented_source": (
                    SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
                ),
                "initial_enabled_readback": None,
                "final_enabled_readback": None,
                "current_updates_enabled": None,
                "persistent_enabled": True,
                "rotation_enabled": False,
                "rotation_disproof": {
                    "run_id": "C6",
                    "status": "failed",
                    "failure": (
                        "render-product destruction did not release GPU memory"
                    ),
                    "receipt_sha256": C6_ROTATION_FAILURE_RECEIPT_SHA256,
                },
                "physics_steps_while_enabled": 0,
                "physics_loop_app_update_count": 0,
                "update_toggle_request_count": 0,
                "render_product_create_count": 0,
                "explicit_render_pass_count": 0,
                "renderer_accumulation_reset_count": 0,
                "explicit_render_passes": [],
            },
            "phase_timings_s": {
                phase: {"count": 0, "total": 0.0, "maximum": 0.0}
                for phase in RENDER_TIMING_PHASES
            },
        },
        "memory": {
            "hard_limit_mib": GPU_MEMORY_LIMIT_MIB,
            "peak_used_mib": None,
            "hard_limit_exceeded": False,
            "sample_source": None,
            "sample_scope": None,
            "stage_growth_stop_mib": STAGE_GPU_MEMORY_GROWTH_STOP_MIB,
            "stage_growth_guard": {
                "applicable": not count_driven_twin_liquid_mesh,
                "mode": (
                    "absolute_hard_limit_only"
                    if count_driven_twin_liquid_mesh
                    else "legacy_per_stage_delta"
                ),
                "legacy_threshold_mib": STAGE_GPU_MEMORY_GROWTH_STOP_MIB,
                "reason": (
                    "count_driven_mesh_has_no_isosurface_sparse_grid"
                    if count_driven_twin_liquid_mesh
                    else "legacy_non_count_driven_render_path"
                ),
            },
            "stage_growth_events": [],
            "samples_mib": [],
            "periodic_capture_interval_frames": (
                PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES
            ),
            "periodic_capture_samples": [],
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
            "simulation_owner": "SimulationManager.step",
            "physics_step_method": (
                "SimulationManager.step(steps=1, update_fabric=False)"
            ),
            "physics_step_documented_source": (
                SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
            ),
            "callbacks_expected": plan["timing"]["primary_final_capture_step"],
            "callbacks_observed": 0,
            "callback_indices": [],
            "callback_dts_s": [],
            "callback_times_s": [],
            "callback_direct_step_indices": [],
            "callback_logical_times_s": [],
            "timeline_initialization": None,
            "callback_contract_failure": None,
            "one_callback_per_direct_step": False,
            "physics_loop_app_update_count": 0,
            "direct_step_probe": {
                "step_limit": DIRECT_STEP_PROBE_STEPS,
                "first_capture_step": DIRECT_STEP_PROBE_STEPS,
                "completed_steps": 0,
                "emission_events_observed": 0,
                "physics_contract_passed": False,
                "capture_contract_passed": False,
                "status": "pending",
                "step_evidence": [],
            },
            "emissions_authored_on_main_thread": False,
            "emission_boundary_verified": False,
            "emission_boundaries": [],
            "points_session_layer_only": False,
            "velocities_session_layer_only": False,
            "widths_session_layer_only": False,
            "ids_session_layer_only": False,
            "positions_session_layer_only": False,
            "orientations_session_layer_only": False,
            "scales_session_layer_only": False,
            "proto_indices_session_layer_only": False,
            "simulation_points_root_layer_absent": False,
            "simulation_points_runtime_readback_passed": False,
            "capture_callbacks_advanced": None,
            "capture_direct_physics_steps_advanced": None,
            "capture_simulation_time_advanced": None,
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
            "wall_limit_seconds": args.wall_limit_seconds,
        },
        "mixing_assumption": plan["mixing_assumption"],
        "claims": dict(plan["claims"]),
        "errors": [],
        "warnings": [],
    }
    tank_colour_resolution = getattr(args, "_tank_colour_resolution", None)
    if tank_colour_resolution is None:
        execution_contract = _execution_schedule_contract(args)
        if execution_contract is not None:
            tank_colour_resolution = execution_contract.get(
                "tank_colour_resolution"
            )
    if tank_colour_resolution is not None:
        resolved = validate_tank_colour_resolution(tank_colour_resolution)
        receipt["tank_colour_resolution"] = resolved
        receipt["arguments"].update(
            {
                "tank_colours": resolved["preset_name"],
                "t100": deepcopy(resolved["transmission_overrides"].get("t100")),
                "t200": deepcopy(resolved["transmission_overrides"].get("t200")),
                "t400": deepcopy(resolved["transmission_overrides"].get("t400")),
                "dye_strength_scale": resolved["dye_strength_scale"],
                "t100_clear_lock": resolved["t100_clear_lock"],
                "randomize_around_preset": resolved["randomize_around_preset"],
                "tank_colour_seed": resolved["randomization_seed"],
                "tank_colour_label": resolved["free_text_label"],
            }
        )
    return receipt


def _empty_paint_ball_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Build the emitter receipt while retaining private legacy work buffers."""

    receipt = _empty_receipt(args)
    runtime_plan = _runtime_paint_ball_plan(args)
    runtime_capture_schedule = build_runtime_capture_schedule(
        runtime_plan, capture_stride_steps=args.capture_stride
    )
    required_particle_capacity = max(
        int(RUNTIME_PARTICLE_CONFIGURATION["max_particles"]),
        int(runtime_plan["totals"]["particle_count"]),
    )
    execution_contract = _execution_schedule_contract(args)
    qualification_mode = bool(getattr(args, "qualification_mode", False))
    qualification_target_step = (
        int(runtime_plan["paint_ball_emitter"]["qualification_first_stage_step"])
        if qualification_mode
        else None
    )
    physics_substeps = int(getattr(args, "physics_substeps", 1))
    clock_report_interval_count = (
        qualification_target_step
        if qualification_target_step is not None
        else (
            int(runtime_capture_schedule["final_plus_3_0_step"])
            if physics_substeps == 2
            else int(runtime_plan["timing"]["primary_final_capture_step"])
        )
    )
    clock_contract = _build_physics_clock_contract(
        physics_substeps=physics_substeps,
        report_interval_count=clock_report_interval_count,
    )
    receipt.update(
        {
            "schema_version": PAINT_BALL_RECEIPT_SCHEMA_VERSION,
            "experiment": "physx_three_pour_f2_paint_ball_emitter",
            "render_path": (
                COUNT_DRIVEN_RENDER_PATH
                if getattr(args, "count_driven_twin_liquid_mesh", False)
                else EMITTER_RENDER_PATH
            ),
            "render_limitations": {
                "continuous_surface": False,
                "palette_uncalibrated": True,
                "pbd_not_navier_stokes": True,
            },
            "captures": [],
            "point_instancer": {
                "path": f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared",
                "prototype_count": len(COMPOSITION_COMPONENTS),
                "prototype_specs": [],
                "max_particles": int(
                    required_particle_capacity
                ),
                "semantic_label": "liquid",
                "ids_stable": False,
                "instance_segmentation_diagnostic": None,
            },
            "containment": {
                "vial_interior_bounds_m": None,
                "capture_records": [],
                "leakage_detected": False,
            },
            "source_emission_geometry": {
                "profile_id": getattr(
                    args,
                    "particle_resolution_profile",
                    LEGACY_RESOLUTION_PROFILE_ID,
                ),
                "events": [],
                "expected_event_count": sum(
                    len(pour["emission_schedule"])
                    for pour in runtime_plan["pours"]
                ),
                "observed_event_count": 0,
                "all_support_gates_passed": None,
                "all_centroid_gates_passed": None,
                "all_separation_gates_passed": None,
                "all_flux_gates_passed": None,
                "gate_passed": None,
            },
            "qualification": {
                "first_stage_only": qualification_mode,
                "target_step": qualification_target_step,
                "status": "not_run",
                "semantic_blocker": None,
            },
            "plan": runtime_plan,
            "runtime_capture_schedule": runtime_capture_schedule,
        }
    )
    receipt["arguments"].update(
        {
            "solver_position_iterations": _requested_solver_position_iterations(
                args
            ),
            "recipe8_residual_volume_ml": getattr(
                args, "recipe8_residual_volume_ml", None
            ),
            "recipe8_effective_particle_volume_ml": getattr(
                args, "recipe8_effective_particle_volume_ml", None
            ),
            "physics_substeps": physics_substeps,
            "particle_resolution_profile": getattr(
                args,
                "particle_resolution_profile",
                LEGACY_RESOLUTION_PROFILE_ID,
            ),
        }
    )
    runtime_run_palette = runtime_plan.get("run_palette")
    receipt["material"].update(
        {
            "run_palette": deepcopy(runtime_run_palette),
            "tank_transmission_colors": (
                deepcopy(runtime_run_palette["transmission_1cm"])
                if isinstance(runtime_run_palette, Mapping)
                else {
                    tank: list(color)
                    for tank, color in NOMINAL_TANK_TRANSMISSION_COLORS.items()
                }
            ),
            "stream_scattering_depth_m": 2.0 * RENDER_STREAM_RADIUS_M,
            "vial_scattering_depth_m": O1B_MATERIAL_PARAMETERS[
                "specular_transmission_scattering_depth_m"
            ],
            "stream_uses_dedicated_path_depth": True,
            "spatial_body_mdl_path": (
                os.fspath(SPATIAL_OMNISURFACE_MDL)
                if getattr(args, "height_banded_body", False)
                else None
            ),
            "spatial_body_mdl_sha256": (
                _sha256(SPATIAL_OMNISURFACE_MDL)
                if getattr(args, "height_banded_body", False)
                else None
            ),
        }
    )
    if getattr(args, "height_banded_body", False):
        receipt["global_blend"].update(
            {
                "composition_state": "tracked_pbd_state_height_banded_mesh_color",
                "spatial_mixing_color_rendered": True,
                "spatial_layering_color_rendered": True,
                "height_band_count": HEIGHT_BAND_COUNT,
            }
        )
        receipt["composition_evidence"]["internal_unrendered"] = False
        receipt["claims"].update(
            {
                "particle_composition_rendered": True,
                "spatial_mixing_color_rendered": True,
                "spatial_layering_color_rendered": True,
                "global_vial_blend_rendered": False,
            }
        )
    resolution_profile_id = receipt["arguments"]["particle_resolution_profile"]
    receipt["particle_resolution_profile"] = deepcopy(
        runtime_plan.get(
            "particle_resolution_profile",
            particle_resolution_profile(resolution_profile_id),
        )
    )
    if physics_substeps == 2:
        if (
            execution_contract is not None
            and execution_contract.get("contract_kind")
            in TRAIN_RESIDUAL_CONTRACT_KINDS
        ):
            receipt["ablation"] = {
                "id": "accepted_two_substep_train_case",
                "qualification_only": False,
                "training_or_cohort_use_authorized": True,
                "changed_variable": "physics_substeps_per_report_interval",
                "accepted_qualification_receipt_sha256": (
                    execution_contract["qualification"]["receipt_sha256"]
                ),
            }
        else:
            if resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
                receipt["ablation"] = {
                    "id": "c21_2p0mm_resolution_and_source_boundary",
                    "qualification_only": True,
                    "training_or_cohort_use_authorized": False,
                    "parent_qualification": "c20_two_physx_substeps_per_report_interval",
                    "changed_variable": "particle_resolution_profile",
                    "authorized_source_sampler_correction": True,
                    "literal_one_variable_claim": False,
                }
            else:
                receipt["ablation"] = {
                    "id": "c20_two_physx_substeps_per_report_interval",
                    "qualification_only": True,
                    "training_or_cohort_use_authorized": False,
                    "changed_variable": "physics_substeps_per_report_interval",
                }
    if "recipe8_leading_residual" in runtime_plan:
        receipt["recipe8_leading_residual"] = deepcopy(
            runtime_plan["recipe8_leading_residual"]
        )
        receipt["point_instancer"]["particle_capacity"] = {
            "required_max_particles": required_particle_capacity,
            "required_value_source": "runtime_plan.totals.particle_count",
            "authored_max_particles": None,
            "readback_max_particles": None,
            "capacity_gate_passed": False,
        }
        receipt["totals"].update(
            {
                "programmed_volume_ml": runtime_plan["totals"][
                    "programmed_volume_ml"
                ],
                "nominal_mass_convention_equivalent_ml": runtime_plan[
                    "totals"
                ]["nominal_mass_convention_equivalent_ml"],
                "packing_equivalent_volume_ml": runtime_plan["totals"][
                    "packing_equivalent_volume_ml"
                ],
                "nominal_error_ml": runtime_plan["totals"]["nominal_error_ml"],
            }
        )
    nuisance_binding = getattr(args, "_nuisance_receipt_binding", None)
    if nuisance_binding is not None:
        receipt["arguments"].update(
            {
                "nuisance_receipt": str(args.nuisance_receipt),
                "nuisance_receipt_sha256": args.nuisance_receipt_sha256,
                "nuisance_case_id": args.nuisance_case_id,
            }
        )
        receipt["nuisance_profile_binding"] = deepcopy(nuisance_binding)
    if execution_contract is not None:
        if execution_contract.get("contract_kind") in TRAIN_RESIDUAL_CONTRACT_KINDS:
            receipt["arguments"].update(
                {
                    "train_render_plan": os.fspath(args.train_render_plan),
                    "train_render_plan_sha256": args.train_render_plan_sha256,
                    "train_case_id": args.train_case_id,
                    "accepted_qualification_receipt": os.fspath(
                        args.accepted_qualification_receipt
                    ),
                    "accepted_qualification_receipt_sha256": (
                        args.accepted_qualification_receipt_sha256
                    ),
                }
            )
        else:
            receipt["arguments"].update(
                {
                    "execution_schedule": os.fspath(args.execution_schedule),
                    "execution_schedule_sha256": args.execution_schedule_sha256,
                }
            )
        receipt["point_instancer"].update(
            {
                "max_particles": int(
                    execution_contract["required_max_particles"]
                ),
                "particle_capacity": {
                    "required_max_particles": int(
                        execution_contract["required_max_particles"]
                    ),
                    "required_value_source": (
                        "validated_schedule.totals.programmed_particle_count"
                        if execution_contract.get("contract_kind")
                        in TRAIN_RESIDUAL_CONTRACT_KINDS
                        else "validated_schedule.totals.maximum_particle_count"
                    ),
                    "authored_max_particles": None,
                    "readback_max_particles": None,
                    "capacity_gate_passed": False,
                },
            }
        )
        receipt["execution_schedule"] = deepcopy(
            runtime_plan["execution_schedule"]
        )
        receipt["execution_schedule"].update(
            {
                "particle_capacity": deepcopy(
                    receipt["point_instancer"]["particle_capacity"]
                ),
                "particle_ids_and_counts_match_every_event": False,
                "event_compositions_match": False,
                "executed_emission_event_count": 0,
                "event_composition_attestation_path": None,
                "event_composition_attestation_sha256": None,
                "render_only_stream_color_policy": (
                    runtime_plan["execution_schedule"].get(
                        "stream_color_policy",
                        "current_tank_pure_source_material",
                    )
                ),
                "all_three_programmed_volume_numeric_gates_passed": False,
            }
        )
        if execution_contract.get("contract_kind") in TRAIN_RESIDUAL_CONTRACT_KINDS:
            receipt["totals"].update(
                {
                    "programmed_volume_ml": runtime_plan["totals"][
                        "programmed_volume_ml"
                    ],
                    "nominal_mass_convention_equivalent_ml": runtime_plan[
                        "totals"
                    ]["nominal_mass_convention_equivalent_ml"],
                    "packing_equivalent_volume_ml": runtime_plan["totals"][
                        "packing_equivalent_volume_ml"
                    ],
                    "nominal_error_ml": runtime_plan["totals"][
                        "nominal_error_ml"
                    ],
                }
            )
    receipt["claims"] = dict(runtime_plan["claims"])
    receipt["global_blend"]["model"] = runtime_plan["mixing_model"]
    receipt["global_blend"]["global_blend_rule"] = runtime_plan["fallback_render_model"]["global_blend_rule"]
    receipt["timing"].update(
        {
            **runtime_plan["timing"],
            "report_hz": clock_contract["report_hz"],
            "report_dt_s": clock_contract["report_dt_s"],
            "report_interval_count": clock_contract["report_interval_count"],
            # Compatibility key: this is now explicitly the PhysX rate.  All
            # schedule consumers use report_hz/runtime_capture_schedule.
            "physics_hz": clock_contract["physics_steps_per_second"],
            "physics_steps_per_second": clock_contract[
                "physics_steps_per_second"
            ],
            "physics_substeps_per_report_interval": physics_substeps,
            "physics_dt_s": clock_contract["physics_dt_s"],
            "physics_callback_count": clock_contract[
                "physics_callback_count"
            ],
            "min_frame_rate": clock_contract["min_frame_rate"],
            "timeline_ticks_per_frame": clock_contract[
                "timeline_ticks_per_frame"
            ],
            "timeline_end_s": clock_contract["timeline_end_s"],
            "physical_duration_s": clock_contract["physical_duration_s"],
            "executed_report_intervals": 0,
            "diffusion_updates_per_report_interval": 1,
            "diffusion_update_count": 0,
            "diffusion_dt_s": clock_contract["diffusion_dt_s"],
            "diffusion_integrated_time_s": 0.0,
            "final_plus_0_3_step": runtime_capture_schedule[
                "final_plus_0_3_step"
            ],
            "final_plus_3_0_step": runtime_capture_schedule[
                "final_plus_3_0_step"
            ],
            "continued_physics_and_diffusion_through_step": (
                runtime_capture_schedule["final_plus_3_0_step"]
            ),
            "pour_duration_steps": [
                int(pour["duration_steps"]) for pour in runtime_plan["pours"]
            ],
            "pour_end_steps": [
                int(pour["end_step"]) for pour in runtime_plan["pours"]
            ],
            "callbacks_expected": clock_contract["physics_callback_count"],
            "callback_dt_sum_s": 0.0,
            "callbacks_per_app_update_contract_passed": False,
            "callback_group_trace_path": None,
            "callback_group_trace_sha256": None,
            "callback_group_trace_count": 0,
            "physics_manager_readback": None,
            "physics_settings_readback": None,
        }
    )
    receipt["timing"]["capture_hz"] = runtime_capture_schedule["capture_hz"]
    receipt["particle_system"][
        "runtime_particle_ids_exact_each_report_interval"
    ] = False
    receipt["particle_system"].update(
        {
            "runtime_configuration": build_runtime_particle_configuration(
                solver_position_iterations=(
                    _requested_solver_position_iterations(args)
                ),
                max_particles=required_particle_capacity,
                resolution_profile_id=resolution_profile_id,
            ),
            "max_velocity_m_s": PAINT_BALL_MAX_VELOCITY_M_S,
            "max_velocity_source": "ParticlePostProcessingDemo.py:250.0 at metersPerUnit=0.01",
            "smoothing": True,
            "smoothing_enabled": True,
            "smoothing_strength": 1.0,
            "anisotropy": True,
            "anisotropy_enabled": True,
            "anisotropy_scale": 1.0,
            "ccd": False,
            "fabric_enabled": None,
        }
    )
    if _water_pbd_preset_requested(args):
        receipt["particle_system"]["pbd_material_treatment"] = {
            "requested": True,
            "applied": False,
            "requested_via": _water_pbd_preset_requested_via(args),
            "plan": deepcopy(runtime_plan["pbd_material_treatment"]),
        }
    receipt["composition_evidence"]["diffusion"]["dt_s"] = 1.0 / 120.0
    # The current emitter receipt deliberately has no retired isosurface or
    # canonical mesh lifecycle fields.  Keep the older private scaffold only
    # for compatibility with historical CPU helpers, never in persisted F2.
    receipt.pop("isosurface", None)
    receipt.pop("mesh_proxy", None)
    receipt["material"].update(
        {
            "vial_representation": "point_instancer_sphere_emitter",
            "stream_representation": "render_only_mesh_current_tank_material",
            "point_instancer_global_material_binding": False,
        }
    )
    receipt["material"].pop("isosurface_global_material_binding", None)
    receipt["global_blend"].update(
        {
            "global_blend_rule": "diagnostic_global_conserved_blend_not_rendered_as_vial_shader",
            "composition_state": "palette_quantized_per_particle_rendered",
            "spatial_mixing_color_rendered": True,
            "spatial_layering_color_rendered": True,
        }
    )
    receipt["composition_evidence"].update(
        {
            "internal_unrendered": False,
            "rendered_representation": "nearest_frozen_palette_point_instancer_proto_indices",
            "spatial_mixing_rendered": True,
        }
    )
    receipt["semantics"].update(
        {
            "mask_source": "exact_renderer_semantic_class_ids",
            "generated_isosurface_path": None,
            "isosurface_path": None,
            "required_isosurface_path": None,
            "isosurface_direct_binding_authored": False,
            "isosurface_direct_binding_step": None,
            "isosurface_direct_binding": None,
            "isosurface_semantic_bound": False,
            "transparent_annotation_override": {
                "prim_path": TRANSPARENT_ANNOTATION_OVERRIDE.prim_path,
                "parent_path": TRANSPARENT_ANNOTATION_OVERRIDE.parent_path,
                "attribute_name": TRANSPARENT_ANNOTATION_OVERRIDE.attribute_name,
                "semantic_pass_value": TRANSPARENT_ANNOTATION_OVERRIDE.authored_value,
                "rgb_and_restore_value": False,
                "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                "documentation_url": TRANSPARENT_ANNOTATION_DOCUMENTATION_URL,
                "source": None,
                "initial_rgb_state": None,
                "transitions": [],
                "property_stack_strongest_is_session_layer": False,
                "authored_after_untouched_reference_capture": False,
                "source_layer_unchanged_expected": True,
            },
            "paired_capture_contract": {
                "shared_single_render_pass": False,
                "paired_same_physics_state": True,
                "kept_rgb_uses_source_equivalent_glass": True,
                "semantic_pass_uses_transparent_annotation_flag": True,
                "flag_restored_false_after_semantic_pass": True,
                "state_fields_required_unchanged": list(PAIRED_CAPTURE_STATE_FIELDS),
                "state_records": [],
            },
        }
    )
    if getattr(args, "count_driven_twin_liquid_mesh", False):
        height_banded = bool(getattr(args, "height_banded_body", False))
        receipt["render_limitations"] = dict(COUNT_DRIVEN_RENDER_LIMITATIONS)
        receipt["material"].update(
            {
                "vial_representation": COUNT_DRIVEN_RENDER_PATH,
                "point_instancer_global_material_binding": False,
            }
        )
        receipt["global_blend"].update(
            {
                "global_blend_rule": (
                    "height_banded_tracked_particle_beer_law_transmission_uncalibrated"
                    if height_banded
                    else "contained_particle_mean_beer_law_transmission_uncalibrated"
                ),
                "composition_state": (
                    "tracked_pbd_state_height_banded_mesh_color"
                    if height_banded
                    else "tracked_pbd_state_global_mesh_blend"
                ),
                "spatial_mixing_color_rendered": height_banded,
                "spatial_layering_color_rendered": height_banded,
                "height_band_count": HEIGHT_BAND_COUNT if height_banded else None,
            }
        )
        receipt["composition_evidence"].update(
            {
                "internal_unrendered": not height_banded,
                "rendered_representation": (
                    "height_banded_count_driven_twin_liquid_mesh"
                    if height_banded
                    else "global_uncalibrated_count_driven_twin_liquid_mesh"
                ),
                "spatial_mixing_rendered": height_banded,
            }
        )
        receipt["semantics"]["paired_capture_contract"][
            "state_fields_required_unchanged"
        ] = [
            *PAIRED_CAPTURE_STATE_FIELDS,
            "compositions",
            "body_points",
            "body_extent",
            "body_visibility",
            "body_material_color",
            "stream_visibility",
        ]
        receipt["semantics"].update(
            {
                "count_driven_body_semantic_bound": False,
                "point_instancer_rgb_visible": False,
                "point_instancer_semantic_visible": False,
                "hidden_particle_control": None,
            }
        )
        receipt["count_driven_liquid_body"] = {
            "representation": COUNT_DRIVEN_RENDER_PATH,
            "source_xform_path": SOURCE_LIQUID_PATH,
            "source_mesh_path": SOURCE_LIQUID_MESH_PATH,
            "body_xform_path": COUNT_DRIVEN_BODY_XFORM_PATH,
            "body_mesh_path": COUNT_DRIVEN_BODY_MESH_PATH,
            "requested_source_time_code": COUNT_DRIVEN_SOURCE_TIME_CODE,
            "resolved_points_time_code": None,
            "resolved_normals_time_code": None,
            "resolved_extent_time_code": None,
            "copied_primvar_names": [],
            "source_world_bottom_z_m": COUNT_DRIVEN_SOURCE_WORLD_BOTTOM_Z_M,
            "particle_volume_m3": runtime_plan["render_treatment"][
                "particle_volume_m3"
            ],
            "cross_section_area_m2": COUNT_DRIVEN_DODECAGON_AREA_M2,
            "body_starts_hidden_at_zero_count": True,
            "sphere_instances_hidden": True,
            "physics_point_instancer_preserved": True,
            "pbd_state_independent": True,
            "global_uncalibrated_blend": not height_banded,
            "spatial_mixing_color_rendered": height_banded,
            "spatial_layering_color_rendered": height_banded,
            "normals_policy": "omit_authored_normals_after_anisotropic_scale",
            "extent_recomputed_each_update": True,
            "geometry_authoring_numeric_path": (
                "local_points_vec3f_then_inherited_world_transform"
            ),
            "post_transform_volume_residual_abs_tolerance_m3": 2.0e-10,
            "material_path": (
                f"{FLUIDMIX_NAMESPACE}/SpatialLiquidMaterial"
                if height_banded
                else f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial"
            ),
            "semantic_label": "liquid",
            "stage_geometry": [],
            "capture_geometry": [],
            "training_eligible": False,
        }
        if height_banded:
            receipt["count_driven_liquid_body"].update(
                {
                    "height_band_count": HEIGHT_BAND_COUNT,
                    "stage_band_evidence": [],
                }
            )
        # The count-driven qualification/cohort path owns physics through the
        # playing Kit timeline.  NVIDIA's Replicator example documents that an
        # app update advances the timeline and implicitly the simulation; this
        # is also the owner exercised by the repository's successful F2 runs.
        # Keep the one render product disabled between explicit captures so a
        # 120 Hz physics update is not also an RTX render tick.
        receipt["render"]["render_product_update_gating"] = {
            "mode": "timeline_app_update_explicit_capture",
            "documented_source": RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
            "physics_step_documented_source": (
                TIMELINE_APP_UPDATE_DOCUMENTATION_URL
            ),
            "initial_disabled_readback": None,
            "final_disabled_readback": None,
            "current_updates_enabled": None,
            "enabled_only_for_explicit_render_passes": True,
            "rotation_enabled": False,
            "rotation_disproof": {
                "run_id": "C6",
                "status": "failed",
                "failure": (
                    "render-product destruction did not release GPU memory"
                ),
                "receipt_sha256": C6_ROTATION_FAILURE_RECEIPT_SHA256,
            },
            "physics_updates_while_disabled": 0,
            "physics_loop_app_update_count": 0,
            "render_product_create_count": 0,
            "enable_request_count": 0,
            "disable_request_count": 0,
            "explicit_render_pass_count": 0,
            "renderer_accumulation_reset_count": 0,
            "transitions": [],
            "explicit_render_passes": [],
        }
        receipt["render"]["phase_timings_s"] = {
            phase: {"count": 0, "total": 0.0, "maximum": 0.0}
            for phase in TIMELINE_RENDER_TIMING_PHASES
        }
        receipt["timing"].update(
            {
                "simulation_owner": "timeline_app_update",
                "physics_step_method": "app.update()",
                "physics_step_documented_source": (
                    TIMELINE_APP_UPDATE_DOCUMENTATION_URL
                ),
                "callback_app_update_indices": [],
                "one_callback_per_app_update": False,
                "physics_loop_app_update_count": 0,
                "motion_falsifier": {
                    "probe_step": MOTION_FALSIFIER_STEP,
                    "status": "pending",
                    "passed": False,
                    "artifact_path": None,
                    "artifact_sha256": None,
                },
            }
        )
    receipt["qualification"]["first_vial_bottom_roi"] = {
        "xyxy_exclusive": list(FIRST_VIAL_BOTTOM_ROI_XYXY_EXCLUSIVE),
        "resolution": [640, 480],
        "minimum_liquid_pixels": FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS,
        "definition": "frozen_fillcam_first_vial_interior_bottom",
    }
    return receipt


def _finalize_paint_ball_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Remove retired render-path buffers before receipt persistence."""

    receipt.pop("isosurface", None)
    receipt.pop("mesh_proxy", None)
    receipt["captures"] = list(receipt.get("point_instancer", {}).get("capture_records", []))
    return receipt


def _persist_completed_capture_artifacts(
    receipt: dict[str, Any],
    *,
    output: Path,
    width: int,
    height: int,
    frame_paths: list[Path],
    frame_steps: list[int],
    semantic_paths: list[Path],
    liquid_mask_paths: list[Path],
    liquid_pixel_counts: list[int],
    semantic_mask_records: list[dict[str, Any]],
    stage_paths: list[Path],
    stage_frame_indices: list[int],
    video_path: Path | None,
    video_codec: str | None,
    video_writer_fallback_used: bool | None,
) -> None:
    """Hash-bind the completed prefix of a failed run without repairing it.

    A capture pair is complete only after its RGB, semantic array, liquid mask,
    pixel count, and semantic metadata have all been appended.  The function
    intentionally records only that common prefix.  It leaves the run status,
    original exception and status untouched; partial evidence cannot turn a
    failed run into a pass.  An unreadable referenced artifact is appended as
    an additional real failure.
    """

    artifact_failures: list[dict[str, str]] = []

    def describe(path: Path, *, kind: str) -> tuple[str | None, str | None]:
        try:
            relative = str(path.relative_to(output))
        except ValueError:
            artifact_failures.append(
                {
                    "kind": kind,
                    "path": str(path),
                    "failure": "path_outside_output",
                }
            )
            return None, None
        try:
            digest = _sha256(path)
        except OSError as exc:
            artifact_failures.append(
                {
                    "kind": kind,
                    "path": relative,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            return relative, None
        return relative, digest

    capture_inputs: dict[str, int] = {
        "rgb": len(frame_paths),
        "physics_steps": len(frame_steps),
        "semantic": len(semantic_paths),
        "liquid_masks": len(liquid_mask_paths),
        "liquid_pixel_counts": len(liquid_pixel_counts),
        "semantic_records": len(semantic_mask_records),
    }
    completed_pair_count = min(capture_inputs.values(), default=0)
    capture_arrays_aligned = len(set(capture_inputs.values())) <= 1

    rgb_relpaths: list[str | None] = []
    rgb_hashes: list[str | None] = []
    completed_render_count = min(len(frame_paths), len(frame_steps))
    for index in range(completed_render_count):
        rgb_path, rgb_sha256 = describe(frame_paths[index], kind="rgb")
        rgb_relpaths.append(rgb_path)
        rgb_hashes.append(rgb_sha256)

    semantic_relpaths: list[str | None] = []
    semantic_hashes: list[str | None] = []
    mask_relpaths: list[str | None] = []
    mask_hashes: list[str | None] = []
    semantic_pairs: list[dict[str, Any]] = []
    for index in range(completed_pair_count):
        semantic_path, semantic_sha256 = describe(
            semantic_paths[index], kind="semantic"
        )
        mask_path, mask_sha256 = describe(
            liquid_mask_paths[index], kind="liquid_mask"
        )
        semantic_relpaths.append(semantic_path)
        semantic_hashes.append(semantic_sha256)
        mask_relpaths.append(mask_path)
        mask_hashes.append(mask_sha256)
        semantic_pairs.append(
            {
                "physics_step": int(frame_steps[index]),
                "rgb_path": rgb_relpaths[index],
                "rgb_sha256": rgb_hashes[index],
                "mask_path": mask_path,
                "mask_sha256": mask_sha256,
                "semantic_path": semantic_path,
                "semantic_sha256": semantic_sha256,
                **deepcopy(semantic_mask_records[index]),
                "mask_readable": mask_sha256 is not None,
                "mask_resolution": [width, height],
                "liquid_pixel_count": int(liquid_pixel_counts[index]),
            }
        )

    stage_inputs = {
        "stage_paths": len(stage_paths),
        "stage_frame_indices": len(stage_frame_indices),
    }
    completed_stage_count = min(stage_inputs.values(), default=0)
    stage_arrays_aligned = len(set(stage_inputs.values())) <= 1
    stage_relpaths: list[str | None] = []
    stage_hashes: list[str | None] = []
    completed_stage_indices: list[int] = []
    stage_steps: list[int] = []
    stage_mask_hashes: list[str | None] = []
    for stage_list_index in range(completed_stage_count):
        frame_index = int(stage_frame_indices[stage_list_index])
        if frame_index < 0 or frame_index >= completed_pair_count:
            artifact_failures.append(
                {
                    "kind": "stage_frame",
                    "path": str(stage_paths[stage_list_index]),
                    "failure": "stage_frame_index_outside_completed_capture_prefix",
                }
            )
            continue
        stage_path, stage_sha256 = describe(
            stage_paths[stage_list_index], kind="stage_frame"
        )
        stage_relpaths.append(stage_path)
        stage_hashes.append(stage_sha256)
        completed_stage_indices.append(frame_index)
        stage_steps.append(int(frame_steps[frame_index]))
        stage_mask_hashes.append(mask_hashes[frame_index])

    render = receipt.setdefault("render", {})
    render.update(
        {
            "frame_count": completed_render_count,
            "frame_paths": rgb_relpaths,
            "frame_sha256": rgb_hashes,
            "frame_physics_steps": [
                int(value) for value in frame_steps[:completed_render_count]
            ],
            "stage_frame_paths": stage_relpaths,
            "stage_frame_sha256": stage_hashes,
            "stage_frame_indices": completed_stage_indices,
            "stage_frame_physics_steps": stage_steps,
            "primary_final_frame_sha256": None,
            "stage_frames_are_video_subsets": bool(
                stage_hashes
                and all(
                    value is not None and value in rgb_hashes
                    for value in stage_hashes
                )
            ),
        }
    )
    if video_path is not None:
        relative_video_path, video_sha256 = describe(
            video_path, kind="partial_video"
        )
        render.update(
            {
                "video_path": relative_video_path,
                "video_sha256": video_sha256,
                "video_codec": video_codec,
                "video_container": video_path.suffix.removeprefix("."),
                "video_writer_fallback_used": video_writer_fallback_used,
                "video_partial": True,
            }
        )

    semantics = receipt.setdefault("semantics", {})
    semantics.update(
        {
            "pairs": semantic_pairs,
            "stage_mask_sha256": stage_mask_hashes,
            "frame_semantic_paths": semantic_relpaths,
            "frame_semantic_sha256": semantic_hashes,
        }
    )
    render["artifact_finalization"] = {
        "mode": "partial_failure",
        "completed_render_frame_count": completed_render_count,
        "completed_capture_pair_count": completed_pair_count,
        "completed_stage_frame_count": len(completed_stage_indices),
        "capture_input_counts": capture_inputs,
        "capture_arrays_aligned": capture_arrays_aligned,
        "stage_input_counts": stage_inputs,
        "stage_arrays_aligned": stage_arrays_aligned,
        "artifact_failures": artifact_failures,
    }
    for failure in artifact_failures:
        message = (
            "partial capture artifact failure: "
            f"{failure['kind']} {failure['path']}: {failure['failure']}"
        )
        if message not in receipt.setdefault("errors", []):
            receipt["errors"].append(message)


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


def _stage_memory_growth_stop_required(
    *,
    count_driven_twin_liquid_mesh: bool,
    memory_before_mib: int | None,
    memory_after_mib: int,
) -> bool:
    """Return whether the legacy per-stage GPU-memory stop should fire."""

    return bool(
        not count_driven_twin_liquid_mesh
        and memory_before_mib is not None
        and memory_after_mib - memory_before_mib
        >= STAGE_GPU_MEMORY_GROWTH_STOP_MIB
    )


def _sample_periodic_capture_memory(
    receipt: dict[str, Any],
    *,
    completed_frame_count: int,
    physics_step: int,
    memory_sample: Any = _memory_sample,
) -> dict[str, int] | None:
    """Persist one hard-gated VRAM sample every frozen frame interval."""

    if (
        isinstance(completed_frame_count, bool)
        or not isinstance(completed_frame_count, int)
        or completed_frame_count < 0
        or isinstance(physics_step, bool)
        or not isinstance(physics_step, int)
        or physics_step < 0
    ):
        raise ValueError("periodic memory coordinates must be non-negative integers")
    memory = receipt.get("memory")
    if not isinstance(memory, dict):
        raise RuntimeError("receipt memory section is missing")
    interval = memory.get("periodic_capture_interval_frames")
    rows = memory.get("periodic_capture_samples")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or not isinstance(rows, list)
        or not callable(memory_sample)
    ):
        raise RuntimeError("periodic capture memory receipt is malformed")
    if completed_frame_count == 0 or completed_frame_count % interval != 0:
        return None
    if rows and (
        not isinstance(rows[-1], dict)
        or int(rows[-1].get("completed_frame_count", -1))
        >= completed_frame_count
    ):
        raise RuntimeError("periodic capture memory samples are not monotonic")
    row = {
        "completed_frame_count": completed_frame_count,
        "physics_step": physics_step,
        "used_gpu_memory_mib": int(memory_sample(receipt)),
    }
    rows.append(row)
    return row


def _author_omnisurface(
    stage: Any,
    material: Any,
    color: tuple[float, float, float],
    Sdf: Any,
    Gf: Any,
    UsdShade: Any,
    *,
    scattering_depth_m: float | None = None,
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
            (
                O1B_MATERIAL_PARAMETERS[
                    "specular_transmission_scattering_depth_m"
                ]
                if scattering_depth_m is None
                else float(scattering_depth_m)
            ),
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


def _author_spatial_omnisurface(
    stage: Any,
    material: Any,
    fallback_color: tuple[float, float, float],
    Sdf: Any,
    Gf: Any,
    UsdShade: Any,
) -> Any:
    """Author the capability-probed displayColor-driven OmniSurface wrapper."""

    shader = UsdShade.Shader.Define(
        stage, f"{material.GetPath()}/SpatialOmniSurface"
    )
    shader.SetSourceAsset(Sdf.AssetPath(SPATIAL_OMNISURFACE_MDL.name), "mdl")
    shader.SetSourceAssetSubIdentifier(
        SPATIAL_OMNISURFACE_SUBIDENTIFIER, "mdl"
    )
    shader.CreateInput("fallback_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*fallback_color)
    )
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    material.CreateVolumeOutput("mdl").ConnectToSource(output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(output)
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


_OMNISURFACE_RENDER_INPUTS = (
    "enable_specular_transmission",
    "specular_transmission_weight",
    "specular_transmission_color",
    "specular_transmission_scattering_depth",
    "thin_walled",
    "diffuse_reflection_weight",
    "specular_reflection_ior",
    "specular_reflection_roughness",
    "enable_opacity",
    "geometry_opacity",
    "geometry_opacity_threshold",
)


def _serial_shader_input(value: Any, *, name: str) -> bool | float | list[float]:
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    try:
        components = [float(component) for component in value]
    except (TypeError, ValueError):
        components = []
    if components and all(math.isfinite(component) for component in components):
        return components
    raise RuntimeError(f"OmniSurface input {name} has no finite readback")


def _read_count_driven_body_material_state(
    body_mesh: Any,
    body_shader: Any,
    UsdShade: Any,
    UsdGeom: Any,
) -> dict[str, Any]:
    """Read the bound material path and every authored appearance input."""

    material, _relationship = UsdShade.MaterialBindingAPI(
        body_mesh.GetPrim()
    ).ComputeBoundMaterial()
    material_path = (
        str(material.GetPrim().GetPath())
        if material and material.GetPrim().IsValid()
        else None
    )
    shader_path = str(body_shader.GetPrim().GetPath())
    inputs: dict[str, Any] = {}
    shader_inputs = {
        str(shader_input.GetBaseName()): shader_input
        for shader_input in body_shader.GetInputs()
    }
    for name, shader_input in sorted(shader_inputs.items()):
        value = shader_input.Get()
        inputs[name] = _serial_shader_input(value, name=name)
    primvar = UsdGeom.PrimvarsAPI(body_mesh).GetPrimvar("displayColor")
    primvar_values = primvar.Get() if primvar and primvar.HasValue() else None
    display_color_sha256 = None
    if primvar_values is not None:
        import numpy as np

        color_rows = np.asarray(
            [tuple(float(channel) for channel in row) for row in primvar_values],
            dtype="<f4",
        )
        display_color_sha256 = hashlib.sha256(color_rows.tobytes()).hexdigest()
    state = {
        "body_mesh_path": str(body_mesh.GetPrim().GetPath()),
        "material_path": material_path,
        "shader_path": shader_path,
        "shader_inputs": inputs,
        "shader_inputs_sha256": _json_value_sha256(inputs),
        "display_color_sha256": display_color_sha256,
    }
    if material_path is None:
        raise RuntimeError("count-driven liquid body has no bound render material")
    return state


def _read_render_stream_geometry_state(
    record: dict[str, Any],
    UsdGeom: Any,
) -> dict[str, Any]:
    """Read the persistent stream geometry, transform, visibility and material."""

    cylinder = UsdGeom.Cylinder(record["prim"])
    height = cylinder.GetHeightAttr().Get()
    radius = cylinder.GetRadiusAttr().Get()
    axis = cylinder.GetAxisAttr().Get()
    translate = record["translate"].Get()
    if (
        height is None
        or radius is None
        or axis is None
        or translate is None
    ):
        raise RuntimeError("persistent stream geometry readback is missing")
    state = {
        "prim_path": str(record["prim"].GetPath()),
        "height_m": float(height),
        "radius_m": float(radius),
        "axis": str(axis),
        "translate": [float(translate[index]) for index in range(3)],
        "visibility": str(
            UsdGeom.Imageable(record["prim"]).ComputeVisibility()
        ),
        "active_source": record.get("active_source"),
        "material_path": record.get("readback_material_path"),
    }
    numeric = [state["height_m"], state["radius_m"], *state["translate"]]
    if (
        any(not math.isfinite(float(value)) for value in numeric)
        or state["height_m"] <= 0.0
        or state["radius_m"] <= 0.0
    ):
        raise RuntimeError("persistent stream geometry readback is invalid")
    return {
        **state,
        "state_sha256": _json_value_sha256(state),
    }


def _resolved_source_sample_time(
    attribute: Any, *, requested_time_code: int
) -> int | float | None:
    """Return the latest authored sample no later than the requested time."""

    samples = [
        float(value)
        for value in (attribute.GetTimeSamples() or [])
        if float(value) <= float(requested_time_code)
    ]
    if not samples:
        return None
    resolved = max(samples)
    return int(resolved) if resolved.is_integer() else resolved


def _mesh_world_points(
    mesh: Any,
    points: Any,
    *,
    time_code: int,
    Usd: Any,
    UsdGeom: Any,
    Gf: Any,
) -> Any:
    """Evaluate local mesh points in metre-scale world coordinates."""

    import numpy as np

    matrix = UsdGeom.XformCache(Usd.TimeCode(time_code)).GetLocalToWorldTransform(
        mesh.GetPrim()
    )
    return np.asarray(
        [
            tuple(float(value) for value in matrix.Transform(Gf.Vec3d(*row)))
            for row in np.asarray(points, dtype=np.float64)
        ],
        dtype=np.float64,
    )


def _evaluated_mesh_hashes(
    world_points: Any, face_counts: Any, face_indices: Any
) -> tuple[str, str]:
    """Hash evaluated geometry using the frozen A31 byte contract."""

    import numpy as np

    point_bytes = np.asarray(world_points, dtype="<f8", order="C").tobytes()
    count_bytes = np.asarray(face_counts, dtype="<i8", order="C").tobytes()
    index_bytes = np.asarray(face_indices, dtype="<i8", order="C").tobytes()
    return (
        hashlib.sha256(point_bytes + count_bytes + index_bytes).hexdigest(),
        hashlib.sha256(count_bytes + index_bytes).hexdigest(),
    )


def _author_count_driven_twin_liquid_body(
    *,
    stage: Any,
    body_material: Any,
    body_shader: Any,
    height_banded: bool,
    Usd: Any,
    UsdGeom: Any,
    UsdShade: Any,
    UsdSemantics: Any,
    LegacySemantics: Any,
    Sdf: Any,
    Gf: Any,
    Vt: Any,
    requested_source_time_code: int,
    particle_volume_m3: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the authored twin liquid mesh into a hidden session-layer body."""

    import numpy as np

    source_xform_prim = stage.GetPrimAtPath(SOURCE_LIQUID_PATH)
    source_mesh_prim = stage.GetPrimAtPath(SOURCE_LIQUID_MESH_PATH)
    if (
        not source_xform_prim.IsValid()
        or not source_mesh_prim.IsValid()
        or not source_mesh_prim.IsA(UsdGeom.Mesh)
    ):
        raise RuntimeError("reviewed authored twin liquid mesh is missing")
    source_mesh = UsdGeom.Mesh(source_mesh_prim)
    requested = int(requested_source_time_code)
    if requested != COUNT_DRIVEN_SOURCE_TIME_CODE:
        raise RuntimeError("count-driven source time code drifted")
    requested_time = Usd.TimeCode(requested)
    source_points = np.asarray(
        source_mesh.GetPointsAttr().Get(requested_time), dtype=np.float64
    )
    face_counts = [
        int(value)
        for value in (source_mesh.GetFaceVertexCountsAttr().Get(requested_time) or [])
    ]
    face_indices = [
        int(value)
        for value in (source_mesh.GetFaceVertexIndicesAttr().Get(requested_time) or [])
    ]
    if source_points.shape != (512, 3):
        raise RuntimeError("authored twin liquid source point count drifted")
    fan_triangle_count = sum(value - 2 for value in face_counts)
    if len(face_counts) != 448 or len(face_indices) != 1664 or fan_triangle_count != 768:
        raise RuntimeError("authored twin liquid topology drifted")
    world_points = _mesh_world_points(
        source_mesh,
        source_points,
        time_code=requested,
        Usd=Usd,
        UsdGeom=UsdGeom,
        Gf=Gf,
    )
    geometry_sha256, topology_sha256 = _evaluated_mesh_hashes(
        world_points, face_counts, face_indices
    )
    if geometry_sha256 != COUNT_DRIVEN_SOURCE_GEOMETRY_SHA256:
        raise RuntimeError("evaluated authored twin liquid geometry hash drifted")
    if topology_sha256 != COUNT_DRIVEN_SOURCE_TOPOLOGY_SHA256:
        raise RuntimeError("authored twin liquid topology hash drifted")
    world_bottom = float(np.min(world_points[:, 2]))
    world_height = float(np.ptp(world_points[:, 2]))
    source_signed_volume = abs(
        signed_mesh_volume_m3(world_points, face_counts, face_indices)
    )
    source_effective_area = source_signed_volume / world_height
    if not math.isclose(
        world_bottom,
        COUNT_DRIVEN_SOURCE_WORLD_BOTTOM_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("authored twin liquid bottom anchor drifted")
    if not math.isclose(
        world_height,
        COUNT_DRIVEN_SOURCE_WORLD_HEIGHT_M,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ) or not math.isclose(
        source_signed_volume,
        COUNT_DRIVEN_SOURCE_SIGNED_VOLUME_M3,
        rel_tol=0.0,
        abs_tol=2.0e-10,
    ):
        raise RuntimeError("authored twin liquid evaluated volume drifted")

    source_xformable = UsdGeom.Xformable(source_xform_prim)
    source_xform_matrix = source_xformable.GetLocalTransformation(requested_time)
    if isinstance(source_xform_matrix, tuple):
        source_xform_matrix = source_xform_matrix[0]
    source_mesh_matrix = UsdGeom.Xformable(source_mesh_prim).GetLocalTransformation(
        requested_time
    )
    if isinstance(source_mesh_matrix, tuple):
        source_mesh_matrix = source_mesh_matrix[0]

    body_xform = UsdGeom.Xform.Define(stage, COUNT_DRIVEN_BODY_XFORM_PATH)
    UsdGeom.Xformable(body_xform.GetPrim()).MakeMatrixXform().Set(source_xform_matrix)
    body_mesh = UsdGeom.Mesh.Define(stage, COUNT_DRIVEN_BODY_MESH_PATH)
    UsdGeom.Xformable(body_mesh.GetPrim()).MakeMatrixXform().Set(source_mesh_matrix)
    body_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    body_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    for source_attr, target_attr in (
        (source_mesh.GetOrientationAttr(), body_mesh.GetOrientationAttr()),
        (source_mesh.GetSubdivisionSchemeAttr(), body_mesh.GetSubdivisionSchemeAttr()),
        (source_mesh.GetInterpolateBoundaryAttr(), body_mesh.GetInterpolateBoundaryAttr()),
        (
            source_mesh.GetFaceVaryingLinearInterpolationAttr(),
            body_mesh.GetFaceVaryingLinearInterpolationAttr(),
        ),
        (
            source_mesh.GetTriangleSubdivisionRuleAttr(),
            body_mesh.GetTriangleSubdivisionRuleAttr(),
        ),
        (source_mesh.GetDoubleSidedAttr(), body_mesh.GetDoubleSidedAttr()),
    ):
        value = source_attr.Get(requested_time)
        if value is not None:
            target_attr.Set(value)
    copied_primvar_names: list[str] = []
    for source_attr in source_mesh_prim.GetAttributes():
        name = str(source_attr.GetName())
        if not name.startswith("primvars:") or name == "primvars:normals":
            continue
        value = source_attr.Get(requested_time)
        if value is None:
            continue
        target_attr = body_mesh.GetPrim().CreateAttribute(
            name,
            source_attr.GetTypeName(),
            source_attr.IsCustom(),
            source_attr.GetVariability(),
        )
        target_attr.Set(value)
        for metadata_key in ("interpolation", "elementSize"):
            if source_attr.HasMetadata(metadata_key):
                target_attr.SetMetadata(
                    metadata_key, source_attr.GetMetadata(metadata_key)
                )
        copied_primvar_names.append(name)
    # Face-varying normals resolved at t681 are invalid after anisotropic z
    # scaling; leave normals unauthored so the renderer derives them.
    body_mesh.GetNormalsAttr().Clear()
    initial_geometry = build_count_driven_liquid_body_geometry(
        source_points,
        contained_particle_count=0,
        particle_volume_m3=particle_volume_m3,
        source_world_height_m=world_height,
        source_effective_area_m2=source_effective_area,
    )
    initial_points = np.asarray(initial_geometry["points"], dtype=np.float64)
    body_mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*row) for row in initial_points])
    )
    initial_extent = np.asarray(
        [np.min(initial_points, axis=0), np.max(initial_points, axis=0)],
        dtype=np.float64,
    )
    body_mesh.GetExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*row) for row in initial_extent])
    )
    body_mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdShade.MaterialBindingAPI.Apply(body_mesh.GetPrim()).Bind(body_material)
    display_color_primvar = None
    if height_banded:
        display_color_primvar = UsdGeom.PrimvarsAPI(body_mesh).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        fallback_color = tuple(
            float(value)
            for value in BASE_TANK_TRANSMISSION_1CM[COMPOSITION_COMPONENTS[0]]
        )
        display_color_primvar.Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(*fallback_color) for _ in range(len(initial_points))]
            )
        )
    _author_dual_liquid_semantics(
        body_mesh.GetPrim(), UsdSemantics, LegacySemantics
    )
    runtime = {
        "mesh": body_mesh,
        "source_points": source_points,
        "face_counts": face_counts,
        "face_indices": face_indices,
        "source_world_height_m": world_height,
        "source_effective_area_m2": source_effective_area,
        "particle_volume_m3": particle_volume_m3,
        "shader": body_shader,
        "display_color_primvar": display_color_primvar,
        "height_banded": height_banded,
    }
    evidence = {
        "resolved_points_time_code": _resolved_source_sample_time(
            source_mesh.GetPointsAttr(), requested_time_code=requested
        ),
        "resolved_normals_time_code": _resolved_source_sample_time(
            source_mesh.GetNormalsAttr(), requested_time_code=requested
        ),
        "resolved_extent_time_code": _resolved_source_sample_time(
            source_mesh.GetExtentAttr(), requested_time_code=requested
        ),
        "source_point_count": len(source_points),
        "source_face_count": len(face_counts),
        "source_corner_count": len(face_indices),
        "source_fan_triangle_count": fan_triangle_count,
        "source_geometry_sha256": geometry_sha256,
        "source_topology_sha256": topology_sha256,
        "copied_primvar_names": sorted(copied_primvar_names),
        "source_world_bottom_z_m": world_bottom,
        "source_world_height_m": world_height,
        "source_signed_volume_m3": source_signed_volume,
        "source_effective_area_m2": source_effective_area,
        "xy_area_correction_scale": initial_geometry[
            "xy_area_correction_scale"
        ],
        "initial_points_sha256": initial_geometry["points_sha256"],
        "initial_visibility": "invisible",
        "body_color_mode": (
            "height_banded_particle_composition"
            if height_banded
            else "global_mean_composition"
        ),
    }
    return runtime, evidence


def _count_driven_contained_mask(positions: Any) -> Any:
    """Return the exact regular-dodecagon containment mask used by F2 gates."""

    import numpy as np

    rows = np.asarray(positions, dtype=np.float64)
    if rows.ndim == 1 and rows.size == 0:
        rows = rows.reshape((0, 3))
    if rows.ndim != 2 or rows.shape[1:] != (3,) or not np.all(np.isfinite(rows)):
        raise RuntimeError("count-driven body received invalid PBD positions")
    apothem = SOURCE_LIQUID_RADIUS_M * math.cos(
        math.pi / VIAL_COLLIDER_WALL_COUNT
    )
    angles = (
        2.0
        * math.pi
        * np.arange(VIAL_COLLIDER_WALL_COUNT)
        / VIAL_COLLIDER_WALL_COUNT
    )
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    centered = rows[:, :2] - np.asarray(VIAL_CENTER_XY_M)[None, :]
    inside_xy = np.all(centered @ normals.T <= apothem + 1.0e-6, axis=1)
    inside_z = (
        (rows[:, 2] >= SOURCE_LIQUID_BOTTOM_M - 1.0e-6)
        & (rows[:, 2] <= VIAL_MOUTH_M + 1.0e-6)
    )
    return inside_xy & inside_z


def _update_count_driven_twin_liquid_body(
    runtime: dict[str, Any],
    *,
    contained_positions: Any,
    contained_compositions: Any,
    global_shader: Any,
    source_transmissions: Any,
    Usd: Any,
    UsdGeom: Any,
    Gf: Any,
    Vt: Any,
) -> dict[str, Any]:
    """Update body height/color from live contained PBD rows and prove volume."""

    import numpy as np

    compositions = np.asarray(contained_compositions, dtype=np.float64)
    if compositions.ndim != 2 or compositions.shape[1:] != (3,) or not np.all(
        np.isfinite(compositions)
    ):
        raise RuntimeError("contained compositions must be finite with shape (N, 3)")
    positions = np.asarray(contained_positions, dtype=np.float64)
    if positions.shape != compositions.shape or not np.all(np.isfinite(positions)):
        raise RuntimeError(
            "contained positions must be finite and match contained compositions"
        )
    count = len(compositions)
    geometry = build_count_driven_liquid_body_geometry(
        runtime["source_points"],
        contained_particle_count=count,
        particle_volume_m3=runtime["particle_volume_m3"],
        source_world_height_m=runtime["source_world_height_m"],
        source_effective_area_m2=runtime["source_effective_area_m2"],
    )
    points = np.asarray(geometry["points"], dtype=np.float64)
    body_mesh = runtime["mesh"]
    body_mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*row) for row in points])
    )
    readback_points = np.asarray(
        [tuple(float(value) for value in row) for row in body_mesh.GetPointsAttr().Get()],
        dtype=np.float64,
    )
    extent = np.asarray(
        [np.min(readback_points, axis=0), np.max(readback_points, axis=0)],
        dtype=np.float64,
    )
    body_mesh.GetExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*row) for row in extent])
    )
    visibility = UsdGeom.Tokens.inherited if count else UsdGeom.Tokens.invisible
    body_mesh.GetVisibilityAttr().Set(visibility)
    # PhysX particle positions are in stage/world metres, while the copied
    # twin mesh points remain in their authored local coordinate system.  The
    # height-band projection must compare like coordinates or the mesh's
    # centimetre-to-metre parent transform collapses most vertices into the
    # first/last colour band.
    body_world_points = _mesh_world_points(
        body_mesh,
        readback_points,
        time_code=COUNT_DRIVEN_SOURCE_TIME_CODE,
        Usd=Usd,
        UsdGeom=UsdGeom,
        Gf=Gf,
    )
    source_extinctions = _source_extinctions_from_transmissions(
        {
            tank: source_transmissions[index]
            for index, tank in enumerate(COMPOSITION_COMPONENTS)
        }
    )
    band_evidence = None
    vertex_color_sha256 = None
    if count:
        color = mix_extinction_transmission(
            compositions.mean(axis=0, keepdims=True), source_extinctions
        )[0]
        if runtime.get("height_banded") is True:
            vertex_colors, band_evidence = (
                build_height_banded_transmission_colors(
                    positions,
                    compositions,
                    body_world_points,
                    source_transmissions=source_transmissions,
                    band_count=HEIGHT_BAND_COUNT,
                )
            )
            primvar = runtime.get("display_color_primvar")
            if primvar is None:
                raise RuntimeError("height-banded body lacks displayColor primvar")
            primvar.Set(
                Vt.Vec3fArray([Gf.Vec3f(*row) for row in vertex_colors])
            )
            readback = np.asarray(
                [
                    tuple(float(value) for value in row)
                    for row in (primvar.Get() or [])
                ],
                dtype=np.float32,
            )
            expected = np.asarray(vertex_colors, dtype=np.float32)
            if readback.shape != expected.shape or not np.array_equal(
                readback, expected
            ):
                raise RuntimeError(
                    "height-banded displayColor readback differs from authored values"
                )
            vertex_color_sha256 = hashlib.sha256(
                np.asarray(readback, dtype="<f4", order="C").tobytes()
            ).hexdigest()
    else:
        color = np.asarray(
            source_transmissions[0],
            dtype=np.float64,
        )
    color_readback = _set_global_blend_color(global_shader, Gf, color)
    world_points = body_world_points
    signed_volume = signed_mesh_volume_m3(
        world_points, runtime["face_counts"], runtime["face_indices"]
    )
    target_volume = count * float(runtime["particle_volume_m3"])
    residual = abs(abs(signed_volume) - target_volume)
    world_bottom = float(np.min(world_points[:, 2]))
    world_extent = np.asarray(
        [np.min(world_points, axis=0), np.max(world_points, axis=0)],
        dtype=np.float64,
    )
    if not math.isclose(
        world_bottom,
        COUNT_DRIVEN_SOURCE_WORLD_BOTTOM_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("count-driven body bottom anchor drifted")
    if residual > 2.0e-10:
        raise RuntimeError("count-driven body signed volume residual exceeded 0.0002 mL")
    geometry.update(
        {
            "points": readback_points.tolist(),
            "points_sha256": _float_array_sha256(readback_points),
            "extent": extent.tolist(),
            "extent_sha256": _float_array_sha256(extent),
            "world_points_sha256": _float_array_sha256(world_points),
            "world_bottom_z_m": world_bottom,
            "world_height_m": float(np.ptp(world_points[:, 2])),
            "world_extent_m": world_extent.tolist(),
            "post_transform_signed_volume_m3": signed_volume,
            "post_transform_volume_residual_m3": residual,
            "volume_proof_coordinate_path": (
                "local_points_vec3f_then_inherited_world_transform"
            ),
            "body_visibility": str(visibility),
            "body_material_color": list(color_readback),
            "body_material_color_sha256": _json_value_sha256(
                list(color_readback)
            ),
            "body_color_mode": (
                "height_banded_particle_composition"
                if runtime.get("height_banded") is True
                else "global_mean_composition"
            ),
            "body_vertex_color_sha256": vertex_color_sha256,
            "height_band_evidence": band_evidence,
        }
    )
    return geometry


def _select_count_driven_body_capture_geometry(
    live_geometry: dict[str, Any],
    *,
    physics_step: int,
    phase: str,
) -> dict[str, Any]:
    """Project live body evidence into one persisted capture/stage record."""

    return {
        **{
            field: live_geometry[field]
            for field in COUNT_DRIVEN_BODY_CAPTURE_FIELDS
        },
        "physics_step": physics_step,
        "phase": phase,
    }


def _with_session_edit_context(
    Usd: Any,
    stage: Any,
    session_layer: Any,
    action: Any,
) -> Any:
    """Run one live USD mutation against the experiment's strongest layer."""

    with Usd.EditContext(stage, session_layer):
        return action()


def _set_persistent_session_edit_target(
    stage: Any,
    session_layer: Any,
    Usd: Any,
) -> dict[str, object]:
    """Keep deferred PhysX/Replicator writes on the strongest runtime layer."""

    requested_identifier = str(session_layer.identifier)
    stage.SetEditTarget(Usd.EditTarget(session_layer))
    observed_layer = stage.GetEditTarget().GetLayer()
    observed_identifier = str(observed_layer.identifier)
    evidence = {
        "requested_layer_identifier": requested_identifier,
        "observed_layer_identifier": observed_identifier,
        "readback_matches_session": observed_identifier == requested_identifier,
    }
    if evidence["readback_matches_session"] is not True:
        raise RuntimeError("stage did not retain the persistent session edit target")
    return evidence


def _runtime_particle_layer_evidence(
    *,
    stage: Any,
    session_layer: Any,
    shared_points_path: Any,
    Sdf: Any,
) -> dict[str, object]:
    """Report every runtime-row layer spec before enforcing session-only writes."""

    authored_property_names = (
        "positions",
        "velocities",
        "orientations",
        "scales",
        "protoIndices",
        "ids",
    )
    read_only_property_name = "simulationPoints"
    property_names = (*authored_property_names, read_only_property_name)
    property_specs: dict[str, bool] = {}
    for property_name in property_names:
        property_path = Sdf.Path(shared_points_path).AppendProperty(property_name)
        property_specs[f"{property_name}_session_layer_spec"] = (
            session_layer.GetPropertyAtPath(property_path) is not None
        )
        property_specs[f"{property_name}_root_layer_spec"] = (
            stage.GetRootLayer().GetPropertyAtPath(property_path) is not None
        )
    failed_properties = [
        property_name
        for property_name in authored_property_names
        if (
            not property_specs[f"{property_name}_session_layer_spec"]
            or property_specs[f"{property_name}_root_layer_spec"]
        )
    ]
    if property_specs[f"{read_only_property_name}_root_layer_spec"]:
        failed_properties.append(read_only_property_name)
    final_layer = stage.GetEditTarget().GetLayer()
    session_identifier = str(session_layer.identifier)
    final_identifier = str(final_layer.identifier)
    final_edit_target_is_session = final_identifier == session_identifier
    return {
        "property_specs": property_specs,
        "failed_properties": failed_properties,
        "authored_session_only_properties": list(authored_property_names),
        "read_only_runtime_property": read_only_property_name,
        "read_only_runtime_property_session_spec_required": False,
        "read_only_runtime_property_root_spec_forbidden": True,
        "read_only_runtime_property_documentation_url": (
            SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL
        ),
        "session_layer_identifier": session_identifier,
        "final_edit_target_identifier": final_identifier,
        "final_edit_target_is_session": final_edit_target_is_session,
        "passed": not failed_properties and final_edit_target_is_session,
    }


def _runtime_simulation_points_evidence(
    simulation_points: object,
    *,
    expected_particle_count: int,
    particle_id_count: int,
    containment_particle_count: int,
) -> dict[str, object]:
    """Attest the documented read-only PhysX simulation-state readback."""

    import numpy as np

    try:
        rows = np.asarray(simulation_points, dtype=np.float64)
    except (TypeError, ValueError):
        rows = np.empty((0,), dtype=np.float64)
    shape = [int(value) for value in rows.shape]
    valid_xyz_shape = rows.ndim == 2 and rows.shape[1:] == (3,)
    row_count = int(rows.shape[0]) if rows.ndim >= 1 else 0
    finite_xyz_rows = bool(valid_xyz_shape and np.all(np.isfinite(rows)))
    row_counts_match = bool(
        valid_xyz_shape
        and row_count
        == expected_particle_count
        == particle_id_count
        == containment_particle_count
    )
    return {
        "attribute": "physxParticle:simulationPoints",
        "documented_read_only": True,
        "documentation_url": (
            SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL
        ),
        "runtime_readback_required": True,
        "runtime_readback_shape": shape,
        "runtime_readback_row_count": row_count,
        "expected_particle_count": int(expected_particle_count),
        "particle_id_count": int(particle_id_count),
        "containment_particle_count": int(containment_particle_count),
        "finite_xyz_rows": finite_xyz_rows,
        "row_counts_match": row_counts_match,
        "passed": finite_xyz_rows and row_counts_match,
    }


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


class _PointInstancerParticleAdapter:
    """Expose the emitter's PointInstancer rows through the runner boundary."""

    def __init__(self, geom: Any, particle_set: Any) -> None:
        self.geom = geom
        self.particle_set = particle_set

    def GetPrim(self) -> Any:
        return self.geom.GetPrim()

    def GetPointsAttr(self) -> Any:
        return self.geom.GetPositionsAttr()

    def GetPositionsAttr(self) -> Any:
        return self.geom.GetPositionsAttr()

    def GetVelocitiesAttr(self) -> Any:
        return self.geom.GetVelocitiesAttr()

    def GetOrientationsAttr(self) -> Any:
        return self.geom.GetOrientationsAttr()

    def GetScalesAttr(self) -> Any:
        return self.geom.GetScalesAttr()

    def GetWidthsAttr(self) -> Any:
        # Legacy runner compatibility; PointInstancer uses unit Vec3f scales.
        return self.geom.GetScalesAttr()

    def GetProtoIndicesAttr(self) -> Any:
        return self.geom.GetProtoIndicesAttr()

    def GetIdsAttr(self) -> Any:
        return self.geom.GetIdsAttr()

    def CreateIdsAttr(self) -> Any:
        return self.geom.CreateIdsAttr()

    def GetSimulationPointsAttr(self) -> Any:
        return self.particle_set.GetSimulationPointsAttr()


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


def _read_effective_camera_render_state(
    camera_prim: Any,
    camera: Any,
    Usd: Any,
    UsdGeom: Any,
    *,
    time_code: float,
) -> dict[str, Any]:
    """Read back the fixed FillCam pose and effective projection attributes."""

    usd_time = Usd.TimeCode(time_code)
    matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(usd_time)
    world_transform = [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]
    if any(not math.isfinite(value) for row in world_transform for value in row):
        raise RuntimeError("FillCam effective world transform is non-finite")

    def scalar(attribute: Any, name: str) -> float:
        value = attribute.Get(usd_time)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"FillCam {name} readback is missing")
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError(f"FillCam {name} readback is non-finite")
        return result

    projection_value = camera.GetProjectionAttr().Get(usd_time)
    projection = str(projection_value) if projection_value is not None else ""
    clipping_value = camera.GetClippingRangeAttr().Get(usd_time)
    clipping = (
        [float(clipping_value[0]), float(clipping_value[1])]
        if clipping_value is not None
        else None
    )
    result = {
        "time_code": float(time_code),
        "world_transform_matrix": world_transform,
        "projection": projection,
        "focal_length": scalar(camera.GetFocalLengthAttr(), "focal length"),
        "horizontal_aperture": scalar(
            camera.GetHorizontalApertureAttr(), "horizontal aperture"
        ),
        "vertical_aperture": scalar(
            camera.GetVerticalApertureAttr(), "vertical aperture"
        ),
        "focus_distance": scalar(
            camera.GetFocusDistanceAttr(), "focus distance"
        ),
        "f_stop": scalar(camera.GetFStopAttr(), "f-stop"),
        "clipping_range": clipping,
    }
    if (
        not projection
        or result["focal_length"] <= 0.0
        or result["horizontal_aperture"] <= 0.0
        or result["vertical_aperture"] <= 0.0
        or result["focus_distance"] < 0.0
        or result["f_stop"] < 0.0
        or clipping is None
        or not all(math.isfinite(value) for value in clipping)
        or clipping[0] <= 0.0
        or clipping[1] <= clipping[0]
    ):
        raise RuntimeError("FillCam effective projection readback is invalid")
    return result


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
    *,
    dt_s: float = DIFFUSION_DT_S,
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
            particle_ids, positions, before, dt_s=dt_s
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


def _json_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_paired_capture_state(state: Any) -> dict[str, Any]:
    """Validate and copy the exact physics/camera state around one capture pair."""

    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("paired capture state must be a mapping")
    missing = [field for field in PAIRED_CAPTURE_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"paired capture state is missing fields: {missing}")
    callback_count = state["callback_count"]
    if (
        isinstance(callback_count, bool)
        or not isinstance(callback_count, int)
        or callback_count < 0
    ):
        raise ValueError("paired capture callback_count must be a non-negative integer")
    timeline_time_s = state["timeline_time_s"]
    if (
        isinstance(timeline_time_s, bool)
        or not isinstance(timeline_time_s, (int, float))
        or not math.isfinite(float(timeline_time_s))
    ):
        raise ValueError("paired capture timeline_time_s must be finite")
    camera_path = state["camera_path"]
    if not isinstance(camera_path, str) or not camera_path:
        raise ValueError("paired capture camera_path must be non-empty")

    arrays: dict[str, list[list[float]]] = {}
    for field in ("positions", "simulation_points", "velocities"):
        rows = np.asarray(state[field], dtype=np.float64)
        if (
            rows.ndim != 2
            or rows.shape[1:] != (3,)
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError(f"paired capture {field} must be finite with shape (N, 3)")
        arrays[field] = rows.tolist()

    integer_rows: dict[str, list[int]] = {}
    for field in ("ids", "proto_indices"):
        values = state[field]
        if not isinstance(values, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"paired capture {field} must contain only integers")
        integer_rows[field] = list(values)

    body_fields = (
        "compositions",
        "body_points",
        "body_extent",
        "body_visibility",
        "body_material_color",
        "stream_visibility",
    )
    body_present = [field in state for field in body_fields]
    body_state: dict[str, Any] = {}
    if any(body_present):
        if not all(body_present):
            missing_body = [
                field for field, present in zip(body_fields, body_present) if not present
            ]
            raise ValueError(
                f"paired count-driven body state is missing fields: {missing_body}"
            )
        body_points = np.asarray(state["body_points"], dtype=np.float64)
        body_extent = np.asarray(state["body_extent"], dtype=np.float64)
        body_color = np.asarray(state["body_material_color"], dtype=np.float64)
        compositions = np.asarray(state["compositions"], dtype=np.float64)
        if (
            compositions.shape != (len(integer_rows["ids"]), 3)
            or not np.all(np.isfinite(compositions))
        ):
            raise ValueError(
                "paired capture compositions must be finite with shape (N, 3)"
            )
        if (
            body_points.ndim != 2
            or body_points.shape[1:] != (3,)
            or not np.all(np.isfinite(body_points))
        ):
            raise ValueError("paired capture body_points must be finite with shape (N, 3)")
        if body_extent.shape != (2, 3) or not np.all(np.isfinite(body_extent)):
            raise ValueError("paired capture body_extent must be finite with shape (2, 3)")
        if body_color.shape != (3,) or not np.all(np.isfinite(body_color)):
            raise ValueError("paired capture body_material_color must contain three finite values")
        if not isinstance(state["body_visibility"], str) or not state["body_visibility"]:
            raise ValueError("paired capture body_visibility must be non-empty")
        streams = state["stream_visibility"]
        if not isinstance(streams, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in streams
        ):
            raise ValueError("paired capture stream_visibility must contain strings")
        body_state = {
            "compositions": compositions.tolist(),
            "body_points": body_points.tolist(),
            "body_extent": body_extent.tolist(),
            "body_visibility": state["body_visibility"],
            "body_material_color": body_color.tolist(),
            "stream_visibility": list(streams),
        }

    return {
        "callback_count": callback_count,
        "timeline_time_s": float(timeline_time_s),
        "camera_path": camera_path,
        **arrays,
        **integer_rows,
        **body_state,
    }


def _paired_capture_state_evidence(before: Any, after: Any) -> dict[str, Any]:
    """Hash and compare every frozen state field across RGB and semantic passes."""

    import numpy as np

    normalized_before = _normalize_paired_capture_state(before)
    normalized_after = _normalize_paired_capture_state(after)
    failed: list[str] = []
    if normalized_before["callback_count"] != normalized_after["callback_count"]:
        failed.append("callback_count")
    if not math.isclose(
        normalized_before["timeline_time_s"],
        normalized_after["timeline_time_s"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failed.append("timeline_time_s")
    if normalized_before["camera_path"] != normalized_after["camera_path"]:
        failed.append("camera_path")
    for normalized in (normalized_before, normalized_after):
        row_count = len(normalized["positions"])
        for field in ("simulation_points", "velocities", "ids", "proto_indices"):
            if len(normalized[field]) != row_count and field not in failed:
                failed.append(field)
    for field in ("positions", "simulation_points", "velocities"):
        if not np.array_equal(
            np.asarray(normalized_before[field], dtype=np.float64),
            np.asarray(normalized_after[field], dtype=np.float64),
        ) and field not in failed:
            failed.append(field)
    for field in ("ids", "proto_indices"):
        if normalized_before[field] != normalized_after[field] and field not in failed:
            failed.append(field)
    body_fields = (
        "compositions",
        "body_points",
        "body_extent",
        "body_visibility",
        "body_material_color",
        "stream_visibility",
    )
    before_has_body = all(field in normalized_before for field in body_fields)
    after_has_body = all(field in normalized_after for field in body_fields)
    if before_has_body != after_has_body:
        failed.extend(field for field in body_fields if field not in failed)
    elif before_has_body:
        for field in body_fields:
            before_value = normalized_before[field]
            after_value = normalized_after[field]
            if isinstance(before_value, list) and field in {
                "compositions",
                "body_points",
                "body_extent",
                "body_material_color",
            }:
                equal = np.array_equal(
                    np.asarray(before_value, dtype=np.float64),
                    np.asarray(after_value, dtype=np.float64),
                )
            else:
                equal = before_value == after_value
            if not equal and field not in failed:
                failed.append(field)

    evidence: dict[str, Any] = {
        "passed": not failed,
        "failed_predicates": failed,
        "row_count": len(normalized_before["positions"]),
        "callback_count_before": normalized_before["callback_count"],
        "callback_count_after": normalized_after["callback_count"],
        "timeline_time_s_before": normalized_before["timeline_time_s"],
        "timeline_time_s_after": normalized_after["timeline_time_s"],
        "camera_path_before": normalized_before["camera_path"],
        "camera_path_after": normalized_after["camera_path"],
    }
    for field in ("positions", "simulation_points", "velocities"):
        evidence[f"{field}_sha256_before"] = _float_array_sha256(
            normalized_before[field]
        )
        evidence[f"{field}_sha256_after"] = _float_array_sha256(
            normalized_after[field]
        )
    for field in ("ids", "proto_indices"):
        evidence[f"{field}_sha256_before"] = _particle_ids_sha256(
            normalized_before[field]
        )
        evidence[f"{field}_sha256_after"] = _particle_ids_sha256(
            normalized_after[field]
        )
    if before_has_body and after_has_body:
        for field in body_fields:
            evidence[f"{field}_sha256_before"] = _json_value_sha256(
                normalized_before[field]
            )
            evidence[f"{field}_sha256_after"] = _json_value_sha256(
                normalized_after[field]
            )
    return evidence


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
        "simulation_points": len(particle_points.GetSimulationPointsAttr().Get() or []),
        "positions": len(_positions(particle_points.GetPrim(), instancer=True)),
        "velocities": len(particle_points.GetVelocitiesAttr().Get() or []),
        "orientations": len(particle_points.GetOrientationsAttr().Get() or []),
        "scales": len(particle_points.GetScalesAttr().Get() or []),
        "proto_indices": len(particle_points.GetProtoIndicesAttr().Get() or []),
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


def _author_dual_liquid_semantics(
    prim: Any, UsdSemantics: Any, LegacySemantics: Any, *, label: str = "liquid"
) -> dict[str, Any]:
    """Author current class labels plus NVIDIA's legacy semantic schema."""

    direct = _author_direct_liquid_semantics(prim, UsdSemantics)
    legacy = {"available": LegacySemantics is not None, "authored": False, "semantic_type": "class", "semantic_data": label}
    if LegacySemantics is None:
        raise RuntimeError("legacy Semantics module is unavailable for PointInstancer prototypes")
    api = LegacySemantics.SemanticsAPI.Apply(prim, "Semantics")
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(label)
    legacy["authored"] = True
    legacy["semantic_type_readback"] = str(api.GetSemanticTypeAttr().Get())
    legacy["semantic_data_readback"] = str(api.GetSemanticDataAttr().Get())
    if legacy["semantic_type_readback"] != "class" or legacy["semantic_data_readback"] != label:
        raise RuntimeError(f"legacy semantic readback failed for {prim.GetPath()}")
    direct["legacy"] = legacy
    return direct


def _set_transparent_vial_annotation_state(
    stage: Any,
    session_layer: Any,
    Usd: Any,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Session-author and read back the exact transparent-vial flag."""

    override = TRANSPARENT_ANNOTATION_OVERRIDE
    mesh = stage.GetPrimAtPath(override.prim_path)
    if not mesh.IsValid() or mesh.GetTypeName() != "Mesh":
        raise RuntimeError("transparent vial annotation target is missing or not a Mesh")
    attribute = mesh.GetAttribute(override.attribute_name)
    if not attribute.IsValid() or str(attribute.GetTypeName()) != "bool":
        raise RuntimeError("transparent vial annotation attribute is missing or not bool")
    with Usd.EditContext(stage, session_layer):
        if not attribute.Set(bool(enabled)):
            raise RuntimeError(
                f"could not set {override.attribute_name}={enabled!r} on "
                f"{override.prim_path}"
            )
    composed = stage.GetPrimAtPath(override.prim_path).GetAttribute(
        override.attribute_name
    )
    property_stack = composed.GetPropertyStack(Usd.TimeCode.Default())
    strongest_is_session = bool(
        property_stack and property_stack[0].layer == session_layer
    )
    composed_value = composed.Get()
    if (
        str(composed.GetTypeName()) != "bool"
        or type(composed_value) is not bool
        or composed_value is not enabled
        or not strongest_is_session
    ):
        raise RuntimeError(
            "transparent vial annotation state/readback did not match the requested "
            "session-layer value"
        )
    return {
        "requested_value": enabled,
        "composed_type_name": str(composed.GetTypeName()),
        "composed_value": composed_value,
        "property_stack_strongest_is_session_layer": strongest_is_session,
    }


def _prepare_transparent_vial_annotation_control(
    stage: Any,
    session_layer: Any,
    Usd: Any,
    Sdf: Any,
) -> dict[str, Any]:
    """Validate the reviewed mesh, then establish the source-equivalent RGB state."""

    override = TRANSPARENT_ANNOTATION_OVERRIDE
    parent = stage.GetPrimAtPath(override.parent_path)
    mesh = stage.GetPrimAtPath(override.prim_path)
    if not parent.IsValid() or parent.GetTypeName() != "Xform":
        raise RuntimeError("transparent vial parent is missing or not an Xform")
    if not mesh.IsValid() or mesh.GetTypeName() != "Mesh":
        raise RuntimeError("transparent vial annotation target is missing or not a Mesh")
    source_class = parent.GetAttribute("userProperties:synthetic_class")
    source_class_value = source_class.Get() if source_class.IsValid() else None
    if source_class_value != override.required_parent_class:
        raise RuntimeError(
            "transparent vial source class differs from the reviewed annotation contract"
        )
    source_attribute = mesh.GetAttribute(override.attribute_name)
    attribute_exists = bool(source_attribute and source_attribute.IsValid())
    source = TransparentAnnotationSourceFact(
        attribute_exists=attribute_exists,
        type_name=(
            str(source_attribute.GetTypeName()) if attribute_exists else None
        ),
        has_authored_value=bool(
            attribute_exists and source_attribute.HasAuthoredValueOpinion()
        ),
        value=source_attribute.Get() if attribute_exists else None,
    )
    plan = plan_transparent_annotation_override(
        {
            override.parent_path: parent.GetTypeName(),
            override.prim_path: mesh.GetTypeName(),
        },
        {override.parent_path: source_class_value},
        source,
    )
    if source.value is not None and source.value is not False:
        raise RuntimeError(
            "transparent vial source already enables doNotCastShadows; false would not "
            "be source-equivalent"
        )
    with Usd.EditContext(stage, session_layer):
        attribute = mesh.CreateAttribute(
            override.attribute_name,
            Sdf.ValueTypeNames.Bool,
            custom=True,
        )
        if not attribute.IsValid():
            raise RuntimeError("could not create transparent vial session attribute")
    initial_rgb_state = _set_transparent_vial_annotation_state(
        stage,
        session_layer,
        Usd,
        enabled=False,
    )
    return {
        "prim_path": override.prim_path,
        "parent_path": override.parent_path,
        "attribute_name": override.attribute_name,
        "semantic_pass_value": True,
        "rgb_and_restore_value": False,
        "capture_mode": "paired_same_physics_state_rgb_then_semantic",
        "documentation_url": TRANSPARENT_ANNOTATION_DOCUMENTATION_URL,
        "source": {
            "prim_type": plan.prim_type,
            "planned_parent_class": plan.planned_parent_class,
            "attribute_exists": plan.source.attribute_exists,
            "type_name": plan.source.type_name,
            "has_authored_value": plan.source.has_authored_value,
            "value": plan.source.value,
        },
        "initial_rgb_state": {
            **initial_rgb_state,
            "phase": "after_untouched_reference_before_prefill",
            "source_equivalent": True,
        },
        "transitions": [],
        "property_stack_strongest_is_session_layer": True,
        "authored_after_untouched_reference_capture": True,
        "source_layer_unchanged_expected": True,
    }


def _capture_paired_particle_state(
    *,
    particle_points: Any,
    callback_count: int,
    timeline: Any,
    camera_prim: Any,
    compositions: Any | None = None,
    count_driven_body_runtime: dict[str, Any] | None = None,
    global_shader: Any | None = None,
    stream_records: dict[str, dict[str, Any]] | None = None,
    UsdGeom: Any | None = None,
) -> dict[str, Any]:
    """Read the exact frozen fields required around the semantic-only pass."""

    camera_path = str(camera_prim.GetPath()) if camera_prim.IsValid() else ""
    state = {
        "callback_count": int(callback_count),
        "timeline_time_s": float(timeline.get_current_time()),
        "camera_path": camera_path,
        "positions": [
            [float(value[0]), float(value[1]), float(value[2])]
            for value in _positions(particle_points.GetPrim(), instancer=True)
        ],
        "simulation_points": [
            [float(value[0]), float(value[1]), float(value[2])]
            for value in (particle_points.GetSimulationPointsAttr().Get() or [])
        ],
        "velocities": [
            [float(value[0]), float(value[1]), float(value[2])]
            for value in (particle_points.GetVelocitiesAttr().Get() or [])
        ],
        "ids": [int(value) for value in (particle_points.GetIdsAttr().Get() or [])],
        "proto_indices": [
            int(value) for value in (particle_points.GetProtoIndicesAttr().Get() or [])
        ],
    }
    if count_driven_body_runtime is not None:
        import numpy as np

        if (
            compositions is None
            or global_shader is None
            or stream_records is None
            or UsdGeom is None
        ):
            raise RuntimeError("count-driven paired state inputs are incomplete")
        body_mesh = count_driven_body_runtime["mesh"]
        color = global_shader.GetInput("specular_transmission_color").Get()
        if color is None:
            raise RuntimeError("count-driven body material color readback is missing")
        state.update(
            {
                "compositions": np.asarray(compositions, dtype=np.float64).tolist(),
                "body_points": [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (body_mesh.GetPointsAttr().Get() or [])
                ],
                "body_extent": [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (body_mesh.GetExtentAttr().Get() or [])
                ],
                "body_visibility": str(body_mesh.GetVisibilityAttr().Get()),
                "body_material_color": [float(color[index]) for index in range(3)],
                "stream_visibility": sorted(
                    str(record["active_source"])
                    for record in stream_records.values()
                    if UsdGeom.Imageable(record["prim"]).ComputeVisibility()
                    != UsdGeom.Tokens.invisible
                    and isinstance(record.get("active_source"), str)
                ),
            }
        )
    return _normalize_paired_capture_state(state)


def _reset_renderer_accumulation_if_supported(context: Any) -> dict[str, bool]:
    """Reset accumulation after a glass-state toggle when the context exposes it."""

    reset = getattr(context, "reset_renderer_accumulation", None)
    if not callable(reset):
        return {"supported": False, "called": False}
    reset()
    return {"supported": True, "called": True}


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


def _configure_pathtracing_sampling(
    settings: Any,
    receipt: dict[str, Any],
    *,
    requested: int,
) -> None:
    """Bind PathTracing SPP to the value multiplied by Replicator subframes."""

    total_path = "/rtx/pathtracing/totalSpp"
    iteration_path = "/rtx/pathtracing/spp"
    initial_total = int(settings.get_as_int(total_path))
    initial_iteration = int(settings.get_as_int(iteration_path))
    settings.set_int(total_path, int(requested))
    settings.set_int(iteration_path, int(requested))
    readback_total = int(settings.get_as_int(total_path))
    readback_iteration = int(settings.get_as_int(iteration_path))
    receipt["render"].update(
        {
            "pathtracing_total_spp_requested": int(requested),
            "pathtracing_total_spp_initial": initial_total,
            "pathtracing_total_spp_readback": readback_total,
            "pathtracing_spp_requested": int(requested),
            "pathtracing_spp_initial": initial_iteration,
            "pathtracing_spp_readback": readback_iteration,
            "pathtracing_sampling_documented_source": (
                PATHTRACING_SUBFRAME_DOCUMENTATION_URL
            ),
        }
    )
    if readback_total != requested or readback_iteration != requested:
        raise RuntimeError("PathTracing sampling readback did not match request")


def _record_phase_timing(
    phase_timings: dict[str, Any], phase: str, elapsed_s: float
) -> None:
    """Accumulate finite monotonic phase timing without storing 4,806 samples."""

    if phase not in phase_timings:
        raise RuntimeError(f"undeclared render timing phase: {phase}")
    if not math.isfinite(float(elapsed_s)) or float(elapsed_s) < 0.0:
        raise RuntimeError(f"render timing phase is not finite/non-negative: {phase}")
    row = phase_timings[phase]
    if not isinstance(row, dict):
        raise RuntimeError(f"render timing phase record is malformed: {phase}")
    count = row.get("count")
    total = row.get("total")
    maximum = row.get("maximum")
    if (
        type(count) is not int
        or count < 0
        or not isinstance(total, (int, float))
        or isinstance(total, bool)
        or not math.isfinite(float(total))
        or float(total) < 0.0
        or not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
    ):
        raise RuntimeError(f"render timing phase record is malformed: {phase}")
    elapsed = float(elapsed_s)
    row.update(
        {
            "count": count + 1,
            "total": float(total) + elapsed,
            "maximum": max(float(maximum), elapsed),
        }
    )


def _set_render_product_updates(
    render_product: Any,
    enabled: bool,
    gating: dict[str, Any],
    *,
    phase: str,
) -> bool:
    """Set and read back one Hydra render-product update state transition."""

    if type(enabled) is not bool or not isinstance(phase, str) or not phase:
        raise RuntimeError("render-product update request is malformed")
    hydra_texture = getattr(render_product, "hydra_texture", None)
    get_enabled = getattr(hydra_texture, "get_updates_enabled", None)
    set_enabled = getattr(hydra_texture, "set_updates_enabled", None)
    if not callable(get_enabled) or not callable(set_enabled):
        raise RuntimeError("render product does not expose Hydra update controls")
    before = get_enabled()
    if type(before) is not bool:
        raise RuntimeError("render-product update readback is not boolean")
    started = time.monotonic()
    set_enabled(enabled)
    elapsed = time.monotonic() - started
    after = get_enabled()
    if type(after) is not bool:
        raise RuntimeError("render-product update readback is not boolean")
    transition = {
        "phase": phase,
        "requested_updates_enabled": enabled,
        "updates_enabled_before": before,
        "updates_enabled_after": after,
        "readback_matches_request": after is enabled,
        "set_and_readback_seconds": float(elapsed),
    }
    gating.setdefault("transitions", []).append(transition)
    count_key = "enable_request_count" if enabled else "disable_request_count"
    gating[count_key] = int(gating.get(count_key, 0)) + 1
    gating["current_updates_enabled"] = after
    if after is not enabled:
        raise RuntimeError(
            f"render-product update readback differs from request during {phase}"
        )
    return after


def _run_explicit_render_pass(
    render_product: Any,
    gating: dict[str, Any],
    phase_timings: dict[str, Any],
    *,
    phase: str,
    timing_phase: str,
    operation: Any,
    prepare: Any = None,
) -> Any:
    """Enable, prepare temporal state, render, and always disable afterwards."""

    if prepare is not None and not callable(prepare):
        raise RuntimeError("explicit render preparation must be callable")
    if not callable(operation):
        raise RuntimeError("explicit render operation must be callable")
    _set_render_product_updates(
        render_product,
        True,
        gating,
        phase=f"{phase}:enable",
    )
    gating["explicit_render_pass_count"] = int(
        gating.get("explicit_render_pass_count", 0)
    ) + 1
    started = time.monotonic()
    passed = False
    try:
        if prepare is not None:
            prepare()
            gating["renderer_accumulation_reset_count"] = int(
                gating.get("renderer_accumulation_reset_count", 0)
            ) + 1
        result = operation()
        passed = True
        return result
    finally:
        elapsed = time.monotonic() - started
        try:
            _record_phase_timing(phase_timings, timing_phase, elapsed)
        finally:
            _set_render_product_updates(
                render_product,
                False,
                gating,
                phase=f"{phase}:disable",
            )
            gating.setdefault("explicit_render_passes", []).append(
                {
                    "phase": phase,
                    "timing_phase": timing_phase,
                    "elapsed_seconds": float(elapsed),
                    "passed": passed,
                    "disabled_after": (
                        gating.get("current_updates_enabled") is False
                    ),
                }
            )


def _run_persistent_render_pass(
    render_product: Any,
    gating: dict[str, Any],
    phase_timings: dict[str, Any],
    *,
    phase: str,
    timing_phase: str,
    operation: Any,
    prepare: Any = None,
) -> Any:
    """Render once while proving the one persistent product remains enabled."""

    if prepare is not None and not callable(prepare):
        raise RuntimeError("persistent render preparation must be callable")
    if not callable(operation):
        raise RuntimeError("persistent render operation must be callable")
    hydra_texture = getattr(render_product, "hydra_texture", None)
    get_enabled = getattr(hydra_texture, "get_updates_enabled", None)
    if not callable(get_enabled):
        raise RuntimeError("render product does not expose Hydra update readback")
    enabled_before = get_enabled()
    if enabled_before is not True or gating.get("current_updates_enabled") is not True:
        raise RuntimeError("persistent render product was not enabled before capture")
    gating["explicit_render_pass_count"] = int(
        gating.get("explicit_render_pass_count", 0)
    ) + 1
    started = time.monotonic()
    passed = False
    reset_before = False
    try:
        if prepare is not None:
            prepare()
            reset_before = True
            gating["renderer_accumulation_reset_count"] = int(
                gating.get("renderer_accumulation_reset_count", 0)
            ) + 1
        result = operation()
        passed = True
        return result
    finally:
        elapsed = time.monotonic() - started
        _record_phase_timing(phase_timings, timing_phase, elapsed)
        enabled_after = get_enabled()
        gating["current_updates_enabled"] = enabled_after
        gating.setdefault("explicit_render_passes", []).append(
            {
                "phase": phase,
                "timing_phase": timing_phase,
                "elapsed_seconds": float(elapsed),
                "renderer_accumulation_reset_before": reset_before,
                "enabled_before": enabled_before,
                "enabled_after": enabled_after,
                "passed": bool(passed and enabled_after is True),
            }
        )
        if enabled_after is not True:
            raise RuntimeError(
                "persistent render product became disabled during explicit capture"
            )


def _rotate_render_product_generation(
    *,
    current_render_product: Any,
    annotators: tuple[Any, ...],
    rep: Any,
    create_render_product: Any,
    flush_destroyed_resources: Any,
    memory_sample: Any,
    gating: dict[str, Any],
    completed_frame_count: int,
) -> Any:
    """Destroy one retained RTX generation and recreate the same render product."""

    if (
        type(completed_frame_count) is not int
        or completed_frame_count <= 0
        or completed_frame_count % RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES != 0
    ):
        raise RuntimeError("render-product rotation frame boundary is invalid")
    if not callable(create_render_product):
        raise RuntimeError("render-product recreation must be callable")
    if not callable(flush_destroyed_resources):
        raise RuntimeError("render-product destruction flush must be callable")
    if not callable(memory_sample):
        raise RuntimeError("render-product memory sampler must be callable")
    if (
        gating.get("current_updates_enabled") is not False
        or current_render_product.hydra_texture.get_updates_enabled() is not False
    ):
        raise RuntimeError("render product must be disabled before rotation")
    rotations = gating.get("rotations")
    rotation_count = gating.get("rotation_count")
    if (
        not isinstance(rotations, list)
        or type(rotation_count) is not int
        or rotation_count != len(rotations)
    ):
        raise RuntimeError("render-product rotation receipt is malformed")
    if rotations and rotations[-1].get("completed_frame_count") == completed_frame_count:
        raise RuntimeError("render product was already rotated at this frame boundary")

    generation_before = rotation_count
    record: dict[str, Any] = {
        "completed_frame_count": completed_frame_count,
        "generation_before": generation_before,
        "generation_after": generation_before + 1,
        "wait_until_complete": False,
        "annotator_count": len(annotators),
        "detached_annotator_count": 0,
        "destroy_called": False,
        "flush_invariants": None,
        "memory_before_destroy_mib": None,
        "memory_after_destroy_mib": None,
        "memory_released_mib": None,
        "measurable_memory_release": False,
        "new_product_disabled_readback": False,
        "reattached_annotator_count": 0,
        "passed": False,
    }
    rotations.append(record)
    gating["rotation_count"] = generation_before + 1

    rep.orchestrator.wait_until_complete()
    record["wait_until_complete"] = True
    memory_before = memory_sample()
    record["memory_before_destroy_mib"] = memory_before
    for annotator in annotators:
        annotator.detach([current_render_product])
        record["detached_annotator_count"] += 1
    current_render_product.destroy()
    record["destroy_called"] = True
    flush_invariants = flush_destroyed_resources()
    record["flush_invariants"] = flush_invariants
    if (
        not isinstance(flush_invariants, Mapping)
        or flush_invariants.get("passed") is not True
    ):
        raise RuntimeError("render-product destruction flush changed simulation state")
    memory_after = memory_sample()
    record["memory_after_destroy_mib"] = memory_after
    if (
        type(memory_before) is not int
        or type(memory_after) is not int
        or memory_before < 0
        or memory_after < 0
    ):
        raise RuntimeError("render-product rotation memory evidence is invalid")
    released = memory_before - memory_after
    record["memory_released_mib"] = released
    record["measurable_memory_release"] = released > 0
    if released <= 0:
        raise RuntimeError("render-product destruction did not release GPU memory")

    replacement = create_render_product()
    disabled = _set_render_product_updates(
        replacement,
        False,
        gating,
        phase=f"rotation_generation_{generation_before + 1:02d}_after_create",
    )
    record["new_product_disabled_readback"] = disabled is False
    for annotator in annotators:
        annotator.attach([replacement])
        record["reattached_annotator_count"] += 1
    record["passed"] = bool(
        record["wait_until_complete"]
        and record["detached_annotator_count"] == len(annotators)
        and record["destroy_called"]
        and record["measurable_memory_release"]
        and record["new_product_disabled_readback"]
        and record["reattached_annotator_count"] == len(annotators)
    )
    if record["passed"] is not True:
        raise RuntimeError("render-product rotation did not complete")
    return replacement


def _enable_syntheticdata_volume_capture(
    settings: Any, receipt: dict[str, Any]
) -> bool:
    """Opt supported liquid render prims into synthetic-data rays and preserve readback."""

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
    rep: Any,
    *,
    rt_subframes: int,
    callback_count: Any,
    timeline: Any,
    physics_state: Any = None,
) -> dict[str, object]:
    """Capture output without permitting physics or timeline clocks to step."""

    if physics_state is not None and not callable(physics_state):
        raise RuntimeError("same-state capture physics-state reader is unavailable")
    before = int(callback_count())
    before_time = float(timeline.get_current_time())
    was_playing = bool(timeline.is_playing())
    before_physics = dict(physics_state()) if physics_state is not None else None
    _replicator_same_state_step(rep, rt_subframes=rt_subframes)
    after = int(callback_count())
    after_physics = dict(physics_state()) if physics_state is not None else None
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
    evidence: dict[str, object] = {
        "callbacks_before": before,
        "callbacks_after": after,
        "timeline_time_before_s": before_time,
        "timeline_time_after_s": float(timeline.get_current_time()),
        "timeline_playing_before": was_playing,
        "timeline_playing_after": bool(timeline.is_playing()),
    }
    if before_physics is not None and after_physics is not None:
        required_physics_fields = {"num_physics_steps", "simulation_time_s"}
        if (
            set(before_physics) != required_physics_fields
            or set(after_physics) != required_physics_fields
        ):
            raise RuntimeError(
                "same-state capture physics-state evidence is malformed"
            )
        if int(after_physics["num_physics_steps"]) != int(
            before_physics["num_physics_steps"]
        ):
            raise RuntimeError(
                "same-state Replicator capture advanced direct physics steps"
            )
        if not math.isclose(
            float(after_physics["simulation_time_s"]),
            float(before_physics["simulation_time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "same-state Replicator capture advanced simulation time"
            )
        evidence.update(
            {
                "physics_steps_before": int(
                    before_physics["num_physics_steps"]
                ),
                "physics_steps_after": int(
                    after_physics["num_physics_steps"]
                ),
                "simulation_time_before_s": float(
                    before_physics["simulation_time_s"]
                ),
                "simulation_time_after_s": float(
                    after_physics["simulation_time_s"]
                ),
            }
        )
    return evidence


def _replicator_same_state_step_checked_in_session(
    *,
    stage: Any,
    session_layer: Any,
    Usd: Any,
    rep: Any,
    rt_subframes: int,
    callback_count: Any,
    timeline: Any,
    physics_state: Any = None,
) -> dict[str, object]:
    """Capture frozen output while keeping PhysX writeback off the root layer."""

    return _with_session_edit_context(
        Usd,
        stage,
        session_layer,
        lambda: _replicator_same_state_step_checked(
            rep,
            rt_subframes=rt_subframes,
            callback_count=callback_count,
            timeline=timeline,
            physics_state=physics_state,
        ),
    )


def _refresh_stage_isosurface_lifecycle(
    *,
    timeline: Any,
    callback_count: Any,
    state_snapshot: Any,
    set_isosurface_enabled: Any,
    read_isosurface_enabled: Any,
    app_update: Any,
    source_evidence: Any,
    ensure_proxy: Any,
    proxy_created: Any,
    physics_step: int,
    phase: str,
) -> dict[str, object]:
    """Regenerate a deferred isosurface without advancing particle physics.

    NVIDIA's installed PhysX 110.4 tests create a non-empty particle set before
    enabling its isosurface and wait two Kit updates because initial creation is
    deferred.  They also exercise disable/update/enable regeneration with one
    update after each toggle.  F2 emits the first particle set after playback
    starts, so this bounded stage refresh applies the documented two-update
    initial-creation wait after re-enable as an explicit adaptation.  It may
    create an exact render proxy only if callback count, timeline time, stable
    particle IDs and the exact particle-position hash remain unchanged.
    """

    def observed_state() -> dict[str, object]:
        raw = state_snapshot()
        if not isinstance(raw, dict):
            raise TypeError("stage isosurface state snapshot must be a dictionary")
        return dict(raw)

    before_callbacks = int(callback_count())
    before_time = float(timeline.get_current_time())
    before_playing = bool(timeline.is_playing())
    before_state = observed_state()
    failures: list[str] = []
    error: str | None = None
    after_disable_callbacks: int | None = None
    after_disable_time: float | None = None
    after_disable_state: dict[str, object] | None = None
    after_enable_callbacks: int | None = None
    after_enable_time: float | None = None
    after_enable_state: dict[str, object] | None = None
    enabled_before = bool(read_isosurface_enabled())
    enabled_after_disable: bool | None = None
    enabled_after_enable: bool | None = None
    disable_update_calls = 0
    disable_updates_completed = 0
    reenable_update_calls = 0
    reenable_updates_completed = 0
    evidence: dict[str, object] = {}

    if not before_playing:
        failures.append("timeline_not_playing_before_refresh")
    if not enabled_before:
        failures.append("isosurface_not_enabled_before_refresh")

    try:
        timeline.pause()
        timeline.commit()
        set_isosurface_enabled(False)
        enabled_after_disable = bool(read_isosurface_enabled())
        disable_update_calls += 1
        app_update()
        disable_updates_completed += 1
        after_disable_callbacks = int(callback_count())
        after_disable_time = float(timeline.get_current_time())
        after_disable_state = observed_state()

        set_isosurface_enabled(True)
        enabled_after_enable = bool(read_isosurface_enabled())
        for _ in range(2):
            reenable_update_calls += 1
            app_update()
            reenable_updates_completed += 1
        after_enable_callbacks = int(callback_count())
        after_enable_time = float(timeline.get_current_time())
        after_enable_state = observed_state()
        raw_evidence = source_evidence()
        if isinstance(raw_evidence, dict):
            evidence = dict(raw_evidence)
        else:
            failures.append("source_isosurface_evidence_not_mapping")
    except Exception as exc:  # preserve a failed refresh as structured evidence
        error = f"{type(exc).__name__}: {exc}"
        failures.append("lifecycle_refresh_exception")
    finally:
        if before_playing and not bool(timeline.is_playing()):
            timeline.set_looping(False)
            timeline.play(looping=False)
            timeline.commit()
        elif not before_playing and bool(timeline.is_playing()):
            timeline.pause()
            timeline.commit()

    after_callbacks = int(callback_count())
    after_time = float(timeline.get_current_time())
    after_playing = bool(timeline.is_playing())
    after_state = observed_state()

    for label, callbacks, timeline_time, particle_state in (
        (
            "after_disable",
            after_disable_callbacks,
            after_disable_time,
            after_disable_state,
        ),
        (
            "after_enable",
            after_enable_callbacks,
            after_enable_time,
            after_enable_state,
        ),
        ("after_restore", after_callbacks, after_time, after_state),
    ):
        if callbacks is None or timeline_time is None or particle_state is None:
            failures.append(
                f"{label.removeprefix('after_')}_observation_missing"
            )
            continue
        if callbacks != before_callbacks:
            failures.append(f"callbacks_changed_{label}")
        if not math.isclose(
            timeline_time, before_time, rel_tol=0.0, abs_tol=1.0e-12
        ):
            failures.append(f"timeline_time_changed_{label}")
        if particle_state != before_state:
            failures.append(f"particle_state_changed_{label}")
    if enabled_after_disable is not False:
        failures.append("isosurface_disable_readback_failed")
    if enabled_after_enable is not True:
        failures.append("isosurface_enable_readback_failed")
    if after_playing != before_playing:
        failures.append("timeline_playback_not_restored")
    if evidence.get("mesh_geometry_valid") is not True:
        failures.append("substantive_isosurface_not_observed")

    # Keep one stable occurrence of each predicate in first-observed order.
    failures = list(dict.fromkeys(failures))
    if not failures:
        try:
            ensure_proxy(physics_step)
        except Exception as exc:  # keep proxy failures in the hashed gate record
            error = f"{type(exc).__name__}: {exc}"
            failures.append("exact_proxy_creation_exception")
    created = bool(proxy_created())
    if not created:
        failures.append("exact_proxy_not_created")
    failures = list(dict.fromkeys(failures))
    return {
        "physics_step": int(physics_step),
        "phase": str(phase),
        "attempt": 1,
        "purpose": "stage_isosurface_lifecycle_refresh",
        "documentation_classification": (
            "installed_nvidia_test_source_plus_bounded_adaptation"
        ),
        "documentation_reference": (
            "omni.physx.tests-110.4.4+110.4.0/omni/physxtests/tests/"
            "PhysxParticleIsosurfaceAPI.py: particle set precedes API apply; "
            "initial creation waits two updates; disable and re-enable toggles "
            "each wait one update"
        ),
        "documented_initial_creation_update_count": 2,
        "documented_disable_toggle_update_count": 1,
        "documented_reenable_toggle_update_count": 1,
        "experiment_disable_update_count": 1,
        "experiment_reenable_update_count": 2,
        "disable_app_update_calls": disable_update_calls,
        "disable_app_updates_completed": disable_updates_completed,
        "reenable_app_update_calls": reenable_update_calls,
        "reenable_app_updates_completed": reenable_updates_completed,
        "callbacks_before": before_callbacks,
        "callbacks_after_disable": after_disable_callbacks,
        "callbacks_after_enable": after_enable_callbacks,
        "callbacks_after": after_callbacks,
        "timeline_time_before_s": before_time,
        "timeline_time_after_disable_s": after_disable_time,
        "timeline_time_after_enable_s": after_enable_time,
        "timeline_time_after_s": after_time,
        "timeline_playing_before": before_playing,
        "timeline_playing_after": after_playing,
        "isosurface_enabled_readback_before": enabled_before,
        "isosurface_enabled_readback_after_disable": enabled_after_disable,
        "isosurface_enabled_readback_after_enable": enabled_after_enable,
        "particle_state_before": before_state,
        "particle_state_after_disable": after_disable_state,
        "particle_state_after_enable": after_enable_state,
        "particle_state_after_restore": after_state,
        "source_isosurface_evidence_after_refresh": evidence,
        "proxy_created_after_refresh": created,
        "failed_predicates": failures,
        "error": error,
        "passed": not failures,
    }


def _app_update_in_session(stage: Any, session_layer: Any, app: Any, Usd: Any) -> None:
    """Run Kit work in the session target so PhysX USD writeback cannot hit root."""

    with Usd.EditContext(stage, session_layer):
        app.update()


def _timeline_state_readback(stage: Any, timeline: Any) -> dict[str, object]:
    """Return the stage and main-timeline range/state without changing either."""

    return {
        "stage_start_time_code": float(stage.GetStartTimeCode()),
        "stage_end_time_code": float(stage.GetEndTimeCode()),
        "stage_time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
        "timeline_start_time_s": float(timeline.get_start_time()),
        "timeline_end_time_s": float(timeline.get_end_time()),
        "timeline_current_time_s": float(timeline.get_current_time()),
        "timeline_time_codes_per_second": float(
            timeline.get_time_codes_per_second()
        ),
        "timeline_looping": bool(timeline.is_looping()),
        "timeline_playing": bool(timeline.is_playing()),
    }


def _configure_zero_origin_timeline(
    stage: Any,
    session_layer: Any,
    app: Any,
    timeline: Any,
    Usd: Any,
    *,
    physics_hz: int,
    final_callback_index: int,
) -> dict[str, object]:
    """Configure and verify the zero-origin fixed-step timeline in session state."""

    if (
        isinstance(physics_hz, bool)
        or not isinstance(physics_hz, int)
        or physics_hz <= 0
        or isinstance(final_callback_index, bool)
        or not isinstance(final_callback_index, int)
        or final_callback_index <= 0
    ):
        raise ValueError("physics_hz and final_callback_index must be positive integers")
    before = _timeline_state_readback(stage, timeline)
    required_end_time_s = (
        final_callback_index + TIMELINE_END_MARGIN_CALLBACKS
    ) / float(physics_hz)
    with Usd.EditContext(stage, session_layer):
        timeline.stop()
        timeline.set_time_codes_per_second(float(physics_hz))
        # The source twin begins at time code 1. Kit clamps a requested current
        # time below the stage start, so move the session start before asking
        # for the zero-origin callback trace exercised by A18.
        timeline.set_start_time(0.0)
        timeline.set_end_time(required_end_time_s)
        timeline.set_current_time(0.0)
        timeline.set_ticks_per_frame(1)
        timeline.set_target_framerate(float(physics_hz))
        timeline.set_play_every_frame(True)
        timeline.set_looping(False)
        timeline.commit()
    _app_update_in_session(stage, session_layer, app, Usd)
    observed = _timeline_state_readback(stage, timeline)
    expected = {
        "stage_start_time_code": 0.0,
        "stage_end_time_code": float(
            final_callback_index + TIMELINE_END_MARGIN_CALLBACKS
        ),
        "stage_time_codes_per_second": float(physics_hz),
        "timeline_start_time_s": 0.0,
        "timeline_end_time_s": required_end_time_s,
        "timeline_current_time_s": 0.0,
        "timeline_time_codes_per_second": float(physics_hz),
        "timeline_looping": False,
        "timeline_playing": False,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if (
            observed.get(key) is not value
            if isinstance(value, bool)
            else not math.isclose(
                float(observed[key]), value, rel_tol=0.0, abs_tol=1.0e-9
            )
        )
    ]
    if mismatches:
        raise RuntimeError(
            "timeline zero-origin readback mismatch: "
            + json.dumps(
                {
                    "actual": observed,
                    "expected": expected,
                    "failed_predicates": mismatches,
                },
                sort_keys=True,
            )
        )
    return {
        "required_final_callback_index": final_callback_index,
        "end_margin_callbacks": TIMELINE_END_MARGIN_CALLBACKS,
        "required_end_time_s": required_end_time_s,
        "before_configuration": before,
        "after_configuration": observed,
    }


def _start_timeline_non_looping(
    stage: Any,
    session_layer: Any,
    timeline: Any,
    Usd: Any,
    *,
    configured: Mapping[str, object],
) -> dict[str, object]:
    """Start the configured timeline without relying on ``play`` defaults."""

    configured_state = configured.get("after_configuration")
    if not isinstance(configured_state, Mapping):
        raise ValueError("configured timeline evidence lacks after_configuration")
    expected = dict(configured_state)
    expected["timeline_looping"] = False
    expected["timeline_playing"] = True
    with Usd.EditContext(stage, session_layer):
        # Kit's play() default is looping=True.  Set and pass False explicitly so
        # neither stored state nor the call default can wrap the fixed-step run.
        timeline.set_looping(False)
        timeline.play(looping=False)
        timeline.commit()
    observed = _timeline_state_readback(stage, timeline)
    mismatches = [
        key
        for key, value in expected.items()
        if (
            observed.get(key) is not value
            if isinstance(value, bool)
            else not math.isclose(
                float(observed[key]), float(value), rel_tol=0.0, abs_tol=1.0e-9
            )
        )
    ]
    if mismatches:
        raise RuntimeError(
            "timeline playback readback mismatch: "
            + json.dumps(
                {
                    "actual": observed,
                    "expected": expected,
                    "failed_predicates": mismatches,
                },
                sort_keys=True,
            )
        )
    return observed


def _timeline_callback_contract_evidence(
    callback: dict[str, Any],
    *,
    global_step: int,
    app_update_index: int,
    physics_hz: int,
) -> dict[str, Any]:
    """Return evidence for one timeline-owned fixed-step Kit update."""

    actual = {
        "callback_index": int(callback["callback_index"]),
        "dt_s": float(callback["dt_s"]),
        "timeline_time_s": float(callback["timeline_time_s"]),
        "app_update_index": int(callback["app_update_index"]),
    }
    expected = {
        "callback_index": int(global_step),
        "dt_s": 1.0 / float(physics_hz),
        "timeline_time_s": global_step / float(physics_hz),
        "app_update_index": int(app_update_index),
    }
    failed_predicates: list[str] = []
    if not math.isclose(
        actual["dt_s"], expected["dt_s"], rel_tol=0.0, abs_tol=1.0e-6
    ):
        failed_predicates.append("dt_s")
    if not math.isclose(
        actual["timeline_time_s"],
        expected["timeline_time_s"],
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        failed_predicates.append("timeline_time_s")
    if actual["app_update_index"] != expected["app_update_index"]:
        failed_predicates.append("app_update_index")
    if actual["callback_index"] != expected["callback_index"]:
        failed_predicates.append("callback_index")
    return {
        "actual_callback": actual,
        "expected": expected,
        "failed_predicates": failed_predicates,
        "passed": not failed_predicates,
    }


def _timeline_callback_group_contract_evidence(
    callbacks: list[dict[str, Any]],
    *,
    report_index: int,
    app_update_index: int,
    physics_substeps: int,
    callback_count_before: int,
    physics_steps_before: int,
    physics_steps_after: int,
    simulation_time_before_s: float,
    simulation_time_after_s: float,
    timeline_time_before_s: float,
    timeline_time_after_s: float,
) -> dict[str, Any]:
    """Fail-closed evidence for all PhysX callbacks in one report interval."""

    clock = _build_physics_clock_contract(
        physics_substeps=physics_substeps,
        report_interval_count=report_index,
    )
    expected_first_callback = (report_index - 1) * physics_substeps + 1
    expected_callback_indices = list(
        range(
            expected_first_callback,
            expected_first_callback + physics_substeps,
        )
    )
    expected_app_update_indices = [app_update_index] * physics_substeps
    expected_physics_dt_s = float(clock["physics_dt_s"])
    expected_timeline_time_before_s = (report_index - 1) / float(REPORT_HZ)
    expected_timeline_time_after_s = report_index / float(REPORT_HZ)

    callback_indices: list[object] = []
    callback_dts_s: list[object] = []
    callback_timeline_samples_s: list[object] = []
    callback_app_update_indices: list[object] = []
    for callback in callbacks:
        if not isinstance(callback, dict):
            callback_indices.append(None)
            callback_dts_s.append(None)
            callback_timeline_samples_s.append(None)
            callback_app_update_indices.append(None)
            continue
        callback_indices.append(callback.get("callback_index"))
        callback_dts_s.append(callback.get("dt_s"))
        callback_timeline_samples_s.append(callback.get("timeline_time_s"))
        callback_app_update_indices.append(callback.get("app_update_index"))

    actual = {
        "report_index": report_index,
        "app_update_index": app_update_index,
        "callback_count_before": callback_count_before,
        "callback_count_after": callback_count_before + len(callbacks),
        "callback_count_delta": len(callbacks),
        "callback_indices": callback_indices,
        "callback_dts_s": callback_dts_s,
        "callback_timeline_samples_s": callback_timeline_samples_s,
        "callback_app_update_indices": callback_app_update_indices,
        "physics_steps_before": physics_steps_before,
        "physics_steps_after": physics_steps_after,
        "physics_step_count_delta": physics_steps_after - physics_steps_before,
        "simulation_time_before_s": simulation_time_before_s,
        "simulation_time_after_s": simulation_time_after_s,
        "simulation_time_delta_s": (
            simulation_time_after_s - simulation_time_before_s
        ),
        "timeline_time_before_s": timeline_time_before_s,
        "timeline_time_after_s": timeline_time_after_s,
    }
    expected = {
        "report_index": report_index,
        "app_update_index": app_update_index,
        "callback_count_before": expected_first_callback - 1,
        "callback_count_delta": physics_substeps,
        "callback_indices": expected_callback_indices,
        "app_update_indices": expected_app_update_indices,
        "callback_dt_s": expected_physics_dt_s,
        "logical_physics_times_s": [
            index / float(clock["physics_steps_per_second"])
            for index in expected_callback_indices
        ],
        "physics_step_count_delta": physics_substeps,
        "simulation_time_delta_s": 1.0 / float(REPORT_HZ),
        "timeline_time_before_s": expected_timeline_time_before_s,
        "timeline_time_after_s": expected_timeline_time_after_s,
    }
    failed_predicates: list[str] = []
    if len(callbacks) != physics_substeps:
        failed_predicates.append("callback_count_delta")
    if callback_count_before != expected_first_callback - 1:
        failed_predicates.append("callback_count_before")
    if callback_indices != expected_callback_indices:
        failed_predicates.append("callback_indices")
    if callback_app_update_indices != expected_app_update_indices:
        failed_predicates.append("app_update_indices")
    if len(callback_dts_s) != physics_substeps or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isclose(
            float(value), expected_physics_dt_s, rel_tol=0.0, abs_tol=1.0e-6
        )
        for value in callback_dts_s
    ):
        failed_predicates.append("callback_dts_s")
    finite_timeline_samples = bool(
        len(callback_timeline_samples_s) == physics_substeps
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in callback_timeline_samples_s
        )
    )
    if not finite_timeline_samples:
        failed_predicates.append("callback_timeline_samples_finite")
    elif any(
        float(current) < float(previous) - 1.0e-12
        for previous, current in zip(
            callback_timeline_samples_s,
            callback_timeline_samples_s[1:],
        )
    ):
        failed_predicates.append("callback_timeline_samples_nondecreasing")
    if physics_steps_after - physics_steps_before != physics_substeps:
        failed_predicates.append("physics_step_count_delta")
    if not math.isclose(
        simulation_time_after_s - simulation_time_before_s,
        1.0 / float(REPORT_HZ),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        failed_predicates.append("simulation_time_delta_s")
    if not math.isclose(
        timeline_time_before_s,
        expected_timeline_time_before_s,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        failed_predicates.append("timeline_time_before_s")
    if not math.isclose(
        timeline_time_after_s,
        expected_timeline_time_after_s,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        failed_predicates.append("timeline_time_after_s")
    return {
        "actual": actual,
        "expected": expected,
        "failed_predicates": failed_predicates,
        "passed": not failed_predicates,
    }


def _direct_step_callback_contract_evidence(
    callback: dict[str, Any],
    *,
    global_step: int,
    physics_hz: int,
    paused_timeline_time_s: float,
) -> dict[str, Any]:
    """Return named evidence for one direct ``SimulationManager.step`` callback."""

    actual = {
        "callback_index": int(callback["callback_index"]),
        "dt_s": float(callback["dt_s"]),
        "timeline_time_s": float(callback["timeline_time_s"]),
        "direct_step_index": int(callback["direct_step_index"]),
    }
    expected = {
        "callback_index": int(global_step),
        "dt_s": 1.0 / float(physics_hz),
        "logical_time_s": global_step / float(physics_hz),
        "timeline_time_s": float(paused_timeline_time_s),
        "direct_step_index": int(global_step),
    }
    failed_predicates = []
    if not math.isclose(
        actual["dt_s"], expected["dt_s"], rel_tol=0.0, abs_tol=1.0e-6
    ):
        failed_predicates.append("dt_s")
    if not math.isclose(
        actual["timeline_time_s"],
        expected["timeline_time_s"],
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        failed_predicates.append("timeline_time_s")
    if actual["direct_step_index"] != expected["direct_step_index"]:
        failed_predicates.append("direct_step_index")
    if actual["callback_index"] != expected["callback_index"]:
        failed_predicates.append("callback_index")
    return {
        "actual_callback": actual,
        "expected": expected,
        "failed_predicates": failed_predicates,
        "passed": not failed_predicates,
    }


def _direct_physics_step_checked(
    *,
    SimulationManager: Any,
    callback_records: list[dict[str, Any]],
    direct_step_index: int,
    physics_hz: int,
    timeline: Any,
    paused_timeline_time_s: float,
) -> dict[str, Any]:
    """Advance physics once and fail closed on owner, clock, or callback drift."""

    callback_count_before = len(callback_records)
    physics_steps_before = int(SimulationManager.get_num_physics_steps())
    simulation_time_before = float(SimulationManager.get_simulation_time())
    timeline_time_before = float(timeline.get_current_time())
    timeline_playing_before = bool(timeline.is_playing())

    SimulationManager.step(steps=1, update_fabric=False)

    callback_count_after = len(callback_records)
    physics_steps_after = int(SimulationManager.get_num_physics_steps())
    simulation_time_after = float(SimulationManager.get_simulation_time())
    timeline_time_after = float(timeline.get_current_time())
    timeline_playing_after = bool(timeline.is_playing())
    callback_delta = callback_count_after - callback_count_before
    callback_contract = (
        _direct_step_callback_contract_evidence(
            callback_records[-1],
            global_step=direct_step_index,
            physics_hz=physics_hz,
            paused_timeline_time_s=paused_timeline_time_s,
        )
        if callback_delta == 1
        else None
    )
    expected_dt = 1.0 / float(physics_hz)
    simulation_time_delta = simulation_time_after - simulation_time_before
    failed_predicates: list[str] = []
    if callback_delta != 1:
        failed_predicates.append("callback_count_delta")
    if physics_steps_after - physics_steps_before != 1:
        failed_predicates.append("physics_step_count_delta")
    if not math.isclose(
        simulation_time_delta, expected_dt, rel_tol=0.0, abs_tol=1.0e-6
    ):
        failed_predicates.append("simulation_time_delta_s")
    if not math.isclose(
        timeline_time_before,
        paused_timeline_time_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        timeline_time_after,
        paused_timeline_time_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failed_predicates.append("paused_timeline_time_s")
    if timeline_playing_before or timeline_playing_after:
        failed_predicates.append("timeline_playing")
    if callback_contract is None or callback_contract["passed"] is not True:
        failed_predicates.append("callback_contract")
    evidence = {
        "direct_step_index": int(direct_step_index),
        "logical_time_s": direct_step_index / float(physics_hz),
        "callback_count_before": callback_count_before,
        "callback_count_after": callback_count_after,
        "callback_count_delta": callback_delta,
        "physics_step_count_before": physics_steps_before,
        "physics_step_count_after": physics_steps_after,
        "physics_step_count_delta": physics_steps_after - physics_steps_before,
        "simulation_time_before_s": simulation_time_before,
        "simulation_time_after_s": simulation_time_after,
        "simulation_time_delta_s": simulation_time_delta,
        "timeline_time_before_s": timeline_time_before,
        "timeline_time_after_s": timeline_time_after,
        "timeline_playing_before": timeline_playing_before,
        "timeline_playing_after": timeline_playing_after,
        "callback_contract": callback_contract,
        "failed_predicates": failed_predicates,
        "passed": not failed_predicates,
    }
    if failed_predicates:
        raise RuntimeError(
            "direct physics step differs from the frozen 120 Hz contract: "
            + ",".join(failed_predicates)
        )
    return evidence


def _direct_particle_vec3_array(np: Any, rows: Any) -> Any:
    """Return finite particle vectors with an explicit ``(N, 3)`` shape."""

    try:
        values = np.asarray(rows, dtype=np.float64).reshape((-1, 3))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("direct particle state is not a Vec3 array") from exc
    if not np.all(np.isfinite(values)):
        raise RuntimeError("direct particle state contains non-finite values")
    return values


def _runtime_particle_motion_evidence(
    *,
    initial_positions: Any,
    current_positions: Any,
    current_velocities: Any,
    stable_particle_count: int,
    displacement_tolerance_m: float = 1.0e-6,
    velocity_tolerance_m_s: float = 1.0e-6,
) -> dict[str, Any]:
    """Falsify callback-only progress using stable emitted-particle rows.

    The first emitted rows are stable by authored USD id.  A real gravity
    update must move at least one of those rows downward and change its
    velocity away from the authored nozzle velocity ``[0, 0, -0.1]``.
    """

    import numpy as np

    initial = _direct_particle_vec3_array(np, initial_positions)
    current = _direct_particle_vec3_array(np, current_positions)
    velocities = _direct_particle_vec3_array(np, current_velocities)
    if (
        isinstance(stable_particle_count, bool)
        or not isinstance(stable_particle_count, int)
        or stable_particle_count <= 0
        or stable_particle_count > len(initial)
        or stable_particle_count > len(current)
        or stable_particle_count > len(velocities)
    ):
        raise RuntimeError("motion falsifier stable-particle count is invalid")
    stable_initial = initial[:stable_particle_count]
    stable_current = current[:stable_particle_count]
    stable_velocities = velocities[:stable_particle_count]
    downward = stable_initial[:, 2] - stable_current[:, 2]
    authored_velocity = np.asarray([0.0, 0.0, -0.1], dtype=np.float64)
    velocity_delta = np.linalg.norm(
        stable_velocities - authored_velocity[None, :], axis=1
    )
    moved_downward = downward > float(displacement_tolerance_m)
    changed_velocity = velocity_delta > float(velocity_tolerance_m_s)
    same_particle_proves_motion = moved_downward & changed_velocity
    failed_predicates: list[str] = []
    if not np.any(moved_downward):
        failed_predicates.append("stable_particle_downward_displacement")
    if not np.any(changed_velocity):
        failed_predicates.append(
            "stable_particle_velocity_changed_from_authored"
        )
    if not failed_predicates and not np.any(same_particle_proves_motion):
        failed_predicates.append("same_stable_particle_proves_motion")
    return {
        "stable_particle_count": stable_particle_count,
        "displacement_tolerance_m": float(displacement_tolerance_m),
        "velocity_tolerance_m_s": float(velocity_tolerance_m_s),
        "max_downward_displacement_m": float(max(0.0, np.max(downward))),
        "max_velocity_delta_m_s": float(np.max(velocity_delta)),
        "moving_particle_count": int(np.count_nonzero(moved_downward)),
        "changed_velocity_particle_count": int(
            np.count_nonzero(changed_velocity)
        ),
        "same_particle_motion_count": int(
            np.count_nonzero(same_particle_proves_motion)
        ),
        "failed_predicates": failed_predicates,
        "passed": not failed_predicates,
    }


def _render_stream_surface_z_m(
    particle_positions: Any,
    *,
    count_driven_body: bool,
    count_driven_body_geometry: Mapping[str, Any] | None,
    render_radius_m: float = FLUID_REST_OFFSET_M,
) -> float:
    """Resolve the render-only stream bottom from its visible liquid body."""

    if count_driven_body:
        if count_driven_body_geometry is None:
            return SOURCE_LIQUID_BOTTOM_M
        try:
            bottom = float(count_driven_body_geometry["world_bottom_z_m"])
            height = float(count_driven_body_geometry["world_height_m"])
            extent = count_driven_body_geometry["world_extent_m"]
            extent_top = float(extent[1][2])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "count-driven body lacks a finite volume-derived surface"
            ) from exc
        expected_top = bottom + height
        if (
            not all(math.isfinite(value) for value in (bottom, height, extent_top))
            or height < 0.0
            or not math.isclose(
                extent_top, expected_top, rel_tol=0.0, abs_tol=2.0e-6
            )
        ):
            raise RuntimeError(
                "count-driven body surface conflicts with its world geometry"
            )
        return min(max(extent_top, SOURCE_LIQUID_BOTTOM_M), VIAL_MOUTH_M)
    return robust_contained_surface_z_m(
        particle_positions,
        bottom_z_m=SOURCE_LIQUID_BOTTOM_M,
        mouth_z_m=VIAL_MOUTH_M,
        render_radius_m=render_radius_m,
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


def _twin_chunk_positions(
    Gf: Any,
    count: int,
    rng: random.Random,
    *,
    resolution_profile_id: str = LEGACY_RESOLUTION_PROFILE_ID,
    reference_particle_count: int | None = None,
    attestation_sink: list[dict[str, Any]] | None = None,
) -> list[Any]:
    if resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
        if reference_particle_count is None:
            raise RuntimeError("C21 emission event lacks its C20 reference count")
        sample = build_resolution_aware_source_sample(
            particle_count=count,
            reference_particle_count=reference_particle_count,
            resolution_profile_id=resolution_profile_id,
            source_velocity_m_s=(0.0, 0.0, -0.1),
        )
        if attestation_sink is not None:
            attestation_sink.append(deepcopy(sample["attestation"]))
        top_z = min(NOZZLE_EXIT_CENTER_M[2] - 0.003, VIAL_MOUTH_M - 0.001)
        return [
            Gf.Vec3f(
                NOZZLE_EXIT_CENTER_M[0] + float(offset[0]),
                NOZZLE_EXIT_CENTER_M[1] + float(offset[1]),
                top_z + float(offset[2]),
            )
            for offset in sample["local_positions_m"]
        ]
    values = []
    spacing = float(RUNTIME_PARTICLE_CONFIGURATION["spawn_spacing_m"])
    for index in range(count):
        ix = index % 3
        iy = (index // 3) % 3
        iz = index // 9
        values.append(
            Gf.Vec3f(
                NOZZLE_EXIT_CENTER_M[0]
                + (ix - 1) * spacing
                + (rng.random() - 0.5) * spacing * 0.1,
                NOZZLE_EXIT_CENTER_M[1]
                + (iy - 1) * spacing
                + (rng.random() - 0.5) * spacing * 0.1,
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


def _q2_runtime_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Return the historical Q2 plan or a count-only packing diagnostic."""

    plan = _runtime_paint_ball_plan(args)
    requested_count = int(args.q2_particle_count)
    if requested_count == int(plan["pours"][0]["particle_count"]):
        return plan
    plan = deepcopy(plan)
    first = plan["pours"][0]
    events = first["emission_schedule"]
    counts = distribute_particle_count(requested_count, len(events))
    first["emission_schedule"] = [
        {**dict(event), "particle_count": count}
        for event, count in zip(events, counts, strict=True)
    ]
    first["particle_count"] = requested_count
    first["cumulative_particle_count"] = requested_count
    first["nominal_mass_convention_equivalent_ml"] = (
        requested_count * NOMINAL_PARTICLE_VOLUME_ML
    )
    first["nominal_error_ml"] = (
        requested_count * NOMINAL_PARTICLE_VOLUME_ML - 10.0
    )
    first_stage = plan["fallback_render_model"]["stage_states"][0]
    first_stage.update(
        {
            "cumulative_particle_counts": [requested_count, 0, 0],
            "cumulative_particle_count": requested_count,
            "component_fractions": [1.0, 0.0, 0.0],
        }
    )
    plan["q2_particle_count_override"] = {
        "requested_particle_count": requested_count,
        "historical_particle_count": 640,
        "programmed_volume_ml": 10.0,
        "particle_radius_unchanged": True,
        "diagnostic_only": True,
    }
    return plan


def _empty_q2_no_render_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Return the dedicated, render-free Q2 receipt envelope."""

    plan = _q2_runtime_plan(args)
    script_path = Path(__file__).resolve()
    contract_path = ROOT / "src" / "fluidmix" / "physx_particle_contract.py"
    emission_end_step = int(
        plan["paint_ball_emitter"]["qualification_first_stage_step"]
    )
    settle_steps = int(args.q2_post_emission_settle_steps)
    target_step = emission_end_step + settle_steps
    q2_runtime_configuration = dict(RUNTIME_PARTICLE_CONFIGURATION)
    q2_runtime_configuration["solver_position_iterations"] = int(
        args.q2_solver_position_iterations
    )
    q2_runtime_configuration["max_particles"] = max(
        int(q2_runtime_configuration["max_particles"]),
        int(args.q2_particle_count),
    )
    expected_particle_count = int(args.q2_particle_count)
    expected_total_mass_kg = expected_particle_count * 0.000015625
    return {
        "schema_version": 1,
        "artifact_type": "q2_no_render_qualification",
        "experiment": "physx_three_pour_f2_q2_no_render",
        "status": "insufficient_evidence",
        "exit_status": None,
        "arguments": {
            "stage": str(args.stage),
            "seed": int(args.seed),
            "q2_no_render": True,
            "wall_limit_seconds": int(args.wall_limit_seconds),
            "q2_post_emission_settle_steps": settle_steps,
            "q2_solver_position_iterations": int(
                args.q2_solver_position_iterations
            ),
            "q2_particle_count": expected_particle_count,
        },
        "target": {
            "programmed_volume_ml": 10.0,
            "physics_step": target_step,
            "emission_end_step": emission_end_step,
            "post_emission_settle_steps": settle_steps,
            "post_emission_settle_s": settle_steps / float(PHYSICS_HZ),
            "expected_particle_count": expected_particle_count,
            "expected_stable_ids": [0, expected_particle_count - 1],
            "stream_hidden": True,
            "rendering_disabled": True,
            "height_curve_sha256": volume_height_curve_digest(),
            "height_bounds_m": list(height_bounds_for_volume_ml(10.0)),
        },
        "plan": plan,
        "ablation": {
            "diagnostic_only": True,
            "production_defaults_modified": False,
            "emission_end_step": emission_end_step,
            "analysis_step": target_step,
            "post_emission_settle_steps": settle_steps,
            "post_emission_settle_s": settle_steps / float(PHYSICS_HZ),
            "solver_position_iterations": int(
                args.q2_solver_position_iterations
            ),
            **(
                {"particle_count": expected_particle_count}
                if expected_particle_count != 640
                else {}
            ),
        },
        "render": {
            "enabled": False,
            "capture_enabled": False,
            "render_product_created": False,
            "rgb_artifacts": [],
            "semantic_artifacts": [],
            "video_artifacts": [],
        },
        "runtime": {
            "versions": {"isaac": None, "kit": None, "physx": None},
            "source_layer_records": [],
            "one_particle_system": False,
            "one_point_instancer": False,
            "particle_system_count": 0,
            "point_instancer_count": 0,
            "point_instancer_path": f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared",
            "particle_system_path": f"{FLUIDMIX_NAMESPACE}/ParticleSystem",
            "runtime_configuration_requested": q2_runtime_configuration,
            "max_particles": int(q2_runtime_configuration["max_particles"]),
            "max_neighborhood": int(q2_runtime_configuration["max_neighborhood"]),
            "density_kg_m3": float(q2_runtime_configuration["density_kg_m3"]),
            "fluid_rest_offset_m": float(q2_runtime_configuration["fluid_rest_offset_m"]),
            "solver_position_iterations_requested": int(
                q2_runtime_configuration["solver_position_iterations"]
            ),
            "solver_position_iterations_readback": None,
            "scene_min_position_iterations_readback": None,
            "scene_max_position_iterations_readback": None,
            "mass_api_count": None,
            "source_visibility": {},
            "camera_transform_untouched": None,
            "vial_transform_untouched": None,
            "nozzle_transform_untouched": None,
            "scene_transform_untouched": None,
        },
        "hashes": {
            "runner_script_sha256": _sha256(script_path),
            "particle_contract_sha256": _sha256(contract_path),
            "stage_sha256": None,
            "source_state_sha256": None,
            "runtime_versions_sha256": None,
        },
        "arrays": {
            "saved_before_assertions": False,
            "snapshot": None,
            "failure_snapshot": None,
        },
        "evidence": {
            "particle_identity": {
                "expected_count": expected_particle_count,
                "observed_count": None,
                "stable_unique": False,
                "ids_exact": False,
                "ids_sha256": None,
                "identity_checks": 0,
                "identity_mismatches": 0,
            },
            "mass": {
                "density_kg_m3": float(q2_runtime_configuration["density_kg_m3"]),
                "nominal_particle_mass_kg": 0.000015625,
                "expected_total_mass_kg": expected_total_mass_kg,
                "observed_total_mass_kg": None,
                "density_derived": True,
                "mass_api_present": None,
                "mass_api_count": None,
                "passed": False,
            },
            "arrays_finite": {"passed": False, "fields": []},
            "containment": {
                "emitted_particle_count": None,
                "inside_vial_interior_count": None,
                "outside_particle_count": None,
                "escaped_particle_count": None,
                "expected_in_flight_stream_count": None,
                "finite_state": False,
                "passed": False,
            },
            "neighbor_cap": {
                "physx_max_neighborhood": int(q2_runtime_configuration["max_neighborhood"]),
                "max_observed_diffusion_degree": None,
                "cap_saturated": None,
                "unsaturated": None,
                "measurement_source": "runtime_diffusion_graph_not_physx_neighbor_readback",
                "gate_required": False,
                "passed": None,
                "status": "pending_diagnostic",
            },
            "height": {
                "geometry_curve_sha256": volume_height_curve_digest(),
                "programmed_volume_ml": 10.0,
                "expected_height_m": height_for_volume_ml(10.0),
                "lower_bound_m": height_bounds_for_volume_ml(10.0)[0],
                "upper_bound_m": height_bounds_for_volume_ml(10.0)[1],
                "contained_z_min_m": None,
                "contained_z_q05_m": None,
                "contained_z_q95_m": None,
                "contained_z_max_m": None,
                "particle_bulk_height_m": None,
                "passed": False,
            },
            "source_consistency": {
                "source_states": [],
                "all_rows_match_source_state": False,
                "all_events_match_plan": False,
                "passed": False,
            },
            "timeline": {
                "target_step": target_step,
                "callbacks_observed": 0,
                "callback_contract_passed": False,
                "timeline_looping": None,
                "timeline_playing": None,
            },
        },
        "failure_artifacts": [],
        "errors": [],
        "warnings": [],
    }


def run_q2_no_render(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    """Run the first 10 mL shared PointInstancer graph without rendering."""

    import traceback as _traceback

    receipt = _empty_q2_no_render_receipt(args)
    q2_runtime_configuration = dict(
        receipt["runtime"]["runtime_configuration_requested"]
    )
    started = time.monotonic()
    deadline = started + int(args.wall_limit_seconds)
    timeline = None
    physics_subscription = None
    settings = None
    particles_writeback_setting: str | None = None
    particles_writeback_initial: bool | None = None
    velocity_writeback_setting: str | None = None
    velocity_writeback_initial: bool | None = None
    fabric_initial: bool | None = None
    stage = None
    session_layer = None
    failure_phase = "import_runtime"
    try:
        import carb
        import omni.physx
        from omni.physx.scripts import particleUtils
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
        from omni.physx.bindings import _physx as physx_bindings
        import numpy as np

        settings = carb.settings.get_settings()
        receipt["runtime"]["versions"] = _version_info(app)
        receipt["hashes"]["runtime_versions_sha256"] = _canonical_json_sha256(
            receipt["runtime"]["versions"]
        )
        manager = app.get_extension_manager()
        if not manager.is_extension_enabled("omni.physx"):
            raise RuntimeError("omni.physx extension is not enabled")
        foundation = get_physx_foundation_interface()
        receipt["runtime"]["cuda_device_available"] = bool(
            foundation.cuda_device_check()
        )
        if not receipt["runtime"]["cuda_device_available"]:
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
        receipt["runtime"]["source_layer_records"] = layer_records
        receipt["hashes"]["stage_sha256"] = _canonical_json_sha256(layer_records)
        default_prim = stage.GetDefaultPrim()
        if (
            not default_prim.IsValid()
            or str(default_prim.GetPath()) != "/FillStation"
            or float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
            or str(UsdGeom.GetStageUpAxis(stage)) != "Z"
        ):
            raise RuntimeError("opened stage is not the reviewed metre-scale Z-up FillStation twin")

        timeline = omni.timeline.get_timeline_interface()
        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0.0:
            raise RuntimeError("twin timeCodesPerSecond is invalid")
        timeline.stop()
        timeline.set_current_time(TWIN_REFERENCE_TIME_CODE / time_codes_per_second)
        app.update()
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"reviewed FillCam missing: {CAMERA_PATH}")
        camera = UsdGeom.Camera(camera_prim)
        clipping = camera.GetClippingRangeAttr().Get(Usd.TimeCode(TWIN_REFERENCE_TIME_CODE))
        if not camera.GetClippingRangeAttr().HasAuthoredValueOpinion() or any(
            not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-9)
            for value, expected in zip(clipping, CAMERA_CLIPPING_RANGE)
        ):
            raise RuntimeError("FillCam authored clipping range differs from source contract")
        receipt["runtime"]["camera_transform_untouched"] = True
        receipt["runtime"]["vial_transform_untouched"] = True
        receipt["runtime"]["nozzle_transform_untouched"] = True
        receipt["runtime"]["scene_transform_untouched"] = True

        plan = receipt["plan"]
        target_step = int(receipt["target"]["physics_step"])
        source_states_by_ordinal = _build_pour_source_states(plan)
        source_state_rows = [
            {
                "pour_ordinal": ordinal,
                "scheduled_tank": state.scheduled_tank,
                "composition": list(state.composition),
                "birth_composition": list(state.birth_composition),
                "palette_prototype_index": state.palette_prototype_index,
                "material_path": state.material_path,
                "pour_event_id": state.pour_event_id,
                "birth_event_id": state.birth_event_id,
                "birth_step": state.birth_step,
            }
            for ordinal, state in sorted(source_states_by_ordinal.items())
        ]
        receipt["evidence"]["source_consistency"]["source_states"] = source_state_rows
        receipt["hashes"]["source_state_sha256"] = _canonical_json_sha256(source_state_rows)
        event_index = _q2_first_pour_event_index(plan)
        events = [
            event
            for step, rows in event_index.items()
            if int(step) <= target_step
            for event in rows
        ]
        expected_events = [
            event
            for pour in plan["pours"][:1]
            for event in pour["emission_schedule"]
            if int(event["physics_step"]) <= target_step
        ]
        if len(events) != len(expected_events):
            raise RuntimeError("Q2 event schedule does not contain exactly the first 10 mL")

        session_layer = stage.GetSessionLayer()
        failure_phase = "author_runtime_settings"
        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_writeback_setting = getattr(
            physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
        )
        velocity_writeback_setting = getattr(
            physx_bindings, "SETTING_UPDATE_VELOCITIES_TO_USD", None
        )
        if particles_writeback_setting is None or velocity_writeback_setting is None:
            raise RuntimeError("PhysX particle USD writeback settings are unavailable")
        particles_writeback_initial = bool(settings.get_as_bool(particles_writeback_setting))
        velocity_writeback_initial = bool(settings.get_as_bool(velocity_writeback_setting))
        settings.set_bool(particles_writeback_setting, True)
        settings.set_bool(velocity_writeback_setting, True)
        if not settings.get_as_bool(particles_writeback_setting) or not settings.get_as_bool(velocity_writeback_setting):
            raise RuntimeError("PhysX particle position/velocity USD writeback did not enable")
        fabric_initial = bool(SimulationManager.is_fabric_enabled())
        SimulationManager.enable_fabric(False)
        if SimulationManager.is_fabric_enabled():
            raise RuntimeError("Fabric must be disabled for runtime USD particle append")

        failure_phase = "author_particle_system"
        with Usd.EditContext(stage, session_layer):
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                prim = stage.GetPrimAtPath(hidden_path)
                if not prim.IsValid():
                    raise RuntimeError(f"source prim missing: {hidden_path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdGeom.Xform.Define(stage, FLUIDMIX_NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{FLUIDMIX_NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            scene_api.CreateSolverTypeAttr().Set("TGS")
            scene_min_iterations = scene_api.GetMinPositionIterationCountAttr().Get()
            scene_max_iterations = scene_api.GetMaxPositionIterationCountAttr().Get()
            if (
                isinstance(scene_min_iterations, bool)
                or not isinstance(scene_min_iterations, int)
                or isinstance(scene_max_iterations, bool)
                or not isinstance(scene_max_iterations, int)
            ):
                raise RuntimeError(
                    "PhysX scene position-iteration bounds are unavailable"
                )
            requested_iterations = int(
                q2_runtime_configuration["solver_position_iterations"]
            )
            if not scene_min_iterations <= requested_iterations <= scene_max_iterations:
                raise RuntimeError(
                    "requested particle solver iterations fall outside PhysX scene bounds"
                )
            system_path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateEnableCCDAttr().Set(True)
            system.CreateParticleContactOffsetAttr().Set(q2_runtime_configuration["particle_contact_offset_m"])
            system.CreateFluidRestOffsetAttr().Set(q2_runtime_configuration["fluid_rest_offset_m"])
            system.CreateContactOffsetAttr().Set(q2_runtime_configuration["contact_offset_m"])
            system.CreateRestOffsetAttr().Set(q2_runtime_configuration["rest_offset_m"])
            system.CreateSolidRestOffsetAttr().Set(q2_runtime_configuration["solid_rest_offset_m"])
            system.CreateSolverPositionIterationCountAttr().Set(requested_iterations)
            system.CreateMaxNeighborhoodAttr().Set(q2_runtime_configuration["max_neighborhood"])
            system.CreateMaxVelocityAttr().Set(PAINT_BALL_MAX_VELOCITY_M_S)
            particleUtils.add_physx_particle_smoothing(stage, system_path, enabled=True, strength=1.0)
            particleUtils.add_physx_particle_anisotropy(stage, system_path, enabled=True, scale=1.0)
            physics_material = UsdShade.Material.Define(stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial")
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(physics_material.GetPrim())
            _author_pbd_material_treatment(
                pbd=pbd,
                material_prim=physics_material.GetPrim(),
                particle_utils=particleUtils,
                use_water_preset=_water_pbd_preset_requested(args),
            )
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
                physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
            instancer_path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared")
            instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
            instancer_prim = instancer.GetPrim()
            point_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(instancer_prim)
            particle_api = PhysxSchema.PhysxParticleAPI(point_set_api)
            particle_api.CreateSelfCollisionAttr().Set(True)
            point_set_api.CreateFluidAttr().Set(True)
            particle_api.CreateParticleGroupAttr().Set(0)
            particle_api.CreateParticleSystemRel().SetTargets([system_path])
            capacity = _author_and_attest_particle_capacity(
                instancer_prim,
                value_type=Sdf.ValueTypeNames.Int,
                required_max_particles=int(q2_runtime_configuration["max_particles"]),
                required_value_source="fluidmix.physx_particle_contract.required_particle_capacity",
            )
            prototype_path = instancer_path.AppendChild("particlePrototype0")
            prototype = UsdGeom.Sphere.Define(stage, prototype_path)
            prototype.CreateRadiusAttr().Set(FLUID_REST_OFFSET_M)
            instancer.GetPrototypesRel().AddTarget(prototype_path)
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray([]))
            instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray([]))
            instancer.GetOrientationsAttr().Set(Vt.QuathArray([]))
            instancer.GetScalesAttr().Set(Vt.Vec3fArray([]))
            point_set_api.GetSimulationPointsAttr().Set(Vt.Vec3fArray([]))
            instancer.CreateIdsAttr().Set(Vt.Int64Array([]))
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([]))
            collider_prims = []
            collider_collision_readbacks = []
            for spec in build_twin_collider_specs():
                cube = UsdGeom.Cube.Define(stage, Sdf.Path(str(spec["path"])))
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                collider_collision_readbacks.append(
                    _author_collider_collision_offsets(
                        PhysxSchema, cube.GetPrim(), spec
                    )
                )
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())

        receipt["runtime"].update(
            {
                "particle_system_count": sum(
                    1 for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)
                ),
                "point_instancer_count": sum(
                    1
                    for prim in stage.Traverse()
                    if prim.IsA(UsdGeom.PointInstancer)
                    and str(prim.GetPath()) == str(instancer_path)
                ),
                "particle_capacity": capacity,
                "mass_api_count": int(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
                "mass_api_present": bool(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
                "collider_count": len(collider_prims),
                "collider_paths": [str(prim.GetPath()) for prim in collider_prims],
                "collider_collision_readbacks": collider_collision_readbacks,
                "solver_position_iterations_readback": int(
                    system.GetSolverPositionIterationCountAttr().Get()
                ),
                "scene_min_position_iterations_readback": int(
                    scene_min_iterations
                ),
                "scene_max_position_iterations_readback": int(
                    scene_max_iterations
                ),
            }
        )
        if (
            receipt["runtime"]["solver_position_iterations_readback"]
            != receipt["runtime"]["solver_position_iterations_requested"]
        ):
            raise RuntimeError(
                "particle solver-position iteration readback differs from request"
            )
        receipt["runtime"]["one_particle_system"] = (
            receipt["runtime"]["particle_system_count"] == 1
        )
        receipt["runtime"]["one_point_instancer"] = (
            receipt["runtime"]["point_instancer_count"] == 1
        )
        if receipt["runtime"]["mass_api_present"]:
            raise RuntimeError("Q2 PointInstancer unexpectedly has MassAPI")
        receipt["runtime"]["source_visibility"] = {
            hidden_path: {
                str(code): str(
                    UsdGeom.Imageable(stage.GetPrimAtPath(hidden_path))
                    .GetVisibilityAttr()
                    .Get(Usd.TimeCode(code))
                )
                for code in VISIBILITY_CHECK_TIME_CODES
            }
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH)
        }
        if any(
            value != "invisible"
            for row in receipt["runtime"]["source_visibility"].values()
            for value in row.values()
        ):
            raise RuntimeError("source visibility did not remain hidden")

        scene_api = PhysxSchema.PhysxSceneAPI(scene.GetPrim())
        with Usd.EditContext(stage, session_layer):
            scene_api.GetTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(physx_bindings.SETTING_MIN_FRAME_RATE, PHYSICS_HZ)
        timeline_initialization = _configure_zero_origin_timeline(
            stage,
            session_layer,
            app,
            timeline,
            Usd,
            physics_hz=PHYSICS_HZ,
            final_callback_index=target_step,
        )
        _start_timeline_non_looping(
            stage,
            session_layer,
            timeline,
            Usd,
            configured=timeline_initialization,
        )
        callback_records: list[dict[str, Any]] = []
        app_update_index = 0

        def on_physics_step(callback_dt: float) -> None:
            callback_records.append(
                {
                    "callback_index": len(callback_records) + 1,
                    "dt_s": float(callback_dt),
                    "timeline_time_s": float(timeline.get_current_time()),
                    "app_update_index": app_update_index,
                }
            )

        physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_physics_step
        )
        points = _PointInstancerParticleAdapter(instancer, point_set_api)
        shared_ids: list[int] = []
        next_id = 0
        compositions_by_id: dict[int, Any] = {}
        active_records: list[dict[str, Any]] = []
        source_consistency = True
        actual_events: list[dict[str, Any]] = []
        component_counts = np.zeros(len(COMPOSITION_COMPONENTS), dtype=np.float64)
        rng = random.Random(args.seed)
        max_degree = 0
        identity_checks = 0
        identity_mismatches = 0
        failure_phase = "run_q2_particle_timeline"
        while len(callback_records) < target_step:
            global_step = len(callback_records) + 1
            if time.monotonic() > deadline:
                raise RuntimeError("Q2 wall-time safety stop reached")
            for event in event_index.get(global_step, []):
                count = int(event["particle_count"])
                particle_ids = list(range(next_id, next_id + count))
                source_state = source_states_by_ordinal[int(event["pour_ordinal"])]
                prepared = _prepare_emission_event(event, component_counts=component_counts)
                source_consistency = source_consistency and (
                    source_state.scheduled_tank == event["tank_id"]
                    and np.allclose(
                        prepared["particle_composition"],
                        source_state.birth_composition,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and int(prepared["prototype_index"]) == source_state.palette_prototype_index
                )
                positions = _twin_chunk_positions(Gf, count, rng)
                existing_positions = list(points.GetPositionsAttr().Get() or [])
                existing_simulation = list(points.GetSimulationPointsAttr().Get() or [])
                existing_velocities = list(points.GetVelocitiesAttr().Get() or [])
                existing_orientations = list(points.GetOrientationsAttr().Get() or [])
                existing_scales = list(points.GetScalesAttr().Get() or [])
                existing_ids = [int(value) for value in (points.GetIdsAttr().Get() or [])]
                if existing_ids != shared_ids:
                    raise RuntimeError("shared particle rows lost stable identity before append")
                with Usd.EditContext(stage, session_layer):
                    points.GetPositionsAttr().Set(Vt.Vec3fArray([*existing_positions, *positions]))
                    points.GetSimulationPointsAttr().Set(Vt.Vec3fArray([*existing_simulation, *positions]))
                    points.GetVelocitiesAttr().Set(
                        Vt.Vec3fArray([*existing_velocities, *[Gf.Vec3f(0.0, 0.0, -0.10)] * count])
                    )
                    points.GetOrientationsAttr().Set(
                        Vt.QuathArray([*existing_orientations, *[Gf.Quath(1.0, 0.0, 0.0, 0.0)] * count])
                    )
                    points.GetScalesAttr().Set(
                        Vt.Vec3fArray([*existing_scales, *[Gf.Vec3f(1.0, 1.0, 1.0)] * count])
                    )
                    points.CreateIdsAttr().Set(Vt.Int64Array([*existing_ids, *particle_ids]))
                    points.GetProtoIndicesAttr().Set(
                        Vt.IntArray([*list(points.GetProtoIndicesAttr().Get() or []), *[0] * count])
                    )
                initial_positions = [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in _positions(points.GetPrim(), instancer=True)[-count:]
                ]
                active_records.append(
                    {
                        "particle_ids": particle_ids,
                        "particle_count": count,
                        "source_compositions": prepared["source_compositions"],
                        "scheduled_source": source_state.scheduled_tank,
                        "birth_composition": list(source_state.birth_composition),
                        "pour_event_id": source_state.pour_event_id,
                        "birth_event_id": f"{source_state.birth_event_id}-{int(event['ordinal']):03d}",
                        "pour_ordinal": int(event["pour_ordinal"]),
                        "event": dict(event),
                        "initial_positions": initial_positions,
                    }
                )
                shared_ids.extend(particle_ids)
                next_id += count
                for particle_id, composition in zip(particle_ids, prepared["source_compositions"]):
                    compositions_by_id[int(particle_id)] = np.asarray(composition, dtype=np.float64).copy()
                component_counts = prepared["next_component_counts"]
                actual_events.append(
                    {
                        "physics_step": global_step,
                        "pour_ordinal": int(event["pour_ordinal"]),
                        "particle_count": count,
                        "particle_id_range": [particle_ids[0], particle_ids[-1] + 1],
                        "source": source_state.scheduled_tank,
                        "composition": list(source_state.birth_composition),
                    }
                )
            before = len(callback_records)
            app_update_index += 1
            _app_update_in_session(stage, session_layer, app, Usd)
            if len(callback_records) != before + 1:
                raise RuntimeError("timeline produced not exactly one callback per app update")
            callback = callback_records[-1]
            callback_contract = _timeline_callback_contract_evidence(
                callback,
                global_step=global_step,
                app_update_index=app_update_index,
                physics_hz=PHYSICS_HZ,
            )
            if not callback_contract["passed"]:
                raise RuntimeError(
                    "Q2 timeline callback contract failed: "
                    + ",".join(callback_contract["failed_predicates"])
                )
            try:
                observed_ids = _read_runtime_particle_identity(points, shared_ids)
            except RuntimeError:
                identity_mismatches += 1
                raise
            identity_checks += 1
            positions_after = np.asarray(
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (points.GetSimulationPointsAttr().Get() or [])
                ],
                dtype=np.float64,
            )
            updated, diagnostics = _diffuse_active_particle_rows(
                active_records,
                compositions_by_id,
                positions_after,
                observed_ids,
                dt_s=1.0 / float(PHYSICS_HZ),
            )
            max_degree = max(max_degree, int(diagnostics.get("max_degree", 0)))
            with Usd.EditContext(stage, session_layer):
                points.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(observed_ids)))

        if len(callback_records) != target_step:
            raise RuntimeError("Q2 timeline did not reach its frozen analysis checkpoint")
        final_ids = _read_runtime_particle_identity(points, shared_ids)
        final_positions = np.asarray(
            [
                [float(value[0]), float(value[1]), float(value[2])]
                for value in (points.GetSimulationPointsAttr().Get() or [])
            ],
            dtype=np.float64,
        )
        final_velocities = np.asarray(
            [
                [float(value[0]), float(value[1]), float(value[2])]
                for value in (points.GetVelocitiesAttr().Get() or [])
            ],
            dtype=np.float64,
        )
        final_orientations = np.asarray(
            [
                [float(value.GetReal()), *[float(value) for value in value.GetImaginary()]]
                for value in (points.GetOrientationsAttr().Get() or [])
            ],
            dtype=np.float64,
        )
        final_scales = np.asarray(
            [
                [float(value[0]), float(value[1]), float(value[2])]
                for value in (points.GetScalesAttr().Get() or [])
            ],
            dtype=np.float64,
        )
        birth_by_id: dict[int, list[float]] = {}
        source_by_id: dict[int, str] = {}
        birth_step_by_id: dict[int, int] = {}
        birth_event_by_id: dict[int, str] = {}
        pour_event_by_id: dict[int, str] = {}
        for record in active_records:
            source_state = source_states_by_ordinal[int(record["pour_ordinal"])]
            for particle_id, composition in zip(record["particle_ids"], record["source_compositions"]):
                birth_by_id[int(particle_id)] = [float(value) for value in composition]
                source_by_id[int(particle_id)] = source_state.scheduled_tank
                birth_step_by_id[int(particle_id)] = int(record["event"]["physics_step"])
                birth_event_by_id[int(particle_id)] = str(record["birth_event_id"])
                pour_event_by_id[int(particle_id)] = source_state.pour_event_id
        snapshot_state = {
            "simulation_points": final_positions.tolist(),
            "positions": np.asarray(
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in _positions(points.GetPrim(), instancer=True)
                ],
                dtype=np.float64,
            ).tolist(),
            "velocities": final_velocities.tolist(),
            "orientations": final_orientations.tolist(),
            "scales": final_scales.tolist(),
            "ids": final_ids,
            "proto_indices": [int(value) for value in (points.GetProtoIndicesAttr().Get() or [])],
            "birth_compositions": [birth_by_id[value] for value in final_ids],
            "current_compositions": [compositions_by_id[value].tolist() for value in final_ids],
            "scheduled_sources": [source_by_id[value] for value in final_ids],
            "birth_steps": [birth_step_by_id[value] for value in final_ids],
            "birth_event_ids": [birth_event_by_id[value] for value in final_ids],
            "pour_event_ids": [pour_event_by_id[value] for value in final_ids],
        }
        snapshot = persist_runtime_particle_snapshot(
            output / "particle-snapshots",
            (
                f"q2-step-{target_step:05d}-t200-stage"
                if int(receipt["target"]["post_emission_settle_steps"]) == 0
                else f"q2-step-{target_step:05d}-t200-settled"
            ),
            snapshot_state,
            expected_ids=list(
                range(int(receipt["target"]["expected_particle_count"]))
            ),
            artifact_path_prefix="particle-snapshots",
        )
        receipt["arrays"].update({"saved_before_assertions": True, "snapshot": snapshot})
        receipt["evidence"]["arrays_finite"] = {
            "passed": all(row["finite"] for row in snapshot["arrays"].values()),
            "fields": sorted(snapshot["arrays"]),
        }

        lower_bounds = (
            VIAL_CENTER_XY_M[0] - SOURCE_LIQUID_RADIUS_M,
            VIAL_CENTER_XY_M[1] - SOURCE_LIQUID_RADIUS_M,
            SOURCE_LIQUID_BOTTOM_M,
        )
        upper_bounds = (
            VIAL_CENTER_XY_M[0] + SOURCE_LIQUID_RADIUS_M,
            VIAL_CENTER_XY_M[1] + SOURCE_LIQUID_RADIUS_M,
            VIAL_MOUTH_M,
        )
        inner_apothem = SOURCE_LIQUID_RADIUS_M * math.cos(math.pi / VIAL_COLLIDER_WALL_COUNT)
        containment = containment_evidence(
            final_positions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            expected_in_flight_mask=np.zeros(len(final_positions), dtype=bool),
            cylindrical_center_xy=VIAL_CENTER_XY_M,
            inner_radius_m=inner_apothem,
            polygon_center_xy=VIAL_CENTER_XY_M,
            polygon_apothem_m=inner_apothem,
            polygon_wall_count=VIAL_COLLIDER_WALL_COUNT,
        )
        inside_count = int(containment["inside_vial_interior_count"])
        outside_count = int(containment["outside_vial_interior_count"])
        escaped_count = int(containment["leaked_particle_count"])
        expected_particle_count = int(
            receipt["target"]["expected_particle_count"]
        )
        containment_passed = bool(
            len(final_positions) == expected_particle_count
            and inside_count == expected_particle_count
            and outside_count == 0
            and escaped_count == 0
            and containment["expected_in_flight_stream_count"] == 0
            and containment["finite_state"] is True
        )
        receipt["evidence"]["containment"] = {
            **containment,
            "emitted_particle_count": len(final_positions),
            "outside_particle_count": outside_count,
            "escaped_particle_count": escaped_count,
            "passed": containment_passed,
        }
        particle_ids_passed = (
            len(final_ids) == expected_particle_count
            and len(set(final_ids)) == expected_particle_count
            and final_ids == list(range(expected_particle_count))
        )
        receipt["evidence"]["particle_identity"].update(
            {
                "observed_count": len(final_ids),
                "stable_unique": len(final_ids) == len(set(final_ids)),
                "ids_exact": particle_ids_passed,
                "ids_sha256": _particle_ids_sha256(final_ids),
                "identity_checks": identity_checks,
                "identity_mismatches": identity_mismatches,
            }
        )
        mass_evidence = receipt["evidence"]["mass"]
        observed_total_mass = len(final_ids) * float(mass_evidence["nominal_particle_mass_kg"])
        mass_evidence.update(
            {
                "observed_total_mass_kg": observed_total_mass,
                "mass_api_present": bool(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
                "mass_api_count": int(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
                "passed": bool(
                    math.isclose(
                        observed_total_mass,
                        float(mass_evidence["expected_total_mass_kg"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and not instancer_prim.HasAPI(UsdPhysics.MassAPI)
                ),
            }
        )
        height = particle_bulk_height_from_quantiles_m(
            containment["contained_z_q05_m"],
            containment["contained_z_q95_m"],
            FLUID_REST_OFFSET_M,
        )
        expected_height = height_for_volume_ml(10.0)
        lower_height, upper_height = height_bounds_for_volume_ml(10.0)
        receipt["evidence"]["height"].update(
            {
                "contained_z_min_m": containment["contained_z_min_m"],
                "contained_z_q05_m": containment["contained_z_q05_m"],
                "contained_z_q95_m": containment["contained_z_q95_m"],
                "contained_z_max_m": containment["contained_z_max_m"],
                "particle_bulk_height_m": height,
                "passed": lower_height <= height <= upper_height,
            }
        )
        neighbor = _q2_diffusion_neighbor_diagnostic(
            max_degree=max_degree,
            physx_max_neighborhood=int(
                receipt["evidence"]["neighbor_cap"]["physx_max_neighborhood"]
            ),
        )
        receipt["evidence"]["neighbor_cap"] = neighbor
        source_rows_match = all(
            source_by_id.get(particle_id) == source_states_by_ordinal[int(record["pour_ordinal"])].scheduled_tank
            and np.allclose(
                np.asarray(record["source_compositions"], dtype=np.float64),
                np.asarray(
                    [source_states_by_ordinal[int(record["pour_ordinal"])].birth_composition] * int(record["particle_count"]),
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=1.0e-12,
            )
            for record in active_records
            for particle_id in record["particle_ids"]
        )
        receipt["evidence"]["source_consistency"].update(
            {
                "all_rows_match_source_state": bool(source_rows_match),
                "all_events_match_plan": bool(source_consistency and len(actual_events) == len(events)),
                "passed": bool(source_rows_match and source_consistency and len(actual_events) == len(events)),
            }
        )
        callback_passed = bool(
            len(callback_records) == target_step
            and all(
                int(row["callback_index"]) == index
                and math.isclose(float(row["dt_s"]), 1.0 / PHYSICS_HZ, rel_tol=0.0, abs_tol=1.0e-6)
                and math.isclose(float(row["timeline_time_s"]), index / PHYSICS_HZ, rel_tol=0.0, abs_tol=1.0e-6)
                for index, row in enumerate(callback_records, start=1)
            )
        )
        receipt["evidence"]["timeline"].update(
            {
                "callbacks_observed": len(callback_records),
                "callback_contract_passed": callback_passed,
                "timeline_looping": bool(timeline.is_looping()),
                "timeline_playing": bool(timeline.is_playing()),
            }
        )
        gate_values = [
            particle_ids_passed,
            bool(receipt["evidence"]["arrays_finite"]["passed"]),
            bool(mass_evidence["passed"]),
            containment_passed,
            bool(receipt["evidence"]["height"]["passed"]),
            bool(receipt["evidence"]["source_consistency"]["passed"]),
            callback_passed,
        ]
        receipt["status"] = "pass" if all(gate_values) else "insufficient_evidence"
        if receipt["status"] != "pass":
            failed_gates = [
                name
                for name, passed in zip(
                    (
                        "stable_unique_particles",
                        "finite_arrays",
                        "density_mass_no_mass_api",
                        "zero_outside_escaped",
                        "particle_height_bounds",
                        "source_consistency",
                        "timeline",
                    ),
                    gate_values,
                )
                if not passed
            ]
            gate_failure_path = output / "q2-gate-failure.json"
            _write_json(
                gate_failure_path,
                {
                    "artifact_type": "q2_gate_failure_artifact",
                    "failed_gates": failed_gates,
                    "arrays_saved_before_assertions": bool(
                        receipt["arrays"]["saved_before_assertions"]
                    ),
                },
            )
            receipt["failure_artifacts"].append(
                {
                    "kind": "gate_failure",
                    "path": str(gate_failure_path.relative_to(output)),
                    "sha256": _sha256(gate_failure_path),
                }
            )
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(f"Q2 execution failed: {type(exc).__name__}: {exc}")
        failure_path = output / "q2-failure.json"
        failure_payload = {
            "artifact_type": "q2_failure_artifact",
            "phase": failure_phase,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": _traceback.format_exc(),
            "arrays_saved_before_assertions": bool(receipt["arrays"]["saved_before_assertions"]),
        }
        _write_json(failure_path, failure_payload)
        receipt["failure_artifacts"].append(
            {
                "kind": "failure",
                "path": str(failure_path.relative_to(output)),
                "sha256": _sha256(failure_path),
            }
        )
    finally:
        if physics_subscription is not None:
            physics_subscription = None
        if timeline is not None:
            try:
                timeline.pause()
                timeline.commit()
            except Exception as exc:
                receipt["warnings"].append(f"timeline pause failed: {exc}")
        if fabric_initial is not None:
            try:
                SimulationManager.enable_fabric(fabric_initial)
                receipt["runtime"]["fabric_restored"] = bool(SimulationManager.is_fabric_enabled())
            except Exception as exc:
                receipt["warnings"].append(f"Fabric restore failed: {exc}")
        if settings is not None and particles_writeback_setting and particles_writeback_initial is not None:
            try:
                settings.set_bool(particles_writeback_setting, particles_writeback_initial)
                receipt["runtime"]["position_writeback_restored"] = bool(settings.get_as_bool(particles_writeback_setting))
            except Exception as exc:
                receipt["warnings"].append(f"position writeback restore failed: {exc}")
        if settings is not None and velocity_writeback_setting and velocity_writeback_initial is not None:
            try:
                settings.set_bool(velocity_writeback_setting, velocity_writeback_initial)
                receipt["runtime"]["velocity_writeback_restored"] = bool(settings.get_as_bool(velocity_writeback_setting))
            except Exception as exc:
                receipt["warnings"].append(f"velocity writeback restore failed: {exc}")
        receipt["evidence"]["timeline"]["wall_seconds"] = time.monotonic() - started
    return receipt


def run_sequence(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    """Run F2 on the existing fill-station twin; only liquid physics is added."""

    width, height = _validate_args(args)
    receipt = _empty_paint_ball_receipt(args)
    runtime_particle_configuration = dict(
        receipt["particle_system"]["runtime_configuration"]
    )
    render_radius_m = float(
        receipt["particle_resolution_profile"]["render_radius_m"]
    )
    timeline_owned_count_driven = bool(args.count_driven_twin_liquid_mesh)
    gating = receipt["render"]["render_product_update_gating"]
    phase_timings = receipt["render"]["phase_timings_s"]
    started = time.monotonic()
    deadline = started + args.wall_limit_seconds
    timeline = None
    physics_step_subscription = None
    parity_capture_subscription = None
    settings = None
    capture_volumes_initial: bool | None = None
    particles_writeback_setting: str | None = None
    particles_writeback_initial: bool | None = None
    velocity_writeback_setting: str | None = None
    velocity_writeback_initial: bool | None = None
    fabric_initial: bool | None = None
    render_product = None
    rgb_annotator = None
    semantic_annotator = None
    instance_annotator = None
    writer = None
    video_path: Path | None = None
    video_codec: str | None = None
    video_fallback: bool | None = None
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
    video_frame_steps: list[int] = []
    video_frame_rgb_hashes: list[str] = []
    callback_records: list[dict[str, Any]] = []
    timeline_callback_group_records: list[dict[str, Any]] = []
    capture_guard_records: list[dict[str, object]] = []
    count_driven_body_runtime: dict[str, Any] | None = None
    latest_count_driven_body_geometry: dict[str, Any] | None = None
    neuraylib = None
    registered_mdl_search_path = None
    active_direct_step_index = 0
    timeline_app_update_index = 0
    timeline_physics_manager_baseline: dict[str, int | float] | None = None
    physics_substeps = int(getattr(args, "physics_substeps", 1))
    failure_phase = "import_runtime"
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        from omni.physx.scripts import particleUtils
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, Vt
        try:
            from pxr import Semantics as LegacySemantics
        except ImportError:
            LegacySemantics = None

        settings = carb.settings.get_settings()
        capture_volumes_initial = _enable_syntheticdata_volume_capture(
            settings, receipt
        )
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            REPLICATOR_ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is still enabled")
        _configure_pathtracing_sampling(
            settings,
            receipt,
            requested=args.pathtracing_total_spp,
        )
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        manager = app.get_extension_manager()
        if args.height_banded_body:
            if not manager.is_extension_enabled("omni.mdl.neuraylib"):
                manager.set_extension_enabled_immediate(
                    "omni.mdl.neuraylib", True
                )
            if not manager.is_extension_enabled("omni.mdl.neuraylib"):
                raise RuntimeError("omni.mdl.neuraylib could not be enabled")
            import omni.mdl.neuraylib

            neuraylib = omni.mdl.neuraylib.get_neuraylib()
            registered_mdl_search_path = neuraylib.addMdlSearchPath(
                os.fspath(SPATIAL_OMNISURFACE_MDL.parent)
            )
            if not registered_mdl_search_path:
                raise RuntimeError("spatial OmniSurface MDL path registration failed")
            receipt["material"]["mdl_search_path_registered"] = bool(
                registered_mdl_search_path
            )
            receipt["material"]["mdl_search_path"] = os.fspath(
                SPATIAL_OMNISURFACE_MDL.parent
            )
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
        session_layer = stage.GetSessionLayer()
        receipt["scene_contract"]["runtime_edit_target"] = (
            _set_persistent_session_edit_target(stage, session_layer, Usd)
        )
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
        receipt["render"]["tone_exposure_state"] = (
            _pin_tone_exposure_state(settings, camera_prim, Sdf)
        )
        _set_rtx_render_mode(
            settings,
            receipt,
            mode="pt",
            phase="initial-pathtraced-parity-and-stills",
        )

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (width, height))
        gating["render_product_create_count"] = int(
            gating["render_product_create_count"]
        ) + 1
        if gating["render_product_create_count"] != 1:
            raise RuntimeError("F2 must create exactly one persistent render product")
        if timeline_owned_count_driven:
            initial_updates_enabled = _set_render_product_updates(
                render_product,
                False,
                gating,
                phase="initial_after_create",
            )
            gating["initial_disabled_readback"] = (
                initial_updates_enabled is False
            )
        else:
            hydra_texture = getattr(render_product, "hydra_texture", None)
            get_updates_enabled = getattr(
                hydra_texture, "get_updates_enabled", None
            )
            if not callable(get_updates_enabled):
                raise RuntimeError(
                    "render product does not expose Hydra update readback"
                )
            initial_updates_enabled = get_updates_enabled()
            if initial_updates_enabled is not True:
                raise RuntimeError(
                    "persistent render product was not enabled after creation"
                )
            gating["initial_enabled_readback"] = True
            gating["current_updates_enabled"] = True
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        _configure_replicator_capture(rep)

        parity_dir = output / "pre-pour-parity"
        frames_dir = output / "frames"
        stages_dir = output / "stages"
        semantic_dir = output / "semantic"
        instance_dir = output / "instance"
        instance_mapping_dir = output / "instance-mapping"
        liquid_masks_dir = output / "liquid"
        semantic_controls_dir = output / "semantic-controls"
        stage_states_dir = output / "stage-states"
        stream_states_dir = output / "stream-states"
        trace_dir = output / "blend"
        parity_dir.mkdir()
        frames_dir.mkdir()
        stages_dir.mkdir()
        semantic_dir.mkdir()
        instance_dir.mkdir()
        instance_mapping_dir.mkdir()
        liquid_masks_dir.mkdir()
        semantic_controls_dir.mkdir()
        stage_states_dir.mkdir()
        stream_states_dir.mkdir()
        trace_dir.mkdir()
        def run_render_warmup() -> None:
            for _ in range(args.warmup):
                app.update()

        def read_direct_physics_state() -> dict[str, float | int]:
            return {
                "num_physics_steps": int(
                    SimulationManager.get_num_physics_steps()
                ),
                "simulation_time_s": float(
                    SimulationManager.get_simulation_time()
                ),
            }

        capture_render_pass = (
            _run_explicit_render_pass
            if timeline_owned_count_driven
            else _run_persistent_render_pass
        )
        capture_physics_state = (
            None if timeline_owned_count_driven else read_direct_physics_state
        )

        capture_render_pass(
            render_product,
            gating,
            phase_timings,
            phase="warmup",
            timing_phase="warmup_app_update",
            operation=run_render_warmup,
        )

        def check_time_limits() -> None:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("F2 wall-time safety stop reached")

        def acquire_rgb(
            step: int,
            phase: str,
            *,
            included_in_video: bool,
            rt_subframes_override: int | None = None,
        ) -> Any:
            invalid_reasons: list[str] = []
            rgb = None
            last_rejected_rgb = None
            for _attempt in range(1, args.max_capture_tries + 1):
                check_time_limits()
                guard_record = capture_render_pass(
                    render_product,
                    gating,
                    phase_timings,
                    phase="single_rgb",
                    timing_phase="capture_rgb_orchestrator",
                    prepare=lambda: _reset_renderer_accumulation_if_supported(
                        context
                    ),
                    operation=lambda: _replicator_same_state_step_checked_in_session(
                        stage=stage,
                        session_layer=session_layer,
                        Usd=Usd,
                        rep=rep,
                        rt_subframes=(
                            args.rt_subframes
                            if rt_subframes_override is None
                            else rt_subframes_override
                        ),
                        callback_count=lambda: len(callback_records),
                        timeline=timeline,
                        physics_state=capture_physics_state,
                    ),
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

        def acquire_paired_frame(
            step: int,
            phase: str,
            visible_stream_tanks: tuple[str, ...],
        ) -> tuple[Any, Any, Any, Any, int, dict[str, Any], dict[str, Any]]:
            invalid_reasons: list[str] = []
            last_rgb = None
            last_id_to_labels: object = None
            last_semantic_unique_ids: list[int] = []
            attempts = 0
            rgb_guard_indices: list[int] = []
            semantic_guard_indices: list[int] = []
            state_record_indices: list[int] = []
            transition_indices: list[int] = []
            accumulation_resets: list[dict[str, Any]] = []
            override_record = receipt["semantics"]["transparent_annotation_override"]
            paired_contract = receipt["semantics"]["paired_capture_contract"]

            def set_annotation_state(value: bool, capture_pass: str, attempt: int) -> None:
                transition = _set_transparent_vial_annotation_state(
                    stage,
                    session_layer,
                    Usd,
                    enabled=value,
                )
                transition.update(
                    {
                        "physics_step": step,
                        "phase": phase,
                        "attempt": attempt,
                        "capture_pass": capture_pass,
                    }
                )
                override_record["transitions"].append(transition)
                transition_indices.append(len(override_record["transitions"]) - 1)

            def reset_accumulation(capture_pass: str, attempt: int) -> None:
                reset = _reset_renderer_accumulation_if_supported(context)
                accumulation_resets.append(
                    {
                        **reset,
                        "physics_step": step,
                        "phase": phase,
                        "attempt": attempt,
                        "capture_pass": capture_pass,
                    }
                )

            for _attempt in range(1, args.max_capture_tries + 1):
                attempts = _attempt
                receipt["render"]["capture_attempt_count"] += 1
                check_time_limits()
                set_annotation_state(False, "kept_rgb_source_equivalent", _attempt)
                rgb_guard = capture_render_pass(
                    render_product,
                    gating,
                    phase_timings,
                    phase="paired_rgb",
                    timing_phase="capture_rgb_orchestrator",
                    prepare=lambda: reset_accumulation(
                        "kept_rgb_source_equivalent", _attempt
                    ),
                    operation=lambda: _replicator_same_state_step_checked_in_session(
                        stage=stage,
                        session_layer=session_layer,
                        Usd=Usd,
                        rep=rep,
                        rt_subframes=args.rt_subframes,
                        callback_count=lambda: len(callback_records),
                        timeline=timeline,
                        physics_state=capture_physics_state,
                    ),
                )
                rgb_guard.update(
                    {
                        "physics_step": step,
                        "phase": phase,
                        "attempt": _attempt,
                        "capture_pass": "kept_rgb_source_equivalent",
                    }
                )
                capture_guard_records.append(rgb_guard)
                rgb_guard_indices.append(len(capture_guard_records) - 1)
                receipt["render"]["paired_render_pass_count"] += 1
                rgb = np.asarray(rgb_annotator.get_data())
                if rgb.ndim != 3 or rgb.shape[:2] != (height, width) or rgb.shape[2] < 3:
                    invalid_reasons.append(f"rgb_shape={rgb.shape}")
                    continue
                rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
                last_rgb = rgb
                if not np.any(rgb != rgb[0, 0]):
                    invalid_reasons.append("rgb_spatially_constant")
                    continue
                paired_state_kwargs: dict[str, Any] = {}
                if count_driven_body_runtime is not None:
                    paired_ids = [
                        int(value)
                        for value in (
                            shared_particle_points.GetIdsAttr().Get() or []
                        )
                    ]
                    paired_state_kwargs = {
                        "compositions": [
                            compositions_by_id[particle_id].tolist()
                            for particle_id in paired_ids
                        ],
                        "count_driven_body_runtime": count_driven_body_runtime,
                        "global_shader": global_shader,
                        "stream_records": stream_records,
                        "UsdGeom": UsdGeom,
                    }
                state_before = _capture_paired_particle_state(
                    particle_points=shared_particle_points,
                    callback_count=len(callback_records),
                    timeline=timeline,
                    camera_prim=camera_prim,
                    **paired_state_kwargs,
                )
                raw_semantic = None
                raw_instance = None
                state_after = None
                try:
                    set_annotation_state(
                        True, "semantic_transparent_annotation", _attempt
                    )
                    semantic_guard = capture_render_pass(
                        render_product,
                        gating,
                        phase_timings,
                        phase="paired_semantic",
                        timing_phase="capture_semantic_orchestrator",
                        prepare=lambda: reset_accumulation(
                            "semantic_transparent_annotation", _attempt
                        ),
                        operation=lambda: _replicator_same_state_step_checked_in_session(
                            stage=stage,
                            session_layer=session_layer,
                            Usd=Usd,
                            rep=rep,
                            rt_subframes=args.rt_subframes,
                            callback_count=lambda: len(callback_records),
                            timeline=timeline,
                            physics_state=capture_physics_state,
                        ),
                    )
                    semantic_guard.update(
                        {
                            "physics_step": step,
                            "phase": phase,
                            "attempt": _attempt,
                            "capture_pass": "semantic_transparent_annotation",
                        }
                    )
                    capture_guard_records.append(semantic_guard)
                    semantic_guard_indices.append(len(capture_guard_records) - 1)
                    receipt["render"]["paired_render_pass_count"] += 1
                    raw_semantic = semantic_annotator.get_data()
                    raw_instance = (
                        instance_annotator.get_data()
                        if instance_annotator is not None
                        else None
                    )
                    state_after = _capture_paired_particle_state(
                        particle_points=shared_particle_points,
                        callback_count=len(callback_records),
                        timeline=timeline,
                        camera_prim=camera_prim,
                        **paired_state_kwargs,
                    )
                finally:
                    set_annotation_state(False, "restore_after_semantic", _attempt)
                if state_after is None:
                    raise RuntimeError("semantic pass did not produce a post-pass state snapshot")
                state_evidence = _paired_capture_state_evidence(
                    state_before, state_after
                )
                state_evidence.update(
                    {
                        "physics_step": step,
                        "phase": phase,
                        "attempt": _attempt,
                        "rgb_guard_record_index": rgb_guard_indices[-1],
                        "semantic_guard_record_index": semantic_guard_indices[-1],
                    }
                )
                paired_contract["state_records"].append(state_evidence)
                state_record_indices.append(len(paired_contract["state_records"]) - 1)
                if state_evidence["passed"] is not True:
                    receipt["render"]["capture_attempts"].append(
                        {
                            "frame_index": len(frame_paths) + 1,
                            "physics_step": step,
                            "phase": phase,
                            "tries": attempts,
                            "invalid_reasons": [
                                "paired_capture_state_drift="
                                + ",".join(state_evidence["failed_predicates"])
                            ],
                            "included_in_video": False,
                            "semantic_class_mask_required": True,
                            "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                            "shared_single_render_pass": False,
                            "rgb_guard_record_indices": rgb_guard_indices,
                            "semantic_guard_record_indices": semantic_guard_indices,
                            "paired_state_record_indices": state_record_indices,
                            "annotation_transition_indices": transition_indices,
                            "renderer_accumulation_resets": accumulation_resets,
                            "failed": True,
                        }
                    )
                    raise RuntimeError(
                        "paired RGB/semantic capture changed frozen state fields: "
                        f"{state_evidence['failed_predicates']}"
                    )
                receipt["point_instancer"]["instance_segmentation_diagnostic"] = {
                    "available": isinstance(raw_instance, dict),
                    "keys": sorted(raw_instance.keys()) if isinstance(raw_instance, dict) else [],
                    "strict_class_mask_still_required": True,
                }
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
                if not isinstance(raw_instance, dict):
                    invalid_reasons.append("instance_envelope_not_mapping")
                    continue
                instance_info = raw_instance.get("info")
                if (
                    not isinstance(instance_info, dict)
                    or "idToLabels" not in instance_info
                ):
                    invalid_reasons.append("instance_mapping_missing")
                    continue
                instance = np.asarray(raw_instance.get("data"))
                if instance.ndim == 3 and instance.shape[2] == 1:
                    instance = instance[..., 0]
                if (
                    instance.shape != (height, width)
                    or np.issubdtype(instance.dtype, np.bool_)
                    or not np.issubdtype(instance.dtype, np.integer)
                ):
                    invalid_reasons.append(
                        f"instance_shape_or_dtype={instance.shape}/{instance.dtype}"
                    )
                    continue
                try:
                    instance_mapping = resolve_instance_paths(
                        instance_info["idToLabels"]
                    )
                    stream_overlap = exact_instance_semantic_overlap(
                        instance,
                        instance_info["idToLabels"],
                        exact.mask,
                        instance_path=PERSISTENT_RENDER_STREAM_PATH,
                        expected_visible=bool(visible_stream_tanks),
                    )
                except Exception as exc:
                    reason = getattr(exc, "reason_code", type(exc).__name__)
                    invalid_reasons.append(
                        f"stream_instance_overlap_invalid={reason}"
                    )
                    continue
                if stream_overlap.passed is not True:
                    invalid_reasons.append(
                        "stream_instance_semantic_overlap="
                        f"{stream_overlap.liquid_overlap_pixel_count}/"
                        f"{stream_overlap.instance_pixel_count}"
                    )
                    continue
                source_tank = (
                    str(visible_stream_tanks[0])
                    if len(visible_stream_tanks) == 1
                    else None
                )
                if len(visible_stream_tanks) > 1:
                    invalid_reasons.append("multiple_visible_stream_sources")
                    continue
                bound_material, _relationship = stream_record[
                    "binding"
                ].ComputeBoundMaterial()
                live_material_path = (
                    str(bound_material.GetPrim().GetPath())
                    if bound_material and bound_material.GetPrim().IsValid()
                    else None
                )
                requested_material_path = stream_record.get(
                    "requested_material_path"
                )
                material_match = (
                    requested_material_path is not None
                    and live_material_path == requested_material_path
                    and stream_record.get("active_source") == source_tank
                    if source_tank is not None
                    else None
                )
                if source_tank is not None and material_match is not True:
                    invalid_reasons.append("stream_material_source_readback_mismatch")
                    continue
                rgb_glass_transition = next(
                    (
                        override_record["transitions"][transition_index]
                        for transition_index in reversed(transition_indices)
                        if override_record["transitions"][transition_index].get(
                            "attempt"
                        )
                        == _attempt
                        and override_record["transitions"][transition_index].get(
                            "capture_pass"
                        )
                        == "kept_rgb_source_equivalent"
                    ),
                    None,
                )
                if (
                    not isinstance(rgb_glass_transition, dict)
                    or rgb_glass_transition.get("requested_value") is not False
                    or rgb_glass_transition.get("composed_type_name") != "bool"
                    or rgb_glass_transition.get("composed_value") is not False
                    or rgb_glass_transition.get(
                        "property_stack_strongest_is_session_layer"
                    )
                    is not True
                ):
                    invalid_reasons.append("rgb_glass_annotation_state_mismatch")
                    continue
                rgb_glass_annotation_state = {
                    "prim_path": override_record["prim_path"],
                    "attribute_name": override_record["attribute_name"],
                    "requested_value": rgb_glass_transition[
                        "requested_value"
                    ],
                    "composed_type_name": rgb_glass_transition[
                        "composed_type_name"
                    ],
                    "composed_value": rgb_glass_transition["composed_value"],
                    "property_stack_strongest_is_session_layer": (
                        rgb_glass_transition[
                            "property_stack_strongest_is_session_layer"
                        ]
                    ),
                    "source_equivalent": True,
                }
                effective_material_and_stream_state: dict[str, Any] = {}
                if count_driven_body_runtime is not None:
                    body_material_render_state = (
                        _read_count_driven_body_material_state(
                            count_driven_body_runtime["mesh"],
                            count_driven_body_runtime["shader"],
                            UsdShade,
                            UsdGeom,
                        )
                    )
                    stream_geometry_render_state = (
                        _read_render_stream_geometry_state(
                            stream_record,
                            UsdGeom,
                        )
                    )
                    effective_material_and_stream_state = {
                        "body_material_render_state": (
                            body_material_render_state
                        ),
                        "body_material_render_state_sha256": (
                            _json_value_sha256(body_material_render_state)
                        ),
                        "stream_geometry_render_state": (
                            stream_geometry_render_state
                        ),
                        "stream_geometry_render_state_sha256": (
                            _json_value_sha256(stream_geometry_render_state)
                        ),
                    }
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
                    "first_vial_bottom_roi": _first_vial_bottom_roi_evidence(
                        exact.mask
                    ),
                    "instance_unique_ids": sorted(
                        int(value) for value in np.unique(instance).tolist()
                    ),
                    "instance_id_to_prim_path": {
                        str(record.instance_id): record.path
                        for record in instance_mapping
                    },
                    "stream_semantic_overlap": {
                        "instance_path": stream_overlap.instance_path,
                        "expected_visible": stream_overlap.expected_visible,
                        "instance_ids": list(stream_overlap.instance_ids),
                        "instance_pixel_count": (
                            stream_overlap.instance_pixel_count
                        ),
                        "liquid_overlap_pixel_count": (
                            stream_overlap.liquid_overlap_pixel_count
                        ),
                        "non_liquid_instance_pixel_count": (
                            stream_overlap.non_liquid_instance_pixel_count
                        ),
                        "coverage_fraction": stream_overlap.coverage_fraction,
                        "passed": stream_overlap.passed,
                    },
                    "stream_material_state": {
                        "source_tank": source_tank,
                        "requested_material_path": requested_material_path,
                        "readback_material_path": live_material_path,
                        "matched": material_match,
                    },
                    "rgb_glass_annotation_state": rgb_glass_annotation_state,
                    **effective_material_and_stream_state,
                }
                receipt["render"]["capture_attempts"].append(
                    {
                        "frame_index": len(frame_paths) + 1,
                        "physics_step": step,
                        "phase": phase,
                        "tries": attempts,
                        "invalid_reasons": invalid_reasons,
                        "included_in_video": True,
                        "semantic_class_mask_required": True,
                        "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                        "shared_single_render_pass": False,
                        "rgb_guard_record_indices": rgb_guard_indices,
                        "semantic_guard_record_indices": semantic_guard_indices,
                        "paired_state_record_indices": state_record_indices,
                        "annotation_transition_indices": transition_indices,
                        "renderer_accumulation_resets": accumulation_resets,
                    }
                )
                return (
                    rgb,
                    semantic,
                    instance,
                    exact.mask,
                    exact.pixel_count,
                    mask_record,
                    state_evidence,
                )
            if last_rgb is not None:
                diagnostic = _write_rejected_rgb_diagnostic(
                    cv2, np, output, last_rgb, step=step, phase=phase
                )
                receipt["render"]["rejected_rgb_diagnostics"].append(diagnostic)
            receipt["render"]["capture_attempts"].append(
                {
                    "frame_index": len(frame_paths) + 1,
                    "physics_step": step,
                    "phase": phase,
                    "tries": attempts,
                    "invalid_reasons": invalid_reasons,
                    "included_in_video": False,
                    "semantic_class_mask_required": True,
                    "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                    "shared_single_render_pass": False,
                    "rgb_guard_record_indices": rgb_guard_indices,
                    "semantic_guard_record_indices": semantic_guard_indices,
                    "paired_state_record_indices": state_record_indices,
                    "annotation_transition_indices": transition_indices,
                    "renderer_accumulation_resets": accumulation_resets,
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
                str(receipt["point_instancer"]["path"]),
                *[
                    str(spec["path"])
                    for spec in receipt["point_instancer"].get("prototype_specs", [])
                ],
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
                f"paired RGB/semantic capture failed after {args.max_capture_tries} tries: "
                f"{invalid_reasons[-1] if invalid_reasons else 'unknown'}"
            )

        failure_phase = "capture_twin_reference"
        reference_rgb = acquire_rgb(0, "twin-reference-time-0001", included_in_video=False)
        reference_path = parity_dir / "twin-reference-time-0001.png"
        if not cv2.imwrite(os.fspath(reference_path), cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write twin reference image")

        plan = receipt["plan"]
        receipt["source_states"] = _runtime_source_state_rows(plan)
        failure_phase = "prepare_transparent_vial_annotation_control"
        receipt["semantics"]["transparent_annotation_override"] = (
            _prepare_transparent_vial_annotation_control(
                stage,
                session_layer,
                Usd,
                Sdf,
            )
        )

        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_writeback_setting = getattr(
            physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
        )
        if particles_writeback_setting is None:
            raise RuntimeError("PhysX lacks SETTING_UPDATE_PARTICLES_TO_USD")
        particles_writeback_initial = bool(
            settings.get_as_bool(particles_writeback_setting)
        )
        settings.set_bool(particles_writeback_setting, True)
        particles_writeback_enabled = bool(
            settings.get_as_bool(particles_writeback_setting)
        )
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
                "position_writeback_initial_readback": particles_writeback_initial,
                "position_writeback_enabled_readback": particles_writeback_enabled,
                "velocity_writeback_initial_readback": velocity_writeback_initial,
                "velocity_writeback_enabled_readback": velocity_writeback_enabled,
            }
        )
        if not particles_writeback_enabled or not velocity_writeback_enabled:
            raise RuntimeError("PhysX particle position/velocity USD writeback did not enable")
        fabric_initial = bool(SimulationManager.is_fabric_enabled())
        SimulationManager.enable_fabric(False)
        fabric_enabled = bool(SimulationManager.is_fabric_enabled())
        receipt["gpu_simulation"].update(
            {
                "fabric_initial_readback": fabric_initial,
                "fabric_disabled_readback": not fabric_enabled,
            }
        )
        receipt["particle_system"]["fabric_enabled"] = fabric_enabled
        if fabric_enabled:
            raise RuntimeError("Fabric must be disabled for runtime USD particle append")
        authored_paths: list[str] = [
            f"{TRANSPARENT_ANNOTATION_OVERRIDE.prim_path}."
            f"{TRANSPARENT_ANNOTATION_OVERRIDE.attribute_name}"
        ]
        failure_phase = "author_particle_system"
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
            receipt["particle_system"]["ccd"] = receipt["particle_system"]["ccd_enabled"]
            if receipt["particle_system"]["ccd_enabled"] is not True:
                raise RuntimeError("particle-system CCD did not retain the requested value")
            system.CreateParticleContactOffsetAttr().Set(
                runtime_particle_configuration["particle_contact_offset_m"]
            )
            system.CreateFluidRestOffsetAttr().Set(
                runtime_particle_configuration["fluid_rest_offset_m"]
            )
            system.CreateContactOffsetAttr().Set(
                runtime_particle_configuration["contact_offset_m"]
            )
            system.CreateRestOffsetAttr().Set(
                runtime_particle_configuration["rest_offset_m"]
            )
            system.CreateSolidRestOffsetAttr().Set(
                runtime_particle_configuration["solid_rest_offset_m"]
            )
            system.CreateSolverPositionIterationCountAttr().Set(
                _requested_solver_position_iterations(args)
            )
            system.CreateMaxNeighborhoodAttr().Set(
                runtime_particle_configuration["max_neighborhood"]
            )
            system.CreateMaxVelocityAttr().Set(PAINT_BALL_MAX_VELOCITY_M_S)
            particleUtils.add_physx_particle_smoothing(
                stage, system_path, enabled=True, strength=1.0
            )
            particleUtils.add_physx_particle_anisotropy(
                stage, system_path, enabled=True, scale=1.0
            )
            smoothing_api = PhysxSchema.PhysxParticleSmoothingAPI(system.GetPrim())
            anisotropy_api = PhysxSchema.PhysxParticleAnisotropyAPI(system.GetPrim())
            receipt["particle_system"].update(
                {
                    "max_velocity_m_s": float(system.GetMaxVelocityAttr().Get()),
                    "particle_contact_offset_m": float(
                        system.GetParticleContactOffsetAttr().Get()
                    ),
                    "fluid_rest_offset_m": float(
                        system.GetFluidRestOffsetAttr().Get()
                    ),
                    "contact_offset_m": float(system.GetContactOffsetAttr().Get()),
                    "rest_offset_m": float(system.GetRestOffsetAttr().Get()),
                    "solid_rest_offset_m": float(
                        system.GetSolidRestOffsetAttr().Get()
                    ),
                    "solver_position_iteration_count": int(
                        system.GetSolverPositionIterationCountAttr().Get()
                    ),
                    "solver_position_iteration_count_requested": (
                        _requested_solver_position_iterations(args)
                    ),
                    "max_neighborhood": int(system.GetMaxNeighborhoodAttr().Get()),
                    "smoothing": system.GetPrim().HasAPI(PhysxSchema.PhysxParticleSmoothingAPI),
                    "smoothing_enabled": bool(smoothing_api.GetParticleSmoothingEnabledAttr().Get()),
                    "smoothing_strength": float(smoothing_api.GetStrengthAttr().Get()),
                    "anisotropy": system.GetPrim().HasAPI(PhysxSchema.PhysxParticleAnisotropyAPI),
                    "anisotropy_enabled": bool(anisotropy_api.GetParticleAnisotropyEnabledAttr().Get()),
                    "anisotropy_scale": float(anisotropy_api.GetScaleAttr().Get()),
                }
            )
            if (
                receipt["particle_system"]["solver_position_iteration_count"]
                != receipt["particle_system"][
                    "solver_position_iteration_count_requested"
                ]
            ):
                raise RuntimeError(
                    "particle solver position-iteration readback drifted"
                )
            if (
                receipt["particle_system"]["smoothing"] is not True
                or receipt["particle_system"]["smoothing_enabled"] is not True
                or receipt["particle_system"]["smoothing_strength"] != 1.0
                or receipt["particle_system"]["anisotropy"] is not True
                or receipt["particle_system"]["anisotropy_enabled"] is not True
                or receipt["particle_system"]["anisotropy_scale"] != 1.0
            ):
                raise RuntimeError("particle smoothing/anisotropy readback did not retain enabled unit settings")

            physics_material = UsdShade.Material.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial"
            )
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(physics_material.GetPrim())
            _author_pbd_material_treatment(
                pbd=pbd,
                material_prim=physics_material.GetPrim(),
                particle_utils=particleUtils,
                use_water_preset=_water_pbd_preset_requested(args),
            )
            receipt["particle_system"]["pbd_material_readback"] = {
                "density_kg_m3": float(pbd.GetDensityAttr().Get()),
                "friction": float(pbd.GetFrictionAttr().Get()),
                "cohesion": float(pbd.GetCohesionAttr().Get()),
                "damping": float(pbd.GetDampingAttr().Get()),
                "surface_tension": float(pbd.GetSurfaceTensionAttr().Get()),
                "viscosity": float(pbd.GetViscosityAttr().Get()),
                "vorticity_confinement": float(
                    pbd.GetVorticityConfinementAttr().Get()
                ),
                "density_authored": bool(pbd.GetDensityAttr().HasAuthoredValueOpinion()),
                "friction_authored": bool(pbd.GetFrictionAttr().HasAuthoredValueOpinion()),
                "cohesion_authored": bool(pbd.GetCohesionAttr().HasAuthoredValueOpinion()),
                "damping_authored": bool(pbd.GetDampingAttr().HasAuthoredValueOpinion()),
                "surface_tension_authored": bool(
                    pbd.GetSurfaceTensionAttr().HasAuthoredValueOpinion()
                ),
                "viscosity_authored": bool(
                    pbd.GetViscosityAttr().HasAuthoredValueOpinion()
                ),
                "vorticity_confinement_authored": bool(
                    pbd.GetVorticityConfinementAttr().HasAuthoredValueOpinion()
                ),
            }
            if _water_pbd_preset_requested(args):
                treatment = receipt["particle_system"]["pbd_material_treatment"]
                meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
                readback_matches_plan = _water_pbd_preset_readback_matches_plan(
                    receipt["particle_system"]["pbd_material_readback"],
                    meters_per_unit=meters_per_unit,
                )
                treatment.update(
                    {
                        "applied": True,
                        "meters_per_unit_readback": meters_per_unit,
                        "readback": dict(
                            receipt["particle_system"]["pbd_material_readback"]
                        ),
                        "readback_matches_plan": readback_matches_plan,
                    }
                )
                if not readback_matches_plan:
                    raise RuntimeError(
                        "water PBD preset readback did not match the frozen A29 plan"
                    )
            binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
            binding.Bind(
                physics_material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            failure_phase = "author_global_liquid_material"
            global_material = UsdShade.Material.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial"
            )
            effective_palette = plan.get("run_palette")
            effective_transmissions = (
                effective_palette["transmission_1cm"]
                if isinstance(effective_palette, Mapping)
                else NOMINAL_TANK_TRANSMISSION_COLORS
            )
            tank_transmission_colors = {
                tank: tuple(
                    float(value)
                    for value in effective_transmissions[tank]
                )
                for tank in COMPOSITION_COMPONENTS
            }
            tank_transmission_matrix = np.asarray(
                [
                    tank_transmission_colors[tank]
                    for tank in COMPOSITION_COMPONENTS
                ],
                dtype=np.float64,
            )
            tank_extinctions = _source_extinctions_from_transmissions(
                tank_transmission_colors
            )
            global_shader = _author_omnisurface(
                stage,
                global_material,
                tank_transmission_colors[COMPOSITION_COMPONENTS[0]],
                Sdf,
                Gf,
                UsdShade,
            )
            body_material = global_material
            body_shader = global_shader
            if args.height_banded_body:
                body_material = UsdShade.Material.Define(
                    stage, f"{FLUIDMIX_NAMESPACE}/SpatialLiquidMaterial"
                )
                body_shader = _author_spatial_omnisurface(
                    stage,
                    body_material,
                    tank_transmission_colors[COMPOSITION_COMPONENTS[0]],
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
            if getattr(args, "count_driven_twin_liquid_mesh", False):
                failure_phase = "author_count_driven_twin_liquid_body"
                (
                    count_driven_body_runtime,
                    count_driven_source_evidence,
                ) = _author_count_driven_twin_liquid_body(
                    stage=stage,
                    body_material=body_material,
                    body_shader=body_shader,
                    height_banded=args.height_banded_body,
                    Usd=Usd,
                    UsdGeom=UsdGeom,
                    UsdShade=UsdShade,
                    UsdSemantics=UsdSemantics,
                    LegacySemantics=LegacySemantics,
                    Sdf=Sdf,
                    Gf=Gf,
                    Vt=Vt,
                    requested_source_time_code=COUNT_DRIVEN_SOURCE_TIME_CODE,
                    particle_volume_m3=float(
                        plan["render_treatment"]["particle_volume_m3"]
                    ),
                )
                receipt["count_driven_liquid_body"].update(
                    count_driven_source_evidence
                )
                receipt["semantics"]["count_driven_body_semantic_bound"] = True
                authored_paths.extend(
                    [COUNT_DRIVEN_BODY_XFORM_PATH, COUNT_DRIVEN_BODY_MESH_PATH]
                )

            # NVIDIA Paint Ball Emitter contract: one shared PointInstancer,
            # one sphere prototype per fixed palette colour, and all row
            # attributes extended together at each emission boundary.
            instancer_path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared")
            failure_phase = "author_shared_point_instancer"
            instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
            instancer_prim = instancer.GetPrim()
            point_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(instancer_prim)
            particle_api = PhysxSchema.PhysxParticleAPI(point_set_api)
            particle_api.CreateSelfCollisionAttr().Set(True)
            point_set_api.CreateFluidAttr().Set(True)
            particle_api.CreateParticleGroupAttr().Set(0)
            particle_api.CreateParticleSystemRel().SetTargets([system_path])
            capacity_evidence = _author_and_attest_particle_capacity(
                instancer_prim,
                value_type=Sdf.ValueTypeNames.Int,
                required_max_particles=max(
                    int(runtime_particle_configuration["max_particles"]),
                    int(plan["totals"]["particle_count"]),
                ),
                required_value_source=(
                    "max(fluidmix.physx_particle_contract.required_particle_"
                    "capacity,runtime_plan.totals.particle_count)"
                ),
            )
            receipt["point_instancer"]["particle_capacity"] = capacity_evidence
            receipt["point_instancer"]["max_particles"] = capacity_evidence[
                "readback_max_particles"
            ]
            if receipt.get("execution_schedule") is not None:
                receipt["execution_schedule"]["particle_capacity"] = deepcopy(
                    capacity_evidence
                )
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray([]))
            instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray([]))
            point_set_api.GetSimulationPointsAttr().Set(Vt.Vec3fArray([]))
            instancer.CreateIdsAttr().Set(Vt.Int64Array([]))
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([]))
            palette_rows = build_composition_palette()
            palette_colors = mix_extinction_transmission(
                palette_rows, tank_extinctions
            )
            pure_palette_indices = {
                int(np.argmax(row)): index
                for index, row in enumerate(palette_rows)
                if int(np.count_nonzero(row == 1.0)) == 1
                and int(np.count_nonzero(row)) == 1
            }
            # Define the pure source materials before any sphere prototype is
            # bound to them.  The prototype and render-only stream then share
            # these exact OmniSurface material prim paths.
            failure_phase = "author_source_palette_materials"
            stream_materials: dict[str, Any] = {}
            for tank_id in COMPOSITION_COMPONENTS:
                material_path = f"{FLUIDMIX_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
                stream_material = UsdShade.Material.Define(stage, material_path)
                _author_omnisurface(
                    stage,
                    stream_material,
                    tank_transmission_colors[tank_id],
                    Sdf,
                    Gf,
                    UsdShade,
                    scattering_depth_m=2.0 * RENDER_STREAM_RADIUS_M,
                )
                stream_materials[tank_id] = stream_material
            failure_phase = "author_palette_prototypes"
            prototype_specs = []
            for prototype_index, color in enumerate(palette_colors.tolist()):
                prototype_path = instancer_path.AppendChild(f"particlePrototype{prototype_index}")
                sphere = UsdGeom.Sphere.Define(stage, prototype_path)
                sphere.CreateRadiusAttr().Set(render_radius_m)
                if getattr(args, "count_driven_twin_liquid_mesh", False):
                    sphere.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                sphere.CreateDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in color])])
                )
                UsdGeom.PointInstancer(instancer_prim).GetPrototypesRel().AddTarget(prototype_path)
                pure_component = next(
                    (component for component, index in pure_palette_indices.items() if index == prototype_index),
                    None,
                )
                if pure_component is not None:
                    tank_id = COMPOSITION_COMPONENTS[pure_component]
                    material_path = f"{FLUIDMIX_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
                else:
                    material_path = build_palette_material_path(
                        FLUIDMIX_NAMESPACE, prototype_index
                    )
                    palette_material = UsdShade.Material.Define(stage, material_path)
                    _author_omnisurface(stage, palette_material, tuple(float(value) for value in color), Sdf, Gf, UsdShade)
                UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                    UsdShade.Material(stage.GetPrimAtPath(material_path))
                )
                _author_dual_liquid_semantics(
                    sphere.GetPrim(), UsdSemantics, LegacySemantics
                )
                prototype_specs.append(
                    {
                        "prototype_index": prototype_index,
                        "tank_id": None if pure_component is None else COMPOSITION_COMPONENTS[pure_component],
                        "composition": [float(value) for value in palette_rows[prototype_index].tolist()],
                        "path": str(prototype_path),
                        "material_path": material_path,
                        "semantic_label": "liquid",
                        "render_visibility": (
                            "invisible"
                            if getattr(args, "count_driven_twin_liquid_mesh", False)
                            else "inherited"
                        ),
                    }
                )
            _author_dual_liquid_semantics(instancer_prim, UsdSemantics, LegacySemantics)
            prototypes_render_hidden = bool(
                getattr(args, "count_driven_twin_liquid_mesh", False)
                and all(
                    UsdGeom.Imageable(stage.GetPrimAtPath(spec["path"]))
                    .ComputeVisibility()
                    == UsdGeom.Tokens.invisible
                    for spec in prototype_specs
                )
            )
            if (
                getattr(args, "count_driven_twin_liquid_mesh", False)
                and not prototypes_render_hidden
            ):
                raise RuntimeError("count-driven mode did not hide every sphere prototype")
            receipt["point_instancer"].update(
                {
                    "path": str(instancer_path),
                    "prototype_count": len(prototype_specs),
                    "prototype_specs": prototype_specs,
                    "max_particles": capacity_evidence["readback_max_particles"],
                    "prototypes_semantic_bound": True,
                    "instancer_semantic_bound": True,
                    "sphere_instances_render_hidden": prototypes_render_hidden,
                }
            )
            receipt["point_instancer"]["mass_density_readback"] = {
                "mass_api_present": bool(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
                "particle_mass_argument_kg": None,
                "initial_particle_count_at_helper_call": 0,
                "total_mass_attr_kg": None,
                "total_mass_attr_authored": False,
                "density_attr_kg_m3": float(pbd.GetDensityAttr().Get()),
                "density_attr_authored": bool(pbd.GetDensityAttr().HasAuthoredValueOpinion()),
                "nominal_per_particle_mass_kg": (
                    runtime_particle_configuration["density_kg_m3"]
                    * (2.0 * runtime_particle_configuration["fluid_rest_offset_m"]) ** 3
                ),
                "precedence": "pbd_material_density_and_fluid_rest_offset_compute_mass",
                "source": (
                    "NVIDIA Omni Physics particle-set mass computation"
                ),
            }
            if receipt["point_instancer"]["mass_density_readback"]["mass_api_present"]:
                raise RuntimeError("density-derived particle set unexpectedly has MassAPI")
            authored_paths.append(str(instancer_path))
            instancer_binding = UsdShade.MaterialBindingAPI.Apply(instancer_prim)
            instancer_binding.Bind(global_material)
            receipt["particle_system"].update(
                {
                    "pbd_material_binding": bool(binding.GetDirectBinding("physics").GetMaterial()),
                    "render_material_binding": bool(
                        instancer_binding.GetDirectBinding().GetMaterial()
                    ),
                }
            )
            receipt["material"]["point_instancer_global_material_binding"] = bool(
                instancer_binding.GetDirectBinding().GetMaterial()
            )
            system_semantics = UsdSemantics.LabelsAPI.Apply(system.GetPrim(), "class")
            if not system_semantics or not system_semantics.CreateLabelsAttr().Set(["liquid"]):
                raise RuntimeError("could not author liquid semantics on the particle system")
            receipt["semantics"]["system_binding_authored"] = True

            failure_phase = "author_render_streams"
            full_stream_height = NOZZLE_EXIT_CENTER_M[2] - SOURCE_LIQUID_BOTTOM_M
            stream = UsdGeom.Cylinder.Define(stage, PERSISTENT_RENDER_STREAM_PATH)
            stream.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            stream.CreateRadiusAttr().Set(RENDER_STREAM_RADIUS_M)
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
            initial_stream_material = stream_materials[COMPOSITION_COMPONENTS[0]]
            stream_binding = UsdShade.MaterialBindingAPI.Apply(stream.GetPrim())
            stream_binding.Bind(initial_stream_material)
            stream_semantics = UsdSemantics.LabelsAPI.Apply(
                stream.GetPrim(), "class"
            )
            if (
                not stream_semantics
                or not stream_semantics.CreateLabelsAttr().Set(["liquid"])
            ):
                raise RuntimeError(
                    "could not author liquid semantics on persistent render stream"
                )
            stream_record: dict[str, Any] = {
                "prim": stream.GetPrim(),
                "height": stream.GetHeightAttr(),
                "translate": translate,
                "binding": stream_binding,
                "active_source": None,
                "requested_material_path": None,
                "readback_material_path": str(initial_stream_material.GetPath()),
            }
            stream_records: dict[str, dict[str, Any]] = {
                "persistent": stream_record
            }
            authored_paths.append(PERSISTENT_RENDER_STREAM_PATH)
            receipt["semantics"].update(
                {
                    "stream_binding_paths": [PERSISTENT_RENDER_STREAM_PATH],
                    "stream_semantic_bound": True,
                }
            )

            failure_phase = "author_vial_colliders"
            collider_prims = []
            collider_collision_readbacks = []
            for spec in build_twin_collider_specs():
                cube = UsdGeom.Cube.Define(stage, Sdf.Path(str(spec["path"])))
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                collider_collision_readbacks.append(
                    _author_collider_collision_offsets(
                        PhysxSchema, cube.GetPrim(), spec
                    )
                )
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())
                authored_paths.append(str(spec["path"]))
            receipt["particle_system"].update(
                {
                    "collider_types": ["Cube"] * len(collider_prims),
                    "collider_paths": [str(prim.GetPath()) for prim in collider_prims],
                    "collider_collision_readbacks": (
                        collider_collision_readbacks
                    ),
                    "floor_tunneling_mitigation": {
                        "documented_source": (
                            FLOOR_COLLISION_DOCUMENTATION_URL
                        ),
                        "observed_failure_receipt_sha256": (
                            FLOOR_TUNNELING_FAILURE_RECEIPT_SHA256
                        ),
                        "floor_contact_offset_m": FLOOR_CONTACT_OFFSET_M,
                        "floor_rest_offset_m": FLOOR_REST_OFFSET_M,
                        "maximum_capped_displacement_per_step_m": (
                            PAINT_BALL_MAX_VELOCITY_M_S
                            / (REPORT_HZ * physics_substeps)
                        ),
                        "contact_offset_exceeds_capped_step": (
                            FLOOR_CONTACT_OFFSET_M
                            > PAINT_BALL_MAX_VELOCITY_M_S
                            / (REPORT_HZ * physics_substeps)
                        ),
                        "floor_top_unchanged_m": SOURCE_LIQUID_BOTTOM_M,
                        "readback_passed": bool(
                            collider_collision_readbacks
                            and math.isclose(
                                float(
                                    collider_collision_readbacks[0][
                                        "contact_offset_m"
                                    ]
                                ),
                                FLOOR_CONTACT_OFFSET_M,
                                rel_tol=0.0,
                                abs_tol=1.0e-9,
                            )
                            and math.isclose(
                                float(
                                    collider_collision_readbacks[0][
                                        "rest_offset_m"
                                    ]
                                ),
                                FLOOR_REST_OFFSET_M,
                                rel_tol=0.0,
                                abs_tol=1.0e-9,
                            )
                        ),
                    },
                    "wall_tunneling_mitigation": (
                        _build_wall_tunneling_mitigation(
                            collider_collision_readbacks,
                            physics_steps_per_second=(
                                REPORT_HZ * physics_substeps
                            ),
                        )
                    ),
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

            # The Points set is deliberately not authored here.  A18 established
            # that PhysX must first see a non-empty set on the scheduled main
            # thread boundary immediately before its owning timeline callback.

        failure_phase = "attach_semantic_annotators"
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        semantic_annotator.attach([render_product])
        instance_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        instance_annotator.attach([render_product])
        receipt["semantics"]["annotator_attached_after_stream_labels"] = True
        receipt["point_instancer"]["instance_segmentation_annotator"] = "instance_segmentation"

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
        video_repeat_plan: dict[str, object] | None = None
        video_repeat_count_by_step: dict[int, int] = {}
        video_frame_source_capture_indices: list[int] = []

        def set_stream_state(
            active_tank: str | None, contained_positions: Any
        ) -> tuple[str, ...]:
            surface_z = _render_stream_surface_z_m(
                contained_positions,
                count_driven_body=timeline_owned_count_driven,
                count_driven_body_geometry=(
                    latest_count_driven_body_geometry
                ),
                render_radius_m=render_radius_m,
            )
            height = max(NOZZLE_EXIT_CENTER_M[2] - surface_z, 0.0005)
            center_z = surface_z + height / 2.0
            def author_and_read() -> dict[str, Any]:
                record = stream_record
                record["height"].Set(height)
                record["translate"].Set(
                    Gf.Vec3d(
                        NOZZLE_EXIT_CENTER_M[0],
                        NOZZLE_EXIT_CENTER_M[1],
                        center_z,
                    )
                )
                requested_material_path = None
                if active_tank is not None:
                    if active_tank not in stream_materials:
                        raise RuntimeError(
                            f"persistent stream has no source material for {active_tank}"
                        )
                    requested_material = stream_materials[active_tank]
                    requested_material_path = str(
                        requested_material.GetPrim().GetPath()
                    )
                    if record.get("requested_material_path") != requested_material_path:
                        record["binding"].Bind(requested_material)
                UsdGeom.Imageable(record["prim"]).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                    if active_tank is not None
                    else UsdGeom.Tokens.invisible
                )
                bound_material, _relationship = record[
                    "binding"
                ].ComputeBoundMaterial()
                readback_material_path = (
                    str(bound_material.GetPrim().GetPath())
                    if bound_material and bound_material.GetPrim().IsValid()
                    else None
                )
                if (
                    requested_material_path is not None
                    and readback_material_path != requested_material_path
                ):
                    raise RuntimeError(
                        "persistent stream material readback mismatch: "
                        f"{readback_material_path!r} != {requested_material_path!r}"
                    )
                record.update(
                    {
                        "active_source": active_tank,
                        "requested_material_path": requested_material_path,
                        "readback_material_path": readback_material_path,
                    }
                )
                return {
                    "active_source": active_tank,
                    "requested_material_path": requested_material_path,
                    "readback_material_path": readback_material_path,
                    "visible": (
                        UsdGeom.Imageable(record["prim"]).ComputeVisibility()
                        != UsdGeom.Tokens.invisible
                    ),
                }

            stream_state = _with_session_edit_context(
                Usd,
                stage,
                session_layer,
                author_and_read,
            )
            expected = () if active_tank is None else (active_tank,)
            visible = (
                (str(stream_state["active_source"]),)
                if stream_state["visible"] is True
                else ()
            )
            if visible != expected:
                raise RuntimeError(
                    f"render-only stream visibility mismatch: expected {expected}, got {visible}"
                )
            return visible

        def capture_hidden_particle_control(
            *,
            step: int,
            emitted_particle_count: int,
            inside_particle_count: int,
        ) -> None:
            """Prove emitted sphere prototypes stay absent from liquid semantics."""

            if count_driven_body_runtime is None:
                return
            if receipt["semantics"].get("hidden_particle_control") is not None:
                return
            if emitted_particle_count <= 0 or inside_particle_count != emitted_particle_count:
                raise RuntimeError(
                    "hidden-particle control requires a non-empty fully contained state"
                )
            if any(
                UsdGeom.Imageable(stage.GetPrimAtPath(spec["path"]))
                .ComputeVisibility()
                != UsdGeom.Tokens.invisible
                for spec in prototype_specs
            ):
                raise RuntimeError(
                    "hidden-particle control found a visible sphere prototype"
                )
            if any(
                UsdGeom.Imageable(record["prim"]).ComputeVisibility()
                != UsdGeom.Tokens.invisible
                for record in stream_records.values()
            ):
                raise RuntimeError(
                    "hidden-particle control requires every render stream hidden"
                )

            body_mesh = count_driven_body_runtime["mesh"]

            def hide_body() -> None:
                body_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

            _with_session_edit_context(Usd, stage, session_layer, hide_body)
            if (
                UsdGeom.Imageable(body_mesh.GetPrim()).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            ):
                raise RuntimeError("count-driven body did not hide for isolation control")
            accumulation_reset: dict[str, Any] | None = None

            def reset_hidden_capture() -> None:
                nonlocal accumulation_reset
                accumulation_reset = _reset_renderer_accumulation_if_supported(
                    context
                )
            transition_before = _set_transparent_vial_annotation_state(
                stage, session_layer, Usd, enabled=True
            )
            transition_before.update(
                {
                    "physics_step": step,
                    "phase": "hidden-emitted-particle-control",
                    "attempt": 1,
                    "capture_pass": "semantic_transparent_annotation",
                }
            )
            receipt["semantics"]["transparent_annotation_override"][
                "transitions"
            ].append(transition_before)
            raw_control = None
            try:
                control_guard = capture_render_pass(
                    render_product,
                    gating,
                    phase_timings,
                    phase="hidden_particle_semantic",
                    timing_phase="capture_semantic_orchestrator",
                    prepare=reset_hidden_capture,
                    operation=lambda: _replicator_same_state_step_checked_in_session(
                        stage=stage,
                        session_layer=session_layer,
                        Usd=Usd,
                        rep=rep,
                        rt_subframes=args.rt_subframes,
                        callback_count=lambda: len(callback_records),
                        timeline=timeline,
                        physics_state=capture_physics_state,
                    ),
                )
                control_guard.update(
                    {
                        "physics_step": step,
                        "phase": "hidden-emitted-particle-control",
                        "attempt": 1,
                        "capture_pass": "semantic_transparent_annotation",
                    }
                )
                capture_guard_records.append(control_guard)
                raw_control = semantic_annotator.get_data()
            finally:
                transition_after = _set_transparent_vial_annotation_state(
                    stage, session_layer, Usd, enabled=False
                )
                transition_after.update(
                    {
                        "physics_step": step,
                        "phase": "hidden-emitted-particle-control",
                        "attempt": 1,
                        "capture_pass": "restore_after_semantic",
                    }
                )
                receipt["semantics"]["transparent_annotation_override"][
                    "transitions"
                ].append(transition_after)

                def show_body() -> None:
                    body_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

                _with_session_edit_context(Usd, stage, session_layer, show_body)
            if not isinstance(raw_control, dict):
                raise RuntimeError("hidden-particle semantic control is not a mapping")
            raw_control_info = raw_control.get("info")
            control_semantic = np.asarray(raw_control.get("data"))
            if control_semantic.ndim == 3 and control_semantic.shape[2] == 1:
                control_semantic = control_semantic[..., 0]
            if (
                not isinstance(raw_control_info, dict)
                or "idToLabels" not in raw_control_info
                or control_semantic.shape != (height, width)
                or not np.issubdtype(control_semantic.dtype, np.integer)
            ):
                raise RuntimeError("hidden-particle semantic control is unreadable")
            control_mapping = resolve_class_ids(raw_control_info["idToLabels"])
            control_liquid_ids = control_mapping.ids_for("liquid")
            control_liquid_pixels = int(
                np.count_nonzero(np.isin(control_semantic, control_liquid_ids))
            )
            if control_liquid_pixels != 0:
                raise RuntimeError(
                    "hidden emitted spheres still produced liquid semantic pixels"
                )
            control_path = (
                semantic_controls_dir
                / f"emitted-spheres-hidden-step-{step:05d}.npy"
            )
            np.save(control_path, control_semantic, allow_pickle=False)
            receipt["semantics"]["hidden_particle_control"] = {
                "physics_step": step,
                "emitted_particle_count": emitted_particle_count,
                "inside_vial_interior_count": inside_particle_count,
                "body_and_stream_hidden_liquid_pixel_count": 0,
                "hidden_sphere_prototypes": True,
                "renderer_accumulation_reset_after_hide": accumulation_reset,
                "control_semantic_path": str(control_path.relative_to(output)),
                "control_semantic_sha256": _sha256(control_path),
                "control_liquid_semantic_ids": list(control_liquid_ids),
                "body_only_target_path": COUNT_DRIVEN_BODY_MESH_PATH,
                "body_only_target_ids": [],
                "passed": False,
            }

        def capture(
            step: int,
            phase: str,
            visible_stream_tanks: tuple[str, ...],
        ) -> tuple[Path, list[list[float]], Any]:
            liquid_representation = (
                COUNT_DRIVEN_RENDER_PATH
                if count_driven_body_runtime is not None
                else "point_instancer"
            )
            if (
                count_driven_body_runtime is not None
                and latest_count_driven_body_geometry is None
            ):
                raise RuntimeError("count-driven body was not updated before capture")
            (
                rgb,
                semantic,
                instance,
                liquid_mask,
                liquid_pixels,
                mask_record,
                paired_state_evidence,
            ) = acquire_paired_frame(step, phase, visible_stream_tanks)
            artifact_io_started = time.monotonic()
            raw_capture_positions = np.asarray(
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (shared_particle_points.GetSimulationPointsAttr().Get() or [])
                ],
                dtype=np.float64,
            )
            smoothed_render_positions = np.asarray(
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in _positions(
                        shared_particle_points.GetPrim(), instancer=True
                    )
                ],
                dtype=np.float64,
            )
            point_instancer_scales = np.asarray(
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (
                        shared_particle_points.GetScalesAttr().Get() or []
                    )
                ],
                dtype=np.float64,
            )
            observed_ids = [
                int(value)
                for value in (shared_particle_points.GetIdsAttr().Get() or [])
            ]
            birth_by_id: dict[int, list[float]] = {}
            scheduled_source_by_id: dict[int, str] = {}
            birth_step_by_id: dict[int, int] = {}
            birth_event_by_id: dict[int, str] = {}
            pour_event_by_id: dict[int, str] = {}
            for metadata in active_row_metadata:
                birth_step = int(metadata["event"]["physics_step"])
                birth_event_id = str(metadata["birth_event_id"])
                for particle_id, composition in zip(
                    metadata["particle_ids"], metadata["source_compositions"]
                ):
                    particle_id = int(particle_id)
                    birth_by_id[particle_id] = [
                        float(value) for value in composition
                    ]
                    scheduled_source_by_id[particle_id] = str(
                        metadata["scheduled_source"]
                    )
                    birth_step_by_id[particle_id] = birth_step
                    birth_event_by_id[particle_id] = birth_event_id
                    pour_event_by_id[particle_id] = str(
                        metadata["pour_event_id"]
                    )
            snapshot_state = {
                "simulation_points": raw_capture_positions.tolist(),
                "positions": smoothed_render_positions.tolist(),
                "velocities": [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (
                        shared_particle_points.GetVelocitiesAttr().Get() or []
                    )
                ],
                "orientations": [
                    [
                        float(value.GetReal()),
                        float(value.GetImaginary()[0]),
                        float(value.GetImaginary()[1]),
                        float(value.GetImaginary()[2]),
                    ]
                    for value in (
                        shared_particle_points.GetOrientationsAttr().Get() or []
                    )
                ],
                "scales": point_instancer_scales.tolist(),
                "ids": observed_ids,
                "proto_indices": [
                    int(value)
                    for value in (
                        shared_particle_points.GetProtoIndicesAttr().Get() or []
                    )
                ],
                "birth_compositions": [birth_by_id[value] for value in observed_ids],
                "current_compositions": [
                    compositions_by_id[value].tolist() for value in observed_ids
                ],
                "scheduled_sources": [
                    scheduled_source_by_id[value] for value in observed_ids
                ],
                "birth_steps": [birth_step_by_id[value] for value in observed_ids],
                "birth_event_ids": [
                    birth_event_by_id[value] for value in observed_ids
                ],
                "pour_event_ids": [pour_event_by_id[value] for value in observed_ids],
            }
            particle_snapshot = persist_runtime_particle_snapshot(
                output / "particle-snapshots",
                f"step-{step:05d}-{phase}",
                snapshot_state,
                expected_ids=shared_particle_ids,
                artifact_path_prefix="particle-snapshots",
            )

            def in_flight_mask_for(positions: np.ndarray) -> np.ndarray:
                return (
                    (positions[:, 2] > VIAL_MOUTH_M)
                    & (
                        np.sum(
                            (
                                positions[:, :2]
                                - np.asarray(NOZZLE_EXIT_CENTER_M[:2])[None, :]
                            )
                            ** 2,
                            axis=1,
                        )
                        <= render_radius_m**2
                    )
                    if len(positions)
                    else np.zeros(0, dtype=bool)
                )

            in_flight_mask = in_flight_mask_for(raw_capture_positions)
            lower_bounds = (
                VIAL_CENTER_XY_M[0] - SOURCE_LIQUID_RADIUS_M,
                VIAL_CENTER_XY_M[1] - SOURCE_LIQUID_RADIUS_M,
                SOURCE_LIQUID_BOTTOM_M,
            )
            upper_bounds = (
                VIAL_CENTER_XY_M[0] + SOURCE_LIQUID_RADIUS_M,
                VIAL_CENTER_XY_M[1] + SOURCE_LIQUID_RADIUS_M,
                VIAL_MOUTH_M,
            )
            inner_apothem = SOURCE_LIQUID_RADIUS_M * math.cos(
                math.pi / VIAL_COLLIDER_WALL_COUNT
            )
            cross_section_area_m2 = (
                0.5
                * VIAL_COLLIDER_WALL_COUNT
                * SOURCE_LIQUID_RADIUS_M**2
                * math.sin(2.0 * math.pi / VIAL_COLLIDER_WALL_COUNT)
            )
            containment = containment_evidence(
                raw_capture_positions,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                expected_in_flight_mask=in_flight_mask,
                cylindrical_center_xy=VIAL_CENTER_XY_M,
                inner_radius_m=inner_apothem,
                polygon_center_xy=VIAL_CENTER_XY_M,
                polygon_apothem_m=inner_apothem,
                polygon_wall_count=VIAL_COLLIDER_WALL_COUNT,
            )
            smoothed_render_containment = containment_evidence(
                smoothed_render_positions,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                expected_in_flight_mask=in_flight_mask_for(
                    smoothed_render_positions
                ),
                cylindrical_center_xy=VIAL_CENTER_XY_M,
                inner_radius_m=inner_apothem,
                polygon_center_xy=VIAL_CENTER_XY_M,
                polygon_apothem_m=inner_apothem,
                polygon_wall_count=VIAL_COLLIDER_WALL_COUNT,
            )
            stage_volume_target = _programmed_stage_volume_target(
                receipt["plan"]["pours"],
                step=step,
                visible_stream_tanks=tuple(visible_stream_tanks),
            )
            stage_kwargs: dict[str, object] = {}
            if stage_volume_target is not None:
                stage_kwargs = {
                    "programmed_cumulative_volume_ml": stage_volume_target[0],
                    "expected_cumulative_particle_count": stage_volume_target[1],
                    "cross_section_area_m2": cross_section_area_m2,
                    "height_tolerance_m": VOLUME_HEIGHT_TOLERANCE_M,
                    "render_radius_m": render_radius_m,
                    "point_instancer_scales": point_instancer_scales,
                    "smoothed_render_containment": smoothed_render_containment,
                }
            capture_record = build_capture_record(
                physics_step=step,
                phase=phase,
                containment=containment,
                prototype_count=int(receipt["point_instancer"]["prototype_count"]),
                instancer_path=str(receipt["point_instancer"]["path"]),
                emitted_particle_count=len(raw_capture_positions),
                **stage_kwargs,
            )
            capture_record["particle_snapshot"] = particle_snapshot
            if latest_count_driven_body_geometry is not None:
                body_capture_geometry = (
                    _select_count_driven_body_capture_geometry(
                        latest_count_driven_body_geometry,
                        physics_step=step,
                        phase=phase,
                    )
                )
                capture_record["render_body_geometry"] = body_capture_geometry
                receipt["count_driven_liquid_body"]["capture_geometry"].append(
                    body_capture_geometry
                )
            if stage_volume_target is not None:
                capture_record = attach_visible_volume_evidence(
                    capture_record,
                    liquid_mask,
                    center_xy_m=VIAL_CENTER_XY_M,
                    bottom_z_m=SOURCE_LIQUID_BOTTOM_M,
                    mouth_z_m=VIAL_MOUTH_M,
                    circumradius_m=SOURCE_LIQUID_RADIUS_M,
                )
                if latest_count_driven_body_geometry is not None:
                    if (
                        latest_count_driven_body_geometry[
                            "contained_particle_count"
                        ]
                        != containment["inside_vial_interior_count"]
                    ):
                        raise RuntimeError(
                            "count-driven body count differs from exact containment"
                        )
                    receipt["count_driven_liquid_body"][
                        "stage_geometry"
                    ].append(body_capture_geometry)
                    if args.height_banded_body:
                        band_evidence = latest_count_driven_body_geometry.get(
                            "height_band_evidence"
                        )
                        if not isinstance(band_evidence, dict):
                            raise RuntimeError(
                                "height-banded stage lacks band evidence"
                            )
                        receipt["count_driven_liquid_body"][
                            "stage_band_evidence"
                        ].append(
                            {
                                "physics_step": step,
                                "phase": phase,
                                **deepcopy(band_evidence),
                            }
                        )
            receipt["point_instancer"].setdefault("capture_records", []).append(capture_record)
            receipt["point_instancer"]["ids_stable"] = True
            receipt["containment"]["capture_records"].append(capture_record)
            receipt["containment"]["vial_interior_bounds_m"] = {
                "center_xy_m": list(VIAL_CENTER_XY_M),
                "inner_apothem_m": inner_apothem,
                "circumradius_m": SOURCE_LIQUID_RADIUS_M,
                "wall_count": VIAL_COLLIDER_WALL_COUNT,
                "cross_section_area_m2": cross_section_area_m2,
                "official_containment": "regular_dodecagon_halfspaces",
                "conservative_diagnostic": "inscribed_circle",
                "bottom_z_m": SOURCE_LIQUID_BOTTOM_M,
                "mouth_z_m": VIAL_MOUTH_M,
                "numerical_tolerance_m": 1.0e-6,
            }
            if containment["leakage_detected"] is True:
                raise RuntimeError(
                    f"particle leakage detected at shared capture step {step}"
                )
            if not visible_stream_tanks and containment["outside_vial_interior_count"] != 0:
                raise RuntimeError(
                    f"stream-hidden capture has particles outside vial interior at step {step}"
                )
            index = len(frame_paths) + 1
            path = frames_dir / f"frame-{index:04d}-step-{step:05d}-{phase}.png"
            semantic_path = semantic_dir / f"frame-{index:04d}-step-{step:05d}.npy"
            instance_path = instance_dir / f"frame-{index:04d}-step-{step:05d}.npy"
            instance_mapping_path = (
                instance_mapping_dir
                / f"frame-{index:04d}-step-{step:05d}.json"
            )
            liquid_path = liquid_masks_dir / f"frame-{index:04d}-step-{step:05d}.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(os.fspath(path), bgr):
                raise RuntimeError(f"could not write RGB frame {path}")
            np.save(semantic_path, semantic, allow_pickle=False)
            np.save(instance_path, instance, allow_pickle=False)
            _write_json(
                instance_mapping_path,
                mask_record["instance_id_to_prim_path"],
            )
            mask_record.update(
                {
                    "instance_path": str(instance_path.relative_to(output)),
                    "instance_sha256": _sha256(instance_path),
                    "instance_mapping_path": str(
                        instance_mapping_path.relative_to(output)
                    ),
                    "instance_mapping_sha256": _sha256(instance_mapping_path),
                }
            )
            if not cv2.imwrite(os.fspath(liquid_path), liquid_mask):
                raise RuntimeError(f"could not write liquid semantic mask {liquid_path}")
            reloaded_semantic = np.load(semantic_path, allow_pickle=False)
            reloaded_instance = np.load(instance_path, allow_pickle=False)
            reloaded_instance_mapping = json.loads(
                instance_mapping_path.read_text(encoding="utf-8")
            )
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
            saved_stream_overlap = exact_instance_semantic_overlap(
                reloaded_instance,
                reloaded_instance_mapping,
                saved_exact.mask,
                instance_path=PERSISTENT_RENDER_STREAM_PATH,
                expected_visible=bool(visible_stream_tanks),
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
                "first_vial_bottom_roi": _first_vial_bottom_roi_evidence(
                    saved_exact.mask
                ),
                "instance_unique_ids": sorted(
                    int(value) for value in np.unique(reloaded_instance).tolist()
                ),
                "instance_id_to_prim_path": reloaded_instance_mapping,
                "stream_semantic_overlap": {
                    "instance_path": saved_stream_overlap.instance_path,
                    "expected_visible": saved_stream_overlap.expected_visible,
                    "instance_ids": list(saved_stream_overlap.instance_ids),
                    "instance_pixel_count": (
                        saved_stream_overlap.instance_pixel_count
                    ),
                    "liquid_overlap_pixel_count": (
                        saved_stream_overlap.liquid_overlap_pixel_count
                    ),
                    "non_liquid_instance_pixel_count": (
                        saved_stream_overlap.non_liquid_instance_pixel_count
                    ),
                    "coverage_fraction": saved_stream_overlap.coverage_fraction,
                    "passed": saved_stream_overlap.passed,
                },
                "stream_material_state": mask_record["stream_material_state"],
                **copy_semantic_effective_render_state(
                    mask_record,
                    count_driven=count_driven_body_runtime is not None,
                ),
                "instance_path": str(instance_path.relative_to(output)),
                "instance_sha256": _sha256(instance_path),
                "instance_mapping_path": str(
                    instance_mapping_path.relative_to(output)
                ),
                "instance_mapping_sha256": _sha256(instance_mapping_path),
            }
            if (
                reloaded_semantic.shape != (height, width)
                or reloaded_mask is None
                or reloaded_mask.shape != (height, width)
                or not np.array_equal(reloaded_semantic, semantic)
                or not np.array_equal(reloaded_instance, instance)
                or not np.array_equal(reloaded_mask, liquid_mask)
                or not np.array_equal(reloaded_mask, saved_exact.mask)
                or saved_exact.pixel_count != liquid_pixels
                or saved_record != mask_record
            ):
                raise RuntimeError("saved semantic evidence is unreadable or misaligned")
            if not args.realtime_video:
                repeat_count = video_repeat_count_by_step.get(step)
                if (
                    isinstance(repeat_count, bool)
                    or not isinstance(repeat_count, int)
                    or repeat_count <= 0
                ):
                    raise RuntimeError(
                        f"video repeat plan has no positive count for capture step {step}"
                    )
                source_capture_index = len(frame_paths)
                for _ in range(repeat_count):
                    writer.write(bgr)
                    video_frame_source_capture_indices.append(
                        source_capture_index
                    )
            frame_paths.append(path)
            frame_steps.append(step)
            semantic_paths.append(semantic_path)
            liquid_mask_paths.append(liquid_path)
            liquid_pixel_counts.append(liquid_pixels)
            if (
                count_driven_body_runtime is not None
                and not visible_stream_tanks
                and receipt["semantics"]["hidden_particle_control"].get(
                    "passed"
                )
                is not True
            ):
                body_only_ids = list(mask_record["liquid_semantic_ids"])
                if not body_only_ids or liquid_pixels <= 0:
                    raise RuntimeError(
                        "count-driven body-only semantic target has no pixels"
                    )
                receipt["semantics"]["hidden_particle_control"].update(
                    {
                        "body_only_target_ids": body_only_ids,
                        "body_only_capture_step": step,
                        "body_only_liquid_pixel_count": liquid_pixels,
                        "passed": True,
                    }
                )
            semantic_mask_records.append(
                {
                    "phase": phase,
                    "visible_stream_tanks": list(visible_stream_tanks),
                    "liquid_representation": liquid_representation,
                    "shared_rgb_semantic_capture": False,
                    "paired_same_physics_state_rgb_semantic_capture": True,
                    "paired_capture_state_passed": paired_state_evidence[
                        "passed"
                    ],
                    "paired_capture_state_evidence": paired_state_evidence,
                    **(
                        {
                            "body_points_sha256": paired_state_evidence[
                                "body_points_sha256_before"
                            ],
                            "body_extent_sha256": paired_state_evidence[
                                "body_extent_sha256_before"
                            ],
                            "body_visibility": latest_count_driven_body_geometry[
                                "body_visibility"
                            ],
                            "body_material_color_sha256": paired_state_evidence[
                                "body_material_color_sha256_before"
                            ],
                            "stream_visibility_sha256": paired_state_evidence[
                                "stream_visibility_sha256_before"
                            ],
                            "compositions_sha256": paired_state_evidence[
                                "compositions_sha256_before"
                            ],
                        }
                        if latest_count_driven_body_geometry is not None
                        else {}
                    ),
                    "liquid_pixel_count_in_first_vial_bottom_roi": int(
                        mask_record["first_vial_bottom_roi"][
                            "liquid_pixel_count_in_roi"
                        ]
                    ),
                    **mask_record,
                }
            )
            _record_phase_timing(
                phase_timings,
                "capture_artifact_io",
                time.monotonic() - artifact_io_started,
            )
            return (
                path,
                raw_capture_positions.tolist(),
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
        nuisance_binding = getattr(args, "_nuisance_receipt_binding", None)
        if nuisance_binding is not None:
            failure_phase = "author_nuisance_session_layer"
            receipt["nuisance_runtime"] = apply_nuisance_session_layer(
                stage,
                Usd,
                nuisance_binding,
                case_id=args.nuisance_case_id,
                seed=args.seed,
            )
        failure_phase = "read_effective_render_state"
        receipt["camera"]["effective_render_state"] = (
            _read_effective_camera_render_state(
                camera_prim,
                camera,
                Usd,
                UsdGeom,
                time_code=TWIN_REFERENCE_TIME_CODE,
            )
        )
        receipt["scene_contract"]["effective_light_state"] = (
            read_existing_light_inventory(stage)
        )
        plan = receipt["plan"]
        runtime_capture_schedule = receipt["runtime_capture_schedule"]
        event_index = _build_emission_event_index(plan)
        if args.realtime_video:
            capture_steps = {
                int(pour["stage_capture_step"]) for pour in plan["pours"]
            }
            if args.qualification_mode:
                capture_steps = {min(capture_steps)}
        else:
            capture_steps = (
                {706, 1404, 1412}
                if args.qualification_mode
                else set(
                    _build_capture_steps(
                        plan,
                        stride=args.capture_stride,
                        runtime_schedule=runtime_capture_schedule,
                    )
                )
            )
        ordered_capture_steps = sorted(int(step) for step in capture_steps)
        qualification_target_step = 1412 if args.qualification_mode else int(
            runtime_capture_schedule["final_plus_3_0_step"]
        )
        video_source_steps = (
            _build_realtime_video_capture_steps(
                final_step=qualification_target_step,
                report_hz=REPORT_HZ,
                video_fps=float(args.video_fps),
            )
            if args.realtime_video
            else tuple(ordered_capture_steps)
        )
        video_repeat_plan = build_time_preserving_video_repeat_plan(
            video_source_steps,
            physics_hz=REPORT_HZ,
            playback_fps=float(args.video_fps),
        )
        video_repeat_count_by_step = dict(
            zip(
                video_repeat_plan["source_frame_steps"],
                video_repeat_plan["repeat_counts"],
                strict=True,
            )
        )
        realtime_video_steps = set(video_source_steps)
        physics_clock = _build_physics_clock_contract(
            physics_substeps=physics_substeps,
            report_interval_count=qualification_target_step,
        )
        if not timeline_owned_count_driven:
            receipt["timing"]["direct_step_probe"][
                "first_capture_step"
            ] = min(capture_steps)
        receipt["timing"].update(
            {
                "report_hz": physics_clock["report_hz"],
                "report_dt_s": physics_clock["report_dt_s"],
                "report_interval_count": qualification_target_step,
                "physics_hz": physics_clock["physics_steps_per_second"],
                "physics_steps_per_second": physics_clock[
                    "physics_steps_per_second"
                ],
                "physics_substeps_per_report_interval": physics_substeps,
                "physics_dt_s": physics_clock["physics_dt_s"],
                "physics_callback_count": physics_clock[
                    "physics_callback_count"
                ],
                "min_frame_rate": physics_clock["min_frame_rate"],
                "timeline_ticks_per_frame": 1,
                "timeline_end_s": physics_clock["timeline_end_s"],
                "physical_duration_s": physics_clock["physical_duration_s"],
                "callbacks_expected": physics_clock["physics_callback_count"],
                "diffusion_update_count": 0,
                "diffusion_dt_s": physics_clock["diffusion_dt_s"],
                "diffusion_integrated_time_s": 0.0,
            }
        )
        stage_by_step = {
            int(pour["stage_capture_step"]): (index, pour)
            for index, pour in enumerate(plan["pours"], start=1)
        }
        actual_events: dict[int, list[dict[str, int]]] = {
            int(pour["ordinal"]): [] for pour in plan["pours"]
        }
        emission_trace: list[dict[str, Any]] = []
        execution_event_attestations: list[dict[str, Any]] = []
        stream_active_steps: dict[str, list[int]] = {
            tank_id: [] for tank_id in COMPOSITION_COMPONENTS
        }
        stage_memory_before = {
            int(pour["ordinal"]): None for pour in plan["pours"]
        }
        component_counts = np.zeros(len(COMPOSITION_COMPONENTS), dtype=np.float64)
        compositions_by_id: dict[int, Any] = {}
        active_row_metadata: list[dict[str, Any]] = []
        shared_particle_points = _PointInstancerParticleAdapter(instancer, point_set_api)
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
        failure_phase = "configure_timeline"
        requested_physics_hz = int(physics_clock["physics_steps_per_second"])
        with Usd.EditContext(stage, session_layer):
            scene_api.GetTimeStepsPerSecondAttr().Set(requested_physics_hz)
        readback_physics_hz = int(
            scene_api.GetTimeStepsPerSecondAttr().Get()
        )
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(physx_bindings.SETTING_MIN_FRAME_RATE, REPORT_HZ)
        readback_min_frame_rate = int(
            settings.get_as_int(physx_bindings.SETTING_MIN_FRAME_RATE)
        )
        readback_fixed_time_stepping = bool(
            settings.get_as_bool("/app/player/useFixedTimeStepping")
        )
        receipt["timing"]["physics_settings_readback"] = {
            "physics_scene_path": str(scene.GetPrim().GetPath()),
            "requested_time_steps_per_second": requested_physics_hz,
            "readback_time_steps_per_second": readback_physics_hz,
            "requested_min_frame_rate": REPORT_HZ,
            "readback_min_frame_rate": readback_min_frame_rate,
            "requested_fixed_time_stepping": True,
            "readback_fixed_time_stepping": readback_fixed_time_stepping,
            "requested_timeline_rate": REPORT_HZ,
            "readback_timeline_rate": None,
            "requested_timeline_ticks_per_frame": 1,
            "readback_timeline_ticks_per_frame": None,
            "passed": False,
        }
        if (
            readback_physics_hz != requested_physics_hz
            or readback_min_frame_rate != REPORT_HZ
            or readback_fixed_time_stepping is not True
        ):
            raise RuntimeError("PhysX/report clock setting readback mismatch")
        timeline = omni.timeline.get_timeline_interface()
        timeline_initialization = _configure_zero_origin_timeline(
            stage,
            session_layer,
            app,
            timeline,
            Usd,
            physics_hz=REPORT_HZ,
            final_callback_index=qualification_target_step,
        )
        receipt["timing"]["timeline_initialization"] = timeline_initialization
        timeline_readback = timeline_initialization["after_configuration"]
        readback_timeline_rate = float(
            timeline_readback["timeline_time_codes_per_second"]
        )
        readback_timeline_ticks = int(timeline.get_ticks_per_frame())
        clock_settings_passed = bool(
            readback_physics_hz == requested_physics_hz
            and readback_min_frame_rate == REPORT_HZ
            and readback_fixed_time_stepping is True
            and math.isclose(
                readback_timeline_rate,
                float(REPORT_HZ),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and readback_timeline_ticks == 1
        )
        receipt["timing"]["physics_settings_readback"].update(
            {
                "readback_timeline_rate": readback_timeline_rate,
                "readback_timeline_ticks_per_frame": readback_timeline_ticks,
                "passed": clock_settings_passed,
            }
        )
        if not clock_settings_passed:
            raise RuntimeError("timeline/report clock setting readback mismatch")
        print(
            "F2_TIMELINE_INITIALIZATION "
            + json.dumps(timeline_initialization, sort_keys=True),
            flush=True,
        )

        if timeline_owned_count_driven:
            timeline_playback_start = _start_timeline_non_looping(
                stage,
                session_layer,
                timeline,
                Usd,
                configured=timeline_initialization,
            )
            receipt["timing"][
                "timeline_playback_start"
            ] = timeline_playback_start
            print(
                "F2_TIMELINE_PLAYBACK_START "
                + json.dumps(timeline_playback_start, sort_keys=True),
                flush=True,
            )
            if physics_substeps == 2:
                timeline_physics_manager_baseline = {
                    "num_physics_steps": int(
                        SimulationManager.get_num_physics_steps()
                    ),
                    "simulation_time_s": float(
                        SimulationManager.get_simulation_time()
                    ),
                }
        else:
            multitick_setting_path = "/rtx/hydra/supportMultiTickRate"
            multitick_enabled = bool(
                settings.get_as_bool(multitick_setting_path)
            )
            receipt["timing"]["multitick_support"] = {
                "setting_path": multitick_setting_path,
                "enabled_readback": multitick_enabled,
            }
            if not multitick_enabled:
                raise RuntimeError(
                    "Isaac 6.2 direct stepping requires Hydra multi-tick support"
                )
            SimulationManager.initialize_physics()
            timeline.pause()
            timeline.commit()
            paused_timeline_time_s = float(timeline.get_current_time())
            if bool(timeline.is_playing()):
                raise RuntimeError(
                    "timeline remained playing before direct physics stepping"
                )
            direct_step_baseline = read_direct_physics_state()
            receipt["timing"]["direct_step_initialization"] = {
                "timeline_playing": False,
                "paused_timeline_time_s": paused_timeline_time_s,
                "baseline_num_physics_steps": direct_step_baseline[
                    "num_physics_steps"
                ],
                "baseline_simulation_time_s": direct_step_baseline[
                    "simulation_time_s"
                ],
                "update_fabric": False,
            }
            receipt["timing"]["timeline_playback_start"] = None
            print(
                "F2_DIRECT_STEP_INITIALIZATION "
                + json.dumps(
                    receipt["timing"]["direct_step_initialization"],
                    sort_keys=True,
                ),
                flush=True,
            )

        def on_physics_step(callback_dt: float) -> None:
            record = {
                "callback_index": len(callback_records) + 1,
                "dt_s": float(callback_dt),
                "timeline_time_s": float(timeline.get_current_time()),
            }
            if timeline_owned_count_driven:
                record["app_update_index"] = timeline_app_update_index
            else:
                record["direct_step_index"] = active_direct_step_index
            callback_records.append(record)

        physics_step_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_physics_step
        )

        if args.realtime_video:
            _set_rtx_render_mode(
                settings,
                receipt,
                mode="rt",
                phase="realtime-video-sequence",
            )

        failure_phase = (
            "run_particle_timeline"
            if timeline_owned_count_driven
            else "run_particle_direct_steps"
        )
        for global_step in range(1, qualification_target_step + 1):
            check_time_limits()
            active_tank = _active_tank_for_step(plan, global_step)
            active_stream_source = _active_stream_source_for_step(
                plan, global_step
            )
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
                emission_state = _prepare_emission_event(
                    event, component_counts=component_counts
                )
                source_state = _source_state_for_event(event)
                if (
                    source_state.scheduled_tank != event["tank_id"]
                    or not np.allclose(
                        emission_state["particle_composition"],
                        source_state.birth_composition,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or int(emission_state["prototype_index"])
                    != source_state.palette_prototype_index
                ):
                    raise RuntimeError(
                        "SourceState stream/emission composition identity drifted"
                    )
                source_compositions = emission_state["source_compositions"]
                next_component_counts = emission_state[
                    "next_component_counts"
                ]
                emitted_prototype_index = source_state.palette_prototype_index
                scheduled_particle_id_range = event.get("particle_id_range")
                if receipt.get("execution_schedule") is not None:
                    particle_id_range_match = (
                        isinstance(scheduled_particle_id_range, list)
                        and scheduled_particle_id_range
                        == [next_particle_id, next_particle_id + count]
                    )
                    execution_event_attestations.append(
                        {
                            "pour_ordinal": int(event["pour_ordinal"]),
                            "emission_ordinal": int(event["ordinal"]),
                            "physics_step": int(event["physics_step"]),
                            "particle_count": count,
                            "particle_id_range": [
                                next_particle_id,
                                next_particle_id + count,
                            ],
                            "scheduled_particle_id_range": list(
                                scheduled_particle_id_range or []
                            ),
                            "particle_id_range_match": particle_id_range_match,
                            "particle_composition": list(
                                emission_state["particle_composition"]
                            ),
                            "composition_rows_match": bool(
                                emission_state["composition_rows_match"]
                            ),
                            "prototype_index": emitted_prototype_index,
                        }
                    )
                    if (
                        not particle_id_range_match
                        or emission_state["composition_rows_match"] is not True
                    ):
                        raise RuntimeError(
                            "scheduled emission identity or composition drifted"
                        )
                color = _global_transmission_color(
                    next_component_counts, tank_extinctions
                )

                def append_and_update_color() -> tuple[float, float, float]:
                    nonlocal shared_particle_points, next_particle_id
                    positions = _twin_chunk_positions(
                        Gf,
                        count,
                        rng,
                        resolution_profile_id=getattr(
                            args,
                            "particle_resolution_profile",
                            LEGACY_RESOLUTION_PROFILE_ID,
                        ),
                        reference_particle_count=event.get(
                            "source_reference_particle_count"
                        ),
                        attestation_sink=receipt["source_emission_geometry"][
                            "events"
                        ],
                    )
                    existing_positions = _positions(
                        shared_particle_points.GetPrim(), instancer=True
                    )
                    existing_simulation_positions = list(
                        shared_particle_points.GetSimulationPointsAttr().Get() or []
                    )
                    existing_velocities = list(
                        shared_particle_points.GetVelocitiesAttr().Get() or []
                    )
                    existing_orientations = list(
                        shared_particle_points.GetOrientationsAttr().Get() or []
                    )
                    existing_scales = list(
                        shared_particle_points.GetScalesAttr().Get() or []
                    )
                    existing_ids = [
                        int(value)
                        for value in (shared_particle_points.GetIdsAttr().Get() or [])
                    ]
                    if (
                        len(existing_positions) != len(existing_velocities)
                        or len(existing_positions) != len(existing_orientations)
                        or len(existing_positions) != len(existing_simulation_positions)
                        or len(existing_positions) != len(existing_scales)
                        or len(existing_positions) != len(existing_ids)
                        or existing_ids != shared_particle_ids
                    ):
                        raise RuntimeError("shared particle row attributes lost identity")
                    combined_positions = [*existing_positions, *positions]
                    combined_simulation_positions = [
                        *existing_simulation_positions,
                        *positions,
                    ]
                    shared_particle_points.GetSimulationPointsAttr().Set(
                        Vt.Vec3fArray(combined_simulation_positions)
                    )
                    shared_particle_points.GetPositionsAttr().Set(
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
                    shared_particle_points.GetOrientationsAttr().Set(
                        Vt.QuathArray(
                            [
                                *existing_orientations,
                                *[Gf.Quath(1.0, 0.0, 0.0, 0.0)] * count,
                            ]
                        )
                    )
                    shared_particle_points.GetScalesAttr().Set(
                        Vt.Vec3fArray(
                            [
                                *existing_scales,
                                *[Gf.Vec3f(1.0, 1.0, 1.0)] * count,
                            ]
                        )
                    )
                    shared_particle_points.CreateIdsAttr().Set(
                        Vt.Int64Array([*existing_ids, *particle_ids])
                    )
                    shared_particle_points.GetProtoIndicesAttr().Set(
                        Vt.IntArray(
                            [
                                *list(shared_particle_points.GetProtoIndicesAttr().Get() or []),
                                *[emitted_prototype_index] * count,
                            ]
                        )
                    )
                    return _set_global_blend_color(global_shader, Gf, color)

                last_global_color_readback = _with_session_edit_context(
                    Usd,
                    stage,
                    session_layer,
                    append_and_update_color,
                )
                initial_positions = [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in _positions(shared_particle_points.GetPrim(), instancer=True)[-count:]
                ]
                active_row_metadata.append(
                    {
                        "particle_ids": particle_ids,
                        "particle_count": count,
                        "source_compositions": source_compositions,
                        "tank_id": event["tank_id"],
                        "scheduled_source": source_state.scheduled_tank,
                        "birth_composition": list(source_state.birth_composition),
                        "pour_event_id": source_state.pour_event_id,
                        "birth_event_id": (
                            source_state.birth_event_id
                        ),
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
                emission_boundary = _emission_callback_boundary(
                    report_step=global_step,
                    physics_substeps=physics_substeps,
                )
                trace_row = {
                    "update_ordinal": global_blend_update_count,
                    "pour_ordinal": int(event["pour_ordinal"]),
                    "tank_id": str(event["tank_id"]),
                    "emission_ordinal": int(event["ordinal"]),
                    "physics_step": int(event["physics_step"]),
                    **emission_boundary,
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
                if receipt.get("execution_schedule") is not None:
                    trace_row.update(
                        {
                            "particle_id_range": [
                                int(particle_ids[0]),
                                int(particle_ids[-1] + 1),
                            ],
                            "particle_composition": list(
                                emission_state["particle_composition"]
                            ),
                            "particle_composition_rows": len(
                                source_compositions
                            ),
                            "prototype_index": emitted_prototype_index,
                            "cumulative_particle_counts": (
                                component_counts.tolist()
                            ),
                            "cumulative_particle_count": cumulative,
                        }
                    )
                emission_trace.append(trace_row)
                actual_events[int(event["pour_ordinal"])].append(
                    {
                        key: int(event[key])
                        for key in ("ordinal", "physics_step", "particle_count")
                    }
                )
            stream_surface_positions = [
                [float(value[0]), float(value[1]), float(value[2])]
                for value in (
                    shared_particle_points.GetSimulationPointsAttr().Get() or []
                )
            ]
            visible_streams = set_stream_state(
                active_stream_source, stream_surface_positions
            )
            for tank_id in visible_streams:
                stream_active_steps[tank_id].append(global_step)
            if timeline_owned_count_driven:
                before_callback_count = len(callback_records)
                timeline_app_update_index += 1
                if physics_substeps == 2:
                    physics_steps_before = int(
                        SimulationManager.get_num_physics_steps()
                    )
                    simulation_time_before_s = float(
                        SimulationManager.get_simulation_time()
                    )
                    timeline_time_before_s = float(timeline.get_current_time())
                if (
                    gating.get("current_updates_enabled") is not False
                    or render_product.hydra_texture.get_updates_enabled()
                    is not False
                ):
                    raise RuntimeError(
                        "render product was enabled during a physics-only app update"
                    )
                physics_update_started = time.monotonic()
                try:
                    _app_update_in_session(
                        stage, session_layer, app, Usd
                    )
                finally:
                    _record_phase_timing(
                        phase_timings,
                        "physics_app_update",
                        time.monotonic() - physics_update_started,
                    )
                if physics_substeps == 2:
                    physics_steps_after = int(
                        SimulationManager.get_num_physics_steps()
                    )
                    simulation_time_after_s = float(
                        SimulationManager.get_simulation_time()
                    )
                    timeline_time_after_s = float(timeline.get_current_time())
                gating["physics_updates_while_disabled"] = int(
                    gating.get("physics_updates_while_disabled", 0)
                ) + 1
                gating["physics_loop_app_update_count"] = int(
                    gating.get("physics_loop_app_update_count", 0)
                ) + 1
                receipt["timing"]["physics_loop_app_update_count"] = int(
                    receipt["timing"].get(
                        "physics_loop_app_update_count", 0
                    )
                ) + 1
                if physics_substeps == 2:
                    callback_contract = (
                        _timeline_callback_group_contract_evidence(
                            callback_records[before_callback_count:],
                            report_index=global_step,
                            app_update_index=timeline_app_update_index,
                            physics_substeps=physics_substeps,
                            callback_count_before=before_callback_count,
                            physics_steps_before=physics_steps_before,
                            physics_steps_after=physics_steps_after,
                            simulation_time_before_s=(
                                simulation_time_before_s
                            ),
                            simulation_time_after_s=simulation_time_after_s,
                            timeline_time_before_s=timeline_time_before_s,
                            timeline_time_after_s=timeline_time_after_s,
                        )
                    )
                    timeline_callback_group_records.append(callback_contract)
                else:
                    if len(callback_records) != before_callback_count + 1:
                        raise RuntimeError(
                            "timeline produced not exactly one callback per app update"
                        )
                    callback_contract = _timeline_callback_contract_evidence(
                        callback_records[-1],
                        global_step=global_step,
                        app_update_index=timeline_app_update_index,
                        physics_hz=REPORT_HZ,
                    )
                if callback_contract["passed"] is not True:
                    diagnostic_path = output / "timeline-callback-failure.json"
                    _write_json(diagnostic_path, callback_contract)
                    receipt["timing"]["callback_contract_failure"] = {
                        **callback_contract,
                        "artifact_path": str(
                            diagnostic_path.relative_to(output)
                        ),
                        "artifact_sha256": _sha256(diagnostic_path),
                    }
                    raise RuntimeError(
                        "timeline callback group differs from the frozen report/PhysX contract: "
                        + ",".join(
                            callback_contract["failed_predicates"]
                        )
                    )
            else:
                active_direct_step_index = global_step
                if (
                    gating.get("current_updates_enabled") is not True
                    or render_product.hydra_texture.get_updates_enabled()
                    is not True
                ):
                    raise RuntimeError(
                        "persistent render product was not enabled during direct physics"
                    )
                physics_update_started = time.monotonic()
                try:
                    direct_step_evidence = _direct_physics_step_checked(
                        SimulationManager=SimulationManager,
                        callback_records=callback_records,
                        direct_step_index=global_step,
                        physics_hz=PHYSICS_HZ,
                        timeline=timeline,
                        paused_timeline_time_s=paused_timeline_time_s,
                    )
                finally:
                    _record_phase_timing(
                        phase_timings,
                        "physics_direct_step",
                        time.monotonic() - physics_update_started,
                    )
                gating["physics_steps_while_enabled"] = int(
                    gating.get("physics_steps_while_enabled", 0)
                ) + 1
                receipt["timing"]["direct_physx_api_used"] = True
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
                    "runtime_particle_ids_exact_each_report_interval": True,
                    "runtime_particle_identity_check_count": particle_identity_check_count,
                    "runtime_particle_id_mismatch_count": particle_identity_mismatch_count,
                    "runtime_particle_ids_final_sha256": _particle_ids_sha256(
                        observed_particle_ids
                    ),
                }
            )
            # The Paint Ball PointInstancer is created before the timeline;
            # no deferred mesh/proxy lifecycle is called at this boundary.
            positions_after_physics = _direct_particle_vec3_array(
                np,
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (shared_particle_points.GetSimulationPointsAttr().Get() or [])
                ]
                if shared_particle_points is not None
                else [],
            )
            velocities_after_physics = _direct_particle_vec3_array(
                np,
                [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in (
                        shared_particle_points.GetVelocitiesAttr().Get() or []
                    )
                ]
                if shared_particle_points is not None
                else [],
            )
            particle_state_passed = bool(
                positions_after_physics.shape
                == velocities_after_physics.shape
                == (len(observed_particle_ids), 3)
                and np.all(np.isfinite(positions_after_physics))
                and np.all(np.isfinite(velocities_after_physics))
            )
            if not particle_state_passed:
                raise RuntimeError(
                    "physics step produced missing or non-finite particle state"
                )
            if global_step == MOTION_FALSIFIER_STEP:
                stable_particle_count = (
                    int(active_row_metadata[0]["particle_count"])
                    if active_row_metadata
                    else 0
                )
                motion_evidence = _runtime_particle_motion_evidence(
                    initial_positions=shared_initial_positions,
                    current_positions=positions_after_physics,
                    current_velocities=velocities_after_physics,
                    stable_particle_count=stable_particle_count,
                )
                motion_evidence.update(
                    {
                        "probe_step": global_step,
                        "simulation_owner": receipt["timing"][
                            "simulation_owner"
                        ],
                    }
                )
                motion_path = output / "runtime-motion-falsifier.json"
                _write_json(motion_path, motion_evidence)
                receipt["timing"]["motion_falsifier"] = {
                    **motion_evidence,
                    "status": (
                        "pass" if motion_evidence["passed"] else "failed"
                    ),
                    "artifact_path": str(motion_path.relative_to(output)),
                    "artifact_sha256": _sha256(motion_path),
                }
                print(
                    "F2_RUNTIME_MOTION_FALSIFIER "
                    + json.dumps(motion_evidence, sort_keys=True),
                    flush=True,
                )
                if motion_evidence["passed"] is not True:
                    raise RuntimeError(
                        "runtime particle motion falsifier rejected callback-only progress"
                    )
            if not timeline_owned_count_driven:
                direct_step_evidence["particle_state"] = {
                    "particle_count": len(observed_particle_ids),
                    "positions_shape": list(positions_after_physics.shape),
                    "velocities_shape": list(velocities_after_physics.shape),
                    "finite_positions": True,
                    "finite_velocities": True,
                    "ids_exact": observed_particle_ids == shared_particle_ids,
                    "passed": True,
                }
                direct_step_probe = receipt["timing"]["direct_step_probe"]
                if global_step <= int(direct_step_probe["step_limit"]):
                    direct_step_probe["completed_steps"] = global_step
                    direct_step_probe["emission_events_observed"] = int(
                        direct_step_probe["emission_events_observed"]
                    ) + len(event_index.get(global_step, []))
                    direct_step_probe["step_evidence"].append(
                        direct_step_evidence
                    )
                    if global_step == int(direct_step_probe["step_limit"]):
                        direct_step_probe["physics_contract_passed"] = bool(
                            len(direct_step_probe["step_evidence"])
                            == int(direct_step_probe["step_limit"])
                            and direct_step_probe["emission_events_observed"]
                            > 0
                            and all(
                                row.get("passed") is True
                                and row.get("particle_state", {}).get(
                                    "passed"
                                )
                                is True
                                for row in direct_step_probe[
                                    "step_evidence"
                                ]
                            )
                        )
                        if not direct_step_probe["physics_contract_passed"]:
                            raise RuntimeError(
                                "bounded direct-step physics probe failed"
                            )
                        direct_step_probe["status"] = (
                            "physics_passed_capture_pending"
                        )
            diffusion_started = time.monotonic()
            try:
                (
                    _updated_compositions,
                    diffusion_diagnostics,
                ) = _diffuse_active_particle_rows(
                    active_row_metadata,
                    compositions_by_id,
                    positions_after_physics,
                    observed_particle_ids,
                    dt_s=1.0 / float(REPORT_HZ),
                )
            finally:
                _record_phase_timing(
                    phase_timings,
                    "diffusion",
                    time.monotonic() - diffusion_started,
                )
            # Paint Ball Emitter visual state follows the tracked composition;
            # nearest frozen palette prototype is recomputed every physics update.
            palette_started = time.monotonic()
            mapped_prototypes = None
            if len(_updated_compositions):
                mapped_prototypes = map_compositions_to_palette(_updated_compositions)
            _record_phase_timing(
                phase_timings,
                "palette_mapping",
                time.monotonic() - palette_started,
            )
            prototype_started = time.monotonic()
            if mapped_prototypes is not None:
                with Usd.EditContext(stage, session_layer):
                    shared_particle_points.GetProtoIndicesAttr().Set(
                        Vt.IntArray([int(value) for value in mapped_prototypes.tolist()])
                    )
            _record_phase_timing(
                phase_timings,
                "prototype_authoring",
                time.monotonic() - prototype_started,
            )
            if cumulative > 0:
                mean_composition = _updated_compositions.mean(axis=0)
                mean_color = mix_extinction_transmission(
                    mean_composition[None, :], tank_extinctions
                )[0]
                if not np.allclose(
                    mean_color,
                    _global_transmission_color(
                        component_counts, tank_extinctions
                    ),
                    rtol=0.0,
                    atol=1.0e-10,
                ):
                    raise RuntimeError(
                        "mean composition drifted from the conserved global render colour"
                    )
            if count_driven_body_runtime is not None:
                contained_mask = _count_driven_contained_mask(
                    positions_after_physics
                )

                def update_count_driven_body() -> dict[str, Any]:
                    return _update_count_driven_twin_liquid_body(
                        count_driven_body_runtime,
                        contained_positions=positions_after_physics[
                            contained_mask
                        ],
                        contained_compositions=_updated_compositions[contained_mask],
                        global_shader=global_shader,
                        source_transmissions=tank_transmission_matrix,
                        Usd=Usd,
                        UsdGeom=UsdGeom,
                        Gf=Gf,
                        Vt=Vt,
                    )

                latest_count_driven_body_geometry = _with_session_edit_context(
                    Usd,
                    stage,
                    session_layer,
                    update_count_driven_body,
                )
                last_global_color_readback = tuple(
                    float(value)
                    for value in latest_count_driven_body_geometry[
                        "body_material_color"
                    ]
                )
                receipt["count_driven_liquid_body"]["update_count"] = (
                    int(
                        receipt["count_driven_liquid_body"].get(
                            "update_count", 0
                        )
                    )
                    + 1
                )
                # The visible body has just been resized from contained
                # volume.  Re-anchor the render-only stream to that exact
                # surface before any capture in this state.
                visible_streams = set_stream_state(
                    active_stream_source,
                    positions_after_physics,
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
            receipt["timing"].update(
                {
                    "executed_report_intervals": global_step,
                    "executed_physics_steps": len(callback_records),
                    "diffusion_update_count": diffusion_update_count,
                    "diffusion_integrated_time_s": (
                        diffusion_update_count / float(REPORT_HZ)
                    ),
                }
            )

            if args.realtime_video and global_step in realtime_video_steps:
                failure_phase = "capture_realtime_video_frame"
                realtime_rgb = acquire_rgb(
                    global_step,
                    "realtime-video",
                    included_in_video=True,
                    rt_subframes_override=1,
                )
                realtime_bgr = cv2.cvtColor(
                    realtime_rgb, cv2.COLOR_RGB2BGR
                )
                writer.write(realtime_bgr)
                video_frame_source_capture_indices.append(
                    len(video_frame_steps)
                )
                video_frame_steps.append(global_step)
                video_frame_rgb_hashes.append(
                    hashlib.sha256(
                        np.asarray(realtime_rgb, dtype=np.uint8).tobytes()
                    ).hexdigest()
                )

            if global_step not in capture_steps:
                continue
            receipt["totals"]["final_particle_count"] = len(
                positions_after_physics
            )
            stage_record = stage_by_step.get(global_step)
            if stage_record is not None:
                stage_phase = f"{stage_record[1]['tank_id']}-stage"
                visible_streams = set_stream_state(
                    None,
                    positions_after_physics,
                )
                phase = stage_phase
            elif active_tank is not None:
                phase = f"{active_tank}-pour"
            else:
                phase = "post-fill"
            if (
                stage_record is not None
                and count_driven_body_runtime is not None
                and receipt["semantics"].get("hidden_particle_control") is None
            ):
                failure_phase = "capture_hidden_emitted_particle_control"
                capture_hidden_particle_control(
                    step=global_step,
                    emitted_particle_count=len(positions_after_physics),
                    inside_particle_count=int(np.count_nonzero(contained_mask)),
                )
                failure_phase = "capture_paired_frame"
            if args.realtime_video:
                _set_rtx_render_mode(
                    settings,
                    receipt,
                    mode="pt",
                    phase=f"pathtraced-stage-step-{global_step}",
                )
            frame_path, positions, _ = capture(
                global_step,
                phase,
                visible_streams,
            )
            if args.realtime_video:
                _set_rtx_render_mode(
                    settings,
                    receipt,
                    mode="rt",
                    phase=f"resume-realtime-after-stage-{global_step}",
                )
            _sample_periodic_capture_memory(
                receipt,
                completed_frame_count=len(frame_paths),
                physics_step=global_step,
            )
            direct_step_probe = receipt["timing"].get("direct_step_probe")
            if (
                not timeline_owned_count_driven
                and isinstance(direct_step_probe, dict)
                and global_step
                == int(direct_step_probe["first_capture_step"])
            ):
                probe_capture_guards = [
                    row
                    for row in capture_guard_records
                    if int(row.get("physics_step", -1)) == global_step
                ]
                direct_step_probe["capture_contract_passed"] = bool(
                    len(probe_capture_guards) >= 2
                    and all(
                        int(row["callbacks_before"])
                        == int(row["callbacks_after"])
                        and int(row["physics_steps_before"])
                        == int(row["physics_steps_after"])
                        and math.isclose(
                            float(row["simulation_time_before_s"]),
                            float(row["simulation_time_after_s"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        and math.isclose(
                            float(row["timeline_time_before_s"]),
                            float(row["timeline_time_after_s"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        for row in probe_capture_guards
                    )
                )
                if (
                    direct_step_probe["physics_contract_passed"] is not True
                    or direct_step_probe["capture_contract_passed"] is not True
                ):
                    direct_step_probe["status"] = "failed"
                    raise RuntimeError(
                        "bounded direct-step capture probe failed"
                    )
                direct_step_probe["status"] = "pass"
                print(
                    "F2_DIRECT_STEP_CAPTURE_PROBE "
                    + json.dumps(
                        {
                            key: direct_step_probe[key]
                            for key in (
                                "step_limit",
                                "first_capture_step",
                                "completed_steps",
                                "emission_events_observed",
                                "physics_contract_passed",
                                "capture_contract_passed",
                                "status",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
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
            observed_color = _global_transmission_color(
                component_counts, tank_extinctions
            ).tolist()
            if last_global_color_readback is None or not np.allclose(
                last_global_color_readback,
                observed_color,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("stage global blend lacks matching live USD readback")
            if (
                not np.allclose(
                    component_counts,
                    expected_state["cumulative_particle_counts"],
                    rtol=0.0,
                    atol=1.0e-9,
                )
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
                    "spatial_mixing_color_rendered": (
                        count_driven_body_runtime is None
                        or args.height_banded_body
                    ),
                    "spatial_layering_color_rendered": (
                        count_driven_body_runtime is None
                        or args.height_banded_body
                    ),
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
            instancer_prim_for_stage = stage.GetPrimAtPath(
                str(receipt["point_instancer"]["path"])
            )
            if not instancer_prim_for_stage.IsValid() or not instancer_prim_for_stage.IsA(UsdGeom.PointInstancer):
                raise RuntimeError("shared Paint Ball PointInstancer is missing")
            labels = {
                str(value)
                for value in UsdSemantics.LabelsQuery(
                    "class", Usd.TimeCode.Default()
                ).ComputeUniqueInheritedLabels(instancer_prim_for_stage)
            }
            if "liquid" not in labels:
                raise RuntimeError(
                    f"PointInstancer lacks the liquid semantic label: {sorted(labels)!r}"
                )
            receipt["semantics"]["point_instancer_semantic_bound"] = True
            if count_driven_body_runtime is not None:
                body_prim_for_stage = stage.GetPrimAtPath(
                    COUNT_DRIVEN_BODY_MESH_PATH
                )
                if (
                    not body_prim_for_stage.IsValid()
                    or not body_prim_for_stage.IsA(UsdGeom.Mesh)
                ):
                    raise RuntimeError("count-driven liquid body mesh is missing")
                body_labels = {
                    str(value)
                    for value in UsdSemantics.LabelsQuery(
                        "class", Usd.TimeCode.Default()
                    ).ComputeUniqueInheritedLabels(body_prim_for_stage)
                }
                if "liquid" not in body_labels:
                    raise RuntimeError(
                        "count-driven body lacks the liquid semantic label: "
                        f"{sorted(body_labels)!r}"
                    )
                receipt["semantics"][
                    "count_driven_body_semantic_bound"
                ] = True
            memory_after = _memory_sample(receipt)
            memory_before = stage_memory_before[stage_index]
            if _stage_memory_growth_stop_required(
                count_driven_twin_liquid_mesh=(
                    count_driven_body_runtime is not None
                ),
                memory_before_mib=memory_before,
                memory_after_mib=memory_after,
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
                "point_instancer_nonempty": True,
                "used_gpu_memory_mib_before": stage_memory_before[stage_index],
                "used_gpu_memory_mib_after": memory_after,
                "global_component_counts_particles": component_counts.tolist(),
                "global_composition_fractions": fractions.tolist(),
                "global_transmission_color": _global_transmission_color(
                    component_counts, tank_extinctions
                ).tolist(),
            }
            receipt["pours"].append(row)
            if row["escaped_particle_count"]:
                raise RuntimeError(f"{pour['tank_id']} has escaped particles")

        if timeline_owned_count_driven and physics_substeps == 2:
            if timeline_physics_manager_baseline is None:
                raise RuntimeError("timeline physics-manager baseline is missing")
            manager_final_steps = int(
                SimulationManager.get_num_physics_steps()
            )
            manager_final_time_s = float(
                SimulationManager.get_simulation_time()
            )
            manager_step_delta = (
                manager_final_steps
                - int(timeline_physics_manager_baseline["num_physics_steps"])
            )
            manager_time_delta_s = (
                manager_final_time_s
                - float(timeline_physics_manager_baseline["simulation_time_s"])
            )
            expected_callback_count = int(
                physics_clock["physics_callback_count"]
            )
            manager_passed = bool(
                manager_step_delta == expected_callback_count
                and math.isclose(
                    manager_time_delta_s,
                    float(physics_clock["physical_duration_s"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            )
            manager_readback = {
                "baseline_num_physics_steps": int(
                    timeline_physics_manager_baseline["num_physics_steps"]
                ),
                "final_num_physics_steps": manager_final_steps,
                "delta_num_physics_steps": manager_step_delta,
                "baseline_simulation_time_s": float(
                    timeline_physics_manager_baseline["simulation_time_s"]
                ),
                "final_simulation_time_s": manager_final_time_s,
                "delta_simulation_time_s": manager_time_delta_s,
                "passed": manager_passed,
            }
            callback_groups_passed = bool(
                len(timeline_callback_group_records)
                == qualification_target_step
                and all(
                    row.get("passed") is True
                    for row in timeline_callback_group_records
                )
                and len(callback_records) == expected_callback_count
                and manager_passed
            )
            callback_group_trace_path = (
                output / "timeline-callback-groups.json"
            )
            _write_json(
                callback_group_trace_path,
                {
                    "clock": physics_clock,
                    "group_count": len(timeline_callback_group_records),
                    "groups": timeline_callback_group_records,
                    "physics_manager_readback": manager_readback,
                    "passed": callback_groups_passed,
                },
            )
            receipt["timing"].update(
                {
                    "callbacks_per_app_update_contract_passed": (
                        callback_groups_passed
                    ),
                    "callback_group_trace_path": str(
                        callback_group_trace_path.relative_to(output)
                    ),
                    "callback_group_trace_sha256": _sha256(
                        callback_group_trace_path
                    ),
                    "callback_group_trace_count": len(
                        timeline_callback_group_records
                    ),
                    "physics_manager_readback": manager_readback,
                }
            )
            if not callback_groups_passed:
                raise RuntimeError(
                    "timeline callback grouping or physics-manager totals drifted"
                )

        if args.qualification_mode:
            qualification_frames = [
                {
                    "physics_step": int(step),
                    "rgb_path": str(path.relative_to(output)),
                    "rgb_sha256": _sha256(path),
                    "semantic_path": str(semantic_paths[index].relative_to(output)),
                    "semantic_sha256": _sha256(semantic_paths[index]),
                    "mask_path": str(liquid_mask_paths[index].relative_to(output)),
                    "mask_sha256": _sha256(liquid_mask_paths[index]),
                    "liquid_pixel_count": int(liquid_pixel_counts[index]),
                    "liquid_pixel_count_in_first_vial_bottom_roi": int(
                        semantic_mask_records[index].get(
                            "liquid_pixel_count_in_first_vial_bottom_roi", 0
                        )
                    ),
                    "visible_stream_tanks": list(
                        semantic_mask_records[index].get("visible_stream_tanks", [])
                    ),
                    "shared_rgb_semantic_capture": semantic_mask_records[
                        index
                    ].get("shared_rgb_semantic_capture"),
                    "paired_same_physics_state_rgb_semantic_capture": (
                        semantic_mask_records[index].get(
                            "paired_same_physics_state_rgb_semantic_capture"
                        )
                    ),
                    "paired_capture_state_passed": semantic_mask_records[
                        index
                    ].get("paired_capture_state_passed"),
                    "paired_capture_state_evidence": semantic_mask_records[
                        index
                    ].get("paired_capture_state_evidence"),
                    "liquid_representation": semantic_mask_records[index].get(
                        "liquid_representation"
                    ),
                    **{
                        key: semantic_mask_records[index][key]
                        for key in (
                            "body_points_sha256",
                            "body_extent_sha256",
                            "body_visibility",
                            "body_material_color_sha256",
                            "stream_visibility_sha256",
                            "compositions_sha256",
                        )
                        if key in semantic_mask_records[index]
                    },
                }
                for index, (step, path) in enumerate(zip(frame_steps, frame_paths))
            ]
            receipt["qualification"].update(
                {
                    "status": "completed_first_stream_hidden_stage_only",
                    "target_step": qualification_target_step,
                    "capture_steps": sorted(capture_steps),
                    "full_run_not_attempted": True,
                    "semantic_blocker": None,
                    "paired_capture_evidence": qualification_frames,
                    "instance_segmentation_diagnostic": receipt["point_instancer"].get(
                        "instance_segmentation_diagnostic"
                    ),
                    "containment_gate_passed": not receipt["containment"]["leakage_detected"],
                }
            )
            target_records = [
                record
                for record in qualification_frames
                if int(record["physics_step"]) == qualification_target_step
            ]
            target_volume_records = [
                record
                for record in receipt["containment"]["capture_records"]
                if int(record["physics_step"]) == qualification_target_step
                and isinstance(record.get("volume_gate"), dict)
                and record["volume_gate"].get("required") is True
            ]
            target_volume_gate = (
                target_volume_records[0]["volume_gate"]
                if len(target_volume_records) == 1
                else None
            )
            volume_state_gate_passed = bool(
                isinstance(target_volume_gate, dict)
                and target_volume_gate.get("state_passed") is True
            )
            volume_visible_gate_passed = bool(
                isinstance(target_volume_gate, dict)
                and target_volume_gate.get("visible_passed") is True
            )
            volume_gate_passed = bool(
                isinstance(target_volume_gate, dict)
                and target_volume_gate.get("passed") is True
            )
            receipt["qualification"].update(
                {
                    "volume_state_gate_passed": volume_state_gate_passed,
                    "volume_visible_gate_passed": volume_visible_gate_passed,
                    "volume_gate_passed": volume_gate_passed,
                    "target_volume_evidence": target_volume_gate,
                }
            )
            semantic_gate = _qualification_semantic_gate(
                records=qualification_frames,
                target_step=qualification_target_step,
                capture_steps=capture_steps,
                frame_count=len(frame_paths),
                frame_pixel_count=width * height,
                point_instancer_semantic_bound=(
                    receipt["semantics"].get(
                        "count_driven_body_semantic_bound"
                        if count_driven_body_runtime is not None
                        else "point_instancer_semantic_bound"
                    )
                    is True
                ),
                stream_semantic_bound=(
                    receipt["semantics"].get("stream_semantic_bound") is True
                ),
            )
            target_roi_count = (
                int(
                    target_records[0][
                        "liquid_pixel_count_in_first_vial_bottom_roi"
                    ]
                )
                if len(target_records) == 1
                else None
            )
            target_roi_passed = bool(
                target_roi_count is not None
                and target_roi_count
                >= FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS
            )
            receipt["qualification"]["semantic_gate_passed"] = semantic_gate
            receipt["qualification"].update(
                {
                    "target_liquid_pixel_count_in_first_vial_bottom_roi": (
                        target_roi_count
                    ),
                    "target_liquid_roi_threshold_passed": target_roi_passed,
                    "semantic_blocker": (
                        None
                        if semantic_gate
                        else (
                            "first_vial_bottom_roi_below_minimum"
                            if not target_roi_passed
                            else "qualification_semantic_contract_failed"
                        )
                    ),
                }
            )
            qualification_passed = bool(
                semantic_gate
                and receipt["qualification"]["containment_gate_passed"]
                and volume_gate_passed
                and receipt["pre_pour_parity"].get("passed") is True
                and receipt["timing"].get("physics_hz") == 120
            )
            receipt["scene_contract"]["source_layers_unchanged"] = all(
                Path(row["resolved_path"]).is_file()
                and _sha256(Path(row["resolved_path"])) == row["sha256"]
                for row in layer_records
            )
            if receipt["scene_contract"]["source_layers_unchanged"] is not True:
                raise RuntimeError("a source USD layer changed during qualification")
            receipt["qualification"]["qualification_hypotheses_passed"] = qualification_passed
            receipt["status"] = "pass" if qualification_passed else "insufficient_evidence"
            receipt["qualification"]["status"] = (
                "passed_first_stream_hidden_stage_only"
                if qualification_passed
                else "qualification_gate_failed"
            )
            scope_note = "qualification-first-stage-only completed; no 60mL pass claim is made"
            receipt["qualification"]["scope_note"] = scope_note
            (receipt["warnings"] if qualification_passed else receipt["errors"]).append(
                scope_note
            )
            receipt["semantics"]["pairs"] = qualification_frames
            receipt["timing"].update(
                {
                    "callbacks_observed": len(callback_records),
                    "callback_indices": [
                        int(row["callback_index"])
                        for row in callback_records
                    ],
                    "callback_dts_s": [
                        float(row["dt_s"]) for row in callback_records
                    ],
                    "callback_times_s": [
                        float(row["timeline_time_s"])
                        for row in callback_records
                    ],
                    "callback_direct_step_indices": [
                        int(row["direct_step_index"])
                        for row in callback_records
                    ] if not timeline_owned_count_driven else [],
                    "callback_app_update_indices": [
                        int(row["app_update_index"])
                        for row in callback_records
                    ] if timeline_owned_count_driven else [],
                    "callback_logical_times_s": [
                        int(
                            row[
                                "app_update_index"
                                if timeline_owned_count_driven
                                else "direct_step_index"
                            ]
                        )
                        / float(PHYSICS_HZ)
                        for row in callback_records
                    ],
                    "one_callback_per_direct_step": bool(
                        not timeline_owned_count_driven
                        and len(callback_records)
                        == qualification_target_step
                        and [
                            int(row["direct_step_index"])
                            for row in callback_records
                        ]
                        == list(range(1, qualification_target_step + 1))
                    ),
                    "one_callback_per_app_update": bool(
                        timeline_owned_count_driven
                        and len(callback_records)
                        == qualification_target_step
                        and [
                            int(row["app_update_index"])
                            for row in callback_records
                        ]
                        == list(range(1, qualification_target_step + 1))
                    ),
                    "physics_loop_app_update_count": (
                        timeline_app_update_index
                        if timeline_owned_count_driven
                        else 0
                    ),
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
                    "capture_callbacks_advanced": sum(
                        int(row["callbacks_after"])
                        - int(row["callbacks_before"])
                        for row in capture_guard_records
                    ),
                    "capture_direct_physics_steps_advanced": sum(
                        int(row["physics_steps_after"])
                        - int(row["physics_steps_before"])
                        for row in capture_guard_records
                    ) if not timeline_owned_count_driven else 0,
                    "capture_simulation_time_advanced": any(
                        not math.isclose(
                            float(row["simulation_time_before_s"]),
                            float(row["simulation_time_after_s"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        for row in capture_guard_records
                    ) if not timeline_owned_count_driven else False,
                    "capture_timeline_time_advanced": any(
                        not math.isclose(
                            float(row["timeline_time_before_s"]),
                            float(row["timeline_time_after_s"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        for row in capture_guard_records
                    ),
                    "capture_playback_preserved": all(
                        bool(row["timeline_playing_before"])
                        == bool(row["timeline_playing_after"])
                        for row in capture_guard_records
                    ),
                    "capture_guard_records": capture_guard_records,
                }
            )
            if timeline_owned_count_driven:
                final_updates_enabled = _set_render_product_updates(
                    render_product,
                    False,
                    gating,
                    phase="final_before_acceptance",
                )
                gating["final_disabled_readback"] = (
                    final_updates_enabled is False
                )
            else:
                final_updates_enabled = (
                    render_product.hydra_texture.get_updates_enabled()
                )
                gating["final_enabled_readback"] = (
                    final_updates_enabled is True
                )
                gating["current_updates_enabled"] = final_updates_enabled
                if final_updates_enabled is not True:
                    raise RuntimeError(
                        "persistent render product was disabled before qualification acceptance"
                    )
            receipt = _finalize_paint_ball_receipt(receipt)
            receipt["qualification"]["receipt_validation_failures"] = list(
                validate_paint_ball_receipt(receipt)
            )
            if receipt["qualification"]["receipt_validation_failures"]:
                receipt["status"] = "insufficient_evidence"
                receipt["errors"].extend(
                    receipt["qualification"]["receipt_validation_failures"]
                )
            return receipt

        expected_update_count = sum(
            int(pour["emission_chunk_count"]) for pour in plan["pours"]
        )
        if global_blend_update_count != expected_update_count:
            raise RuntimeError(
                f"global blend updated {global_blend_update_count} times; "
                f"expected {expected_update_count}"
            )
        if any(
            actual_events[int(pour["ordinal"])]
            != [
                {
                    key: int(event[key])
                    for key in ("ordinal", "physics_step", "particle_count")
                }
                for event in pour["emission_schedule"]
            ]
            for pour in plan["pours"]
        ):
            raise RuntimeError("executed emission schedule differs from the frozen plan")
        if receipt.get("execution_schedule") is not None:
            particle_ids_and_counts_match = bool(
                len(execution_event_attestations) == expected_update_count
                and all(
                    row["particle_id_range_match"] is True
                    and row["particle_count"]
                    == row["particle_id_range"][1]
                    - row["particle_id_range"][0]
                    for row in execution_event_attestations
                )
            )
            event_compositions_match = bool(
                len(execution_event_attestations) == expected_update_count
                and all(
                    row["composition_rows_match"] is True
                    for row in execution_event_attestations
                )
            )
            if not particle_ids_and_counts_match or not event_compositions_match:
                raise RuntimeError(
                    "scheduled emission attestation did not cover every event"
                )
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
        if receipt.get("execution_schedule") is not None:
            attestation_path = trace_dir / "execution-event-attestations.json"
            _write_json(
                attestation_path,
                {
                    "schedule_sha256": receipt["execution_schedule"][
                        "schedule_sha256"
                    ],
                    "event_count": len(execution_event_attestations),
                    "events": execution_event_attestations,
                },
            )
            receipt["execution_schedule"].update(
                {
                    "particle_ids_and_counts_match_every_event": (
                        particle_ids_and_counts_match
                    ),
                    "event_compositions_match": event_compositions_match,
                    "executed_emission_event_count": len(
                        execution_event_attestations
                    ),
                    "event_composition_attestation_path": str(
                        attestation_path.relative_to(output)
                    ),
                    "event_composition_attestation_sha256": _sha256(
                        attestation_path
                    ),
                }
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
        if diffusion_update_count != qualification_target_step:
            raise RuntimeError(
                "diffusion update count does not cover every report interval"
            )
        receipt["composition_evidence"].update(
            {
                "diffusion_update_count": diffusion_update_count,
                "diffusion_dt_s": 1.0 / float(REPORT_HZ),
                "diffusion_integrated_time_s": (
                    diffusion_update_count / float(REPORT_HZ)
                ),
                "diffusion_neighbor_count_total": diffusion_neighbor_count,
                "diffusion_max_degree": diffusion_max_degree,
                "diffusion_pair_weight_max": diffusion_pair_weight_max,
                "stage_count": len(receipt["composition_evidence"]["stages"]),
            }
        )

        stream_evidence_records = []
        expected_steps_by_source: dict[str, list[int]] = {
            tank_id: [] for tank_id in COMPOSITION_COMPONENTS
        }
        for source_row in receipt["source_states"]:
            expected_steps_by_source[str(source_row["scheduled_tank"])].extend(
                range(
                    int(source_row["start_step"]),
                    int(source_row["end_step"]) + 1,
                )
            )
        for tank_id, expected_steps in expected_steps_by_source.items():
            if stream_active_steps[tank_id] != expected_steps:
                raise RuntimeError(f"{tank_id} render-only stream timing drifted")
        for source_row in receipt["source_states"]:
            ordinal = int(source_row["pour_ordinal"])
            segment_ordinal = int(source_row["segment_ordinal"])
            tank_id = str(source_row["scheduled_tank"])
            start_step = int(source_row["start_step"])
            end_step = int(source_row["end_step"])
            observed_active_steps = [
                step
                for step in stream_active_steps[tank_id]
                if start_step <= step <= end_step
            ]
            state_path = stream_states_dir / (
                f"stream-{ordinal}-segment-{segment_ordinal}-{tank_id}.json"
            )
            state = {
                "ordinal": ordinal,
                "segment_ordinal": segment_ordinal,
                "segment_role": source_row["segment_role"],
                "programmed_tank": source_row["programmed_tank"],
                "tank_id": tank_id,
                "start_step": start_step,
                "end_step": end_step,
                "transmission_color": list(
                    mix_extinction_transmission(
                        [source_row["stream_composition"]]
                    )[0]
                ),
                "source_state_composition": list(source_row["composition"]),
                "emitted_birth_composition": list(
                    source_row["birth_composition"]
                ),
                "source_state_material_path": source_row["material_path"],
                "source_state_palette_prototype_index": (
                    source_row["palette_prototype_index"]
                ),
                "stream_and_birth_composition_equal": (
                    source_row["stream_composition"]
                    == source_row["birth_composition"]
                ),
                "active_only_during_own_pour": True,
                "active_only_during_own_segment": True,
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
                            "segment_ordinal",
                            "segment_role",
                            "programmed_tank",
                            "tank_id",
                            "start_step",
                            "end_step",
                            "transmission_color",
                            "source_state_composition",
                            "emitted_birth_composition",
                            "source_state_material_path",
                            "source_state_palette_prototype_index",
                            "stream_and_birth_composition_equal",
                            "active_only_during_own_pour",
                            "active_only_during_own_segment",
                        )
                    },
                    "state_artifact_path": str(state_path.relative_to(output)),
                    "state_artifact_sha256": _sha256(state_path),
                }
            )
        receipt["stream_render"]["records"] = stream_evidence_records
        receipt["timing"].update(
            {
                "executed_report_intervals": int(qualification_target_step),
                "executed_physics_steps": len(callback_records),
            }
        )

        if shared_particle_points is None:
            raise RuntimeError("timeline completed without a runtime particle set")
        final_particle_ids = _read_runtime_particle_identity(
            shared_particle_points, shared_particle_ids
        )
        if (
            particle_identity_check_count
            != qualification_target_step
            or particle_identity_mismatch_count != 0
        ):
            raise RuntimeError(
                "particle identity was not observed at every report interval"
            )
        expected_particle_count = int(plan["totals"]["particle_count"])
        containment_records = receipt["containment"]["capture_records"]
        final_containment_record = (
            containment_records[-1]
            if isinstance(containment_records, list) and containment_records
            else None
        )
        final_contained_particle_count = (
            final_containment_record.get("inside_vial_interior_count")
            if isinstance(final_containment_record, Mapping)
            else None
        )
        if (
            isinstance(final_contained_particle_count, bool)
            or not isinstance(final_contained_particle_count, int)
        ):
            raise RuntimeError("final contained particle count is missing")
        simulation_points_readback = (
            shared_particle_points.GetSimulationPointsAttr().Get()
        )
        simulation_points_evidence = _runtime_simulation_points_evidence(
            simulation_points_readback,
            expected_particle_count=expected_particle_count,
            particle_id_count=len(final_particle_ids),
            containment_particle_count=final_contained_particle_count,
        )
        shared_points_path = shared_particle_points.GetPrim().GetPath()
        runtime_layer_evidence = _runtime_particle_layer_evidence(
            stage=stage,
            session_layer=session_layer,
            shared_points_path=shared_points_path,
            Sdf=Sdf,
        )
        runtime_layer_evidence["simulation_points_runtime_evidence"] = (
            simulation_points_evidence
        )
        if simulation_points_evidence["passed"] is not True:
            runtime_layer_evidence["failed_properties"].append(
                "simulationPoints_runtime_readback"
            )
            runtime_layer_evidence["passed"] = False
        runtime_property_specs = dict(runtime_layer_evidence["property_specs"])
        receipt["point_instancer"]["runtime_layer_evidence"] = (
            runtime_layer_evidence
        )
        receipt["point_instancer"]["runtime_property_specs"] = (
            runtime_property_specs
        )
        receipt["timing"].update(
            {
                "final_edit_target_is_session": runtime_layer_evidence[
                    "final_edit_target_is_session"
                ],
                **{
                    f"{receipt_name}_session_layer_only": (
                        runtime_property_specs[
                            f"{property_name}_session_layer_spec"
                        ]
                        and not runtime_property_specs[
                            f"{property_name}_root_layer_spec"
                        ]
                    )
                    for property_name, receipt_name in (
                        ("positions", "positions"),
                        ("velocities", "velocities"),
                        ("orientations", "orientations"),
                        ("scales", "scales"),
                        ("protoIndices", "proto_indices"),
                        ("ids", "ids"),
                    )
                },
                "simulation_points_root_layer_absent": not runtime_property_specs[
                    "simulationPoints_root_layer_spec"
                ],
                "simulation_points_runtime_readback_passed": (
                    simulation_points_evidence["passed"]
                ),
            }
        )
        if runtime_layer_evidence["passed"] is not True:
            raise RuntimeError("runtime particle rows escaped the session edit target")
        if (
            len(shared_particle_ids) != expected_particle_count
            or shared_particle_ids != list(range(expected_particle_count))
            or len(_positions(shared_particle_points.GetPrim(), instancer=True)) != expected_particle_count
            or len(active_row_metadata)
            != sum(int(pour["emission_chunk_count"]) for pour in plan["pours"])
        ):
            raise RuntimeError("shared runtime particle identity/count contract failed")
        callback_indices = [int(row["callback_index"]) for row in callback_records]
        callback_dts = [float(row["dt_s"]) for row in callback_records]
        callback_times = [float(row["timeline_time_s"]) for row in callback_records]
        callback_direct_step_indices = (
            []
            if timeline_owned_count_driven
            else [
                int(row["direct_step_index"])
                for row in callback_records
            ]
        )
        callback_app_update_indices = (
            [
                int(row["app_update_index"])
                for row in callback_records
            ]
            if timeline_owned_count_driven
            else []
        )
        callback_logical_times = [
            callback_index / float(physics_clock["physics_steps_per_second"])
            for callback_index in callback_indices
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
        capture_direct_physics_steps_advanced = (
            0
            if timeline_owned_count_driven
            else sum(
                int(row["physics_steps_after"])
                - int(row["physics_steps_before"])
                for row in capture_guard_records
            )
        )
        capture_simulation_time_advanced = (
            False
            if timeline_owned_count_driven
            else any(
                not math.isclose(
                    float(row["simulation_time_before_s"]),
                    float(row["simulation_time_after_s"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for row in capture_guard_records
            )
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
                "callback_direct_step_indices": callback_direct_step_indices,
                "callback_app_update_indices": callback_app_update_indices,
                "callback_logical_times_s": callback_logical_times,
                "callback_dt_sum_s": float(sum(callback_dts)),
                "one_callback_per_direct_step": (
                    not timeline_owned_count_driven
                    and len(callback_records)
                    == qualification_target_step
                    and callback_direct_step_indices
                    == list(range(1, qualification_target_step + 1))
                ),
                "one_callback_per_app_update": (
                    timeline_owned_count_driven
                    and physics_substeps == 1
                    and len(callback_records) == qualification_target_step
                    and callback_app_update_indices
                    == list(range(1, qualification_target_step + 1))
                ),
                "physics_loop_app_update_count": (
                    timeline_app_update_index
                    if timeline_owned_count_driven
                    else 0
                ),
                "emissions_authored_on_main_thread": all(
                    row["authored_on_main_thread"] is True
                    for row in emission_trace
                ),
                "emission_boundary_verified": all(
                    row["physics_step"] == row["report_step"]
                    and row["authored_after_callback_count"]
                    == (row["report_step"] - 1) * physics_substeps
                    and row["scheduled_before_callback_index"]
                    == (row["report_step"] - 1) * physics_substeps + 1
                    and row["scheduled_before_callback_indices"]
                    == list(
                        range(
                            (row["report_step"] - 1) * physics_substeps + 1,
                            row["report_step"] * physics_substeps + 1,
                        )
                    )
                    for row in emission_trace
                ),
                "emission_boundaries": [
                    {
                        key: row[key]
                        for key in (
                            "physics_step",
                            "report_step",
                            "authored_after_callback_count",
                            "scheduled_before_callback_index",
                            "scheduled_before_callback_indices",
                            "authored_on_main_thread",
                        )
                    }
                    for row in emission_trace
                ],
                "positions_session_layer_only": runtime_property_specs[
                    "positions_session_layer_spec"
                ] and not runtime_property_specs["positions_root_layer_spec"],
                "simulation_points_root_layer_absent": not runtime_property_specs[
                    "simulationPoints_root_layer_spec"
                ],
                "simulation_points_runtime_readback_passed": (
                    simulation_points_evidence["passed"]
                ),
                "velocities_session_layer_only": runtime_property_specs[
                    "velocities_session_layer_spec"
                ] and not runtime_property_specs["velocities_root_layer_spec"],
                "orientations_session_layer_only": runtime_property_specs[
                    "orientations_session_layer_spec"
                ] and not runtime_property_specs["orientations_root_layer_spec"],
                "scales_session_layer_only": runtime_property_specs[
                    "scales_session_layer_spec"
                ] and not runtime_property_specs["scales_root_layer_spec"],
                "proto_indices_session_layer_only": runtime_property_specs[
                    "protoIndices_session_layer_spec"
                ] and not runtime_property_specs["protoIndices_root_layer_spec"],
                "ids_session_layer_only": runtime_property_specs[
                    "ids_session_layer_spec"
                ] and not runtime_property_specs["ids_root_layer_spec"],
                "capture_callbacks_advanced": capture_callbacks_advanced,
                "capture_direct_physics_steps_advanced": (
                    capture_direct_physics_steps_advanced
                ),
                "capture_simulation_time_advanced": (
                    capture_simulation_time_advanced
                ),
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
                "runtime_particle_ids_exact_each_report_interval": True,
                "runtime_particle_identity_check_count": particle_identity_check_count,
                "runtime_particle_id_mismatch_count": particle_identity_mismatch_count,
                "runtime_particle_ids_final_sha256": _particle_ids_sha256(
                    final_particle_ids
                ),
                "runtime_emission_event_count": len(active_row_metadata),
            }
        )
        final_movement = _movement(
            shared_initial_positions, _positions(shared_particle_points.GetPrim(), instancer=True)
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
        final_positions = [
            [float(value[0]), float(value[1]), float(value[2])]
            for value in (shared_particle_points.GetSimulationPointsAttr().Get() or [])
        ]
        receipt["totals"].update({"emitted_particle_count": cumulative, "final_particle_count": len(final_positions)})
        count_driven_render = count_driven_body_runtime is not None
        receipt["render_limitations"] = dict(
            COUNT_DRIVEN_RENDER_LIMITATIONS
            if count_driven_render
            else RENDER_LIMITATIONS
        )
        liquid_representations = [
            str(row["liquid_representation"])
            for row in semantic_mask_records
        ]
        expected_liquid_representation = (
            COUNT_DRIVEN_RENDER_PATH
            if count_driven_render
            else "point_instancer"
        )
        rendered_liquid_indices = [
            index
            for index, value in enumerate(liquid_representations)
            if value == expected_liquid_representation
        ]
        if not rendered_liquid_indices:
            raise RuntimeError(
                f"no {expected_liquid_representation} capture was observed"
            )
        first_rendered_liquid_index = rendered_liquid_indices[0]
        representation_monotonic = (
            liquid_representations[:first_rendered_liquid_index]
            == [expected_liquid_representation] * first_rendered_liquid_index
            and liquid_representations[first_rendered_liquid_index:]
            == [expected_liquid_representation]
            * (len(liquid_representations) - first_rendered_liquid_index)
        )
        if not representation_monotonic:
            raise RuntimeError("liquid representation changed during the run")
        if any(index < first_rendered_liquid_index for index in stage_frame_indices):
            raise RuntimeError("a stage frame used stream-only representation")
        receipt["containment"]["leakage_detected"] = any(
            record.get("leakage_detected") is True
            for record in receipt["containment"]["capture_records"]
        )
        writer.release()
        writer = None
        decodable, decoded_count = _decode_video(cv2, video_path)
        if video_repeat_plan is None:
            raise RuntimeError("video repeat plan was not built")
        if (
            video_frame_source_capture_indices
            != video_repeat_plan["source_capture_indices"]
        ):
            raise RuntimeError("encoded video source-index timeline drifted")
        if decoded_count != video_repeat_plan["video_frame_count"]:
            raise RuntimeError(
                "decoded video frame count differs from cumulative repeat plan"
            )
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
                "video_playback_fps": video_repeat_plan["playback_fps"],
                "video_timing_algorithm": video_repeat_plan["algorithm"],
                "video_encoded_frame_time_rule": video_repeat_plan[
                    "encoded_frame_time_rule"
                ],
                "video_timing_source_frame_steps": video_repeat_plan[
                    "source_frame_steps"
                ],
                "video_source_capture_onset_frame_indices": video_repeat_plan[
                    "source_capture_onset_frame_indices"
                ],
                "video_source_capture_onset_times_s": video_repeat_plan[
                    "source_capture_onset_times_s"
                ],
                "video_source_capture_onset_errors_s": video_repeat_plan[
                    "source_capture_onset_errors_s"
                ],
                "video_maximum_absolute_onset_error_s": video_repeat_plan[
                    "maximum_absolute_onset_error_s"
                ],
                "video_frame_repeat_counts": video_repeat_plan[
                    "repeat_counts"
                ],
                "video_frame_source_capture_indices": (
                    video_frame_source_capture_indices
                ),
                "video_frame_rgb_sha256": video_frame_rgb_hashes,
                "video_distinct_rgb_sha256_count": len(
                    set(video_frame_rgb_hashes)
                ),
                "video_source_duration_s": video_repeat_plan[
                    "source_duration_s"
                ],
                "video_playback_duration_s": video_repeat_plan[
                    "playback_duration_s"
                ],
                "video_duration_error_s": video_repeat_plan[
                    "duration_error_s"
                ],
                "stage_frames_are_video_subsets": (
                    False
                    if args.realtime_video
                    else bool(
                        len(stage_hashes) == 3
                        and set(stage_hashes).issubset(set(frame_hashes))
                    )
                ),
                "stream_only_frame_count": first_rendered_liquid_index,
                **(
                    {
                        "count_driven_body_frame_count": len(
                            rendered_liquid_indices
                        ),
                        "count_driven_body_transition_step": frame_steps[
                            first_rendered_liquid_index
                        ],
                    }
                    if count_driven_render
                    else {
                        "vial_proxy_frame_count": len(rendered_liquid_indices),
                        "vial_proxy_transition_step": frame_steps[
                            first_rendered_liquid_index
                        ],
                    }
                ),
                "liquid_representation_monotonic": representation_monotonic,
            }
        )
        expected_frame_steps = (
            sorted(
                int(pour["stage_capture_step"])
                for pour in plan["pours"]
            )
            if args.realtime_video
            else list(
                _build_capture_steps(
                    plan,
                    stride=args.capture_stride,
                    runtime_schedule=runtime_capture_schedule,
                )
            )
        )
        if frame_steps != expected_frame_steps:
            raise RuntimeError(
                "kept RGB timeline differs from the frozen safe sequence"
            )
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
                    receipt["semantics"].get(
                        "count_driven_body_semantic_bound"
                        if count_driven_render
                        else "point_instancer_semantic_bound"
                    )
                    is True
                    and receipt["semantics"]["stream_semantic_bound"]
                    and receipt["semantics"]["capture_volumes_enabled_readback"]
                    is True
                    and len(semantic_pairs) == len(frame_paths)
                    and all(
                        0 < pair["liquid_pixel_count"] < width * height
                        for pair in semantic_pairs
                    )
                    and all(
                        pair.get(
                            "paired_same_physics_state_rgb_semantic_capture"
                        )
                        is True
                        and pair.get("paired_capture_state_passed") is True
                        for pair in semantic_pairs
                    )
                    and all(
                        pair.get("stream_semantic_overlap", {}).get("passed")
                        is True
                        and (
                            pair.get("stream_material_state", {}).get("matched")
                            is True
                            if pair.get("visible_stream_tanks")
                            else pair.get("stream_material_state", {}).get(
                                "matched"
                            )
                            is None
                        )
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
        if count_driven_render:
            render_state_duplicate_evidence = (
                build_render_state_duplicate_evidence(receipt)
            )
            receipt["render"]["render_state_duplicate_evidence"] = (
                render_state_duplicate_evidence
            )
            if render_state_duplicate_evidence["passed"] is not True:
                raise RuntimeError(
                    "kept RGB sequence contains byte-identical frames across "
                    "non-equivalent render states"
                )
        receipt["scene_contract"]["source_layers_unchanged"] = all(
            Path(row["resolved_path"]).is_file()
            and _sha256(Path(row["resolved_path"])) == row["sha256"]
            for row in layer_records
        )
        if timeline_owned_count_driven:
            final_updates_enabled = _set_render_product_updates(
                render_product,
                False,
                gating,
                phase="final_before_acceptance",
            )
            gating["final_disabled_readback"] = (
                final_updates_enabled is False
            )
        else:
            final_updates_enabled = (
                render_product.hydra_texture.get_updates_enabled()
            )
            gating["final_enabled_readback"] = (
                final_updates_enabled is True
            )
            gating["current_updates_enabled"] = final_updates_enabled
            if final_updates_enabled is not True:
                raise RuntimeError(
                    "persistent render product was disabled before acceptance"
                )
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
        source_geometry = receipt["source_emission_geometry"]
        source_geometry["observed_event_count"] = len(
            source_geometry["events"]
        )
        if source_geometry["profile_id"] == C21_RESOLUTION_PROFILE_ID:
            source_geometry["all_support_gates_passed"] = all(
                row.get("support_gate_passed") is True
                for row in source_geometry["events"]
            )
            source_geometry["all_centroid_gates_passed"] = all(
                row.get("centroid_gate_passed") is True
                for row in source_geometry["events"]
            )
            source_geometry["all_flux_gates_passed"] = all(
                row.get("flux_gate_passed") is True
                for row in source_geometry["events"]
            )
            source_geometry["all_separation_gates_passed"] = all(
                row.get("separation_gate_passed") is True
                for row in source_geometry["events"]
            )
            source_geometry["gate_passed"] = bool(
                source_geometry["observed_event_count"]
                == source_geometry["expected_event_count"]
                and source_geometry["all_support_gates_passed"]
                and source_geometry["all_centroid_gates_passed"]
                and source_geometry["all_separation_gates_passed"]
                and source_geometry["all_flux_gates_passed"]
            )
            if source_geometry["gate_passed"] is not True:
                raise RuntimeError("C21 source-emission geometry gate failed")
        else:
            source_geometry["gate_passed"] = True
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["failure"] = _failure_evidence(exc, phase=failure_phase)
        receipt["errors"].append(f"F2 execution failed: {type(exc).__name__}: {exc}")
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                receipt["errors"].append(
                    "partial video writer release failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                writer = None
        if receipt.get("status") != "pass":
            try:
                _persist_completed_capture_artifacts(
                    receipt,
                    output=output,
                    width=width,
                    height=height,
                    frame_paths=frame_paths,
                    frame_steps=frame_steps,
                    semantic_paths=semantic_paths,
                    liquid_mask_paths=liquid_mask_paths,
                    liquid_pixel_counts=liquid_pixel_counts,
                    semantic_mask_records=semantic_mask_records,
                    stage_paths=stage_paths,
                    stage_frame_indices=stage_frame_indices,
                    video_path=video_path,
                    video_codec=video_codec,
                    video_writer_fallback_used=video_fallback,
                )
            except Exception as exc:
                receipt["errors"].append(
                    "partial artifact metadata finalization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        final_gate_key = (
            "final_disabled_readback"
            if timeline_owned_count_driven
            else "final_enabled_readback"
        )
        if render_product is not None and gating.get(final_gate_key) is not True:
            try:
                cleanup_updates_enabled = (
                    render_product.hydra_texture.get_updates_enabled()
                )
                gating[final_gate_key] = (
                    cleanup_updates_enabled is False
                    if timeline_owned_count_driven
                    else cleanup_updates_enabled is True
                )
                gating["current_updates_enabled"] = cleanup_updates_enabled
            except Exception as exc:
                receipt["warnings"].append(
                    "render-product update readback during cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )
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
        if instance_annotator is not None and render_product is not None:
            try:
                instance_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"instance annotator detach failed: {exc}")
        if timeline is not None:
            try:
                timeline.pause()
                timeline.commit()
            except Exception as exc:
                receipt["warnings"].append(f"timeline pause failed: {exc}")
        physics_step_subscription = None
        if fabric_initial is not None:
            try:
                SimulationManager.enable_fabric(fabric_initial)
                receipt["gpu_simulation"]["fabric_restored_readback"] = bool(
                    SimulationManager.is_fabric_enabled()
                )
            except Exception as exc:
                receipt["warnings"].append(
                    f"Fabric setting restore failed: {exc}"
                )
        if (
            settings is not None
            and particles_writeback_setting is not None
            and particles_writeback_initial is not None
        ):
            try:
                settings.set_bool(
                    particles_writeback_setting, particles_writeback_initial
                )
                receipt["gpu_simulation"][
                    "position_writeback_restored_readback"
                ] = bool(settings.get_as_bool(particles_writeback_setting))
            except Exception as exc:
                receipt["warnings"].append(
                    f"position writeback setting restore failed: {exc}"
                )
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
        if neuraylib is not None and registered_mdl_search_path:
            try:
                receipt["material"]["mdl_search_path_removed"] = bool(
                    neuraylib.removeMdlSearchPath(registered_mdl_search_path)
                )
            except Exception as exc:
                receipt["material"]["mdl_search_path_removed"] = False
                receipt["warnings"].append(
                    f"MDL search path removal failed: {exc}"
                )
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
    receipt = _finalize_paint_ball_receipt(receipt)
    if receipt["status"] == "pass":
        failures = (
            _validate_realtime_height_banded_receipt(receipt)
            if args.realtime_video
            else validate_paint_ball_receipt(receipt)
        )
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    return receipt


def _validate_realtime_height_banded_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate only the decision-changing gates for the revised cohort."""

    failures: list[str] = []
    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    plan = receipt.get("plan")
    plan = plan if isinstance(plan, Mapping) else {}
    pours = plan.get("pours")
    pours = pours if isinstance(pours, list) else []
    video_hashes = render.get("video_frame_rgb_sha256")
    repeat_counts = render.get("video_frame_repeat_counts")
    source_indices = render.get("video_frame_source_capture_indices")
    decoded_count = render.get("video_decoded_frame_count")
    if (
        receipt.get("status") != "pass"
        or render.get("video_renderer") != "RaytracedLighting"
        or render.get("stage_still_renderer") != "PathTracing"
        or render.get("stage_frames_are_video_subsets") is not False
        or render.get("video_decodable") is not True
        or not isinstance(decoded_count, int)
        or decoded_count <= 0
        or not isinstance(video_hashes, list)
        or len(video_hashes) != decoded_count
        or any(not _valid_lower_sha256(value) for value in video_hashes)
        or not isinstance(repeat_counts, list)
        or repeat_counts != [1] * decoded_count
        or source_indices != list(range(decoded_count))
    ):
        failures.append("realtime_video_contract")
    if (
        render.get("frame_count") != len(pours)
        or not isinstance(render.get("stage_frame_sha256"), list)
        or len(render["stage_frame_sha256"]) != len(pours)
        or any(
            not _valid_lower_sha256(value)
            for value in render.get("stage_frame_sha256", [])
        )
    ):
        failures.append("pathtraced_stage_still_contract")
    tone = render.get("tone_exposure_state")
    transitions = render.get("mode_transitions")
    if (
        not isinstance(tone, Mapping)
        or tone.get("passed") is not True
        or not isinstance(transitions, list)
        or not transitions
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in transitions
        )
        or {row.get("requested") for row in transitions} != {"rt", "pt"}
    ):
        failures.append("renderer_tone_exposure_readback")
    body = receipt.get("count_driven_liquid_body")
    body = body if isinstance(body, Mapping) else {}
    band_rows = body.get("stage_band_evidence")
    if (
        body.get("spatial_mixing_color_rendered") is not True
        or body.get("spatial_layering_color_rendered") is not True
        or body.get("height_band_count") != HEIGHT_BAND_COUNT
        or not isinstance(band_rows, list)
        or len(band_rows) != len(pours)
        or any(
            row.get("spatial_mixing_color_rendered") is not True
            or row.get("spatial_layering_color_rendered") is not True
            for row in band_rows
            if isinstance(row, Mapping)
        )
    ):
        failures.append("height_banded_mixing_contract")
    material = receipt.get("material")
    material = material if isinstance(material, Mapping) else {}
    tank_colors = material.get("tank_transmission_colors")
    source_resolution_valid = True
    expected_t100 = [1.0, 1.0, 1.0]
    if "tank_colour_resolution" in plan:
        try:
            source_resolution = validate_tank_colour_resolution(
                plan["tank_colour_resolution"]
            )
            source_resolution_valid = plan.get("run_palette") == (
                run_palette_from_tank_colour_resolution(source_resolution)
            )
            expected_t100 = source_resolution["transmission_1cm"]["t100"]
        except (TypeError, ValueError):
            source_resolution_valid = False
    if (
        not isinstance(tank_colors, Mapping)
        or not source_resolution_valid
        or tank_colors.get("t100") != expected_t100
        or material.get("stream_uses_dedicated_path_depth") is not True
    ):
        failures.append("run_palette_and_stream_material_contract")
    semantics = receipt.get("semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    containment = receipt.get("containment")
    containment = containment if isinstance(containment, Mapping) else {}
    if semantics.get("semantic_gate_passed") is not True:
        failures.append("stage_semantic_gate")
    if containment.get("leakage_detected") is not False:
        failures.append("particle_containment_gate")
    return tuple(failures)


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
        plan = _runtime_paint_ball_plan(args)
        _write_json(args.out / "plan.json", plan)
        if "tank_colour_resolution" in plan:
            _write_json(
                args.out / "label-provenance.json",
                {
                    "schema_version": 1,
                    "artifact_type": "tank_colour_label_provenance",
                    "tank_colour_resolution": deepcopy(
                        plan["tank_colour_resolution"]
                    ),
                    "excluded_from_model_inputs": True,
                },
            )
        return 0
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    owned_app = None
    exit_code = 2
    try:
        app, owned_app = _start_or_reuse_kit()
        if args.q2_no_render:
            receipt = run_q2_no_render(args, app, args.out)
        else:
            receipt = run_sequence(args, app, args.out)
    except Exception as exc:
        receipt = (
            _empty_q2_no_render_receipt(args)
            if args.q2_no_render
            else _empty_paint_ball_receipt(args)
        )
        receipt["errors"].append(
            f"Isaac application startup failed: {type(exc).__name__}: {exc}"
        )
        if args.q2_no_render:
            failure_path = args.out / "q2-startup-failure.json"
            _write_json(
                failure_path,
                {
                    "artifact_type": "q2_failure_artifact",
                    "phase": "startup",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "arrays_saved_before_assertions": False,
                },
            )
            receipt["failure_artifacts"].append(
                {
                    "kind": "startup_failure",
                    "path": str(failure_path.relative_to(args.out)),
                    "sha256": _sha256(failure_path),
                }
            )
        if not args.q2_no_render:
            receipt = _finalize_paint_ball_receipt(receipt)
    try:
        receipt_path = args.out / "receipt.json"
        exit_code = 0 if receipt["status"] == "pass" else 2
        receipt["exit_status"] = exit_code
        _write_json(receipt_path, receipt)
        if "tank_colour_resolution" in receipt:
            _write_json(
                args.out / "label-provenance.json",
                {
                    "schema_version": 1,
                    "artifact_type": "tank_colour_label_provenance",
                    "tank_colour_resolution": deepcopy(
                        receipt["tank_colour_resolution"]
                    ),
                    "excluded_from_model_inputs": True,
                },
            )
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    finally:
        if owned_app is not None:
            owned_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
