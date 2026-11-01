#!/usr/bin/env python3
"""Test whether Isaac 6.0.1 or 6.2.0 renders mutable per-particle prototype colours.

This is a bounded renderer-compatibility smoke, not a liquid-mixing or physical-
colour result.  A top-level visual review of the saved images is required.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from twin.physx.three_pour import (
    CAMERA_PATH,
    SOURCE_LIQUID_PATH,
    SOURCE_STREAM_PATH,
    VIAL_CENTER_XY_M,
    build_twin_collider_specs,
)
from twin.render.render_capture import resolve_class_ids


NAMESPACE = "/FillStation/FluidmixPaletteSmoke"
PARTICLE_COUNT = 12
INITIAL_PROTO_INDICES = [0, 1, 2] * 4
TANK_COLORS = {
    "t200": (0.95, 0.38, 0.08),
    "t400": (0.18, 0.55, 0.75),
    "t100": (0.92, 0.98, 1.0),
}
ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
EXTRA_ARGS = (f"--{ASYNC_SETTING}=false",)
DEFAULT_EXPECTED_ISAAC_SIM_VERSION = "6.0.1"
SUPPORTED_ISAAC_SIM_VERSION_FAMILIES = ("6.0.1", "6.2.0")


def _is_supported_isaac_sim_version_family(
    observed: object, expected_family: object
) -> bool:
    """Accept only the two reviewed three-component Isaac package families."""
    if (
        not isinstance(observed, str)
        or not isinstance(expected_family, str)
        or expected_family not in SUPPORTED_ISAAC_SIM_VERSION_FAMILIES
    ):
        return False
    if observed == expected_family:
        return True
    suffix = observed.removeprefix(expected_family + ".")
    return bool(suffix and suffix.isdigit() and observed.startswith(expected_family + "."))


def build_smoke_plan(
    expected_isaac_sim_version: str = DEFAULT_EXPECTED_ISAAC_SIM_VERSION,
) -> dict[str, Any]:
    if expected_isaac_sim_version not in SUPPORTED_ISAAC_SIM_VERSION_FAMILIES:
        raise ValueError(
            "expected_isaac_sim_version must be a supported Isaac Sim version family"
        )
    return {
        "schema_version": 1,
        "experiment": "physx_particle_palette_smoke",
        "seed": 20261025,
        "particle_count": PARTICLE_COUNT,
        "particle_system_count": 1,
        "prototype_tanks": list(TANK_COLORS),
        "prototype_transmission_colors": {
            tank: list(color) for tank, color in TANK_COLORS.items()
        },
        "initial_proto_indices": list(INITIAL_PROTO_INDICES),
        "swapped_proto_indices": swap_orange_blue(INITIAL_PROTO_INDICES),
        "isaac_sim_version": expected_isaac_sim_version,
        "expected_isaac_sim_version": expected_isaac_sim_version,
        "frozen_time_code": 1.0,
        "physics_steps_before_swap": 30,
        "physics_steps_during_swap": 0,
        "particle_ccd_enabled": True,
        "physics_hz": 60,
        "renderer": "PathTracing",
        "resolution": [640, 480],
        "warmup_updates": 60,
        "rt_subframes": 64,
        "max_bounces": 16,
        "max_specular_and_transmission_bounces": 16,
        "async_throttling_enabled": False,
        "camera_path": CAMERA_PATH,
        "material_controls": {
            "specular_transmission_scattering_depth_m": 0.01,
            "thin_walled": False,
        },
        "runtime_edit_target": "session_layer",
        "semantic_gate": {
            "label": "liquid",
            "type": "class",
            "required_on_both_captures": True,
            "nonempty_aligned_mask_required": True,
        },
        "visual_decision": "pending_top_level_review",
        "claim_limit": (
            "Renderer/API feasibility only; not smooth fluid, measured tank colour, "
            "mixing validation, physical accuracy, or sim-to-real evidence."
        ),
    }


def swap_orange_blue(indices: list[int]) -> list[int]:
    if any(type(value) is not int or value not in (0, 1, 2) for value in indices):
        raise ValueError("prototype index outside frozen palette")
    return [1 if value == 0 else 0 if value == 1 else 2 for value in indices]


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _semantic_pair_is_valid(pair: object) -> bool:
    if not isinstance(pair, Mapping):
        return False
    liquid_ids = pair.get("liquid_ids")
    id_to_labels = pair.get("idToLabels")
    pixel_count = pair.get("liquid_pixel_count")
    try:
        resolved_liquid_ids = list(resolve_class_ids(id_to_labels).liquid_ids)
    except Exception:
        return False
    return bool(
        isinstance(pair.get("mask_path"), str)
        and pair.get("mask_readable") is True
        and pair.get("mask_aligned_to_rgb") is True
        and _valid_hash(pair.get("mask_sha256"))
        and _valid_hash(pair.get("rgb_sha256"))
        and isinstance(liquid_ids, list)
        and liquid_ids
        and all(type(value) is int for value in liquid_ids)
        and sorted(liquid_ids) == resolved_liquid_ids
        and isinstance(pixel_count, int)
        and not isinstance(pixel_count, bool)
        and pixel_count > 0
        and isinstance(id_to_labels, Mapping)
    )


def validate_technical_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    failures: list[str] = []
    if receipt.get("schema_version") != 1 or receipt.get("experiment") != "physx_particle_palette_smoke":
        failures.append("receipt_identity")
    if receipt.get("status") != "pending_visual":
        failures.append("status_must_remain_pending_visual")
    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    plan = receipt.get("plan")
    plan = plan if isinstance(plan, Mapping) else {}
    expected_isaac_version = arguments.get("expected_isaac_sim_version")
    if (
        expected_isaac_version not in SUPPORTED_ISAAC_SIM_VERSION_FAMILIES
        or receipt.get("expected_isaac_sim_version") != expected_isaac_version
        or plan.get("expected_isaac_sim_version") != expected_isaac_version
        or plan.get("isaac_sim_version") != expected_isaac_version
    ):
        failures.append("expected_isaac_sim_version")
    scene = receipt.get("scene_contract")
    scene = scene if isinstance(scene, Mapping) else {}
    frozen_time_code = scene.get("frozen_time_code")
    if (
        scene.get("default_prim") != "/FillStation"
        or scene.get("meters_per_unit") != 1.0
        or scene.get("up_axis") != "Z"
        or isinstance(frozen_time_code, bool)
        or not isinstance(frozen_time_code, (int, float))
        or not math.isclose(
            float(frozen_time_code), 1.0, rel_tol=0.0, abs_tol=1e-9
        )
        or scene.get("session_layer_only") is not True
        or scene.get("hidden_source_paths") != [SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH]
        or scene.get("custom_camera_count") != 0
        or scene.get("custom_light_count") != 0
        or scene.get("custom_backdrop_count") != 0
        or scene.get("forbidden_authored_paths") != []
    ):
        failures.append("twin_scene_contract")
    technical = receipt.get("technical")
    technical = technical if isinstance(technical, Mapping) else {}
    isaac_version = technical.get("isaac_sim_version")
    if not _is_supported_isaac_sim_version_family(
        isaac_version, expected_isaac_version
    ):
        failures.append("isaac_version")
    if technical.get("cuda_device_available") is not True or technical.get("gpu_dynamics") is not True:
        failures.append("gpu_dynamics")
    if technical.get("particle_ccd_enabled") is not True:
        failures.append("particle_ccd")
    if technical.get("particle_system_count") != 1:
        failures.append("particle_system_count")
    if technical.get("particle_count") != PARTICLE_COUNT or technical.get("prototype_count") != 3:
        failures.append("particle_or_prototype_count")
    if (
        technical.get("fluid_particle_set") is not True
        or technical.get("creation_api")
        != "omni.physx.scripts.particleUtils.add_physx_particleset_pointinstancer"
        or technical.get("particle_system_target_count") != 1
        or technical.get("sphere_prototype_count") != 3
    ):
        failures.append("pointinstancer_contract")
    if technical.get("prototype_material_bindings") != 3:
        failures.append("prototype_material_binding")
    expected_materials = [
        {
            "tank_id": tank,
            "specular_transmission_color": list(color),
            "specular_transmission_scattering_depth_m": 0.01,
            "thin_walled": False,
            "bound": True,
        }
        for tank, color in TANK_COLORS.items()
    ]
    if technical.get("material_variants") != expected_materials:
        failures.append("material_variant_contract")
    if (
        technical.get("collider_count") != len(build_twin_collider_specs())
        or technical.get("all_colliders_invisible") is not True
    ):
        failures.append("collider_contract")
    maximum_movement = technical.get("maximum_movement_m")
    if (
        technical.get("physics_steps_before_swap") != 30
        or technical.get("physics_steps_during_swap") != 0
        or technical.get("finite_positions") is not True
        or technical.get("moved_under_physics") is not True
        or isinstance(maximum_movement, bool)
        or not isinstance(maximum_movement, (int, float))
        or not math.isfinite(float(maximum_movement))
        or float(maximum_movement) <= 1e-6
    ):
        failures.append("finite_motion")
    if technical.get("indices_before") != INITIAL_PROTO_INDICES or technical.get("indices_after") != swap_orange_blue(INITIAL_PROTO_INDICES):
        failures.append("prototype_swap")
    if technical.get("positions_identical_across_swap") is not True:
        failures.append("swap_changed_positions")
    before_position_hash = technical.get("positions_before_sha256")
    after_position_hash = technical.get("positions_after_sha256")
    if (
        technical.get("position_bytes_identical_across_swap") is not True
        or not _valid_hash(before_position_hash)
        or before_position_hash != after_position_hash
    ):
        failures.append("swap_changed_position_bytes")
    if (
        technical.get("positions_session_layer_spec") is not True
        or technical.get("positions_root_layer_spec") is not False
        or technical.get("proto_indices_session_layer_spec") is not True
        or technical.get("proto_indices_root_layer_spec") is not False
    ):
        failures.append("runtime_property_layer_contract")
    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    if (
        render.get("renderer") != "PathTracing"
        or render.get("camera_path") != CAMERA_PATH
        or render.get("resolution") != [640, 480]
        or render.get("async_throttling_enabled") is not False
        or render.get("warmup_updates") != 60
        or render.get("rt_subframes") != 64
        or render.get("max_bounces") != 16
        or render.get("max_specular_and_transmission_bounces") != 16
    ):
        failures.append("render_configuration")
    if (
        render.get("before_readable") is not True
        or render.get("after_readable") is not True
        or render.get("before_nonconstant") is not True
        or render.get("after_nonconstant") is not True
        or any(
            not _valid_hash(render.get(key))
            for key in (
                "before_sha256",
                "after_sha256",
                "before_crop_sha256",
                "after_crop_sha256",
                "contact_pair_sha256",
            )
        )
    ):
        failures.append("render_missing_or_constant")
    semantics = receipt.get("semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    if (
        semantics.get("semantic_label") != "liquid"
        or semantics.get("semantic_type") != "class"
        or semantics.get("annotator") != "semantic_segmentation"
        or semantics.get("resolution") != [640, 480]
        or semantics.get("labels_authored_before_stage_attach") is not True
        or semantics.get("annotator_attached_after_labels_and_stage_attach") is not True
        or not _semantic_pair_is_valid(semantics.get("before"))
        or not _semantic_pair_is_valid(semantics.get("after"))
    ):
        failures.append("semantic_mask_contract")
    visual = receipt.get("visual_review")
    visual = visual if isinstance(visual, Mapping) else {}
    if visual.get("status") != "pending_top_level_review":
        failures.append("visual_review_not_pending")
    errors = receipt.get("errors")
    if not isinstance(errors, list) or errors:
        failures.append("run_error_recorded")
    return tuple(dict.fromkeys(failures))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--expected-isaac-sim-version",
        "--expected-isaac-version",
        dest="expected_isaac_sim_version",
        choices=SUPPORTED_ISAAC_SIM_VERSION_FAMILIES,
        default=DEFAULT_EXPECTED_ISAAC_SIM_VERSION,
        help="reviewed Isaac Sim version family expected on this host",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.stage.is_absolute():
        raise ValueError("stage must be an absolute path")
    if not args.stage.is_file():
        raise ValueError("stage must be an existing file")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("seed must be an integer")
    if args.expected_isaac_sim_version not in SUPPORTED_ISAAC_SIM_VERSION_FAMILIES:
        raise ValueError(
            "expected_isaac_sim_version must be a supported Isaac Sim version family"
        )


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


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    expected_isaac_sim_version = args.expected_isaac_sim_version
    return {
        "schema_version": 1,
        "experiment": "physx_particle_palette_smoke",
        "status": "insufficient_evidence",
        "expected_isaac_sim_version": expected_isaac_sim_version,
        "plan": build_smoke_plan(expected_isaac_sim_version),
        "arguments": {
            "stage": str(args.stage),
            "seed": args.seed,
            "expected_isaac_sim_version": expected_isaac_sim_version,
        },
        "provenance": {
            "argv": list(sys.argv),
            "stage_sha256": _sha256(args.stage),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "scene_contract": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "frozen_time_code": None,
            "session_layer_only": False,
            "hidden_source_paths": [],
            "custom_camera_count": None,
            "custom_light_count": None,
            "custom_backdrop_count": None,
            "forbidden_authored_paths": None,
        },
        "technical": {
            "isaac_sim_version": None,
            "cuda_device_available": False,
            "gpu_dynamics": False,
            "particle_ccd_enabled": False,
            "particle_system_count": 0,
            "particle_count": 0,
            "fluid_particle_set": False,
            "creation_api": None,
            "particle_system_target_count": 0,
            "prototype_count": 0,
            "sphere_prototype_count": 0,
            "prototype_material_bindings": 0,
            "material_variants": [],
            "collider_count": 0,
            "all_colliders_invisible": False,
            "physics_steps_before_swap": 0,
            "physics_steps_during_swap": None,
            "finite_positions": False,
            "moved_under_physics": False,
            "maximum_movement_m": None,
            "indices_before": [],
            "indices_after": [],
            "positions_identical_across_swap": False,
            "position_bytes_identical_across_swap": False,
            "positions_before_sha256": None,
            "positions_after_sha256": None,
            "positions_before_swap": [],
            "positions_after_swap": [],
            "positions_session_layer_spec": False,
            "positions_root_layer_spec": None,
            "proto_indices_session_layer_spec": False,
            "proto_indices_root_layer_spec": None,
        },
        "render": {
            "renderer": "PathTracing",
            "camera_path": CAMERA_PATH,
            "resolution": [640, 480],
            "async_throttling_enabled": None,
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_bounces": 16,
            "max_specular_and_transmission_bounces": 16,
            "before_path": None,
            "before_sha256": None,
            "after_path": None,
            "after_sha256": None,
            "before_crop_path": None,
            "before_crop_sha256": None,
            "after_crop_path": None,
            "after_crop_sha256": None,
            "contact_pair_path": None,
            "contact_pair_sha256": None,
            "before_readable": False,
            "after_readable": False,
            "before_nonconstant": False,
            "after_nonconstant": False,
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "annotator": "semantic_segmentation",
            "resolution": [640, 480],
            "labels_authored_before_stage_attach": False,
            "annotator_attached_after_labels_and_stage_attach": False,
            "before": {
                "mask_path": None,
                "mask_sha256": None,
                "rgb_sha256": None,
                "mask_readable": False,
                "mask_aligned_to_rgb": False,
                "liquid_ids": [],
                "liquid_pixel_count": 0,
                "idToLabels": {},
            },
            "after": {
                "mask_path": None,
                "mask_sha256": None,
                "rgb_sha256": None,
                "mask_readable": False,
                "mask_aligned_to_rgb": False,
                "liquid_ids": [],
                "liquid_pixel_count": 0,
                "idToLabels": {},
            },
            "rejected_diagnostics": [],
        },
        "visual_review": {
            "status": "pending_top_level_review",
            "criteria": [
                "three prototype colours visibly distinct",
                "orange and blue visibly swap on the same particles",
                "particle rendering readable enough to justify the bead fallback",
            ],
        },
        "errors": [],
        "warnings": [],
    }


def _author_omnisurface(stage: Any, path: str, color: tuple[float, float, float], Sdf: Any, Gf: Any, UsdShade: Any) -> Any:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/OmniSurface")
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_transmission_color": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
        "specular_transmission_scattering_depth": (Sdf.ValueTypeNames.Float, 0.01),
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
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    material.CreateVolumeOutput("mdl").ConnectToSource(output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(output)
    return material, shader


def _initial_positions(Gf: Any) -> list[Any]:
    positions = []
    for index in range(PARTICLE_COUNT):
        column = index % 4
        row = (index // 4) % 3
        positions.append(
            Gf.Vec3f(
                VIAL_CENTER_XY_M[0] + (column - 1.5) * 0.004,
                VIAL_CENTER_XY_M[1] + (row - 1.0) * 0.004,
                1.035 + (index // 6) * 0.008,
            )
        )
    return positions


def _finite_positions(values: Any) -> bool:
    return bool(values) and all(
        math.isfinite(float(component)) for row in values for component in row
    )


def _capture_rgb(rep: Any, annotator: Any, np: Any, *, tries: int = 20) -> Any:
    for _ in range(tries):
        rep.orchestrator.step(
            delta_time=0.0,
            pause_timeline=True,
            rt_subframes=64,
            wait_for_render=True,
        )
        candidate = np.asarray(annotator.get_data())
        if candidate.ndim == 3 and candidate.shape[:2] == (480, 640) and candidate.shape[2] >= 3:
            candidate = np.asarray(candidate[..., :3], dtype=np.uint8)
            if np.any(candidate != candidate[0, 0]):
                return candidate
    raise RuntimeError("RGB capture remained missing or spatially constant after 20 tries")


def _liquid_ids(id_to_labels: object) -> list[int]:
    try:
        return list(resolve_class_ids(id_to_labels).liquid_ids)
    except Exception:
        return []


class _SemanticCaptureError(RuntimeError):
    def __init__(self, capture_name: str, diagnostics: Mapping[str, object], message: str):
        super().__init__(message)
        self.capture_name = capture_name
        self.diagnostics = dict(diagnostics)


def _capture_rgb_semantic(
    rep: Any,
    rgb_annotator: Any,
    semantic_annotator: Any,
    np: Any,
    *,
    tries: int = 20,
    capture_name: str = "unknown",
) -> tuple[Any, Any, dict[str, Any], list[int]]:
    last_nonconstant_rgb = None
    last_id_to_labels: dict[object, object] = {}
    last_unique_semantic_ids: list[int] = []
    for _ in range(tries):
        rep.orchestrator.step(
            delta_time=0.0,
            pause_timeline=True,
            rt_subframes=64,
            wait_for_render=True,
        )
        rgb = np.asarray(rgb_annotator.get_data())
        raw_semantic = semantic_annotator.get_data()
        if (
            rgb.ndim != 3
            or rgb.shape[:2] != (480, 640)
            or rgb.shape[2] < 3
            or not isinstance(raw_semantic, Mapping)
        ):
            continue
        rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
        if np.any(rgb != rgb[0, 0]):
            last_nonconstant_rgb = rgb
        info = raw_semantic.get("info")
        if not isinstance(info, Mapping):
            continue
        if isinstance(info.get("idToLabels"), Mapping):
            id_to_labels = dict(info["idToLabels"])
            last_id_to_labels = id_to_labels
        else:
            continue
        semantic = np.asarray(raw_semantic.get("data"))
        if semantic.ndim == 3 and semantic.shape[2] == 1:
            semantic = semantic[..., 0]
        if semantic.shape != (480, 640) or not np.issubdtype(
            semantic.dtype, np.integer
        ):
            continue
        last_unique_semantic_ids = [int(value) for value in np.unique(semantic)]
        liquid_ids = _liquid_ids(id_to_labels)
        if not liquid_ids:
            continue
        mask = np.where(np.isin(semantic, liquid_ids), 255, 0).astype(np.uint8)
        if int(np.count_nonzero(mask)) <= 0:
            continue
        return rgb, mask, id_to_labels, liquid_ids
    raise _SemanticCaptureError(
        capture_name,
        {
            "rgb": last_nonconstant_rgb,
            "idToLabels": last_id_to_labels,
            "unique_semantic_ids": last_unique_semantic_ids,
        },
        "RGB/semantic capture lacked a liquid ID or nonempty aligned liquid mask",
    )


def _record_rejected_semantic_diagnostic(
    receipt: dict[str, Any],
    output: Path,
    capture_name: str,
    diagnostics: Mapping[str, object],
    cv2: Any,
) -> None:
    """Persist bounded debugging evidence while keeping it outside accepted pairs."""
    rgb = diagnostics.get("rgb")
    rgb_path: Path | None = None
    rgb_sha256: str | None = None
    if rgb is not None:
        candidate = output / f"{capture_name}-rejected-rgb.png"
        if cv2.imwrite(os.fspath(candidate), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            rgb_path = candidate
            rgb_sha256 = _sha256(candidate)
        else:
            receipt["warnings"].append(
                f"could not write {capture_name} rejected RGB diagnostic"
            )
    rejected = receipt["semantics"].setdefault("rejected_diagnostics", [])
    rejected.append(
        {
            "capture": capture_name,
            "status": "rejected",
            "rgb_path": rgb_path.name if rgb_path is not None else None,
            "rgb_sha256": rgb_sha256,
            "idToLabels": dict(diagnostics.get("idToLabels", {})),
            "unique_semantic_ids": list(diagnostics.get("unique_semantic_ids", [])),
        }
    )


def _runtime_property_specs(stage: Any, instancer: Any, Sdf: Any) -> dict[str, bool | None]:
    session = stage.GetSessionLayer()
    root = stage.GetRootLayer()
    path = instancer.GetPath()
    positions_path = Sdf.Path(path).AppendProperty("positions")
    proto_indices_path = Sdf.Path(path).AppendProperty("protoIndices")
    return {
        "positions_session_layer_spec": session.GetPropertyAtPath(positions_path)
        is not None,
        "positions_root_layer_spec": root.GetPropertyAtPath(positions_path) is not None,
        "proto_indices_session_layer_spec": session.GetPropertyAtPath(
            proto_indices_path
        )
        is not None,
        "proto_indices_root_layer_spec": root.GetPropertyAtPath(proto_indices_path)
        is not None,
    }


def run_smoke(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    receipt = _empty_receipt(args)
    simulation = None
    attached = False
    annotator = None
    semantic_annotator = None
    render_product = None
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physx.scripts import particleUtils
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import (
            Gf,
            PhysxSchema,
            Sdf,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdSemantics,
            UsdShade,
            UsdUtils,
        )

        settings = carb.settings.get_settings()
        receipt["technical"]["isaac_sim_version"] = version("isaacsim")
        if not _is_supported_isaac_sim_version_family(
            receipt["technical"]["isaac_sim_version"],
            args.expected_isaac_sim_version,
        ):
            raise RuntimeError(
                "palette smoke expected Isaac Sim "
                f"{args.expected_isaac_sim_version}, observed "
                f"{receipt['technical']['isaac_sim_version']}"
            )
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is enabled")
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        receipt["render"]["max_bounces"] = settings.get_as_int(
            "/rtx/pathtracing/maxBounces"
        )
        receipt["render"][
            "max_specular_and_transmission_bounces"
        ] = settings.get_as_int(
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces"
        )
        if (
            receipt["render"]["max_bounces"] != 16
            or receipt["render"]["max_specular_and_transmission_bounces"] != 16
        ):
            raise RuntimeError("path-tracing bounce settings were not retained")
        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError("could not open canonical twin stage")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened twin stage")
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
            raise RuntimeError("opened stage is not the reviewed FillStation twin")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0.0:
            raise RuntimeError("canonical twin has invalid timeCodesPerSecond")
        timeline.set_current_time(1.0 / time_codes_per_second)
        app.update()
        frozen_time_code = float(timeline.get_current_time()) * time_codes_per_second
        receipt["scene_contract"]["frozen_time_code"] = frozen_time_code
        if not math.isclose(frozen_time_code, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("canonical twin did not freeze at time code 1")
        camera = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
            raise RuntimeError("reviewed FillCam is missing")

        foundation = get_physx_foundation_interface()
        receipt["technical"]["cuda_device_available"] = bool(foundation.cuda_device_check())
        if not receipt["technical"]["cuda_device_available"]:
            raise RuntimeError("GPU particles require a CUDA device")
        simulation = omni.physx.get_physx_simulation_interface()
        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particle_update = getattr(physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None)
        if particle_update is None:
            raise RuntimeError("PhysX lacks particle-to-USD updates")
        settings.set(particle_update, True)

        session = stage.GetSessionLayer()
        collider_prims = []
        material_records = []
        prototype_prims = []
        bound_material_prims = []
        with Usd.EditContext(stage, session):
            for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                source_prim = stage.GetPrimAtPath(path)
                if not source_prim.IsValid():
                    raise RuntimeError(f"canonical twin source prim is missing: {path}")
                UsdGeom.Imageable(source_prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
            UsdGeom.Xform.Define(stage, NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            receipt["technical"]["gpu_dynamics"] = bool(
                scene_api.GetEnableGPUDynamicsAttr().Get()
            ) and str(scene_api.GetBroadphaseTypeAttr().Get()) == "GPU"

            system_path = Sdf.Path(f"{NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateEnableCCDAttr().Set(True)
            system.CreateParticleContactOffsetAttr().Set(0.003)
            system.CreateFluidRestOffsetAttr().Set(0.0015)
            system.CreateContactOffsetAttr().Set(0.003)
            system.CreateRestOffsetAttr().Set(0.0029)
            system.CreateSolidRestOffsetAttr().Set(0.0029)
            receipt["technical"]["particle_ccd_enabled"] = bool(
                system.GetEnableCCDAttr().Get()
            )
            pbd_material = UsdShade.Material.Define(stage, f"{NAMESPACE}/PBDMaterial")
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(pbd_material.GetPrim())
            pbd.CreateDensityAttr().Set(1000.0)
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
                pbd_material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

            for spec in build_twin_collider_specs():
                smoke_path = str(spec["path"]).replace(
                    "/FillStation/FluidmixF2", NAMESPACE
                )
                cube = UsdGeom.Cube.Define(stage, smoke_path)
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())

            initial_positions = _initial_positions(Gf)
            initial_velocities = [Gf.Vec3f(0.0, 0.0, -0.05)] * PARTICLE_COUNT
            instancer_path = Sdf.Path(f"{NAMESPACE}/Particles")
            instancer_prim = particleUtils.add_physx_particleset_pointinstancer(
                stage=stage,
                path=instancer_path,
                positions=initial_positions,
                velocities=initial_velocities,
                particle_system_path=system_path,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=0.0,
                density=1000.0,
                num_prototypes=3,
                prototype_indices=list(INITIAL_PROTO_INDICES),
            )
            bindings = 0
            for ordinal, (tank, color) in enumerate(TANK_COLORS.items()):
                prototype_path = f"{instancer_path}/particlePrototype{ordinal}"
                sphere = UsdGeom.Sphere.Get(stage, prototype_path)
                sphere.CreateRadiusAttr().Set(0.0025)
                prototype_prims.append(sphere.GetPrim())
                material, shader = _author_omnisurface(
                    stage,
                    f"{NAMESPACE}/Materials/{tank}",
                    color,
                    Sdf,
                    Gf,
                    UsdShade,
                )
                binding = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
                binding.Bind(material)
                bound_material_prims.append(material.GetPrim())
                bound = bool(binding.GetDirectBinding().GetMaterial())
                bindings += bound
                effective_color = shader.GetInput(
                    "specular_transmission_color"
                ).Get()
                if any(
                    not math.isclose(
                        float(observed), expected, rel_tol=0.0, abs_tol=1e-6
                    )
                    for observed, expected in zip(effective_color, color)
                ):
                    raise RuntimeError(f"{tank} transmission color was not retained")
                effective_depth = float(
                    shader.GetInput(
                        "specular_transmission_scattering_depth"
                    ).Get()
                )
                effective_thin_walled = bool(
                    shader.GetInput("thin_walled").Get()
                )
                if not math.isclose(
                    effective_depth, 0.01, rel_tol=0.0, abs_tol=1e-8
                ) or effective_thin_walled:
                    raise RuntimeError(f"{tank} nonthin depth controls were not retained")
                material_records.append(
                    {
                        "tank_id": tank,
                        "specular_transmission_color": list(color),
                        "specular_transmission_scattering_depth_m": 0.01,
                        "thin_walled": False,
                        "bound": bound,
                    }
                )
            receipt["technical"]["prototype_material_bindings"] = int(bindings)
            semantic_prims = [instancer_prim, *prototype_prims, *bound_material_prims]
            for semantic_prim in semantic_prims:
                labels_api = UsdSemantics.LabelsAPI.Apply(semantic_prim, "class")
                if not labels_api or not labels_api.CreateLabelsAttr().Set(["liquid"]):
                    raise RuntimeError(
                        f"could not author liquid semantics on {semantic_prim.GetPath()}"
                    )
            receipt["semantics"]["labels_authored_before_stage_attach"] = True

        hidden_paths = []
        for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
            property_path = Sdf.Path(path).AppendProperty("visibility")
            session_spec = session.GetPropertyAtPath(property_path)
            visibility = UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Get(
                Usd.TimeCode(1.0)
            )
            if session_spec and visibility == UsdGeom.Tokens.invisible:
                hidden_paths.append(path)
        namespace_prims = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(NAMESPACE)
        ]
        forbidden = ["/World"] if session.GetPrimAtPath("/World") else []
        receipt["scene_contract"].update(
            {
                "session_layer_only": bool(session.GetPrimAtPath(NAMESPACE))
                and not bool(stage.GetRootLayer().GetPrimAtPath(NAMESPACE)),
                "hidden_source_paths": hidden_paths,
                "custom_camera_count": sum(
                    prim.IsA(UsdGeom.Camera) for prim in namespace_prims
                ),
                "custom_light_count": sum(
                    prim.GetTypeName().endswith("Light") for prim in namespace_prims
                ),
                "custom_backdrop_count": sum(
                    "backdrop" in str(prim.GetPath()).lower()
                    for prim in namespace_prims
                ),
                "forbidden_authored_paths": forbidden,
            }
        )
        particle_set_api = PhysxSchema.PhysxParticleSetAPI(instancer_prim)
        particle_api = PhysxSchema.PhysxParticleAPI(particle_set_api)
        instancer_for_contract = UsdGeom.PointInstancer(instancer_prim)
        prototype_targets = instancer_for_contract.GetPrototypesRel().GetTargets()
        receipt["technical"].update(
            {
                "creation_api": "omni.physx.scripts.particleUtils.add_physx_particleset_pointinstancer",
                "fluid_particle_set": bool(particle_set_api.GetFluidAttr().Get()),
                "particle_system_target_count": len(
                    particle_api.GetParticleSystemRel().GetTargets()
                ),
                "sphere_prototype_count": sum(
                    stage.GetPrimAtPath(path).IsA(UsdGeom.Sphere)
                    for path in prototype_targets
                ),
                "material_variants": material_records,
                "collider_count": len(collider_prims),
                "all_colliders_invisible": all(
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
                    == UsdGeom.Tokens.invisible
                    for prim in collider_prims
                ),
            }
        )

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("opened twin has no stage-cache ID")
        simulation.attach_stage(stage_id)
        attached = True
        initial_array = np.asarray(
            [[float(component) for component in row] for row in initial_positions],
            dtype=np.float64,
        )
        dt = 1.0 / 60.0
        with Usd.EditContext(stage, session):
            for _ in range(2):
                app.update()
            for step in range(1, 31):
                simulation.simulate(dt, step * dt)
                simulation.fetch_results()
                app.update()
        receipt["technical"]["physics_steps_before_swap"] = 30
        instancer = UsdGeom.PointInstancer(instancer_prim)
        positions_before_value = instancer.GetPositionsAttr().Get()
        positions_before = np.asarray(
            [[float(component) for component in row] for row in positions_before_value],
            dtype=np.float64,
        )
        indices_before = [int(value) for value in instancer.GetProtoIndicesAttr().Get()]
        systems = [
            prim for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)
        ]
        movement = np.linalg.norm(positions_before - initial_array, axis=1)
        finite_before = _finite_positions(positions_before_value)
        receipt["technical"].update(
            {
                "particle_system_count": len(systems),
                "particle_count": int(len(positions_before)),
                "prototype_count": len(instancer.GetPrototypesRel().GetTargets()),
                "finite_positions": bool(finite_before),
                "moved_under_physics": bool(np.max(movement) > 1e-6),
                "maximum_movement_m": float(np.max(movement)),
                "indices_before": indices_before,
                "positions_before_sha256": hashlib.sha256(
                    positions_before.tobytes(order="C")
                ).hexdigest(),
                "positions_before_swap": positions_before.tolist(),
            }
        )
        receipt["technical"].update(
            _runtime_property_specs(stage, instancer, Sdf)
        )

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        annotator.attach([render_product])
        semantic_annotator.attach([render_product])
        receipt["semantics"][
            "annotator_attached_after_labels_and_stage_attach"
        ] = True
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(60):
            app.update()
        receipt["render"]["warmup_updates"] = 60

        try:
            before, before_mask, before_id_to_labels, before_liquid_ids = _capture_rgb_semantic(
                rep, annotator, semantic_annotator, np, capture_name="before"
            )
        except _SemanticCaptureError as exc:
            _record_rejected_semantic_diagnostic(
                receipt, output, exc.capture_name, exc.diagnostics, cv2
            )
            raise
        before_path = output / "before-swap.png"
        if not cv2.imwrite(os.fspath(before_path), cv2.cvtColor(before, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write before-swap image")
        before_mask_path = output / "before-liquid-mask.png"
        if not cv2.imwrite(os.fspath(before_mask_path), before_mask):
            raise RuntimeError("could not write before liquid mask")
        before_mask_reopened = cv2.imread(
            os.fspath(before_mask_path), cv2.IMREAD_GRAYSCALE
        )
        before_mask_readable = bool(
            before_mask_reopened is not None
            and before_mask_reopened.shape == before.shape[:2]
            and np.array_equal(before_mask_reopened, before_mask)
        )
        if not before_mask_readable:
            raise RuntimeError("before liquid mask did not round-trip exactly")
        receipt["semantics"]["before"] = {
            "mask_path": before_mask_path.name,
            "mask_sha256": _sha256(before_mask_path),
            "rgb_sha256": _sha256(before_path),
            "mask_readable": before_mask_readable,
            "mask_aligned_to_rgb": bool(before_mask.shape == before.shape[:2]),
            "liquid_ids": before_liquid_ids,
            "liquid_pixel_count": int(np.count_nonzero(before_mask)),
            "idToLabels": before_id_to_labels,
        }

        indices_after_expected = swap_orange_blue(indices_before)
        with Usd.EditContext(stage, session):
            if not instancer.GetProtoIndicesAttr().Set(indices_after_expected):
                raise RuntimeError("could not author swapped prototype indices")
            app.update()
        positions_after_value = instancer.GetPositionsAttr().Get()
        positions_after = np.asarray(
            [[float(component) for component in row] for row in positions_after_value],
            dtype=np.float64,
        )
        indices_after = [int(value) for value in instancer.GetProtoIndicesAttr().Get()]
        receipt["technical"]["physics_steps_during_swap"] = 0
        receipt["technical"].update(
            {
                "finite_positions": bool(
                    receipt["technical"]["finite_positions"]
                    and _finite_positions(positions_after_value)
                ),
                "indices_after": indices_after,
                "positions_identical_across_swap": bool(
                    np.array_equal(positions_before, positions_after)
                ),
                "position_bytes_identical_across_swap": bool(
                    positions_before.tobytes(order="C")
                    == positions_after.tobytes(order="C")
                ),
                "positions_after_sha256": hashlib.sha256(
                    positions_after.tobytes(order="C")
                ).hexdigest(),
                "positions_after_swap": positions_after.tolist(),
            }
        )
        receipt["technical"].update(
            _runtime_property_specs(stage, instancer, Sdf)
        )
        try:
            after, after_mask, after_id_to_labels, after_liquid_ids = _capture_rgb_semantic(
                rep, annotator, semantic_annotator, np, capture_name="after"
            )
        except _SemanticCaptureError as exc:
            _record_rejected_semantic_diagnostic(
                receipt, output, exc.capture_name, exc.diagnostics, cv2
            )
            raise
        after_path = output / "after-swap.png"
        if not cv2.imwrite(os.fspath(after_path), cv2.cvtColor(after, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write after-swap image")
        after_mask_path = output / "after-liquid-mask.png"
        if not cv2.imwrite(os.fspath(after_mask_path), after_mask):
            raise RuntimeError("could not write after liquid mask")
        after_mask_reopened = cv2.imread(
            os.fspath(after_mask_path), cv2.IMREAD_GRAYSCALE
        )
        after_mask_readable = bool(
            after_mask_reopened is not None
            and after_mask_reopened.shape == after.shape[:2]
            and np.array_equal(after_mask_reopened, after_mask)
        )
        if not after_mask_readable:
            raise RuntimeError("after liquid mask did not round-trip exactly")
        receipt["semantics"]["after"] = {
            "mask_path": after_mask_path.name,
            "mask_sha256": _sha256(after_mask_path),
            "rgb_sha256": _sha256(after_path),
            "mask_readable": after_mask_readable,
            "mask_aligned_to_rgb": bool(after_mask.shape == after.shape[:2]),
            "liquid_ids": after_liquid_ids,
            "liquid_pixel_count": int(np.count_nonzero(after_mask)),
            "idToLabels": after_id_to_labels,
        }

        crop_slice = (slice(80, 465), slice(100, 430))
        before_crop = before[crop_slice]
        after_crop = after[crop_slice]
        before_crop_path = output / "before-swap-crop.png"
        after_crop_path = output / "after-swap-crop.png"
        if not cv2.imwrite(os.fspath(before_crop_path), cv2.cvtColor(before_crop, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write before crop")
        if not cv2.imwrite(os.fspath(after_crop_path), cv2.cvtColor(after_crop, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write after crop")
        contact_pair = np.concatenate([before_crop, after_crop], axis=1)
        contact_pair_path = output / "swap-contact-pair.png"
        if not cv2.imwrite(
            os.fspath(contact_pair_path),
            cv2.cvtColor(contact_pair, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("could not write swap contact pair")
        before_reopened = cv2.imread(os.fspath(before_path), cv2.IMREAD_COLOR)
        after_reopened = cv2.imread(os.fspath(after_path), cv2.IMREAD_COLOR)
        before_readable = (
            before_reopened is not None and before_reopened.shape[:2] == (480, 640)
        )
        after_readable = (
            after_reopened is not None and after_reopened.shape[:2] == (480, 640)
        )

        receipt["render"].update(
            {
                "before_path": before_path.name,
                "before_sha256": _sha256(before_path),
                "after_path": after_path.name,
                "after_sha256": _sha256(after_path),
                "before_crop_path": before_crop_path.name,
                "before_crop_sha256": _sha256(before_crop_path),
                "after_crop_path": after_crop_path.name,
                "after_crop_sha256": _sha256(after_crop_path),
                "contact_pair_path": contact_pair_path.name,
                "contact_pair_sha256": _sha256(contact_pair_path),
                "before_readable": bool(before_readable),
                "after_readable": bool(after_readable),
                "before_nonconstant": bool(np.any(before != before[0, 0])),
                "after_nonconstant": bool(np.any(after != after[0, 0])),
            }
        )
        receipt["status"] = "pending_visual"
        failures = validate_technical_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(
            f"palette smoke failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if annotator is not None and render_product is not None:
            try:
                annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"annotator detach failed: {exc}")
        if semantic_annotator is not None and render_product is not None:
            try:
                semantic_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(
                    f"semantic annotator detach failed: {exc}"
                )
        if attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception as exc:
                receipt["warnings"].append(f"stage detach failed: {exc}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    from isaacsim import SimulationApp

    kit = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "width": 640,
            "height": 480,
            "extra_args": list(EXTRA_ARGS),
        }
    )
    exit_code = 2
    try:
        import omni.kit.app

        receipt = run_smoke(args, omni.kit.app.get_app(), args.out)
        _write_json(args.out / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pending_visual" else 2
    finally:
        kit.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
