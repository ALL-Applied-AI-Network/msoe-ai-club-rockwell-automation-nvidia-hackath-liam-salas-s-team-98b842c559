#!/usr/bin/env python3
"""Run the frozen A36 first-two-stage PhysX array diagnostic without rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fluidmix.particle_array_probe import (  # noqa: E402
    ARRAY_NAMES,
    CANONICAL_TWIN_SHA256,
    CAPTURE_STEPS,
    CONTACT_OFFSET_M,
    EXPECTED_PARTICLE_COUNTS,
    EXPERIMENT,
    MAX_VELOCITY_M_S,
    PARTICLE_CONTACT_OFFSET_M,
    PARTICLE_DENSITY_ARGUMENT_KG_M3,
    PARTICLE_MASS_ARGUMENT_KG,
    PARTICLE_TO_NONPARTICLE_REST_OFFSET_M,
    PHYSICS_HZ,
    SCHEMA_VERSION,
    SEED,
    SOLID_REST_OFFSET_M,
    WALL_LIMIT_SECONDS,
    build_probe_plan,
    center_support_diagnostic,
    validate_probe_receipt,
)
from fluidmix.physx_three_pour import (  # noqa: E402
    COMPOSITION_COMPONENTS,
    FLUID_REST_OFFSET_M,
    FLUIDMIX_NAMESPACE,
    NOMINAL_TANK_TRANSMISSION_COLORS,
    NOZZLE_EXIT_CENTER_M,
    SOURCE_LIQUID_BOTTOM_M,
    SOURCE_LIQUID_PATH,
    SOURCE_LIQUID_RADIUS_M,
    SOURCE_STREAM_PATH,
    VIAL_CENTER_XY_M,
    VIAL_MOUTH_M,
    build_composition_palette,
    build_paint_ball_three_pour_plan,
    build_palette_transmission_colors,
    build_twin_collider_specs,
    map_compositions_to_palette,
)
from fluidmix.paint_ball_emitter import containment_evidence  # noqa: E402
from fluidmix.particle_palette import mix_linear_rgb  # noqa: E402

# The A36 runner imports the already exercised A34 authoring and timeline
# helpers.  It does not call Replicator and never creates a render product.
import physx_three_pour_sequence as f2  # noqa: E402


VIAL_COLLIDER_WALL_COUNT = 12
PROBE_NAMESPACE = FLUIDMIX_NAMESPACE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    parser.add_argument(
        "--wall-limit-seconds",
        type=int,
        choices=(WALL_LIMIT_SECONDS,),
        default=WALL_LIMIT_SECONDS,
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
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


def _array(values: object, *, dtype: object) -> Any:
    import numpy as np

    result = np.asarray(values, dtype=dtype)
    if result.dtype.kind in "fc" and not np.all(np.isfinite(result)):
        raise RuntimeError("particle array contains non-finite values")
    return np.ascontiguousarray(result)


def _save_array(output: Path, relative: Path, values: Any) -> dict[str, object]:
    import numpy as np

    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    reloaded = np.load(path, allow_pickle=False)
    if reloaded.dtype != values.dtype or reloaded.shape != values.shape or not np.array_equal(
        reloaded, values
    ):
        raise RuntimeError(f"lossless NPY verification failed for {relative}")
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(
            values.dtype.kind not in "fc" or np.all(np.isfinite(values))
        ),
        "allow_pickle": False,
    }


def _empty_receipt(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "insufficient_evidence",
        "errors": [],
        "plan": build_probe_plan(),
        "command": {
            "argv": [os.fspath(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "cwd": os.fspath(Path.cwd()),
            "script_sha256": _sha256(Path(__file__)),
            "seed": args.seed,
            "denominator": 1,
        },
        "source": {
            "stage": os.fspath(args.stage),
            "stage_sha256": None,
            "canonical_stage_sha256": CANONICAL_TWIN_SHA256,
            "layers_before": None,
            "layers_after": None,
            "source_layers_unchanged": None,
        },
        "runtime": {
            "isaac_sim": None,
            "kit": None,
            "physx": None,
            "render_product_created": False,
            "rgb_captured": False,
            "video_created": False,
            "failure_phase": "startup",
            "wall_seconds": None,
        },
        "gpu_simulation": {
            "cuda_device_available": None,
            "scene_enable_gpu_dynamics": None,
            "broadphase": None,
        },
        "particle_system": {},
        "point_instancer": {},
        "colliders": {},
        "emission": {
            "events": [],
            "origin_m": list(NOZZLE_EXIT_CENTER_M),
            "seed": args.seed,
        },
        "diffusion": {},
        "timing": {},
        "stages": [],
        "artifacts": {},
    }


def _pbd_material_readback(pbd: Any) -> dict[str, float | bool]:
    """Read every A34 water-preset value and authored-value predicate."""

    attributes = {
        "density": pbd.GetDensityAttr(),
        "friction": pbd.GetFrictionAttr(),
        "cohesion": pbd.GetCohesionAttr(),
        "damping": pbd.GetDampingAttr(),
        "surface_tension": pbd.GetSurfaceTensionAttr(),
        "viscosity": pbd.GetViscosityAttr(),
        "vorticity_confinement": pbd.GetVorticityConfinementAttr(),
    }
    readback: dict[str, float | bool] = {}
    for name, attribute in attributes.items():
        value_name = "density_kg_m3" if name == "density" else name
        readback[value_name] = float(attribute.Get())
        readback[f"{name}_authored"] = bool(
            attribute.HasAuthoredValueOpinion()
        )
    return readback


def _vec3_rows(values: object) -> list[list[float]]:
    if values is None:
        values = []
    return [
        [float(value[0]), float(value[1]), float(value[2])]
        for value in values
    ]


def _contained_mask(positions: Any) -> Any:
    return f2._count_driven_contained_mask(positions)


def _capture_stage_arrays(
    *,
    output: Path,
    step: int,
    particle_points: Any,
    compositions: Any,
    expected_ids: list[int],
    particle_to_nonparticle_rest_offset_m: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np

    observed_ids = f2._read_runtime_particle_identity(particle_points, expected_ids)
    arrays = {
        "raw_simulation_positions": _array(
            _vec3_rows(particle_points.GetSimulationPointsAttr().Get()),
            dtype=np.float64,
        ),
        "smoothed_instancer_positions": _array(
            _vec3_rows(f2._positions(particle_points.GetPrim(), instancer=True)),
            dtype=np.float64,
        ),
        "velocities": _array(
            _vec3_rows(particle_points.GetVelocitiesAttr().Get()), dtype=np.float64
        ),
        "scales": _array(
            _vec3_rows(particle_points.GetScalesAttr().Get()), dtype=np.float64
        ),
        "ids": _array(observed_ids, dtype=np.int64),
        "proto_indices": _array(
            [
                int(value)
                for value in (particle_points.GetProtoIndicesAttr().Get() or [])
            ],
            dtype=np.int64,
        ),
        "compositions": _array(compositions, dtype=np.float64),
    }
    expected_count = EXPECTED_PARTICLE_COUNTS[step]
    expected_shapes = {
        "raw_simulation_positions": (expected_count, 3),
        "smoothed_instancer_positions": (expected_count, 3),
        "velocities": (expected_count, 3),
        "scales": (expected_count, 3),
        "ids": (expected_count,),
        "proto_indices": (expected_count,),
        "compositions": (expected_count, len(COMPOSITION_COMPONENTS)),
    }
    mismatched = {
        name: {"actual": list(values.shape), "expected": list(expected_shapes[name])}
        for name, values in arrays.items()
        if values.shape != expected_shapes[name]
    }
    if mismatched:
        raise RuntimeError(f"stage array shapes differ: {mismatched}")
    if not np.array_equal(arrays["ids"], np.arange(expected_count, dtype=np.int64)):
        raise RuntimeError("stage particle IDs differ from exact scheduled identity")
    if np.any(arrays["compositions"] < 0.0) or not np.allclose(
        arrays["compositions"].sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("stage compositions violate nonnegative row-sum contract")

    inner_apothem = SOURCE_LIQUID_RADIUS_M * math.cos(
        math.pi / VIAL_COLLIDER_WALL_COUNT
    )
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
    raw = arrays["raw_simulation_positions"]
    smoothed = arrays["smoothed_instancer_positions"]
    raw_containment = containment_evidence(
        raw,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        cylindrical_center_xy=VIAL_CENTER_XY_M,
        inner_radius_m=inner_apothem,
        polygon_center_xy=VIAL_CENTER_XY_M,
        polygon_apothem_m=inner_apothem,
        polygon_wall_count=VIAL_COLLIDER_WALL_COUNT,
    )
    smoothed_containment = containment_evidence(
        smoothed,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        cylindrical_center_xy=VIAL_CENTER_XY_M,
        inner_radius_m=inner_apothem,
        polygon_center_xy=VIAL_CENTER_XY_M,
        polygon_apothem_m=inner_apothem,
        polygon_wall_count=VIAL_COLLIDER_WALL_COUNT,
    )
    raw_inside = _contained_mask(raw)
    smoothed_inside = _contained_mask(smoothed)
    if (
        int(np.count_nonzero(raw_inside)) != expected_count
        or raw_containment["inside_vial_interior_count"] != expected_count
        or raw_containment["outside_vial_interior_count"] != 0
        or raw_containment["leaked_particle_count"] != 0
    ):
        raise RuntimeError("stage raw particle containment mismatch")
    if int(np.count_nonzero(smoothed_inside)) == 0:
        raise RuntimeError("stage smoothed array has no contained particle centres")

    stage_dir = Path(f"step-{step:05d}")
    artifact_records: dict[str, dict[str, object]] = {}
    manifest_records: list[dict[str, object]] = []
    for name in ARRAY_NAMES:
        record = _save_array(output, stage_dir / f"{name}.npy", arrays[name])
        artifact_records[name] = record
        manifest_records.append({"kind": name, "physics_step": step, **record})

    raw_diagnostic = center_support_diagnostic(
        raw[raw_inside, 2],
        particle_to_nonparticle_rest_offset_m=particle_to_nonparticle_rest_offset_m,
        prototype_radius_m=FLUID_REST_OFFSET_M,
        prototype_scale_z=float(np.max(np.abs(arrays["scales"][:, 2]))),
    )
    smoothed_diagnostic = center_support_diagnostic(
        smoothed[smoothed_inside, 2],
        particle_to_nonparticle_rest_offset_m=particle_to_nonparticle_rest_offset_m,
        prototype_radius_m=FLUID_REST_OFFSET_M,
        prototype_scale_z=float(np.max(np.abs(arrays["scales"][:, 2]))),
    )
    return (
        {
            "physics_step": step,
            "expected_particle_count": expected_count,
            "emitted_particle_count": len(expected_ids),
            "inside_vial_interior_count": int(
                raw_containment["inside_vial_interior_count"]
            ),
            "outside_vial_interior_count": int(
                raw_containment["outside_vial_interior_count"]
            ),
            "escaped_particle_count": int(raw_containment["leaked_particle_count"]),
            "ids_exact": observed_ids == expected_ids,
            "raw_containment": raw_containment,
            "smoothed_containment": smoothed_containment,
            "raw_z_min_m": float(np.min(raw[raw_inside, 2])),
            "raw_z_max_m": float(np.max(raw[raw_inside, 2])),
            "smoothed_z_min_m": float(np.min(smoothed[smoothed_inside, 2])),
            "smoothed_z_max_m": float(np.max(smoothed[smoothed_inside, 2])),
            "arrays": artifact_records,
            "diagnostics": {
                "raw_simulation_positions": raw_diagnostic,
                "smoothed_instancer_positions": smoothed_diagnostic,
            },
        },
        manifest_records,
    )


def run_probe(
    args: argparse.Namespace,
    app: Any,
    output: Path,
    receipt: dict[str, object],
    *,
    particle_mass_argument_kg: float = PARTICLE_MASS_ARGUMENT_KG,
    mass_transition_callback: Any | None = None,
    max_particles: int = 2221,
    diffusion_enabled: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import carb
    import numpy as np
    import omni.physx
    from omni.physx.scripts import particleUtils
    import omni.timeline
    import omni.usd
    from omni.physxfoundation import get_physx_foundation_interface
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    started = time.monotonic()
    deadline = started + args.wall_limit_seconds
    timeline = None
    subscription = None
    velocity_writeback_setting = None
    velocity_writeback_initial = None
    manifest_records: list[dict[str, object]] = []

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise RuntimeError("A36 wall-time safety stop reached")

    try:
        receipt["runtime"]["failure_phase"] = "runtime_and_stage_contract"
        manager = app.get_extension_manager()
        versions = f2._version_info(app)
        receipt["runtime"].update(
            {
                "isaac_sim": versions.get("isaac"),
                "kit": versions.get("kit"),
                "physx": versions.get("physx"),
            }
        )
        if not manager.is_extension_enabled("omni.physx"):
            raise RuntimeError("omni.physx extension is not enabled")
        foundation = get_physx_foundation_interface()
        receipt["gpu_simulation"]["cuda_device_available"] = bool(
            foundation.cuda_device_check()
        )
        if receipt["gpu_simulation"]["cuda_device_available"] is not True:
            raise RuntimeError("GPU particles require a CUDA device")

        stage_sha = _sha256(args.stage)
        receipt["source"]["stage_sha256"] = stage_sha
        if stage_sha != CANONICAL_TWIN_SHA256:
            raise RuntimeError("stage SHA-256 differs from canonical frozen twin")
        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError(f"could not open twin stage: {args.stage}")
        while context.get_stage_loading_status()[2] > 0:
            check_deadline()
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened twin stage")
        layers_before = f2._source_layer_records(stage)
        receipt["source"]["layers_before"] = layers_before
        default_prim = stage.GetDefaultPrim()
        if (
            not default_prim.IsValid()
            or str(default_prim.GetPath()) != "/FillStation"
            or float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
            or str(UsdGeom.GetStageUpAxis(stage)) != "Z"
        ):
            raise RuntimeError("opened stage is not the reviewed metre-scale Z-up twin")
        session_layer = stage.GetSessionLayer()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

        settings = carb.settings.get_settings()
        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_setting = getattr(
            physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
        )
        velocity_writeback_setting = getattr(
            physx_bindings, "SETTING_UPDATE_VELOCITIES_TO_USD", None
        )
        if particles_setting is None or velocity_writeback_setting is None:
            raise RuntimeError("PhysX USD particle/velocity writeback settings are missing")
        settings.set(particles_setting, True)
        velocity_writeback_initial = bool(
            settings.get_as_bool(velocity_writeback_setting)
        )
        settings.set_bool(velocity_writeback_setting, True)
        if not settings.get_as_bool(velocity_writeback_setting):
            raise RuntimeError("PhysX velocity USD writeback did not enable")

        receipt["runtime"]["failure_phase"] = "author_exact_a34_physics_graph"
        with Usd.EditContext(stage, session_layer):
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                prim = stage.GetPrimAtPath(hidden_path)
                if not prim.IsValid():
                    raise RuntimeError(f"source liquid prim missing: {hidden_path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )

            UsdGeom.Xform.Define(stage, PROBE_NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{PROBE_NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            scene_api.CreateSolverTypeAttr().Set("TGS")
            receipt["gpu_simulation"].update(
                {
                    "scene_enable_gpu_dynamics": bool(
                        scene_api.GetEnableGPUDynamicsAttr().Get()
                    ),
                    "broadphase": str(scene_api.GetBroadphaseTypeAttr().Get()),
                }
            )

            system_path = Sdf.Path(f"{PROBE_NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateEnableCCDAttr().Set(True)
            system.CreateParticleContactOffsetAttr().Set(PARTICLE_CONTACT_OFFSET_M)
            system.CreateFluidRestOffsetAttr().Set(FLUID_REST_OFFSET_M)
            system.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
            system.CreateRestOffsetAttr().Set(
                PARTICLE_TO_NONPARTICLE_REST_OFFSET_M
            )
            system.CreateSolidRestOffsetAttr().Set(SOLID_REST_OFFSET_M)
            system.CreateMaxVelocityAttr().Set(MAX_VELOCITY_M_S)
            particleUtils.add_physx_particle_smoothing(
                stage, system_path, enabled=True, strength=1.0
            )
            particleUtils.add_physx_particle_anisotropy(
                stage, system_path, enabled=True, scale=1.0
            )
            smoothing_api = PhysxSchema.PhysxParticleSmoothingAPI(system.GetPrim())
            anisotropy_api = PhysxSchema.PhysxParticleAnisotropyAPI(system.GetPrim())

            physics_material = UsdShade.Material.Define(
                stage, f"{PROBE_NAMESPACE}/PBDMaterial"
            )
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(physics_material.GetPrim())
            f2._author_pbd_material_treatment(
                pbd=pbd,
                material_prim=physics_material.GetPrim(),
                particle_utils=particleUtils,
                use_water_preset=True,
            )
            pbd_readback = _pbd_material_readback(pbd)
            if not f2._water_pbd_preset_readback_matches_plan(
                pbd_readback, meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage))
            ):
                raise RuntimeError("water PBD preset readback differs from A34")
            binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
            binding.Bind(
                physics_material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )

            global_material = UsdShade.Material.Define(
                stage, f"{PROBE_NAMESPACE}/GlobalLiquidMaterial"
            )
            global_shader = f2._author_omnisurface(
                stage,
                global_material,
                NOMINAL_TANK_TRANSMISSION_COLORS[COMPOSITION_COMPONENTS[0]],
                Sdf,
                Gf,
                UsdShade,
            )
            instancer_path = Sdf.Path(
                f"{PROBE_NAMESPACE}/FluidParticles/Shared"
            )
            instancer_prim = particleUtils.add_physx_particleset_pointinstancer(
                stage,
                instancer_path,
                [],
                [],
                system_path,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=particle_mass_argument_kg,
                density=PARTICLE_DENSITY_ARGUMENT_KG_M3,
                num_prototypes=0,
            )
            instancer = UsdGeom.PointInstancer(instancer_prim)
            point_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(instancer_prim)
            instancer_prim.CreateAttribute(
                "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
            ).Set(max_particles)
            point_set_api.GetSimulationPointsAttr().Set(Vt.Vec3fArray([]))
            instancer.CreateIdsAttr().Set(Vt.Int64Array([]))
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([]))

            palette_rows = build_composition_palette()
            palette_colors = build_palette_transmission_colors(palette_rows)
            pure_palette_indices = {
                int(np.argmax(row)): index
                for index, row in enumerate(palette_rows)
                if int(np.count_nonzero(row == 1.0)) == 1
                and int(np.count_nonzero(row)) == 1
            }
            source_prototype_indices = {
                COMPOSITION_COMPONENTS[component]: pure_palette_indices[component]
                for component in range(len(COMPOSITION_COMPONENTS))
            }
            source_materials: dict[str, Any] = {}
            for tank_id in COMPOSITION_COMPONENTS:
                material_path = (
                    f"{PROBE_NAMESPACE}/StreamMaterials/{tank_id.upper()}"
                )
                source_material = UsdShade.Material.Define(stage, material_path)
                f2._author_omnisurface(
                    stage,
                    source_material,
                    NOMINAL_TANK_TRANSMISSION_COLORS[tank_id],
                    Sdf,
                    Gf,
                    UsdShade,
                )
                source_materials[tank_id] = source_material
            for prototype_index, color in enumerate(palette_colors.tolist()):
                prototype_path = instancer_path.AppendChild(
                    f"particlePrototype{prototype_index}"
                )
                sphere = UsdGeom.Sphere.Define(stage, prototype_path)
                sphere.CreateRadiusAttr().Set(FLUID_REST_OFFSET_M)
                sphere.CreateDisplayColorAttr().Set(
                    Vt.Vec3fArray(
                        [Gf.Vec3f(*[float(component) for component in color])]
                    )
                )
                instancer.GetPrototypesRel().AddTarget(prototype_path)
                pure_component = next(
                    (
                        component
                        for component, index in pure_palette_indices.items()
                        if index == prototype_index
                    ),
                    None,
                )
                if pure_component is None:
                    prototype_material = UsdShade.Material.Define(
                        stage,
                        f"{PROBE_NAMESPACE}/PaletteMaterials/Color{prototype_index:02d}",
                    )
                    f2._author_omnisurface(
                        stage,
                        prototype_material,
                        tuple(float(component) for component in color),
                        Sdf,
                        Gf,
                        UsdShade,
                    )
                else:
                    prototype_material = source_materials[
                        COMPOSITION_COMPONENTS[pure_component]
                    ]
                UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                    prototype_material
                )
            binding.Bind(global_material)

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

            scene_api.GetTimeStepsPerSecondAttr().Set(PHYSICS_HZ)

        if scene_api.GetTimeStepsPerSecondAttr().Get() != PHYSICS_HZ:
            raise RuntimeError("PhysicsScene did not retain 120 Hz")
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(physx_bindings.SETTING_MIN_FRAME_RATE, PHYSICS_HZ)

        system_readback = {
            "particle_contact_offset_m": float(
                system.GetParticleContactOffsetAttr().Get()
            ),
            "fluid_rest_offset_m": float(system.GetFluidRestOffsetAttr().Get()),
            "contact_offset_m": float(system.GetContactOffsetAttr().Get()),
            "rest_offset_m": float(system.GetRestOffsetAttr().Get()),
            "solid_rest_offset_m": float(system.GetSolidRestOffsetAttr().Get()),
            "max_velocity_m_s": float(system.GetMaxVelocityAttr().Get()),
            "ccd_enabled": bool(system.GetEnableCCDAttr().Get()),
            "smoothing_enabled": bool(
                smoothing_api.GetParticleSmoothingEnabledAttr().Get()
            ),
            "smoothing_strength": float(smoothing_api.GetStrengthAttr().Get()),
            "anisotropy_enabled": bool(
                anisotropy_api.GetParticleAnisotropyEnabledAttr().Get()
            ),
            "anisotropy_scale": float(anisotropy_api.GetScaleAttr().Get()),
            "solver_position_iteration_count": int(
                system.GetSolverPositionIterationCountAttr().Get()
            ),
            "max_neighborhood": int(system.GetMaxNeighborhoodAttr().Get()),
            "pbd_material_readback": pbd_readback,
        }
        mass_api = UsdPhysics.MassAPI(instancer_prim)
        system_readback["mass_density_readback"] = {
            "particle_mass_argument_kg": particle_mass_argument_kg,
            "density_argument_kg_m3": PARTICLE_DENSITY_ARGUMENT_KG_M3,
            "mass_api_present": bool(instancer_prim.HasAPI(UsdPhysics.MassAPI)),
            "total_mass_attr_kg": float(mass_api.GetMassAttr().Get()),
            "total_mass_attr_authored": bool(
                mass_api.GetMassAttr().HasAuthoredValueOpinion()
            ),
            "density_attr_kg_m3": float(mass_api.GetDensityAttr().Get()),
            "density_attr_authored": bool(
                mass_api.GetDensityAttr().HasAuthoredValueOpinion()
            ),
        }
        receipt["particle_system"] = system_readback
        receipt["point_instancer"] = {
            "path": str(instancer_path),
            "prototype_count": len(palette_rows),
            "max_particles": max_particles,
            "one_shared_point_instancer": True,
            "simulation_points_authored": True,
        }
        collider_specs = json.loads(
            json.dumps(build_twin_collider_specs(), allow_nan=False)
        )
        floor_spec = collider_specs[0]
        receipt["colliders"] = {
            "specs": collider_specs,
            "floor_path": floor_spec["path"],
            "floor_top_z_m": float(floor_spec["center_m"][2])
            + float(floor_spec["size_m"][2]) / 2.0,
            "visible_inner_bottom_z_m": SOURCE_LIQUID_BOTTOM_M,
            "floor_matches_visible_inner_bottom": math.isclose(
                float(floor_spec["center_m"][2])
                + float(floor_spec["size_m"][2]) / 2.0,
                SOURCE_LIQUID_BOTTOM_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "all_invisible": all(
                UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
                == UsdGeom.Tokens.invisible
                for prim in collider_prims
            ),
        }

        # Match A34's two post-authoring Kit updates before configuring the
        # zero-origin timeline.  The timeline is stopped and no render product exists.
        for _ in range(2):
            f2._app_update_in_session(stage, session_layer, app, Usd)

        plan = build_paint_ball_three_pour_plan()
        event_index = f2._build_emission_event_index(plan)
        particle_points = f2._PointInstancerParticleAdapter(instancer, point_set_api)
        callback_records: list[dict[str, object]] = []
        emission_records: list[dict[str, object]] = []
        active_row_metadata: list[dict[str, object]] = []
        compositions_by_id: dict[int, Any] = {}
        shared_particle_ids: list[int] = []
        next_particle_id = 0
        cumulative = 0
        rng = random.Random(args.seed)
        timeline_app_update_index = 0
        diffusion_updates = 0
        diffusion_neighbors = 0

        receipt["runtime"]["failure_phase"] = "configure_timeline"
        timeline_configuration = f2._configure_zero_origin_timeline(
            stage,
            session_layer,
            app,
            timeline,
            Usd,
            physics_hz=PHYSICS_HZ,
            final_callback_index=CAPTURE_STEPS[-1],
        )
        timeline_playback_start = f2._start_timeline_non_looping(
            stage,
            session_layer,
            timeline,
            Usd,
            configured=timeline_configuration,
        )

        def on_physics_step(callback_dt: float) -> None:
            callback_records.append(
                {
                    "callback_index": len(callback_records) + 1,
                    "dt_s": float(callback_dt),
                    "timeline_time_s": float(timeline.get_current_time()),
                    "app_update_index": timeline_app_update_index,
                }
            )

        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_physics_step
        )
        receipt["runtime"]["failure_phase"] = "run_particle_timeline"
        while len(callback_records) < CAPTURE_STEPS[-1]:
            global_step = len(callback_records) + 1
            check_deadline()
            for event in event_index.get(global_step, []):
                count = int(event["particle_count"])
                particle_ids = list(
                    range(next_particle_id, next_particle_id + count)
                )

                def append_rows() -> None:
                    positions = f2._twin_chunk_positions(Gf, count, rng)
                    existing_positions = list(
                        f2._positions(particle_points.GetPrim(), instancer=True)
                    )
                    existing_simulation = list(
                        particle_points.GetSimulationPointsAttr().Get() or []
                    )
                    existing_velocities = list(
                        particle_points.GetVelocitiesAttr().Get() or []
                    )
                    existing_orientations = list(
                        particle_points.GetOrientationsAttr().Get() or []
                    )
                    existing_scales = list(
                        particle_points.GetScalesAttr().Get() or []
                    )
                    existing_ids = [
                        int(value)
                        for value in (particle_points.GetIdsAttr().Get() or [])
                    ]
                    existing_proto_indices = list(
                        particle_points.GetProtoIndicesAttr().Get() or []
                    )
                    if not (
                        len(existing_positions)
                        == len(existing_simulation)
                        == len(existing_velocities)
                        == len(existing_orientations)
                        == len(existing_scales)
                        == len(existing_ids)
                        == len(existing_proto_indices)
                        == len(shared_particle_ids)
                    ) or existing_ids != shared_particle_ids:
                        raise RuntimeError(
                            "shared PointInstancer rows lost exact A34 identity"
                        )
                    particle_points.GetSimulationPointsAttr().Set(
                        Vt.Vec3fArray([*existing_simulation, *positions])
                    )
                    particle_points.GetPositionsAttr().Set(
                        Vt.Vec3fArray([*existing_positions, *positions])
                    )
                    particle_points.GetVelocitiesAttr().Set(
                        Vt.Vec3fArray(
                            [
                                *existing_velocities,
                                *[Gf.Vec3f(0.0, 0.0, -0.10)] * count,
                            ]
                        )
                    )
                    particle_points.GetOrientationsAttr().Set(
                        Vt.QuathArray(
                            [
                                *existing_orientations,
                                *[Gf.Quath(1.0, 0.0, 0.0, 0.0)] * count,
                            ]
                        )
                    )
                    particle_points.GetScalesAttr().Set(
                        Vt.Vec3fArray(
                            [*existing_scales, *[Gf.Vec3f(1.0, 1.0, 1.0)] * count]
                        )
                    )
                    particle_points.CreateIdsAttr().Set(
                        Vt.Int64Array([*existing_ids, *particle_ids])
                    )
                    particle_points.GetProtoIndicesAttr().Set(
                        Vt.IntArray(
                            [
                                *existing_proto_indices,
                                *[
                                    source_prototype_indices[str(event["tank_id"])]
                                ]
                                * count,
                            ]
                        )
                    )

                f2._with_session_edit_context(
                    Usd, stage, session_layer, append_rows
                )
                if mass_transition_callback is not None:
                    transition_ordinal = len(emission_records) + 1
                    active_particle_count = len(shared_particle_ids) + count
                    transition = f2._with_session_edit_context(
                        Usd,
                        stage,
                        session_layer,
                        lambda: mass_transition_callback(
                            mass_api,
                            transition_ordinal=transition_ordinal,
                            physics_step=global_step,
                            active_particle_count=active_particle_count,
                        ),
                    )
                    treatment = receipt.get("explicit_mass_treatment")
                    if not isinstance(treatment, dict) or not isinstance(
                        treatment.get("transitions"), list
                    ):
                        raise RuntimeError(
                            "explicit mass receipt was not initialized"
                        )
                    treatment["transitions"].append(transition)
                source_compositions = np.zeros(
                    (count, len(COMPOSITION_COMPONENTS)), dtype=np.float64
                )
                source_compositions[
                    :, COMPOSITION_COMPONENTS.index(str(event["tank_id"]))
                ] = 1.0
                active_row_metadata.append(
                    {
                        "particle_ids": particle_ids,
                        "particle_count": count,
                        "source_compositions": source_compositions,
                        "tank_id": str(event["tank_id"]),
                        "pour_ordinal": int(event["pour_ordinal"]),
                    }
                )
                shared_particle_ids.extend(particle_ids)
                next_particle_id += count
                cumulative += count
                for particle_id, composition in zip(
                    particle_ids, source_compositions
                ):
                    compositions_by_id[particle_id] = composition.copy()
                emission_records.append(
                    {
                        "tank_id": str(event["tank_id"]),
                        "pour_ordinal": int(event["pour_ordinal"]),
                        "emission_ordinal": int(event["ordinal"]),
                        "physics_step": int(event["physics_step"]),
                        "particle_count": count,
                        "particle_id_start": particle_ids[0],
                        "particle_id_end": particle_ids[-1],
                        "cumulative_particle_count": cumulative,
                        "authored_after_callback_count": global_step - 1,
                        "scheduled_before_callback_index": global_step,
                    }
                )

            before_count = len(callback_records)
            timeline_app_update_index += 1
            f2._app_update_in_session(stage, session_layer, app, Usd)
            if len(callback_records) != before_count + 1:
                raise RuntimeError(
                    "timeline produced not exactly one callback per app update"
                )
            callback_contract = f2._timeline_callback_contract_evidence(
                callback_records[-1],
                global_step=global_step,
                app_update_index=timeline_app_update_index,
                physics_hz=PHYSICS_HZ,
            )
            if callback_contract["passed"] is not True:
                raise RuntimeError(
                    "timeline callback drift: "
                    + ",".join(callback_contract["failed_predicates"])
                )

            observed_ids = f2._read_runtime_particle_identity(
                particle_points, shared_particle_ids
            )
            raw_positions = _array(
                _vec3_rows(particle_points.GetSimulationPointsAttr().Get()),
                dtype=np.float64,
            )
            if diffusion_enabled:
                updated_compositions, diffusion_diagnostics = (
                    f2._diffuse_active_particle_rows(
                        active_row_metadata,
                        compositions_by_id,
                        raw_positions,
                        observed_ids,
                        dt_s=1.0 / PHYSICS_HZ,
                    )
                )
            else:
                updated_compositions = np.asarray(
                    [compositions_by_id[particle_id] for particle_id in observed_ids],
                    dtype=np.float64,
                )
                diffusion_diagnostics = {"neighbor_count": 0}
            if len(updated_compositions):
                mapped = map_compositions_to_palette(updated_compositions)
                with Usd.EditContext(stage, session_layer):
                    particle_points.GetProtoIndicesAttr().Set(
                        Vt.IntArray([int(value) for value in mapped.tolist()])
                    )
                mean_color = mix_linear_rgb(
                    updated_compositions.mean(axis=0)[None, :],
                    np.asarray(
                        [
                            NOMINAL_TANK_TRANSMISSION_COLORS[tank]
                            for tank in COMPOSITION_COMPONENTS
                        ],
                        dtype=np.float64,
                    ),
                )[0]
                f2._with_session_edit_context(
                    Usd,
                    stage,
                    session_layer,
                    lambda: f2._set_global_blend_color(
                        global_shader, Gf, mean_color
                    ),
                )
            diffusion_updates += int(diffusion_enabled)
            diffusion_neighbors += int(diffusion_diagnostics["neighbor_count"])

            if global_step in CAPTURE_STEPS:
                stage_record, records = _capture_stage_arrays(
                    output=output,
                    step=global_step,
                    particle_points=particle_points,
                    compositions=updated_compositions,
                    expected_ids=shared_particle_ids,
                    particle_to_nonparticle_rest_offset_m=float(
                        system.GetRestOffsetAttr().Get()
                    ),
                )
                receipt["stages"].append(stage_record)
                if mass_transition_callback is not None:
                    transitions = receipt["explicit_mass_treatment"][
                        "transitions"
                    ]
                    stage_record["mass_state"] = dict(transitions[-1])
                manifest_records.extend(records)

        timeline.pause()
        timeline.commit()
        timeline_final = f2._timeline_state_readback(stage, timeline)
        if timeline_final["timeline_looping"] is not False:
            raise RuntimeError("timeline looping became enabled")
        callback_trace_path = output / "callback-trace.json"
        _write_json(callback_trace_path, callback_records)
        callback_trace_record = {
            "kind": "callback_trace",
            "path": callback_trace_path.relative_to(output).as_posix(),
            "sha256": _sha256(callback_trace_path),
            "rows": len(callback_records),
        }
        manifest_records.append(callback_trace_record)
        receipt["emission"]["events"] = emission_records
        receipt["emission"]["event_count"] = len(emission_records)
        receipt["emission"]["emitted_particle_count"] = cumulative
        receipt["diffusion"] = {
            "enabled": diffusion_enabled,
            "update_count": diffusion_updates,
            "total_neighbor_count": diffusion_neighbors,
            "composition_components": list(COMPOSITION_COMPONENTS),
            "composition_rows_saved_at_each_stage": True,
        }
        receipt["timing"] = {
            "physics_hz": PHYSICS_HZ,
            "executed_physics_steps": len(callback_records),
            "callbacks_observed": len(callback_records),
            "callback_indices_exact": all(
                record["callback_index"] == index
                for index, record in enumerate(callback_records, start=1)
            ),
            "callback_dts_exact": all(
                math.isclose(
                    float(record["dt_s"]),
                    1.0 / PHYSICS_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for record in callback_records
            ),
            "callback_times_exact": all(
                math.isclose(
                    float(record["timeline_time_s"]),
                    index / PHYSICS_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for index, record in enumerate(callback_records, start=1)
            ),
            "callback_app_update_indices_exact": all(
                record["app_update_index"] == index
                for index, record in enumerate(callback_records, start=1)
            ),
            "trace_path": callback_trace_record["path"],
            "trace_sha256": callback_trace_record["sha256"],
            "timeline_looping": bool(timeline_final["timeline_looping"]),
            "timeline_configuration": timeline_configuration,
            "timeline_playback_start": timeline_playback_start,
            "timeline_final": timeline_final,
        }
        receipt["source"]["layers_after"] = f2._source_layer_records(stage)
        receipt["source"]["source_layers_unchanged"] = (
            receipt["source"]["layers_before"] == receipt["source"]["layers_after"]
        )
        if receipt["source"]["source_layers_unchanged"] is not True:
            raise RuntimeError("source layer stack changed during session-layer probe")
        receipt["runtime"]["failure_phase"] = "complete"
        return receipt, manifest_records
    finally:
        subscription = None
        if timeline is not None:
            try:
                timeline.pause()
                timeline.commit()
            except Exception:
                pass
        if velocity_writeback_setting is not None and velocity_writeback_initial is not None:
            try:
                carb.settings.get_settings().set_bool(
                    velocity_writeback_setting, velocity_writeback_initial
                )
            except Exception:
                pass
        receipt["runtime"]["wall_seconds"] = time.monotonic() - started


def _build_manifest(
    output: Path,
    receipt: dict[str, object],
    runtime_records: list[dict[str, object]],
) -> dict[str, object]:
    records = list(runtime_records)
    for kind, relative in (
        ("plan", Path("plan.json")),
        ("command", Path("command.json")),
    ):
        path = output / relative
        records.append(
            {"kind": kind, "path": relative.as_posix(), "sha256": _sha256(path)}
        )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "seed": SEED,
        "denominator": 1,
        "status": receipt["status"],
        "records": sorted(records, key=lambda record: str(record["path"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.stage.is_absolute() or not args.stage.is_file():
        parser.error("stage must be an existing absolute file")
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    plan = build_probe_plan()
    _write_json(args.out / "plan.json", plan)
    if args.plan_only:
        return 0

    receipt = _empty_receipt(args)
    _write_json(args.out / "command.json", receipt["command"])
    app_owner = None
    runtime_records: list[dict[str, object]] = []
    exit_code = 2
    try:
        app, app_owner = f2._start_or_reuse_kit()
        receipt, runtime_records = run_probe(args, app, args.out, receipt)
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["errors"].append(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
    try:
        # A manifest is evidence even on a failed run.  Only a completed pass is
        # contract-validated and allowed to retain status=pass.
        manifest = _build_manifest(args.out, receipt, runtime_records)
        manifest_path = args.out / "artifact-manifest.json"
        _write_json(manifest_path, manifest)
        receipt["artifacts"] = {
            "manifest_path": manifest_path.relative_to(args.out).as_posix(),
            "manifest_sha256": _sha256(manifest_path),
            "record_count": len(manifest["records"]),
        }
        if receipt["status"] == "pass":
            failures = validate_probe_receipt(receipt, artifact_root=args.out)
            if failures:
                receipt["status"] = "insufficient_evidence"
                receipt["errors"].extend(failures)
        receipt_path = args.out / "receipt.json"
        _write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pass" else 2
    finally:
        if app_owner is not None:
            app_owner.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

