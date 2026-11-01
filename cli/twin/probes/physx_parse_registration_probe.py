#!/usr/bin/env python3
"""Run A39: parse-time registration control using A37's saved compact state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from twin.probes.parse_registration_probe import (  # noqa: E402
    A37_INPUT_ARRAYS,
    A37_MANIFEST_SHA256,
    A37_RECEIPT_SHA256,
    DENSITY_KG_M3,
    EXPECTED_PARTICLE_COUNT,
    EXPERIMENT,
    PARTICLE_MASS_KG,
    PHYSICS_HZ,
    PHYSICS_STEPS,
    SCHEMA_VERSION,
    SEED,
    TOTAL_MASS_KG,
    build_probe_plan,
    evaluate_probe,
    particle_metrics,
    validate_input_arrays,
    validate_probe_receipt,
)
from twin.probes.particle_array_probe import (  # noqa: E402
    CANONICAL_TWIN_SHA256,
    CONTACT_OFFSET_M,
    MAX_VELOCITY_M_S,
    PARTICLE_CONTACT_OFFSET_M,
    PARTICLE_TO_NONPARTICLE_REST_OFFSET_M,
    SOLID_REST_OFFSET_M,
    WALL_LIMIT_SECONDS,
)
from twin.physx.three_pour import (  # noqa: E402
    FLUID_REST_OFFSET_M,
    FLUIDMIX_NAMESPACE,
    SOURCE_LIQUID_PATH,
    SOURCE_LIQUID_RADIUS_M,
    SOURCE_STREAM_PATH,
    VIAL_CENTER_XY_M,
    build_twin_collider_specs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cli.twin.probes.physx_particle_array_probe as a36  # noqa: E402
import cli.twin.physx.physx_three_pour_sequence as f2  # noqa: E402


PARTICLE_SYSTEM_PATH = f"{FLUIDMIX_NAMESPACE}/ParticleSystem"
POINT_INSTANCER_PATH = f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared"
VIAL_WALL_COUNT = 12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--a37-root", type=Path, required=True)
    parser.add_argument(
        "--a37-receipt-sha256", choices=(A37_RECEIPT_SHA256,), required=True
    )
    parser.add_argument(
        "--a37-manifest-sha256", choices=(A37_MANIFEST_SHA256,), required=True
    )
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


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    root = args.a37_root.resolve(strict=True)
    receipt_path = root / "receipt.json"
    manifest_path = root / "artifact-manifest.json"
    if _sha256(receipt_path) != args.a37_receipt_sha256:
        raise RuntimeError("A37 receipt hash differs from frozen input")
    if _sha256(manifest_path) != args.a37_manifest_sha256:
        raise RuntimeError("A37 manifest hash differs from frozen input")
    arrays: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for name, specification in A37_INPUT_ARRAYS.items():
        source = root / str(specification["path"])
        observed_sha = _sha256(source)
        if observed_sha != specification["sha256"]:
            raise RuntimeError(f"A37 input hash differs for {name}")
        value = np.load(source, allow_pickle=False)
        arrays[name] = value
        records[name] = {
            "source_path": os.fspath(source),
            "source_sha256": observed_sha,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    failures = validate_input_arrays(arrays)
    if failures:
        raise RuntimeError("A37 input contract failed: " + ",".join(failures))
    return arrays, {
        "source_root": os.fspath(root),
        "receipt_path": os.fspath(receipt_path),
        "receipt_sha256": args.a37_receipt_sha256,
        "manifest_path": os.fspath(manifest_path),
        "manifest_sha256": args.a37_manifest_sha256,
        "hashes_match_frozen": True,
        "arrays_valid": True,
        "array_records": records,
    }


def _copy_frozen_inputs(
    output: Path, args: argparse.Namespace, input_record: dict[str, Any]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    root = args.a37_root.resolve(strict=True)
    for name, specification in A37_INPUT_ARRAYS.items():
        source = root / str(specification["path"])
        destination = output / "frozen-input" / f"{name}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        observed = _sha256(destination)
        if observed != specification["sha256"]:
            raise RuntimeError(f"copied A37 input hash differs for {name}")
        relative = destination.relative_to(output).as_posix()
        input_record["array_records"][name]["copied_path"] = relative
        input_record["array_records"][name]["copied_sha256"] = observed
        records.append(
            {
                "kind": f"frozen_input_{name}",
                "path": relative,
                "sha256": observed,
                "bytes": destination.stat().st_size,
            }
        )
    return records


def _empty_receipt(
    args: argparse.Namespace, input_record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "insufficient_evidence",
        "errors": [],
        "plan": build_probe_plan(),
        "command": {
            "argv": [os.fspath(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "cwd": os.fspath(Path.cwd()),
            "seed": args.seed,
            "denominator": 1,
            "git_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_status_short": subprocess.run(
                ["git", "status", "--short"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines(),
            "script_sha256": _sha256(Path(__file__)),
        },
        "input": input_record,
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
            "physics_callbacks": 0,
            "callback_contract_passed": False,
            "failure_phase": "startup",
            "wall_seconds": None,
        },
        "gpu_simulation": {},
        "particle_system": {},
        "particle_contract": {},
        "colliders": {},
        "snapshots": {},
        "hypothesis_evaluation": None,
        "experiment_outcome": None,
        "artifacts": {
            "manifest_valid": False,
            "required_arrays_valid": False,
        },
    }


def _vec3_rows(values: object) -> list[list[float]]:
    if values is None:
        values = []
    return [
        [float(value[0]), float(value[1]), float(value[2])]
        for value in values
    ]


def _snapshot(
    *,
    output: Path,
    name: str,
    particle_points: Any,
    expected_ids: list[int],
    compositions: Any,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    import numpy as np

    observed_ids = f2._read_runtime_particle_identity(
        particle_points, expected_ids
    )
    arrays = {
        "raw_simulation_positions": np.asarray(
            _vec3_rows(particle_points.GetSimulationPointsAttr().Get()),
            dtype=np.float64,
        ),
        "smoothed_instancer_positions": np.asarray(
            _vec3_rows(f2._positions(particle_points.GetPrim(), instancer=True)),
            dtype=np.float64,
        ),
        "velocities": np.asarray(
            _vec3_rows(particle_points.GetVelocitiesAttr().Get()),
            dtype=np.float64,
        ),
        "scales": np.asarray(
            _vec3_rows(particle_points.GetScalesAttr().Get()), dtype=np.float64
        ),
        "ids": np.asarray(observed_ids, dtype=np.int64),
        "proto_indices": np.asarray(
            [
                int(value)
                for value in (particle_points.GetProtoIndicesAttr().Get() or [])
            ],
            dtype=np.int64,
        ),
        "compositions": np.asarray(compositions, dtype=np.float64),
    }
    expected_shapes = {
        "raw_simulation_positions": (EXPECTED_PARTICLE_COUNT, 3),
        "smoothed_instancer_positions": (EXPECTED_PARTICLE_COUNT, 3),
        "velocities": (EXPECTED_PARTICLE_COUNT, 3),
        "scales": (EXPECTED_PARTICLE_COUNT, 3),
        "ids": (EXPECTED_PARTICLE_COUNT,),
        "proto_indices": (EXPECTED_PARTICLE_COUNT,),
        "compositions": (EXPECTED_PARTICLE_COUNT, 3),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise RuntimeError(
                f"{name} {key} shape {arrays[key].shape} != {expected}"
            )
        if arrays[key].dtype.kind in "fc" and not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"{name} {key} contains non-finite values")
    raw = arrays["raw_simulation_positions"]
    inside = np.asarray(f2._count_driven_contained_mask(raw), dtype=bool)
    inside_count = int(np.count_nonzero(inside))
    outside_count = EXPECTED_PARTICLE_COUNT - inside_count
    escaped_count = int(f2._twin_escaped_count(raw.tolist()))
    snapshot_records: dict[str, Any] = {}
    manifest_records: list[dict[str, object]] = []
    for key, value in arrays.items():
        relative = Path("snapshots") / name / f"{key}.npy"
        artifact = a36._save_array(output, relative, np.ascontiguousarray(value))
        snapshot_records[key] = artifact
        manifest_records.append({"kind": f"{name}_{key}", **artifact})
    metrics = particle_metrics(raw)
    return (
        {
            "count": len(raw),
            "inside_count": inside_count,
            "outside_count": outside_count,
            "escaped_count": escaped_count,
            "ids_exact": observed_ids == expected_ids,
            "finite": all(
                value.dtype.kind not in "fc" or np.all(np.isfinite(value))
                for value in arrays.values()
            ),
            "z_min_m": float(np.min(raw[:, 2])),
            "z_max_m": float(np.max(raw[:, 2])),
            "metrics": metrics,
            "arrays": snapshot_records,
        },
        manifest_records,
    )


def _run(
    args: argparse.Namespace,
    app: Any,
    output: Path,
    receipt: dict[str, Any],
    arrays: dict[str, Any],
) -> list[dict[str, object]]:
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
    velocity_setting = None
    velocity_initial = None
    records: list[dict[str, object]] = []

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise RuntimeError("A39 wall-time safety stop reached")

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
        particle_setting = getattr(
            physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
        )
        velocity_setting = getattr(
            physx_bindings, "SETTING_UPDATE_VELOCITIES_TO_USD", None
        )
        if particle_setting is None or velocity_setting is None:
            raise RuntimeError("PhysX particle writeback settings are missing")
        settings.set(particle_setting, True)
        velocity_initial = bool(settings.get_as_bool(velocity_setting))
        settings.set_bool(velocity_setting, True)

        receipt["runtime"]["failure_phase"] = "author_nonempty_preplayback_graph"
        with Usd.EditContext(stage, session_layer):
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                prim = stage.GetPrimAtPath(hidden_path)
                if not prim.IsValid():
                    raise RuntimeError(f"source liquid prim missing: {hidden_path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
            UsdGeom.Xform.Define(stage, FLUIDMIX_NAMESPACE)
            scene = UsdPhysics.Scene.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/PhysicsScene"
            )
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            scene_api.CreateSolverTypeAttr().Set("TGS")

            system_path = Sdf.Path(PARTICLE_SYSTEM_PATH)
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
            smoothing_api = PhysxSchema.PhysxParticleSmoothingAPI(
                system.GetPrim()
            )
            anisotropy_api = PhysxSchema.PhysxParticleAnisotropyAPI(
                system.GetPrim()
            )

            material = UsdShade.Material.Define(
                stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial"
            )
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
            f2._author_pbd_material_treatment(
                pbd=pbd,
                material_prim=material.GetPrim(),
                particle_utils=particleUtils,
                use_water_preset=True,
            )
            pbd_readback = a36._pbd_material_readback(pbd)
            if not f2._water_pbd_preset_readback_matches_plan(
                pbd_readback,
                meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
            ):
                raise RuntimeError("water PBD preset readback differs from A37")
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

            positions = [Gf.Vec3f(*row) for row in arrays["raw_simulation_positions"]]
            velocities = [Gf.Vec3f(*row) for row in arrays["velocities"]]
            proto_indices = [int(value) for value in arrays["proto_indices"]]
            instancer_prim = particleUtils.add_physx_particleset_pointinstancer(
                stage,
                Sdf.Path(POINT_INSTANCER_PATH),
                positions,
                velocities,
                system_path,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=PARTICLE_MASS_KG,
                density=DENSITY_KG_M3,
                num_prototypes=15,
                prototype_indices=proto_indices,
            )
            instancer = UsdGeom.PointInstancer(instancer_prim)
            point_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(instancer_prim)
            particle_api = PhysxSchema.PhysxParticleAPI(instancer_prim)
            instancer_prim.CreateAttribute(
                "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
            ).Set(2221)
            point_set_api.GetSimulationPointsAttr().Set(Vt.Vec3fArray(positions))
            instancer.CreateIdsAttr().Set(
                Vt.Int64Array([int(value) for value in arrays["ids"]])
            )
            instancer.CreateOrientationsAttr().Set(
                Vt.QuathArray(
                    [Gf.Quath(1.0, 0.0, 0.0, 0.0)]
                    * EXPECTED_PARTICLE_COUNT
                )
            )
            instancer.CreateScalesAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*row) for row in arrays["scales"]])
            )
            for prototype in instancer.GetPrototypesRel().GetTargets():
                UsdGeom.Sphere(stage.GetPrimAtPath(prototype)).CreateRadiusAttr().Set(
                    FLUID_REST_OFFSET_M
                )

            collider_prims = []
            for specification in build_twin_collider_specs():
                cube = UsdGeom.Cube.Define(
                    stage, Sdf.Path(str(specification["path"]))
                )
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*specification["center_m"]))
                cube.AddRotateZOp().Set(float(specification["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*specification["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())
            scene_api.GetTimeStepsPerSecondAttr().Set(PHYSICS_HZ)

        particle_points = f2._PointInstancerParticleAdapter(
            instancer, point_set_api
        )
        expected_ids = [int(value) for value in arrays["ids"]]
        initial, initial_records = _snapshot(
            output=output,
            name="initial-authored",
            particle_points=particle_points,
            expected_ids=expected_ids,
            compositions=arrays["compositions"],
        )
        records.extend(initial_records)
        receipt["snapshots"]["initial_authored"] = initial

        mass_api = UsdPhysics.MassAPI(instancer_prim)
        mass_readback = float(mass_api.GetMassAttr().Get())
        density_readback = float(mass_api.GetDensityAttr().Get())
        system_targets = [
            str(value) for value in particle_api.GetParticleSystemRel().GetTargets()
        ]
        global_self_attr = system.GetGlobalSelfCollisionEnabledAttr()
        global_self_value = global_self_attr.Get()
        receipt["particle_contract"] = {
            "nonempty_at_helper_call": len(positions) == EXPECTED_PARTICLE_COUNT,
            "nonempty_before_first_post_authoring_update": True,
            "nonempty_before_timeline_play": True,
            "runtime_append_count": 0,
            "fluid": bool(point_set_api.GetFluidAttr().Get()),
            "self_collision": bool(particle_api.GetSelfCollisionAttr().Get()),
            "particle_group": int(particle_api.GetParticleGroupAttr().Get()),
            "particle_system_targets": system_targets,
            "mass_readback_kg": mass_readback,
            "expected_total_mass_kg": TOTAL_MASS_KG,
            "mass_readback_matches": math.isclose(
                mass_readback, TOTAL_MASS_KG, rel_tol=0.0, abs_tol=1.0e-8
            ),
            "density_readback_kg_m3": density_readback,
            "density_readback_matches": math.isclose(
                density_readback, DENSITY_KG_M3, rel_tol=0.0, abs_tol=1.0e-6
            ),
            "global_self_collision_authored": bool(
                global_self_attr.HasAuthoredValueOpinion()
            ),
            "global_self_collision_effective": (
                None if global_self_value is None else bool(global_self_value)
            ),
            "saved_velocities_used_unchanged": True,
            "saved_scales_used_unchanged": True,
            "identity_orientation_deviation": True,
        }
        receipt["particle_system"] = {
            "particle_contact_offset_m": float(
                system.GetParticleContactOffsetAttr().Get()
            ),
            "fluid_rest_offset_m": float(system.GetFluidRestOffsetAttr().Get()),
            "contact_offset_m": float(system.GetContactOffsetAttr().Get()),
            "rest_offset_m": float(system.GetRestOffsetAttr().Get()),
            "solid_rest_offset_m": float(
                system.GetSolidRestOffsetAttr().Get()
            ),
            "max_velocity_m_s": float(system.GetMaxVelocityAttr().Get()),
            "ccd_enabled": bool(system.GetEnableCCDAttr().Get()),
            "smoothing_enabled": bool(
                smoothing_api.GetParticleSmoothingEnabledAttr().Get()
            ),
            "anisotropy_enabled": bool(
                anisotropy_api.GetParticleAnisotropyEnabledAttr().Get()
            ),
            "solver_position_iteration_count": int(
                system.GetSolverPositionIterationCountAttr().Get()
            ),
            "max_neighborhood": int(system.GetMaxNeighborhoodAttr().Get()),
            "pbd_material_readback": pbd_readback,
        }
        floor = build_twin_collider_specs()[0]
        floor_top = float(floor["center_m"][2]) + float(floor["size_m"][2]) / 2
        receipt["colliders"] = {
            "specs": json.loads(json.dumps(build_twin_collider_specs())),
            "floor_top_z_m": floor_top,
            "floor_matches_visible_inner_bottom": math.isclose(
                floor_top, 0.9762065645, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "all_invisible": all(
                UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
                == UsdGeom.Tokens.invisible
                for prim in collider_prims
            ),
        }
        receipt["gpu_simulation"] = {
            **receipt["gpu_simulation"],
            "scene_enable_gpu_dynamics": bool(
                scene_api.GetEnableGPUDynamicsAttr().Get()
            ),
            "broadphase": str(scene_api.GetBroadphaseTypeAttr().Get()),
        }

        for _ in range(2):
            f2._app_update_in_session(stage, session_layer, app, Usd)
        preplay_ids = f2._read_runtime_particle_identity(
            particle_points, expected_ids
        )
        preplay_positions = np.asarray(
            _vec3_rows(particle_points.GetSimulationPointsAttr().Get()),
            dtype=np.float64,
        )
        receipt["particle_contract"].update(
            {
                "preplay_count": len(preplay_positions),
                "preplay_ids_exact": preplay_ids == expected_ids,
                "preplay_positions_byte_equivalent": bool(
                    np.array_equal(
                        preplay_positions,
                        arrays["raw_simulation_positions"].astype(
                            np.float32
                        ).astype(np.float64),
                    )
                ),
            }
        )
        if (
            len(preplay_positions) != EXPECTED_PARTICLE_COUNT
            or preplay_ids != expected_ids
        ):
            raise RuntimeError("preplay particle state changed before timeline play")

        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(physx_bindings.SETTING_MIN_FRAME_RATE, PHYSICS_HZ)
        receipt["runtime"]["failure_phase"] = "configure_timeline"
        configured = f2._configure_zero_origin_timeline(
            stage,
            session_layer,
            app,
            timeline,
            Usd,
            physics_hz=PHYSICS_HZ,
            final_callback_index=PHYSICS_STEPS,
        )
        playback = f2._start_timeline_non_looping(
            stage, session_layer, timeline, Usd, configured=configured
        )
        callback_records: list[dict[str, object]] = []
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

        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_physics_step
        )
        receipt["runtime"]["failure_phase"] = "run_preparsed_state"
        callback_passed = True
        while len(callback_records) < PHYSICS_STEPS:
            check_deadline()
            step = len(callback_records) + 1
            before = len(callback_records)
            app_update_index += 1
            f2._app_update_in_session(stage, session_layer, app, Usd)
            if len(callback_records) != before + 1:
                callback_passed = False
                raise RuntimeError("timeline did not emit exactly one callback per update")
            contract = f2._timeline_callback_contract_evidence(
                callback_records[-1],
                global_step=step,
                app_update_index=app_update_index,
                physics_hz=PHYSICS_HZ,
            )
            if contract["passed"] is not True:
                callback_passed = False
                raise RuntimeError(
                    "timeline callback drift: "
                    + ",".join(contract["failed_predicates"])
                )
        timeline.pause()
        trace_path = output / "callback-trace.json"
        _write_json(trace_path, callback_records)
        records.append(
            {
                "kind": "callback_trace",
                "path": trace_path.relative_to(output).as_posix(),
                "sha256": _sha256(trace_path),
                "bytes": trace_path.stat().st_size,
            }
        )
        receipt["runtime"].update(
            {
                "physics_callbacks": len(callback_records),
                "callback_contract_passed": callback_passed,
                "timeline_configuration": configured,
                "timeline_playback": playback,
            }
        )

        final, final_records = _snapshot(
            output=output,
            name="final-simulated",
            particle_points=particle_points,
            expected_ids=expected_ids,
            compositions=arrays["compositions"],
        )
        records.extend(final_records)
        receipt["snapshots"]["final_simulated"] = final
        receipt["hypothesis_evaluation"] = evaluate_probe(
            initial["metrics"], final["metrics"]
        )
        receipt["experiment_outcome"] = receipt["hypothesis_evaluation"][
            "decision"
        ]
        receipt["source"]["layers_after"] = f2._source_layer_records(stage)
        receipt["source"]["source_layers_unchanged"] = (
            receipt["source"]["layers_after"]
            == receipt["source"]["layers_before"]
        )
        if receipt["source"]["source_layers_unchanged"] is not True:
            raise RuntimeError("source layer hashes changed")
        receipt["runtime"]["failure_phase"] = "complete"
        receipt["runtime"]["wall_seconds"] = time.monotonic() - started
        receipt["status"] = "pass"
        return records
    finally:
        receipt["runtime"]["wall_seconds"] = time.monotonic() - started
        if subscription is not None:
            subscription.unsubscribe()
        if timeline is not None:
            timeline.pause()
        if velocity_setting is not None and velocity_initial is not None:
            carb.settings.get_settings().set_bool(
                velocity_setting, velocity_initial
            )


def _manifest(output: Path, records: list[dict[str, object]]) -> dict[str, Any]:
    paths = {str(record["path"]) for record in records}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        relative = path.relative_to(output).as_posix()
        if relative in paths or relative in {"receipt.json", "artifact-manifest.json"}:
            continue
        records.append(
            {
                "kind": "run_metadata",
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        paths.add(relative)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "records": sorted(records, key=lambda row: str(row["path"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.stage.is_absolute() or not args.stage.is_file():
        parser.error("stage must be an existing absolute file")
    if not args.a37_root.is_absolute() or not args.a37_root.is_dir():
        parser.error("a37-root must be an existing absolute directory")
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    _write_json(args.out / "plan.json", build_probe_plan())
    if args.plan_only:
        return 0

    arrays, input_record = _load_inputs(args)
    records = _copy_frozen_inputs(args.out, args, input_record)
    receipt = _empty_receipt(args, input_record)
    _write_json(args.out / "command.json", receipt["command"])
    app_owner = None
    exit_code = 2
    try:
        app, app_owner = f2._start_or_reuse_kit()
        records.extend(_run(args, app, args.out, receipt, arrays))
    except Exception as exc:
        receipt["errors"].append(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
    try:
        manifest = _manifest(args.out, records)
        manifest_path = args.out / "artifact-manifest.json"
        _write_json(manifest_path, manifest)
        receipt["artifacts"] = {
            "manifest_path": manifest_path.relative_to(args.out).as_posix(),
            "manifest_sha256": _sha256(manifest_path),
            "record_count": len(manifest["records"]),
            "manifest_valid": all(
                _sha256(args.out / str(record["path"])) == record["sha256"]
                for record in manifest["records"]
            ),
            "required_arrays_valid": all(
                record.get("finite", True) is True
                for record in manifest["records"]
                if str(record["path"]).endswith(".npy")
            ),
        }
        if receipt["status"] == "pass":
            failures = validate_probe_receipt(receipt)
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
