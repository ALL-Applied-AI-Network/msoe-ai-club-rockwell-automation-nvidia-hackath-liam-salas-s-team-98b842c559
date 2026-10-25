#!/usr/bin/env python3
"""Run one bounded Isaac 6.2 spatial OmniSurface capability probe."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.physx_three_pour import (
    CAMERA_CLIPPING_RANGE,
    CAMERA_PATH,
    O1B_MATERIAL_PARAMETERS,
    SOURCE_LIQUID_PATH,
    SOURCE_STREAM_PATH,
    TWIN_REFERENCE_TIME_CODE,
)
from fluidmix.spatial_omnisurface_probe import (
    EXPECTED_ISAAC_SIM_VERSION,
    EXPECTED_WRAPPER_MDL_SHA256,
    WALL_LIMIT_SECONDS,
    build_fixture_spatial_mapping,
    build_probe_plan,
    build_static_probe_fixture,
    measure_constant_parity,
    record_vertex_color_readback,
    validate_probe_receipt,
)


NAMESPACE = "/FillStation/FluidmixSpatialOmniSurfaceProbe"
MESH_PATH = f"{NAMESPACE}/StaticLiquidMesh"
ORIGINAL_MATERIAL_PATH = f"{NAMESPACE}/Materials/OriginalOmniSurface"
WRAPPER_MATERIAL_PATH = f"{NAMESPACE}/Materials/SpatialOmniSurface"
WRAPPER_SUBIDENTIFIER = "FluidmixSpatialOmniSurface"
ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
EXTRA_ARGS = (f"--{ASYNC_SETTING}=false",)
MAX_CAPTURE_TRIES = 20


def _progress(phase: str) -> None:
    """Emit a durable watchdog marker without changing probe state."""

    print(
        json.dumps(
            {"event": "spatial_omnisurface_probe_progress", "phase": phase},
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--mdl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate the frozen probe inputs before Kit starts."""

    for name in ("stage", "mdl", "out"):
        value = getattr(args, name)
        if not isinstance(value, Path) or not value.is_absolute():
            raise ValueError(f"{name} path must be absolute")
    if not args.stage.is_file():
        raise ValueError("stage must be an existing regular file")
    if not args.mdl.is_file():
        raise ValueError("mdl must be an existing regular file")
    if args.mdl.name != "FluidmixSpatialOmniSurface.mdl":
        raise ValueError("mdl filename must be FluidmixSpatialOmniSurface.mdl")
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    if _sha256_file(args.mdl) != EXPECTED_WRAPPER_MDL_SHA256:
        raise ValueError("mdl bytes do not match the reviewed asset hash")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise TypeError("seed must be an integer")
    if args.seed != 20261025:
        raise ValueError("spatial OmniSurface capability probe seed must be 20261025")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: object, dtype: str) -> str:
    import numpy as np

    array = np.asarray(value, dtype=dtype, order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_topology(counts: object, indices: object) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(np.asarray(counts, dtype="<i4", order="C").tobytes())
    digest.update(np.asarray(indices, dtype="<i4", order="C").tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-safe receipt which fails closed until every gate passes."""

    plan = build_probe_plan()
    return {
        "schema_version": 1,
        "experiment": "spatial_omnisurface_capability_probe",
        "status": "insufficient_evidence",
        "plan": plan,
        "arguments": {
            "stage": os.fspath(args.stage),
            "mdl": os.fspath(args.mdl),
            "out": os.fspath(args.out),
            "seed": args.seed,
        },
        "runtime": {
            "isaac_sim_version": None,
            "wall_seconds": 0.0,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
        },
        "scene_contract": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "frozen_time_code": None,
            "session_layer_only": False,
            "source_layers_unchanged": False,
            "source_layer_records": [],
            "camera_path": CAMERA_PATH,
            "camera_clipping_range": None,
            "custom_camera_count": None,
            "custom_light_count": None,
            "custom_backdrop_count": None,
            "probe_mesh_path": MESH_PATH,
            "hidden_source_paths": [SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH],
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_bounces": 16,
            "max_specular_and_transmission_bounces": 16,
            "async_throttling_enabled": None,
            "capture_attempt_counts": {},
            "original_constant_path": None,
            "original_constant_sha256": None,
            "wrapper_constant_path": None,
            "wrapper_constant_sha256": None,
            "parity_absdiff_path": None,
            "parity_absdiff_sha256": None,
            "wrapper_spatial_path": None,
            "wrapper_spatial_sha256": None,
            "all_images_readable": False,
            "all_images_nonconstant": False,
        },
        "material": {
            "wrapper_mdl_path": os.fspath(args.mdl),
            "wrapper_mdl_sha256": None,
            "mdl_search_path_requested": os.fspath(args.mdl.parent),
            "mdl_search_path_registered": None,
            "mdl_search_path_removed": None,
            "wrapper_subidentifier": WRAPPER_SUBIDENTIFIER,
            "original_omnisurface_resolved": False,
            "original_resolution_message": None,
            "wrapper_omnisurface_resolved": False,
            "wrapper_resolution_message": None,
            "original_bound": False,
            "wrapper_bound_constant": False,
            "wrapper_bound_spatial": False,
            "same_family_omnisurface_delegate": False,
            "only_transmission_color_source_differs": False,
            "o1b_parameters": plan["omnisurface_parameters"],
        },
        "constant_parity": {
            "thresholds": plan["constant_parity_thresholds"],
            "mean_abs_rgb": None,
            "p95_abs_rgb": None,
            "fraction_abs_rgb_gt_16": None,
            "passed": False,
            "same_mesh_points_sha256": None,
            "same_mesh_topology_sha256": None,
            "same_display_color_sha256": None,
            "same_time_code": False,
        },
        "spatial": {
            "mapping_evidence": None,
            "nearest_particle_ids": None,
            "primvar_readback": None,
            "constant_to_spatial_rgb_change": None,
        },
        "claims": plan["claims"],
        "visual_review": {
            "status": "not_reached",
            "criteria": [
                "spatial capture visibly changes across the static liquid surface",
                "spatial regions are not an all-fallback or uniform-color render",
                "the probe remains inside the reviewed FillCam vial view",
            ],
        },
        "errors": [],
        "warnings": [],
    }


def _deadline_check(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("spatial OmniSurface probe reached its 900-second wall limit")


def _source_layer_records(stage: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if not layer.realPath:
            continue
        path = Path(layer.realPath).resolve(strict=True)
        records.append(
            {
                "identifier": str(layer.identifier),
                "resolved_path": os.fspath(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError("opened twin stage has no hashable non-session layers")
    return records


def _source_layers_unchanged(records: list[dict[str, Any]]) -> bool:
    return all(
        Path(record["resolved_path"]).is_file()
        and Path(record["resolved_path"]).stat().st_size == record["bytes"]
        and _sha256_file(Path(record["resolved_path"])) == record["sha256"]
        for record in records
    )


def _mesh_state(mesh: Any, primvar: Any) -> dict[str, Any]:
    import numpy as np

    points = np.asarray(
        [[float(component) for component in row] for row in mesh.GetPointsAttr().Get()],
        dtype=np.float32,
    )
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    colors = np.asarray(
        [[float(component) for component in row] for row in primvar.Get()],
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
        raise RuntimeError("probe mesh points did not read back as finite Nx3 values")
    if colors.shape != points.shape or not np.all(np.isfinite(points)) or not np.all(np.isfinite(colors)):
        raise RuntimeError("probe mesh or displayColor readback is malformed")
    if counts.ndim != 1 or indices.ndim != 1 or int(counts.sum()) != indices.size:
        raise RuntimeError("probe mesh topology readback is malformed")
    return {
        "points": points,
        "counts": counts,
        "indices": indices,
        "colors": colors,
        "points_sha256": _sha256_array(points, "<f4"),
        "topology_sha256": _sha256_topology(counts, indices),
        "display_color_sha256": _sha256_array(colors, "<f4"),
    }


def _connect_mdl_outputs(material: Any, shader: Any, Sdf: Any) -> None:
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    material.CreateVolumeOutput("mdl").ConnectToSource(output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(output)


def _author_original_material(stage: Any, Sdf: Any, Gf: Any, UsdShade: Any) -> Any:
    material = UsdShade.Material.Define(stage, ORIGINAL_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, f"{ORIGINAL_MATERIAL_PATH}/OmniSurface")
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_transmission_weight"],
        ),
        "specular_transmission_color": (
            Sdf.ValueTypeNames.Color3f,
            Gf.Vec3f(*O1B_MATERIAL_PARAMETERS["specular_transmission_color"]),
        ),
        "specular_transmission_scattering_depth": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_transmission_scattering_depth_m"],
        ),
        "thin_walled": (
            Sdf.ValueTypeNames.Bool,
            O1B_MATERIAL_PARAMETERS["thin_walled"],
        ),
        "diffuse_reflection_weight": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["diffuse_reflection_weight"],
        ),
        "specular_reflection_ior": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_reflection_ior"],
        ),
        "specular_reflection_roughness": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_reflection_roughness"],
        ),
        "enable_opacity": (
            Sdf.ValueTypeNames.Bool,
            O1B_MATERIAL_PARAMETERS["enable_opacity"],
        ),
        "geometry_opacity": (Sdf.ValueTypeNames.Float, 1.0),
        "geometry_opacity_threshold": (Sdf.ValueTypeNames.Float, 0.0),
    }
    for name, (type_name, value) in values.items():
        shader.CreateInput(name, type_name).Set(value)
    _connect_mdl_outputs(material, shader, Sdf)
    return material


def _author_wrapper_material(
    stage: Any, mdl: Path, Sdf: Any, Gf: Any, UsdShade: Any
) -> Any:
    material = UsdShade.Material.Define(stage, WRAPPER_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, f"{WRAPPER_MATERIAL_PATH}/SpatialOmniSurface")
    shader.SetSourceAsset(Sdf.AssetPath(mdl.name), "mdl")
    shader.SetSourceAssetSubIdentifier(WRAPPER_SUBIDENTIFIER, "mdl")
    shader.CreateInput("fallback_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*O1B_MATERIAL_PARAMETERS["specular_transmission_color"])
    )
    _connect_mdl_outputs(material, shader, Sdf)
    return material


def _bind_material(mesh: Any, material: Any, UsdShade: Any) -> bool:
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.UnbindAllBindings()
    binding.Bind(material)
    bound, relationship = binding.ComputeBoundMaterial()
    return bool(
        bound
        and bound.GetPrim().GetPath() == material.GetPrim().GetPath()
        and relationship
        and relationship.GetTargets() == [material.GetPath()]
    )


def _material_is_bound(mesh: Any, material: Any, UsdShade: Any) -> bool:
    bound, relationship = UsdShade.MaterialBindingAPI(
        mesh.GetPrim()
    ).ComputeBoundMaterial()
    return bool(
        bound
        and bound.GetPrim().GetPath() == material.GetPrim().GetPath()
        and relationship
        and relationship.GetTargets() == [material.GetPath()]
    )


def _resolve_material(stage: Any, material: Any, app: Any) -> tuple[bool, str]:
    import omni.mdl.usd_converter

    for _ in range(4):
        app.update()
    status, message = omni.mdl.usd_converter.is_shader_resolved(
        stage, material.GetPrim()
    )
    return bool(status), str(message or "")


def _set_vertex_colors(primvar: Any, colors: object, Gf: Any) -> None:
    import numpy as np

    canonical = np.asarray(colors, dtype=np.float32)
    if canonical.ndim != 2 or canonical.shape[1:] != (3,) or canonical.shape[0] == 0:
        raise ValueError("displayColor authoring requires a nonempty Nx3 array")
    if not np.all(np.isfinite(canonical)) or np.any(canonical < 0.0) or np.any(canonical > 1.0):
        raise ValueError("displayColor authoring values must be finite and in [0, 1]")
    if not primvar.Set([Gf.Vec3f(*[float(value) for value in row]) for row in canonical]):
        raise RuntimeError("could not author primvars:displayColor")


def _capture_rgb(
    rep: Any,
    annotator: Any,
    *,
    deadline: float,
) -> tuple[Any, int]:
    import numpy as np

    invalid: list[str] = []
    for attempt in range(1, MAX_CAPTURE_TRIES + 1):
        _deadline_check(deadline)
        rep.orchestrator.step(
            delta_time=0.0,
            pause_timeline=True,
            rt_subframes=64,
            wait_for_render=True,
        )
        candidate = np.asarray(annotator.get_data())
        if candidate.ndim != 3 or candidate.shape[:2] != (480, 640) or candidate.shape[2] < 3:
            invalid.append(f"shape={candidate.shape}")
            continue
        candidate = np.asarray(candidate[..., :3], dtype=np.uint8)
        if not np.any(candidate != candidate[0, 0]):
            invalid.append("spatially_constant")
            continue
        return candidate, attempt
    raise RuntimeError(
        "RGB capture remained invalid after 20 same-state attempts: "
        + ",".join(invalid)
    )


def _write_rgb_png(cv2: Any, path: Path, rgb: Any) -> tuple[str, bool]:
    import numpy as np

    if not cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write RGB evidence: {path.name}")
    reopened_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    readable = bool(
        reopened_bgr is not None
        and reopened_bgr.shape == rgb.shape
        and np.array_equal(cv2.cvtColor(reopened_bgr, cv2.COLOR_BGR2RGB), rgb)
    )
    if not readable:
        raise RuntimeError(f"RGB evidence did not round-trip exactly: {path.name}")
    return _sha256_file(path), readable


def run_probe(
    args: argparse.Namespace,
    app: Any,
    output: Path,
    *,
    started: float | None = None,
) -> dict[str, Any]:
    """Execute constant parity first, then author and capture spatial color."""

    import numpy as np

    receipt = build_empty_receipt(args)
    began = time.monotonic() if started is None else started
    deadline = began + WALL_LIMIT_SECONDS
    rgb_annotator = None
    render_product = None
    neuraylib = None
    registered_mdl_search_path = None
    try:
        import carb
        import cv2
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        _progress("isaac_startup_complete")
        _deadline_check(deadline)
        observed_version = version("isaacsim")
        receipt["runtime"]["isaac_sim_version"] = observed_version
        suffix = observed_version.removeprefix(EXPECTED_ISAAC_SIM_VERSION + ".")
        if observed_version != EXPECTED_ISAAC_SIM_VERSION and not (
            suffix and suffix.isdigit()
        ):
            raise RuntimeError(
                f"probe requires Isaac Sim {EXPECTED_ISAAC_SIM_VERSION}, observed {observed_version}"
            )

        mdl_hash = _sha256_file(args.mdl)
        receipt["material"]["wrapper_mdl_sha256"] = mdl_hash
        if mdl_hash != EXPECTED_WRAPPER_MDL_SHA256:
            raise RuntimeError("wrapper MDL hash differs from the reviewed asset")

        settings = carb.settings.get_settings()
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is enabled")
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        if (
            settings.get_as_int("/rtx/pathtracing/maxBounces") != 16
            or settings.get_as_int(
                "/rtx/pathtracing/maxSpecularAndTransmissionBounces"
            )
            != 16
        ):
            raise RuntimeError("path-tracing bounce settings did not read back")
        mdl_parent = os.fspath(args.mdl.parent)

        manager = app.get_extension_manager()
        if not manager.is_extension_enabled("omni.mdl.neuraylib"):
            manager.set_extension_enabled_immediate("omni.mdl.neuraylib", True)
        if not manager.is_extension_enabled("omni.mdl.neuraylib"):
            raise RuntimeError("omni.mdl.neuraylib could not be enabled")
        import omni.mdl.neuraylib

        neuraylib = omni.mdl.neuraylib.get_neuraylib()
        registered_mdl_search_path = neuraylib.addMdlSearchPath(mdl_parent)
        receipt["material"][
            "mdl_search_path_registered"
        ] = registered_mdl_search_path
        if not registered_mdl_search_path:
            raise RuntimeError("reviewed MDL parent could not be registered")
        if not manager.is_extension_enabled("omni.mdl.usd_converter"):
            manager.set_extension_enabled_immediate("omni.mdl.usd_converter", True)
        if not manager.is_extension_enabled("omni.mdl.usd_converter"):
            raise RuntimeError("omni.mdl.usd_converter could not be enabled")
        import omni.mdl.usd_converter  # noqa: F401

        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError("could not open canonical fill-station stage")
        while context.get_stage_loading_status()[2] > 0:
            _deadline_check(deadline)
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened stage")
        _progress("twin_stage_loaded")
        source_records = _source_layer_records(stage)
        receipt["scene_contract"]["source_layer_records"] = source_records
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
            raise RuntimeError("opened stage is not the reviewed metre-scale Z-up twin")

        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if not math.isfinite(time_codes_per_second) or time_codes_per_second <= 0.0:
            raise RuntimeError("twin timeCodesPerSecond is invalid")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(TWIN_REFERENCE_TIME_CODE / time_codes_per_second)
        app.update()
        frozen_time = float(timeline.get_current_time()) * time_codes_per_second
        receipt["scene_contract"]["frozen_time_code"] = frozen_time
        if not math.isclose(frozen_time, TWIN_REFERENCE_TIME_CODE, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("twin did not freeze at the reviewed time code")
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError("reviewed FillCam is missing")
        camera = UsdGeom.Camera(camera_prim)
        clipping = camera.GetClippingRangeAttr().Get(Usd.TimeCode(TWIN_REFERENCE_TIME_CODE))
        clipping_values = [float(clipping[0]), float(clipping[1])]
        receipt["scene_contract"]["camera_clipping_range"] = clipping_values
        if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
            for observed, expected in zip(clipping_values, CAMERA_CLIPPING_RANGE)
        ):
            raise RuntimeError("FillCam clipping range differs from the reviewed source")

        fixture = build_static_probe_fixture()
        session = stage.GetSessionLayer()
        with Usd.EditContext(stage, session):
            for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                source_prim = stage.GetPrimAtPath(path)
                if not source_prim.IsValid():
                    raise RuntimeError(f"canonical source prim is missing: {path}")
                UsdGeom.Imageable(source_prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
            UsdGeom.Xform.Define(stage, NAMESPACE)
            UsdGeom.Scope.Define(stage, f"{NAMESPACE}/Materials")
            mesh = UsdGeom.Mesh.Define(stage, MESH_PATH)
            points = [Gf.Vec3f(*[float(value) for value in row]) for row in fixture["mesh_vertices_m"]]
            mesh.CreatePointsAttr().Set(points)
            mesh.CreateFaceVertexCountsAttr().Set(
                [int(value) for value in fixture["face_vertex_counts"]]
            )
            mesh.CreateFaceVertexIndicesAttr().Set(
                [int(value) for value in fixture["face_vertex_indices"]]
            )
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
            points_np = np.asarray(fixture["mesh_vertices_m"], dtype=np.float32)
            mesh.CreateExtentAttr().Set(
                [
                    Gf.Vec3f(*[float(value) for value in points_np.min(axis=0)]),
                    Gf.Vec3f(*[float(value) for value in points_np.max(axis=0)]),
                ]
            )
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            constant_colors = np.repeat(
                np.asarray(
                    [O1B_MATERIAL_PARAMETERS["specular_transmission_color"]],
                    dtype=np.float32,
                ),
                points_np.shape[0],
                axis=0,
            )
            _set_vertex_colors(primvar, constant_colors, Gf)
            original_material = _author_original_material(stage, Sdf, Gf, UsdShade)
            wrapper_material = _author_wrapper_material(
                stage, args.mdl, Sdf, Gf, UsdShade
            )
            original_bound = _bind_material(mesh, original_material, UsdShade)
        receipt["material"]["original_bound"] = original_bound
        if not original_bound:
            raise RuntimeError("original OmniSurface material binding did not resolve")

        non_session_layers = stage.GetLayerStack(includeSessionLayers=False)
        session_only = all(
            layer.GetPrimAtPath(Sdf.Path(NAMESPACE)) is None
            for layer in non_session_layers
        )
        receipt["scene_contract"]["session_layer_only"] = session_only
        if not session_only:
            raise RuntimeError("probe namespace escaped the session layer")
        namespace_prims = list(Usd.PrimRange(stage.GetPrimAtPath(NAMESPACE)))
        receipt["scene_contract"].update(
            {
                "custom_camera_count": sum(
                    prim.IsA(UsdGeom.Camera) for prim in namespace_prims
                ),
                "custom_light_count": sum(
                    "Light" in prim.GetTypeName() for prim in namespace_prims
                ),
                "custom_backdrop_count": sum(
                    prim.IsA(UsdGeom.Mesh)
                    and str(prim.GetPath()) != MESH_PATH
                    for prim in namespace_prims
                ),
            }
        )
        if any(
            receipt["scene_contract"][key] != 0
            for key in (
                "custom_camera_count",
                "custom_light_count",
                "custom_backdrop_count",
            )
        ):
            raise RuntimeError("probe unexpectedly authored a camera, light, or backdrop")

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (640, 480))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(60):
            _deadline_check(deadline)
            app.update()

        resolved, resolution_message = _resolve_material(
            stage, original_material, app
        )
        receipt["material"].update(
            {
                "original_omnisurface_resolved": resolved,
                "original_resolution_message": resolution_message,
            }
        )
        if not resolved:
            raise RuntimeError(
                f"original OmniSurface MDL did not resolve: {resolution_message}"
            )
        original_state = _mesh_state(mesh, primvar)
        original_rgb, original_attempts = _capture_rgb(
            rep, rgb_annotator, deadline=deadline
        )
        _progress("original_constant_captured")
        receipt["render"]["capture_attempt_counts"][
            "original_constant"
        ] = original_attempts

        with Usd.EditContext(stage, session):
            wrapper_bound = _bind_material(mesh, wrapper_material, UsdShade)
        receipt["material"]["wrapper_bound_constant"] = wrapper_bound
        if not wrapper_bound:
            raise RuntimeError("wrapper OmniSurface material binding did not resolve")
        wrapper_resolved, wrapper_message = _resolve_material(
            stage, wrapper_material, app
        )
        receipt["material"].update(
            {
                "wrapper_omnisurface_resolved": wrapper_resolved,
                "wrapper_resolution_message": wrapper_message,
                "same_family_omnisurface_delegate": wrapper_resolved,
                "only_transmission_color_source_differs": wrapper_resolved,
            }
        )
        if not wrapper_resolved:
            raise RuntimeError(
                f"wrapper OmniSurface MDL did not compile/load: {wrapper_message}"
            )
        wrapper_state = _mesh_state(mesh, primvar)
        for field in ("points_sha256", "topology_sha256", "display_color_sha256"):
            if original_state[field] != wrapper_state[field]:
                raise RuntimeError(
                    f"constant parity changed static mesh state: {field}"
                )
        current_time = float(timeline.get_current_time()) * time_codes_per_second
        same_time = math.isclose(
            current_time, TWIN_REFERENCE_TIME_CODE, rel_tol=0.0, abs_tol=1e-9
        )
        if not same_time:
            raise RuntimeError("constant parity captures do not share the frozen time code")

        wrapper_rgb, wrapper_attempts = _capture_rgb(
            rep, rgb_annotator, deadline=deadline
        )
        _progress("wrapper_constant_captured")
        receipt["render"]["capture_attempt_counts"][
            "wrapper_constant"
        ] = wrapper_attempts
        parity_metrics, parity_difference = measure_constant_parity(
            original_rgb, wrapper_rgb
        )
        receipt["constant_parity"].update(
            {
                **parity_metrics,
                "same_mesh_points_sha256": original_state["points_sha256"],
                "same_mesh_topology_sha256": original_state[
                    "topology_sha256"
                ],
                "same_display_color_sha256": original_state[
                    "display_color_sha256"
                ],
                "same_time_code": same_time,
            }
        )

        original_path = output / "original-constant.png"
        wrapper_path = output / "wrapper-constant.png"
        difference_path = output / "constant-parity-absdiff.png"
        original_hash, original_readable = _write_rgb_png(
            cv2, original_path, original_rgb
        )
        wrapper_hash, wrapper_readable = _write_rgb_png(
            cv2, wrapper_path, wrapper_rgb
        )
        difference_hash, difference_readable = _write_rgb_png(
            cv2, difference_path, parity_difference
        )
        receipt["render"].update(
            {
                "original_constant_path": original_path.name,
                "original_constant_sha256": original_hash,
                "wrapper_constant_path": wrapper_path.name,
                "wrapper_constant_sha256": wrapper_hash,
                "parity_absdiff_path": difference_path.name,
                "parity_absdiff_sha256": difference_hash,
            }
        )
        if not parity_metrics["passed"]:
            raise RuntimeError("constant-color OmniSurface parity gate failed")

        mesh_points = np.asarray(
            [
                [float(component) for component in row]
                for row in mesh.GetPointsAttr().Get()
            ],
            dtype=np.float64,
        )
        vertex_colors, nearest_ids, mapping_evidence = (
            build_fixture_spatial_mapping(mesh_points)
        )
        with Usd.EditContext(stage, session):
            _set_vertex_colors(primvar, vertex_colors, Gf)
        readback_raw = primvar.Get()
        readback_values = [
            [float(component) for component in row] for row in readback_raw
        ]
        readback_indices = primvar.GetIndices()
        primvar_evidence = record_vertex_color_readback(
            vertex_colors,
            readback_values,
            interpolation=str(primvar.GetInterpolation()),
            point_count=mesh_points.shape[0],
            indices=readback_indices,
        )
        spatial_bound = _material_is_bound(mesh, wrapper_material, UsdShade)
        receipt["material"]["wrapper_bound_spatial"] = spatial_bound
        if not spatial_bound:
            raise RuntimeError("wrapper binding was lost after spatial primvar authoring")
        receipt["spatial"].update(
            {
                "mapping_evidence": mapping_evidence,
                "nearest_particle_ids": nearest_ids.astype(int).tolist(),
                "primvar_readback": primvar_evidence,
            }
        )
        spatial_rgb, spatial_attempts = _capture_rgb(
            rep, rgb_annotator, deadline=deadline
        )
        _progress("wrapper_spatial_captured")
        receipt["render"]["capture_attempt_counts"][
            "wrapper_spatial"
        ] = spatial_attempts
        spatial_path = output / "wrapper-spatial.png"
        spatial_hash, spatial_readable = _write_rgb_png(
            cv2, spatial_path, spatial_rgb
        )
        spatial_difference = np.abs(
            wrapper_rgb.astype(np.int16) - spatial_rgb.astype(np.int16)
        ).astype(np.uint8)
        receipt["spatial"]["constant_to_spatial_rgb_change"] = {
            "mean_abs_rgb": float(np.mean(spatial_difference)),
            "p95_abs_rgb": float(np.percentile(spatial_difference, 95)),
            "fraction_abs_rgb_gt_16": float(
                np.mean(spatial_difference > 16)
            ),
            "constant_rgb_sha256": wrapper_hash,
            "spatial_rgb_sha256": spatial_hash,
            "interpretation": "pending_visual_review_not_a_physical_or_concentration_gate",
        }
        all_nonconstant = all(
            np.any(image != image[0, 0])
            for image in (original_rgb, wrapper_rgb, spatial_rgb)
        )
        receipt["render"].update(
            {
                "wrapper_spatial_path": spatial_path.name,
                "wrapper_spatial_sha256": spatial_hash,
                "all_images_readable": bool(
                    original_readable
                    and wrapper_readable
                    and difference_readable
                    and spatial_readable
                ),
                "all_images_nonconstant": bool(all_nonconstant),
            }
        )
        if not all_nonconstant:
            raise RuntimeError("one or more kept RGB captures is spatially constant")
        receipt["scene_contract"]["source_layers_unchanged"] = (
            _source_layers_unchanged(source_records)
        )
        if not receipt["scene_contract"]["source_layers_unchanged"]:
            raise RuntimeError("a canonical source layer changed during the probe")
        receipt["visual_review"]["status"] = "pending_top_level_review"
        receipt["status"] = "pending_visual_review"
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(
            f"spatial OmniSurface probe failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if rgb_annotator is not None and render_product is not None:
            try:
                rgb_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"RGB annotator detach failed: {exc}")
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
        receipt["runtime"]["wall_seconds"] = time.monotonic() - began
    if receipt["status"] == "pending_visual_review":
        failures = validate_probe_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    return receipt


def _timeout_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("spatial OmniSurface probe process exceeded 900 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except (FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True)
    receipt = build_empty_receipt(args)
    kit = None
    exit_code = 2
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT_SECONDS)
    try:
        _progress("before_isaac_startup")
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
        import omni.kit.app

        receipt = run_probe(
            args, omni.kit.app.get_app(), args.out, started=started
        )
        exit_code = 0 if receipt["status"] == "pending_visual_review" else 2
    except Exception as exc:
        receipt["runtime"]["wall_seconds"] = time.monotonic() - started
        receipt["errors"].append(
            f"Isaac application startup/probe failed: {type(exc).__name__}: {exc}"
        )
    finally:
        try:
            _write_json(args.out / "receipt.json", receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
            if kit is not None:
                kit.close(exit_code=exit_code)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
