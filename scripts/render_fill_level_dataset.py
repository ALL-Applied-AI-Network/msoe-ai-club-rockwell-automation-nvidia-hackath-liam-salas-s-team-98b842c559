"""Render deterministic FillCam RGB and four-class synthetic labels.

The organizer station and camera inputs are read-only. All camera, material,
geometry, label, and visibility changes are authored into the anonymous USD
session layer. Specs are a JSON list produced by the pure-Python variation
module. Use a fresh output directory and an outer timeout for every run.
"""

import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from fluidmix import fill_level_variation as variation_contract
from fluidmix.fill_level_variation import (
    LABEL_IDS,
    compose_class_mask,
    expected_surface_row,
    validate_class_mask,
    validate_spec,
    volume_ml_for_row,
)


CAMERA_FALLBACKS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/FillCam/FillCam",
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
)
LIQUID_FALLBACKS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
    "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
)
ML_PER_MM = math.pi * (26.41 / 2.0) ** 2 / 1000.0
SYNTHETIC_ROOT = "/FillLevelSynthetic"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def class_tokens(value):
    if not isinstance(value, dict):
        return set()
    return {
        token.strip().casefold()
        for token in str(value.get("class", "")).split(",")
        if token.strip()
    }


def semantic_ids(mapping, wanted):
    wanted = {str(value).casefold() for value in wanted}
    return [
        int(identifier)
        for identifier, value in mapping.items()
        if class_tokens(value) & wanted
    ]


def annotator_payload(value, height, width):
    data = value.get("data", []) if isinstance(value, dict) else value
    info = value.get("info", {}) if isinstance(value, dict) else {}
    array = np.asarray(data)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.shape != (height, width):
        raise RuntimeError(
            f"annotator returned {array.shape}, expected {(height, width)}"
        )
    return array.copy(), info.get("idToLabels", {})


def mask_for(mapping, raster, wanted):
    return np.isin(raster, semantic_ids(mapping, wanted))


def contiguous_runs(values):
    runs = []
    start = None
    for index, value in enumerate(values):
        if bool(value):
            if start is None:
                start = index
        elif start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, len(values) - 1])
    return runs


def bbox_xyxy(mask):
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]


def holder_projection(carrier_mask, column_x):
    runs = contiguous_runs(carrier_mask[:, column_x])
    candidates = [
        run for run in runs
        if run[1] >= 180 and run[0] <= 300
    ]
    selected = max(candidates, key=lambda run: run[1] - run[0], default=None)
    result = {
        "column_x": int(column_x),
        "carrier_runs": runs,
        "selected_run": selected,
        "real_hidden_ml": [53.0, 69.0],
        "real_edge_ml": [69.0, 73.0],
    }
    if selected is not None:
        top_ml = volume_ml_for_row(selected[0])
        bottom_ml = volume_ml_for_row(selected[1])
        hidden = sorted([bottom_ml, top_ml])
        result.update(
            {
                "selected_rows_inclusive": selected,
                "twin_hidden_ml": hidden,
                "twin_edge_ml": [hidden[1], hidden[1] + 4.0],
                "boundary_delta_ml_vs_real": [
                    hidden[0] - 53.0,
                    hidden[1] - 69.0,
                ],
                "shift_oversampling_to_twin": any(
                    abs(delta) > 2.0
                    for delta in (hidden[0] - 53.0, hidden[1] - 69.0)
                ),
            }
        )
    return result


def rgb_surface_candidates(pixels, expected_row, columns, half_window=6, radius=14):
    import cv2

    x_first, x_last = (int(value) for value in columns)
    first = max(half_window, int(round(expected_row)) - radius)
    last = min(pixels.shape[0] - half_window - 1, int(round(expected_row)) + radius)
    if first > last:
        return []
    lab = cv2.cvtColor(pixels[:, :, ::-1], cv2.COLOR_BGR2LAB).astype(np.float64)
    strip = lab[:, x_first:x_last]
    scores = []
    for row in range(first, last + 1):
        above = np.median(strip[row - half_window : row], axis=(0, 1))
        below = np.median(strip[row : row + half_window], axis=(0, 1))
        scores.append((row, float(np.linalg.norm(below - above))))
    selected = []
    for row, score in sorted(scores, key=lambda item: item[1], reverse=True):
        if all(abs(row - earlier[0]) >= half_window for earlier in selected):
            selected.append((row, score))
        if len(selected) == 5:
            break
    return selected


def frame_integrity_ok(record):
    """Check only classes that must be visible for this frame's conditions."""
    counts = record["label_stats"]["class_pixels"]
    level_ml = float(record["level_ml"])
    foam = record["foam"]
    return bool(
        counts["ignore"] > 0
        and counts["blocked"] > 0
        and (level_ml > 0.0 or counts["empty"] > 0)
        and (
            not foam["enabled"]
            or not foam.get("visible_in_camera", True)
            or counts["foam"] > 0
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument(
        "--stage", type=Path,
        default=Path("stations/fill_station/fill_station.usda"),
    )
    parser.add_argument(
        "--camera-json", type=Path, default=Path("cameras/FillCam.json")
    )
    parser.add_argument("--frame", type=int, default=686)
    parser.add_argument("--subframes", type=int, default=32)
    parser.add_argument("--warmup-updates", type=int, default=60)
    parser.add_argument("--retry-updates", type=int, default=30)
    parser.add_argument("--capture-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--cache-state", choices=("warm", "cold", "unknown"), required=True
    )
    args = parser.parse_args()

    twin = args.twin.expanduser().resolve()
    stage_path = args.stage if args.stage.is_absolute() else twin / args.stage
    camera_path = (
        args.camera_json
        if args.camera_json.is_absolute()
        else twin / args.camera_json
    )
    match_path = args.match.expanduser().resolve()
    specs_path = args.specs.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()
    portable_root = args.portable_root.expanduser().resolve()
    if not stage_path.is_file() or not camera_path.is_file():
        parser.error("stage and camera JSON must resolve below --twin")
    if not match_path.is_file() or not specs_path.is_file():
        parser.error("camera match and specs JSON must exist")
    if output_dir.exists():
        parser.error("--out must be a fresh path")
    if args.capture_timeout_seconds <= 0:
        parser.error("capture timeout must be positive")
    output_dir.mkdir(parents=True)
    (output_dir / "rgb").mkdir()
    (output_dir / "mask").mkdir()
    portable_root.mkdir(parents=True, exist_ok=True)

    camera_json = json.loads(camera_path.read_text())
    match = json.loads(match_path.read_text())
    specs_payload = json.loads(specs_path.read_text())
    specs = specs_payload.get("frames", []) if isinstance(specs_payload, dict) else specs_payload
    if not isinstance(specs, list) or not specs:
        parser.error("specs JSON must be a non-empty list or an object with frames")
    for spec in specs:
        validate_spec(spec)
    width, height = camera_json["render_px"]
    if (width, height) != (640, 480):
        parser.error("the fill-level renderer requires the public 640x480 FillCam")
    expected_stage_hash = match.get("source_station_usda_sha256")
    if expected_stage_hash and file_sha256(stage_path) != expected_stage_hash:
        parser.error("station USD hash differs from camera-match provenance")
    expected_camera_hash = match.get("source_camera_json_sha256")
    if expected_camera_hash and file_sha256(camera_path) != expected_camera_hash:
        parser.error("camera JSON hash differs from camera-match provenance")

    source_hashes_before = tree_hashes(stage_path.parent)
    camera_hash_before = file_sha256(camera_path)
    process_started = time.monotonic()
    faulthandler.dump_traceback_later(120, repeat=True)

    from isaacsim import SimulationApp

    app_started = time.monotonic()
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": width,
            "height": height,
            "extra_args": [
                "--portable-root", str(portable_root),
                "--/telemetry/enableAnonymousData=false",
                "--/structuredLog/enable=false",
            ],
        }
    )
    app_start_seconds = time.monotonic() - app_started
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        from isaacsim.core.utils.semantics import add_labels
        import cv2
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

        def make_material(path, color, opacity=1.0, roughness=0.35):
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(
                stage, material.GetPath().AppendChild("Surface")
            )
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            return material, shader

        def bind(prim, material):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        stage_load_started = time.monotonic()
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path.resolve())):
            raise RuntimeError(f"could not open stage: {stage_path}")
        deadline = time.monotonic() + 120.0
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise TimeoutError("stage load exceeded 120 seconds")
            app.update()
        stage = context.get_stage()
        stage_load_seconds = time.monotonic() - stage_load_started
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError("fill-level rendering requires a Z-up stage")
        if not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError("fill-level rendering requires stage units in metres")
        session_layer = stage.GetSessionLayer()
        stage.SetEditTarget(session_layer)

        camera_candidates = (
            camera_json.get("prim_in_csi_line"),
            camera_json.get("prim_in_station_file"),
            *CAMERA_FALLBACKS,
        )
        camera_prim_path = next(
            (
                path for path in camera_candidates
                if path and stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        liquid_prim_path = next(
            (
                path for path in LIQUID_FALLBACKS
                if stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        if camera_prim_path is None or liquid_prim_path is None:
            raise RuntimeError("FillCam or Fill_liquid was not found")
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        liquid_prim = stage.GetPrimAtPath(liquid_prim_path)
        camera = UsdGeom.Camera(camera_prim)
        liquid_mesh = UsdGeom.Mesh(liquid_prim)
        if not camera_prim.IsA(UsdGeom.Camera) or not liquid_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("camera or liquid prim has the wrong schema")

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(args.frame / stage.GetTimeCodesPerSecond())
        frame_code = Usd.TimeCode(args.frame)
        app.update()

        focal_attr = camera.GetFocalLengthAttr()
        horizontal_offset_attr = camera.GetHorizontalApertureOffsetAttr()
        vertical_offset_attr = camera.GetVerticalApertureOffsetAttr()
        base_focal = float(match["overrides"]["focal_length"])
        base_horizontal_offset = float(horizontal_offset_attr.Get(frame_code))
        base_vertical_offset = float(match["overrides"]["vertical_aperture_offset"])
        focal_attr.Set(base_focal)
        vertical_offset_attr.Set(base_vertical_offset)

        points_attr = liquid_mesh.GetPointsAttr()
        original_points = np.asarray(points_attr.Get(frame_code), dtype=float)
        liquid_transform = np.asarray(
            UsdGeom.Xformable(liquid_prim).ComputeLocalToWorldTransform(frame_code),
            dtype=float,
        )
        original_world = (
            np.c_[original_points, np.ones(len(original_points))] @ liquid_transform
        )
        inverse_liquid_transform = np.linalg.inv(liquid_transform)
        bottom_z = float(original_world[:, 2].min())
        original_height = float(original_world[:, 2].max() - bottom_z)
        center_xy = original_world[:, :2].mean(axis=0)

        semantic_prims = []
        holder_roots = []
        active_vial_prim = None
        for prim in stage.Traverse():
            source_label = prim.GetAttribute("userProperties:synthetic_class")
            if not source_label or not source_label.HasValue():
                continue
            category = str(source_label.Get())
            add_labels(prim, labels=[category], instance_name="class")
            path = str(prim.GetPath())
            semantic_prims.append({"path": path, "class": category})
            if (
                category.casefold() == "carrier"
                and path.endswith("/V3_carrier_0_clamp_21")
            ):
                holder_roots.append(prim)
            if category.casefold() == "vial" and "active" in path.casefold():
                active_vial_prim = prim
        if active_vial_prim is None:
            raise RuntimeError("active vial prim was not found")
        add_labels(liquid_prim, labels=["liquid"], instance_name="class")
        semantic_prims.append({"path": liquid_prim_path, "class": "liquid"})

        synthetic_root = UsdGeom.Xform.Define(stage, SYNTHETIC_ROOT)
        liquid_material, liquid_shader = make_material(
            SYNTHETIC_ROOT + "/Materials/Liquid", (0.1, 0.35, 0.7), 0.85, 0.25
        )
        foam_material, foam_shader = make_material(
            SYNTHETIC_ROOT + "/Materials/Foam", (0.94, 0.94, 0.94), 0.75, 0.8
        )
        sticker_material, _ = make_material(
            SYNTHETIC_ROOT + "/Materials/Sticker", (0.92, 0.92, 0.9), 1.0, 0.7
        )
        ink_material, _ = make_material(
            SYNTHETIC_ROOT + "/Materials/Ink", (0.01, 0.01, 0.01), 1.0, 0.6
        )
        wet_material, wet_shader = make_material(
            SYNTHETIC_ROOT + "/Materials/Wet", (0.5, 0.7, 0.8), 0.15, 0.15
        )
        probe_material, _ = make_material(
            SYNTHETIC_ROOT + "/Materials/Probe", (0.6, 0.6, 0.6), 1.0, 1.0
        )
        holder_material, holder_shader = make_material(
            SYNTHETIC_ROOT + "/Materials/Holder", (0.12, 0.12, 0.12), 1.0, 0.6
        )
        bind(liquid_prim, liquid_material)
        holder_prims = [
            descendant
            for holder_root in holder_roots
            for descendant in Usd.PrimRange(holder_root)
            if descendant.IsA(UsdGeom.Gprim)
        ]
        if not holder_prims:
            raise RuntimeError("no active-holder clamp geometry was found for variation")
        for holder_prim in holder_prims:
            bind(holder_prim, holder_material)

        foam = UsdGeom.Cylinder.Define(stage, SYNTHETIC_ROOT + "/Foam")
        foam.CreateAxisAttr("Z")
        foam.CreateRadiusAttr(0.0128)
        foam.CreateHeightAttr(0.001)
        foam_translate = foam.AddTranslateOp()
        bind(foam.GetPrim(), foam_material)
        add_labels(foam.GetPrim(), labels=["foam"], instance_name="class")
        foam_bump_root_xform = UsdGeom.Xform.Define(
            stage, SYNTHETIC_ROOT + "/FoamBumps"
        )
        foam_bump_root = foam_bump_root_xform.GetPrim()
        foam_bumps = []
        for index in range(10):
            bump = UsdGeom.Sphere.Define(
                stage, f"{SYNTHETIC_ROOT}/FoamBumps/Bump{index:02d}"
            )
            bump.CreateRadiusAttr(0.001)
            translate = bump.AddTranslateOp()
            bind(bump.GetPrim(), foam_material)
            foam_bumps.append((bump, translate))
        add_labels(foam_bump_root, labels=["foam"], instance_name="class")

        sticker_root = UsdGeom.Xform.Define(stage, SYNTHETIC_ROOT + "/Sticker")
        sticker_translate = sticker_root.AddTranslateOp()
        sticker_rotate = sticker_root.AddRotateXYZOp()
        sticker_plate = UsdGeom.Cube.Define(
            stage, SYNTHETIC_ROOT + "/Sticker/Plate"
        )
        sticker_plate.CreateSizeAttr(1.0)
        sticker_plate.AddScaleOp().Set(Gf.Vec3f(0.020, 0.0007, 0.026))
        bind(sticker_plate.GetPrim(), sticker_material)
        for stripe_index, stripe_x in enumerate(np.linspace(-0.0075, 0.0075, 7)):
            stripe = UsdGeom.Cube.Define(
                stage, f"{SYNTHETIC_ROOT}/Sticker/Stripe{stripe_index:02d}"
            )
            stripe.CreateSizeAttr(1.0)
            stripe.AddTranslateOp().Set(Gf.Vec3d(float(stripe_x), -0.0008, 0.0))
            stripe.AddScaleOp().Set(
                Gf.Vec3f(0.00065 if stripe_index % 2 else 0.0011, 0.00025, 0.020)
            )
            bind(stripe.GetPrim(), ink_material)
        add_labels(sticker_root.GetPrim(), labels=["sticker"], instance_name="class")

        wet_root_xform = UsdGeom.Xform.Define(stage, SYNTHETIC_ROOT + "/Wet")
        wet_root = wet_root_xform.GetPrim()
        wet_droplets = []
        for index in range(12):
            droplet = UsdGeom.Sphere.Define(
                stage, f"{SYNTHETIC_ROOT}/Wet/Droplet{index:02d}"
            )
            droplet.CreateRadiusAttr(0.001)
            translate = droplet.AddTranslateOp()
            bind(droplet.GetPrim(), wet_material)
            wet_droplets.append((droplet, translate))
        add_labels(wet_root, labels=["wet_wall"], instance_name="class")

        probe_height = 85.0 / ML_PER_MM / 1000.0 + 0.012
        probe = UsdGeom.Cylinder.Define(stage, SYNTHETIC_ROOT + "/InteriorProbe")
        probe.CreateAxisAttr("Z")
        probe.CreateRadiusAttr(0.0125)
        probe.CreateHeightAttr(probe_height)
        probe.AddTranslateOp().Set(
            Gf.Vec3d(
                float(center_xy[0]), float(center_xy[1]),
                bottom_z + probe_height / 2.0,
            )
        )
        bind(probe.GetPrim(), probe_material)
        add_labels(
            probe.GetPrim(), labels=["interior_probe"], instance_name="class"
        )

        variation_light = UsdLux.DistantLight.Define(
            stage, SYNTHETIC_ROOT + "/VariationLight"
        )
        variation_light.CreateIntensityAttr(350.0)
        variation_light.CreateAngleAttr(1.0)
        variation_light.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 15.0, 25.0))

        # All labels, including generated classes, now exist before the product.
        render_product = rep.create.render_product(camera_prim_path, (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        rgb.attach([render_product])
        semantic.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        app.update()

        local_root = camera_prim.GetParent().GetParent()
        active_vial_path = str(active_vial_prim.GetPath())
        active_vial_top = next(
            (
                child for child in local_root.GetChildren()
                if active_vial_path == str(child.GetPath())
                or active_vial_path.startswith(str(child.GetPath()) + "/")
            ),
            None,
        )
        if active_vial_top is None:
            raise RuntimeError("could not identify the active vial top-level prim")
        camera_top = next(
            (
                child for child in local_root.GetChildren()
                if camera_prim_path.startswith(str(child.GetPath()) + "/")
            ),
            None,
        )
        station_imageables = [
            UsdGeom.Imageable(child)
            for child in local_root.GetChildren()
            if child.IsA(UsdGeom.Imageable)
        ]
        original_visibility = {}
        for imageable in station_imageables:
            value = imageable.GetVisibilityAttr().Get(frame_code)
            original_visibility[str(imageable.GetPrim().GetPath())] = (
                value if value is not None else UsdGeom.Tokens.inherited
            )

        def restore_station():
            for imageable in station_imageables:
                path = str(imageable.GetPrim().GetPath())
                imageable.GetVisibilityAttr().Set(original_visibility[path])
            for name in ("Fill_dispensing_stream", "Fill_barcode_label"):
                prim = stage.GetPrimAtPath(str(local_root.GetPath()) + "/" + name)
                if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                        UsdGeom.Tokens.invisible
                    )

        liquid_top = liquid_prim.GetParent()

        def set_generated_visibility(
            foam_visible=False, sticker_visible=False,
            wet_visible=False, probe_visible=False,
        ):
            foam.GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if foam_visible else UsdGeom.Tokens.invisible
            )
            UsdGeom.Imageable(foam_bump_root).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited
                if foam_visible else UsdGeom.Tokens.invisible
            )
            sticker_root.GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited
                if sticker_visible else UsdGeom.Tokens.invisible
            )
            UsdGeom.Imageable(wet_root).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if wet_visible else UsdGeom.Tokens.invisible
            )
            probe.GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if probe_visible else UsdGeom.Tokens.invisible
            )

        render_steps = []

        def step_and_get(annotator, pass_name, frame_index):
            deadline = time.monotonic() + args.capture_timeout_seconds
            attempt = 0
            while time.monotonic() < deadline:
                attempt += 1
                started = time.monotonic()
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
                seconds = time.monotonic() - started
                value = annotator.get_data()
                if annotator is rgb:
                    array = np.asarray(value)
                    valid = (
                        array.ndim == 3
                        and array.shape[:2] == (height, width)
                        and array.shape[2] >= 3
                    )
                    if valid:
                        array = array[:, :, :3].copy()
                        spatial_std = array.astype(np.float64).std(axis=(0, 1))
                        valid = float(spatial_std.max()) > 1.0
                    else:
                        spatial_std = None
                    mapping = {}
                else:
                    try:
                        array, mapping = annotator_payload(value, height, width)
                        valid = True
                    except RuntimeError:
                        array = np.asarray([])
                        mapping = {}
                        valid = False
                    spatial_std = None
                render_steps.append(
                    {
                        "frame_index": frame_index,
                        "pass": pass_name,
                        "attempt": attempt,
                        "seconds": seconds,
                        "shape": list(array.shape),
                        "spatial_std": (
                            spatial_std.tolist()
                            if spatial_std is not None else None
                        ),
                    }
                )
                if valid:
                    return array, mapping, seconds
                for _ in range(args.retry_updates):
                    app.update()
            raise TimeoutError(f"{pass_name} did not return a valid frame")

        warmup_started = time.monotonic()
        for _ in range(args.warmup_updates):
            app.update()
        warmup_seconds = time.monotonic() - warmup_started

        records = []
        holder_reports = []
        manifest_path = output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
            for frame_index, spec in enumerate(specs):
                level_ml = float(spec["level_ml"])
                level_height = level_ml / ML_PER_MM / 1000.0
                target_world = original_world.copy()
                target_world[:, 2] = bottom_z + (
                    (target_world[:, 2] - bottom_z)
                    * level_height / original_height
                )
                local_points = target_world @ inverse_liquid_transform
                points_attr.Set(
                    [
                        Gf.Vec3f(*[float(value) for value in point[:3]])
                        for point in local_points
                    ],
                    frame_code,
                )
                liquid_visible = level_ml > 0.0

                jitter = spec["camera_jitter"]
                horizontal_aperture = float(
                    camera.GetHorizontalApertureAttr().Get(frame_code)
                )
                vertical_aperture = float(
                    camera.GetVerticalApertureAttr().Get(frame_code)
                )
                focal_attr.Set(base_focal * float(jitter["zoom_scale"]))
                horizontal_offset_attr.Set(
                    base_horizontal_offset
                    + float(jitter["x_px"]) / width * horizontal_aperture
                )
                vertical_offset_attr.Set(
                    base_vertical_offset
                    + float(jitter["y_px"]) / height * vertical_aperture
                )

                liquid_shader.GetInput("diffuseColor").Set(
                    Gf.Vec3f(*spec["liquid"]["rgb"])
                )
                liquid_shader.GetInput("opacity").Set(
                    float(spec["liquid"]["opacity"])
                )
                liquid_shader.GetInput("roughness").Set(
                    float(spec["liquid"]["roughness"])
                )
                foam_shader.GetInput("diffuseColor").Set(
                    Gf.Vec3f(*spec["foam"]["tint_rgb"])
                )
                foam_shader.GetInput("opacity").Set(
                    float(spec["foam"]["opacity"])
                )
                foam_shader.GetInput("roughness").Set(
                    0.55 + 0.4 * float(spec["foam"]["bumpiness"])
                )
                foam_height = float(spec["foam"]["thickness_mm"]) / 1000.0
                foam.CreateHeightAttr().Set(max(foam_height, 0.0001))
                foam.CreateRadiusAttr().Set(
                    0.0128 * (0.98 + 0.04 * float(spec["foam"]["bumpiness"]))
                )
                foam_translate.Set(
                    Gf.Vec3d(
                        float(center_xy[0]), float(center_xy[1]),
                        bottom_z + level_height + max(foam_height, 0.0001) / 2.0,
                    )
                )
                foam_rng = np.random.default_rng(
                    np.random.SeedSequence([int(spec["seed"]), 0xF04A])
                )
                bump_count = int(
                    round(float(spec["foam"]["bumpiness"]) * len(foam_bumps))
                )
                for index, (bump, translate) in enumerate(foam_bumps):
                    radius = float(foam_rng.uniform(0.00045, 0.0018))
                    radial = float(foam_rng.uniform(0.0, 0.0105))
                    angle_bump = float(foam_rng.uniform(-math.pi, math.pi))
                    bump.CreateRadiusAttr().Set(radius)
                    translate.Set(
                        Gf.Vec3d(
                            float(center_xy[0] + radial * math.cos(angle_bump)),
                            float(center_xy[1] + radial * math.sin(angle_bump)),
                            bottom_z + level_height + foam_height + 0.35 * radius,
                        )
                    )
                    bump.GetVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited
                        if spec["foam"]["enabled"] and index < bump_count
                        else UsdGeom.Tokens.invisible
                    )

                pose = spec["sticker_pose"]
                angle = math.radians(float(pose["azimuth_deg"]))
                sticker_translate.Set(
                    Gf.Vec3d(
                        float(center_xy[0] + 0.0137 * math.cos(angle)),
                        float(center_xy[1] + 0.0137 * math.sin(angle)),
                        bottom_z + 0.090 + float(pose["height_offset_mm"]) / 1000.0,
                    )
                )
                sticker_rotate.Set(
                    Gf.Vec3f(
                        float(pose["roll_deg"]), 0.0,
                        float(pose["azimuth_deg"]) - 90.0,
                    )
                )

                wet = spec["wet_walls"]
                wet_shader.GetInput("opacity").Set(float(wet["opacity"]))
                wet_rng = np.random.default_rng(int(wet["pattern_seed"]))
                wet_count = int(round(float(wet["coverage"]) * len(wet_droplets)))
                for index, (droplet, translate) in enumerate(wet_droplets):
                    droplet.GetVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited
                        if wet["enabled"] and index < wet_count
                        else UsdGeom.Tokens.invisible
                    )
                    wet_angle = float(wet_rng.uniform(-math.pi, math.pi))
                    wet_z = bottom_z + min(
                        0.16,
                        level_height + float(wet_rng.uniform(0.004, 0.045)),
                    )
                    translate.Set(
                        Gf.Vec3d(
                            float(center_xy[0] + 0.0125 * math.cos(wet_angle)),
                            float(center_xy[1] + 0.0125 * math.sin(wet_angle)),
                            wet_z,
                        )
                    )
                    droplet.CreateRadiusAttr().Set(
                        float(wet_rng.uniform(0.00035, 0.0014))
                    )

                holder = spec["holder_material"]
                holder_brightness = float(holder["brightness"])
                holder_shader.GetInput("diffuseColor").Set(
                    Gf.Vec3f(
                        0.78 * holder_brightness,
                        holder_brightness,
                        holder_brightness,
                    )
                )
                holder_shader.GetInput("roughness").Set(
                    float(holder["roughness"])
                )

                lighting = spec["lighting"]
                temperature = float(lighting["color_temperature_scale"])
                warm = max(0.0, 1.0 - 0.35 * max(temperature - 1.0, 0.0))
                cool = max(0.0, 1.0 - 0.35 * max(1.0 - temperature, 0.0))
                variation_light.GetIntensityAttr().Set(
                    350.0 * float(lighting["brightness_scale"])
                )
                variation_light.GetColorAttr().Set(Gf.Vec3f(warm, 1.0, cool))

                restore_station()
                UsdGeom.Imageable(liquid_top).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                    if liquid_visible else UsdGeom.Tokens.invisible
                )
                set_generated_visibility(
                    foam_visible=bool(spec["foam"]["enabled"] and liquid_visible),
                    sticker_visible=bool(pose["present"]),
                    wet_visible=bool(wet["enabled"]),
                    probe_visible=False,
                )
                app.update()
                rgb_pixels, _, rgb_seconds = step_and_get(
                    rgb, "rgb", frame_index
                )

                for imageable in station_imageables:
                    top_path = str(imageable.GetPrim().GetPath())
                    keep = top_path in {
                        str(active_vial_top.GetPath()),
                        str(camera_top.GetPath()) if camera_top else "",
                    }
                    imageable.GetVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited
                        if keep else UsdGeom.Tokens.invisible
                    )
                set_generated_visibility()
                app.update()
                vial_raster, vial_mapping, vial_seconds = step_and_get(
                    semantic, "glass_only", frame_index
                )
                vial_mask = mask_for(vial_mapping, vial_raster, {"vial"})

                restore_station()
                UsdGeom.Imageable(active_vial_prim).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
                UsdGeom.Imageable(liquid_top).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                    if liquid_visible else UsdGeom.Tokens.invisible
                )
                set_generated_visibility(
                    foam_visible=bool(spec["foam"]["enabled"] and liquid_visible),
                    sticker_visible=bool(pose["present"]),
                    wet_visible=bool(wet["enabled"]),
                    probe_visible=False,
                )
                app.update()
                fluid_raster, fluid_mapping, fluid_seconds = step_and_get(
                    semantic, "glass_hidden_fluid", frame_index
                )
                liquid_mask = (
                    mask_for(fluid_mapping, fluid_raster, {"liquid"}) & vial_mask
                )
                foam_mask = (
                    mask_for(fluid_mapping, fluid_raster, {"foam"}) & vial_mask
                )
                foam_mask &= ~liquid_mask

                restore_station()
                UsdGeom.Imageable(active_vial_prim).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
                UsdGeom.Imageable(liquid_top).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
                set_generated_visibility(
                    sticker_visible=bool(pose["present"]), probe_visible=True
                )
                app.update()
                blocker_raster, blocker_mapping, blocker_seconds = step_and_get(
                    semantic, "interior_probe_blockers", frame_index
                )
                carrier_mask = mask_for(
                    blocker_mapping, blocker_raster, {"carrier"}
                )
                generated_sticker_mask = mask_for(
                    blocker_mapping, blocker_raster, {"sticker"}
                )
                sticker_mask = generated_sticker_mask | mask_for(
                    blocker_mapping, blocker_raster, {"label"}
                )
                blocker_mask = (
                    carrier_mask
                    | sticker_mask
                    | mask_for(blocker_mapping, blocker_raster, {"nozzle"})
                ) & vial_mask
                class_mask = compose_class_mask(
                    vial_mask, liquid_mask, foam_mask, blocker_mask
                )
                validate_class_mask(class_mask)

                restore_station()
                UsdGeom.Imageable(liquid_top).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                    if liquid_visible else UsdGeom.Tokens.invisible
                )
                set_generated_visibility(
                    foam_visible=bool(spec["foam"]["enabled"] and liquid_visible),
                    sticker_visible=bool(pose["present"]),
                    wet_visible=bool(wet["enabled"]),
                    probe_visible=False,
                )
                app.update()

                stem = f"{frame_index:05d}-{spec['role']}-{int(spec['seed']):07d}"
                rgb_path = output_dir / "rgb" / f"{stem}.png"
                mask_path = output_dir / "mask" / f"{stem}.png"
                if not cv2.imwrite(str(rgb_path), rgb_pixels[:, :, ::-1]):
                    raise OSError(f"could not write {rgb_path}")
                if not cv2.imwrite(str(mask_path), class_mask):
                    raise OSError(f"could not write {mask_path}")

                camera_transform = np.asarray(
                    UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
                        frame_code
                    ),
                    dtype=float,
                )
                inverse_camera = np.linalg.inv(camera_transform)
                top_world = target_world[
                    np.isclose(
                        target_world[:, 2], target_world[:, 2].max(), atol=1e-7
                    )
                ]
                top_camera = top_world @ inverse_camera
                depth = -top_camera[:, 2]
                focal_length = float(focal_attr.Get(frame_code))
                fx = focal_length / horizontal_aperture * width
                fy = focal_length / vertical_aperture * height
                cx = (
                    width / 2.0
                    + float(horizontal_offset_attr.Get(frame_code))
                    / horizontal_aperture * width
                )
                cy = (
                    height / 2.0
                    + float(vertical_offset_attr.Get(frame_code))
                    / vertical_aperture * height
                )
                projected_x = cx + fx * top_camera[:, 0] / depth
                projected_y = cy - fy * top_camera[:, 1] / depth
                columns = match["surface_measurement"]["columns_half_open"]
                center_projection = (
                    (projected_x >= int(columns[0]))
                    & (projected_x < int(columns[1]))
                )
                geometry_row = (
                    float(projected_y[center_projection].min())
                    if center_projection.any() else None
                )
                candidates = rgb_surface_candidates(
                    rgb_pixels, expected_surface_row(level_ml), columns
                )
                holder = holder_projection(
                    carrier_mask, int(match["holder_context"]["column_x"])
                )
                holder_reports.append(holder)
                class_counts = {
                    name: int(np.count_nonzero(class_mask == identifier))
                    for name, identifier in LABEL_IDS.items()
                }
                generated_sticker_pixels = int(
                    (generated_sticker_mask & vial_mask).sum()
                )
                foam_pixels = int(foam_mask.sum())
                record = {
                    **spec,
                    "foam": {
                        **spec["foam"],
                        "visible_in_camera": bool(foam_pixels),
                    },
                    "sticker_pose": {
                        **spec["sticker_pose"],
                        "visible_in_camera": bool(generated_sticker_pixels),
                    },
                    "frame_index": frame_index,
                    "label_method": "semantic_segmentation_labels_before_product",
                    "outputs": {
                        "rgb": {
                            "path": str(rgb_path.relative_to(output_dir)).replace("\\", "/"),
                            "sha256": file_sha256(rgb_path),
                        },
                        "mask": {
                            "path": str(mask_path.relative_to(output_dir)).replace("\\", "/"),
                            "sha256": file_sha256(mask_path),
                        },
                    },
                    "label_stats": {
                        "class_pixels": class_counts,
                        "vial_pixels": int(vial_mask.sum()),
                        "sticker_blocked_pixels": generated_sticker_pixels,
                        "vial_bbox_xyxy": bbox_xyxy(vial_mask),
                        "liquid_bbox_xyxy": bbox_xyxy(liquid_mask),
                        "foam_bbox_xyxy": bbox_xyxy(foam_mask),
                        "blocked_bbox_xyxy": bbox_xyxy(blocker_mask),
                    },
                    "surface": {
                        "expected_row_from_frozen_scale": expected_surface_row(level_ml),
                        "geometry_row": geometry_row,
                        "rgb_row": float(candidates[0][0]) if candidates else None,
                        "rgb_candidates": [
                            {"row": row, "score_cielab": score}
                            for row, score in candidates
                        ],
                    },
                    "holder_projection": holder,
                    "timings_seconds": {
                        "rgb": rgb_seconds,
                        "glass_only": vial_seconds,
                        "glass_hidden_fluid": fluid_seconds,
                        "interior_probe_blockers": blocker_seconds,
                        "all_frame_passes": (
                            rgb_seconds + vial_seconds
                            + fluid_seconds + blocker_seconds
                        ),
                    },
                }
                manifest.write(json.dumps(record, sort_keys=True) + "\n")
                manifest.flush()
                records.append(record)
                print(
                    "FILL_LEVEL_DATASET_FRAME "
                    + json.dumps(record, sort_keys=True),
                    flush=True,
                )

        session_path = output_dir / "session-layer.usda"
        if not session_layer.Export(str(session_path)):
            raise OSError(f"could not export {session_path}")
        source_hashes_after = tree_hashes(stage_path.parent)
        camera_hash_after = file_sha256(camera_path)
        source_unchanged = (
            source_hashes_before == source_hashes_after
            and camera_hash_before == camera_hash_after
        )
        pass_names = (
            "rgb", "glass_only", "glass_hidden_fluid",
            "interior_probe_blockers",
        )
        timing_summary = {}
        for pass_name in pass_names:
            values = [
                record["timings_seconds"][pass_name] for record in records
            ]
            timing_summary[pass_name] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
        totals = [
            record["timings_seconds"]["all_frame_passes"] for record in records
        ]
        integrity = [frame_integrity_ok(record) for record in records]
        summary = {
            "schema_version": 1,
            "purpose": "FillCam synthetic variation and four-class label capture",
            "evidence_label": "measured synthetic render",
            "cache_state_at_start": args.cache_state,
            "renderer": "PathTracing",
            "subframes": args.subframes,
            "resolution": [width, height],
            "frame": args.frame,
            "frame_count": len(records),
            "label_method": "semantic_segmentation_labels_before_product",
            "label_ids": LABEL_IDS,
            "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "stage_meters_per_unit": float(
                UsdGeom.GetStageMetersPerUnit(stage)
            ),
            "camera_prim": camera_prim_path,
            "liquid_prim": liquid_prim_path,
            "active_vial_prim": active_vial_path,
            "semantic_prims": semantic_prims,
            "holder_material_prim_paths": [
                str(prim.GetPath()) for prim in holder_prims
            ],
            "source_station_tree_hashes_before": source_hashes_before,
            "source_station_tree_hashes_after": source_hashes_after,
            "source_camera_json_sha256_before": camera_hash_before,
            "source_camera_json_sha256_after": camera_hash_after,
            "source_unchanged": source_unchanged,
            "camera_match_sha256": file_sha256(match_path),
            "specs_sha256": file_sha256(specs_path),
            "renderer_script_sha256": file_sha256(Path(__file__)),
            "variation_module_sha256": file_sha256(
                Path(variation_contract.__file__)
            ),
            "manifest_sha256": file_sha256(manifest_path),
            "session_layer_sha256": file_sha256(session_path),
            "phase_timings_seconds": {
                "simulation_app_start": app_start_seconds,
                "stage_load": stage_load_seconds,
                "warmup": warmup_seconds,
            },
            "per_pass_timing_seconds": timing_summary,
            "total_frame_passes_seconds": {
                "median": float(np.median(totals)),
                "p95": float(np.percentile(totals, 95)),
                "estimated_frames_per_hour_from_median": (
                    3600.0 / float(np.median(totals))
                ),
            },
            "holder_projection": holder_reports,
            "render_steps": render_steps,
            "integrity_passed": all(integrity),
            "seconds": time.monotonic() - process_started,
            "passed": bool(source_unchanged and all(integrity)),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(
            "FILL_LEVEL_DATASET_COMPLETE "
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        faulthandler.cancel_dump_traceback_later()
        if not summary["passed"]:
            raise RuntimeError("dataset capture failed integrity or changed an input")
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == "__main__":
    main()
