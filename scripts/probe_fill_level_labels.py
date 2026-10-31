"""Probe label routes for the matched FillCam scene.

Labels are authored in the anonymous session layer before render products and
annotators are created. Semantic and instance-id outputs are saved so the
dataset renderer can use the first route that preserves the liquid prim.
Organizer inputs remain read-only.
"""

import argparse
import json
import math
from pathlib import Path
import time


CAMERAS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/FillCam/FillCam",
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
)
LIQUIDS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
    "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
)
ML_PER_MM = math.pi * (26.41 / 2.0) ** 2 / 1000.0


def array_payload(value, height, width):
    import numpy as np

    data = value.get("data", []) if isinstance(value, dict) else value
    info = value.get("info", {}) if isinstance(value, dict) else {}
    array = np.asarray(data)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.shape != (height, width):
        raise RuntimeError(f"annotator shape {array.shape}, expected {(height, width)}")
    return array.copy(), info


def id_counts(array):
    import numpy as np

    identifiers, counts = np.unique(array, return_counts=True)
    return {
        str(int(identifier)): int(count)
        for identifier, count in zip(identifiers, counts)
    }


def mapping_contains(mapping, text):
    folded = str(text).casefold()
    return any(folded in str(value).casefold() for value in mapping.values())


def class_tokens(value):
    if not isinstance(value, dict):
        return set()
    return {
        token.strip().casefold()
        for token in str(value.get("class", "")).split(",")
        if token.strip()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument("--match", type=Path, required=True)
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
    parser.add_argument("--volume-ml", type=float, default=70.0)
    parser.add_argument("--subframes", type=int, default=32)
    parser.add_argument("--warmup-updates", type=int, default=60)
    args = parser.parse_args()

    twin = args.twin.expanduser().resolve()
    stage_path = args.stage if args.stage.is_absolute() else twin / args.stage
    camera_path = (
        args.camera_json
        if args.camera_json.is_absolute()
        else twin / args.camera_json
    )
    match_path = args.match.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()
    portable_root = args.portable_root.expanduser().resolve()
    if not stage_path.is_file() or not camera_path.is_file() or not match_path.is_file():
        parser.error("stage, camera JSON and camera match must exist")
    if output_dir.exists():
        parser.error("--out must be a fresh path")
    if not 0.0 <= args.volume_ml <= 85.0:
        parser.error("--volume-ml must be in [0, 85]")
    output_dir.mkdir(parents=True)
    portable_root.mkdir(parents=True, exist_ok=True)
    camera_json = json.loads(camera_path.read_text())
    match = json.loads(match_path.read_text())
    width, height = camera_json["render_px"]

    from isaacsim import SimulationApp

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
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        from isaacsim.core.utils.semantics import add_labels
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path.resolve())):
            raise RuntimeError(f"could not open stage: {stage_path}")
        deadline = time.monotonic() + 120.0
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise TimeoutError("stage load exceeded 120 seconds")
            app.update()
        stage = context.get_stage()
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError("label probe requires a Z-up stage")
        if not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError("label probe requires stage units in metres")
        stage.SetEditTarget(stage.GetSessionLayer())

        camera_path_candidates = (
            camera_json.get("prim_in_csi_line"),
            camera_json.get("prim_in_station_file"),
            *CAMERAS,
        )
        camera_prim_path = next(
            (
                path for path in camera_path_candidates
                if path and stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        liquid_prim_path = next(
            (path for path in LIQUIDS if stage.GetPrimAtPath(path).IsValid()),
            None,
        )
        if camera_prim_path is None or liquid_prim_path is None:
            raise RuntimeError("FillCam or Fill_liquid was not found")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
        liquid_prim = stage.GetPrimAtPath(liquid_prim_path)
        camera.GetFocalLengthAttr().Set(float(match["overrides"]["focal_length"]))
        camera.GetVerticalApertureOffsetAttr().Set(
            float(match["overrides"]["vertical_aperture_offset"])
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(args.frame / stage.GetTimeCodesPerSecond())
        frame_code = Usd.TimeCode(args.frame)
        app.update()

        mesh = UsdGeom.Mesh(liquid_prim)
        points_attr = mesh.GetPointsAttr()
        points = np.asarray(points_attr.Get(frame_code), dtype=float)
        transform = np.asarray(
            UsdGeom.Xformable(liquid_prim).ComputeLocalToWorldTransform(frame_code),
            dtype=float,
        )
        world = np.c_[points, np.ones(len(points))] @ transform
        bottom_z = float(world[:, 2].min())
        original_height = float(world[:, 2].max() - bottom_z)
        target_height = args.volume_ml / ML_PER_MM / 1000.0
        world[:, 2] = (
            bottom_z
            + (world[:, 2] - bottom_z) * target_height / original_height
        )
        local = world @ np.linalg.inv(transform)
        points_attr.Set(
            [Gf.Vec3f(*[float(v) for v in point[:3]]) for point in local],
            frame_code,
        )

        # Review-required first route: labels precede the render product.
        semantic_prims = []
        vial_prims = []
        for prim in stage.Traverse():
            source_label = prim.GetAttribute("userProperties:synthetic_class")
            if not source_label or not source_label.HasValue():
                continue
            category = str(source_label.Get())
            add_labels(prim, labels=[category], instance_name="class")
            path = str(prim.GetPath())
            semantic_prims.append({"path": path, "class": category})
            if (
                category.casefold() == "vial"
                and not liquid_prim_path.startswith(path + "/")
            ):
                vial_prims.append(UsdGeom.Imageable(prim))
        add_labels(liquid_prim, labels=["liquid"], instance_name="class")
        semantic_prims.append({"path": liquid_prim_path, "class": "liquid"})
        for vial in vial_prims:
            vial.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        app.update()

        render_product = rep.create.render_product(
            camera_prim_path, (width, height)
        )
        semantic = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        instance = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params={"colorize": False}
        )
        semantic.attach([render_product])
        instance.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(args.warmup_updates):
            app.update()
        started = time.monotonic()
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        render_seconds = time.monotonic() - started

        semantic_mask, semantic_info = array_payload(
            semantic.get_data(), height, width
        )
        instance_mask, instance_info = array_payload(
            instance.get_data(), height, width
        )
        semantic_mapping = semantic_info.get("idToLabels", {})
        instance_mapping = instance_info.get("idToLabels", {})
        np.save(output_dir / "semantic.npy", semantic_mask)
        np.save(output_dir / "instance-id.npy", instance_mask)
        cv2.imwrite(
            str(output_dir / "semantic-preview.png"),
            semantic_mask.astype(np.uint16),
        )
        cv2.imwrite(
            str(output_dir / "instance-id-preview.png"),
            instance_mask.astype(np.uint16),
        )

        semantic_has_liquid = any(
            "liquid" in class_tokens(value)
            for value in semantic_mapping.values()
        )
        instance_has_liquid = mapping_contains(
            instance_mapping, liquid_prim_path
        )
        working_route = (
            "semantic_segmentation"
            if semantic_has_liquid
            else "instance_id_segmentation"
            if instance_has_liquid
            else None
        )
        summary = {
            "schema_version": 1,
            "purpose": "probe fill-level label routes",
            "label_order": "labels before render product and annotators",
            "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "stage_meters_per_unit": float(
                UsdGeom.GetStageMetersPerUnit(stage)
            ),
            "frame": args.frame,
            "volume_ml": args.volume_ml,
            "camera_prim": camera_prim_path,
            "liquid_prim": liquid_prim_path,
            "hidden_vial_prims": [
                str(vial.GetPrim().GetPath()) for vial in vial_prims
            ],
            "semantic_prims": semantic_prims,
            "semantic_id_counts": id_counts(semantic_mask),
            "semantic_id_to_labels": semantic_mapping,
            "semantic_has_liquid": semantic_has_liquid,
            "instance_id_counts": id_counts(instance_mask),
            "instance_id_to_labels": instance_mapping,
            "instance_has_liquid_path": instance_has_liquid,
            "render_seconds": render_seconds,
            "working_route": working_route,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(
            "FILL_LEVEL_LABEL_PROBE_COMPLETE "
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        if working_route is None:
            raise RuntimeError(
                "neither ordered-label semantics nor instance-id paths worked"
            )
    finally:
        app.close()


if __name__ == "__main__":
    main()
