"""Validate a FillCam match with exact 30 mL and 80 mL liquid probes.

The organizer station and camera files are read-only. Camera attributes,
synthetic labels, liquid probe heights and temporary visibility changes are
authored only in the anonymous USD session layer. Run with Isaac Sim's Python
under an outer timeout and give every run a fresh output directory.
"""

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback


CAMERA_FALLBACKS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/FillCam/FillCam",
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
)
LIQUID_FALLBACKS = (
    "/CSI_Line/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
    "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid",
)
ML_PER_MM = 3.141592653589793 * (26.41 / 2.0) ** 2 / 1000.0


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


def semantic_ids(labels, category):
    return [
        int(key)
        for key, value in labels.items()
        if str(value.get("class", "")).casefold() == category.casefold()
    ]


def contiguous_runs(values, wanted):
    runs = []
    start = None
    for index, value in enumerate(values):
        if bool(wanted(value)):
            if start is None:
                start = index
        elif start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, len(values) - 1])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument(
        "--stage",
        type=Path,
        default=Path("stations/fill_station/fill_station.usda"),
    )
    parser.add_argument(
        "--camera-json", type=Path, default=Path("cameras/FillCam.json")
    )
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=686)
    parser.add_argument("--subframes", type=int, default=32)
    parser.add_argument("--warmup-updates", type=int, default=60)
    parser.add_argument("--retry-updates", type=int, default=30)
    parser.add_argument("--stage-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--capture-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--cache-state", choices=("warm", "cold", "unknown"), required=True
    )
    args = parser.parse_args()

    twin = args.twin.expanduser().resolve()
    stage_path = args.stage.expanduser()
    if not stage_path.is_absolute():
        stage_path = twin / stage_path
    stage_path = stage_path.resolve()
    camera_path = args.camera_json.expanduser()
    if not camera_path.is_absolute():
        camera_path = twin / camera_path
    camera_path = camera_path.resolve()
    match_path = args.match.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()
    portable_root = args.portable_root.expanduser().resolve()

    if not twin.is_dir() or not stage_path.is_file() or not camera_path.is_file():
        parser.error("--twin, --stage and --camera-json must resolve to existing inputs")
    if not match_path.is_file():
        parser.error("--match must be an existing JSON file")
    if output_dir.exists():
        parser.error("--out must be a fresh path")
    if min(args.frame, args.subframes, args.warmup_updates, args.retry_updates) < 1:
        parser.error("frame, subframes and update counts must be positive")
    if args.stage_timeout_seconds <= 0 or args.capture_timeout_seconds <= 0:
        parser.error("timeouts must be positive")

    camera_json = json.loads(camera_path.read_text())
    match = json.loads(match_path.read_text())
    width, height = camera_json["render_px"]
    if (width, height) != (640, 480):
        parser.error("camera matching currently requires a 640x480 FillCam")
    if camera_json.get("overscan", 1.0) != 1.0:
        parser.error("camera matching requires the unwarped public FillCam")
    expected_camera_hash = match.get("source_camera_json_sha256")
    if expected_camera_hash and file_sha256(camera_path) != expected_camera_hash:
        parser.error("camera JSON hash does not match the camera-match provenance")
    expected_stage_hash = match.get("source_station_usda_sha256")
    if expected_stage_hash and file_sha256(stage_path) != expected_stage_hash:
        parser.error("station USD hash does not match the camera-match provenance")
    targets = match.get("targets", [])
    if {float(item["volume_ml"]) for item in targets} != {30.0, 80.0}:
        parser.error("camera match must define exactly the 30 mL and 80 mL targets")

    output_dir.mkdir(parents=True)
    portable_root.mkdir(parents=True, exist_ok=True)
    source_hashes_before = tree_hashes(stage_path.parent)
    camera_hash_before = file_sha256(camera_path)
    process_started = time.monotonic()
    phase_timings = {}
    render_steps = []
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
                "--portable-root",
                str(portable_root),
                "--/telemetry/enableAnonymousData=false",
                "--/structuredLog/enable=false",
            ],
        }
    )
    phase_timings["simulation_app_start_seconds"] = time.monotonic() - app_started
    print("FILL_LEVEL_MATCH_RUNTIME_READY", flush=True)
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

        stage_load_started = time.monotonic()
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"could not open stage: {stage_path}")
        deadline = time.monotonic() + args.stage_timeout_seconds
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"stage load exceeded {args.stage_timeout_seconds:g} seconds"
                )
            app.update()
        stage = context.get_stage()
        phase_timings["stage_load_seconds"] = time.monotonic() - stage_load_started
        print("FILL_LEVEL_MATCH_STAGE_LOADED", flush=True)
        scene_setup_started = time.monotonic()

        camera_prim_path = next(
            (
                path
                for path in (
                    camera_json.get("prim_in_csi_line"),
                    camera_json.get("prim_in_station_file"),
                    *CAMERA_FALLBACKS,
                )
                if path and stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        liquid_prim_path = next(
            (
                path
                for path in LIQUID_FALLBACKS
                if stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        if camera_prim_path is None or liquid_prim_path is None:
            raise RuntimeError("FillCam or Fill_liquid was not found in the stage")
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        liquid_prim = stage.GetPrimAtPath(liquid_prim_path)
        if not camera_prim.IsA(UsdGeom.Camera) or not liquid_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("camera or liquid prim has the wrong USD schema")

        session_layer = stage.GetSessionLayer()
        stage.SetEditTarget(session_layer)
        camera = UsdGeom.Camera(camera_prim)
        focal_attr = camera.GetFocalLengthAttr()
        aperture_offset_attr = camera.GetVerticalApertureOffsetAttr()
        source_camera_values = {
            "focal_length": float(focal_attr.Get(args.frame)),
            "vertical_aperture_offset": float(
                aperture_offset_attr.Get(args.frame)
            ),
        }
        overrides = match["overrides"]
        focal_attr.Set(float(overrides["focal_length"]))
        aperture_offset_attr.Set(float(overrides["vertical_aperture_offset"]))
        app.update()
        camera_override_path = output_dir / "camera-override.usda"
        if not session_layer.Export(str(camera_override_path)):
            raise OSError(f"could not export {camera_override_path}")

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(args.frame / stage.GetTimeCodesPerSecond())
        app.update()

        render_product = rep.create.render_product(camera_prim_path, (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product])
        semantic = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        semantic.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        app.update()

        semantic_prims = []
        vial_prims = []
        for prim in stage.Traverse():
            source_label = prim.GetAttribute("userProperties:synthetic_class")
            if not source_label or not source_label.HasValue():
                continue
            category = str(source_label.Get())
            add_labels(prim, labels=[category], instance_name="class")
            semantic_prims.append({"path": str(prim.GetPath()), "class": category})
            path_text = str(prim.GetPath())
            if (
                category.casefold() == "vial"
                and not liquid_prim_path.startswith(path_text + "/")
            ):
                vial_prims.append(UsdGeom.Imageable(prim))
        # The public USD carries the string "None" on the liquid parent in
        # this copy, despite its documentation naming a liquid class. Author
        # the probe truth explicitly on the mesh in the session layer.
        add_labels(liquid_prim, labels=["liquid"], instance_name="class")
        semantic_prims.append({"path": liquid_prim_path, "class": "liquid"})

        clipping = camera.GetClippingRangeAttr()
        original_clipping = clipping.Get(args.frame)
        clipping.Set(Gf.Vec2f(original_clipping[0] * 0.5, original_clipping[1]))
        app.update()
        clipping.Set(original_clipping)
        app.update()
        phase_timings["scene_setup_seconds"] = time.monotonic() - scene_setup_started

        warmup_started = time.monotonic()
        for _ in range(args.warmup_updates):
            app.update()
        phase_timings["warmup_seconds"] = time.monotonic() - warmup_started
        print("FILL_LEVEL_MATCH_WARMUP_COMPLETE", flush=True)

        def capture_rgb(volume_ml):
            capture_deadline = time.monotonic() + args.capture_timeout_seconds
            attempt = 0
            while time.monotonic() < capture_deadline:
                attempt += 1
                channel_std = None
                step_started = time.monotonic()
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
                step_seconds = time.monotonic() - step_started
                candidate = np.asarray(rgb.get_data())
                valid_shape = (
                    candidate.ndim == 3
                    and candidate.shape[:2] == (height, width)
                    and candidate.shape[2] >= 3
                )
                if valid_shape:
                    candidate = candidate[:, :, :3].copy()
                    channel_std = candidate.astype(np.float64).std(axis=(0, 1))
                render_steps.append(
                    {
                        "volume_ml": volume_ml,
                        "pass": "rgb",
                        "attempt": attempt,
                        "seconds": step_seconds,
                        "shape": list(candidate.shape),
                        "spatial_std": (
                            channel_std.tolist() if channel_std is not None else None
                        ),
                    }
                )
                if valid_shape and float(channel_std.max()) > 1.0:
                    return candidate, attempt
                print(json.dumps(render_steps[-1], sort_keys=True), flush=True)
                for _ in range(args.retry_updates):
                    app.update()
            raise TimeoutError(
                f"{volume_ml:g} mL RGB stayed empty or blank for "
                f"{args.capture_timeout_seconds:g} seconds"
            )

        def capture_semantic(volume_ml):
            capture_deadline = time.monotonic() + args.capture_timeout_seconds
            attempt = 0
            while time.monotonic() < capture_deadline:
                attempt += 1
                step_started = time.monotonic()
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
                step_seconds = time.monotonic() - step_started
                data = semantic.get_data()
                mask = np.asarray(data.get("data", []))
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                record = {
                    "volume_ml": volume_ml,
                    "pass": "semantic_no_glass",
                    "attempt": attempt,
                    "seconds": step_seconds,
                    "shape": list(mask.shape),
                }
                render_steps.append(record)
                if mask.shape == (height, width):
                    return mask.copy(), data.get("info", {}).get("idToLabels", {}), attempt
                print(json.dumps(record, sort_keys=True), flush=True)
                for _ in range(args.retry_updates):
                    app.update()
            raise TimeoutError(
                f"{volume_ml:g} mL semantic mask stayed empty for "
                f"{args.capture_timeout_seconds:g} seconds"
            )

        def rgb_surface_candidates(pixels, target_row):
            measurement = match["surface_measurement"]
            x_first, x_last = (
                int(value) for value in measurement["columns_half_open"]
            )
            half_window = int(measurement["half_window_rows"])
            radius = int(measurement["rgb_search_radius_px"])
            first = max(half_window, int(round(target_row)) - radius)
            last = min(height - half_window - 1, int(round(target_row)) + radius)
            lab = cv2.cvtColor(pixels[:, :, ::-1], cv2.COLOR_BGR2LAB).astype(
                np.float64
            )
            strip = lab[:, x_first:x_last]
            scores = []
            for row in range(first, last + 1):
                above = np.median(strip[row - half_window : row], axis=(0, 1))
                below = np.median(strip[row : row + half_window], axis=(0, 1))
                scores.append((row, float(np.linalg.norm(below - above))))
            candidates = []
            for row, score in sorted(scores, key=lambda item: item[1], reverse=True):
                if all(abs(row - prior[0]) >= half_window for prior in candidates):
                    candidates.append((row, score))
                if len(candidates) == 5:
                    break
            if not candidates:
                raise RuntimeError("RGB surface search produced no candidates")
            return candidates, [x_first, x_last], [first, last]

        mesh = UsdGeom.Mesh(liquid_prim)
        points_attr = mesh.GetPointsAttr()
        frame_code = Usd.TimeCode(args.frame)
        original_points = np.asarray(points_attr.Get(frame_code), dtype=float)
        mesh_transform = np.asarray(
            UsdGeom.Xformable(liquid_prim).ComputeLocalToWorldTransform(frame_code),
            dtype=float,
        )
        original_world = np.c_[original_points, np.ones(len(original_points))] @ mesh_transform
        bottom_z = float(original_world[:, 2].min())
        original_height = float(original_world[:, 2].max() - bottom_z)
        inverse_mesh_transform = np.linalg.inv(mesh_transform)
        camera_transform = np.asarray(
            UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(frame_code),
            dtype=float,
        )
        inverse_camera_transform = np.linalg.inv(camera_transform)
        focal_length = float(focal_attr.Get(frame_code))
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get(frame_code))
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get(frame_code))
        horizontal_offset = float(camera.GetHorizontalApertureOffsetAttr().Get(frame_code))
        vertical_offset = float(aperture_offset_attr.Get(frame_code))
        fx = focal_length / horizontal_aperture * width
        fy = focal_length / vertical_aperture * height
        cx = width / 2.0 + horizontal_offset / horizontal_aperture * width
        cy = height / 2.0 + vertical_offset / vertical_aperture * height
        records = []

        for target in sorted(targets, key=lambda item: float(item["volume_ml"])):
            volume_ml = float(target["volume_ml"])
            target_height_m = volume_ml / ML_PER_MM / 1000.0
            probe_world = original_world.copy()
            probe_world[:, 2] = bottom_z + (
                (probe_world[:, 2] - bottom_z) * target_height_m / original_height
            )
            probe_local = probe_world @ inverse_mesh_transform
            points_attr.Set(
                [Gf.Vec3f(*[float(value) for value in point[:3]]) for point in probe_local],
                frame_code,
            )
            app.update()

            pixels, rgb_tries = capture_rgb(volume_ml)
            image_path = output_dir / f"fillcam-{int(volume_ml):03d}ml.png"
            if not cv2.imwrite(str(image_path), pixels[:, :, ::-1]):
                raise OSError(f"could not write {image_path}")

            for vial in vial_prims:
                vial.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            app.update()
            mask, labels, semantic_tries = capture_semantic(volume_ml)
            for vial in vial_prims:
                vial.GetVisibilityAttr().Clear()
            app.update()

            liquid_ids = semantic_ids(labels, "liquid")
            carrier_ids = semantic_ids(labels, "carrier")
            liquid_region = np.isin(mask, liquid_ids)
            carrier_region = np.isin(mask, carrier_ids)

            top_world = probe_world[
                np.isclose(probe_world[:, 2], probe_world[:, 2].max(), atol=1e-7)
            ]
            top_camera = top_world @ inverse_camera_transform
            depth = -top_camera[:, 2]
            projected_x = cx + fx * top_camera[:, 0] / depth
            projected_y = cy - fy * top_camera[:, 1] / depth
            measure_first, measure_last = (
                int(value)
                for value in match["surface_measurement"]["columns_half_open"]
            )
            center_projection = (projected_x >= measure_first) & (
                projected_x < measure_last
            )
            if not center_projection.any():
                raise RuntimeError("liquid top does not cross the measurement columns")
            geometry_surface_row = float(projected_y[center_projection].min())
            geometry_error_px = geometry_surface_row - float(target["surface_row"])

            rgb_candidates, rgb_columns, rgb_search_rows = rgb_surface_candidates(
                pixels, float(target["surface_row"])
            )
            rgb_surface_row = float(rgb_candidates[0][0])
            rgb_error_px = rgb_surface_row - float(target["surface_row"])

            semantic_surface_row = None
            semantic_column_tops = []
            liquid_bbox = None
            if liquid_region.any():
                ys, xs = np.nonzero(liquid_region)
                semantic_center_x = int(round(float(np.median(xs))))
                sample_columns = range(
                    max(0, semantic_center_x - 4), min(width, semantic_center_x + 5)
                )
                for column in sample_columns:
                    column_rows = np.flatnonzero(liquid_region[:, column])
                    if column_rows.size:
                        semantic_column_tops.append(int(column_rows.min()))
                if semantic_column_tops:
                    semantic_surface_row = float(np.median(semantic_column_tops))
                liquid_bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ]

            class_mask = np.zeros((height, width), dtype=np.uint8)
            class_mask[carrier_region] = 2
            class_mask[liquid_region] = 1
            class_mask_path = output_dir / f"fillcam-{int(volume_ml):03d}ml.classes.png"
            if not cv2.imwrite(str(class_mask_path), class_mask):
                raise OSError(f"could not write {class_mask_path}")
            np.save(output_dir / f"fillcam-{int(volume_ml):03d}ml.semantic.npy", mask)

            gray = cv2.cvtColor(pixels[:, :, ::-1], cv2.COLOR_BGR2GRAY)
            holder_x = int(match["holder_context"]["column_x"])
            rgb_runs = contiguous_runs(gray[:, holder_x], lambda value: value < 70)
            carrier_runs = contiguous_runs(
                carrier_region[:, holder_x], lambda value: bool(value)
            )
            records.append(
                {
                    "volume_ml": volume_ml,
                    "target_height_mm": target_height_m * 1000.0,
                    "target_surface_row": float(target["surface_row"]),
                    "surface_row": rgb_surface_row,
                    "error_px": rgb_error_px,
                    "measurement_method": "strongest local RGB CIELAB edge",
                    "rgb_surface_row": rgb_surface_row,
                    "rgb_error_px": rgb_error_px,
                    "rgb_surface_candidates": [
                        {"row": row, "score_cielab": score}
                        for row, score in rgb_candidates
                    ],
                    "rgb_measurement_columns_half_open": rgb_columns,
                    "rgb_search_rows_inclusive": rgb_search_rows,
                    "geometry_surface_row": geometry_surface_row,
                    "geometry_error_px": geometry_error_px,
                    "semantic_surface_row": semantic_surface_row,
                    "semantic_center_column_top_rows": semantic_column_tops,
                    "semantic_labels": labels,
                    "semantic_liquid_label_found": bool(
                        liquid_ids and liquid_region.any()
                    ),
                    "tolerance_px": float(target["tolerance_px"]),
                    "liquid_pixels": int(liquid_region.sum()),
                    "liquid_bbox_xyxy": liquid_bbox,
                    "projected_top_x_range": [
                        float(projected_x.min()),
                        float(projected_x.max()),
                    ],
                    "holder_column_x": holder_x,
                    "rgb_dark_runs_at_holder_x": rgb_runs,
                    "carrier_runs_at_holder_x": carrier_runs,
                    "rgb_tries": rgb_tries,
                    "semantic_tries": semantic_tries,
                    "rgb_mean": pixels.astype(np.float64).mean(axis=(0, 1)).tolist(),
                    "rgb_spatial_std": pixels.astype(np.float64)
                    .std(axis=(0, 1))
                    .tolist(),
                    "png_sha256": file_sha256(image_path),
                }
            )
            print("FILL_LEVEL_MATCH_PROBE " + json.dumps(records[-1], sort_keys=True), flush=True)

        source_hashes_after = tree_hashes(stage_path.parent)
        camera_hash_after = file_sha256(camera_path)
        source_unchanged = (
            source_hashes_before == source_hashes_after
            and camera_hash_before == camera_hash_after
        )
        passed = source_unchanged and all(
            abs(record["rgb_error_px"]) <= record["tolerance_px"]
            and abs(record["geometry_error_px"]) <= record["tolerance_px"]
            for record in records
        )
        summary = {
            "schema_version": 1,
            "purpose": "FillCam match against video-5-informed liquid-row targets",
            "evidence_label": "measured synthetic probe; targets informed by real video 5",
            "cache_state_at_start": args.cache_state,
            "twin": str(twin),
            "stage": str(stage_path),
            "camera_json": str(camera_path),
            "camera_prim": camera_prim_path,
            "liquid_prim": liquid_prim_path,
            "frame": args.frame,
            "renderer": "PathTracing",
            "resolution": [width, height],
            "subframes": args.subframes,
            "source_camera_values": source_camera_values,
            "camera_overrides": {
                "focal_length": float(focal_attr.Get(args.frame)),
                "vertical_aperture_offset": float(
                    aperture_offset_attr.Get(args.frame)
                ),
            },
            "camera_intrinsics_px": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
            "camera_match": str(match_path),
            "camera_match_sha256": file_sha256(match_path),
            "camera_override_layer": str(camera_override_path),
            "camera_override_layer_sha256": file_sha256(camera_override_path),
            "renderer_script_sha256": file_sha256(Path(__file__)),
            "source_camera_json_sha256_before": camera_hash_before,
            "source_camera_json_sha256_after": camera_hash_after,
            "source_station_tree_hashes_before": source_hashes_before,
            "source_station_tree_hashes_after": source_hashes_after,
            "source_unchanged": source_unchanged,
            "semantic_prims": semantic_prims,
            "hidden_vial_prims_for_semantic_pass": [
                str(vial.GetPrim().GetPath()) for vial in vial_prims
            ],
            "semantic_liquid_labels_found": all(
                record["semantic_liquid_label_found"] for record in records
            ),
            "phase_timings_seconds": phase_timings,
            "render_steps": render_steps,
            "records": records,
            "seconds": time.monotonic() - process_started,
            "passed": passed,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("FILL_LEVEL_CAMERA_MATCH_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
        faulthandler.cancel_dump_traceback_later()
        if not passed:
            raise RuntimeError("camera match did not meet every target or changed an input")
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == "__main__":
    main()
