"""Inspect FillCam and animated liquid geometry without starting Isaac Sim.

Run with an environment that provides Pixar USD, such as Isaac Sim's Python.
This is a read-only diagnostic for deriving a camera override; it never saves the
opened stage or authors into any of its layers.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


CAMERA = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
LIQUID = "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
ML_PER_MM = np.pi * (26.41 / 2.0) ** 2 / 1000.0


def world_points(prim, frame):
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(frame), dtype=float)
    transform = np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(frame), dtype=float
    )
    return np.c_[points, np.ones(len(points))] @ transform


def projected_pixel(point, camera_matrix, intrinsics):
    camera_point = np.r_[point, 1.0] @ np.linalg.inv(camera_matrix)
    depth = -camera_point[2]
    if depth <= 0:
        raise ValueError(f"point is behind camera: {camera_point.tolist()}")
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    return [
        cx + fx * camera_point[0] / depth,
        cy - fy * camera_point[1] / depth,
    ]


def op_record(op, frame):
    attr = op.GetAttr()
    return {
        "name": op.GetOpName(),
        "type": str(op.GetOpType()),
        "value": str(op.Get(frame)),
        "time_samples": attr.GetTimeSamples(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=686)
    args = parser.parse_args()

    stage_path = args.stage.expanduser().resolve()
    camera_path = args.camera_json.expanduser().resolve()
    if not stage_path.is_file() or not camera_path.is_file():
        parser.error("stage and camera JSON must exist")

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not open {stage_path}")
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if up_axis != UsdGeom.Tokens.z:
        raise RuntimeError(f"expected a Z-up stage, got {up_axis}")
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"expected stage units in metres, got {meters_per_unit:g} metres per unit"
        )
    camera_json = json.loads(camera_path.read_text())
    camera_prim = stage.GetPrimAtPath(CAMERA)
    liquid_prim = stage.GetPrimAtPath(LIQUID)
    if not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"camera not found at {CAMERA}")
    if not liquid_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"liquid mesh not found at {LIQUID}")

    frame = Usd.TimeCode(args.frame)
    camera_matrix = np.asarray(
        UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(frame), dtype=float
    )
    camera = UsdGeom.Camera(camera_prim)
    camera_chain = []
    prim = camera_prim
    while prim and prim.GetPath() != stage.GetPseudoRoot().GetPath():
        if prim.IsA(UsdGeom.Xformable):
            xform = UsdGeom.Xformable(prim)
            camera_chain.append(
                {
                    "path": str(prim.GetPath()),
                    "reset_stack": xform.GetResetXformStack(),
                    "ordered_ops": [op_record(op, frame) for op in xform.GetOrderedXformOps()],
                }
            )
        prim = prim.GetParent()

    heights = []
    for sample_frame in range(
        int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode()) + 1
    ):
        imageable = UsdGeom.Imageable(liquid_prim)
        if imageable.ComputeVisibility(sample_frame) != UsdGeom.Tokens.inherited:
            continue
        points = world_points(liquid_prim, sample_frame)
        low = points[:, :3].min(axis=0)
        high = points[:, :3].max(axis=0)
        height_mm = (high[2] - low[2]) * 1000.0
        heights.append((sample_frame, height_mm * ML_PER_MM, low, high))
    if not heights:
        raise RuntimeError("liquid has no visible animation frames")

    requested_points = {}
    for volume_ml in (30.0, 80.0):
        closest = min(heights, key=lambda record: abs(record[1] - volume_ml))
        bottom = closest[2]
        center_xy = (closest[2][:2] + closest[3][:2]) / 2.0
        top_z = bottom[2] + volume_ml / ML_PER_MM / 1000.0
        point = np.array([center_xy[0], center_xy[1], top_z])
        requested_points[str(int(volume_ml))] = {
            "closest_animation_frame": closest[0],
            "closest_animation_volume_ml": closest[1],
            "world_surface_center": point.tolist(),
            "projected_pixel": projected_pixel(
                point, camera_matrix, camera_json["intrinsics_K_px"]
            ),
        }

    selected_points = world_points(liquid_prim, args.frame)
    selected_low = selected_points[:, :3].min(axis=0)
    selected_high = selected_points[:, :3].max(axis=0)
    semantic_prims = []
    for scene_prim in stage.Traverse():
        label = scene_prim.GetAttribute("userProperties:synthetic_class")
        if label and label.HasValue():
            semantic_prims.append(
                {"path": str(scene_prim.GetPath()), "class": str(label.Get())}
            )
    point_samples = UsdGeom.Mesh(liquid_prim).GetPointsAttr().GetTimeSamples()
    result = {
        "stage": str(stage_path),
        "stage_up_axis": str(up_axis),
        "stage_meters_per_unit": meters_per_unit,
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "start_time_code": stage.GetStartTimeCode(),
        "end_time_code": stage.GetEndTimeCode(),
        "frame": args.frame,
        "camera": {
            "path": CAMERA,
            "world_matrix": camera_matrix.tolist(),
            "focal_length": camera.GetFocalLengthAttr().Get(frame),
            "horizontal_aperture": camera.GetHorizontalApertureAttr().Get(frame),
            "vertical_aperture": camera.GetVerticalApertureAttr().Get(frame),
            "xform_chain_leaf_first": camera_chain,
        },
        "liquid": {
            "path": LIQUID,
            "ml_per_mm": ML_PER_MM,
            "selected_world_bounds": [selected_low.tolist(), selected_high.tolist()],
            "selected_volume_ml": float(
                (selected_high[2] - selected_low[2]) * 1000.0 * ML_PER_MM
            ),
            "requested_surfaces": requested_points,
            "xform_ops": [
                op_record(op, frame)
                for op in UsdGeom.Xformable(liquid_prim).GetOrderedXformOps()
            ],
            "points_time_samples": {
                "count": len(point_samples),
                "first": point_samples[0] if point_samples else None,
                "last": point_samples[-1] if point_samples else None,
            },
        },
        "synthetic_class_prims": semantic_prims,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
