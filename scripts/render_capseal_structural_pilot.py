#!/usr/bin/env python3
"""Render an evidence-bounded, paired CapSealCam structural pilot.

The CPU-only preflight runs before Isaac starts.  The runtime deliberately records
synthetic-only evidence; successful rendering does not establish a real defect signal,
physical cap displacement, or transfer to the production cap/seal camera.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.capseal_structural import (  # noqa: E402
    FINAL_RASTER_SIZE,
    InputSnapshot,
    SourcePose,
    apply_structural_state,
    build_structural_receipt,
    evaluate_structural_gates,
    mask_statistics,
    preflight_output_path,
    publish_directory_no_clobber,
    sha256_file,
    snapshot_input_file,
    validate_camera_record,
    validate_states,
    validate_target,
    verify_input_snapshots,
    warp_ids_nearest,
    warp_rgb_linear,
)


COMPLETE_STAGE = "csi_line.usda"
CAMERA_RECORD = Path("cameras") / "CapSealCam.json"
REQUIRED_TWIN_SCRIPTS = (Path("tools") / "render_example.py", Path("tools") / "apply_lens.py")
ASYNC_RENDERING_DISABLED = "--/exts/isaacsim.core.throttling/enable_async=false"


@dataclass(frozen=True)
class Preflight:
    twin: Path
    stage: Path
    camera_file: Path
    camera: dict[str, float | int]
    output: Path
    target: str
    states: tuple[str, ...]
    subframes: int
    seed: int
    snapshots: tuple[InputSnapshot, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--states", default="intact,missing,unseated")
    parser.add_argument("--subframes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> Preflight:
    """Reject malformed requests and absent immutable inputs before Isaac startup."""

    target = validate_target(args.target)
    states = validate_states(args.states)
    if type(args.subframes) is not int or args.subframes < 1:
        raise ValueError("subframes must be a positive integer")
    if type(args.seed) is not int:
        raise ValueError("seed must be an integer")
    output = preflight_output_path(args.out)
    twin = Path(args.twin).expanduser().resolve(strict=True)
    if not twin.is_dir():
        raise ValueError(f"twin must be a directory: {twin}")
    def required(relative: Path, name: str) -> Path:
        lexical = twin / relative
        if not lexical.is_file() or lexical.is_symlink():
            raise ValueError(f"missing required {name}: {lexical}")
        return lexical.resolve(strict=True)

    stage = required(Path(COMPLETE_STAGE), "complete stage")
    camera_file = required(CAMERA_RECORD, "camera record")
    script_paths = tuple(
        required(relative, name)
        for relative, name in zip(REQUIRED_TWIN_SCRIPTS, ("render script", "lens script"), strict=True)
    )
    own_script = Path(__file__).resolve(strict=True)
    with camera_file.open("r", encoding="utf-8") as stream:
        camera = validate_camera_record(json.load(stream))
    snapshots = (
        snapshot_input_file("complete_stage", stage),
        snapshot_input_file("camera_json", camera_file),
        snapshot_input_file("pilot_script", own_script),
        snapshot_input_file("organizer_render_script", script_paths[0]),
        snapshot_input_file("organizer_lens_script", script_paths[1]),
    )
    return Preflight(twin, stage, camera_file, camera, output, target, states, args.subframes, args.seed, snapshots)


def _matrix_values(matrix: object) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _target_roots(stage: object, target: str) -> tuple[object, object]:
    """Find the selected tray slot's cap and vial roots without falling back to another vial."""

    matches = [prim for prim in stage.Traverse() if target.lower() in prim.GetPath().pathString.lower()]
    if not matches:
        raise RuntimeError(f"no prim path identifies requested target {target}")
    cap_candidates = [prim for prim in matches if "cap" in prim.GetName().lower()]
    vial_candidates = [prim for prim in matches if any(token in prim.GetName().lower() for token in ("vial", "tube"))]
    if len(cap_candidates) != 1 or len(vial_candidates) != 1:
        raise RuntimeError(f"target {target} does not resolve exactly one cap root and one vial root")
    return cap_candidates[0], vial_candidates[0]


def _author_class_label(prim: object, label: str, Usd: object, UsdSemantics: object) -> None:
    api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    if not api or not api.CreateLabelsAttr().Set([label]):
        raise RuntimeError(f"could not author session-only class label {label}")
    labels = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default()).ComputeUniqueInheritedLabels(prim)
    if set(str(item) for item in labels) != {label}:
        raise RuntimeError(f"class label readback failed for {prim.GetPath()}")


def _source_pose(prim: object, UsdGeom: object) -> SourcePose:
    imageable = UsdGeom.Imageable(prim)
    visibility = str(imageable.GetVisibilityAttr().Get())
    common = UsdGeom.XformCommonAPI(prim)
    translation, rotation, _scale, _pivot, _order = common.GetXformVectors()
    return SourcePose(
        visibility=visibility,
        translation=(float(translation[0]), float(translation[1]), float(translation[2])),
        rotation_xyz_degrees=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
    )


def _apply_pose(prim: object, pose: object, UsdGeom: object, Gf: object) -> None:
    imageable = UsdGeom.Imageable(prim)
    if not imageable.GetVisibilityAttr().Set(pose.visibility):
        raise RuntimeError("could not author cap visibility")
    common = UsdGeom.XformCommonAPI(prim)
    if not common.SetTranslate(Gf.Vec3d(*pose.translation)):
        raise RuntimeError("could not author cap translation")
    if not common.SetRotate(Gf.Vec3f(*pose.rotation_xyz_degrees), UsdGeom.XformCommonAPI.RotationOrderXYZ):
        raise RuntimeError("could not author cap rotation")


def _annotator_data(raw: object, name: str) -> tuple[np.ndarray, Mapping[object, object]]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("info"), Mapping) or "idToLabels" not in raw["info"]:
        raise RuntimeError(f"{name} annotator did not provide data and idToLabels")
    data = np.asarray(raw.get("data"))
    mapping = raw["info"]["idToLabels"]
    if not isinstance(mapping, Mapping):
        raise RuntimeError(f"{name} idToLabels is malformed")
    return data, mapping


def _ids_for_label(mapping: Mapping[object, object], label: str) -> tuple[int, ...]:
    found: list[int] = []
    for raw_id, raw_labels in mapping.items():
        try:
            identifier = int(raw_id)
        except (TypeError, ValueError):
            continue
        text = json.dumps(raw_labels, sort_keys=True).lower()
        if label.lower() in text:
            found.append(identifier)
    if not found:
        raise RuntimeError(f"annotator mapping does not contain {label}")
    return tuple(sorted(set(found)))


def _lens_coordinates(camera: Mapping[str, float | int], cv2: object) -> tuple[np.ndarray, np.ndarray]:
    width, height = FINAL_RASTER_SIZE
    focal = float(camera["focal_px_final"])
    distortion = float(camera["lens_k1"])
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]], dtype=np.float64)
    undistorted = cv2.undistortPoints(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:, None, :], intrinsic, np.array([distortion, 0, 0, 0, 0.0])).reshape(height, width, 2)
    return undistorted[..., 0] * focal + float(camera["width"]) / 2, undistorted[..., 1] * focal + float(camera["height"]) / 2


def _write_png(cv2: object, path: Path, value: np.ndarray) -> None:
    if not cv2.imwrite(str(path), value):
        raise OSError(f"could not write {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape[:2] != (480, 640):
        raise OSError(f"could not read back a 640 x 480 PNG: {path}")


def _write_npy(path: Path, value: np.ndarray) -> None:
    np.save(path, value, allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    if loaded.shape != (480, 640) or loaded.dtype != value.dtype or not np.array_equal(loaded, value):
        raise OSError(f"could not read back NPY: {path}")


def _stage_layer_snapshots(stage: object) -> tuple[InputSnapshot, ...]:
    snapshots: list[InputSnapshot] = []
    for ordinal, layer in enumerate(stage.GetLayerStack(includeSessionLayers=False), start=1):
        real_path = Path(str(layer.realPath)).resolve(strict=True) if layer.realPath else None
        if real_path is None or not real_path.is_file():
            raise RuntimeError(f"non-session USD layer {ordinal} is not a regular file")
        snapshots.append(snapshot_input_file(f"usd_layer_{ordinal}", real_path))
    if not snapshots:
        raise RuntimeError("complete stage has no non-session USD layers")
    return tuple(snapshots)


def _capture(pre: Preflight) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object], tuple[InputSnapshot, ...]]:
    """The only Isaac entry path; all imports below require the preflight to have passed."""

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": 768, "height": 576, "extra_args": [ASYNC_RENDERING_DISABLED]})
    try:
        import cv2
        import omni.timeline
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, Usd, UsdGeom, UsdSemantics

        enable_extension("omni.replicator.core")
        context = omni.usd.get_context()
        if not context.open_stage(str(pre.stage)):
            raise RuntimeError(f"could not open complete stage: {pre.stage}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        layer_snapshots = _stage_layer_snapshots(stage)
        input_snapshots = pre.snapshots + layer_snapshots
        verify_input_snapshots(input_snapshots)
        camera_path = "/CSI_Line/TrayInspection/CapSealCam/CapSealCam"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"authored CapSealCam is missing: {camera_path}")
        cap_root, vial_root = _target_roots(stage, pre.target)
        camera_transform_before = _matrix_values(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        camera = UsdGeom.Camera(camera_prim)
        clipping = camera.GetClippingRangeAttr()
        clipping_before = tuple(float(value) for value in clipping.Get())
        source_pose = _source_pose(cap_root, UsdGeom)
        stage.SetEditTarget(stage.GetSessionLayer())
        _author_class_label(cap_root, "target_cap", Usd, UsdSemantics)
        _author_class_label(vial_root, "target_vial", Usd, UsdSemantics)
        # Organizer workaround: toggle then restore CapSealCam's clipping range in session only.
        clipping.Set(Gf.Vec2f(clipping_before[0] * 0.5, clipping_before[1]))
        app.update()
        clipping.Set(Gf.Vec2f(*clipping_before))
        app.update()
        source_x, source_y = _lens_coordinates(pre.camera, cv2)
        rep.set_global_seed(pre.seed)
        product = rep.create.render_product(camera_prim, (768, 576))
        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False, "semanticTypes": ["class"]})
        instance_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
        for annotator in (rgb_ann, semantic_ann, instance_ann):
            annotator.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        artifacts: dict[str, dict[str, np.ndarray]] = {}
        for state in pre.states:
            _apply_pose(cap_root, apply_structural_state(source_pose, state), UsdGeom, Gf)
            app.update()
            rep.orchestrator.step(rt_subframes=pre.subframes, delta_time=0.0, wait_for_render=True)
            raw_rgb = np.asarray(rgb_ann.get_data())
            semantic_raw, semantic_labels = _annotator_data(semantic_ann.get_data(), "semantic")
            instance_raw, instance_labels = _annotator_data(instance_ann.get_data(), "instance")
            if raw_rgb.ndim != 3 or raw_rgb.shape[:2] != (576, 768) or raw_rgb.shape[2] < 3:
                raise RuntimeError(f"RGB capture has wrong shape: {raw_rgb.shape}")
            if semantic_raw.shape != (576, 768) or instance_raw.shape != (576, 768):
                raise RuntimeError("semantic and instance AOVs must match overscan raster")
            rgb = warp_rgb_linear(raw_rgb[..., :3], source_x, source_y)
            semantic = warp_ids_nearest(semantic_raw, source_x, source_y)
            instance = warp_ids_nearest(instance_raw, source_x, source_y)
            cap_semantic = np.isin(semantic, _ids_for_label(semantic_labels, "target_cap"))
            vial_semantic = np.isin(semantic, _ids_for_label(semantic_labels, "target_vial"))
            cap_instance = np.isin(instance, _ids_for_label(instance_labels, str(cap_root.GetPath())))
            artifacts[state] = {"rgb": rgb, "semantic": semantic, "instance": instance, "target_cap_semantic_mask": cap_semantic, "target_cap_instance_mask": cap_instance, "target_vial_mask": vial_semantic}
        _apply_pose(cap_root, apply_structural_state(source_pose, "intact"), UsdGeom, Gf)
        clipping.Set(Gf.Vec2f(*clipping_before))
        app.update()
        camera_transform_after = _matrix_values(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        clipping_after = tuple(float(value) for value in clipping.Get())
        camera_gate = {"passed": camera_transform_before == camera_transform_after and clipping_before == clipping_after, "path": camera_path, "transform_before": list(camera_transform_before), "transform_after": list(camera_transform_after), "clipping_before": list(clipping_before), "clipping_after": list(clipping_after), "resync_restored": clipping_before == clipping_after}
        if not camera_gate["passed"]:
            raise RuntimeError("CapSealCam transform or clipping range changed during pilot")
        return artifacts, camera_gate, input_snapshots
    finally:
        app.close()


def _publish(pre: Preflight, artifacts: Mapping[str, Mapping[str, np.ndarray]], camera_gate: Mapping[str, object], snapshots: tuple[InputSnapshot, ...]) -> Path:
    gates = evaluate_structural_gates(artifacts, require_pass=True)
    gates["camera_path_transform_clipping_unchanged"] = dict(camera_gate)
    input_hashes = {snapshot.role: snapshot.sha256 for snapshot in snapshots}
    receipt = build_structural_receipt(target=pre.target, states=pre.states, input_hashes=input_hashes, gates=gates)

    def populate(partial: Path) -> None:
        for state in pre.states:
            destination = partial / state
            destination.mkdir()
            value = artifacts[state]
            _write_png(__import__("cv2"), destination / "rgb.png", value["rgb"][..., ::-1])
            _write_npy(destination / "semantic.npy", value["semantic"])
            _write_npy(destination / "instance.npy", value["instance"])
            _write_png(__import__("cv2"), destination / "target_cap_mask.png", value["target_cap_semantic_mask"].astype(np.uint8) * 255)
            _write_png(__import__("cv2"), destination / "target_vial_mask.png", value["target_vial_mask"].astype(np.uint8) * 255)
            state_record = {"state": state, "artifacts": {name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in (("rgb", destination / "rgb.png"), ("semantic", destination / "semantic.npy"), ("instance", destination / "instance.npy"), ("target_cap_mask", destination / "target_cap_mask.png"), ("target_vial_mask", destination / "target_vial_mask.png"))}, "counts_and_geometry": {"target_cap": mask_statistics(value["target_cap_semantic_mask"]), "target_vial": mask_statistics(value["target_vial_mask"])}, "gates": gates["states"][state]}
            (destination / "receipt.json").write_text(json.dumps(state_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_input_snapshots(snapshots)
    return publish_directory_no_clobber(pre.output, populate)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pre = preflight(args)
        artifacts, camera_gate, snapshots = _capture(pre)
        output = _publish(pre, artifacts, camera_gate, snapshots)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(f"published synthetic CapSealCam structural pilot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
