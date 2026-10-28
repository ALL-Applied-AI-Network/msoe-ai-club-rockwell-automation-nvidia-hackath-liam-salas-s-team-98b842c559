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
import traceback
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.capseal_structural import (  # noqa: E402
    FINAL_RASTER_SIZE,
    GateError,
    InputSnapshot,
    SourcePose,
    STATES,
    StateValidationError,
    apply_structural_state,
    build_structural_receipt,
    evaluate_structural_gates,
    mask_statistics,
    preflight_output_path,
    publish_directory_no_clobber,
    sha256_file,
    snapshot_input_file,
    target_instance_mask,
    target_semantic_mask,
    validate_cap_pose_layout,
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
STRUCTURAL_MODULE = ROOT / "src" / "fluidmix" / "capseal_structural.py"
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


@dataclass(frozen=True)
class TargetPrims:
    cap_root: object
    cap_leaf: object
    tube_root: object
    tube_leaf: object
    cap_root_path: str
    cap_leaf_path: str
    tube_root_path: str
    tube_leaf_path: str

    def provenance(self) -> dict[str, str]:
        return {
            "cap_root": self.cap_root_path,
            "cap_leaf": self.cap_leaf_path,
            "tube_root": self.tube_root_path,
            "tube_leaf": self.tube_leaf_path,
        }


@dataclass(frozen=True)
class RootPose:
    source: SourcePose
    scale: tuple[float, float, float]
    op_order: tuple[str, str, str]
    local_transform: tuple[float, ...]
    translate_op: object
    rotate_op: object
    scale_op: object


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
    if set(states) != set(STATES):
        raise StateValidationError("states must include all three required structural states")
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
    structural_module = STRUCTURAL_MODULE.resolve(strict=True)
    with camera_file.open("r", encoding="utf-8") as stream:
        camera = validate_camera_record(json.load(stream))
    snapshots = (
        snapshot_input_file("complete_stage", stage),
        snapshot_input_file("camera_json", camera_file),
        snapshot_input_file("pilot_script", own_script),
        snapshot_input_file("structural_module", structural_module),
        snapshot_input_file("organizer_render_script", script_paths[0]),
        snapshot_input_file("organizer_lens_script", script_paths[1]),
    )
    return Preflight(twin, stage, camera_file, camera, output, target, states, args.subframes, args.seed, snapshots)


def _matrix_values(matrix: object) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _target_prims(stage: object, target: str) -> TargetPrims:
    """Resolve only the reviewed complete-line target root and leaf prim paths."""

    prefix = f"/CSI_Line/TrayInspection/{target}"
    paths = {
        "cap_root": prefix + "_cap",
        "cap_leaf": prefix + "_cap/vial_cap",
        "tube_root": prefix + "_tube",
        "tube_leaf": prefix + "_tube/vial_tube",
    }
    prims: dict[str, object] = {}
    for role, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"target {target} {role} is missing: {path}")
        expected_type = "Xform" if role.endswith("root") else "Mesh"
        if prim.GetTypeName() != expected_type:
            raise RuntimeError(
                f"target {target} {role} has type {prim.GetTypeName()!r}, expected {expected_type}"
            )
        prims[role] = prim
    return TargetPrims(
        cap_root=prims["cap_root"],
        cap_leaf=prims["cap_leaf"],
        tube_root=prims["tube_root"],
        tube_leaf=prims["tube_leaf"],
        cap_root_path=paths["cap_root"],
        cap_leaf_path=paths["cap_leaf"],
        tube_root_path=paths["tube_root"],
        tube_leaf_path=paths["tube_leaf"],
    )


def _author_class_label(prim: object, label: str, Usd: object, UsdSemantics: object) -> None:
    api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    if not api or not api.CreateLabelsAttr().Set([label]):
        raise RuntimeError(f"could not author session-only class label {label}")
    labels = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default()).ComputeUniqueInheritedLabels(prim)
    if set(str(item) for item in labels) != {label}:
        raise RuntimeError(f"class label readback failed for {prim.GetPath()}")


def _root_pose(prim: object, UsdGeom: object) -> RootPose:
    imageable = UsdGeom.Imageable(prim)
    visibility = str(imageable.GetVisibilityAttr().Get())
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    names = tuple(str(op.GetOpName()) for op in ops)
    type_names = tuple(name.split(":")[-1] for name in names)
    validate_cap_pose_layout(names, type_names)
    if tuple(op.GetOpType() for op in ops) != (
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeScale,
    ):
        raise RuntimeError("cap root xform op types are not translate, rotateXYZ, scale")
    translation = ops[0].Get()
    rotation = ops[1].Get()
    scale = ops[2].Get()
    return RootPose(
        source=SourcePose(
            visibility=visibility,
            translation=(float(translation[0]), float(translation[1]), float(translation[2])),
            rotation_xyz_degrees=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
        ),
        scale=(float(scale[0]), float(scale[1]), float(scale[2])),
        op_order=names,
        local_transform=_matrix_values(xformable.GetLocalTransformation()),
        translate_op=ops[0],
        rotate_op=ops[1],
        scale_op=ops[2],
    )


def _apply_pose(prim: object, source: RootPose, pose: object, UsdGeom: object, Gf: object) -> None:
    """Mutate only existing translate and rotateXYZ attributes in the session layer."""

    imageable = UsdGeom.Imageable(prim)
    if not imageable.GetVisibilityAttr().Set(pose.visibility):
        raise RuntimeError("could not author cap visibility")
    if not source.translate_op.Set(Gf.Vec3d(*pose.translation)):
        raise RuntimeError("could not author cap translation")
    if not source.rotate_op.Set(Gf.Vec3f(*pose.rotation_xyz_degrees)):
        raise RuntimeError("could not author cap rotation")


def _verify_restored_pose(prim: object, source: RootPose, UsdGeom: object) -> None:
    """Fail closed unless op order, scale, and effective local transform are restored exactly."""

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    names = tuple(str(op.GetOpName()) for op in ops)
    types = tuple(name.split(":")[-1] for name in names)
    if names != source.op_order:
        raise RuntimeError("cap root xform op order changed during structural states")
    validate_cap_pose_layout(names, types)
    scale = tuple(float(value) for value in ops[2].Get())
    local = _matrix_values(xformable.GetLocalTransformation())
    visible = str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get())
    translation = tuple(float(value) for value in ops[0].Get())
    rotation = tuple(float(value) for value in ops[1].Get())
    if (
        scale != source.scale
        or local != source.local_transform
        or visible != source.source.visibility
        or translation != source.source.translation
        or rotation != source.source.rotation_xyz_degrees
    ):
        raise RuntimeError("cap root source pose was not restored exactly")


def _annotator_data(raw: object, name: str) -> tuple[np.ndarray, Mapping[object, object]]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("info"), Mapping) or "idToLabels" not in raw["info"]:
        raise RuntimeError(f"{name} annotator did not provide data and idToLabels")
    data = np.asarray(raw.get("data"))
    mapping = raw["info"]["idToLabels"]
    if not isinstance(mapping, Mapping):
        raise RuntimeError(f"{name} idToLabels is malformed")
    return data, mapping


def _target_instance_mask_from_envelope(
    raw: object,
    cap_root_path: str,
    *,
    expected_visible: bool,
    warped_instance: np.ndarray | None = None,
):
    """Resolve target-cap IDs from the documented exact root-path envelope mapping."""

    instance, id_to_paths = _annotator_data(raw, "instance")
    return target_instance_mask(
        instance if warped_instance is None else warped_instance,
        id_to_paths,
        cap_root_path,
        expected_visible=expected_visible,
    )


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
    session_identifier = str(stage.GetSessionLayer().identifier)
    for ordinal, layer in enumerate(stage.GetUsedLayers(), start=1):
        if str(layer.identifier) == session_identifier:
            continue
        real_path = Path(str(layer.realPath)).resolve(strict=True) if layer.realPath else None
        if real_path is None or not real_path.is_file():
            raise RuntimeError(f"used non-session USD layer {ordinal} is not a regular file")
        snapshots.append(snapshot_input_file(f"usd_used_layer_{len(snapshots) + 1}", real_path))
    if not snapshots:
        raise RuntimeError("complete stage has no non-session USD layers")
    return tuple(snapshots)


def _capture(
    pre: Preflight,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, object],
    dict[str, str],
    tuple[InputSnapshot, ...],
]:
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
        target_prims = _target_prims(stage, pre.target)
        camera_transform_before = _matrix_values(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        camera = UsdGeom.Camera(camera_prim)
        clipping = camera.GetClippingRangeAttr()
        clipping_before = tuple(float(value) for value in clipping.Get())
        source_pose = _root_pose(target_prims.cap_root, UsdGeom)
        stage.SetEditTarget(stage.GetSessionLayer())
        _author_class_label(target_prims.cap_root, "target_cap", Usd, UsdSemantics)
        _author_class_label(target_prims.tube_root, "target_vial", Usd, UsdSemantics)
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
        instance_mapping_evidence: dict[str, dict[str, object]] = {}
        for state in pre.states:
            _apply_pose(
                target_prims.cap_root,
                source_pose,
                apply_structural_state(source_pose.source, state),
                UsdGeom,
                Gf,
            )
            app.update()
            rep.orchestrator.step(rt_subframes=pre.subframes, delta_time=0.0, wait_for_render=True)
            raw_rgb = np.asarray(rgb_ann.get_data())
            semantic_raw, semantic_labels = _annotator_data(semantic_ann.get_data(), "semantic")
            raw_instance = instance_ann.get_data()
            instance_raw, _instance_labels = _annotator_data(raw_instance, "instance")
            if raw_rgb.ndim != 3 or raw_rgb.shape[:2] != (576, 768) or raw_rgb.shape[2] < 3:
                raise RuntimeError(f"RGB capture has wrong shape: {raw_rgb.shape}")
            if semantic_raw.shape != (576, 768) or instance_raw.shape != (576, 768):
                raise RuntimeError("semantic and instance AOVs must match overscan raster")
            rgb = warp_rgb_linear(raw_rgb[..., :3], source_x, source_y)
            semantic = warp_ids_nearest(semantic_raw, source_x, source_y)
            instance = warp_ids_nearest(instance_raw, source_x, source_y)
            expected_cap_visible = state != "missing"
            cap_semantic = target_semantic_mask(
                semantic,
                semantic_labels,
                "target_cap",
                expected_visible=expected_cap_visible,
            ).mask
            vial_semantic = target_semantic_mask(
                semantic,
                semantic_labels,
                "target_vial",
                expected_visible=True,
            ).mask
            cap_instance_result = _target_instance_mask_from_envelope(
                raw_instance,
                target_prims.cap_root_path,
                expected_visible=expected_cap_visible,
                warped_instance=instance,
            )
            cap_instance = cap_instance_result.mask
            instance_mapping_evidence[state] = {
                "path": target_prims.cap_root_path,
                "selected_ids": list(cap_instance_result.selected_ids),
                "expected_visible": expected_cap_visible,
                "mapping": "info.idToLabels exact ID-to-path equality",
            }
            artifacts[state] = {"rgb": rgb, "semantic": semantic, "instance": instance, "target_cap_semantic_mask": cap_semantic, "target_cap_instance_mask": cap_instance, "target_vial_mask": vial_semantic}
        _apply_pose(
            target_prims.cap_root,
            source_pose,
            apply_structural_state(source_pose.source, "intact"),
            UsdGeom,
            Gf,
        )
        _verify_restored_pose(target_prims.cap_root, source_pose, UsdGeom)
        clipping.Set(Gf.Vec2f(*clipping_before))
        app.update()
        camera_transform_after = _matrix_values(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        clipping_after = tuple(float(value) for value in clipping.Get())
        camera_gate = {"passed": camera_transform_before == camera_transform_after and clipping_before == clipping_after, "path": camera_path, "transform_before": list(camera_transform_before), "transform_after": list(camera_transform_after), "clipping_before": list(clipping_before), "clipping_after": list(clipping_after), "resync_restored": clipping_before == clipping_after}
        camera_gate["target_cap_instance_mapping"] = instance_mapping_evidence
        if not camera_gate["passed"]:
            raise RuntimeError("CapSealCam transform or clipping range changed during pilot")
        captured = artifacts, camera_gate, target_prims.provenance(), input_snapshots
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    return _publish_then_close(pre, captured, app)


def _publish(
    pre: Preflight,
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
    camera_gate: Mapping[str, object],
    target_prims: Mapping[str, str],
    snapshots: tuple[InputSnapshot, ...],
) -> Path:
    gates = evaluate_structural_gates(artifacts)
    gates["camera_path_transform_clipping_unchanged"] = dict(camera_gate)
    gates["target_cap_instance_mapping"] = dict(
        camera_gate["target_cap_instance_mapping"]
    )
    input_hashes = {snapshot.role: snapshot.sha256 for snapshot in snapshots}
    receipt = build_structural_receipt(target=pre.target, states=pre.states, input_hashes=input_hashes, gates=gates)
    receipt["provenance"] = {"target_prims": dict(target_prims)}

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
    output = publish_directory_no_clobber(pre.output, populate)
    if not gates["passed"]:
        failed = ", ".join(str(name) for name in gates["failed"])
        raise GateError(f"structural gates failed; failed artifacts published at {output}: {failed}")
    return output


def _publish_then_close(pre: Preflight, captured: tuple, app: object, *, publisher=None) -> Path:
    """Publish one completed capture around Kit's fast-shutdown boundary."""

    if publisher is None:
        publisher = _publish
    try:
        output = publisher(pre, *captured)
        print(f"published synthetic CapSealCam structural pilot: {output}", flush=True)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pre = preflight(args)
        _capture(pre)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
