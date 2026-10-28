"""CPU contracts for the CapSealCam paired structural pilot.

These tests exercise only deterministic validation, raster and receipt contracts.  They
do not start Isaac Sim and cannot establish rendered or real-world correctness.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import importlib.util
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidmix.capseal_structural import (
    FINAL_RASTER_SIZE,
    GateError,
    OutputPreflightError,
    SourcePose,
    StateValidationError,
    apply_structural_state,
    build_structural_receipt,
    central_crop_coordinates,
    evaluate_structural_gates,
    preflight_output_path,
    validate_states,
    validate_target,
    warp_ids_nearest,
    warp_rgb_linear,
    validate_camera_record,
    target_instance_mask,
    target_semantic_mask,
    validate_cap_pose_layout,
)


RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_capseal_structural_pilot.py"


def _load_renderer_module():
    spec = importlib.util.spec_from_file_location("capseal_structural_renderer_test", RENDER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_target_and_states_accept_only_the_fixed_paired_pilot_domain() -> None:
    assert validate_target("G15") == "G15"
    assert validate_states("intact,missing,unseated") == (
        "intact",
        "missing",
        "unseated",
    )

    for target in ("G00", "G31", "G1", "g15", "G015", " G15"):
        with pytest.raises(StateValidationError, match="target"):
            validate_target(target)
    for states in ("intact,intact", "intact,unknown", "", "intact,"):
        with pytest.raises(StateValidationError):
            validate_states(states)


def test_unseated_pose_is_derived_from_frozen_source_without_compounding() -> None:
    source = SourcePose(
        visibility="inherited",
        translation=(1.5, -2.0, 3.25),
        rotation_xyz_degrees=(4.0, 5.0, 6.0),
    )

    unseated = apply_structural_state(source, "unseated")
    missing = apply_structural_state(source, "missing")

    assert source == SourcePose("inherited", (1.5, -2.0, 3.25), (4.0, 5.0, 6.0))
    assert unseated.visibility == "inherited"
    assert unseated.translation == (1.5, -2.0, 3.45)
    assert unseated.rotation_xyz_degrees == (14.0, 5.0, 6.0)
    assert unseated.displacement_mm == 5.08
    assert missing.visibility == "invisible"
    assert missing.translation == source.translation
    assert missing.rotation_xyz_degrees == source.rotation_xyz_degrees


def test_identity_lens_coordinates_centrally_crop_overscan_without_changing_rgb() -> None:
    height, width = 576, 768
    rgb = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    source_x, source_y = central_crop_coordinates()

    warped = warp_rgb_linear(rgb, source_x, source_y)

    assert FINAL_RASTER_SIZE == (640, 480)
    assert np.array_equal(warped, rgb[48:528, 64:704])


def test_capseal_camera_record_requires_reviewed_overscan_and_lens_fields() -> None:
    record = {
        "camera": "CapSealCam",
        "render_px": [768, 576],
        "final_image_px": [640, 480],
        "focal_px_final": 652.91,
        "lens_k1": -0.1527001634003988,
    }

    normalized = validate_camera_record(record)

    assert normalized["width"] == 768
    assert normalized["height"] == 576
    assert normalized["focal_px_final"] == 652.91
    record["lens_k1"] = "bad"
    with pytest.raises(ValueError, match="lens_k1"):
        validate_camera_record(record)


def test_nearest_id_warp_preserves_integer_labels_and_uses_zero_outside_bounds() -> None:
    labels = np.array([[5, 11], [17, 23]], dtype=np.uint32)
    source_x = np.array([[0.0, 1.0, 3.0], [0.49, 0.51, -0.51]])
    source_y = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    warped = warp_ids_nearest(labels, source_x, source_y)

    assert warped.dtype == np.uint32
    assert warped.tolist() == [[5, 23, 0], [17, 23, 0]]


def test_output_preflight_refuses_an_occupied_destination(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    assert preflight_output_path(output) == output.resolve()
    output.mkdir()

    with pytest.raises(OutputPreflightError, match="occupied"):
        preflight_output_path(output)


def test_capseal_renderer_cli_help_is_safe_without_isaac() -> None:
    completed = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--target" in completed.stdout
    assert "--states" in completed.stdout
    assert "--subframes" in completed.stdout


def test_runtime_publishes_and_announces_before_fast_shutdown(capsys) -> None:
    renderer = _load_renderer_module()
    events = []

    class FastShutdownApp:
        def close(self, exit_code=0):
            events.append(("close", exit_code))
            raise SystemExit(exit_code)

    def publish(pre, *captured):
        events.append(("publish", pre, captured))
        return Path("/tmp/pilot")

    with pytest.raises(SystemExit) as stopped:
        renderer._publish_then_close(
            "preflight",
            ("artifacts", "camera", "targets", "snapshots"),
            FastShutdownApp(),
            publisher=publish,
        )

    assert stopped.value.code == 0
    assert events == [
        (
            "publish",
            "preflight",
            ("artifacts", "camera", "targets", "snapshots"),
        ),
        ("close", 0),
    ]
    assert "published synthetic CapSealCam structural pilot: /tmp/pilot" in capsys.readouterr().out


def test_runtime_reports_publication_failure_before_nonzero_fast_shutdown(capsys) -> None:
    renderer = _load_renderer_module()
    events = []

    class FastShutdownApp:
        def close(self, exit_code=0):
            events.append(("close", exit_code))
            raise SystemExit(exit_code)

    def fail_publish(pre, *captured):
        events.append(("publish", pre, captured))
        raise RuntimeError("publication failed")

    with pytest.raises(SystemExit) as stopped:
        renderer._publish_then_close(
            "preflight",
            ("artifacts", "camera", "targets", "snapshots"),
            FastShutdownApp(),
            publisher=fail_publish,
        )

    assert stopped.value.code == 1
    assert events[-1] == ("close", 1)
    assert "RuntimeError: publication failed" in capsys.readouterr().err


def test_preflight_rejects_missing_required_twin_lens_script_before_isaac(tmp_path: Path) -> None:
    renderer = _load_renderer_module()
    twin = tmp_path / "twin"
    (twin / "cameras").mkdir(parents=True)
    (twin / "tools").mkdir()
    (twin / "csi_line.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (twin / "tools" / "render_example.py").write_text("# organizer\n", encoding="utf-8")
    (twin / "cameras" / "CapSealCam.json").write_text(
        '{"camera":"CapSealCam","render_px":[768,576],"final_image_px":[640,480],'
        '"focal_px_final":652.91,"lens_k1":-0.1527}',
        encoding="utf-8",
    )
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    args = renderer.parse_args(
        ["--twin", str(twin), "--target", "G15", "--out", str(output_parent / "pilot")]
    )

    with pytest.raises(ValueError, match="missing required lens script"):
        renderer.preflight(args)


def test_preflight_requires_all_three_structural_states_before_runtime(tmp_path: Path) -> None:
    renderer = _load_renderer_module()
    twin = tmp_path / "twin"
    (twin / "cameras").mkdir(parents=True)
    (twin / "tools").mkdir()
    (twin / "csi_line.usda").write_text("#usda 1.0\n", encoding="utf-8")
    for name in ("render_example.py", "apply_lens.py"):
        (twin / "tools" / name).write_text("# organizer\n", encoding="utf-8")
    (twin / "cameras" / "CapSealCam.json").write_text(
        '{"camera":"CapSealCam","render_px":[768,576],"final_image_px":[640,480],'
        '"focal_px_final":652.91,"lens_k1":-0.1527}',
        encoding="utf-8",
    )
    parent = tmp_path / "out"
    parent.mkdir()
    args = renderer.parse_args(
        ["--twin", str(twin), "--target", "G15", "--states", "intact", "--out", str(parent / "pilot")]
    )

    with pytest.raises(StateValidationError, match="all three"):
        renderer.preflight(args)


def test_target_masks_use_faithful_envelopes_and_exact_cap_root_mapping() -> None:
    renderer = _load_renderer_module()
    semantic = np.array([[0, 4], [0, 0]], dtype=np.uint32)
    instance = np.array([[0, 9], [0, 0]], dtype=np.uint32)
    semantic_envelope = {
        "data": semantic,
        "info": {"idToLabels": {0: {"class": "background"}, 4: {"class": "target_cap"}}},
    }
    cap_root = "/CSI_Line/TrayInspection/G15_cap"
    instance_envelope = {"data": instance, "info": {"idToLabels": {9: cap_root}}}
    hidden_semantic_envelope = {
        "data": np.zeros((2, 2), dtype=np.uint32),
        "info": {"idToLabels": {0: {"class": "background"}}},
    }
    hidden_instance_envelope = {
        "data": np.zeros((2, 2), dtype=np.uint32),
        "info": {"idToLabels": {}},
    }

    visible_semantic = target_semantic_mask(*renderer._annotator_data(semantic_envelope, "semantic"), "target_cap", expected_visible=True)
    visible_instance = renderer._target_instance_mask_from_envelope(
        instance_envelope, cap_root, expected_visible=True
    )
    hidden_semantic = target_semantic_mask(
        *renderer._annotator_data(hidden_semantic_envelope, "semantic"), "target_cap", expected_visible=False
    )
    hidden_instance = renderer._target_instance_mask_from_envelope(
        hidden_instance_envelope, cap_root, expected_visible=False
    )

    assert visible_semantic.mask.tolist() == [[False, True], [False, False]]
    assert visible_instance.mask.tolist() == [[False, True], [False, False]]
    assert not hidden_semantic.mask.any()
    assert not hidden_instance.mask.any()
    with pytest.raises(ValueError, match="semantic class absent"):
        target_semantic_mask(np.zeros((2, 2), dtype=np.uint32), {0: {"class": "background"}}, "target_cap", expected_visible=True)
    with pytest.raises(ValueError, match="instance path absent"):
        renderer._target_instance_mask_from_envelope(hidden_instance_envelope, cap_root, expected_visible=True)


def test_cap_pose_layout_requires_only_existing_translate_rotatexyz_scale_ops() -> None:
    assert validate_cap_pose_layout(
        ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"),
        ("translate", "rotateXYZ", "scale"),
    ) == ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
    with pytest.raises(ValueError, match="op layout"):
        validate_cap_pose_layout(("xformOp:transform",), ("transform",))


class _Layer:
    def __init__(self, path: Path | None, identifier: str) -> None:
        self.realPath = "" if path is None else str(path)
        self.identifier = identifier


class _Stage:
    def __init__(self, layers, session) -> None:
        self.layers = layers
        self.session = session

    def GetUsedLayers(self):
        return self.layers

    def GetLayerStack(self, includeSessionLayers=False):
        return self.layers[-1:]

    def GetSessionLayer(self):
        return self.session


class _Path:
    def __init__(self, value: str) -> None:
        self.pathString = value


class _Prim:
    def __init__(self, path: str, type_name: str) -> None:
        self.path = path
        self.type_name = type_name

    def IsValid(self):
        return True

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return _Path(self.path)


class _PathStage:
    def __init__(self, prims):
        self.prims = prims
        self.requested = []

    def GetPrimAtPath(self, path):
        self.requested.append(path)
        return self.prims[path]


def test_stage_layer_snapshot_uses_all_non_session_used_file_layers(tmp_path: Path) -> None:
    renderer = _load_renderer_module()
    files = []
    for number in range(6):
        path = tmp_path / f"layer-{number}.usdc"
        path.write_text(f"layer {number}\n", encoding="utf-8")
        files.append(_Layer(path, str(path)))
    session = _Layer(None, "anon:session")

    snapshots = renderer._stage_layer_snapshots(_Stage([session, *files], session))

    assert [snapshot.role for snapshot in snapshots] == [f"usd_used_layer_{number}" for number in range(1, 7)]
    assert [snapshot.path.name for snapshot in snapshots] == [f"layer-{number}.usdc" for number in range(6)]


def test_target_lookup_uses_exact_root_and_leaf_paths_with_reviewed_types() -> None:
    renderer = _load_renderer_module()
    base = "/CSI_Line/TrayInspection/G15"
    cap_root = base + "_cap"
    tube_root = base + "_tube"
    cap_leaf = cap_root + "/vial_cap"
    tube_leaf = tube_root + "/vial_tube"
    stage = _PathStage(
        {
            cap_root: _Prim(cap_root, "Xform"),
            cap_leaf: _Prim(cap_leaf, "Mesh"),
            tube_root: _Prim(tube_root, "Xform"),
            tube_leaf: _Prim(tube_leaf, "Mesh"),
        }
    )

    targets = renderer._target_prims(stage, "G15")

    assert targets.cap_root_path == cap_root
    assert targets.cap_leaf_path == cap_leaf
    assert targets.tube_root_path == tube_root
    assert targets.tube_leaf_path == tube_leaf
    assert stage.requested == [cap_root, cap_leaf, tube_root, tube_leaf]


def _artifacts(*, cap: np.ndarray, vial: np.ndarray, cap_instance: np.ndarray | None = None):
    height, width = 480, 640
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[10, 10] = (1, 2, 3)
    semantic = np.zeros((height, width), dtype=np.uint32)
    semantic[cap] = 7
    instance = np.zeros((height, width), dtype=np.uint32)
    instance[cap if cap_instance is None else cap_instance] = 101
    return {
        "rgb": rgb,
        "semantic": semantic,
        "instance": instance,
        "target_cap_semantic_mask": cap,
        "target_cap_instance_mask": cap if cap_instance is None else cap_instance,
        "target_vial_mask": vial,
    }


def test_receipt_records_all_gate_failures_and_never_claims_real_data() -> None:
    intact_cap = np.zeros((480, 640), dtype=bool)
    intact_cap[100:110, 100:110] = True
    unseated_cap = np.zeros((480, 640), dtype=bool)
    unseated_cap[100:110, 103:113] = True
    vial = np.zeros((480, 640), dtype=bool)
    vial[200:230, 200:230] = True
    wrong_cap_instance = intact_cap.copy()
    wrong_cap_instance[100, 100] = False

    artifacts = {
        "intact": _artifacts(cap=intact_cap, vial=vial, cap_instance=wrong_cap_instance),
        "missing": _artifacts(cap=np.zeros_like(intact_cap), vial=vial),
        "unseated": _artifacts(cap=unseated_cap, vial=vial),
    }
    gates = evaluate_structural_gates(artifacts)
    receipt = build_structural_receipt(
        target="G15",
        states=("intact", "missing", "unseated"),
        input_hashes={"complete_stage": "a" * 64},
        gates=gates,
    )

    assert gates["passed"] is False
    assert "intact.target_cap_semantic_instance_agreement" in gates["failed"]
    assert receipt["real_data_used"] is False
    assert receipt["claim_boundary"] == "synthetic structural pilot only; no real-data claim"
    with pytest.raises(GateError, match="semantic_instance_agreement"):
        evaluate_structural_gates(artifacts, require_pass=True)


def test_failed_runtime_gates_publish_failed_receipt_and_artifacts_before_raising(tmp_path: Path) -> None:
    renderer = _load_renderer_module()
    intact_cap = np.zeros((480, 640), dtype=bool)
    intact_cap[100:110, 100:110] = True
    unseated_cap = np.zeros((480, 640), dtype=bool)
    unseated_cap[100:110, 103:113] = True
    vial = np.zeros((480, 640), dtype=bool)
    vial[200:230, 200:230] = True
    wrong_cap_instance = intact_cap.copy()
    wrong_cap_instance[100, 100] = False
    artifacts = {
        "intact": _artifacts(cap=intact_cap, vial=vial, cap_instance=wrong_cap_instance),
        "missing": _artifacts(cap=np.zeros_like(intact_cap), vial=vial),
        "unseated": _artifacts(cap=unseated_cap, vial=vial),
    }
    output = tmp_path / "failed-pilot"
    pre = renderer.Preflight(
        twin=tmp_path,
        stage=tmp_path / "stage.usda",
        camera_file=tmp_path / "camera.json",
        camera={},
        output=output,
        target="G15",
        states=("intact", "missing", "unseated"),
        subframes=32,
        seed=20261027,
        snapshots=(),
    )
    camera_gate = {"passed": True, "target_cap_instance_mapping": {}}

    with pytest.raises(GateError, match="failed artifacts published"):
        renderer._publish(pre, artifacts, camera_gate, {}, ())

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["gates"]["passed"] is False
    assert "intact.target_cap_semantic_instance_agreement" in receipt["gates"]["failed"]
    assert (output / "intact" / "rgb.png").is_file()
    assert (output / "intact" / "target_cap_mask.png").is_file()
