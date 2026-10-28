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
