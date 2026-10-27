"""Public contract for label-free Varied V3 stage measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_receipt(root: Path) -> dict[str, str]:
    path = root / "runtime-receipt.json"
    _write_json(
        path,
        {
            "schema_version": 3,
            "artifact_type": "f2_varied_runtime_acceptance_receipt",
            "status": "accepted",
            "provider": "brev",
            "runtime_id": "brev-isaac62-varied-v3",
            "isaac_sim_version": "6.2.0.0",
            "source_sha256": "b" * 64,
            "hard_gates": {
                "scene_parity": True,
                "finite_particle_state": True,
                "containment": True,
                "zero_leakage": True,
                "rendered_body_volume_height": True,
                "stream_continuity": True,
                "source_color": True,
                "numeric_mixing": True,
                "visual_mixing": True,
                "semantics_alignment": True,
                "timing": True,
            },
            "particle_height_diagnostic": {
                "hard_gate": False,
                "physics_pass_claimed": False,
                "status": "within_bounds",
                "bounds_passed": True,
                "warning_count": 0,
                "warned_vial_count": 0,
                "vial_count": 3,
                "claim": "particle_height_within_bounds_diagnostic_only",
                "vials": [
                    {
                        "case_id": f"CASE_{ordinal:016x}",
                        "vial_id": f"V{ordinal}",
                        "status": "within_bounds",
                        "bounds_passed": True,
                        "warning_count": 0,
                    }
                    for ordinal in (1, 2, 3)
                ],
            },
        },
    )
    return {"path": str(path), "sha256": _sha(path)}


def _manifest_path(
    tmp_path: Path,
    *,
    active_tanks: tuple[str, ...] = ("t200", "t100"),
) -> Path:
    root = tmp_path / "published"
    root.mkdir()
    events = [
        {
            "event_id": f"dispense-{ordinal}-{tank}",
            "stage_ordinal": ordinal,
            "tank_id": tank,
            "tank_order": ordinal,
            "physics_step": ordinal * 120,
            "frame_id": f"frame-{ordinal:04d}",
            "event_relative_timestamp_s": float(ordinal),
            "final_observation_offset_s": 0.3 if ordinal == len(active_tanks) else 0.0,
        }
        for ordinal, tank in enumerate(active_tanks, start=1)
    ]
    colors = {
        "V1": ((255, 0, 0), (128, 128, 128), (90, 100, 110)),
        "V2": ((0, 255, 0), (128, 128, 128), (100, 110, 120)),
        "V3": ((0, 0, 255), (128, 128, 128), (110, 120, 130)),
    }
    rows: list[dict[str, object]] = []
    for vial_id in ("V1", "V2", "V3"):
        for event in events:
            ordinal = int(event["stage_ordinal"])
            stem = f"{vial_id}-{event['event_id']}"
            rgb_path = root / f"{stem}-rgb.png"
            mask_path = root / f"{stem}-mask.png"
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[:] = colors[vial_id][ordinal - 1]
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[100:300, 200:300] = 255
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            rows.append(
                {
                    "vial_id": vial_id,
                    "event_id": event["event_id"],
                    "stage_ordinal": ordinal,
                    "physics_step": event["physics_step"],
                    "frame_id": event["frame_id"],
                    "event_relative_timestamp_s": event["event_relative_timestamp_s"],
                    "final_observation_offset_s": event["final_observation_offset_s"],
                    "view_id": "vial_fill",
                    "verified": True,
                    "rgb": {"path": str(rgb_path), "sha256": _sha(rgb_path)},
                    "liquid_mask": {"path": str(mask_path), "sha256": _sha(mask_path)},
                }
            )
    manifest = {
        "schema_version": 3,
        "artifact_type": "f2_stage_capture_manifest",
        "status": "accepted",
        "data_origin": "twin",
        "batch_id": "BATCH-V3",
        "run_id": "RUN-TRAIN-01",
        "recipe_id": "6",
        "split_role": "train",
        "split_version": "whole-run-varied-v3",
        "view_id": "vial_fill",
        "runtime_binding": {
            "provider": "brev",
            "runtime_name": "NVIDIA Isaac Sim",
            "isaac_sim_version": "6.2.0.0",
            "runtime_id": "brev-isaac62-varied-v3",
            "source_sha256": "b" * 64,
        },
        "runtime_receipt": _runtime_receipt(root),
        "stage_events": events,
        "rows": rows,
    }
    path = root / "f2-stage-capture-manifest.json"
    _write_json(path, manifest)
    return path


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(key for child in value.values() for key in _all_keys(child))]
    if isinstance(value, list):
        return [key for child in value for key in _all_keys(child)]
    return []


def _manifest_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_stage_manifest_yields_robust_stats_and_all_same_stage_vial_pairs(tmp_path):
    from fluidmix.f2_varied_stage_measurement import build_varied_stage_measurements

    manifest_path = _manifest_path(tmp_path)

    result = build_varied_stage_measurements([manifest_path])

    assert result["artifact_type"] == "f2_varied_stage_measured_evidence"
    assert result["status"] == "measured"
    assert result["real_data_used"] is False
    assert result["denominators"] == {
        "source_manifests": 1,
        "runs": 1,
        "vials": 3,
        "active_stage_events": 2,
        "stage_measurements": 6,
        "same_run_stage_pairs": 6,
        "two_stage_runs": 1,
        "three_stage_runs": 0,
    }
    assert len(result["stage_measurements"]) == 6
    assert len(result["same_run_stage_pairs"]) == 6
    first = result["stage_measurements"][0]
    assert first["source_manifest_sha256"] == _sha(manifest_path)
    assert first["case_key"] == "BATCH-V3:V1"
    assert first["rgb_sha256"] == _sha(
        manifest_path.parent / "V1-dispense-1-t200-rgb.png"
    )
    assert first["mask_sha256"] == _sha(
        manifest_path.parent / "V1-dispense-1-t200-mask.png"
    )
    assert first["median_lab"] == pytest.approx(
        [53.240588, 80.092308, 67.202751], abs=1e-5
    )
    assert first["mad_lab"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert first["iqr_lab"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert first["median_srgb_0_255"] == [255.0, 0.0, 0.0]
    assert all(pair["unit"] == "delta_e76" for pair in result["same_run_stage_pairs"])
    assert all(pair["run_id"] == "RUN-TRAIN-01" for pair in result["same_run_stage_pairs"])
    red_green = next(
        pair
        for pair in result["same_run_stage_pairs"]
        if pair["stage_ordinal"] == 1
        and pair["left_vial_id"] == "V1"
        and pair["right_vial_id"] == "V2"
    )
    assert red_green["delta_e76"] == pytest.approx(170.565595, abs=1e-5)
    assert red_green["left_rgb_sha256"] == first["rgb_sha256"]
    assert not any(
        fragment in key.lower()
        for key in _all_keys(result)
        for fragment in ("label", "residual", "palette", "disposition", "decision")
    )


def test_three_active_stages_are_measured_without_assuming_a_fixed_stage_set(tmp_path):
    from fluidmix.f2_varied_stage_measurement import build_varied_stage_measurements

    result = build_varied_stage_measurements(
        [_manifest_path(tmp_path, active_tanks=("t200", "t400", "t100"))]
    )

    assert result["denominators"] == {
        "source_manifests": 1,
        "runs": 1,
        "vials": 3,
        "active_stage_events": 3,
        "stage_measurements": 9,
        "same_run_stage_pairs": 9,
        "two_stage_runs": 0,
        "three_stage_runs": 1,
    }
    assert {row["tank_id"] for row in result["stage_measurements"]} == {
        "t100",
        "t200",
        "t400",
    }


@pytest.mark.parametrize("forbidden_key", ["label", "residual_particle_count", "palette"])
def test_oracle_fields_are_rejected_before_measurement(tmp_path, forbidden_key):
    from fluidmix.f2_varied_stage_measurement import (
        VariedStageMeasurementError,
        build_varied_stage_measurements,
    )

    path = _manifest_path(tmp_path)
    manifest = _manifest_payload(path)
    manifest[forbidden_key] = "oracle"
    _write_json(path, manifest)

    with pytest.raises(VariedStageMeasurementError):
        build_varied_stage_measurements([path])


def test_missing_rgb_fails_closed(tmp_path):
    from fluidmix.f2_varied_stage_measurement import (
        VariedStageMeasurementError,
        build_varied_stage_measurements,
    )

    path = _manifest_path(tmp_path)
    manifest = _manifest_payload(path)
    Path(manifest["rows"][0]["rgb"]["path"]).unlink()

    with pytest.raises(VariedStageMeasurementError, match="rgb_missing"):
        build_varied_stage_measurements([path])


def test_misaligned_mask_fails_closed_even_when_its_new_hash_is_declared(tmp_path):
    from fluidmix.f2_varied_stage_measurement import (
        VariedStageMeasurementError,
        build_varied_stage_measurements,
    )

    path = _manifest_path(tmp_path)
    manifest = _manifest_payload(path)
    mask_path = Path(manifest["rows"][0]["liquid_mask"]["path"])
    Image.fromarray(np.full((479, 640), 255, dtype=np.uint8), mode="L").save(mask_path)
    manifest["rows"][0]["liquid_mask"]["sha256"] = _sha(mask_path)
    _write_json(path, manifest)

    with pytest.raises(VariedStageMeasurementError, match="stage_measurement_invalid"):
        build_varied_stage_measurements([path])


def test_nonfinite_lab_conversion_fails_closed(tmp_path, monkeypatch):
    from fluidmix.f2_varied_stage_measurement import (
        VariedStageMeasurementError,
        build_varied_stage_measurements,
    )
    import skimage.color

    path = _manifest_path(tmp_path)
    monkeypatch.setattr(
        skimage.color,
        "rgb2lab",
        lambda rgb: np.full(rgb.shape, np.inf, dtype=np.float64),
    )

    with pytest.raises(VariedStageMeasurementError, match="stage_measurement_invalid"):
        build_varied_stage_measurements([path])


def test_one_vial_stage_set_drift_fails_closed(tmp_path):
    from fluidmix.f2_varied_stage_measurement import (
        VariedStageMeasurementError,
        build_varied_stage_measurements,
    )

    path = _manifest_path(tmp_path)
    manifest = _manifest_payload(path)
    vial3_rows = [row for row in manifest["rows"] if row["vial_id"] == "V3"]
    vial3_rows[1]["event_id"] = vial3_rows[0]["event_id"]
    vial3_rows[1]["stage_ordinal"] = vial3_rows[0]["stage_ordinal"]
    vial3_rows[1]["physics_step"] = vial3_rows[0]["physics_step"]
    vial3_rows[1]["frame_id"] = vial3_rows[0]["frame_id"]
    vial3_rows[1]["event_relative_timestamp_s"] = vial3_rows[0][
        "event_relative_timestamp_s"
    ]
    vial3_rows[1]["final_observation_offset_s"] = vial3_rows[0][
        "final_observation_offset_s"
    ]
    _write_json(path, manifest)

    with pytest.raises(VariedStageMeasurementError):
        build_varied_stage_measurements([path])


def test_cli_publishes_once_and_refuses_to_clobber(tmp_path):
    manifest = _manifest_path(tmp_path)
    output = tmp_path / "measured-evidence.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_f2_varied_stage_measurements.py"),
        "--capture-manifest",
        str(manifest),
        "--out",
        str(output),
    ]

    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    assert "measured 6 stages" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "measured"
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
