"""CPU-only checks for the FillCam pilot summarizer."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_fill_level_pilot.py"
SPEC = importlib.util.spec_from_file_location("summarize_fill_level_pilot", SCRIPT)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_image(left, right):
    image = np.full((480, 640, 3), 235, dtype=np.uint8)
    image[:, left:right] = 35
    image[:, :, 1] = np.clip(image[:, :, 1] + 5, 0, 255)
    return image


def write_pilot(tmp_path, include_75=True):
    pilot_dir = tmp_path / "pilot"
    (pilot_dir / "rgb").mkdir(parents=True)
    (pilot_dir / "mask").mkdir()
    definitions = [(8155, 30.0, 329.0, 329.0), (8114, 70.0, 210.0, 208.0)]
    if include_75:
        definitions.append((8117, 75.0, 195.0, 194.0))
    records = []
    for index, (seed, level, geometry_row, rgb_row) in enumerate(definitions):
        rgb = edge_image(280, 340)
        mask = np.full((480, 640), 255, dtype=np.uint8)
        mask[70:450, 280:340] = 0
        mask[250:450, 280:340] = 1
        mask[180:240, 280:287] = 3
        rgb_path = pilot_dir / "rgb" / f"{index:05d}.png"
        mask_path = pilot_dir / "mask" / f"{index:05d}.png"
        assert cv2.imwrite(str(rgb_path), rgb)
        assert cv2.imwrite(str(mask_path), mask)
        records.append(
            {
                "schema_version": 1,
                "seed": seed,
                "role": "pilot",
                "level_ml": level,
                "holder_material": {
                    "brightness": 0.04 + index * 0.1,
                    "roughness": 0.3 + index * 0.2,
                },
                "foam": {
                    "enabled": False,
                    "thickness_mm": 0.0,
                    "visible_in_camera": False,
                },
                "sticker_pose": {
                    "present": False,
                    "visible_in_camera": False,
                },
                "wet_walls": {"enabled": index == 0},
                "outputs": {
                    "rgb": {
                        "path": f"rgb/{rgb_path.name}",
                        "sha256": sha256(rgb_path),
                    },
                    "mask": {
                        "path": f"mask/{mask_path.name}",
                        "sha256": sha256(mask_path),
                    },
                },
                "label_stats": {
                    "sticker_blocked_pixels": 0,
                    "vial_pixels": 22_800,
                    "class_pixels": {
                        "empty": 10_380,
                        "liquid": 12_000,
                        "foam": 0,
                        "blocked": 420,
                        "ignore": 284_400,
                    },
                },
                "surface": {"geometry_row": geometry_row, "rgb_row": rgb_row},
                "timings_seconds": {
                    "rgb": 2.0 + index,
                    "glass_only": 0.8 + index * 0.1,
                    "glass_hidden_fluid": 1.2 + index * 0.1,
                    "interior_probe_blockers": 1.0 + index * 0.1,
                    "all_frame_passes": 5.0 + index,
                },
            }
        )
    manifest = pilot_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    (pilot_dir / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "frame_count": len(records),
                "renderer": "PathTracing",
                "subframes": 32,
                "resolution": [640, 480],
                "manifest_sha256": sha256(manifest),
                "seconds": 60.0,
                "phase_timings_seconds": {
                    "simulation_app_start": 10.0,
                    "stage_load": 1.0,
                    "warmup": 1.0,
                },
            }
        )
    )
    real = tmp_path / "real-reference.jpg"
    assert cv2.imwrite(str(real), edge_image(282, 338))
    return pilot_dir, real, records


def test_build_report_measures_rows_edges_labels_and_timings(tmp_path):
    pilot_dir, real, records = write_pilot(tmp_path)
    report = PILOT.build_report(pilot_dir, real, 8155)
    assert report["integrity"]["passed"]
    assert report["frame_count"] == 3
    held_out = {
        measurement["level_ml"]: measurement
        for measurement in report["held_out_surface_rows"]
    }
    assert held_out[70.0]["geometry_delta_px_vs_real"] == pytest.approx(4.0)
    assert held_out[75.0]["rgb_delta_px_vs_real"] == pytest.approx(0.0)
    position = report["horizontal_vial_position"]
    assert position["synthetic"]["left"]["x"] == 279
    assert position["synthetic"]["right"]["x"] == 339
    assert position["delta_px_synthetic_minus_real"] == {
        "left": -2,
        "right": 2,
        "center": 0.0,
        "width": 4,
    }
    assert report["timing_seconds"]["all_frame_passes"]["median"] == 6.0
    assert report["label_coverage"]["frames_with_class"]["liquid"] == 3
    assert report["case_coverage"]["holder_material"]["brightness_min"] == 0.04
    assert report["integrity"]["render_summary_matches_manifest"]
    assert report["batch_sizing"]["end_to_end_frames_per_hour"] == 180.0
    assert report["batch_sizing"]["startup_adjusted_frames_per_hour"] == 225.0
    assert report["batch_sizing"][
        "estimated_hours_for_selected_total_conservative"
    ] == pytest.approx(3800 / 180)

    montage = tmp_path / "visual" / "montage.png"
    PILOT.write_montage(records, pilot_dir / "manifest.jsonl", montage, columns=2)
    image = cv2.imread(str(montage), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (356, 800, 3)


def test_missing_held_out_level_is_rejected(tmp_path):
    pilot_dir, real, _ = write_pilot(tmp_path, include_75=False)
    with pytest.raises(ValueError, match="75 mL record"):
        PILOT.build_report(pilot_dir, real, 8155)


def test_stale_render_summary_is_rejected(tmp_path):
    pilot_dir, real, _ = write_pilot(tmp_path)
    summary_path = pilot_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    report = PILOT.build_report(pilot_dir, real, 8155)
    assert not report["integrity"]["passed"]
    assert not report["integrity"]["render_summary_matches_manifest"]
    assert any("manifest_sha256" in error for error in report["integrity"]["errors"])


def test_edge_ranges_and_outputs_are_non_destructive(tmp_path):
    with pytest.raises(ValueError, match="outside image height"):
        PILOT.measure_vial_edges(
            edge_image(280, 340), (470, 490), (260, 300), (320, 360)
        )
    existing = tmp_path / "report.json"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        PILOT._write_json_new(existing, {"replacement": True})
    assert existing.read_text() == "keep"
