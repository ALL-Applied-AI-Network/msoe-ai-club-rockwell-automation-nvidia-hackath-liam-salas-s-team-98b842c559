"""CPU-only contracts for closed real-appearance measurement evidence."""

import hashlib
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix import fill_level_real_appearance as appearance


def _png_bytes(vial):
    y, x = np.mgrid[:480, :640]
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[..., 0] = (x // 4 + vial) % 256
    image[..., 1] = (y // 3 + vial * 2) % 256
    image[..., 2] = ((x + y) // 6 + vial * 3) % 256
    passed, encoded = cv2.imencode(".png", image)
    assert passed
    return encoded.tobytes()


def _write_closed_root(tmp_path):
    root = tmp_path / "calibration-frames"
    root.mkdir()
    frames = []
    for vial, expected in appearance.EXPECTED_WINDOWS.items():
        encoded = _png_bytes(vial)
        sha256 = hashlib.sha256(encoded).hexdigest()
        for frame_index in range(
            expected["first_frame"], expected["last_frame"] + 1
        ):
            relative = (
                f"{expected['path_prefix']}/frame-{frame_index:06d}.png"
            )
            image_path = root / Path(relative)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(encoded)
            frames.append(
                {
                    "frame_index": frame_index,
                    "height": 480,
                    "level_ml": expected["level_ml"],
                    "path": relative,
                    "recipe_id": expected["recipe_id"],
                    "role": "calibration",
                    "sha256": sha256,
                    "source_window_role": "real_validation",
                    "vial": vial,
                    "width": 640,
                }
            )
    index = {
        "frame_count": 98,
        "frames": frames,
        "image_contract": {
            "encoding": "PNG",
            "height": 480,
            "lossless": True,
            "width": 640,
        },
        "schema_version": 1,
        "selection_policy": "all frames; synthetic test fixture",
        "source": {
            "video": {
                "sha256": appearance.EXPECTED_VIDEO_SHA256,
                "expected_sha256": appearance.EXPECTED_VIDEO_SHA256,
            },
            "windows_csv": {"sha256": appearance.EXPECTED_WINDOWS_SHA256},
        },
    }
    (root / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, index


def _rewrite_index(root, index):
    (root / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _external_jpeg_bytes(camera):
    image = np.full((480, 640, 3), (170, 190, 205), dtype=np.uint8)
    if camera == "vial_fill":
        image[220:425, 290:330] = (115, 45, 18)
        image[330:346, 290:330] = (150, 115, 100)
    else:
        colors = [(120, 145, 125), (135, 120, 185), (145, 160, 145), (190, 115, 65)]
        for roi, color in zip(appearance.EXTERNAL_TANK_ROIS_XYXY.values(), colors):
            x0, y0, x1, y1 = roi
            image[y0:y1, x0:x1] = color
    passed, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert passed
    return encoded.tobytes()


def _write_external_root(tmp_path):
    root = tmp_path / "csi-vial-fill-dataset"
    root.mkdir()
    for camera in ("vial_fill", "tank_level"):
        (root / camera / "images").mkdir(parents=True)

    with (root / "videos.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "duration_s", "cameras", "n_vials", "labeled", "notes"])
        for video_id, vial_count in appearance.EXTERNAL_VIDEO_VIAL_COUNTS.items():
            writer.writerow(
                [video_id, 100, "cap_seal;tank_level;vial_fill", vial_count, "no", "fixture"]
            )

    rows_by_camera = {"vial_fill": [], "tank_level": []}
    candidate_names = {"vial_fill": [], "tank_level": []}
    serial = 0
    for video_id, row_count in appearance.EXTERNAL_VIDEO_INDEX_COUNTS.items():
        vial_count = appearance.EXTERNAL_VIDEO_VIAL_COUNTS[video_id]
        candidate_positions = {
            appearance._external_selected_position(row_count, vial_count, cycle, fraction)
            for cycle in range(vial_count)
            for fraction in appearance.EXTERNAL_CANDIDATE_FRACTIONS
        }
        for position in range(row_count):
            serial += 1
            suffix = f"20260101_000000_{serial:06d}"
            timestamp = f"2026-01-01 00:00:00.{serial:06d}"
            for camera in rows_by_camera:
                filename = f"{camera}_{suffix}.jpg"
                rows_by_camera[camera].append(
                    {"filename": filename, "timestamp": timestamp, "video_id": str(video_id)}
                )
                if position in candidate_positions:
                    candidate_names[camera].append(filename)
                    (root / camera / "images" / filename).write_bytes(
                        _external_jpeg_bytes(camera)
                    )

    # A forbidden-video row deliberately points outside the root.  The gate
    # must discard video 5 before filename/path inspection and never resolve it.
    for camera in rows_by_camera:
        rows_by_camera[camera].append(
            {
                "filename": "../must-not-open.jpg",
                "timestamp": "2026-01-02 00:00:00.000000",
                "video_id": "5",
            }
        )

    for camera, rows in rows_by_camera.items():
        with (root / camera / "image_index.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["filename", "timestamp", "video_id"]
            )
            writer.writeheader()
            writer.writerows(rows)

    expected_csv = {
        "videos.csv": appearance.file_sha256(root / "videos.csv"),
        "vial_fill/image_index.csv": appearance.file_sha256(
            root / "vial_fill" / "image_index.csv"
        ),
        "tank_level/image_index.csv": appearance.file_sha256(
            root / "tank_level" / "image_index.csv"
        ),
    }
    expected_allowlist = {}
    for camera, names in candidate_names.items():
        hashes = [
            (name, appearance.file_sha256(root / camera / "images" / name))
            for name in names
        ]
        expected_allowlist[camera] = appearance._allowlist_digest(hashes)
    return root, expected_csv, expected_allowlist, candidate_names


def test_closed_set_produces_deterministic_compact_measurements(tmp_path):
    root, _index = _write_closed_root(tmp_path)

    report = appearance.measure_real_appearance(root)
    text = appearance.json_text(report)

    assert text == appearance.json_text(report)
    assert report["additional_video5_frames_opened"] == 0
    assert report["other_real_frames_opened"] == 0
    assert "video-5-only counter" in report["other_real_frames_opened_definition"]
    assert report["provenance"]["frame_count"] == 98
    assert report["provenance"]["vial_counts"] == {"1": 28, "13": 70}
    assert report["provenance"]["validation_order"].endswith("before first decode")
    assert report["liquids"]["vial_1_30ml"]["roi_xyxy_half_open"] == [
        298,
        354,
        314,
        420,
    ]
    assert report["brady_label"]["bbox_xyxy_half_open"] == [285, 168, 341, 240]
    assert not report["global_look"]["haze_proxy"]["physical_haze_identifiable"]
    assert "proxies only" in report["global_look"]["haze_proxy"]["wording"]
    json.loads(text)


def test_external_videos_1_to_4_sample_is_gated_and_reproducible(tmp_path):
    root, _index = _write_closed_root(tmp_path)
    external, csv_hashes, allowlist_hashes, _names = _write_external_root(tmp_path)

    report = appearance.measure_real_appearance(
        root,
        external_root=external,
        expected_external_csv_sha256=csv_hashes,
        expected_external_allowlist_sha256=allowlist_hashes,
    )
    sample = report["external_reference_sample"]
    provenance = sample["provenance"]

    assert provenance["allowed_video_ids"] == [1, 2, 3, 4]
    assert provenance["forbidden_video_ids"] == [5]
    assert provenance["video5_paths_resolved_hashed_or_decoded"] == 0
    assert provenance["video5_vial_colors_csv_opened"] == 0
    assert provenance["candidate_pair_count"] == 68
    assert provenance["candidate_image_count_hashed_before_first_decode"] == 136
    assert provenance["selected_pair_count"] == 34
    assert sample["vial_fill"]["liquid"]["colored_or_dark_settled_count"] == 34
    assert report["renderer_guidance"]["dark_liquid_top_surface"]["value_range"] == [
        0.16,
        0.28,
    ]
    assert len(appearance.json_text(report)) < 200_000


def test_external_last_candidate_corruption_fails_before_first_decode(tmp_path):
    root, _index = _write_closed_root(tmp_path)
    external, csv_hashes, allowlist_hashes, names = _write_external_root(tmp_path)
    (external / "tank_level" / "images" / names["tank_level"][-1]).write_bytes(b"bad")
    decode_calls = []

    with pytest.raises(ValueError, match="tank_level candidate allowlist SHA-256 mismatch"):
        appearance.measure_real_appearance(
            root,
            external_root=external,
            expected_external_csv_sha256=csv_hashes,
            expected_external_allowlist_sha256=allowlist_hashes,
            decoder=lambda *args: decode_calls.append(args),
        )
    assert decode_calls == []


def test_external_allowed_video_path_traversal_fails_before_decode(tmp_path):
    root, _index = _write_closed_root(tmp_path)
    external, _csv_hashes, allowlist_hashes, _names = _write_external_root(tmp_path)
    index_path = external / "vial_fill" / "image_index.csv"
    rows = list(csv.DictReader(index_path.open(newline="", encoding="utf-8")))
    rows[0]["filename"] = "../forbidden.jpg"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "timestamp", "video_id"])
        writer.writeheader()
        writer.writerows(rows)
    csv_hashes = {
        "videos.csv": appearance.file_sha256(external / "videos.csv"),
        "vial_fill/image_index.csv": appearance.file_sha256(index_path),
        "tank_level/image_index.csv": appearance.file_sha256(
            external / "tank_level" / "image_index.csv"
        ),
    }
    decode_calls = []

    with pytest.raises(ValueError, match="invalid filename"):
        appearance.measure_real_appearance(
            root,
            external_root=external,
            expected_external_csv_sha256=csv_hashes,
            expected_external_allowlist_sha256=allowlist_hashes,
            decoder=lambda *args: decode_calls.append(args),
        )
    assert decode_calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "test", "role must be calibration"),
        ("source_window_role", "real_test", "must be real_validation"),
        ("width", 320, "dimensions 640x480"),
        ("vial", 2, "forbidden vial"),
    ],
)
def test_metadata_gate_fails_before_any_decode(tmp_path, field, value, message):
    root, index = _write_closed_root(tmp_path)
    index["frames"][0][field] = value
    _rewrite_index(root, index)
    decode_calls = []

    def forbidden_decoder(*args):
        decode_calls.append(args)
        raise AssertionError("decoder must not run")

    with pytest.raises(ValueError, match=message):
        appearance.measure_real_appearance(root, decoder=forbidden_decoder)
    assert decode_calls == []


def test_path_traversal_fails_before_any_decode(tmp_path):
    root, index = _write_closed_root(tmp_path)
    index["frames"][0]["path"] = "vial-01-030ml/../frame-000816.png"
    _rewrite_index(root, index)
    decode_calls = []

    with pytest.raises(ValueError, match="path traversal"):
        appearance.measure_real_appearance(
            root, decoder=lambda *args: decode_calls.append(args)
        )
    assert decode_calls == []


def test_last_hash_mismatch_proves_all_hashes_precede_decode(tmp_path):
    root, index = _write_closed_root(tmp_path)
    index["frames"][-1]["sha256"] = "0" * 64
    _rewrite_index(root, index)
    decode_calls = []

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        appearance.measure_real_appearance(
            root, decoder=lambda *args: decode_calls.append(args)
        )
    assert decode_calls == []


def test_exact_counts_and_complete_windows_are_required(tmp_path):
    root, index = _write_closed_root(tmp_path)
    index["frames"].pop()
    index["frame_count"] = 97
    _rewrite_index(root, index)

    with pytest.raises(ValueError, match="exactly 98"):
        appearance.validate_calibration_root(root)


def test_report_writer_refuses_to_overwrite(tmp_path):
    target = tmp_path / "appearance.json"
    appearance.write_json_new(target, {"schema_version": 1})
    with pytest.raises(FileExistsError):
        appearance.write_json_new(target, {"schema_version": 2})
