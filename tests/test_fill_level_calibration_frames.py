"""CPU-only checks for deterministic Step 8 calibration-frame extraction."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "extract_fill_level_calibration_frames.py"
)
SPEC = importlib.util.spec_from_file_location("extract_fill_level_calibration_frames", SCRIPT)
EXTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT)


FIELDS = [
    "video_id",
    "vial",
    "recipe_id",
    "window_start_s",
    "window_end_s",
    "start_frame",
    "end_frame",
    "n_frames",
    "role",
]


def write_windows(path, *, vial1_role="real_validation", vial13_recipe="9"):
    rows = [
        {
            "video_id": "5",
            "vial": "1",
            "recipe_id": "5",
            "window_start_s": "0.033",
            "window_end_s": "0.067",
            "start_frame": "1",
            "end_frame": "2",
            "n_frames": "2",
            "role": vial1_role,
        },
        {
            "video_id": "5",
            "vial": "13",
            "recipe_id": vial13_recipe,
            "window_start_s": "0.133",
            "window_end_s": "0.200",
            "start_frame": "4",
            "end_frame": "6",
            "n_frames": "3",
            "role": "real_validation",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.position = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def set(self, property_id, value):
        if property_id != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame


def test_window_contract_pins_vials_levels_roles_and_inclusive_counts(tmp_path):
    windows_path = tmp_path / "windows.csv"
    write_windows(windows_path)
    windows = EXTRACT.load_calibration_windows(windows_path)

    assert [(item["vial"], item["level_ml"]) for item in windows] == [
        (1, 30.0),
        (13, 80.0),
    ]
    assert [item["role"] for item in windows] == ["calibration", "calibration"]
    assert [item["n_frames"] for item in windows] == [2, 3]


@pytest.mark.parametrize(
    ("role", "recipe", "message"),
    [
        ("test", "9", "source role"),
        ("real_validation", "8", "recipe must be"),
    ],
)
def test_window_contract_rejects_wrong_role_or_calibration_recipe(
    tmp_path, role, recipe, message
):
    windows_path = tmp_path / "windows.csv"
    write_windows(windows_path, vial1_role=role, vial13_recipe=recipe)
    with pytest.raises(ValueError, match=message):
        EXTRACT.load_calibration_windows(windows_path)


def test_extracts_every_window_frame_losslessly_with_provenance(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"synthetic-video-identity")
    windows_path = tmp_path / "windows.csv"
    write_windows(windows_path)
    frames = [
        np.full((480, 640, 3), index, dtype=np.uint8) for index in range(7)
    ]
    video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    output = tmp_path / "fresh-output"

    result = EXTRACT.extract_calibration_frames(
        video_path,
        windows_path,
        output,
        expected_video_sha256=video_hash.upper(),
        capture_factory=lambda _path: FakeCapture(frames),
        generated_utc="2026-11-01T00:00:00Z",
    )

    assert result["frame_count"] == 5
    assert [item["frame_index"] for item in result["frames"]] == [1, 2, 4, 5, 6]
    assert result["selection_policy"].startswith("all frames")
    assert result["source"]["video"]["sha256"] == video_hash
    assert result["source"]["video"]["expected_sha256"] == video_hash
    saved = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert saved == result
    for record in result["frames"]:
        image_path = output / record["path"]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == record["sha256"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert image.shape == (480, 640, 3)
        assert np.all(image == record["frame_index"])

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        EXTRACT.extract_calibration_frames(
            video_path,
            windows_path,
            output,
            capture_factory=lambda _path: FakeCapture(frames),
        )


def test_sha_gate_fails_before_creating_output(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"unexpected")
    windows_path = tmp_path / "windows.csv"
    write_windows(windows_path)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EXTRACT.extract_calibration_frames(
            video_path,
            windows_path,
            output,
            expected_video_sha256="0" * 64,
            capture_factory=lambda _path: FakeCapture([]),
        )
    assert not output.exists()


def test_rejects_decoded_frames_that_are_not_640_by_480(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"synthetic-video-identity")
    windows_path = tmp_path / "windows.csv"
    write_windows(windows_path)
    frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(7)]

    with pytest.raises(ValueError, match="expected 640x480"):
        EXTRACT.extract_calibration_frames(
            video_path,
            windows_path,
            tmp_path / "partial-evidence",
            capture_factory=lambda _path: FakeCapture(frames),
        )
