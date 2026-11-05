import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import vial_analysis

from vial_analysis import (
    AnalysisConfig,
    ReferenceProfile,
    VideoFrames,
    _result_is_current,
    apply_calibration,
    build_reference,
    configuration_fingerprint,
    extract_frames,
    measure_frame,
    postprocess_measurements,
    row_medians,
    measurements_from_masks,
    track_vial,
)


def rectangular_mask(height=10, width=12, x0=2, x1=8, y0=1, y1=9):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_row_medians_uses_only_masked_pixels_and_requires_minimum_width():
    lab = np.zeros((4, 6, 3), dtype=np.float32)
    lab[1, 1:5] = [50, 4, -2]
    lab[2, 2:4] = [90, 9, 9]
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 1:5] = True
    mask[2, 2:4] = True

    rows = row_medians(lab, mask, min_row_pixels=3)

    np.testing.assert_allclose(rows[1], [50, 4, -2])
    assert np.isnan(rows[0]).all()
    assert np.isnan(rows[2]).all()


def test_build_reference_uses_multiple_empty_frames_not_one_outlier():
    mask = rectangular_mask()
    labs = []
    for lightness in (48.0, 50.0, 90.0):
        lab = np.zeros((10, 12, 3), dtype=np.float32)
        lab[mask] = [lightness, 2.0, -3.0]
        labs.append(lab)

    ref = build_reference(
        labs, [mask, mask, mask], dark_l=35, min_row_pixels=3, min_vial_pixels=20
    )

    assert ref.frame_count == 3
    assert ref.bottom_y == 8
    assert ref.center_x == pytest.approx(4.5)
    np.testing.assert_allclose(ref.row_lab[4], [50.0, 2.0, -3.0])


def test_build_reference_rejects_tiny_masks_in_reference_window():
    lab = np.full((10, 12, 3), [50.0, 0.0, 0.0], dtype=np.float32)
    tiny = rectangular_mask(y0=1, y1=2)

    with pytest.raises(ValueError, match="no valid masks"):
        build_reference([lab], [tiny], min_row_pixels=3, min_vial_pixels=20)


def test_measure_frame_detects_fill_after_lateral_shift_without_discarding_it():
    ref_mask = rectangular_mask(x0=2, x1=8)
    shifted_mask = rectangular_mask(x0=4, x1=10)
    row_lab = np.full((10, 3), np.nan, dtype=np.float32)
    row_lab[1:9] = [50.0, 0.0, 0.0]
    ref = ReferenceProfile(
        row_lab=row_lab,
        usable_rows=np.array([False] + [True] * 8 + [False]),
        bottom_y=8,
        center_x=4.5,
        frame_count=5,
    )
    current = np.zeros((10, 12, 3), dtype=np.float32)
    current[shifted_mask] = [50.0, 0.0, 0.0]
    for y in (6, 7, 8):
        current[y, shifted_mask[y]] = [70.0, 0.0, 0.0]

    result = measure_frame(
        current,
        shifted_mask,
        ref,
        de_threshold=6.0,
        gap_rows=2,
        min_row_pixels=3,
        min_vial_pixels=20,
        large_shift_px=1.0,
    )

    assert result["status"] == "ok"
    assert result["fill_top_y"] == 6
    assert result["fill_height_px"] == 3
    assert result["vial_shift_px"] == pytest.approx(2.0)
    assert "vial_shift_large" in result["quality_flags"]


def test_measure_frame_returns_no_vial_for_missing_or_tiny_masks():
    ref = ReferenceProfile(
        row_lab=np.zeros((10, 3), dtype=np.float32),
        usable_rows=np.ones(10, dtype=bool),
        bottom_y=8,
        center_x=4.5,
        frame_count=3,
    )
    lab = np.zeros((10, 12, 3), dtype=np.float32)

    assert measure_frame(lab, None, ref, min_vial_pixels=20)["status"] == "no_vial"
    assert measure_frame(lab, rectangular_mask(y0=1, y1=2), ref, min_vial_pixels=20)["status"] == "no_vial"


def test_postprocess_adds_rolling_monotonic_signal_and_drop_flag():
    raw = pd.DataFrame(
        {
            "frame_sampled": range(5),
            "fill_height_px": [0.0, 10.0, 2.0, 12.0, 14.0],
            "quality_flags": ["", "", "", "", ""],
        }
    )

    result = postprocess_measurements(raw, smoothing_window=3, downward_jump_px=5)

    np.testing.assert_allclose(result["fill_height_smoothed_px"], [5, 2, 10, 12, 13])
    np.testing.assert_allclose(result["fill_height_monotonic_px"], [5, 5, 10, 12, 13])
    assert result.loc[2, "quality_flags"] == "downward_jump"


def test_postprocess_resets_smoothing_and_monotonic_height_after_vial_gap():
    raw = pd.DataFrame(
        {
            "frame_sampled": range(7),
            "status": ["ok", "ok", "no_vial", "no_vial", "no_vial", "ok", "ok"],
            "fill_height_px": [1.0, 5.0, np.nan, np.nan, np.nan, 2.0, 4.0],
            "quality_flags": [""] * 7,
        }
    )

    result = postprocess_measurements(raw, smoothing_window=1, cycle_gap_frames=2)

    assert result["vial_cycle_id"].tolist() == [0, 0, pd.NA, pd.NA, pd.NA, 1, 1]
    assert result.loc[1, "fill_height_monotonic_px"] == 5.0
    assert result.loc[5, "fill_height_monotonic_px"] == 2.0
    assert pd.isna(result.loc[3, "fill_height_smoothed_px"])


def test_calibration_is_explicit_and_never_fabricates_physical_units():
    values = pd.DataFrame({"fill_height_smoothed_px": [10.0, 20.0]})

    uncalibrated = apply_calibration(values, mm_per_pixel=None, volume_coefficients=None)
    assert uncalibrated["fill_height_mm"].isna().all()
    assert uncalibrated["volume_ml"].isna().all()
    assert set(uncalibrated["calibration_status"]) == {"uncalibrated"}

    calibrated = apply_calibration(values, mm_per_pixel=0.5, volume_coefficients=(0.1, 0.0))
    np.testing.assert_allclose(calibrated["fill_height_mm"], [5.0, 10.0])
    np.testing.assert_allclose(calibrated["volume_ml"], [0.5, 1.0])
    assert set(calibrated["calibration_status"]) == {"calibrated"}


def _write_test_video(path: Path, fps=10.0, frame_count=9):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 12))
    assert writer.isOpened()
    for i in range(frame_count):
        frame = np.full((12, 16, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_uses_measured_fps_full_duration_and_invalidates_stale_stride(tmp_path):
    video = tmp_path / "clip.mp4"
    frame_dir = tmp_path / "frames"
    _write_test_video(video, fps=10.0, frame_count=9)

    first = extract_frames(video, frame_dir, stride=3)
    assert first.sampled_frame_count == 3
    assert first.source_frame_indices == [0, 3, 6]
    np.testing.assert_allclose(first.sample_times_s, [0.0, 0.3, 0.6], atol=0.02)
    assert len(list(frame_dir.glob("*.jpg"))) == 3

    second = extract_frames(video, frame_dir, stride=2)
    assert second.sampled_frame_count == 5
    assert second.source_frame_indices == [0, 2, 4, 6, 8]
    assert len(list(frame_dir.glob("*.jpg"))) == 5


def test_extract_frames_rejects_equal_count_cache_with_wrong_filenames(tmp_path):
    video = tmp_path / "clip.mp4"
    frame_dir = tmp_path / "frames"
    _write_test_video(video, fps=10.0, frame_count=9)
    extract_frames(video, frame_dir, stride=3)
    (frame_dir / "000001.jpg").unlink()
    (frame_dir / "999999.jpg").write_bytes((frame_dir / "000000.jpg").read_bytes())

    refreshed = extract_frames(video, frame_dir, stride=3)

    assert refreshed.sampled_frame_count == 3
    assert sorted(path.name for path in frame_dir.glob("*.jpg")) == [
        "000000.jpg", "000001.jpg", "000002.jpg"
    ]


def test_unverified_initial_reference_only_emits_relative_change_not_absolute_height(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    mask = rectangular_mask()
    for index, lightness in enumerate((200, 200, 245)):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[mask] = [lightness, lightness, lightness]
        cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), image)
    frames = VideoFrames(
        video_path=tmp_path / "clip.mp4",
        frame_dir=frame_dir,
        source_fps=10.0,
        source_frame_count=3,
        sampled_frame_count=3,
        width=12,
        height=10,
        stride=1,
        source_frame_indices=[0, 1, 2],
        sample_times_s=[0.0, 0.1, 0.2],
    )
    config = AnalysisConfig(
        video_dir=tmp_path,
        reference_frame_count=2,
        minimum_reference_frames=2,
        min_row_pixels=3,
        min_vial_pixels=20,
        gap_rows=2,
        smoothing_window=1,
    )

    result, reference, provenance = measurements_from_masks(
        "clip.mp4", frames, {0: mask, 1: mask, 2: mask}, config
    )

    assert reference.frame_count == 2
    assert provenance.used_indices == (0, 1)
    assert set(result["reference_basis"]) == {"relative_initial_unverified"}
    assert result["fill_height_px"].isna().all()
    assert "fill_height_change_px" in result


def test_missing_reference_masks_emit_unusable_rows_instead_of_crashing(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(2):
        cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), np.full((10, 12, 3), 200, np.uint8))
    frames = VideoFrames(
        video_path=tmp_path / "clip.mp4", frame_dir=frame_dir, source_fps=10.0,
        source_frame_count=2, sampled_frame_count=2, width=12, height=10, stride=1,
        source_frame_indices=[0, 1], sample_times_s=[0.0, 0.1],
    )
    config = AnalysisConfig(video_dir=tmp_path, min_vial_pixels=20, smoothing_window=1)

    result, reference, provenance = measurements_from_masks("clip.mp4", frames, {}, config)

    assert reference is None
    assert provenance.used_indices == ()
    assert set(result.status) == {"reference_unavailable"}
    assert result.fill_height_px.isna().all()
    assert result.fill_height_change_px.isna().all()


def test_verified_reference_requires_multiple_valid_frames_and_emits_absolute_height(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    mask = rectangular_mask()
    for index, lightness in enumerate((200, 200, 200, 245)):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[mask] = [lightness] * 3
        cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), image)
    frames = VideoFrames(
        video_path=tmp_path / "clip.mp4", frame_dir=frame_dir, source_fps=10.0,
        source_frame_count=4, sampled_frame_count=4, width=12, height=10, stride=1,
        source_frame_indices=[0, 1, 2, 3], sample_times_s=[0.0, 0.1, 0.2, 0.3],
    )
    base = dict(
        video_dir=tmp_path, reference_frame_count=3, minimum_reference_frames=3,
        min_row_pixels=3, min_vial_pixels=20, gap_rows=2, smoothing_window=1,
    )
    verified = AnalysisConfig(**base, verified_empty_frames={"clip.mp4": (0, 1, 2)})

    result, reference, provenance = measurements_from_masks(
        "clip.mp4", frames, {0: mask, 1: mask, 2: mask, 3: mask}, verified
    )

    assert reference is not None
    assert provenance.basis == "verified_empty"
    assert provenance.requested_indices == (0, 1, 2)
    assert provenance.used_indices == (0, 1, 2)
    assert result.fill_height_px.notna().all()

    insufficient = AnalysisConfig(**base, verified_empty_frames={"clip.mp4": (0, 1, 2)})
    unavailable, reference, provenance = measurements_from_masks(
        "clip.mp4", frames, {0: mask, 1: mask}, insufficient
    )
    assert reference is None
    assert provenance.basis == "reference_unavailable"
    assert provenance.used_indices == ()
    assert set(unavailable.status) == {"reference_unavailable"}


def test_track_vial_selects_requested_object_id_and_covers_final_frame(tmp_path, monkeypatch):
    frames = VideoFrames(
        video_path=tmp_path / "clip.mp4", frame_dir=tmp_path, source_fps=10.0,
        source_frame_count=3, sampled_frame_count=3, width=2, height=2, stride=1,
        source_frame_indices=[0, 1, 2], sample_times_s=[0.0, 0.1, 0.2],
    )
    config = AnalysisConfig(video_dir=tmp_path, min_vial_pixels=1)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())

    class FakePredictor:
        max_frames = None

        def init_state(self, **kwargs):
            return {}

        def add_new_points_or_box(self, *args, **kwargs):
            return None

        def propagate_in_video(self, *args, **kwargs):
            self.max_frames = kwargs["max_frame_num_to_track"]
            logits = torch.tensor([
                [[[-1.0, -1.0], [-1.0, -1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ])
            yield 0, [2, 1], logits
            yield 2, [2, 1], logits

    predictor = FakePredictor()
    masks = track_vial(predictor, frames, config)

    assert masks[0].all()
    assert masks[2].all()
    assert predictor.max_frames == 2


def test_batch_reuses_predictor_and_continues_after_per_video_failure(tmp_path, monkeypatch):
    videos = [tmp_path / name for name in ("a.mp4", "b.mp4", "c.mp4")]
    for video in videos:
        video.write_bytes(b"video")
    config = AnalysisConfig(
        video_dir=tmp_path, work_dir=tmp_path / "work", results_dir=tmp_path / "results"
    )
    loads = []

    monkeypatch.setattr(vial_analysis, "load_predictor", lambda model_id: loads.append(model_id) or object())
    monkeypatch.setattr(vial_analysis, "_result_is_current", lambda *args: False)

    def fake_analyze(video_path, config, predictor):
        if video_path.name == "b.mp4":
            raise RuntimeError("broken clip")
        frame = pd.DataFrame({
            "video_name": [video_path.name], "status": ["relative_change"],
            "reference_basis": ["relative_initial_unverified"],
        })
        manifest = {
            "video_name": video_path.name, "status": "completed", "sampled_frames": 1,
            "measured_rows": 1, "ok_rows": 0, "relative_rows": 1,
            "reference_basis": "relative_initial_unverified", "scientifically_usable": False,
            "elapsed_s": 0.1, "error": "",
        }
        return frame, manifest

    monkeypatch.setattr(vial_analysis, "analyze_video", fake_analyze)

    combined, manifest = vial_analysis.analyze_batch(config, videos)

    assert loads == [config.model_id]
    assert manifest.status.tolist() == ["completed", "failed", "completed"]
    assert combined.video_name.tolist() == ["a.mp4", "c.mp4"]
    persisted = pd.read_csv(config.results_dir / "processing_manifest.csv")
    assert persisted.status.tolist() == ["completed", "failed", "completed"]


def test_resume_rejects_truncated_csv_even_when_metadata_matches(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    config = AnalysisConfig(video_dir=tmp_path, results_dir=tmp_path / "results")
    csv_path = config.results_dir / "per_video" / "clip.csv"
    metadata_path = config.results_dir / "per_video" / "clip.json"
    diagnostic_path = config.results_dir / "diagnostics" / "clip.png"
    csv_path.parent.mkdir(parents=True)
    diagnostic_path.parent.mkdir(parents=True)
    csv_path.write_text("status,config_fingerprint\nok,bad\n")
    diagnostic_path.write_bytes(b"png")
    stat = video.stat()
    metadata_path.write_text(json.dumps({
        "config_fingerprint": configuration_fingerprint(config),
        "video_signature": {
            "path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        },
        "row_count": 10,
    }))

    assert _result_is_current(video, config) is False


def test_configuration_fingerprint_changes_when_scientific_settings_change(tmp_path):
    base = AnalysisConfig(video_dir=tmp_path)
    same = AnalysisConfig(video_dir=tmp_path)
    changed = AnalysisConfig(video_dir=tmp_path, de_threshold=8.0)

    assert configuration_fingerprint(base) == configuration_fingerprint(same)
    assert configuration_fingerprint(base) != configuration_fingerprint(changed)

