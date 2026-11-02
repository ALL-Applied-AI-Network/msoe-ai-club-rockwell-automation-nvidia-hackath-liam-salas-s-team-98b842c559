import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vial_review import (
    estimate_mm_per_pixel,
    fit_volume_calibration,
    make_empty_reference_candidates,
    make_validation_sample,
    parse_verified_empty_flags,
    save_verified_empty_frames,
    validation_metrics,
    validate_empty_reference_map,
    generate_review_artifacts,
)


def _measurements():
    rows = []
    for video in ("a.mp4", "b.mp4"):
        for frame, status in enumerate(
            ["no_change_from_reference", "relative_change", "relative_change", "no_vial", "relative_change"]
        ):
            rows.append(
                {
                    "video_name": video,
                    "frame_sampled": frame,
                    "video_time_s": frame / 10,
                    "status": status,
                    "fill_height_change_px": float(frame),
                    "fill_top_y": 10 - frame,
                    "fill_bottom_y": 12,
                }
            )
    return pd.DataFrame(rows)


def _frames(tmp_path: Path):
    root = tmp_path / "frames"
    for stem in ("a", "b"):
        directory = root / stem
        directory.mkdir(parents=True, exist_ok=True)
        for frame in range(5):
            (directory / f"{frame:06d}.jpg").write_bytes(b"jpg")
    return root


def test_empty_reference_candidates_cover_every_video_without_claiming_verification(tmp_path):
    candidates = make_empty_reference_candidates(_measurements(), _frames(tmp_path), count_per_video=3)

    assert candidates.groupby("video_name").size().to_dict() == {"a.mp4": 3, "b.mp4": 3}
    assert candidates["frame_path"].map(lambda value: Path(value).is_file()).all()
    assert candidates["verified_empty"].isna().all()
    assert set(candidates.groupby("video_name").first().frame_sampled) == {0}


def test_validation_sample_is_deterministic_stratified_and_unlabeled(tmp_path):
    measurements = _measurements()
    frames = _frames(tmp_path)
    sample = make_validation_sample(measurements, frames, count_per_video=4)
    shuffled = make_validation_sample(
        measurements.sample(frac=1, random_state=7), frames, count_per_video=4
    )

    assert sample.groupby("video_name").size().to_dict() == {"a.mp4": 4, "b.mp4": 4}
    assert sample.groupby("video_name").status.nunique().min() >= 3
    assert sample["manual_fill_top_y"].isna().all()
    assert sample["manual_fill_bottom_y"].isna().all()
    assert sample["known_volume_ml"].isna().all()
    assert sample["frame_path"].map(lambda value: Path(value).is_file()).all()
    pd.testing.assert_frame_equal(
        sample.sort_values(["video_name", "frame_sampled"]).reset_index(drop=True),
        shuffled.sort_values(["video_name", "frame_sampled"]).reset_index(drop=True),
    )


def test_verified_empty_map_requires_three_distinct_existing_sampled_frames(tmp_path):
    frames = _frames(tmp_path)
    videos = ["a.mp4", "b.mp4"]

    with pytest.raises(ValueError, match="at least 3"):
        validate_empty_reference_map({"a.mp4": [0, 1]}, videos, frames)
    with pytest.raises(FileNotFoundError):
        validate_empty_reference_map({"a.mp4": [0, 1, 99]}, videos, frames)
    with pytest.raises(ValueError, match="exact integer"):
        validate_empty_reference_map({"a.mp4": [0, 1, 2.5]}, videos, frames)

    normalized = validate_empty_reference_map({"a.mp4": [2, 1, 0, 1]}, videos, frames)
    assert normalized == {"a.mp4": [0, 1, 2]}

    output = tmp_path / "verified_empty_frames.json"
    save_verified_empty_frames(normalized, output, videos, frames)
    assert json.loads(output.read_text()) == {"a.mp4": [0, 1, 2]}


def test_verified_empty_flags_reject_unknown_values():
    candidates = pd.DataFrame(
        {
            "video_name": ["a.mp4"] * 4,
            "frame_sampled": [0, 1, 2, 3],
            "verified_empty": ["True", "yes", "False", ""],
        }
    )
    assert parse_verified_empty_flags(candidates) == {"a.mp4": [0, 1]}
    candidates.loc[3, "verified_empty"] = "tru"
    with pytest.raises(ValueError, match="unknown verified_empty"):
        parse_verified_empty_flags(candidates)


def test_physical_calibration_recovers_known_scale_and_linear_volume_relation():
    points = pd.DataFrame(
        {
            "known_height_mm": [5.0, 10.0, 15.0],
            "measured_height_px": [10.0, 20.0, 30.0],
            "known_volume_ml": [1.0, 2.0, 3.0],
        }
    )

    mm_per_pixel = estimate_mm_per_pixel(points)
    coefficients, metrics = fit_volume_calibration(points, mm_per_pixel, degree=1)

    assert mm_per_pixel == pytest.approx(0.5)
    np.testing.assert_allclose(coefficients, [0.2, 0.0], atol=1e-10)
    assert metrics == pytest.approx({"n": 3, "rmse_ml": 0.0, "r2": 1.0})


@pytest.mark.parametrize(
    "points, function, message",
    [
        (
            pd.DataFrame({"known_height_mm": [5.0], "measured_height_px": [np.nan]}),
            estimate_mm_per_pixel,
            "missing measured_height_px",
        ),
        (
            pd.DataFrame(
                {"known_height_mm": [5.0, 10.0], "measured_height_px": [10.0, 10.0]}
            ),
            estimate_mm_per_pixel,
            "duplicate measured heights",
        ),
        (
            pd.DataFrame({"known_height_mm": [np.inf], "measured_height_px": [10.0]}),
            estimate_mm_per_pixel,
            "finite",
        ),
        (
            pd.DataFrame({"measured_height_px": [10.0, 20.0], "known_volume_ml": [1.0, np.inf]}),
            lambda points: fit_volume_calibration(points, 0.5),
            "finite",
        ),
        (
            pd.DataFrame({"measured_height_px": [10.0, 10.0], "known_volume_ml": [1.0, 2.0]}),
            lambda points: fit_volume_calibration(points, 0.5),
            "duplicate measured heights",
        ),
    ],
)
def test_calibration_rejects_invalid_entered_rows(points, function, message):
    with pytest.raises(ValueError, match=message):
        function(points)


def test_validation_metrics_ignore_unlabeled_rows_and_report_coverage():
    labels = pd.DataFrame(
        {
            "fill_height_px": [10.0, 20.0, 99.0],
            "manual_fill_top_y": [11.0, 10.0, np.nan],
            "manual_fill_bottom_y": [20.0, 30.0, np.nan],
            "label_status": ["complete", "complete", "unlabeled"],
        }
    )

    metrics = validation_metrics(labels)

    assert metrics["labeled_rows"] == 2
    assert metrics["total_rows"] == 3
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["mae_px"] == pytest.approx(0.5)
    assert metrics["bias_px"] == pytest.approx(-0.5)


def test_validation_metrics_ignore_numeric_drafts_and_reject_unknown_status():
    labels = pd.DataFrame(
        {
            "fill_height_px": [10.0, 99.0],
            "manual_fill_top_y": [1.0, 1.0],
            "manual_fill_bottom_y": [10.0, 99.0],
            "label_status": ["complete", "unlabeled"],
        }
    )
    metrics = validation_metrics(labels)
    assert metrics["labeled_rows"] == 1
    assert metrics["coverage"] == pytest.approx(0.5)
    labels.loc[1, "label_status"] = "done-ish"
    with pytest.raises(ValueError, match="unknown label_status"):
        validation_metrics(labels)


def test_regenerating_review_artifacts_preserves_human_entries(tmp_path):
    review = tmp_path / "review"
    paths = generate_review_artifacts(
        _measurements(), _frames(tmp_path), review, candidates_per_video=3, validation_per_video=3
    )
    candidates = pd.read_csv(paths["candidates"])
    candidates["verified_empty"] = candidates["verified_empty"].astype(object)
    candidates["reviewer_notes"] = candidates["reviewer_notes"].astype(object)
    candidates.loc[0, "verified_empty"] = "True"
    candidates.loc[0, "reviewer_notes"] = "confirmed"
    candidates.to_csv(paths["candidates"], index=False)
    validation = pd.read_csv(paths["validation"])
    validation.loc[0, "manual_fill_top_y"] = 10
    validation.loc[0, "manual_fill_bottom_y"] = 20
    validation.loc[0, "label_status"] = "complete"
    validation.to_csv(paths["validation"], index=False)

    changed = _measurements().sample(frac=1, random_state=11).copy()
    changed.loc[changed.frame_sampled == 0, "status"] = "no_vial"
    generate_review_artifacts(changed, _frames(tmp_path), review, candidates_per_video=3, validation_per_video=1)

    candidates = pd.read_csv(paths["candidates"])
    validation = pd.read_csv(paths["validation"])
    reviewed_candidate = candidates[
        (candidates.video_name == "a.mp4") & (candidates.frame_sampled == 0)
    ].iloc[0]
    assert pd.notna(reviewed_candidate.verified_empty)
    assert str(reviewed_candidate.verified_empty).lower() == "true"
    assert reviewed_candidate.reviewer_notes == "confirmed"
    reviewed_validation = validation[
        (validation.video_name == "a.mp4") & (validation.frame_sampled == 0)
    ].iloc[0]
    assert reviewed_validation.manual_fill_top_y == 10
    assert reviewed_validation.manual_fill_bottom_y == 20
    assert reviewed_validation.label_status == "complete"

