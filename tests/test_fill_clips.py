"""CPU contract tests for Part-B fill clipping and stage measurement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.fill_clips import (  # noqa: E402
    TRIAD_CONFIG,
    FillBoundary,
    StageBoundary,
    build_fill_clips,
    cross_batch_threshold,
    measure_stage_boundaries,
    real_final_mixture_window,
)
from fluidmix.contracts.evidence import validate_evidence_record_v1  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video(path: Path, count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24), True
    )
    assert writer.isOpened()
    for index in range(count):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, :] = (index * 10, 30, 80)
        writer.write(frame)
    writer.release()


def _stage_assets(root: Path, name: str, colour: tuple[int, int, int]) -> tuple[Path, Path]:
    rgb = root / f"{name}.png"
    mask = root / f"{name}-mask.png"
    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[:, :] = colour
    mask_array = np.zeros((24, 32), dtype=np.uint8)
    mask_array[5:18, 8:24] = 255
    Image.fromarray(array, mode="RGB").save(rgb)
    Image.fromarray(mask_array, mode="L").save(mask)
    return rgb, mask


def test_fill_clips_write_half_open_segment_and_transition_metadata(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    output = tmp_path / "out"
    result = build_fill_clips(
        video,
        [
            {"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400", "t100"]},
            {"vial_id": "V2", "recipe_id": "R1", "dispense_order": ["t200", "t400", "t100"]},
        ],
        output,
        fill_boundaries=[
            FillBoundary("V1", 1, 3, transition_frame_ranges=((0, 1),)),
            FillBoundary("V2", 5, 7, transition_frame_ranges=((3, 5), (7, 8))),
        ],
    )
    assert result["schema_version"] == 1
    assert len(result["fills"]) == 2
    assert result["fills"][0]["fill_frame_range"] == [1, 3]
    assert result["fills"][0]["transition_frame_ranges"] == [[0, 1]]
    assert result["fills"][0]["clip_sha256"] == _sha(output / "fills" / "V1.mp4")
    assert result["fills"][0]["fill_clip_path"] == "fills/V1.mp4"
    assert result["fills"][0]["transport_frame_ranges"] == [[0, 1]]
    assert result["fills"][0]["transport_frame_ranges"] != result["fills"][0]["fill_frame_range"]


def test_stage_measurement_uses_mask_and_emits_pairwise_delta(tmp_path: Path) -> None:
    rgb1, mask1 = _stage_assets(tmp_path, "a", (200, 30, 20))
    rgb2, mask2 = _stage_assets(tmp_path, "b", (20, 30, 200))
    stages = [
        StageBoundary("V1", "after-t200", 1, "t200", 3, rgb1, mask1),
        StageBoundary("V2", "after-t200", 1, "t200", 3, rgb2, mask2),
    ]
    measured = measure_stage_boundaries(stages)
    assert len(measured["measurements"]) == 2
    assert set(measured["measurements"][0]) >= {
        "evidence_id", "measurement_id", "rgb_sha256", "mask_sha256",
        "median_lab", "mad_lab", "iqr_lab",
    }
    assert len(measured["pairwise_delta_e76"]) == 1
    assert measured["pairwise_delta_e76"][0]["left_evidence_id"] != measured["pairwise_delta_e76"][0]["right_evidence_id"]


def test_build_fill_clips_emits_stage_still_crop_and_measurement(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    rgb, mask = _stage_assets(tmp_path, "stage", (80, 100, 120))
    result = build_fill_clips(
        video,
        [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400", "t100"]}],
        tmp_path / "out",
        fill_boundaries=[FillBoundary("V1", 1, 3)],
        stage_boundaries=[
            StageBoundary("V1", "after-t200", 1, "t200", 3, rgb, mask),
            StageBoundary("V1", "after-t400", 2, "t400", 3, rgb, mask),
            StageBoundary("V1", "after-t100", 3, "t100", 3, rgb, mask),
        ],
    )
    assert result["status"] == "measured"
    assert len(result["stages"]) == len(result["measurements"]) == 3
    stage = result["stages"][0]
    assert (tmp_path / "out" / stage["still_path"]).is_file()
    assert (tmp_path / "out" / stage["crop_path"]).is_file()
    assert stage["mask_source"] == "oracle"


def test_timing_evidence_is_exact_projected_part_b_contract(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    recipe = [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200"]}]
    timing = {
        "fills": [{"vial_id": "V1", "fill_start_frame": 1, "fill_end_frame": 3}],
        "unapproved": "must fail",
    }
    with pytest.raises(ValueError, match="timing evidence fields"):
        build_fill_clips(video, recipe, tmp_path / "out", timing_evidence=timing)
    timing["unapproved"] = {"tank_colour_preset": "base"}
    with pytest.raises(ValueError, match="simulation metadata"):
        build_fill_clips(video, recipe, tmp_path / "out2", timing_evidence=timing)


def test_fill_and_transition_intervals_are_global_and_nonoverlapping(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    recipe = [
        {"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200"]},
        {"vial_id": "V2", "recipe_id": "R1", "dispense_order": ["t200"]},
    ]
    with pytest.raises(ValueError, match="global frame intervals"):
        build_fill_clips(
            video,
            recipe,
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 1, 4), FillBoundary("V2", 3, 5)],
        )
    with pytest.raises(ValueError, match="global frame intervals"):
        build_fill_clips(
            video,
            recipe,
            tmp_path / "out2",
            fill_boundaries=[
                FillBoundary("V1", 1, 3, transition_frame_ranges=((0, 1),)),
                FillBoundary("V2", 5, 7, transition_frame_ranges=((0, 1),)),
            ],
        )


def test_measured_fill_requires_each_programmed_stage_once(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    rgb, mask = _stage_assets(tmp_path, "stage", (80, 100, 120))
    recipe = [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400"]}]
    with pytest.raises(ValueError, match="exactly one stage"):
        build_fill_clips(
            video,
            recipe,
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 1, 3)],
            stage_boundaries=[StageBoundary("V1", "after-t200", 1, "t200", 2, rgb, mask)],
        )
    with pytest.raises(ValueError, match="duplicate stage"):
        build_fill_clips(
            video,
            recipe,
            tmp_path / "out2",
            fill_boundaries=[FillBoundary("V1", 1, 3)],
            stage_boundaries=[
                StageBoundary("V1", "after-t200", 1, "t200", 2, rgb, mask),
                StageBoundary("V1", "after-t200-copy", 1, "t200", 2, rgb, mask),
            ],
        )


def test_same_stage_delta_requires_comparable_context(tmp_path: Path) -> None:
    rgb1, mask1 = _stage_assets(tmp_path, "context-a", (200, 30, 20))
    rgb2, mask2 = _stage_assets(tmp_path, "context-b", (20, 30, 200))
    with pytest.raises(ValueError, match="comparable context"):
        measure_stage_boundaries(
            [
                StageBoundary("V1", "same-stage", 1, "t200", 1, rgb1, mask1, recipe_id="R1", run_id="RUN1"),
                StageBoundary("V2", "same-stage", 1, "t200", 1, rgb2, mask2, recipe_id="R2", run_id="RUN1"),
            ]
        )


def test_output_contains_validated_measured_evidence_records_and_source_bindings(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    rgb, mask = _stage_assets(tmp_path, "source", (80, 100, 120))
    recipe = [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400"]}]
    result = build_fill_clips(
        video,
        recipe,
        tmp_path / "out",
        fill_boundaries=[FillBoundary("V1", 1, 3)],
        stage_boundaries=[
            StageBoundary("V1", "after-t200", 1, "t200", 2, rgb, mask),
            StageBoundary("V1", "after-t400", 2, "t400", 2, rgb, mask),
        ],
    )
    records = result["evidence_records"]
    assert len(records) == 2
    for record in records:
        assert validate_evidence_record_v1(record, phase="measured", mode="storage")["stage_id"]
    locator = result["image_locators"][0]
    assert locator["mask_path"] == str(mask.resolve())
    assert locator["mask_sha256"] == _sha(mask)


def test_rebuilt_stage_preserves_colour_regime_and_reference_role(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    rgb, mask = _stage_assets(tmp_path, "context", (80, 100, 120))
    result = build_fill_clips(
        video,
        [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400"]}],
        tmp_path / "out",
        fill_boundaries=[FillBoundary("V1", 1, 3)],
        stage_boundaries=[
            StageBoundary("V1", "after-t200", 1, "t200", 2, rgb, mask, colour_regime_id="REG1", reference_role="target"),
            StageBoundary("V1", "after-t400", 2, "t400", 2, rgb, mask, colour_regime_id="REG1", reference_role="target"),
        ],
    )
    assert {row["colour_regime_id"] for row in result["measurements"]} == {"REG1"}
    assert {row["reference_role"] for row in result["measurements"]} == {"target"}


def test_measurement_hashes_match_the_bytes_actually_measured(tmp_path: Path) -> None:
    source_rgb, source_mask = _stage_assets(tmp_path, "source", (20, 30, 40))
    measured_rgb, measured_mask = _stage_assets(tmp_path, "measured", (200, 30, 40))
    result = measure_stage_boundaries([
        StageBoundary(
            "V1", "after-t200", 1, "t200", 1, measured_rgb, measured_mask,
            source_rgb_path=source_rgb, source_mask_path=source_mask,
        )
    ])
    row = result["measurements"][0]
    assert row["rgb_sha256"] == _sha(measured_rgb)
    assert row["mask_sha256"] == _sha(measured_mask)
    assert row["rgb_sha256"] != _sha(source_rgb)


def test_video_extracted_stage_locators_survive_final_publish_rename(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    _, mask = _stage_assets(tmp_path, "mask", (80, 100, 120))
    result = build_fill_clips(
        video,
        [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400"]}],
        tmp_path / "out",
        fill_boundaries=[FillBoundary("V1", 1, 3)],
        stage_boundaries=[
            StageBoundary("V1", "after-t200", 1, "t200", 2, None, mask, frame_index=1),
            StageBoundary("V1", "after-t400", 2, "t400", 2, None, mask, frame_index=2),
        ],
    )
    for locator in result["image_locators"]:
        assert Path(locator["derived_still_path"]).is_file()
        assert Path(locator["rgb_path"]).is_file()


def test_reversed_stage_input_fails_and_full_sequence_is_validated(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    rgb, mask = _stage_assets(tmp_path, "ordered", (80, 100, 120))
    with pytest.raises(ValueError, match="ordered stage sequence"):
        build_fill_clips(
            video,
            [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200", "t400"]}],
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 1, 3)],
            stage_boundaries=[
                StageBoundary("V1", "after-t400", 2, "t400", 2, rgb, mask),
                StageBoundary("V1", "after-t200", 1, "t200", 2, rgb, mask),
            ],
        )


def test_transition_range_must_fit_source_video(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video, count=8)
    with pytest.raises(ValueError, match="transition frame range exceeds"):
        build_fill_clips(
            video,
            [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200"]}],
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 1, 3, transition_frame_ranges=((3, 9),))],
        )


def test_boundary_rejects_simulation_truth_and_bad_order(tmp_path: Path) -> None:
    video = tmp_path / "camera.mp4"
    _video(video)
    with pytest.raises(ValueError, match="simulation metadata"):
        build_fill_clips(
            video,
            [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200"], "label": "keep"}],
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 1, 3)],
        )
    with pytest.raises(ValueError, match="fill_frame_range"):
        build_fill_clips(
            video,
            [{"vial_id": "V1", "recipe_id": "R1", "dispense_order": ["t200"]}],
            tmp_path / "out",
            fill_boundaries=[FillBoundary("V1", 3, 1)],
        )
    with pytest.raises(ValueError, match="predicted masks are not accepted"):
        StageBoundary("V1", "after-t200", 1, "t200", 1, "rgb.png", "mask.png", mask_source="predicted")


def test_threshold_is_frozen_cross_batch_rule() -> None:
    assert TRIAD_CONFIG == {
        "tight_peer_pair_max_delta_e76": 2.0,
        "target_outlier_margin_delta_e76": 1.0,
    }
    assert cross_batch_threshold(0.5) == 2.0
    assert cross_batch_threshold(2.25) == 3.25


def test_timing_evidence_is_optional_boundary_locator_and_real_window_is_gated() -> None:
    result = real_final_mixture_window(100, 120, fps=10)
    assert result["start_frame"] == 103
    assert result["end_frame"] == 120
    assert result["eligible"] is True


def test_preserved_v3_regression_when_artifacts_are_available() -> None:
    evidence = Path(
        "/home/hackathon5/liamreporoot/f2-varied-v3-measurements-20261027/measured-evidence.json"
    )
    locators = evidence.with_name("image-locators.json")
    if not evidence.is_file() or not locators.is_file():
        pytest.skip("preserved V3 artifacts are not available in this checkout")
    from fluidmix.fill_clips import remeasure_preserved_v3

    result = remeasure_preserved_v3(evidence, locators)
    assert result["status"] == "matched"
    assert result["lab_rows_expected"] == result["lab_rows_remeasured"] == 42
    assert result["delta_e_rows_expected"] == result["delta_e_rows_remeasured"] == 42
    assert result["inputs_unchanged"] is True
