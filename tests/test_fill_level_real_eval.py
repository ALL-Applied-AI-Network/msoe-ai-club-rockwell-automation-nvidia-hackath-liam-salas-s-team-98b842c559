from pathlib import Path
import hashlib

import numpy as np
import pytest

from fluidmix import fill_level_real_eval as REAL


ROOT = Path(__file__).resolve().parents[1]
RULER = {
    "slope_ml_per_row": -0.32894736842105265,
    "intercept_ml": 138.22368421052633,
}


class FakeCv2:
    CAP_PROP_POS_FRAMES = 1


class FakeCapture:
    def __init__(self, frames, *, seek_offset=0):
        self.frames = dict(frames)
        self.position = 0
        self.seek_offset = seek_offset
        self.read_indices = []

    def isOpened(self):
        return True

    def set(self, property_id, value):
        assert property_id == FakeCv2.CAP_PROP_POS_FRAMES
        self.position = int(value) + self.seek_offset
        return True

    def get(self, property_id):
        assert property_id == FakeCv2.CAP_PROP_POS_FRAMES
        return float(self.position)

    def read(self):
        index = self.position
        self.read_indices.append(index)
        frame = self.frames.get(index)
        if frame is None:
            return False, None
        self.position += 1
        return True, frame.copy()


def _window(**changes):
    value = REAL.RealWindow(
        video_id=5,
        vial=2,
        recipe_id=5,
        previous_recipe_id=5,
        recipe_changed=False,
        transition_or_repeat="repeat",
        zone="visible",
        start_frame=10,
        end_frame=14,
        frame_count=5,
        recipe_volume_ml=30.0,
    )
    return REAL.RealWindow(**{**value.__dict__, **changes})


def _prediction(frame_index, status, *, row=None, top=None, bottom=None, edge=False):
    return {
        "frame_index": frame_index,
        "status": status,
        "liquid_top_row": row,
        "range_top_row": row if top is None else top,
        "range_bottom_row": row if bottom is None else bottom,
        "edge_flag": edge,
    }


def test_validate_real_inputs_accepts_only_the_frozen_27_vials():
    result = REAL.validate_real_inputs(
        ROOT / "data/real_check_windows.csv",
        REAL.EXPECTED_WINDOWS_SHA256,
        ROOT / "data/video5_vial_recipes.csv",
        REAL.EXPECTED_RECIPES_SHA256,
    )

    assert [item.vial for item in result["windows"]] == list(range(2, 10)) + [11, 12] + list(range(14, 31))
    assert result["frame_count"] == 1470
    assert result["zone_counts"] == {"visible": 10, "edge": 15, "hidden": 2}
    assert result["transition_or_repeat_counts"] == {"transition": 7, "repeat": 20}
    assert result["excluded_vials"] == {"ineligible": [1, 10], "calibration": [13]}


def test_validate_real_inputs_rejects_tampered_windows_before_parsing(tmp_path):
    tampered = tmp_path / "windows.csv"
    tampered.write_bytes((ROOT / "data/real_check_windows.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="windows CSV SHA-256"):
        REAL.validate_real_inputs(
            tampered,
            REAL.EXPECTED_WINDOWS_SHA256,
            ROOT / "data/video5_vial_recipes.csv",
            REAL.EXPECTED_RECIPES_SHA256,
        )


def test_validate_real_inputs_rejects_alternate_hash_even_if_it_matches_bytes(tmp_path):
    alternate = tmp_path / "windows.csv"
    alternate.write_bytes((ROOT / "data/real_check_windows.csv").read_bytes() + b"\n")
    alternate_hash = hashlib.sha256(alternate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="argument is not the frozen value"):
        REAL.validate_real_inputs(
            alternate,
            alternate_hash,
            ROOT / "data/video5_vial_recipes.csv",
            REAL.EXPECTED_RECIPES_SHA256,
        )


def test_bgr_to_rgb_is_explicit_and_non_aliasing():
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    rgb = REAL.bgr_to_rgb(bgr)
    assert rgb.tolist() == [[[3, 2, 1], [6, 5, 4]]]
    rgb[0, 0, 0] = 99
    assert bgr[0, 0, 2] == 3


def test_decode_window_seeks_once_and_reads_exact_inclusive_span():
    frames = {index: np.full((480, 640, 3), [index, 2, 3], dtype=np.uint8) for index in range(10, 16)}
    capture = FakeCapture(frames)
    decoded = REAL.decode_window(capture, _window(), FakeCv2)
    assert [item["frame_index"] for item in decoded] == [10, 11, 12, 13, 14]
    assert capture.read_indices == [10, 11, 12, 13, 14]
    assert decoded[0]["rgb"][0, 0].tolist() == [3, 2, 10]


def test_decode_window_rejects_inexact_seek_or_bad_frame_geometry():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="seek"):
        REAL.decode_window(FakeCapture({i: frame for i in range(10, 15)}, seek_offset=1), _window(), FakeCv2)
    bad = np.zeros((479, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="640x480"):
        REAL.decode_window(FakeCapture({10: bad, **{i: frame for i in range(11, 15)}}), _window(), FakeCv2)


def test_decode_window_partial_failure_reports_frames_already_decoded():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(REAL.FrameDecodeError) as caught:
        REAL.decode_window(FakeCapture({10: frame, 11: frame}), _window(), FakeCv2)
    assert caught.value.frames_decoded == 2


def test_visible_majority_uses_median_row_and_hand_ruler():
    results = [_prediction(10 + i, "visible", row=row) for i, row in enumerate((320.0, 322.0, 324.0))]
    results += [_prediction(13, "hidden", row=280.0, top=217.0, bottom=309.0), _prediction(14, "abstain")]
    result = REAL.aggregate_vial(_window(), results, RULER)
    assert result["status"] == "visible"
    assert result["frames_used"] == 3
    assert result["raw_surface_row"] == 322.0
    assert result["predicted_ml"] == pytest.approx(32.3026315789)
    assert result["signed_error_ml"] == pytest.approx(2.3026315789)
    assert result["absolute_error_ml"] == pytest.approx(2.3026315789)


def test_hidden_majority_medians_endpoints_separately_and_never_uses_midpoint():
    results = [
        _prediction(10 + i, "hidden", row=bottom, top=top, bottom=bottom)
        for i, (top, bottom) in enumerate(((200.0, 290.0), (202.0, 292.0), (204.0, 294.0)))
    ]
    results += [_prediction(13, "visible", row=250.0, edge=True), _prediction(14, "abstain")]
    result = REAL.aggregate_vial(_window(recipe_volume_ml=60.0, zone="hidden"), results, RULER)
    assert result["status"] == "hidden"
    assert result["frames_used"] == 3
    assert result["raw_row_range"] == [202.0, 292.0]
    assert result["predicted_ml_range"] == pytest.approx([42.1710526316, 71.7763157895])
    assert result["recipe_in_range"] is True
    assert result["distance_to_range_ml"] == 0.0
    assert "predicted_ml" not in result
    assert result["edge_flagged_frames"] == 1


def test_tie_abstains_and_fewer_than_three_frames_is_no_clean_frame():
    visible = _prediction(10, "visible", row=320.0)
    hidden = _prediction(11, "hidden", row=292.0, top=202.0, bottom=292.0)
    tied = REAL.aggregate_vial(
        _window(frame_count=4, end_frame=13),
        [visible, {**visible, "frame_index": 12}, hidden, {**hidden, "frame_index": 13}],
        RULER,
    )
    no_clean = REAL.aggregate_vial(_window(), [visible, _prediction(12, "abstain")], RULER)
    assert tied["status"] == "abstain"
    assert tied["frames_used"] == 0
    assert no_clean["status"] == "no_clean_frame"
    assert no_clean["decoded_frames"] == 2


def test_aggregate_rejects_duplicate_out_of_span_and_invalid_rows():
    visible = _prediction(10, "visible", row=320.0)
    with pytest.raises(ValueError, match="duplicate"):
        REAL.aggregate_vial(_window(), [visible, visible, _prediction(11, "abstain")], RULER)
    with pytest.raises(ValueError, match="outside"):
        REAL.aggregate_vial(
            _window(),
            [visible, _prediction(11, "visible", row=321.0), _prediction(99, "visible", row=322.0)],
            RULER,
        )
    with pytest.raises(ValueError, match="finite"):
        REAL.aggregate_vial(
            _window(),
            [
                visible,
                _prediction(11, "visible", row=float("nan")),
                _prediction(12, "visible", row=322.0),
            ],
            RULER,
        )
    with pytest.raises(ValueError, match="unsupported status"):
        REAL.aggregate_vial(
            _window(),
            [visible, _prediction(11, "visible", row=321.0), _prediction(12, "other")],
            RULER,
        )


def test_summarize_vials_reports_thresholds_zones_and_transition_split():
    visible = REAL.aggregate_vial(
        _window(vial=2, recipe_volume_ml=30.0),
        [_prediction(10 + i, "visible", row=329.0) for i in range(5)],
        RULER,
    )
    hidden = REAL.aggregate_vial(
        _window(vial=11, recipe_volume_ml=60.0, zone="hidden"),
        [_prediction(10 + i, "hidden", row=292.0, top=202.0, bottom=292.0) for i in range(5)],
        RULER,
    )
    edge = REAL.aggregate_vial(
        _window(vial=7, recipe_volume_ml=70.0, zone="edge", recipe_changed=True, transition_or_repeat="transition"),
        [_prediction(10 + i, "abstain") for i in range(5)],
        RULER,
    )
    summary = REAL.summarize_vials([visible, hidden, edge])
    assert summary["overall"]["vials"] == 3
    assert summary["overall"]["status_counts"] == {"visible": 1, "hidden": 1, "abstain": 1, "no_clean_frame": 0}
    for threshold in (1, 2, 5):
        assert summary["overall"][f"visible_within_{threshold}_ml_count"] == 1
    assert summary["overall"]["hidden_recipe_in_range_count"] == 1
    assert summary["by_zone"]["hidden"]["hidden_recipe_in_range_rate"] == 1.0
    assert summary["by_zone"]["edge"]["abstain_count"] == 1
    assert summary["by_transition_or_repeat"]["transition"]["vials"] == 1
