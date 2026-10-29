"""Receipt projection for accepted V3 fill-clip inputs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


V3_DOWNSTREAM = Path(
    "/home/hackathon5/liamreporoot/f2-varied-v3-downstream-v2-20261027/"
    "downstream-validation-receipt.json"
)
V3_LOCATORS = Path(
    "/home/hackathon5/liamreporoot/f2-varied-v3-measurements-20261027/"
    "image-locators.json"
)


@pytest.mark.skipif(
    not V3_DOWNSTREAM.is_file() or not V3_LOCATORS.is_file(),
    reason="preserved accepted V3 artifacts are unavailable",
)
def test_public_builder_projects_preserved_v3_active_fill_intervals(tmp_path):
    from fluidmix.f2_fill_clip_inputs import build_f2_fill_clip_input_manifest

    output = tmp_path / "fill-clip-inputs.json"
    assert build_f2_fill_clip_input_manifest(
        downstream_validation_receipt=V3_DOWNSTREAM,
        image_locators=V3_LOCATORS,
        output_path=output,
    ) == output

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert set(artifact) == {
        "schema_version",
        "artifact_type",
        "status",
        "data_origin",
        "split_version",
        "source_downstream_validation_receipt_sha256",
        "source_request_materialization_receipt_sha256",
        "source_image_locators_sha256",
        "denominators",
        "records",
        "claim_boundary",
    }
    assert artifact["denominators"] == {
        "batches": 6,
        "runs": 6,
        "vials": 18,
        "stages": 42,
    }
    assert len(artifact["records"]) == 18
    assert set(artifact["records"][0]) == {
        "batch_id",
        "run_id",
        "case_id",
        "vial_id",
        "vial_position",
        "recipe_id",
        "previous_recipe_id",
        "split_role",
        "split_version",
        "recipe_ml",
        "dispense_order",
        "source_receipt_sha256",
        "video_path",
        "video_sha256",
        "video_decoded_frame_count",
        "video_playback_fps",
        "video_codec",
        "video_container",
        "fill_frame_range",
        "excluded_post_fill_frame_range",
        "fill_end_physics_step",
        "interval_rule",
        "video_timing_source_steps_sha256",
        "stages",
    }
    assert set(artifact["records"][0]["stages"][0]) == {
        "image_id",
        "stage_id",
        "stage_ordinal",
        "tank_id",
        "physics_step",
    }
    assert sum(len(row["stages"]) for row in artifact["records"]) == 42
    assert {
        row["fill_frame_range"][1] for row in artifact["records"]
    } == {222, 228, 274, 307, 370, 381}
    assert {
        row["excluded_post_fill_frame_range"][1]
        - row["excluded_post_fill_frame_range"][0]
        for row in artifact["records"]
    } == {30}
    assert all(
        "frame_index" not in stage
        for row in artifact["records"]
        for stage in row["stages"]
    )


@pytest.mark.skipif(
    not V3_DOWNSTREAM.is_file() or not V3_LOCATORS.is_file(),
    reason="preserved accepted V3 artifacts are unavailable",
)
def test_public_cli_publishes_once_and_refuses_clobber(tmp_path):
    output = tmp_path / "fill-clip-inputs.json"
    command = [
        sys.executable,
        "scripts/build_f2_fill_clip_inputs.py",
        "--downstream-validation-receipt",
        str(V3_DOWNSTREAM),
        "--image-locators",
        str(V3_LOCATORS),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert output.is_file()

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_active_fill_range_rejects_whole_video_and_nonmonotonic_steps():
    from fluidmix.f2_fill_clip_inputs import F2FillClipInputError, _active_fill_range

    render = {
        "video_decoded_frame_count": 4,
        "video_timing_source_frame_steps": [10, 20, 30, 40],
        "video_frame_source_capture_indices": [0, 1, 2, 3],
        "video_frame_repeat_counts": [1, 1, 1, 1],
    }
    assert _active_fill_range(render, {"fill_end_step": 25}) == (2, 4)
    with pytest.raises(F2FillClipInputError, match="active_fill_range_invalid"):
        _active_fill_range(render, {"fill_end_step": 40})
    with pytest.raises(F2FillClipInputError, match="video_timing_invalid"):
        _active_fill_range(
            dict(render, video_timing_source_frame_steps=[10, 30, 20, 40]),
            {"fill_end_step": 25},
        )


def test_video_metadata_rejects_truncated_or_wrong_resolution_decode(monkeypatch, tmp_path):
    import fluidmix.f2_fill_clip_inputs as clip_inputs

    class FakeCapture:
        def __init__(self, frames):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == clip_inputs.cv2.CAP_PROP_FPS:
                return 10.0
            if property_id == clip_inputs.cv2.CAP_PROP_FOURCC:
                return float(sum(ord(char) << (8 * index) for index, char in enumerate("MJPG")))
            return 0.0

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            pass

    render = {
        "resolution": [640, 480],
        "video_playback_fps": 10.0,
        "video_codec": "MJPG",
        "video_container": "avi",
        "video_decodable": True,
        "video_decoded_frame_count": 2,
    }
    one_frame = [np.zeros((480, 640, 3), dtype=np.uint8)]
    monkeypatch.setattr(clip_inputs.cv2, "VideoCapture", lambda _: FakeCapture(one_frame))
    with pytest.raises(clip_inputs.F2FillClipInputError, match="video_decode_mismatch"):
        clip_inputs._validate_video_metadata(tmp_path / "video.avi", render)

    wrong_shape = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((479, 640, 3), dtype=np.uint8),
    ]
    monkeypatch.setattr(clip_inputs.cv2, "VideoCapture", lambda _: FakeCapture(wrong_shape))
    with pytest.raises(clip_inputs.F2FillClipInputError, match="video_decode_mismatch"):
        clip_inputs._validate_video_metadata(tmp_path / "video.avi", render)


@pytest.mark.parametrize(
    "field",
    ["oracle_path", "mask_source", "residue_ml", "paint_palette", "label"],
)
def test_safe_projection_rejects_private_or_oracle_fields(field):
    from fluidmix.f2_fill_clip_inputs import F2FillClipInputError, _assert_safe_projection

    with pytest.raises(F2FillClipInputError, match="forbidden_output_field"):
        _assert_safe_projection({"records": [{field: "not-safe"}]})

