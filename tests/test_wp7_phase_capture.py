"""Synthetic contracts for generic WP7 phase capture plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _sequence() -> dict[str, object]:
    return {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "test",
        "split_version": "synthetic-v1",
        "runs": [{
            "run_id": "RUN-1",
            "vials": [
                {"case_id": "CASE-1"},
                {"case_id": "CASE-2"},
            ],
        }],
    }


def _fixture(tmp_path: Path, *, preattested: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = [
        np.full((480, 640, 3), index, dtype=np.uint8)
        for index in range(8)
    ]
    source = tmp_path / "timing-source.json"
    source.write_text("synthetic timing source\n")
    transitions = [
        {"segment_index": 0, "start_frame_index": 0, "stop_frame_index": 2,
         "frame_count": 2, "decoded_bgr_sha256": None},
        {"segment_index": 1, "start_frame_index": 3, "stop_frame_index": 5,
         "frame_count": 2, "decoded_bgr_sha256": None},
        {"segment_index": 2, "start_frame_index": 6, "stop_frame_index": 8,
         "frame_count": 2, "decoded_bgr_sha256": None},
    ]
    if preattested:
        for transition in transitions:
            start = transition["start_frame_index"]
            stop = transition["stop_frame_index"]
            transition["decoded_bgr_sha256"] = [
                hashlib.sha256(frame[..., ::-1].copy().tobytes()).hexdigest()
                for frame in frames[start:stop]
            ]
    video = b"synthetic claimed video"
    plan = {
        "schema_version": "wp7-phase-capture-plan-v1",
        "artifact_type": "wp7_phase_capture_plan",
        "phase": "run-test",
        "capture_origin": "synthetic_camera",
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "fill_clips": [
            {"case_id": "CASE-1", "frame_count": 2},
            {"case_id": "CASE-2", "frame_count": 2},
        ],
        "encoded_transition_evidence": transitions,
        "source_provenance": [{
            "role": "synthetic_timing_source",
            "path": str(source.absolute()),
            "sha256": _sha(source),
        }],
    }
    plan_path = tmp_path / "capture-plan.json"
    plan_sha = _write_json(plan_path, plan)
    return plan_path, plan_sha, video, frames, source


def test_computed_hash_plan_publishes_exact_runtime_capture_and_provenance(
    tmp_path: Path,
) -> None:
    from fluidmix.wp7_phase_capture import publish_phase_capture

    plan, plan_sha, video, frames, source = _fixture(tmp_path)
    capture_out = tmp_path / "capture.json"
    provenance_out = tmp_path / "capture-provenance.json"
    result = publish_phase_capture(
        plan_path=plan,
        expected_plan_sha256=plan_sha,
        expected_phase="run-test",
        expected_capture_origin="synthetic_camera",
        video_bytes=video,
        recipe_sequence=_sequence(),
        decoded_rgb_frames=frames,
        capture_receipt_out=capture_out,
        provenance_out=provenance_out,
    )

    capture = json.loads(capture_out.read_text())
    assert set(capture) == {
        "schema_version", "artifact_type", "capture_origin", "video_sha256",
        "receipt_sha256", "encoded_transition_evidence", "fill_clips",
    }
    assert len(capture["encoded_transition_evidence"]) == 3
    assert all(
        len(row["decoded_bgr_sha256"]) == row["frame_count"]
        for row in capture["encoded_transition_evidence"]
    )
    assert result.capture_receipt == capture
    provenance = json.loads(provenance_out.read_text())
    assert provenance["plan_binding"]["raw_file_sha256"] == plan_sha
    assert provenance["capture_receipt_binding"]["raw_file_sha256"] == _sha(capture_out)
    assert provenance["source_provenance"] == [{
        "role": "synthetic_timing_source",
        "path": str(source.absolute()),
        "sha256": _sha(source),
    }]


def _publish(tmp_path: Path, *, preattested: bool = False):
    from fluidmix.wp7_phase_capture import publish_phase_capture

    plan, plan_sha, video, frames, source = _fixture(
        tmp_path, preattested=preattested
    )
    capture = tmp_path / "capture.json"
    provenance = tmp_path / "capture-provenance.json"
    result = publish_phase_capture(
        plan_path=plan,
        expected_plan_sha256=plan_sha,
        expected_phase="run-test",
        expected_capture_origin="synthetic_camera",
        video_bytes=video,
        recipe_sequence=_sequence(),
        decoded_rgb_frames=frames,
        capture_receipt_out=capture,
        provenance_out=provenance,
    )
    return result, plan, capture, provenance, source


def test_pre_attested_hashes_are_recomputed_and_verified(tmp_path: Path) -> None:
    result, _, _, _, _ = _publish(tmp_path, preattested=True)
    assert result.capture_receipt["encoded_transition_evidence"][0][
        "decoded_bgr_sha256"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan["fill_clips"].reverse(), "case order"),
        (lambda plan: plan["fill_clips"].pop(), r"N\+1/N"),
        (
            lambda plan: plan["encoded_transition_evidence"][0].__setitem__(
                "stop_frame_index", 9
            ),
            "range",
        ),
        (
            lambda plan: plan["encoded_transition_evidence"][0][
                "decoded_bgr_sha256"
            ].__setitem__(0, "0" * 64),
            "drifted",
        ),
    ],
)
def test_plan_rejects_order_count_range_and_attestation_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path, preattested=True)
    plan = json.loads(plan_path.read_text())
    mutation(plan)
    plan_sha = _write_json(plan_path, plan)
    with pytest.raises(PhaseCaptureError, match=message):
        publish_phase_capture(
            plan_path=plan_path,
            expected_plan_sha256=plan_sha,
            expected_phase="run-test",
            expected_capture_origin="synthetic_camera",
            video_bytes=video,
            recipe_sequence=_sequence(),
            decoded_rgb_frames=frames,
            capture_receipt_out=tmp_path / "capture.json",
            provenance_out=tmp_path / "provenance.json",
        )


@pytest.mark.parametrize("role", ["capture", "provenance"])
def test_existing_output_rejects_before_plan_read(tmp_path: Path, role: str) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    capture = tmp_path / "capture.json"
    provenance = tmp_path / "provenance.json"
    occupied = capture if role == "capture" else provenance
    occupied.write_text("owner data")
    with pytest.raises(PhaseCaptureError, match="output already exists"):
        publish_phase_capture(
            plan_path=tmp_path / "sealed-plan-must-not-be-read",
            expected_plan_sha256="0" * 64,
            expected_phase="run-test",
            expected_capture_origin="synthetic_camera",
            video_bytes=b"unread",
            recipe_sequence=_sequence(),
            decoded_rgb_frames=[],
            capture_receipt_out=capture,
            provenance_out=provenance,
        )
    assert occupied.read_text() == "owner data"


def test_plan_and_sources_reject_symlink_nonregular_and_stale_hash(
    tmp_path: Path,
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan, plan_sha, video, frames, source = _fixture(tmp_path / "base")
    common = {
        "expected_phase": "run-test",
        "expected_capture_origin": "synthetic_camera",
        "video_bytes": video,
        "recipe_sequence": _sequence(),
        "decoded_rgb_frames": frames,
    }
    with pytest.raises(PhaseCaptureError, match="raw-file SHA-256 mismatch"):
        publish_phase_capture(
            plan_path=plan, expected_plan_sha256="0" * 64,
            capture_receipt_out=tmp_path / "stale-capture.json",
            provenance_out=tmp_path / "stale-provenance.json", **common,
        )
    link = tmp_path / "plan-link.json"
    link.symlink_to(plan)
    with pytest.raises(PhaseCaptureError, match="regular non-symlink"):
        publish_phase_capture(
            plan_path=link, expected_plan_sha256=plan_sha,
            capture_receipt_out=tmp_path / "link-capture.json",
            provenance_out=tmp_path / "link-provenance.json", **common,
        )
    plan_value = json.loads(plan.read_text())
    plan_value["source_provenance"][0]["path"] = str((tmp_path / "directory"))
    (tmp_path / "directory").mkdir()
    rebound = _write_json(plan, plan_value)
    with pytest.raises(PhaseCaptureError, match="regular non-symlink"):
        publish_phase_capture(
            plan_path=plan, expected_plan_sha256=rebound,
            capture_receipt_out=tmp_path / "dir-capture.json",
            provenance_out=tmp_path / "dir-provenance.json", **common,
        )
    stale_value = json.loads(plan.read_text())
    stale_value["source_provenance"][0] = {
        "role": "synthetic_timing_source",
        "path": str(source.absolute()),
        "sha256": "0" * 64,
    }
    stale_sha = _write_json(plan, stale_value)
    with pytest.raises(PhaseCaptureError, match="raw-file SHA-256 mismatch"):
        publish_phase_capture(
            plan_path=plan, expected_plan_sha256=stale_sha,
            capture_receipt_out=tmp_path / "source-stale-capture.json",
            provenance_out=tmp_path / "source-stale-provenance.json", **common,
        )
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(source)
    link_value = json.loads(plan.read_text())
    link_value["source_provenance"][0] = {
        "role": "synthetic_timing_source",
        "path": str(source_link.absolute()),
        "sha256": _sha(source),
    }
    source_link_sha = _write_json(plan, link_value)
    with pytest.raises(PhaseCaptureError, match="regular non-symlink"):
        publish_phase_capture(
            plan_path=plan, expected_plan_sha256=source_link_sha,
            capture_receipt_out=tmp_path / "source-link-capture.json",
            provenance_out=tmp_path / "source-link-provenance.json", **common,
        )
    assert source.read_text() == "synthetic timing source\n"


@pytest.mark.parametrize(
    ("mutation", "expected_phase", "expected_origin", "video_override", "sequence", "message"),
    [
        (lambda plan: plan.__setitem__("phase", "run-real-check"), "run-test",
         "synthetic_camera", None, None, "phase/origin"),
        (lambda plan: None, "run-test", "real_camera", None, None, "phase/origin"),
        (lambda plan: None, "run-test", "synthetic_camera", b"wrong video", None,
         "video SHA-256"),
        (lambda plan: None, "run-test", "synthetic_camera", None,
         {"runs": [{"vials": [{"case_id": "CASE-1"}]}]}, "case order"),
    ],
)
def test_phase_origin_video_and_recipe_denominator_are_exact(
    tmp_path: Path,
    mutation,
    expected_phase: str,
    expected_origin: str,
    video_override: bytes | None,
    sequence: dict[str, object] | None,
    message: str,
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    mutation(plan)
    plan_sha = _write_json(plan_path, plan)
    with pytest.raises(PhaseCaptureError, match=message):
        publish_phase_capture(
            plan_path=plan_path,
            expected_plan_sha256=plan_sha,
            expected_phase=expected_phase,
            expected_capture_origin=expected_origin,
            video_bytes=video if video_override is None else video_override,
            recipe_sequence=_sequence() if sequence is None else sequence,
            decoded_rgb_frames=frames,
            capture_receipt_out=tmp_path / "capture.json",
            provenance_out=tmp_path / "provenance.json",
        )
