from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.transport.vial_transport import (  # noqa: E402
    TransportSpec,
    _fill_masked_rows,
    compose_transport_frames,
    extract_fillcam_mover_masks,
    transport_offsets,
)
from twin.transport.transport_stage import (  # noqa: E402
    PILOT_VISUAL_REVIEW_CHECKS,
    validate_pilot_visual_review,
)
from cli.twin.render.compose_run_sequence_videos import compose_case_video, compose_run_video  # noqa: E402
import cli.twin.render.compose_run_sequence_videos as run_composer  # noqa: E402


def _synthetic_frame():
    frame = np.full((480, 640, 3), (12, 34, 56), dtype=np.uint8)
    outgoing = np.zeros((480, 640), dtype=np.uint8)
    incoming = np.zeros((480, 640), dtype=np.uint8)
    outgoing[180:260, 240:280] = 255
    incoming[180:260, 430:470] = 255
    frame[outgoing > 0] = (220, 30, 20)
    frame[incoming > 0] = (20, 220, 30)
    return frame, outgoing, incoming


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_rejected_2d_transport_compositor_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="2D post-fill compositor is retired"):
        compose_case_video(
            case_output=tmp_path / "case-output",
            output_mp4=tmp_path / "transport" / "vial.mp4",
            spec=TransportSpec(frame_count=3, fps=10.0),
        )

    assert not (tmp_path / "transport").exists()


@pytest.mark.parametrize("mutation", ["caller-path", "same-path-bytes", "build-binding"])
def test_supplied_fill_binding_rejects_video_drift_before_partial_output(
    tmp_path: Path, mutation: str
) -> None:
    supplied_path = tmp_path / "supplied.mp4"
    supplied_path.write_bytes(b"supplied-fill")
    alternate_path = tmp_path / "alternate.mp4"
    alternate_path.write_bytes(b"alternate-fill")
    supplied = {
        "case_id": "CASE_0000000000000001",
        "path": str(supplied_path.resolve()),
        "sha256": _sha256(supplied_path),
    }
    endpoints = {
        "accepted_start": None,
        "accepted_end": {
            "video_path": str(supplied_path.resolve()),
            "video_sha256": _sha256(supplied_path),
        },
    }
    if mutation == "caller-path":
        supplied["path"] = str(alternate_path.resolve())
        supplied["sha256"] = _sha256(alternate_path)
    elif mutation == "same-path-bytes":
        supplied_path.write_bytes(b"changed-after-decode")
    else:
        endpoints["accepted_end"] = {
            "video_path": str(alternate_path.resolve()),
            "video_sha256": _sha256(alternate_path),
        }

    with pytest.raises(ValueError, match="differs from supplied fill video"):
        run_composer._validate_supplied_fill_bindings(
            endpoints,
            mode="entry",
            expected_start_fill=None,
            expected_end_fill=supplied,
        )

    assert not (tmp_path / "run.partial.mp4").exists()


def test_run_compositor_uses_entry_between_exit_endpoint_substitution(
    tmp_path: Path,
):
    case_ids = ["CASE_0000000000000001", "CASE_0000000000000002"]
    owner_thread_id = "test-owner"
    implementation = {"sha256": "e" * 64}
    pilot_segment = tmp_path / "pilot-build"
    pilot_build_path = pilot_segment / "transport-build-receipt.json"
    _write_json(
        pilot_build_path,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
            "status": "pass",
            "mode": "between",
            "owner_thread_id": owner_thread_id,
            "transport_implementation": implementation,
        },
    )
    pilot_output = tmp_path / "pilot-render"
    contact_sheet = pilot_output / "contact-sheet.png"
    contact_sheet.parent.mkdir(parents=True)
    contact_sheet.write_bytes(b"pilot-contact-sheet")
    pilot_receipt = pilot_output / "transport-render-receipt.json"
    _write_json(
        pilot_receipt,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_render_receipt_v1",
            "status": "pilot",
            "mode": "between",
            "owner_thread_id": owner_thread_id,
            "segment_dir": str(pilot_segment),
            "build_receipt_sha256": _sha256(pilot_build_path),
            "state_indices": [0, 6, 12],
            "source_time_code": 1,
            "physics_steps": 0,
            "physics_callbacks": 0,
            "rendered_states_byte_unique": True,
            "pilot_visual_review": None,
            "contact_sheet_path": contact_sheet.name,
            "contact_sheet_sha256": _sha256(contact_sheet),
            "checker_evaluation": {
                "schema_version": 1,
                "artifact_type": "fluidmix_transport_stage_checker_v1",
                "status": "pass",
                "pilot": True,
                "manual_visual_review_required": True,
                "pilot_visual_review_status": "required",
                "automatic_acceptance_gates": {
                    "evidence_closure_exact": True,
                    "runtime_partition_exact": True,
                    "render_mode_transitions_exact": True,
                    "camera_and_lighting_unchanged": True,
                    "source_time_one_and_zero_physics": True,
                    "instance_mappings_and_masks_valid": True,
                    "background_sentinels_within_2": True,
                    "mode_occupancy_valid": True,
                    "start_seam_passed": None,
                    "end_seam_passed": None,
                    "packaged_sequence_byte_unique": None,
                    "pilot_visual_review_accepted": None,
                },
            },
        },
    )
    review_path = tmp_path / "pilot-review.json"
    _write_json(
        review_path,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_pilot_visual_review_v1",
            "status": "accepted",
            "owner_thread_id": owner_thread_id,
            "transport_implementation_sha256": implementation["sha256"],
            "reviewer_id": "independent-reviewer",
            "independent_from_implementation": True,
            "reviewed_utc": "2026-10-28T18:00:00Z",
            "decision_reference": "test-review",
            "pilot_render_receipt": {
                "path": str(pilot_receipt),
                "sha256": _sha256(pilot_receipt),
            },
            "contact_sheet": {
                "path": str(contact_sheet),
                "sha256": _sha256(contact_sheet),
            },
            "checks": {key: True for key in PILOT_VISUAL_REVIEW_CHECKS},
        },
    )
    visual_review = validate_pilot_visual_review(
        review_path,
        transport_implementation_sha256=str(implementation["sha256"]),
        owner_thread_id=owner_thread_id,
    )
    fill_videos = []
    for case_index, case_id in enumerate(case_ids):
        path = tmp_path / case_id / "fill.mp4"
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
        )
        assert writer.isOpened()
        fill_values = (0, 255)
        writer.write(np.full((480, 640, 3), fill_values[0], dtype=np.uint8))
        writer.write(np.full((480, 640, 3), fill_values[1], dtype=np.uint8))
        writer.release()
        fill_videos.append(path)

    endpoint_cases = (
        ("entry", None, case_ids[0]),
        ("between", case_ids[0], case_ids[1]),
        ("exit", case_ids[1], None),
    )
    render_dirs = []
    for segment_index, (mode, outgoing, next_case) in enumerate(endpoint_cases):
        segment = tmp_path / "builds" / f"segment-{segment_index:02d}-{mode}"
        render = tmp_path / "renders" / f"segment-{segment_index:02d}-{mode}"
        render.mkdir(parents=True)
        build_path = segment / "transport-build-receipt.json"
        _write_json(
            build_path,
            {
                "schema_version": 1,
                "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
                "status": "pass",
                "mode": mode,
                "owner_thread_id": owner_thread_id,
                "transport_implementation": implementation,
                "adjacent_endpoints": {
                    "outgoing_case_output": (
                        str(tmp_path / outgoing / "app-output") if outgoing else None
                    ),
                    "next_case_output": (
                        str(tmp_path / next_case / "app-output") if next_case else None
                    ),
                    "accepted_start": (
                        {
                            "video_path": str(
                                fill_videos[case_ids.index(outgoing)].resolve()
                            ),
                            "video_sha256": _sha256(
                                fill_videos[case_ids.index(outgoing)]
                            ),
                        }
                        if outgoing
                        else None
                    ),
                    "accepted_end": (
                        {
                            "video_path": str(
                                fill_videos[case_ids.index(next_case)].resolve()
                            ),
                            "video_sha256": _sha256(
                                fill_videos[case_ids.index(next_case)]
                            ),
                        }
                        if next_case
                        else None
                    ),
                },
            },
        )
        captures = []
        for state_index in range(13):
            rgb_path = render / f"state-{state_index:02d}.png"
            assert cv2.imwrite(
                str(rgb_path),
                np.full(
                    (480, 640, 3),
                        30 + segment_index * 20 + state_index * 10,
                    dtype=np.uint8,
                ),
            )
            captures.append(
                {
                    "state_index": state_index,
                    "rgb_path": rgb_path.name,
                    "rgb_sha256": _sha256(rgb_path),
                }
            )
        _write_json(
            render / "transport-render-receipt.json",
            {
                "schema_version": 1,
                "artifact_type": "fluidmix_transport_stage_render_receipt_v1",
                "status": "candidate",
                "mode": mode,
                "owner_thread_id": owner_thread_id,
                "state_indices": list(range(13)),
                "rendered_states_byte_unique": True,
                "segment_dir": str(segment),
                "build_receipt_sha256": _sha256(build_path),
                "captures": captures,
                "pilot_visual_review": visual_review,
                "checker_evaluation": {
                    "schema_version": 1,
                    "artifact_type": "fluidmix_transport_stage_checker_v1",
                    "status": "pass",
                    "pilot": False,
                    "manual_visual_review_required": False,
                    "pilot_visual_review_status": "accepted",
                    "automatic_acceptance_gates": {
                        "all_reviewed_gates": True,
                        "pilot_visual_review_accepted": True,
                    },
                },
            },
        )
        render_dirs.append(render)

    output = tmp_path / "package" / "run.mp4"
    receipt = compose_run_video(
        run_id="RUN-TRAIN-01",
        case_ids=case_ids,
        fill_videos=fill_videos,
        transport_render_dirs=render_dirs,
        output_mp4=output,
    )

    assert receipt["transport_segment_count"] == 3
    assert receipt["output_frame_count"] == 39
    assert receipt["run_id"] == "RUN-TRAIN-01"
    assert receipt["encoded_transition_sequences_byte_unique"] is True
    assert receipt["encoded_transition_seams_passed"] is True
    assert receipt["transport_fill_video_bindings"][0] == {
        "segment_index": 0,
        "mode": "entry",
        "accepted_start_fill": None,
        "accepted_end_fill": {
            "case_id": case_ids[0],
            "path": str(fill_videos[0].resolve()),
            "sha256": _sha256(fill_videos[0]),
        },
    }
    assert receipt["transport_fill_video_bindings"][1]["accepted_start_fill"] == {
        "case_id": case_ids[0],
        "path": str(fill_videos[0].resolve()),
        "sha256": _sha256(fill_videos[0]),
    }
    assert receipt["transport_fill_video_bindings"][1]["accepted_end_fill"] == {
        "case_id": case_ids[1],
        "path": str(fill_videos[1].resolve()),
        "sha256": _sha256(fill_videos[1]),
    }
    assert receipt["transport_fill_video_bindings"][2]["accepted_start_fill"] == {
        "case_id": case_ids[1],
        "path": str(fill_videos[1].resolve()),
        "sha256": _sha256(fill_videos[1]),
    }
    assert receipt["transport_fill_video_bindings"][2]["accepted_end_fill"] is None
    assert set(receipt["transport_segments"][0]) == {
        "render_dir",
        "receipt_path",
        "receipt_sha256",
        "mode",
    }
    assert [row["frame_count"] for row in receipt["encoded_transition_evidence"]] == [
        13,
        13,
        13,
    ]
    assert output.is_file()
    assert json.loads(output.with_suffix(".run.json").read_text())["run_id"] == (
        "RUN-TRAIN-01"
    )


def test_rowwise_background_fill_preserves_unmasked_pixels_and_horizontal_bands():
    frame = np.empty((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        frame[y] = (y % 251, (2 * y) % 251, (3 * y) % 251)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:420, 250:390] = 255
    obscured = frame.copy()
    obscured[mask > 0] = (0, 0, 0)

    filled = _fill_masked_rows(obscured, mask)

    assert np.array_equal(filled[mask == 0], obscured[mask == 0])
    assert np.array_equal(filled[mask > 0], frame[mask > 0])


def test_transport_moves_only_foregrounds_over_fixed_background():
    frame, outgoing, incoming = _synthetic_frame()
    spec = TransportSpec(
        frame_count=11,
        fps=10.0,
        outgoing_shift_px=-300,
        incoming_shift_px=-190,
        inpaint_radius_px=3.0,
        feather_radius_px=0,
    )

    frames = compose_transport_frames(
        frame,
        outgoing_mask=outgoing,
        incoming_mask=incoming,
        spec=spec,
    )

    assert len(frames) == 11
    assert np.array_equal(frames[0], frame)
    assert all(tuple(row[10, 10]) == (12, 34, 56) for row in frames)
    assert tuple(frames[-1][200, 250]) == (20, 220, 30)
    assert not np.any(np.all(frames[-1] == (220, 30, 20), axis=2))
    assert np.max(
        np.abs(frames[-1][200, 450].astype(int) - np.asarray((12, 34, 56)))
    ) <= 8
    hashes = {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in frames
    }
    assert len(hashes) == len(frames)


def test_transport_offsets_are_monotonic_and_hit_exact_endpoints():
    spec = TransportSpec(
        frame_count=9,
        outgoing_shift_px=-360,
        incoming_shift_px=-180,
    )
    offsets = transport_offsets(spec)

    assert offsets[0] == (0, 0)
    assert offsets[-1] == (-360, -180)
    assert all(
        later[0] <= earlier[0] and later[1] <= earlier[1]
        for earlier, later in zip(offsets, offsets[1:])
    )
    assert len(set(offsets)) == 9


def test_fillcam_extraction_keeps_liquid_as_outgoing_foreground():
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    frame[210:470, 250:395] = 40
    frame[215:470, 415:575] = 70
    liquid = np.zeros((480, 640), dtype=np.uint8)
    liquid[280:420, 290:345] = 255

    outgoing, incoming = extract_fillcam_mover_masks(frame, liquid)

    assert outgoing.shape == liquid.shape
    assert incoming.shape == liquid.shape
    assert outgoing.dtype == np.uint8
    assert incoming.dtype == np.uint8
    assert np.all(outgoing[liquid > 0] == 255)
    assert outgoing[250, 320] == 255
    assert incoming[300, 480] == 255
    assert np.count_nonzero(outgoing) > np.count_nonzero(liquid)
    assert np.count_nonzero(incoming) > 0
    assert np.count_nonzero((outgoing > 0) & (incoming > 0)) == 0


def test_fillcam_extraction_rejects_dark_camera_background_bands():
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    frame[45:125] = 5
    frame[185:235, 260:400] = 35
    frame[365:480, 260:400] = 70
    frame[185:235, 405:585] = 35
    frame[365:480, 405:585] = 70
    liquid = np.zeros((480, 640), dtype=np.uint8)
    liquid[285:410, 295:350] = 255

    outgoing, incoming = extract_fillcam_mover_masks(frame, liquid)

    assert outgoing[100, 385] == 0
    assert incoming[100, 550] == 0
    assert np.count_nonzero(outgoing) < 65_000
    assert np.count_nonzero(incoming) < 68_000


def test_transport_rejects_bad_frame_or_overlapping_masks():
    frame, outgoing, incoming = _synthetic_frame()
    with pytest.raises(ValueError, match="640x480"):
        compose_transport_frames(
            frame[:400],
            outgoing_mask=outgoing[:400],
            incoming_mask=incoming[:400],
        )
    incoming[200, 250] = 255
    with pytest.raises(ValueError, match="overlap"):
        compose_transport_frames(
            frame,
            outgoing_mask=outgoing,
            incoming_mask=incoming,
        )
