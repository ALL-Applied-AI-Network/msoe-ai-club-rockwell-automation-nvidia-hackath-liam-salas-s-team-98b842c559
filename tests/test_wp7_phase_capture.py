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


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


_ROLES = {
    "run-test": {
        "ordered_wp1_composition_receipt": "wp7_ordered_test_video_v1",
        "ordered_wp1_review_receipt": (
            "fluidmix_wp7_ordered_test_video_final_output_review_v1"
        ),
        "ordered_wp1_run_receipt": "wp7_ordered_test_run_receipt_v1",
    },
    "run-real-check": {
        "video5_dataset_manifest": "video5_dataset_manifest_v1",
        "video5_recipe_metadata": "video5_recipe_metadata_v1",
        "video5_timing_metadata": "video5_timing_metadata_v1",
        "video5_transition_window_metadata": "video5_transition_window_metadata_v1",
    },
}


def _source_adapters(
    root: Path,
    *,
    phase: str,
    video: bytes,
    frames: list[np.ndarray],
    fills: list[dict[str, object]],
    transitions: list[dict[str, object]],
) -> list[dict[str, str]]:
    normalized_transitions = []
    for transition in transitions:
        start = int(transition["start_frame_index"])
        stop = int(transition["stop_frame_index"])
        normalized_transitions.append({
            **transition,
            "decoded_bgr_sha256": [
                hashlib.sha256(frame[..., ::-1].copy().tobytes()).hexdigest()
                for frame in frames[start:stop]
            ],
        })
    transition_ranges = [
        {key: transition[key] for key in (
            "segment_index", "start_frame_index", "stop_frame_index", "frame_count"
        )}
        for transition in normalized_transitions
    ]
    fill_windows = [
        {
            "fill_index": index,
            "case_id": fill["case_id"],
            "start_frame_index": normalized_transitions[index]["stop_frame_index"] - 1,
            "stop_frame_index": normalized_transitions[index + 1]["start_frame_index"],
            "frame_count": fill["frame_count"],
        }
        for index, fill in enumerate(fills)
    ]
    timing_sha = _canonical_hash({
        "fill_clips": fills,
        "transition_ranges": transition_ranges,
        "fill_windows": fill_windows,
    })
    sources = []
    for role, upstream_type in _ROLES[phase].items():
        upstream = root / f"upstream-{role}.json"
        if role == "ordered_wp1_composition_receipt":
            upstream_value = {
                "schema_version": 1,
                "artifact_type": upstream_type,
                "status": "pass",
                "output": {
                    "sha256": hashlib.sha256(video).hexdigest(),
                    "frame_count": len(frames),
                },
                "ordered_runs": [
                    {"run_id": "RUN-1", "case_ids": [fills[0]["case_id"]]},
                    {"run_id": "RUN-2", "case_ids": [fills[1]["case_id"]]},
                ],
            }
        elif role == "ordered_wp1_review_receipt":
            upstream_value = {
                "schema_version": 1,
                "artifact_type": upstream_type,
                "status": "pass",
                "verdict": "pass",
                "identity": {"output_mp4_sha256": hashlib.sha256(video).hexdigest()},
                "case_contract": {"count": len(fills), "unique": True},
                "output_media": {
                    "sha256": hashlib.sha256(video).hexdigest(),
                    "frame_count": len(frames),
                },
            }
        elif role == "ordered_wp1_run_receipt":
            bindings = []
            for index, fill in enumerate(fills, start=1):
                run = root / f"ordered-run-{index}.json"
                frame_count = len(frames) // 2
                _write_json(run, {
                    "artifact_type": "rendered_transport_run_sequence_v4",
                    "status": "pass",
                    "run_id": f"RUN-{index}",
                    "case_ids": [fill["case_id"]],
                    "output_video_sha256": f"{index:064x}",
                    "output_frame_count": frame_count,
                })
                bindings.append({
                    "path": str(run.absolute()),
                    "sha256": _sha(run),
                    "artifact_type": "rendered_transport_run_sequence_v4",
                    "run_id": f"RUN-{index}",
                    "case_ids": [fill["case_id"]],
                    "output_video_sha256": f"{index:064x}",
                    "output_frame_count": frame_count,
                })
            upstream_value = {
                "schema_version": 1,
                "artifact_type": upstream_type,
                "status": "pass",
                "run_receipts": bindings,
            }
        elif role == "video5_dataset_manifest":
            upstream_value = {
                "schema_version": 1, "artifact_type": upstream_type,
                "status": "pass", "video_sha256": hashlib.sha256(video).hexdigest(),
                "case_ids": [fill["case_id"] for fill in fills],
                "case_count": len(fills),
            }
        elif role == "video5_recipe_metadata":
            upstream_value = {
                "schema_version": 1, "artifact_type": upstream_type,
                "status": "pass", "video_sha256": hashlib.sha256(video).hexdigest(),
                "recipe_sequence_sha256": _canonical_hash(_sequence()),
                "case_ids": [fill["case_id"] for fill in fills],
            }
        elif role == "video5_timing_metadata":
            upstream_value = {
                "schema_version": 1, "artifact_type": upstream_type,
                "status": "pass", "video_sha256": hashlib.sha256(video).hexdigest(),
                "capture_timing_sha256": timing_sha,
                "decoded_frame_count": len(frames),
            }
        else:
            upstream_value = {
                "schema_version": 1, "artifact_type": upstream_type,
                "status": "pass", "video_sha256": hashlib.sha256(video).hexdigest(),
                "capture_timing_sha256": timing_sha,
                "case_ids": [fill["case_id"] for fill in fills],
            }
        _write_json(upstream, upstream_value)
        adapter = root / f"adapter-{role}.json"
        _write_json(adapter, {
            "schema_version": "wp7-capture-source-adapter-v1",
            "artifact_type": "wp7_capture_source_adapter",
            "phase": phase,
            "role": role,
            "video_sha256": hashlib.sha256(video).hexdigest(),
            "recipe_sequence_sha256": _canonical_hash(_sequence()),
            "ordered_case_ids": [str(fill["case_id"]) for fill in fills],
            "capture_timing_sha256": timing_sha,
            "upstream_artifact_type": upstream_type,
            "upstream_path": str(upstream.absolute()),
            "upstream_sha256": _sha(upstream),
            "status": "pass",
        })
        sources.append({
            "role": role,
            "path": str(adapter.absolute()),
            "sha256": _sha(adapter),
        })
    return sources


def _fixture(
    tmp_path: Path, *, preattested: bool = False, phase: str = "run-test"
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = [
        np.full((480, 640, 3), index, dtype=np.uint8)
        for index in range(8)
    ]
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
        "phase": phase,
        "capture_origin": "synthetic_camera" if phase == "run-test" else "real_camera",
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "fill_clips": [
            {"case_id": "CASE-1", "frame_count": 2},
            {"case_id": "CASE-2", "frame_count": 2},
        ],
        "encoded_transition_evidence": transitions,
        "source_provenance": _source_adapters(
            tmp_path,
            phase=phase,
            video=video,
            frames=frames,
            fills=[
                {"case_id": "CASE-1", "frame_count": 2},
                {"case_id": "CASE-2", "frame_count": 2},
            ],
            transitions=transitions,
        ),
    }
    plan_path = tmp_path / "capture-plan.json"
    plan_sha = _write_json(plan_path, plan)
    return plan_path, plan_sha, video, frames, Path(plan["source_provenance"][0]["path"])


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
    assert capture["encoded_transition_evidence"][0]["start_frame_index"] == 0
    assert capture["encoded_transition_evidence"][-1]["stop_frame_index"] == len(frames)
    assert all(
        len(row["decoded_bgr_sha256"]) == row["frame_count"]
        for row in capture["encoded_transition_evidence"]
    )
    assert result.capture_receipt == capture
    provenance = json.loads(provenance_out.read_text())
    assert provenance["plan_binding"]["raw_file_sha256"] == plan_sha
    assert provenance["capture_receipt_binding"]["raw_file_sha256"] == _sha(capture_out)
    assert {row["role"] for row in provenance["source_provenance"]} == set(
        _ROLES["run-test"]
    )
    assert provenance["source_provenance"][0] == {
        "role": "ordered_wp1_composition_receipt",
        "path": str(source.absolute()),
        "sha256": _sha(source),
    }


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


def test_structural_timing_binding_ignores_valid_optional_frame_attestations(
    tmp_path: Path,
) -> None:
    plain, _, _, _, _ = _fixture(tmp_path / "plain")
    attested, _, _, _, _ = _fixture(tmp_path / "attested", preattested=True)
    plain_adapter = json.loads(Path(
        json.loads(plain.read_text())["source_provenance"][0]["path"]
    ).read_text())
    attested_adapter = json.loads(Path(
        json.loads(attested.read_text())["source_provenance"][0]["path"]
    ).read_text())

    assert plain_adapter["capture_timing_sha256"] == attested_adapter[
        "capture_timing_sha256"
    ]


def test_capture_plan_rejects_oracle_semantic_source_role(tmp_path: Path) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"][0]["role"] = "oracle_semantic_labels"
    plan_sha = _write_json(plan_path, plan)

    with pytest.raises(PhaseCaptureError, match="exact phase role set"):
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


@pytest.mark.parametrize(
    ("phase", "origin"),
    [
        ("run-test", "synthetic_camera"),
        ("run-real-check", "real_camera"),
    ],
)
def test_capture_plan_accepts_only_phase_appropriate_safe_metadata_roles(
    tmp_path: Path, phase: str, origin: str
) -> None:
    from fluidmix.wp7_phase_capture import publish_phase_capture

    plan_path, plan_sha, video, frames, _ = _fixture(tmp_path, phase=phase)

    publish_phase_capture(
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        expected_phase=phase,
        expected_capture_origin=origin,
        video_bytes=video,
        recipe_sequence=_sequence(),
        decoded_rgb_frames=frames,
        capture_receipt_out=tmp_path / "capture.json",
        provenance_out=tmp_path / "provenance.json",
    )


@pytest.mark.parametrize("role", ["arbitrary_metadata", "video5_dataset_manifest"])
def test_run_test_rejects_arbitrary_and_real_check_metadata_roles(
    tmp_path: Path, role: str
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"][0]["role"] = role
    plan_sha = _write_json(plan_path, plan)
    with pytest.raises(PhaseCaptureError, match="exact phase role set"):
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


def test_capture_plan_requires_complete_phase_role_set_before_source_reads(
    tmp_path: Path,
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, source = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"].pop()
    plan_sha = _write_json(plan_path, plan)
    source.unlink()
    with pytest.raises(PhaseCaptureError, match="exact phase role set"):
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda adapter: adapter.clear(), "metadata identity"),
        (
            lambda adapter: adapter.__setitem__(
                "upstream_path", Path(str(adapter["upstream_path"])).name
            ),
            "absolute JSON metadata",
        ),
    ],
)
def test_allowed_role_rejects_unrelated_adapter_and_relative_upstream(
    tmp_path: Path, mutation, message: str
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, source = _fixture(tmp_path)
    adapter = json.loads(source.read_text())
    mutation(adapter)
    _write_json(source, adapter)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"][0]["sha256"] = _sha(source)
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


def test_allowed_composition_type_rejects_wrong_video_semantics(tmp_path: Path) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, source = _fixture(tmp_path)
    adapter = json.loads(source.read_text())
    upstream = Path(adapter["upstream_path"])
    upstream_value = json.loads(upstream.read_text())
    upstream_value["output"]["sha256"] = "0" * 64
    _write_json(upstream, upstream_value)
    adapter["upstream_sha256"] = _sha(upstream)
    _write_json(source, adapter)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"][0]["sha256"] = _sha(source)
    plan_sha = _write_json(plan_path, plan)

    with pytest.raises(PhaseCaptureError, match="composition identity"):
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


@pytest.mark.parametrize(
    ("source_name", "source_bytes", "message"),
    [
        ("not-metadata.png", b'{}\n', "absolute JSON metadata"),
        ("malformed.json", b"not JSON\n", "strict JSON metadata"),
    ],
)
def test_allowed_source_role_still_rejects_non_metadata_artifact(
    tmp_path: Path, source_name: str, source_bytes: bytes, message: str
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path)
    source = tmp_path / source_name
    source.write_bytes(source_bytes)
    plan = json.loads(plan_path.read_text())
    plan["source_provenance"][0].update({
        "path": str(source.absolute()),
        "sha256": _sha(source),
    })
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


def test_capture_plan_rejects_transition_coverage_inside_decoded_sequence(
    tmp_path: Path,
) -> None:
    from fluidmix.wp7_phase_capture import PhaseCaptureError, publish_phase_capture

    plan_path, _, video, frames, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    first = plan["encoded_transition_evidence"][0]
    first["start_frame_index"] = 1
    first["frame_count"] = 1
    final = plan["encoded_transition_evidence"][-1]
    final["stop_frame_index"] = len(frames) - 1
    final["frame_count"] = 1
    plan_sha = _write_json(plan_path, plan)

    with pytest.raises(PhaseCaptureError, match="full decoded sequence"):
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
    source_sha = _sha(source)
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
    plan_value["source_provenance"][0]["path"] = str((tmp_path / "directory.json"))
    (tmp_path / "directory.json").mkdir()
    rebound = _write_json(plan, plan_value)
    with pytest.raises(PhaseCaptureError, match="regular non-symlink"):
        publish_phase_capture(
            plan_path=plan, expected_plan_sha256=rebound,
            capture_receipt_out=tmp_path / "dir-capture.json",
            provenance_out=tmp_path / "dir-provenance.json", **common,
        )
    stale_value = json.loads(plan.read_text())
    stale_value["source_provenance"][0] = {
        "role": "ordered_wp1_composition_receipt",
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
        "role": "ordered_wp1_composition_receipt",
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
    assert _sha(source) == source_sha


@pytest.mark.parametrize(
    ("mutation", "expected_phase", "expected_origin", "video_override", "sequence", "message"),
    [
        (lambda plan: plan.__setitem__("phase", "run-real-check"), "run-test",
         "synthetic_camera", None, None, "phase/origin"),
        (lambda plan: plan.__setitem__("capture_origin", "real_camera"), "run-test",
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
