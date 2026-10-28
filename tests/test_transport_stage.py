from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.transport_stage import (  # noqa: E402
    ALL_STATE_INDICES,
    CAMERA_AND_LIGHT_PATHS,
    ENTRY_MODE,
    EXIT_MODE,
    INCOMING_LABEL_SOURCE_PATHS,
    INCOMING_PRIM_PATHS,
    LEGACY_ANIMATED_PATHS,
    OUTGOING_PRIM_PATHS,
    PILOT_VISUAL_REVIEW_CHECKS,
    QUEUED_PRIM_PATHS,
    QUEUE_TAIL_SOURCE_PATHS,
    SOURCE_CURVE_APPROVED_SHA256,
    SOURCE_CURVE_CANONICAL_13_SHA256,
    SOURCE_CURVE_CANONICAL_37_SHA256,
    SOURCE_CURVE_DELTA_37_SHA256,
    SOURCE_CURVE_SAMPLES,
    SOURCE_LAYER_HASHES,
    STATE_COUNT,
    TRANSPORT_FPS,
    BETWEEN_MODE,
    build_run_transport_layout,
    build_state_plan,
    build_transport_layers,
    capture_evidence_closure,
    capture_transport_implementation,
    freeze_runtime_partition,
    package_state_indices,
    validate_evidence_closure,
    validate_runtime_partition,
)
from render_transport_stage import (  # noqa: E402
    PILOT_STATE_INDICES,
    run_transport_stage,
)
from build_transport_stage import _load_accepted_first_frame_endpoint  # noqa: E402
from check_transport_stage import _validate_evidence_closure_shape  # noqa: E402
import check_transport_stage as transport_checker  # noqa: E402


OWNER_THREAD_ID = "01a18128-8dc4-7780-ace1-4d816b49d176"
IMPLEMENTATION_SHA256 = "f" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_next_endpoint_loader_consumes_only_receipt_and_video(tmp_path: Path) -> None:
    case = tmp_path / "app-output"
    video = _write(case / "three-pour.avi", b"accepted-video")
    receipt = {
        "schema_version": 20,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "pass",
        "scene_contract": {
            "source_layers_unchanged": True,
            "source_layer_records": [
                {
                    "identifier": "fill_station.usda",
                    "resolved_path": "/source/fill_station.usda",
                    "bytes": 906,
                    "sha256": SOURCE_LAYER_HASHES["fill_station.usda"],
                },
                {
                    "identifier": "geometry/fill_station.usdc",
                    "resolved_path": "/source/geometry/fill_station.usdc",
                    "bytes": 26_464_568,
                    "sha256": SOURCE_LAYER_HASHES["geometry/fill_station.usdc"],
                },
            ],
        },
        "render": {
            "video_path": video.name,
            "video_sha256": _sha256(video),
            "video_decodable": True,
            "video_decoded_frame_count": 12,
        },
    }
    _write_json(case / "receipt.json", receipt)
    # A next fill's snapshot is deliberately absent: only its accepted first video frame
    # is consumed by the between-mode endpoint substitution.

    endpoint = _load_accepted_first_frame_endpoint(case)

    assert endpoint["first_pre_pour"]["video_path"] == str(video.resolve())
    assert endpoint["dependencies"] == [
        (case / "receipt.json").resolve(),
        video.resolve(),
    ]


def test_frozen_source_curve_and_exact_mover_sets() -> None:
    assert SOURCE_CURVE_APPROVED_SHA256 == (
        "c340e573d8b225326176a2cd3523ea2b0c19895c7d1187e0a67954a8a485a317"
    )
    assert SOURCE_CURVE_CANONICAL_37_SHA256 == (
        "58ac18e079d9c6f99280338046584b21772eea475eac976d9a83d75d15597625"
    )
    assert SOURCE_CURVE_DELTA_37_SHA256 == (
        "621410b9ece7875fc0a0c7e5ef6feb3e7a34d521a6f6490bbf454f578702e528"
    )
    assert SOURCE_CURVE_CANONICAL_13_SHA256 == (
        "aed5b56bcd9443684f376db70272a25940b4715051d047b0a98cf64e06b2a613"
    )
    assert [row.time_code for row in SOURCE_CURVE_SAMPLES] == list(range(180, 217, 3))
    assert [row.progress for row in SOURCE_CURVE_SAMPLES] == [
        0.0,
        0.01967620849609375,
        0.07407379150390625,
        0.15625,
        0.2592582702636719,
        0.3761558532714844,
        0.5,
        0.6238422393798828,
        0.7407398223876953,
        0.84375,
        0.9259243011474609,
        0.9803256988525391,
        1.0,
    ]
    assert STATE_COUNT == 13
    assert TRANSPORT_FPS == 10
    assert ALL_STATE_INDICES == tuple(range(13))
    assert len(OUTGOING_PRIM_PATHS) == 57
    assert len(INCOMING_PRIM_PATHS) == 27
    assert len(QUEUED_PRIM_PATHS) == 27
    assert len(INCOMING_LABEL_SOURCE_PATHS) == 29
    assert QUEUE_TAIL_SOURCE_PATHS == tuple(sorted(QUEUED_PRIM_PATHS))
    assert SOURCE_LAYER_HASHES == {
        "fill_station.usda": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
        "geometry/fill_station.usdc": (
            "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
        ),
    }


@pytest.mark.parametrize("mode", [ENTRY_MODE, BETWEEN_MODE, EXIT_MODE])
def test_state_plan_is_source_time_one_and_hits_exact_mode_endpoints(mode: str) -> None:
    plan = build_state_plan(mode)

    assert len(plan) == 13
    assert [row["state_index"] for row in plan] == list(range(13))
    assert [row["source_time_code"] for row in plan] == [1] * 13
    assert [row["curve_time_code"] for row in plan] == list(range(180, 217, 3))
    assert all(row["physics_steps"] == 0 for row in plan)
    assert plan[0]["progress"] == 0.0
    assert plan[-1]["progress"] == 1.0
    assert all(
        plan[index]["progress"] < plan[index + 1]["progress"]
        for index in range(12)
    )

    if mode == ENTRY_MODE:
        assert plan[-1]["world_delta_y_m_by_group"] == {
            "incoming": 0.0848413319824885,
            "queued": 0.186441330754633,
            "queue_tail": 0.1016,
        }
        assert plan[-1]["outgoing_visibility"] == "invisible"
    elif mode == BETWEEN_MODE:
        assert plan[-1]["world_delta_y_m_by_group"] == {
            "outgoing": 0.19,
            "incoming": 0.0848413319824885,
            "queued": 0.186441330754633,
            "queue_tail": 0.1016,
        }
        assert plan[-1]["outgoing_visibility"] == "inherited"
    else:
        assert plan[-1]["world_delta_y_m_by_group"] == {"outgoing": 0.19}
        assert plan[-1]["incoming_label_root_local_y"] is None
        assert plan[-1]["queue_tail_root_local_y"] is None


def test_packaging_endpoint_substitution_and_wp1_wp1b_layouts() -> None:
    assert package_state_indices(ENTRY_MODE) == {
        "accepted_start_endpoint": None,
        "rendered_state_indices": list(range(12)),
        "accepted_end_endpoint": "next_fill_first_pre_pour",
    }
    assert package_state_indices(BETWEEN_MODE) == {
        "accepted_start_endpoint": "outgoing_fill_final",
        "rendered_state_indices": list(range(1, 12)),
        "accepted_end_endpoint": "next_fill_first_pre_pour",
    }
    assert package_state_indices(EXIT_MODE) == {
        "accepted_start_endpoint": "outgoing_fill_final",
        "rendered_state_indices": list(range(1, 13)),
        "accepted_end_endpoint": None,
    }
    layout = build_run_transport_layout([f"CASE_{index:016x}" for index in range(9)])
    assert [row["mode"] for row in layout] == [ENTRY_MODE] + [BETWEEN_MODE] * 8 + [EXIT_MODE]
    assert len(layout) == 10
    assert layout[0]["next_case_id"] == "CASE_0000000000000000"
    assert layout[-1]["outgoing_case_id"] == "CASE_0000000000000008"

    wp1b_layout = build_run_transport_layout(
        [f"CASE_{index:016x}" for index in range(3)]
    )
    assert [row["mode"] for row in wp1b_layout] == [
        ENTRY_MODE,
        BETWEEN_MODE,
        BETWEEN_MODE,
        EXIT_MODE,
    ]
    assert len(wp1b_layout) == 4


def test_start_seam_uses_actual_last_decoded_fill_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_png = np.full((480, 640, 3), 11, dtype=np.uint8)
    decoded_frames = [
        np.full((480, 640, 3), 22, dtype=np.uint8),
        np.full((480, 640, 3), 33, dtype=np.uint8),
    ]
    monkeypatch.setattr(
        transport_checker, "_load_rgb", lambda *_args, **_kwargs: accepted_png
    )
    monkeypatch.setattr(
        transport_checker, "_decode_video", lambda *_args, **_kwargs: decoded_frames
    )

    start, video, end, end_video = transport_checker._accepted_endpoint_evidence(
        {
            "adjacent_endpoints": {
                "accepted_start": {
                    "path": "/accepted/final.png",
                    "sha256": "a" * 64,
                    "video_path": "/accepted/fill.mp4",
                    "video_sha256": "b" * 64,
                    "frame_index": -1,
                },
                "accepted_end": None,
            }
        },
        EXIT_MODE,
    )

    assert start is decoded_frames[-1]
    assert video is decoded_frames
    assert end is None
    assert end_video is None


def test_builder_materializes_exact_default_valued_layers_without_camera_or_time_samples(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "source" / "fill_station.usda", b"source-usda")
    geometry = _write(tmp_path / "source" / "geometry" / "fill_station.usdc", b"source-usdc")
    liquid = _write(tmp_path / "liquid.usda", b"#usda 1.0\ndef Xform \"FillStation\" {}\n")
    output = tmp_path / "segment"

    manifest = build_transport_layers(
        mode=BETWEEN_MODE,
        owner_thread_id=OWNER_THREAD_ID,
        transport_implementation_sha256=IMPLEMENTATION_SHA256,
        source_stage=source,
        source_geometry=geometry,
        liquid_layer=liquid,
        output_dir=output,
        expected_source_hashes={
            "fill_station.usda": _sha256(source),
            "geometry/fill_station.usdc": _sha256(geometry),
        },
        liquid_evidence_sha256=_sha256(liquid),
    )

    assert manifest["source_time_code"] == 1
    assert manifest["owner_thread_id"] == OWNER_THREAD_ID
    assert manifest["transport_implementation_sha256"] == IMPLEMENTATION_SHA256
    assert len(manifest["state_layers"]) == 13
    assert manifest["mode"] == BETWEEN_MODE
    assert (output / "transport-stage.usda").is_file()
    assert (output / "transport-liquid.usda").read_bytes() == liquid.read_bytes()
    for row in manifest["state_layers"]:
        path = output / row["path"]
        assert _sha256(path) == row["sha256"]
        text = path.read_text(encoding="utf-8")
        assert "timeSamples" not in text
        assert "timeCodesPerSecond = 30" in text
        assert "framesPerSecond = 30" in text
        assert "fluidmixTransport" in text
        assert f'ownerThreadId = "{OWNER_THREAD_ID}"' in text
        assert f'transportImplementationSha256 = "{IMPLEMENTATION_SHA256}"' in text
        assert all(path_value not in text for path_value in CAMERA_AND_LIGHT_PATHS)
        assert all(
            path_value.rsplit("/", 1)[-1] not in text
            for path_value in CAMERA_AND_LIGHT_PATHS
        )
        assert "Physx" not in text and "Physics" not in text
    common = (output / "transport-stage.usda").read_text(encoding="utf-8")
    assert "sourceTimeCode = 1" in common
    assert f'ownerThreadId = "{OWNER_THREAD_ID}"' in common
    assert f'transportImplementationSha256 = "{IMPLEMENTATION_SHA256}"' in common
    assert "timeCodesPerSecond = 30" in common
    assert "framesPerSecond = 30" in common
    assert all(path_value not in common for path_value in CAMERA_AND_LIGHT_PATHS)
    assert all(
        path_value.rsplit("/", 1)[-1] not in common
        for path_value in CAMERA_AND_LIGHT_PATHS
    )
    assert "Physx" not in common and "Physics" not in common
    assert all(path_value.rsplit("/", 1)[-1] in common for path_value in LEGACY_ANIMATED_PATHS)


def test_evidence_closure_binds_all_thirteen_arrays_layers_and_revalidates_mutation(
    tmp_path: Path,
) -> None:
    sources = [
        _write(tmp_path / "source.usda", b"source"),
        _write(tmp_path / "source.usdc", b"geometry"),
    ]
    arrays = [
        _write(
            tmp_path / "arrays" / f"array-{index:02d}.npy",
            f"array-{index}".encode(),
        )
        for index in range(13)
    ]
    layers = [
        _write(
            tmp_path / "states" / f"state-{index:02d}.usda",
            f"state-{index}".encode(),
        )
        for index in range(13)
    ]
    receipt = _write(tmp_path / "receipt.json", b"receipt")
    counters = {
        "physics_steps": 0,
        "physics_callbacks": 0,
        "timeline_playing": False,
        "source_time_code": 1,
    }

    closure = capture_evidence_closure(
        source_files=sources,
        snapshot_arrays=arrays,
        state_layers=layers,
        dependency_files=[receipt],
        counters=counters,
    )

    assert len(closure["snapshot_arrays"]) == 13
    assert len(closure["state_layers"]) == 13
    assert _validate_evidence_closure_shape(closure) == closure
    validate_evidence_closure(
        closure,
        source_files=sources,
        snapshot_arrays=arrays,
        state_layers=layers,
        dependency_files=[receipt],
        counters=counters,
    )
    arrays[-1].write_bytes(b"mutated")
    with pytest.raises(ValueError, match="snapshot array evidence drifted"):
        validate_evidence_closure(
            closure,
            source_files=sources,
            snapshot_arrays=arrays,
            state_layers=layers,
            dependency_files=[receipt],
            counters=counters,
        )

    malformed = deepcopy(closure)
    malformed["snapshot_arrays"] = malformed["snapshot_arrays"][:-1]
    with pytest.raises(ValueError, match="snapshot_arrays closure membership"):
        _validate_evidence_closure_shape(malformed)


def test_post_annotator_runtime_partition_is_exact_and_hash_bound() -> None:
    prims = [
        "/Render",
        "/Render/Vars",
        "/Render/Vars/Rgb",
        "/Render/Product",
        "/Render/PostProcess",
        "/Render/PostProcess/SDGPipeline",
        "/Render/PostProcess/SDGPipeline/InstanceSegmentation",
    ]
    properties = [
        "/Render.purpose",
        "/Render/Product.camera",
        "/Render/Product.resolution",
        "/Render/Vars/Rgb.dataType",
        "/Render/PostProcess/SDGPipeline/InstanceSegmentation.inputs:renderProductPath",
    ]
    frozen = freeze_runtime_partition(prims, properties)

    validate_runtime_partition(frozen, prims, properties)
    changed = deepcopy(properties)
    changed.append("/Render/PostProcess/SDGPipeline/Rogue.inputs:value")
    with pytest.raises(ValueError, match="runtime render partition drifted"):
        validate_runtime_partition(frozen, prims, changed)
    with pytest.raises(ValueError, match="runtime render partition is malformed"):
        freeze_runtime_partition([*prims, "/RendererRogue"], properties)
    assert frozen["sha256"] == hashlib.sha256(
        json.dumps(
            {"prim_paths": sorted(prims), "property_paths": sorted(properties)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _fake_segment(tmp_path: Path) -> tuple[Path, list[Path]]:
    source = _write(tmp_path / "source" / "fill_station.usda", b"source-usda")
    geometry = _write(
        tmp_path / "source" / "geometry" / "fill_station.usdc", b"source-usdc"
    )
    liquid = _write(
        tmp_path / "liquid.usda", b'#usda 1.0\ndef Xform "FillStation" {}\n'
    )
    segment = tmp_path / "segment"
    implementation = capture_transport_implementation(ROOT)
    manifest = build_transport_layers(
        mode=BETWEEN_MODE,
        owner_thread_id=OWNER_THREAD_ID,
        transport_implementation_sha256=str(implementation["sha256"]),
        source_stage=source,
        source_geometry=geometry,
        liquid_layer=liquid,
        output_dir=segment,
        expected_source_hashes={
            "fill_station.usda": _sha256(source),
            "geometry/fill_station.usdc": _sha256(geometry),
        },
        liquid_evidence_sha256=_sha256(liquid),
    )
    arrays = [
        _write(tmp_path / "arrays" / f"array-{index:02d}.npy", f"array-{index}".encode())
        for index in range(13)
    ]
    dependencies = [_write(tmp_path / "receipt.json", b"receipt")]
    state_layers = [segment / row["path"] for row in manifest["state_layers"]]
    closure = capture_evidence_closure(
        source_files=[source, geometry],
        snapshot_arrays=arrays,
        state_layers=state_layers,
        dependency_files=dependencies,
        counters={
            "physics_steps": 0,
            "physics_callbacks": 0,
            "timeline_playing": False,
            "source_time_code": 1,
        },
    )
    _write_json(
        segment / "transport-build-receipt.json",
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
            "status": "pass",
            "mode": BETWEEN_MODE,
            "owner_thread_id": OWNER_THREAD_ID,
            "transport_implementation": implementation,
            "source_time_code": 1,
            "physics_steps": 0,
            "camera_and_light_specs_authored": [],
            "layer_manifest": manifest,
            "pre_output_evidence_closure": closure,
        },
    )
    return segment, arrays


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.partition = freeze_runtime_partition(
            [
                "/Render",
                "/Render/Vars",
                "/Render/Vars/Rgb",
                "/Render/Product",
                "/Render/PostProcess/SDGPipeline",
            ],
            [
                "/Render/Product.camera",
                "/Render/Vars/Rgb.dataType",
                "/Render/PostProcess/SDGPipeline.inputs:renderProductPath",
            ],
        )
        self.current = 0
        self.closed = False

    def _attestation(self, index: int) -> dict[str, object]:
        return {
            "state_index": index,
            "source_time_code": 1,
            "physics_steps": 0,
            "physics_callbacks": 0,
            "timeline_playing": False,
            "runtime_partition": self.partition,
            "camera_light_before_annotators_sha256": "1" * 64,
            "camera_light_sha256": "1" * 64,
            "scene_session_sha256": "2" * 64,
        }

    def preflight_state(self, _path: Path, state_index: int) -> dict[str, object]:
        self.current = state_index
        return self._attestation(state_index)

    def capture_state(self, _path: Path, state_index: int) -> dict[str, object]:
        self.current = state_index
        rgb = np.full((480, 640, 3), state_index + 10, dtype=np.uint8)
        instance = np.zeros((480, 640), dtype=np.uint32)
        empty = np.zeros((480, 640), dtype=np.uint8)
        return {
            "rgb": rgb,
            "instance_ids": instance,
            "instance_id_to_path": {},
            "group_masks": {
                "outgoing": empty.copy(),
                "incoming": empty.copy(),
                "queued": empty.copy(),
                "incoming_labels": empty.copy(),
                "queue_tail": empty.copy(),
            },
            "attestation": self._attestation(state_index),
        }

    def attest_current(self) -> dict[str, object]:
        return self._attestation(self.current)

    def close(self) -> None:
        self.closed = True


def _checker_result(*, pilot: bool = True) -> dict[str, object]:
    pending = None if pilot else True
    return {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_stage_checker_v1",
        "status": "pass",
        "pilot": pilot,
        "automatic_acceptance_gates": {
            "evidence_closure_exact": True,
            "runtime_partition_exact": True,
            "camera_and_lighting_unchanged": True,
            "source_time_one_and_zero_physics": True,
            "instance_mappings_and_masks_valid": True,
            "background_sentinels_within_2": True,
            "outgoing_absent_final": pending,
            "incoming_centered_final": pending,
            "start_seam_passed": pending,
            "end_seam_passed": pending,
            "packaged_sequence_byte_unique": pending,
            "pilot_visual_review_accepted": pending,
        },
        "manual_visual_review_required": pilot,
        "pilot_visual_review_status": "required" if pilot else "accepted",
    }


def test_renderer_freezes_partition_before_output_and_publishes_after_recheck(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()

    receipt = run_transport_stage(
        segment_dir=segment,
        output_dir=output,
        state_indices=PILOT_STATE_INDICES,
        runtime=runtime,
        check_output=lambda **_kwargs: _checker_result(),
    )

    assert receipt["status"] == "pilot"
    assert receipt["owner_thread_id"] == OWNER_THREAD_ID
    assert receipt["pre_output_evidence_closure_sha256"] == receipt[
        "post_checker_evidence_closure"
    ]["sha256"]
    assert len(receipt["post_checker_evidence_closure"]["snapshot_arrays"]) == 13
    assert len(receipt["post_checker_evidence_closure"]["state_layers"]) == 13
    assert receipt["final_runtime_attestation"] == receipt[
        "preflight_attestations"
    ][-1]
    assert receipt["camera_light_before_annotators_sha256"] == receipt[
        "camera_light_sha256"
    ]
    assert output.is_dir()
    assert (output / "contact-sheet.png").is_file()
    assert runtime.closed is True


def test_full_renderer_requires_pilot_visual_review_before_any_preflight(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()

    with pytest.raises(ValueError, match="pilot visual review"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=ALL_STATE_INDICES,
            runtime=runtime,
            check_output=lambda **_kwargs: _checker_result(),
        )

    assert runtime.current == 0
    assert runtime.closed is True
    assert not output.exists()


def test_full_renderer_binds_exact_independently_accepted_pilot_review(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    pilot_output = tmp_path / "pilot"
    run_transport_stage(
        segment_dir=segment,
        output_dir=pilot_output,
        state_indices=PILOT_STATE_INDICES,
        runtime=_FakeRuntime(),
        check_output=lambda **_kwargs: _checker_result(),
    )
    pilot_receipt = pilot_output / "transport-render-receipt.json"
    contact_sheet = pilot_output / "contact-sheet.png"
    review_path = tmp_path / "reviews" / "transport-pilot-review.json"
    review_path.parent.mkdir()
    _write_json(
        review_path,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_pilot_visual_review_v1",
            "status": "accepted",
            "owner_thread_id": OWNER_THREAD_ID,
            "transport_implementation_sha256": str(
                capture_transport_implementation(ROOT)["sha256"]
            ),
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
    full_output = tmp_path / "full"

    receipt = run_transport_stage(
        segment_dir=segment,
        output_dir=full_output,
        state_indices=ALL_STATE_INDICES,
        runtime=_FakeRuntime(),
        check_output=lambda **_kwargs: _checker_result(pilot=False),
        pilot_visual_review_path=review_path,
    )

    assert receipt["status"] == "candidate"
    assert receipt["pilot_visual_review"]["status"] == "accepted"
    assert receipt["pilot_visual_review"]["review_sha256"] == _sha256(review_path)
    assert full_output.is_dir()


def test_renderer_revalidates_all_snapshot_arrays_after_final_checker(
    tmp_path: Path,
) -> None:
    segment, arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()
    calls = 0

    def mutate_after_checker(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            arrays[-1].write_bytes(b"mutated-after-checker")
        return _checker_result()

    with pytest.raises(ValueError, match="snapshot array evidence drifted"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=PILOT_STATE_INDICES,
            runtime=runtime,
            check_output=mutate_after_checker,
        )

    assert not output.exists()
    assert runtime.closed is True


def test_renderer_revalidates_build_receipt_after_final_checker(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()
    calls = 0

    def mutate_build_after_checker(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            path = segment / "transport-build-receipt.json"
            path.write_bytes(path.read_bytes() + b" ")
        return _checker_result()

    with pytest.raises(ValueError, match="build receipt evidence drifted"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=PILOT_STATE_INDICES,
            runtime=runtime,
            check_output=mutate_build_after_checker,
        )

    assert not output.exists()
    assert runtime.closed is True
