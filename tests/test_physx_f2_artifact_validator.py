"""Offline checks must validate artifact bytes rather than receipt metadata alone."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_physx_f2_artifacts",
    ROOT / "scripts" / "validate_physx_f2_artifacts.py",
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.mark.parametrize(
    "unsafe",
    ["", ".", "../escape.png", "frames/../../escape.png", "/tmp/escape.png"],
)
def test_resolve_artifact_rejects_empty_absolute_and_traversal(tmp_path, unsafe):
    with pytest.raises(ValueError):
        validator.resolve_artifact(tmp_path, unsafe)


def test_resolve_artifact_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-artifact.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "link.bin"
    link.symlink_to(outside)

    with pytest.raises(ValueError):
        validator.resolve_artifact(tmp_path, "link.bin")


def test_resolve_artifact_accepts_existing_regular_file(tmp_path):
    target = tmp_path / "frames" / "frame.png"
    target.parent.mkdir()
    target.write_bytes(b"frame")

    assert validator.resolve_artifact(tmp_path, "frames/frame.png") == target.resolve()


def test_verify_hash_reads_actual_bytes_and_rejects_metadata_only_match(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual bytes")
    expected = hashlib.sha256(b"actual bytes").hexdigest()

    validator.verify_hash(artifact, expected)
    with pytest.raises(validator.ArtifactValidationError, match="SHA-256"):
        validator.verify_hash(artifact, "0" * 64)


def test_rgb_mask_and_semantic_readability_checks_actual_shapes(tmp_path):
    rgb = np.full((4, 5, 3), 127, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    semantic = np.arange(20, dtype=np.uint32).reshape(4, 5)
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    semantic_path = tmp_path / "semantic.npy"
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(mask_path), mask)
    np.save(semantic_path, semantic, allow_pickle=False)

    assert validator.read_rgb(rgb_path, 5, 4).shape == (4, 5, 3)
    assert int(np.count_nonzero(validator.read_mask(mask_path, 5, 4))) == 4
    np.testing.assert_array_equal(
        validator.read_semantic(semantic_path, 5, 4), semantic
    )

    with pytest.raises(validator.ArtifactValidationError, match="shape"):
        validator.read_rgb(rgb_path, 6, 4)


def test_mask_must_contain_liquid_and_semantic_must_be_integer(tmp_path):
    empty_mask = tmp_path / "empty.png"
    floating_semantic = tmp_path / "semantic.npy"
    assert cv2.imwrite(str(empty_mask), np.zeros((3, 4), dtype=np.uint8))
    np.save(
        floating_semantic,
        np.zeros((3, 4), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(validator.ArtifactValidationError, match="nonempty"):
        validator.read_mask(empty_mask, 4, 3)
    with pytest.raises(validator.ArtifactValidationError, match="integer"):
        validator.read_semantic(floating_semantic, 4, 3)


def test_semantic_artifact_must_reproduce_exact_class_mask_mapping_and_counts():
    semantic = np.array([[0, 5, 5], [7, 5, 0]], dtype=np.uint32)
    mask = np.array([[0, 255, 255], [0, 255, 0]], dtype=np.uint8)
    pair = {
        "semantic_unique_ids": [0, 5, 7],
        "liquid_semantic_ids": [5],
        "semantic_id_to_class_labels": {
            "0": ["BACKGROUND"],
            "5": ["container", "liquid"],
            "7": ["vial"],
        },
        "liquid_pixel_counts_by_id": {"5": 3},
        "liquid_pixel_count": 3,
    }

    validator.validate_semantic_mask_evidence(semantic, mask, pair)

    tampered = mask.copy()
    tampered[1, 0] = 255
    with pytest.raises(validator.ArtifactValidationError, match="semantic class IDs"):
        validator.validate_semantic_mask_evidence(semantic, tampered, pair)

    bad_counts = dict(pair, liquid_pixel_counts_by_id={"5": 2})
    with pytest.raises(validator.ArtifactValidationError, match="per-ID pixel counts"):
        validator.validate_semantic_mask_evidence(semantic, mask, bad_counts)

    bad_mapping = dict(
        pair,
        semantic_id_to_class_labels={
            "0": ["BACKGROUND"],
            "05": ["liquid"],
            "5": ["vial"],
            "7": ["vial"],
        },
    )
    with pytest.raises(validator.ArtifactValidationError, match="mapping"):
        validator.validate_semantic_mask_evidence(semantic, mask, bad_mapping)


def test_video_decoder_checks_actual_frame_count_and_resolution(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 6)
    )
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    writer.release()

    assert validator.validate_video(video_path, 8, 6, 3) == 3
    with pytest.raises(validator.ArtifactValidationError, match="frame count"):
        validator.validate_video(video_path, 8, 6, 4)
    with pytest.raises(validator.ArtifactValidationError, match="resolution"):
        validator.validate_video(video_path, 9, 6, 3)


def _proxy_geometry(status="proxy_refreshed_exact"):
    return {
        "status": status,
        "source_path": "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
        "proxy_path": "/FillStation/FluidmixF2/ParticleSystem/LiquidSemanticProxy",
        "same_parent_transform": True,
        "point_count": 4,
        "face_count": 4,
        "corner_count": 12,
        "points_sha256": "1" * 64,
        "proxy_points_sha256": "1" * 64,
        "face_vertex_counts_sha256": "2" * 64,
        "proxy_face_vertex_counts_sha256": "2" * 64,
        "face_vertex_indices_sha256": "3" * 64,
        "proxy_face_vertex_indices_sha256": "3" * 64,
        "topology_sha256": "4" * 64,
        "proxy_topology_sha256": "4" * 64,
        "local_transform_sha256": "5" * 64,
        "proxy_local_transform_sha256": "5" * 64,
        "mesh_sha256": "6" * 64,
        "proxy_mesh_sha256": "6" * 64,
        "exact_geometry_match": True,
        "material_binding": "/FillStation/FluidmixF2/GlobalLiquidMaterial",
        "semantic_binding": {"type": "class", "labels": ["liquid"]},
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proxy_artifact_fixture(tmp_path):
    mesh_dir = tmp_path / "mesh-proxy"
    refresh_dir = mesh_dir / "refresh"
    refresh_dir.mkdir(parents=True)
    creation = {
        "physics_step": 12,
        "evidence": _proxy_geometry("proxy_authored_pending_rgb_parity"),
        "source_visibility": "inherited",
        "proxy_visibility": "invisible",
    }
    creation_path = mesh_dir / "creation-step-00012.json"
    creation_hash = _write_json(creation_path, creation)

    original = np.zeros((6, 8, 3), dtype=np.uint8)
    original[..., 0] = np.arange(8, dtype=np.uint8)
    proxy = original.copy()
    proxy[2:4, 3:6] += 2
    difference = np.abs(
        original.astype(np.int16) - proxy.astype(np.int16)
    ).astype(np.uint8)
    original_path = mesh_dir / "parity-original-source.png"
    proxy_path = mesh_dir / "parity-standard-proxy.png"
    difference_path = mesh_dir / "parity-absdiff.png"
    assert cv2.imwrite(str(original_path), original)
    assert cv2.imwrite(str(proxy_path), proxy)
    assert cv2.imwrite(str(difference_path), difference)

    refresh = {
        "physics_step": 12,
        "phase": "t200-pour",
        "evidence": _proxy_geometry(),
        "source_visibility": "invisible",
        "proxy_visibility": "inherited",
        "shared_rgb_semantic_capture": True,
    }
    refresh_path = refresh_dir / "frame-0002-step-00012.json"
    refresh_hash = _write_json(refresh_path, refresh)
    metrics = {
        "mean_abs_rgb": float(np.mean(difference)),
        "p95_abs_rgb": float(np.percentile(difference, 95)),
        "fraction_abs_rgb_gt_16": float(np.mean(difference > 16)),
    }
    record = {
        **refresh,
        "artifact_path": "mesh-proxy/refresh/frame-0002-step-00012.json",
        "artifact_sha256": refresh_hash,
    }
    receipt = {
        "render": {
            "frame_count": 2,
            "frame_physics_steps": [6, 12],
            "stream_only_frame_count": 1,
            "vial_proxy_frame_count": 1,
            "vial_proxy_transition_step": 12,
            "liquid_representation_monotonic": True,
        },
        "mesh_proxy": {
            "creation_step": 12,
            "creation_evidence": creation["evidence"],
            "creation_artifact_path": "mesh-proxy/creation-step-00012.json",
            "creation_artifact_sha256": creation_hash,
            "stage_surface_sync_records": [],
            "stage_isosurface_lifecycle_refresh_records": [],
            "rgb_parity": {
                "status": "rgb_parity_pass_pending_shared_capture",
                "hide_source_allowed": True,
                "same_physics_step": True,
                "same_renderer_settings": True,
                "parity_capture_isolated": True,
                "source_restored_visible": True,
                "physics_step": 12,
                "phase": "t200-pour",
                "parity_geometry_mesh_sha256": "6" * 64,
                "parity_geometry_points_sha256": "1" * 64,
                "original_capture_visibility": ["inherited", "invisible"],
                "proxy_capture_visibility": ["invisible", "inherited"],
                "restored_visibility": ["inherited", "invisible"],
                "thresholds": {
                    "mean_abs_rgb_max": 2.0,
                    "p95_abs_rgb_max": 8.0,
                    "fraction_abs_rgb_gt_16_max": 0.01,
                },
                "metrics": metrics,
                "original_rgb_path": "mesh-proxy/parity-original-source.png",
                "original_rgb_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                "proxy_rgb_path": "mesh-proxy/parity-standard-proxy.png",
                "proxy_rgb_sha256": hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
                "absdiff_path": "mesh-proxy/parity-absdiff.png",
                "absdiff_sha256": hashlib.sha256(difference_path.read_bytes()).hexdigest(),
            },
            "refresh_records": [record],
        },
        "semantics": {
            "pairs": [
                {
                    "phase": "t200-pour",
                    "liquid_representation": "stream_only",
                    "visible_stream_tanks": ["t200"],
                    "mesh_proxy_refresh_artifact_path": None,
                    "mesh_proxy_refresh_artifact_sha256": None,
                    "source_isosurface_visibility": "invisible",
                    "proxy_visibility": "absent",
                    "shared_rgb_semantic_capture": True,
                },
                {
                    "phase": "t200-pour",
                    "liquid_representation": "vial_proxy",
                    "visible_stream_tanks": ["t200"],
                    "mesh_proxy_refresh_artifact_path": record["artifact_path"],
                    "mesh_proxy_refresh_artifact_sha256": record["artifact_sha256"],
                    "source_isosurface_visibility": "invisible",
                    "proxy_visibility": "inherited",
                    "shared_rgb_semantic_capture": True,
                }
            ]
        },
    }
    return receipt, difference_path, refresh_path


def test_schema15_proxy_artifacts_allow_stream_prefix_and_verify_proxy_bytes(tmp_path):
    receipt, _difference_path, _refresh_path = _proxy_artifact_fixture(tmp_path)

    checked, failures = validator.validate_mesh_proxy_artifacts(
        tmp_path, receipt, expected_width=8, expected_height=6
    )

    assert failures == []
    assert len(checked) == 5


def test_schema15_proxy_artifacts_bind_optional_stage_lifecycle_sidecar(tmp_path):
    receipt, _difference_path, _refresh_path = _proxy_artifact_fixture(tmp_path)
    state = {
        "particle_count": 370,
        "particle_ids_sha256": "7" * 64,
        "positions_sha256": "8" * 64,
    }
    lifecycle = {
        "physics_step": 706,
        "phase": "t200-stage",
        "attempt": 1,
        "purpose": "stage_isosurface_lifecycle_refresh",
        "documentation_classification": (
            "installed_nvidia_test_source_plus_bounded_adaptation"
        ),
        "documentation_reference": "installed NVIDIA deferred creation test",
        "documented_initial_creation_update_count": 2,
        "documented_disable_toggle_update_count": 1,
        "documented_reenable_toggle_update_count": 1,
        "experiment_disable_update_count": 1,
        "experiment_reenable_update_count": 2,
        "disable_app_update_calls": 1,
        "disable_app_updates_completed": 1,
        "reenable_app_update_calls": 2,
        "reenable_app_updates_completed": 2,
        "callbacks_before": 706,
        "callbacks_after_disable": 706,
        "callbacks_after_enable": 706,
        "callbacks_after": 706,
        "timeline_time_before_s": 706.0 / 60.0,
        "timeline_time_after_disable_s": 706.0 / 60.0,
        "timeline_time_after_enable_s": 706.0 / 60.0,
        "timeline_time_after_s": 706.0 / 60.0,
        "timeline_playing_before": True,
        "timeline_playing_after": True,
        "isosurface_enabled_readback_before": True,
        "isosurface_enabled_readback_after_disable": False,
        "isosurface_enabled_readback_after_enable": True,
        "particle_state_before": state,
        "particle_state_after_disable": state,
        "particle_state_after_enable": state,
        "particle_state_after_restore": state,
        "proxy_created_after_refresh": True,
        "source_isosurface_evidence_after_refresh": {
            "mesh_geometry_valid": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
        },
        "failed_predicates": [],
        "error": None,
        "passed": True,
    }
    lifecycle_path = (
        tmp_path
        / "mesh-proxy"
        / "stage-isosurface-lifecycle-refresh-step-00706.json"
    )
    lifecycle_hash = _write_json(lifecycle_path, lifecycle)
    receipt["mesh_proxy"]["stage_isosurface_lifecycle_refresh_records"] = [
        {
            **lifecycle,
            "artifact_path": (
                "mesh-proxy/stage-isosurface-lifecycle-refresh-step-00706.json"
            ),
            "artifact_sha256": lifecycle_hash,
        }
    ]

    checked, failures = validator.validate_mesh_proxy_artifacts(
        tmp_path, receipt, expected_width=8, expected_height=6
    )

    assert failures == []
    assert lifecycle_path in checked


def test_schema15_proxy_artifacts_reject_rehashed_wrong_diff_and_refresh_content(tmp_path):
    receipt, difference_path, refresh_path = _proxy_artifact_fixture(tmp_path)
    wrong = np.full((6, 8, 3), 9, dtype=np.uint8)
    assert cv2.imwrite(str(difference_path), wrong)
    receipt["mesh_proxy"]["rgb_parity"]["absdiff_sha256"] = hashlib.sha256(
        difference_path.read_bytes()
    ).hexdigest()
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    refresh["physics_step"] = 7
    receipt["mesh_proxy"]["refresh_records"][0]["artifact_sha256"] = _write_json(
        refresh_path, refresh
    )
    receipt["semantics"]["pairs"][1]["mesh_proxy_refresh_artifact_sha256"] = receipt[
        "mesh_proxy"
    ]["refresh_records"][0]["artifact_sha256"]

    _checked, failures = validator.validate_mesh_proxy_artifacts(
        tmp_path, receipt, expected_width=8, expected_height=6
    )

    assert any("absdiff pixels" in failure for failure in failures)
    assert any("refresh[0]:content mismatch" in failure for failure in failures)


def test_schema15_proxy_artifacts_fail_closed_on_non_string_refresh_path(tmp_path):
    receipt, _difference_path, _refresh_path = _proxy_artifact_fixture(tmp_path)
    receipt["mesh_proxy"]["refresh_records"][0]["artifact_path"] = {}

    _checked, failures = validator.validate_mesh_proxy_artifacts(
        tmp_path, receipt, expected_width=8, expected_height=6
    )

    assert any("artifact paths are not unique" in failure for failure in failures)
    assert any("artifact path must be" in failure for failure in failures)


def test_schema15_proxy_artifacts_bind_parity_to_first_refresh_geometry(tmp_path):
    receipt, _difference_path, _refresh_path = _proxy_artifact_fixture(tmp_path)
    receipt["mesh_proxy"]["rgb_parity"]["parity_geometry_mesh_sha256"] = "f" * 64

    _checked, failures = validator.validate_mesh_proxy_artifacts(
        tmp_path, receipt, expected_width=8, expected_height=6
    )

    assert any("refresh geometry binding mismatch" in failure for failure in failures)


def test_validate_output_calls_schema_contract_and_fails_closed(tmp_path):
    (tmp_path / "receipt.json").write_text("{}\n", encoding="utf-8")

    result = validator.validate_output(tmp_path)

    assert result["status"] == "insufficient_evidence"
    assert "receipt:receipt_identity" in result["failures"]
    assert "receipt:run_status_not_pass" in result["failures"]


def test_validate_output_reports_unsafe_paths_even_when_receipt_is_invalid(tmp_path):
    receipt = {
        "render": {
            "resolution": [640, 480],
            "frame_paths": ["../outside.png"],
            "frame_sha256": ["0" * 64],
        }
    }
    (tmp_path / "receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    result = validator.validate_output(tmp_path)

    assert result["status"] == "insufficient_evidence"
    assert any("unsafe" in failure or "safe relative" in failure for failure in result["failures"])


def test_validate_output_reports_missing_or_unreadable_receipt(tmp_path):
    missing = validator.validate_output(tmp_path)
    assert missing["status"] == "insufficient_evidence"
    assert missing["failures"] == ["receipt_missing_or_unreadable"]

    (tmp_path / "receipt.json").write_text("not json", encoding="utf-8")
    unreadable = validator.validate_output(tmp_path)
    assert unreadable["status"] == "insufficient_evidence"
    assert unreadable["failures"] == ["receipt_missing_or_unreadable"]


def test_validate_output_fails_closed_when_output_directory_is_missing(tmp_path):
    missing_root = tmp_path / "does-not-exist"

    result = validator.validate_output(missing_root)

    assert result["status"] == "insufficient_evidence"
    assert result["failures"] == ["receipt_missing_or_unreadable"]
    assert validator.main([str(missing_root)]) == 2

