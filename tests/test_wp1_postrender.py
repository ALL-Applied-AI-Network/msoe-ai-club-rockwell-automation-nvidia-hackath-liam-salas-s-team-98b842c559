from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.full_transition_manifest import (  # noqa: E402
    build_default_full_transition_source_spec,
    build_full_transition_artifacts,
)
from fluidmix.wp1_postrender import (  # noqa: E402
    canonical_sha256,
    decide_wp1_visibility,
    file_sha256,
    package_wp1_test_videos,
    validate_wp1_acceptance,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_png(path: Path, rgb: tuple[int, int, int], *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask:
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[2:7, 2:7] = 255
        Image.fromarray(pixels, mode="L").save(path)
    else:
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        pixels[:, :] = rgb
        Image.fromarray(pixels, mode="RGB").save(path)


def _manifest_bundle(root: Path) -> dict[str, object]:
    artifacts = build_full_transition_artifacts(
        build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
    )
    for relative, value in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            _write_json(path, value)
    return artifacts


def _minimal_case(
    case_root: Path,
    *,
    case_id: str,
    schedule: dict[str, object],
    model_row: dict[str, object],
    render_plan: dict[str, object],
    plan_path: Path,
    runtime: dict[str, str],
    colours: list[tuple[int, int, int]] | None = None,
) -> None:
    app = case_root / "app-output"
    app.mkdir(parents=True)
    video = app / "three-pour.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480)
    )
    assert writer.isOpened()
    endpoint_frames = []
    for value in (100, 220):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[180:300, 280:330] = value
        endpoint_frames.append(frame)
        writer.write(frame)
    writer.release()
    endpoints = app / "endpoints"
    endpoints.mkdir()
    assert cv2.imwrite(str(endpoints / "first.png"), endpoint_frames[0])
    assert cv2.imwrite(str(endpoints / "final.png"), endpoint_frames[-1])
    pairs = []
    stage_paths = []
    stage_hashes = []
    stages = schedule["physical_sequence"]["dispense_order"]
    mask = app / "masks" / "liquid.png"
    _write_png(mask, (0, 0, 0), mask=True)
    for index, tank in enumerate(stages):
        rgb = app / "frames" / f"frame-{index + 1:04d}-{tank}-stage.png"
        stage_rgb = app / "stages" / f"stage-{index + 1}-{tank}.png"
        colour = (colours or [(70 + index * 20, 80, 90) for index in range(3)])[index]
        _write_png(rgb, colour)
        _write_png(stage_rgb, colour)
        relative = rgb.relative_to(app).as_posix()
        stage_paths.append(stage_rgb.relative_to(app).as_posix())
        stage_hashes.append(file_sha256(stage_rgb))
        pairs.append(
            {
                "physics_step": schedule["pours"][index]["stage_capture_step"],
                "phase": f"{tank}-stage",
                "visible_stream_tanks": [],
                "liquid_representation": "count_driven_twin_liquid_mesh",
                "paired_capture_state_passed": True,
                "rgb_path": relative,
                "rgb_sha256": file_sha256(rgb),
                "mask_path": mask.relative_to(app).as_posix(),
                "mask_sha256": file_sha256(mask),
            }
        )
    receipt = {
        "schema_version": 20,
        "status": "pass",
        "errors": [],
        "warnings": [],
        "render_path": "count_driven_twin_liquid_mesh",
        "arguments": {
            "video_fps": 10.0,
            "train_render_plan_sha256": canonical_sha256(render_plan),
            "train_case_id": case_id,
            "accepted_qualification_receipt_sha256": runtime["qualification_receipt_sha256"],
            "nuisance_receipt_sha256": runtime["nuisance_receipt_sha256"],
            "count_driven_twin_liquid_mesh": True,
            "particle_resolution_profile": render_plan["particle_resolution_profile"]["profile_id"],
            "recipe8_effective_particle_volume_ml": render_plan["particle_resolution_profile"]["effective_particle_volume_ml"],
        },
        "versions": {"isaac": runtime["isaac_version"], "kit": runtime["kit_version"]},
        "provenance": {
            "requested_stage_sha256": runtime["stage_sha256"],
            "script_sha256": runtime["renderer_script_sha256"],
        },
        "render": {
            "video_path": "three-pour.avi",
            "video_sha256": file_sha256(video),
            "video_decodable": True,
            "video_container": "avi",
            "video_codec": "MJPG",
            "video_decoded_frame_count": 2,
            "video_playback_fps": 10.0,
            "video_renderer": "RaytracedLighting",
            "stage_still_renderer": "PathTracing",
            "stage_frames_are_video_subsets": False,
            "stage_frame_paths": stage_paths,
            "stage_frame_sha256": stage_hashes,
            "stage_frame_physics_steps": [
                pour["stage_capture_step"] for pour in schedule["pours"]
            ],
        },
        "semantics": {
            "semantic_gate_passed": True,
            "count_driven_body_semantic_bound": True,
            "mask_source": "exact_renderer_semantic_class_ids",
            "pairs": pairs,
        },
        "count_driven_liquid_body": {
            "representation": "count_driven_twin_liquid_mesh",
        },
        "particle_resolution_profile": {
            **deepcopy(render_plan["particle_resolution_profile"]),
            "source_sampler": "resolution_aware_c20_boundary",
        },
        "particle_system": {
            "runtime_configuration": {
                "spawn_spacing_m": render_plan["particle_resolution_profile"]["spawn_spacing_m"],
                "fluid_rest_offset_m": render_plan["particle_resolution_profile"]["fluid_rest_offset_m"],
                "particle_contact_offset_m": render_plan["particle_resolution_profile"]["particle_contact_offset_m"],
                "solver_position_iterations": 16,
            }
        },
        "tank_colour_resolution": deepcopy(render_plan["tank_colour_resolution"]),
        "execution_schedule": {
            "case_id": case_id,
            "cohort_plan_sha256": canonical_sha256(render_plan),
            "schedule_file_sha256": file_sha256(plan_path),
            "source_run_spec_sha256": render_plan["source_run_spec_sha256"],
            "schedule_sha256": render_plan["execution_schedule_sha256_by_case"][case_id],
            "vial_record_sha256": canonical_sha256(model_row),
        },
    }
    _write_json(app / "receipt.json", receipt)
    result = {
        "case_id": case_id,
        "outcome": "passed",
        "return_code": 0,
        "timed_out": False,
        "evidence": {
            "failures": [],
            "launch_failures": [],
            "receipt_failures": [],
            "artifact_validation": {"status": "pass", "failures": [], "warnings": []},
        },
    }
    _write_json(case_root / "case-result.json", result)
    _write_json(
        case_root / "launch-receipt.json",
        {"case_id": case_id, "execution_host": runtime["execution_host"]},
    )
    (case_root / "exit-code.txt").write_text("0\n", encoding="utf-8")


def _runtime(host: str) -> dict[str, str]:
    return {
        "execution_host": host,
        "gpu_model": "NVIDIA L40S",
        "isaac_version": "6.2.0.0",
        "kit_version": "110.4.0+feature.386251.187cf21b.gl",
        "stage_sha256": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
        "renderer_script_sha256": "0a397e3de8046a3ad4754b0924e4068459939b17fe49073f916f278b30c6548c",
        "qualification_receipt_sha256": "3" * 64,
        "nuisance_receipt_sha256": "4" * 64,
        "python_runtime_sha256": "5" * 64,
        "source_recipe_manifest_sha256": "49ab190bf018f61c868f48ddb83dd96d7e1d0ff77080e68d62953d8f8f9ef49b",
        "source_split_spec_sha256": "e6e4b3b8d2c0932629ffae5872a58df9f4fb1b03612b0e842e53339a4d071bfb",
    }


def _write_actual_readiness(root: Path, runtime: dict[str, str]) -> Path:
    implementation = {
        "source_commit": "16fbb85582935ce7fa3ceeffcfa6ddffb5af5e83",
        "supervisor_sha256": "d27a610fe307f1c4e23986fef56caac547747a57804e59a87e7f3635f92da528",
        "renderer_sha256": "0a397e3de8046a3ad4754b0924e4068459939b17fe49073f916f278b30c6548c",
        "offline_validator_sha256": "f7d5014b5fdccb744ebc4e666fee8ab680b3e9f9c65d5f550c38e283716ed1b0",
        "occlusion_profile_sha256": "b056579aec18b2d2ace726d7640289e0401f6239fa4303057646040e880504f7",
        "fill_station_sha256": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
        "geometry_sha256": "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db",
    }
    profile = {
        "gpu_model": "NVIDIA L40S",
        "isaac_sim_version": "6.2.0.0",
        "kit_version": "110.4.0+feature.386251.187cf21b.gl",
        "implementation": implementation,
    }
    evidence = root / "readiness-evidence"
    proof = evidence / "case-result.json"
    offline = evidence / "offline-validator.json"
    host = evidence / "host-profile.json"
    _write_json(
        proof,
        {
            "case_id": "CASE_91365fbcc3369a1c",
            "ordinal": 4,
            "outcome": "passed",
            "return_code": 0,
            "timed_out": False,
            "launch_error": None,
            "evidence": {
                "exit_code": 0,
                "failures": [],
                "launch_failures": [],
                "receipt_failures": [],
                "observed_counts": {
                    "decoded_video_frame_count": 433,
                    "source_capture_count": 3,
                },
                "artifact_validation": {
                    "status": "pass",
                    "failures": [],
                    "verified_artifact_count": 75,
                    "warnings": [],
                },
            },
        },
    )
    _write_json(
        offline,
        {
            "status": "pass",
            "failures": [],
            "offline_revalidations": [{"name": "occlusion", "status": "pass"}],
            "verified_artifact_count": 75,
            "warnings": [],
        },
    )
    _write_json(
        host,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_wp1_l40s_host_profile_v1",
            "status": "pass",
            **profile,
            "process_conflicts": [],
        },
    )
    receipt = root / "wp1-fill-readiness.json"
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_wp1_fill_readiness_v1",
            "status": "pass",
            "owner_thread_id": "01a18128-8dc4-7780-ace1-4d816b49d176",
            "execution_profile": profile,
            "proof_case_id": "CASE_91365fbcc3369a1c",
            "supervisor_case_result": {"path": str(proof.resolve()), "sha256": file_sha256(proof)},
            "offline_validator_result": {"path": str(offline.resolve()), "sha256": file_sha256(offline)},
            "host_profile": {"path": str(host.resolve()), "sha256": file_sha256(host)},
        },
    )
    return receipt


def _write_transport(
    root: Path,
    *,
    binding: dict[str, object],
    fill_by_case: dict[str, dict[str, object]],
    readiness_path: Path,
) -> Path:
    run_id = binding["run_id"]
    case_ids = binding["case_ids"]
    products = root / run_id
    products.mkdir(parents=True)
    readiness = json.loads(readiness_path.read_text())
    readiness_binding = {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_fill_readiness_binding_v1",
        "status": "pass",
        "owner_thread_id": readiness["owner_thread_id"],
        "proof_case_id": readiness["proof_case_id"],
        "execution_profile": readiness["execution_profile"],
        "receipt_path": str(readiness_path.resolve()),
        "receipt_sha256": file_sha256(readiness_path),
        "supervisor_case_result": readiness["supervisor_case_result"],
        "offline_validator_result": readiness["offline_validator_result"],
        "host_profile": readiness["host_profile"],
    }
    review_dir = products / "pilot-review"
    review_dir.mkdir()
    review_file = review_dir / "review.json"
    pilot_receipt = review_dir / "pilot-receipt.json"
    contact = review_dir / "contact.png"
    _write_json(review_file, {"status": "accepted"})
    _write_json(pilot_receipt, {"status": "pilot"})
    assert cv2.imwrite(str(contact), np.zeros((32, 32, 3), dtype=np.uint8))
    implementation_sha = "a" * 64
    pilot_binding = {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_pilot_visual_review_binding_v1",
        "status": "accepted",
        "owner_thread_id": readiness["owner_thread_id"],
        "transport_implementation_sha256": implementation_sha,
        "reviewer_id": "independent-reviewer",
        "independent_from_implementation": True,
        "reviewed_utc": "2026-10-28T00:00:00Z",
        "decision_reference": "fixture-review",
        "review_path": str(review_file.resolve()),
        "review_sha256": file_sha256(review_file),
        "pilot_render_receipt_path": str(pilot_receipt.resolve()),
        "pilot_render_receipt_sha256": file_sha256(pilot_receipt),
        "contact_sheet_path": str(contact.resolve()),
        "contact_sheet_sha256": file_sha256(contact),
        "checks": {"identity": True, "appearance": True},
    }
    modes = ["entry"] + ["between_fill"] * 8 + ["exit"]
    layout = []
    for index, mode in enumerate(modes):
        layout.append(
            {
                "segment_index": index,
                "mode": mode,
                "outgoing_case_id": None if index == 0 else case_ids[index - 1],
                "next_case_id": None if index == 9 else case_ids[index],
            }
        )

    shared = products / "fixture-shared-states"
    shared.mkdir()
    mapping = {
        "1": "/FillStation/FillStation_LocalRoot/Fill_vial_active",
        "2": "/FillStation/FillStation_LocalRoot/Fill_vial_neighbour_1",
        "3": "/FillStation/FillStation_LocalRoot/FluidmixTransport/IncomingLabels",
        "4": "/FillStation/FillStation_LocalRoot/FluidmixTransport/QueueTail",
    }

    def state_value(mode: str, index: int) -> int:
        if mode == "entry":
            return 5 * (index + 1)
        return 225 - 10 * index

    def shared_state(mode: str, index: int) -> dict[str, Path]:
        state_root = shared / mode / f"state-{index:02d}"
        if not state_root.exists():
            state_root.mkdir(parents=True)
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[180:300, 280:330] = state_value(mode, index)
            assert cv2.imwrite(str(state_root / "rgb.png"), rgb)
            instance = np.zeros((480, 640), dtype=np.uint32)
            if mode != "entry" and index == 0:
                instance[180:300, 100:120] = 1
            if mode in {"entry", "between_fill"} and index in {11, 12}:
                instance[180:300, 300:310] = 2
            with (state_root / "instance.npy").open("xb") as stream:
                np.save(stream, instance, allow_pickle=False)
            _write_json(state_root / "mapping.json", mapping)
            for name, identifier in {
                "outgoing": 1,
                "incoming": 2,
                "queued": 99,
                "incoming_labels": 3,
                "queue_tail": 4,
            }.items():
                mask = np.where(instance == identifier, 255, 0).astype(np.uint8)
                assert cv2.imwrite(str(state_root / f"{name}.png"), mask)
        return {path.name: path for path in state_root.iterdir()}

    fill_frames: dict[str, list[np.ndarray]] = {}
    fill_rows = []
    for case_id in case_ids:
        path = Path(str(fill_by_case[case_id]["path"]))
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        fill_frames[case_id] = frames
        fill_rows.append(
            {
                "case_id": case_id,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "frame_count": len(frames),
                "fps": fps,
            }
        )

    segment_rows = []
    fill_bindings = []
    segment_frames: list[list[np.ndarray]] = []
    for row in layout:
        index = int(row["segment_index"])
        mode = str(row["mode"])
        build_dir = products / "segments" / f"segment-{index:02d}-build"
        render_dir = products / "segments" / f"segment-{index:02d}-render"
        states_dir = render_dir / "states"
        build_dir.mkdir(parents=True)
        states_dir.mkdir(parents=True)
        outgoing_id = row["outgoing_case_id"]
        incoming_id = row["next_case_id"]

        def endpoint(case_id: object, which: str) -> dict[str, object] | None:
            if case_id is None:
                return None
            video = Path(str(fill_by_case[str(case_id)]["path"]))
            endpoint_name = "final.png" if which == "start" else "first.png"
            endpoint_path = video.parent / "endpoints" / endpoint_name
            return {
                "path": str(endpoint_path.resolve()),
                "sha256": file_sha256(endpoint_path),
                "frame_index": -1 if which == "start" else 0,
                "video_path": str(video.resolve()),
                "video_sha256": file_sha256(video),
            }

        build = {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
            "status": "pass",
            "mode": mode,
            "owner_thread_id": readiness["owner_thread_id"],
            "source_time_code": 1,
            "physics_steps": 0,
            "camera_and_light_specs_authored": [],
            "transport_implementation": {"sha256": implementation_sha},
            "adjacent_endpoints": {
                "outgoing_case_output": (
                    None
                    if outgoing_id is None
                    else str(Path(str(fill_by_case[str(outgoing_id)]["path"])).parent)
                ),
                "next_case_output": (
                    None
                    if incoming_id is None
                    else str(Path(str(fill_by_case[str(incoming_id)]["path"])).parent)
                ),
                "accepted_start": endpoint(outgoing_id, "start"),
                "accepted_end": endpoint(incoming_id, "end"),
            },
        }
        build_path = build_dir / "transport-build-receipt.json"
        _write_json(build_path, build)
        captures = []
        decoded_states = []
        for state_index in range(13):
            source_files = shared_state(mode, state_index)
            prefix = f"state-{state_index:02d}"
            target_files: dict[str, Path] = {}
            for source_name, source in source_files.items():
                target = states_dir / f"{prefix}-{source_name}"
                os.link(source, target)
                target_files[source_name] = target
            rgb_path = target_files["rgb.png"]
            decoded_states.append(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR))
            group_masks = {}
            for name in (
                "outgoing",
                "incoming",
                "queued",
                "incoming_labels",
                "queue_tail",
            ):
                mask_path = target_files[f"{name}.png"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                group_masks[name] = {
                    "path": mask_path.relative_to(render_dir).as_posix(),
                    "sha256": file_sha256(mask_path),
                    "pixel_count": int(np.count_nonzero(mask)),
                }
            captures.append(
                {
                    "state_index": state_index,
                    "rgb_path": rgb_path.relative_to(render_dir).as_posix(),
                    "rgb_sha256": file_sha256(rgb_path),
                    "instance_path": target_files["instance.npy"].relative_to(render_dir).as_posix(),
                    "instance_sha256": file_sha256(target_files["instance.npy"]),
                    "mapping_path": target_files["mapping.json"].relative_to(render_dir).as_posix(),
                    "mapping_sha256": file_sha256(target_files["mapping.json"]),
                    "group_masks": group_masks,
                    "attestation": {
                        "state_index": state_index,
                        "source_time_code": 1,
                        "physics_steps": 0,
                        "physics_callbacks": 0,
                        "timeline_playing": False,
                    },
                }
            )
        segment_frames.append(decoded_states)
        start_frames = None if outgoing_id is None else fill_frames[str(outgoing_id)]
        end_frames = None if incoming_id is None else fill_frames[str(incoming_id)]
        start_mad = (
            None
            if start_frames is None
            else float(np.mean(np.abs(start_frames[-1].astype(np.float32) - decoded_states[1].astype(np.float32))))
        )
        end_mad = (
            None
            if end_frames is None
            else float(np.mean(np.abs(decoded_states[11].astype(np.float32) - end_frames[0].astype(np.float32))))
        )
        def p99(frames: list[np.ndarray] | None) -> float | None:
            if frames is None:
                return None
            return float(np.percentile([
                np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))
                for left, right in zip(frames, frames[1:])
            ], 99))
        metrics = {
            "background_sentinel_max_channel_delta": 0,
            "background_sentinels_within_2": True,
            "mode_occupancy_valid": True,
            "start_seam_mad": start_mad,
            "start_seam_threshold_p99": p99(start_frames),
            "start_seam_passed": True,
            "end_seam_mad": end_mad,
            "end_seam_threshold_p99": p99(end_frames),
            "end_seam_passed": True,
            "packaged_sequence_byte_unique": True,
        }
        gates = {
            "evidence_closure_exact": True,
            "runtime_partition_exact": True,
            "camera_and_lighting_unchanged": True,
            "source_time_one_and_zero_physics": True,
            "instance_mappings_and_masks_valid": True,
            "background_sentinels_within_2": True,
            "mode_occupancy_valid": True,
            "start_seam_passed": True,
            "end_seam_passed": True,
            "packaged_sequence_byte_unique": True,
            "pilot_visual_review_accepted": True,
        }
        render_receipt = {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_render_receipt_v1",
            "status": "candidate",
            "mode": mode,
            "owner_thread_id": readiness["owner_thread_id"],
            "segment_dir": str(build_dir.resolve()),
            "build_receipt_sha256": file_sha256(build_path),
            "state_indices": list(range(13)),
            "source_time_code": 1,
            "physics_steps": 0,
            "physics_callbacks": 0,
            "wp1_fill_readiness": readiness_binding,
            "pilot_visual_review": pilot_binding,
            "captures": captures,
            "rendered_states_byte_unique": True,
            "checker_evaluation": {
                "schema_version": 1,
                "artifact_type": "fluidmix_transport_stage_checker_v1",
                "status": "pass",
                "pilot": False,
                "metrics": metrics,
                "automatic_acceptance_gates": gates,
                "manual_visual_review_required": False,
                "pilot_visual_review_status": "accepted",
            },
        }
        render_receipt_path = render_dir / "transport-render-receipt.json"
        _write_json(render_receipt_path, render_receipt)
        segment_rows.append(
            {
                "render_dir": str(render_dir.resolve()),
                "receipt_path": str(render_receipt_path.resolve()),
                "receipt_sha256": file_sha256(render_receipt_path),
                "mode": mode,
            }
        )

        def fill_binding(case_id: object) -> dict[str, str] | None:
            if case_id is None:
                return None
            source = fill_by_case[str(case_id)]
            return {
                "case_id": str(case_id),
                "path": str(Path(str(source["path"])).resolve()),
                "sha256": str(source["sha256"]),
            }
        fill_bindings.append(
            {
                "segment_index": index,
                "mode": mode,
                "accepted_start_fill": fill_binding(outgoing_id),
                "accepted_end_fill": fill_binding(incoming_id),
            }
        )

    sequence = list(segment_frames[0][:12]) + list(fill_frames[case_ids[0]])
    ranges = [(0, 13)]
    for index, case_id in enumerate(case_ids[1:], start=1):
        start = len(sequence) - 1
        sequence.extend(segment_frames[index][1:12])
        sequence.extend(fill_frames[case_id])
        ranges.append((start, start + 13))
    start = len(sequence) - 1
    sequence.extend(segment_frames[-1][1:13])
    ranges.append((start, start + 13))
    run_video = products / "run.mp4"
    writer = cv2.VideoWriter(
        str(run_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
    )
    assert writer.isOpened()
    for frame in sequence:
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(run_video))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    output_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    transition_rows = []
    seam_rows = []
    for layout_row, (start, stop) in zip(layout, ranges, strict=True):
        frames = decoded[start:stop]
        hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]
        outgoing = layout_row["outgoing_case_id"]
        incoming = layout_row["next_case_id"]
        start_mad = None if outgoing is None else float(np.mean(np.abs(frames[0].astype(np.float32) - frames[1].astype(np.float32))))
        end_mad = None if incoming is None else float(np.mean(np.abs(frames[11].astype(np.float32) - frames[12].astype(np.float32))))
        def video_p99(case_id: object) -> float | None:
            if case_id is None:
                return None
            frames_ = fill_frames[str(case_id)]
            return float(np.percentile([
                np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))
                for left, right in zip(frames_, frames_[1:])
            ], 99))
        start_threshold = video_p99(outgoing)
        end_threshold = video_p99(incoming)
        transition_rows.append(
            {
                "segment_index": layout_row["segment_index"],
                "mode": layout_row["mode"],
                "start_frame_index": start,
                "stop_frame_index": stop,
                "frame_count": 13,
                "decoded_bgr_sha256": hashes,
                "byte_unique": len(set(hashes)) == 13,
            }
        )
        seam_rows.append(
            {
                "segment_index": layout_row["segment_index"],
                "mode": layout_row["mode"],
                "start_seam_mad": start_mad,
                "start_seam_threshold_p99": start_threshold,
                "start_seam_passed": start_mad is None or start_mad <= start_threshold,
                "end_seam_mad": end_mad,
                "end_seam_threshold_p99": end_threshold,
                "end_seam_passed": end_mad is None or end_mad <= end_threshold,
            }
        )
    receipt = {
        "schema_version": 2,
        "artifact_type": "rendered_transport_run_sequence_v2",
        "status": "pass",
        "run_id": run_id,
        "case_ids": case_ids,
        "fill_clips": fill_rows,
        "transport_segments": segment_rows,
        "transport_fill_video_bindings": fill_bindings,
        "transport_segment_count": 10,
        "entry_segment_count": 1,
        "between_segment_count": 8,
        "exit_segment_count": 1,
        "endpoint_substitution": {
            "entry": {
                "accepted_start_endpoint": None,
                "rendered_state_indices": list(range(12)),
                "accepted_end_endpoint": "next_fill_first_pre_pour",
            },
            "between_fill": {
                "accepted_start_endpoint": "outgoing_fill_final",
                "rendered_state_indices": list(range(1, 12)),
                "accepted_end_endpoint": "next_fill_first_pre_pour",
            },
            "exit": {
                "accepted_start_endpoint": "outgoing_fill_final",
                "rendered_state_indices": list(range(1, 13)),
                "accepted_end_endpoint": None,
            },
        },
        "output_video": str(run_video.resolve()),
        "output_video_sha256": file_sha256(run_video),
        "output_frame_count": len(decoded),
        "fps": output_fps,
        "resolution": [640, 480],
        "encoded_transition_evidence": transition_rows,
        "encoded_transition_sequences_byte_unique": True,
        "encoded_seam_evidence": seam_rows,
        "encoded_transition_seams_passed": True,
        "qualified_fill_receipts_changed": False,
        "camera_and_lighting_changed": False,
    }
    path = products / f"{run_id}.run.json"
    _write_json(path, receipt)
    return path


@pytest.fixture
def accepted_fixture(tmp_path: Path) -> dict[str, object]:
    manifest_root = tmp_path / "manifests"
    artifacts = _manifest_bundle(manifest_root)
    collection = artifacts["full-transition-manifest.json"]
    run_roots = []
    fill_by_case: dict[str, dict[str, object]] = {}
    runtimes = [_runtime("l40s-host-a"), _runtime("l40s-host-b")]
    for index, binding in enumerate(collection["run_bindings"], start=1):
        run_root = tmp_path / f"merged-{index:02d}"
        run_roots.append(run_root)
        plan_path = manifest_root / binding["render_plan_path"]
        plan = artifacts[binding["render_plan_path"]]
        model = artifacts[binding["model_input_plan_path"]]
        model_by_case = {row["case_id"]: row for row in model["vial_records"]}
        schedules = plan["execution_schedules"]
        for schedule in schedules:
            case_id = schedule["case_binding"]["case_id"]
            _minimal_case(
                run_root / "cases" / case_id,
                case_id=case_id,
                schedule=schedule,
                model_row=model_by_case[case_id],
                render_plan=plan,
                plan_path=plan_path,
                runtime=runtimes[index - 1],
                colours=(
                    [(180, 60, 60), (90, 100, 110), (100, 110, 120)]
                    if case_id == "CASE_d52026e7df33b0cb"
                    else None
                ),
            )
            fill_path = run_root / "cases" / case_id / "app-output" / "three-pour.avi"
            fill_by_case[case_id] = {
                "path": str(fill_path.resolve()),
                "sha256": file_sha256(fill_path),
            }
        _write_json(
            run_root / "shard-merge.json",
            {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_shard_merge",
                "status": "pass",
                "plan_sha256": canonical_sha256(plan),
                "merged_case_count": 9,
                "case_ids": binding["case_ids"],
                "source_shards": [{"execution_host": runtimes[index - 1]["execution_host"]}],
                "full_resume_summary": {
                    "status": "pass",
                    "planned_case_count": 9,
                    "validated_completed_case_count": 9,
                    "case_results": [
                        {
                            "case_id": case_id,
                            "outcome": "recognized_completed",
                            "evidence": {
                                "artifact_validation": {
                                    "status": "pass", "failures": [], "warnings": []
                                }
                            },
                        }
                        for case_id in binding["case_ids"]
                    ],
                },
            },
        )
        _write_actual_readiness(run_root, runtimes[index - 1])

    leakage = manifest_root / "leakage-check-receipt.json"
    _write_json(
        leakage,
        {
            "artifact_type": "fluidmix-wp1-leakage-check-v1",
            "status": "pass",
            "split_version": "inn-7-splits-v3",
            "case_ids": [case for b in collection["run_bindings"] for case in b["case_ids"]],
            "recipe_ids": [recipe for b in collection["run_bindings"] for recipe in b["recipe_ids"]],
            "near_video5_count": 0,
            "leakage_failure_count": 0,
            "leakage_manifest_sha256": file_sha256(manifest_root / "leakage-manifest.csv"),
        },
    )
    boundary = manifest_root / "input-boundary-receipt.json"
    _write_json(
        boundary,
        {
            "artifact_type": "fluidmix-wp1-input-boundary-check-v1",
            "status": "pass",
            "forbidden_field_count": 0,
            "model_plan_canonical_sha256": [
                b["model_input_plan_canonical_sha256"] for b in collection["run_bindings"]
            ],
        },
    )
    transport_root = tmp_path / "transport"
    transport_paths = [
        _write_transport(
            transport_root,
            binding=binding,
            fill_by_case=fill_by_case,
            readiness_path=run_roots[index] / "wp1-fill-readiness.json",
        )
        for index, binding in enumerate(collection["run_bindings"])
    ]
    review_video = tmp_path / "review.mp4"
    review_video.write_bytes(b"lead-reviewed-all-runs")
    review = tmp_path / "review.json"
    _write_json(
        review,
        {
            "artifact_type": "fluidmix-wp1-review-acceptance-v1",
            "status": "accepted",
            "reviewer_role": "lead",
            "all_runs_viewed": True,
            "visually_distinct_presets": True,
            "review_video_path": str(review_video),
            "review_video_sha256": file_sha256(review_video),
            "run_product_sha256": [
                json.loads(path.read_text())["output_video_sha256"]
                for path in transport_paths
            ],
        },
    )
    return {
        "manifest_root": manifest_root,
        "run_roots": run_roots,
        "transport_paths": transport_paths,
        "review": review,
        "collection": collection,
    }


def test_visibility_decision_uses_all_three_receipt_bound_stages_and_keeps_plan_hashes_distinct(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    binding = accepted_fixture["collection"]["run_bindings"][0]
    plan = json.loads((root / binding["render_plan_path"]).read_text())
    by_case = {row["case_binding"]["case_id"]: row for row in plan["execution_schedules"]}
    model = json.loads((root / binding["model_input_plan_path"]).read_text())
    model_by_case = {row["case_id"]: row for row in model["vial_records"]}
    for case_id, colours in (
        ("CASE_d52026e7df33b0cb", [(180, 60, 60), (90, 100, 110), (100, 110, 120)]),
        ("CASE_ab6449c995c4b07d", [(60, 60, 60), (90, 100, 110), (100, 110, 120)]),
    ):
        shutil.rmtree(run_root / "cases" / case_id)
        _minimal_case(
            run_root / "cases" / case_id,
            case_id=case_id,
            schedule=by_case[case_id],
            model_row=model_by_case[case_id],
            render_plan=plan,
            plan_path=root / binding["render_plan_path"],
            runtime=_runtime("l40s-host-a"),
            colours=colours,
        )
    output = tmp_path / "visibility.json"
    decision = decide_wp1_visibility(
        manifest_root=root, run_root=run_root, output_path=output
    )
    assert decision["status"] == "pass"
    assert decision["decision"] == "original_1_5ml_visible"
    assert [row["stage"] for row in decision["stage_measurements"]] == ["t200", "t400", "t100"]
    assert decision["maximum_delta_e76"] >= 2.0
    assert decision["plan_binding"]["canonical_sha256"] == canonical_sha256(plan)
    assert decision["plan_binding"]["file_sha256"] == file_sha256(root / binding["render_plan_path"])
    assert decision["decision_sha256"] == canonical_sha256(
        {key: value for key, value in decision.items() if key != "decision_sha256"}
    )


def test_visibility_rejects_malformed_or_mismatched_masks(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bad_mask = target / "masks" / "bad.png"
    Image.fromarray(np.full((7, 8), 1, dtype=np.uint8), mode="L").save(bad_mask)
    receipt["semantics"]["pairs"][0]["mask_path"] = "masks/bad.png"
    receipt["semantics"]["pairs"][0]["mask_sha256"] = file_sha256(bad_mask)
    _write_json(receipt_path, receipt)
    decision = decide_wp1_visibility(
        manifest_root=root, run_root=run_root, output_path=tmp_path / "invalid-mask.json"
    )
    assert decision["status"] == "insufficient_evidence"
    assert any("mask" in reason for reason in decision["reasons"])


def test_visibility_accepts_real_count_driven_receipt_shape_and_rejects_legacy_proxy(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == 20
    assert receipt["render_path"] == "count_driven_twin_liquid_mesh"
    assert receipt["render"]["stage_frame_paths"] != [
        pair["rgb_path"] for pair in receipt["semantics"]["pairs"]
    ]
    decision = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "count-driven.json",
    )
    assert decision["status"] == "pass"

    receipt["semantics"]["pairs"][0]["liquid_representation"] = "vial_proxy"
    _write_json(receipt_path, receipt)
    failed = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "legacy-proxy.json",
    )
    assert failed["status"] == "insufficient_evidence"
    assert any("count_driven" in reason for reason in failed["reasons"])


def test_visibility_invalid_evidence_publishes_atomic_no_clobber_decision(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["semantics"]["pairs"][0]["mask_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    output = tmp_path / "insufficient.json"

    decision = decide_wp1_visibility(
        manifest_root=root, run_root=run_root, output_path=output
    )
    assert decision["status"] == "insufficient_evidence"
    assert decision["decision"] == "none"
    assert decision["reasons"]
    assert json.loads(output.read_text()) == decision
    assert not list(tmp_path.glob(".insufficient.json.partial-*"))

    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        decide_wp1_visibility(
            manifest_root=root, run_root=run_root, output_path=output
        )
    assert output.read_bytes() == original


def test_manifest_locator_traversal_and_symlink_escape_fail_closed(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    collection_path = root / "full-transition-manifest.json"
    collection = json.loads(collection_path.read_text())
    collection["run_bindings"][0]["render_plan_path"] = "../outside.json"
    _write_json(collection_path, collection)
    decision = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "traversal.json",
    )
    assert decision["status"] == "insufficient_evidence"
    assert any("escape" in reason or "traversal" in reason for reason in decision["reasons"])

    # Restore the frozen collection, then replace one bound RGB with an escaping symlink.
    _write_json(collection_path, accepted_fixture["collection"])
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    rgb_relative = receipt["semantics"]["pairs"][0]["rgb_path"]
    rgb = target / rgb_relative
    outside = tmp_path / "outside.png"
    shutil.copy2(rgb, outside)
    rgb.unlink()
    rgb.symlink_to(outside)
    receipt["semantics"]["pairs"][0]["rgb_sha256"] = file_sha256(outside)
    _write_json(receipt_path, receipt)
    symlinked = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "symlink.json",
    )
    assert symlinked["status"] == "insufficient_evidence"
    assert any("symlink" in reason or "escape" in reason for reason in symlinked["reasons"])


def test_aggregate_acceptance_and_package_exact_wp1_names(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    acceptance_path = tmp_path / "wp1-acceptance.json"
    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        review_receipt=accepted_fixture["review"],
        output_path=acceptance_path,
        repo_root=ROOT,
    )
    assert result["status"] == "pass"
    assert result["counts"] == {"runs": 2, "cases": 18, "recipes": 6, "keep": 9, "discard": 9}
    assert result["runtime"]["gpu_model"] == "NVIDIA L40S"
    assert result["runtime"]["hosts"] == ["l40s-host-a", "l40s-host-b"]
    assert result["transport"]["segment_count"] == 20

    package = tmp_path / "package"
    receipt = package_wp1_test_videos(
        acceptance_path=acceptance_path,
        output_root=package,
    )
    expected_runs = {
        "test-run01-S0439-S0440-S0441-amber_teal.mp4",
        "test-run02-S0442-S0443-S0444-crimson_indigo.mp4",
    }
    assert {path.name for path in (package / "videos" / "test").glob("test-run??-*.mp4") if "-vial" not in path.name} == expected_runs
    vial_paths = list((package / "videos" / "test").glob("test-run??-vial??-*.avi"))
    assert len(vial_paths) == 18
    assert all(path.read_bytes()[:4] == b"RIFF" and path.read_bytes()[8:12] == b"AVI " for path in vial_paths)
    assert (package / "videos" / "test" / "INDEX.csv").is_file()
    assert (package / "videos" / "test" / "README.md").is_file()
    assert receipt["status"] == "pass"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("mixed_gpu", "runtime_profile_mismatch"),
        ("mixed_runtime_hash", "runtime_profile_mismatch"),
        ("mixed_source_hash", "runtime_profile_mismatch"),
        ("missing_case", "case_set"),
        ("mismatched_video", "source_fill"),
        ("avi_relabelled_mp4", "RIFF/AVI"),
        ("relabelled", "label"),
        ("failed_transport", "transport"),
        ("absent_transport", "transport"),
    ],
)
def test_acceptance_fails_closed_for_adversarial_evidence(
    accepted_fixture: dict[str, object], tmp_path: Path, mutation: str, expected: str
) -> None:
    roots = list(accepted_fixture["run_roots"])
    transports = list(accepted_fixture["transport_paths"])
    if mutation in {"mixed_gpu", "mixed_runtime_hash", "mixed_source_hash"}:
        readiness_path = roots[1] / "wp1-fill-readiness.json"
        readiness = json.loads(readiness_path.read_text())
        host_path = Path(readiness["host_profile"]["path"])
        host = json.loads(host_path.read_text())
        if mutation == "mixed_gpu":
            host["gpu_model"] = "NVIDIA GB10"
        elif mutation == "mixed_runtime_hash":
            host["implementation"]["renderer_sha256"] = "f" * 64
        else:
            host["implementation"]["source_commit"] = "f" * 40
        _write_json(host_path, host)
        readiness["host_profile"]["sha256"] = file_sha256(host_path)
        _write_json(readiness_path, readiness)
    elif mutation == "missing_case":
        shutil.rmtree(roots[0] / "cases" / accepted_fixture["collection"]["run_bindings"][0]["case_ids"][0])
    elif mutation == "mismatched_video":
        receipt = json.loads(transports[0].read_text())
        receipt["fill_clips"][0]["sha256"] = "f" * 64
        _write_json(transports[0], receipt)
    elif mutation == "avi_relabelled_mp4":
        case_id = accepted_fixture["collection"]["run_bindings"][0]["case_ids"][0]
        app = roots[0] / "cases" / case_id / "app-output"
        receipt_path = app / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        source = app / receipt["render"]["video_path"]
        relabelled = app / "three-pour.mp4"
        shutil.copy2(source, relabelled)
        receipt["render"]["video_path"] = relabelled.name
        receipt["render"]["video_sha256"] = file_sha256(relabelled)
        _write_json(receipt_path, receipt)
    elif mutation == "relabelled":
        label = accepted_fixture["manifest_root"] / "run-01" / "label-provenance.json"
        value = json.loads(label.read_text())
        value["vial_labels"][0]["expected_disposition"] = "discard"
        _write_json(label, value)
    elif mutation == "failed_transport":
        receipt = json.loads(transports[0].read_text())
        receipt["status"] = "failed"
        _write_json(transports[0], receipt)
    elif mutation == "absent_transport":
        transports.pop()

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=roots,
        transport_receipts=transports,
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / f"{mutation}-wp1-acceptance.json",
        repo_root=ROOT,
    )
    assert result["status"] == "failed"
    assert any(expected in failure for failure in result["failures"])
