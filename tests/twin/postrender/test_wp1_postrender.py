from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping

import cv2
import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

C20_EFFECTIVE_PARTICLE_VOLUME_ML = 0.011272866937444935
C20_SOURCE_SAMPLER = "c20_3x3_lattice"

from twin.cohorts.full_transition_manifest import (  # noqa: E402
    build_default_full_transition_source_spec,
    build_full_transition_artifacts,
)
from twin.scene.nuisance_authoring import EXPECTED_LIGHT_PATHS  # noqa: E402
from twin.cohorts.one_run_manifest_v1 import (  # noqa: E402
    build_one_run_source_artifacts,
)
from twin.physx.run_variation import (  # noqa: E402
    bind_tank_colour_resolution_to_artifacts,
)
import twin.postrender.wp1_postrender as wp1_postrender  # noqa: E402
from twin.postrender.wp1_postrender import (  # noqa: E402
    canonical_sha256,
    decide_wp1_visibility,
    file_sha256,
    package_wp1_test_videos,
    register_wp1_dp3_fallback_case,
    validate_wp1_acceptance,
)
from twin.postrender.wp1_dp3_fallback import (  # noqa: E402
    build_dp3_fallback_plan_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_transport_provenance(fixture: Mapping[str, object]) -> None:
    """Refresh a synthetic trust anchor so a test can reach a deeper validation gate."""

    transport_paths = fixture["transport_paths"]
    transport_review_path = fixture["transport_review"]
    transport_review = json.loads(transport_review_path.read_text())
    products = transport_review["inspected_files"]["products"]
    product_root = Path(transport_review["product_root"])
    for row in products:
        row["sha256"] = file_sha256(product_root / row["path"])
    _write_json(transport_review_path, transport_review)
    inventory_path = fixture["inventory"]
    inventory = json.loads(inventory_path.read_text())
    for row, path in zip(
        inventory["transport_products"], transport_paths, strict=True
    ):
        row["receipt_sha256"] = file_sha256(path)
    inventory["independent_stitch_review"]["sha256"] = file_sha256(
        transport_review_path
    )
    _write_json(inventory_path, inventory)


def _write_png(path: Path, rgb: tuple[int, int, int], *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask:
        pixels = np.zeros((480, 640), dtype=np.uint8)
        pixels[180:300, 280:330] = 255
        Image.fromarray(pixels, mode="L").save(path)
    else:
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)
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
    case_marker = 30 + int(hashlib.sha256(case_id.encode()).hexdigest()[:4], 16) % 190
    video = app / "three-pour.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480)
    )
    assert writer.isOpened()
    for value in (100, 220):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[180:300, 280:330] = value
        frame[400:420, 40:60] = case_marker
        writer.write(frame)
    writer.release()
    pre_pour = app / "pre-pour-parity" / "f2-prefill-time-0001.png"
    parity_reference = app / "pre-pour-parity" / "twin-reference-time-0001.png"
    pre_pour.parent.mkdir(parents=True)
    pre_pour_pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pre_pour_pixels[180:300, 280:330] = 100
    pre_pour_pixels[400:420, 40:60] = case_marker
    assert cv2.imwrite(str(pre_pour), pre_pour_pixels)
    shutil.copy2(pre_pour, parity_reference)
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
        for marked_path in (rgb, stage_rgb):
            marked = cv2.imread(str(marked_path), cv2.IMREAD_COLOR)
            marked[400:420, 40:60] = case_marker
            assert cv2.imwrite(str(marked_path), marked)
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
            "recipe8_effective_particle_volume_ml": C20_EFFECTIVE_PARTICLE_VOLUME_ML,
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
            "video_timing_source_frame_steps": [12, 24],
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
        "pre_pour_parity": {
            "f2_prefill_path": pre_pour.relative_to(app).as_posix(),
            "f2_prefill_sha256": file_sha256(pre_pour),
            "reference_path": parity_reference.relative_to(app).as_posix(),
            "reference_sha256": file_sha256(parity_reference),
            "passed": True,
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
            "effective_particle_volume_ml": C20_EFFECTIVE_PARTICLE_VOLUME_ML,
            "source_sampler": C20_SOURCE_SAMPLER,
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
        "renderer_script_sha256": "deb0d7bb98ec6c45151ce6b0a6c58342a58c5903c0c09163fd33ddec9f723edb",
        "qualification_receipt_sha256": "3" * 64,
        "nuisance_receipt_sha256": "4" * 64,
        "python_runtime_sha256": "5" * 64,
        "source_recipe_manifest_sha256": "49ab190bf018f61c868f48ddb83dd96d7e1d0ff77080e68d62953d8f8f9ef49b",
        "source_split_spec_sha256": "e6e4b3b8d2c0932629ffae5872a58df9f4fb1b03612b0e842e53339a4d071bfb",
    }


def _write_actual_readiness(root: Path, runtime: dict[str, str]) -> Path:
    implementation = {
        "source_commit": "24889cad3cf9d7302622a4f844fb4b2a4977d4c9",
        "supervisor_sha256": "d34c8b5aaf94c90f69dcfe5a4e234d42be2d808609342355ed00c2f9f4cba01e",
        "renderer_sha256": "deb0d7bb98ec6c45151ce6b0a6c58342a58c5903c0c09163fd33ddec9f723edb",
        "offline_validator_sha256": "16b8c53fc00373af9fe94fa0305b4482588a1d77a9b6fab513c9ab6e0aa72ed7",
        "occlusion_profile_sha256": "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e",
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
    modes = ["entry"] + ["between"] * 8 + ["exit"]
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

    def shared_state(mode: str, index: int, segment_index: int) -> dict[str, Path]:
        state_root = shared / f"segment-{segment_index:02d}" / f"state-{index:02d}"
        if not state_root.exists():
            state_root.mkdir(parents=True)
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[180:300, 280:330] = state_value(mode, index)
            rgb[400:420, 40:60] = 20 * (segment_index + 1)
            assert cv2.imwrite(str(state_root / "rgb.png"), rgb)
            instance = np.zeros((480, 640), dtype=np.uint32)
            if mode != "entry" and index == 0:
                instance[180:300, 100:120] = 1
            if mode in {"entry", "between"} and index in {11, 12}:
                instance[180:300, 307:317] = 2
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
            result = {"video_path": str(video.resolve()), "video_sha256": file_sha256(video)}
            if which == "start":
                result["frame_index"] = -1
                fill_receipt = json.loads((video.parent / "receipt.json").read_text())
                final_pair = max(
                    fill_receipt["semantics"]["pairs"],
                    key=lambda pair: pair["physics_step"],
                )
                endpoint_path = video.parent / final_pair["rgb_path"]
                result.update(
                    path=str(endpoint_path.resolve()),
                    sha256=final_pair["rgb_sha256"],
                )
            else:
                receipt_path = video.parent / "receipt.json"
                fill_receipt = json.loads(receipt_path.read_text())
                parity = fill_receipt["pre_pour_parity"]
                render = fill_receipt["render"]
                endpoint_path = video.parent / parity["f2_prefill_path"]
                reference_path = video.parent / parity["reference_path"]
                result.update(
                    path=str(endpoint_path.resolve()),
                    sha256=parity["f2_prefill_sha256"],
                    physics_step=0,
                    continuation_frame_index=0,
                    continuation_physics_step=render["video_timing_source_frame_steps"][0],
                    decoded_video_frame_count=render["video_decoded_frame_count"],
                    included_in_video=False,
                    parity_reference={
                        "path": str(reference_path.resolve()),
                        "sha256": parity["reference_sha256"],
                        "receipt_field": "pre_pour_parity.reference_path",
                    },
                    provenance={
                        "continuation_renderer": render["video_renderer"],
                        "pre_pour_renderer": render["stage_still_renderer"],
                        "receipt_field": "pre_pour_parity.f2_prefill_path",
                        "receipt_path": str(receipt_path.resolve()),
                        "receipt_sha256": file_sha256(receipt_path),
                        "video_codec": render["video_codec"],
                    },
                    video_timing_source_frame_steps=render["video_timing_source_frame_steps"],
                )
            return result

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
            source_files = shared_state(mode, state_index, index)
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
        end_endpoint = None if incoming_id is None else endpoint(incoming_id, "end")
        end_endpoint_frame = (
            None
            if end_endpoint is None
            else cv2.imread(str(end_endpoint["path"]), cv2.IMREAD_COLOR)
        )
        start_mad = (
            None
            if start_frames is None
            else float(np.mean(np.abs(start_frames[-1].astype(np.float32) - decoded_states[1].astype(np.float32))))
        )
        end_mad = (
            None
            if end_endpoint_frame is None
            else float(np.mean(np.abs(decoded_states[11].astype(np.float32) - end_endpoint_frame.astype(np.float32))))
        )
        start_control = (
            None
            if start_frames is None
            else float(np.mean(np.abs(start_frames[-1].astype(np.float32) - decoded_states[0].astype(np.float32))))
        )
        end_control = (
            None
            if end_endpoint_frame is None
            else float(np.mean(np.abs(decoded_states[12].astype(np.float32) - end_endpoint_frame.astype(np.float32))))
        )
        continuation_mad = (
            None
            if end_endpoint_frame is None
            else float(np.mean(np.abs(end_endpoint_frame.astype(np.float32) - end_frames[0].astype(np.float32))))
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
            "start_seam_actual_mad": start_mad,
            "start_seam_control_mad": start_control,
            "start_seam_residual_mad": None if start_mad is None else start_mad - start_control,
            "start_seam_threshold_p99": p99(start_frames),
            "start_seam_control_ceiling_mad": 12.0,
            "start_seam_passed": True,
            "end_seam_actual_mad": end_mad,
            "end_seam_control_mad": end_control,
            "end_seam_residual_mad": None if end_mad is None else end_mad - end_control,
            "end_seam_threshold_p99": p99(end_frames),
            "end_seam_control_ceiling_mad": 12.0,
            "end_seam_passed": True,
            "continuation_seam_mad": continuation_mad,
            "continuation_seam_ceiling_mad": 12.0,
            "continuation_seam_passed": True,
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

    pre_pour_frames = {
        case_id: cv2.imread(
            str(Path(str(fill_by_case[case_id]["path"])).parent / "pre-pour-parity" / "f2-prefill-time-0001.png"),
            cv2.IMREAD_COLOR,
        )
        for case_id in case_ids
    }
    sequence = list(segment_frames[0][:12]) + [pre_pour_frames[case_ids[0]]] + list(fill_frames[case_ids[0]])
    ranges = [(0, 13)]
    for index, case_id in enumerate(case_ids[1:], start=1):
        start = len(sequence) - 1
        sequence.extend(segment_frames[index][1:12])
        sequence.append(pre_pour_frames[case_id])
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
        raw_states = segment_frames[int(layout_row["segment_index"])]
        start_mad = None if outgoing is None else float(np.mean(np.abs(frames[0].astype(np.float32) - frames[1].astype(np.float32))))
        end_mad = None if incoming is None else float(np.mean(np.abs(frames[11].astype(np.float32) - frames[12].astype(np.float32))))
        start_control = None if outgoing is None else float(np.mean(np.abs(fill_frames[str(outgoing)][-1].astype(np.float32) - raw_states[0].astype(np.float32))))
        end_control = None if incoming is None else float(np.mean(np.abs(raw_states[12].astype(np.float32) - frames[12].astype(np.float32))))
        continuation_mad = None if incoming is None else float(np.mean(np.abs(frames[12].astype(np.float32) - decoded[stop].astype(np.float32))))
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
                "start_seam_actual_mad": start_mad,
                "start_seam_control_mad": start_control,
                "start_seam_residual_mad": None if start_mad is None else start_mad - start_control,
                "start_seam_threshold_p99": start_threshold,
                "start_seam_control_ceiling_mad": 12.0,
                "start_seam_passed": start_mad is None or (start_mad - start_control <= start_threshold and start_control <= 12.0),
                "end_seam_actual_mad": end_mad,
                "end_seam_control_mad": end_control,
                "end_seam_residual_mad": None if end_mad is None else end_mad - end_control,
                "end_seam_threshold_p99": end_threshold,
                "end_seam_control_ceiling_mad": 12.0,
                "end_seam_passed": end_mad is None or (end_mad - end_control <= end_threshold and end_control <= 12.0),
                "continuation_seam_mad": continuation_mad,
                "continuation_seam_ceiling_mad": 12.0,
                "continuation_seam_passed": continuation_mad is None or continuation_mad <= 12.0,
            }
        )
    receipt = {
        "schema_version": 4,
        "artifact_type": "rendered_transport_run_sequence_v4",
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
            "between": {
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
        "packaging_frame_count_semantics": {
            "accepted_fill_frame_zero_dropped": False,
            "accepted_fill_frames_preserved": sum(
                len(fill_frames[case_id]) for case_id in case_ids
            ),
            "authored_transport_state_count_per_segment": 13,
            "packaged_transition_frame_count_per_segment": 13,
            "true_pre_pour_frames_added": 9,
        },
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
    transport_review = tmp_path / "transport-final-review.json"
    _write_json(
        transport_review,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_wp1_canonical_stitch_final_output_review_v1",
            "status": "pass",
            "verdict": "pass_with_known_limitations",
            "product_root": str(transport_root.resolve()),
            "inspected_files": {
                "products": [
                    {
                        "path": str(path.relative_to(transport_root)),
                        "sha256": file_sha256(path),
                    }
                    for receipt_path in transport_paths
                    for path in (
                        Path(json.loads(receipt_path.read_text())["output_video"]),
                        receipt_path,
                    )
                ]
            },
        },
    )
    inventory = tmp_path / "finalization-inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix-wp1-finalization-inventory-v1",
            "status": "prepared_awaiting_independent_lead_review",
            "owner_thread_id": "01a18128-test",
            "render_performed": False,
            "canonical_inputs_mutated": False,
            "manifest": {
                "root": str(manifest_root.resolve()),
                "collection_sha256": file_sha256(
                    manifest_root / "full-transition-manifest.json"
                ),
            },
            "aggregate_runs": [
                {
                    "run_id": binding["run_id"],
                    "root": str(run_root.resolve()),
                    "readiness_sha256": file_sha256(
                        run_root / "wp1-fill-readiness.json"
                    ),
                }
                for binding, run_root in zip(
                    collection["run_bindings"], run_roots, strict=True
                )
            ],
            "transport_products": [
                {
                    "run_id": binding["run_id"],
                    "video_path": receipt["output_video"],
                    "video_sha256": receipt["output_video_sha256"],
                    "receipt_path": str(path.resolve()),
                    "receipt_sha256": file_sha256(path),
                }
                for binding, path in zip(
                    collection["run_bindings"], transport_paths, strict=True
                )
                for receipt in [json.loads(path.read_text())]
            ],
            "independent_stitch_review": {
                "path": str(transport_review.resolve()),
                "sha256": file_sha256(transport_review),
            },
        },
    )
    return {
        "manifest_root": manifest_root,
        "run_roots": run_roots,
        "transport_paths": transport_paths,
        "transport_review": transport_review,
        "review": review,
        "inventory": inventory,
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


def _real_fallback_case(
    tmp_path: Path,
    accepted_fixture: Mapping[str, object],
    *,
    ordinal: int,
    colours: list[tuple[int, int, int]],
) -> Path:
    binding = accepted_fixture["collection"]["run_bindings"][0]
    frozen_plan = json.loads(
        (accepted_fixture["manifest_root"] / binding["render_plan_path"]).read_text()
    )
    artifacts = build_dp3_fallback_plan_artifacts(
        frozen_plan, target_ordinal=ordinal
    )
    plan_root = tmp_path / f"fallback-plan-{ordinal:02d}"
    plan_root.mkdir()
    for name, value in artifacts.items():
        _write_json(plan_root / name, value)
    plan = artifacts["render-schedule-plan.json"]
    schedule = plan["execution_schedules"][0]
    model_row = artifacts["model-input-plan.json"]["vial_records"][0]
    case_id = schedule["case_binding"]["case_id"]
    case_root = tmp_path / case_id
    runtime = _runtime(f"l40s-fallback-{ordinal:02d}")
    _minimal_case(
        case_root,
        case_id=case_id,
        schedule=schedule,
        model_row=model_row,
        render_plan=plan,
        plan_path=plan_root / "render-schedule-plan.json",
        runtime=runtime,
        colours=colours,
    )
    receipt_path = case_root / "app-output" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    seed = plan["batches"][0]["render_seed_by_vial"][0]
    nuisance_id = plan["batches"][0]["nuisance_group_ids"][0]
    nuisance_sha = ("5" if ordinal == 5 else "9") * 64
    receipt["arguments"].update(
        {
            "seed": seed,
            "nuisance_receipt_sha256": nuisance_sha,
        }
    )
    receipt["execution_schedule"] = {
        "case_id": case_id,
        "cohort_plan_sha256": canonical_sha256(plan),
        "schedule_file_sha256": file_sha256(
            plan_root / "render-schedule-plan.json"
        ),
        "source_run_spec_sha256": plan["source_run_spec_sha256"],
        "schedule_sha256": canonical_sha256(schedule),
        "vial_record_sha256": canonical_sha256(model_row),
        "renderer_case_binding": deepcopy(schedule["case_binding"]),
        "model_input_record": deepcopy(model_row),
        "render_seed": seed,
        "source_pure_segment_schedule": True,
        "particle_ids_and_counts_match_every_event": True,
        "event_compositions_match": True,
        # The canonical renderer currently reports this aggregate field false even
        # when the individually bound programmed volumes are valid.  Acceptance
        # therefore proves the actual schedule/model/pours instead of inventing a
        # fixture-only gate value.
        "all_three_programmed_volume_numeric_gates_passed": False,
    }
    receipt["plan"] = {"pours": deepcopy(schedule["pours"])}
    receipt["camera"] = {
        "path": "/FillStation/Cameras/FillCam",
        "effective_render_state": {
            "projection": "perspective",
            "world_transform_matrix": [float(index) for index in range(16)],
        },
    }
    lights = {
        path: {
            "type_name": "RectLight",
            "intensity": 1500.0,
            "exposure": 0.0,
            "transform_path": path.rsplit("/", 1)[0],
            "translate": [0.0, 0.0, 1.0],
            "rotate_xyz_degrees": [0.0, 0.0, 0.0],
        }
        for path in EXPECTED_LIGHT_PATHS
    }
    receipt["scene_contract"] = {
        "effective_light_state": deepcopy(lights),
        "carrier_occlusion_profile": {
            "source_stage_sha256": runtime["stage_sha256"],
            "source_geometry_layer_sha256": (
                "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
            ),
            "profile_file_sha256": (
                "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e"
            ),
            "camera_effective_state_attestation": {"passed": True},
            "occluder_effective_state_attestation": {"passed": True},
        },
    }
    receipt["nuisance_runtime"] = {
        "status": "applied",
        "seed": seed,
        "case_id": nuisance_id,
        "source_receipt_sha256": nuisance_sha,
        "authored_light_properties": deepcopy(lights),
        "readback_light_properties": deepcopy(lights),
    }
    _write_json(receipt_path, receipt)
    offline_path = case_root / "offline-validator.json"
    _write_json(
        offline_path,
        {
            "status": "pass",
            "verified_artifact_count": 75,
            "failures": [],
            "warnings": [],
            "offline_revalidations": [
                {
                    "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                    "passed": True,
                }
            ],
        },
    )
    case_result_path = case_root / "case-result.json"
    case_result = json.loads(case_result_path.read_text())
    case_result["evidence"]["artifact_validation"] = json.loads(
        offline_path.read_text()
    )
    _write_json(case_result_path, case_result)
    host_profile = accepted_fixture["run_roots"][0] / "readiness-evidence" / "host-profile.json"
    register_wp1_dp3_fallback_case(
        target_ordinal=ordinal,
        manifest_root=accepted_fixture["manifest_root"],
        repo_root=ROOT,
        plan_root=plan_root,
        case_root=case_root,
        host_profile_path=host_profile,
        offline_validation_path=offline_path,
        output_path=case_root / "fallback-case-acceptance.json",
    )
    return case_root


def _replace_fallback_bundle_with_self_consistent_wrong_derivation(
    plan_root: Path,
    accepted_fixture: Mapping[str, object],
    mutation: str,
) -> None:
    source = json.loads((plan_root / "source-run-spec.json").read_text())
    plan = json.loads((plan_root / "render-schedule-plan.json").read_text())
    palette = deepcopy(plan["tank_colour_resolution"])
    if mutation == "programmed_ml":
        source["recipes"][0]["programmed_ml"]["t200"] += 1.0
    elif mutation == "run_id":
        source["run_id"] = "WP1-DP3-FALLBACK-WRONG-RUN"
    else:
        run2 = accepted_fixture["collection"]["run_bindings"][1]
        run2_plan = json.loads(
            (
                accepted_fixture["manifest_root"] / run2["render_plan_path"]
            ).read_text()
        )
        palette = deepcopy(run2_plan["tank_colour_resolution"])
    derived = bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(source), palette
    )
    values = {"source-run-spec.json": source, **derived}
    for name, value in values.items():
        _write_json(plan_root / name, value)


def test_acceptance_consumes_only_registered_fresh_dp3_fallback_pair(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    run_root = accepted_fixture["run_roots"][0]
    reference = run_root / "cases" / "CASE_ab6449c995c4b07d" / "app-output" / "receipt.json"
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output" / "receipt.json"
    reference_receipt = json.loads(reference.read_text())
    target_receipt = json.loads(target.read_text())
    for left, right in zip(
        target_receipt["semantics"]["pairs"],
        reference_receipt["semantics"]["pairs"],
        strict=True,
    ):
        shutil.copy2(reference.parent / right["rgb_path"], target.parent / left["rgb_path"])
        left["rgb_sha256"] = file_sha256(target.parent / left["rgb_path"])
    _write_json(target, target_receipt)
    roots = [
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=5,
            colours=[(20, 210, 40), (23, 213, 43), (26, 216, 46)],
        ),
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=9,
            colours=[(220, 30, 180), (223, 33, 183), (226, 36, 186)],
        ),
    ]

    result = wp1_postrender._fallback_visibility_payload(
        original=wp1_postrender._visibility_payload(
            manifest_root=accepted_fixture["manifest_root"],
            run_root=accepted_fixture["run_roots"][0],
            repo_root=ROOT,
        ),
        manifest_root=accepted_fixture["manifest_root"],
        run_root=accepted_fixture["run_roots"][0],
        fallback_case_roots=roots,
        repo_root=ROOT,
    )

    assert result["status"] == "approval_required"
    assert result["decision"] == "cross_case_aggregation_policy_required"
    assert result["original"]["status"] == "fallback_required"
    assert [row["target_vial_ordinal"] for row in result["fallback_pairs"]] == [5, 9]
    assert all(row["maximum_delta_e76"] >= 2.0 for row in result["fallback_pairs"])


def test_fallback_rejects_copied_1_5ml_renderer_execution_binding(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    roots = [
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=5,
            colours=[(20, 210, 40), (23, 213, 43), (26, 216, 46)],
        ),
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=9,
            colours=[(220, 30, 180), (223, 33, 183), (226, 36, 186)],
        ),
    ]
    for root in roots:
        acceptance_path = root / "fallback-case-acceptance.json"
        acceptance = json.loads(acceptance_path.read_text())
        renderer_path = Path(acceptance["renderer_receipt"]["path"])
        renderer = json.loads(renderer_path.read_text())
        renderer["execution_schedule"]["renderer_case_binding"][
            "residual_volume_ml"
        ] = 1.5
        _write_json(renderer_path, renderer)
        acceptance["renderer_receipt"]["sha256"] = file_sha256(renderer_path)
        acceptance["acceptance_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in acceptance.items()
                if key != "acceptance_sha256"
            }
        )
        _write_json(acceptance_path, acceptance)

    with pytest.raises(ValueError, match="renderer_case_binding"):
        wp1_postrender._fallback_visibility_payload(
            original={"status": "fallback_required", "threshold_delta_e76": 2.0},
            manifest_root=accepted_fixture["manifest_root"],
            run_root=accepted_fixture["run_roots"][0],
            fallback_case_roots=roots,
            repo_root=ROOT,
        )


def test_registration_rejects_1_5ml_renderer_binding_before_publication(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = _real_fallback_case(
        tmp_path,
        accepted_fixture,
        ordinal=5,
        colours=[(20, 210, 40), (23, 213, 43), (26, 216, 46)],
    )
    acceptance_path = root / "fallback-case-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    acceptance_path.unlink()
    renderer_path = Path(acceptance["renderer_receipt"]["path"])
    renderer = json.loads(renderer_path.read_text())
    renderer["execution_schedule"]["renderer_case_binding"][
        "residual_volume_ml"
    ] = 1.5
    _write_json(renderer_path, renderer)

    with pytest.raises(ValueError, match="schedule|binding"):
        register_wp1_dp3_fallback_case(
            target_ordinal=5,
            manifest_root=accepted_fixture["manifest_root"],
            repo_root=ROOT,
            plan_root=acceptance["plan_root"],
            case_root=root,
            host_profile_path=acceptance["host_profile"]["path"],
            offline_validation_path=acceptance["offline_validation"]["path"],
            output_path=acceptance_path,
        )
    assert not acceptance_path.exists()


def test_registration_rejects_unrelated_passed_offline_revalidation(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = _real_fallback_case(
        tmp_path,
        accepted_fixture,
        ordinal=5,
        colours=[(20, 210, 40), (23, 213, 43), (26, 216, 46)],
    )
    acceptance_path = root / "fallback-case-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    acceptance_path.unlink()
    offline_path = Path(acceptance["offline_validation"]["path"])
    offline = json.loads(offline_path.read_text())
    offline["offline_revalidations"][0]["revalidation_type"] = (
        "unrelated_checksum_revalidation"
    )
    _write_json(offline_path, offline)
    case_result_path = root / "case-result.json"
    case_result = json.loads(case_result_path.read_text())
    case_result["evidence"]["artifact_validation"] = offline
    _write_json(case_result_path, case_result)

    with pytest.raises(ValueError, match="offline runtime gate"):
        register_wp1_dp3_fallback_case(
            target_ordinal=5,
            manifest_root=accepted_fixture["manifest_root"],
            repo_root=ROOT,
            plan_root=acceptance["plan_root"],
            case_root=root,
            host_profile_path=acceptance["host_profile"]["path"],
            offline_validation_path=offline_path,
            output_path=acceptance_path,
        )
    assert not acceptance_path.exists()


@pytest.mark.parametrize("mutation", ["programmed_ml", "run_id", "run2_palette"])
def test_registration_rejects_self_consistent_bundle_not_derived_from_frozen_run1(
    accepted_fixture: dict[str, object], tmp_path: Path, mutation: str
) -> None:
    root = _real_fallback_case(
        tmp_path,
        accepted_fixture,
        ordinal=5,
        colours=[(20, 210, 40), (23, 213, 43), (26, 216, 46)],
    )
    acceptance_path = root / "fallback-case-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    acceptance_path.unlink()
    _replace_fallback_bundle_with_self_consistent_wrong_derivation(
        Path(acceptance["plan_root"]), accepted_fixture, mutation
    )

    with pytest.raises(ValueError, match="exact frozen run-1 derivation"):
        register_wp1_dp3_fallback_case(
            target_ordinal=5,
            manifest_root=accepted_fixture["manifest_root"],
            repo_root=ROOT,
            plan_root=acceptance["plan_root"],
            case_root=root,
            host_profile_path=acceptance["host_profile"]["path"],
            offline_validation_path=acceptance["offline_validation"]["path"],
            output_path=acceptance_path,
        )
    assert not acceptance_path.exists()


@pytest.mark.parametrize("mutation", ["programmed_ml", "run_id", "run2_palette"])
def test_consumption_rejects_rehashed_self_consistent_bundle_not_from_frozen_run1(
    accepted_fixture: dict[str, object], tmp_path: Path, mutation: str
) -> None:
    roots = [
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=5,
            colours=[(10, 200, 30), (13, 203, 33), (16, 206, 36)],
        ),
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=9,
            colours=[(210, 20, 170), (213, 23, 173), (216, 26, 176)],
        ),
    ]
    acceptance_path = roots[0] / "fallback-case-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    plan_root = Path(acceptance["plan_root"])
    _replace_fallback_bundle_with_self_consistent_wrong_derivation(
        plan_root, accepted_fixture, mutation
    )
    for name, binding in acceptance["plan_artifacts"].items():
        path = plan_root / name
        value = json.loads(path.read_text())
        binding["file_sha256"] = file_sha256(path)
        binding["canonical_sha256"] = canonical_sha256(value)
    acceptance["acceptance_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in acceptance.items()
            if key != "acceptance_sha256"
        }
    )
    _write_json(acceptance_path, acceptance)

    with pytest.raises(ValueError, match="exact frozen run-1 derivation"):
        wp1_postrender._fallback_visibility_payload(
            original={"status": "fallback_required", "threshold_delta_e76": 2.0},
            manifest_root=accepted_fixture["manifest_root"],
            run_root=accepted_fixture["run_roots"][0],
            fallback_case_roots=roots,
            repo_root=ROOT,
        )


def test_aggregate_rejects_fallback_roots_after_original_dp3_pass(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        fallback_case_roots=[tmp_path / "unused-a", tmp_path / "unused-b"],
        output_path=tmp_path / "unexpected-fallback.json",
        repo_root=ROOT,
    )

    assert result["status"] == "failed"
    assert result["failures"] == [
        "DP3 fallback cases are allowed only after fallback_required"
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "exactly two"),
        ("extra", "exactly two"),
        ("duplicate", "unique"),
        ("wrong_recipe", "recipe"),
        ("wrong_ordinal", "ordinal"),
        ("wrong_residue", "residue"),
        ("wrong_profile", "lineage"),
        ("wrong_hash", "hash"),
        ("wrong_plan", "plan|rebuild|hash"),
        ("wrong_schedule", "schedule|binding"),
        ("wrong_runtime_residue", "schedule|binding"),
        ("wrong_context", "schedule|binding"),
        ("wrong_seed", "schedule|binding"),
        ("wrong_nuisance", "lineage"),
        ("wrong_camera", "lineage"),
        ("wrong_light", "lineage"),
        ("failed_gate", "gate"),
        ("failed_revalidation", "gate"),
        ("missing_revalidation", "gate"),
        ("unrelated_revalidation", "gate"),
    ],
)
def test_dp3_fallback_contract_mutations_fail_closed(
    accepted_fixture: dict[str, object], tmp_path: Path, mutation: str, expected: str
) -> None:
    roots = [
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=5,
            colours=[(10, 200, 30), (13, 203, 33), (16, 206, 36)],
        ),
        _real_fallback_case(
            tmp_path,
            accepted_fixture,
            ordinal=9,
            colours=[(210, 20, 170), (213, 23, 173), (216, 26, 176)],
        ),
    ]
    if mutation == "missing":
        roots.pop()
    elif mutation == "extra":
        roots.append(roots[0])
    elif mutation == "duplicate":
        roots[1] = roots[0]
    else:
        receipt_path = roots[0] / "fallback-case-acceptance.json"
        receipt = json.loads(receipt_path.read_text())
        if mutation == "wrong_recipe":
            receipt["recipe_id"] = "S0441"
        elif mutation == "wrong_ordinal":
            receipt["target_vial_ordinal"] = 9
        elif mutation == "wrong_residue":
            receipt["residual_volume_ml"] = 1.5
        elif mutation == "wrong_profile":
            receipt["lineage"]["gpu_model"] = "NVIDIA GB10"
        elif mutation == "wrong_hash":
            receipt["renderer_receipt"]["sha256"] = "f" * 64
        elif mutation == "wrong_plan":
            plan_path = Path(
                receipt["plan_artifacts"]["render-schedule-plan.json"]["path"]
            )
            plan = json.loads(plan_path.read_text())
            plan["execution_schedules"][0]["case_binding"][
                "residual_volume_ml"
            ] = 1.5
            _write_json(plan_path, plan)
            receipt["plan_artifacts"]["render-schedule-plan.json"].update(
                file_sha256=file_sha256(plan_path),
                canonical_sha256=canonical_sha256(plan),
            )
        elif mutation in {
            "wrong_context",
            "wrong_schedule",
            "wrong_runtime_residue",
            "wrong_seed",
            "wrong_nuisance",
            "wrong_camera",
            "wrong_light",
        }:
            renderer_path = Path(receipt["renderer_receipt"]["path"])
            renderer = json.loads(renderer_path.read_text())
            if mutation == "wrong_schedule":
                renderer["execution_schedule"]["schedule_sha256"] = "f" * 64
            elif mutation == "wrong_runtime_residue":
                renderer["plan"]["pours"][0]["leading_residual_volume_ml"] = 1.5
            elif mutation == "wrong_context":
                renderer["execution_schedule"]["renderer_case_binding"][
                    "previous_recipe_id"
                ] = "S0439"
            elif mutation == "wrong_seed":
                renderer["execution_schedule"]["render_seed"] += 1
            elif mutation == "wrong_nuisance":
                renderer["nuisance_runtime"]["case_id"] = "WRONG-NUISANCE"
            elif mutation == "wrong_camera":
                renderer["camera"]["effective_render_state"]["projection"] = (
                    "orthographic"
                )
            else:
                renderer["scene_contract"]["effective_light_state"][
                    EXPECTED_LIGHT_PATHS[0]
                ]["intensity"] += 1.0
            _write_json(renderer_path, renderer)
            receipt["renderer_receipt"]["sha256"] = file_sha256(renderer_path)
        elif mutation in {
            "failed_gate",
            "failed_revalidation",
            "missing_revalidation",
            "unrelated_revalidation",
        }:
            offline_path = Path(receipt["offline_validation"]["path"])
            offline = json.loads(offline_path.read_text())
            if mutation == "failed_gate":
                offline["status"] = "insufficient_evidence"
                offline["failures"] = ["containment"]
            elif mutation == "failed_revalidation":
                offline["offline_revalidations"][0]["passed"] = False
            elif mutation == "unrelated_revalidation":
                offline["offline_revalidations"][0]["revalidation_type"] = (
                    "unrelated_checksum_revalidation"
                )
            else:
                offline["offline_revalidations"] = []
            _write_json(offline_path, offline)
            receipt["offline_validation"]["sha256"] = file_sha256(offline_path)
            if mutation == "unrelated_revalidation":
                case_result_path = roots[0] / "case-result.json"
                case_result = json.loads(case_result_path.read_text())
                case_result["evidence"]["artifact_validation"] = offline
                _write_json(case_result_path, case_result)
        receipt["acceptance_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "acceptance_sha256"
            }
        )
        _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match=expected):
        wp1_postrender._fallback_visibility_payload(
            original={"status": "fallback_required", "threshold_delta_e76": 2.0},
            manifest_root=accepted_fixture["manifest_root"],
            run_root=accepted_fixture["run_roots"][0],
            fallback_case_roots=roots,
            repo_root=ROOT,
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
    binding = accepted_fixture["collection"]["run_bindings"][0]
    plan = json.loads((root / binding["render_plan_path"]).read_text())
    assert plan["particle_resolution_profile"]["effective_particle_volume_ml"] is None
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == 20
    assert receipt["render_path"] == "count_driven_twin_liquid_mesh"
    assert receipt["particle_resolution_profile"]["effective_particle_volume_ml"] == C20_EFFECTIVE_PARTICLE_VOLUME_ML
    assert receipt["particle_resolution_profile"]["source_sampler"] == C20_SOURCE_SAMPLER
    assert receipt["render"]["stage_frame_paths"] != [
        pair["rgb_path"] for pair in receipt["semantics"]["pairs"]
    ]
    decision = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "count-driven.json",
    )
    assert decision["status"] == "pass"

    original = deepcopy(receipt)
    receipt["particle_resolution_profile"]["effective_particle_volume_ml"] = 0.02
    receipt["arguments"]["recipe8_effective_particle_volume_ml"] = 0.02
    _write_json(receipt_path, receipt)
    arbitrary = decide_wp1_visibility(
        manifest_root=root,
        run_root=run_root,
        output_path=tmp_path / "arbitrary-c20-volume.json",
    )
    assert arbitrary["status"] == "insufficient_evidence"

    receipt = original
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
    composer = json.loads(accepted_fixture["transport_paths"][0].read_text())
    assert [row["mode"] for row in composer["transport_segments"]] == [
        "entry",
        *("between" for _ in range(8)),
        "exit",
    ]
    assert set(composer["endpoint_substitution"]) == {"entry", "between", "exit"}
    entry_receipt = json.loads(
        Path(composer["transport_segments"][0]["receipt_path"]).read_text()
    )
    entry_build = json.loads(
        (Path(entry_receipt["segment_dir"]) / "transport-build-receipt.json").read_text()
    )
    assert set(entry_build["adjacent_endpoints"]["accepted_end"]) == {
        "continuation_frame_index",
        "continuation_physics_step",
        "decoded_video_frame_count",
        "included_in_video",
        "parity_reference",
        "path",
        "physics_step",
        "provenance",
        "sha256",
        "video_path",
        "video_sha256",
        "video_timing_source_frame_steps",
    }
    between_receipt = json.loads(
        Path(composer["transport_segments"][1]["receipt_path"]).read_text()
    )
    between_build = json.loads(
        (Path(between_receipt["segment_dir"]) / "transport-build-receipt.json").read_text()
    )
    assert set(between_build["adjacent_endpoints"]["accepted_start"]) == {
        "path",
        "sha256",
        "video_path",
        "video_sha256",
        "frame_index",
    }
    acceptance_path = tmp_path / "wp1-acceptance.json"
    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=acceptance_path,
        repo_root=ROOT,
    )
    assert result["status"] == "pass", result["failures"]
    assert result["counts"] == {"runs": 2, "cases": 18, "recipes": 6, "keep": 9, "discard": 9}
    assert result["runtime"]["gpu_model"] == "NVIDIA L40S"
    assert result["runtime"]["hosts"] == ["l40s-host-a", "l40s-host-b"]
    assert result["transport"]["segment_count"] == 20
    for run in result["transport"]["receipts"]:
        assert run["packaging_frame_count_semantics"] == {
            "accepted_fill_frame_zero_dropped": False,
            "accepted_fill_frames_preserved": sum(
                row["frame_count"] for row in run["fill_clips"]
            ),
            "authored_transport_state_count_per_segment": 13,
            "packaged_transition_frame_count_per_segment": 13,
            "true_pre_pour_frames_added": 9,
        }

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


def test_acceptance_accepts_tree_identical_relocated_case_copies(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    aggregate_roots = []
    for index, canonical_root in enumerate(accepted_fixture["run_roots"], start=1):
        aggregate_root = tmp_path / f"aggregate-{index:02d}"
        shutil.copytree(canonical_root, aggregate_root)
        readiness_path = aggregate_root / "wp1-fill-readiness.json"
        readiness = json.loads(readiness_path.read_text())
        for key in (
            "supervisor_case_result",
            "offline_validator_result",
            "host_profile",
        ):
            canonical_path = Path(readiness[key]["path"])
            aggregate_path = aggregate_root / "readiness-evidence" / canonical_path.name
            assert file_sha256(aggregate_path) == readiness[key]["sha256"]
            readiness[key]["path"] = str(aggregate_path.resolve())
        _write_json(readiness_path, readiness)
        aggregate_roots.append(aggregate_root)

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=aggregate_roots,
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / "relocated-wp1-acceptance.json",
        repo_root=ROOT,
    )

    assert result["status"] == "pass", result["failures"]

    case_id = accepted_fixture["collection"]["run_bindings"][0]["case_ids"][0]
    canonical_marker = accepted_fixture["run_roots"][0] / "cases" / case_id / "exit-code.txt"
    canonical_marker.write_text("changed after aggregate copy\n", encoding="utf-8")
    mismatched = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=aggregate_roots,
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / "mismatched-tree-wp1-acceptance.json",
        repo_root=ROOT,
    )
    assert mismatched["status"] == "failed"
    assert mismatched["failures"] == [
        f"transport source_fill case tree differs: {case_id}"
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("endpoint_hash", "accepted-end provenance"),
        ("endpoint_provenance", "accepted-end provenance"),
        ("seam_control", "encoded seam evidence"),
        ("seam_residual", "encoded seam evidence"),
        ("seam_continuation", "encoded seam evidence"),
    ],
)
def test_v4_endpoint_and_seam_evidence_drift_fails_closed(
    accepted_fixture: dict[str, object],
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    transport_path = accepted_fixture["transport_paths"][0]
    transport = json.loads(transport_path.read_text())
    if mutation.startswith("endpoint_"):
        segment_receipt_path = Path(transport["transport_segments"][0]["receipt_path"])
        segment_receipt = json.loads(segment_receipt_path.read_text())
        build_path = Path(segment_receipt["segment_dir"]) / "transport-build-receipt.json"
        build = json.loads(build_path.read_text())
        accepted_end = build["adjacent_endpoints"]["accepted_end"]
        if mutation == "endpoint_hash":
            accepted_end["sha256"] = "f" * 64
        else:
            accepted_end["provenance"]["receipt_field"] = "pre_pour_parity.reference_path"
        _write_json(build_path, build)
        segment_receipt["build_receipt_sha256"] = file_sha256(build_path)
        _write_json(segment_receipt_path, segment_receipt)
        transport["transport_segments"][0]["receipt_sha256"] = file_sha256(
            segment_receipt_path
        )
    else:
        seam = transport["encoded_seam_evidence"][1]
        if mutation == "seam_control":
            seam["start_seam_control_mad"] += 0.25
        elif mutation == "seam_residual":
            seam["end_seam_residual_mad"] += 0.25
        else:
            seam["continuation_seam_mad"] += 0.25
    _write_json(transport_path, transport)
    _refresh_transport_provenance(accepted_fixture)

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / f"{mutation}-wp1-acceptance.json",
        repo_root=ROOT,
    )

    assert result["status"] == "failed"
    assert any(expected in failure for failure in result["failures"]), result["failures"]


def test_v4_decoded_transition_hash_substitution_fails_closed(
    accepted_fixture: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validate_segment = wp1_postrender._validate_segment

    def validate_decoder_tolerant_segment(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        result = validate_segment(*args, **kwargs)
        return {**result, "decoder_tolerant_evidence": True}

    monkeypatch.setattr(
        wp1_postrender, "_validate_segment", validate_decoder_tolerant_segment
    )
    transport_path = accepted_fixture["transport_paths"][0]
    accepted_receipt_sha256 = file_sha256(transport_path)
    transport = json.loads(transport_path.read_text())
    transport["encoded_transition_evidence"][0]["decoded_bgr_sha256"][0] = "f" * 64
    _write_json(transport_path, transport)
    assert file_sha256(transport_path) != accepted_receipt_sha256

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / "substituted-transition-hash-wp1-acceptance.json",
        repo_root=ROOT,
    )

    assert result["status"] == "failed"
    assert any(
        "immutable finalization provenance" in failure
        for failure in result["failures"]
    ), result["failures"]


def test_exact_real_v4_receipts_pass_with_available_decoder() -> None:
    inventory_path = Path(
        "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
        "wp1-finalization-01a18128-attempt01/finalization-inventory.json"
    )
    if not inventory_path.is_file():
        pytest.skip("exact WP1 finalization evidence is not available on this host")
    inventory = json.loads(inventory_path.read_text())
    manifest_root = Path(inventory["manifest"]["root"])
    collection = json.loads(
        (manifest_root / "full-transition-manifest.json").read_text()
    )
    receipt_paths = [
        Path(row["receipt_path"]) for row in inventory["transport_products"]
    ]
    accepted = wp1_postrender._validate_finalization_inventory(
        inventory_path,
        manifest_root=manifest_root,
        bindings=collection["run_bindings"],
        transport_receipts=receipt_paths,
    )

    for binding, aggregate, product in zip(
        collection["run_bindings"],
        inventory["aggregate_runs"],
        inventory["transport_products"],
        strict=True,
    ):
        run_root = Path(aggregate["root"])
        fills = {}
        for case_id in binding["case_ids"]:
            receipt_path = run_root / "cases" / case_id / "app-output" / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            app = receipt_path.parent
            render = receipt["render"]
            final_pair = max(
                receipt["semantics"]["pairs"], key=lambda row: row["physics_step"]
            )
            pre_pour = receipt["pre_pour_parity"]
            video = (app / render["video_path"]).resolve()
            fills[case_id] = {
                "path": str(video),
                "sha256": file_sha256(video),
                "final_rgb_path": str((app / final_pair["rgb_path"]).resolve()),
                "final_rgb_sha256": final_pair["rgb_sha256"],
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": file_sha256(receipt_path),
                "pre_pour_path": str((app / pre_pour["f2_prefill_path"]).resolve()),
                "pre_pour_sha256": pre_pour["f2_prefill_sha256"],
                "parity_reference_path": str(
                    (app / pre_pour["reference_path"]).resolve()
                ),
                "parity_reference_sha256": pre_pour["reference_sha256"],
                "video_timing_source_frame_steps": render[
                    "video_timing_source_frame_steps"
                ],
                "decoded_video_frame_count": render["video_decoded_frame_count"],
                "video_codec": render["video_codec"],
                "continuation_renderer": render["video_renderer"],
                "pre_pour_renderer": render["stage_still_renderer"],
            }
        readiness_path = run_root / "wp1-fill-readiness.json"
        readiness = json.loads(readiness_path.read_text())
        readiness.update(
            {
                "receipt_path": str(readiness_path.resolve()),
                "receipt_sha256": file_sha256(readiness_path),
            }
        )
        result = wp1_postrender._validate_transport(
            Path(product["receipt_path"]),
            binding=binding,
            source_fills=fills,
            readiness=readiness,
            accepted_receipt_sha256=accepted[binding["run_id"]],
        )
        assert result["receipt_sha256"] == product["receipt_sha256"]
        assert result["run_product"]["sha256"] == product["video_sha256"]


@pytest.mark.parametrize(("row_index", "segment_index"), [(1, True), (0, 0.5)])
def test_v4_seam_segment_index_requires_exact_integer(
    accepted_fixture: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_index: int,
    segment_index: object,
) -> None:
    validate_segment = wp1_postrender._validate_segment

    def validate_decoder_tolerant_segment(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        result = validate_segment(*args, **kwargs)
        return {**result, "decoder_tolerant_evidence": True}

    monkeypatch.setattr(
        wp1_postrender, "_validate_segment", validate_decoder_tolerant_segment
    )
    transport_path = accepted_fixture["transport_paths"][0]
    transport = json.loads(transport_path.read_text())
    transport["encoded_seam_evidence"][row_index]["segment_index"] = segment_index
    _write_json(transport_path, transport)
    _refresh_transport_provenance(accepted_fixture)

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / "invalid-seam-index-wp1-acceptance.json",
        repo_root=ROOT,
    )

    assert result["status"] == "failed"
    assert any(
        "encoded seam evidence differs" in failure for failure in result["failures"]
    ), result["failures"]


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
        ("legacy_transport_v2", "transport composer identity"),
        ("altered_packaging_semantics", "packaging frame-count semantics"),
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
    elif mutation == "legacy_transport_v2":
        receipt = json.loads(transports[0].read_text())
        receipt["schema_version"] = 2
        receipt["artifact_type"] = "rendered_transport_run_sequence_v2"
        _write_json(transports[0], receipt)
    elif mutation == "altered_packaging_semantics":
        receipt = json.loads(transports[0].read_text())
        receipt["packaging_frame_count_semantics"][
            "accepted_fill_frames_preserved"
        ] -= 1
        _write_json(transports[0], receipt)
    if mutation in {
        "mismatched_video",
        "failed_transport",
        "legacy_transport_v2",
        "altered_packaging_semantics",
    }:
        _refresh_transport_provenance(accepted_fixture)

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=roots,
        transport_receipts=transports,
        finalization_inventory=accepted_fixture["inventory"],
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / f"{mutation}-wp1-acceptance.json",
        repo_root=ROOT,
    )
    assert result["status"] == "failed"
    assert any(expected in failure for failure in result["failures"])
