"""Materialize and run the frozen Brev validation-video cohort."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone

from twin.render.render_process import (
    atomic_write_json,
    hash_regular_file,
    load_strict_json_snapshot,
)
from twin.scene.twin_stage_appearance import (
    create_stage_appearance_layer,
    stage_appearance_command_plan,
)


SOURCE_RENDER_PLAN_SHA256 = (
    "98e73854c4817bd3e2ef2dfa0c21bd6c63a6f2d0ac83021365a0ccd665d5296b"
)
VIDEO_TIME_CODES = tuple(sorted(set(range(31, 687, 5)) | {342}))
STAGE_TIME_CODES = (171, 342, 686)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_BATCHES = (
    ("BATCH_72ADF5DC9C45", "reference_only", "none", "none", (0.0, 0.0, 0.0), (1310001, 1310002, 1310003)),
    ("BATCH_A51288D2AE92", "reference_only", "first_only", "after_t400", (0.3, 0.0, 0.0), (1310001, 1310002, 1310003)),
    ("BATCH_907E15C8E9B1", "reference_only", "first_two", "after_t400", (0.3, 0.12, 0.0), (1310001, 1310002, 1310003)),
    ("BATCH_D7A0146E9E09", "reference_only", "all_three", "after_t400", (0.3, 0.18, 0.08), (1310001, 1310002, 1310003)),
    ("BATCH_F77F6F81E31A", "scored_target", "none", "none", (0.0, 0.0, 0.0), (1310101, 1310102, 1310103)),
    ("BATCH_D5944B023021", "scored_target", "first_only", "after_t400", (0.3, 0.0, 0.0), (1310101, 1310102, 1310103)),
    ("BATCH_A583FC4BC148", "scored_target", "first_two", "after_t400", (0.3, 0.12, 0.0), (1310101, 1310102, 1310103)),
    ("BATCH_E4033277D748", "scored_target", "all_three", "after_t400", (0.3, 0.18, 0.08), (1310101, 1310102, 1310103)),
)
_SOURCE_FILES = (
    "cli/twin/render/build_brev_video_cohort.py",
    "cli/twin/render/encode_twin_capture_video.py",
    "cli/twin/render/run_brev_video_cohort.py",
    "cli/twin/render/twin_render_sequence.py",
    "cli/twin/render/render_twin_dataset.py",
    "src/twin/render/brev_video_cohort.py",
    "src/twin/render/render_capture.py",
    "src/twin/render/render_process.py",
    "src/twin/scene/twin_stage_appearance.py",
    "src/twin/render/twin_video.py",
)


def _runtime(isaac_python: str) -> dict[str, object]:
    return {
        "provider": "Brev",
        "instance_name": "claude-isaac-smoke-1024",
        "instance_id": "m1vgz4tbs",
        "machine_type": "g6e.xlarge",
        "gpu": "L40S",
        "machine": "x86_64",
        "runtime_name": "NVIDIA Isaac Sim",
        "isaac_sim_version": "6.0.1.0",
        "environment_profile": "generic",
        "isaac_python": isaac_python,
    }


def _validate_frozen_plan(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("render plan must be an object")
    if payload.get("schema_version") != 3 or payload.get("artifact_type") != "render_plan":
        raise ValueError("render plan schema or artifact type drifted")
    families = payload.get("families")
    batches = payload.get("batches")
    launches = payload.get("render_launches")
    if not all(isinstance(value, list) for value in (families, batches, launches)):
        raise ValueError("render plan family, batch, or launch records are malformed")
    selected_families = [row for row in families if isinstance(row, Mapping) and row.get("family_id") == "VA"]
    if len(selected_families) != 1:
        raise ValueError("render plan must contain exactly one VA family")
    family = selected_families[0]
    expected_family = {
        "family_id": "VA",
        "partition": "val",
        "previous_sim_recipe_id": "S0160",
        "current_sim_recipe_id": "S0380",
        "onset_id": "after_t400",
    }
    if any(family.get(key) != value for key, value in expected_family.items()):
        raise ValueError("VA family identity drifted")
    selected_batches = [row for row in batches if isinstance(row, Mapping) and row.get("family_id") == "VA"]
    if len(selected_batches) != 8:
        raise ValueError("VA must contain exactly eight batches")
    for row, expected in zip(selected_batches, _EXPECTED_BATCHES, strict=True):
        batch_id, cohort, profile, onset, alphas, seeds = expected
        actual = (
            row.get("batch_id"), row.get("cohort"), row.get("synthetic_profile_id"),
            row.get("synthetic_onset_id"), tuple(row.get("appearance_blend_alpha_by_vial", ())),
            tuple(row.get("render_seed_by_vial", ())),
        )
        if actual != (batch_id, cohort, profile, onset, alphas, seeds) or row.get("vial_count") != 3:
            raise ValueError(f"frozen VA batch drifted: {batch_id}")
    selected_launches = [dict(row) for row in launches if isinstance(row, Mapping) and row.get("family_id") == "VA"]
    if len(selected_launches) != 24:
        raise ValueError("VA must contain exactly 24 render launches")
    expected_launch_ids = [f"{batch[0]}-V{position}" for batch in _EXPECTED_BATCHES for position in (1, 2, 3)]
    if [row.get("launch_id") for row in selected_launches] != expected_launch_ids:
        raise ValueError("VA render launch order or identity drifted")
    batch_lookup = {row[0]: row for row in _EXPECTED_BATCHES}
    for row in selected_launches:
        batch = batch_lookup[str(row["batch_id"])]
        position = int(str(row["vial_id"])[1:])
        if row.get("render_seed") != batch[5][position - 1]:
            raise ValueError("VA launch seed drifted")
        samples = row.get("stage_samples")
        if not isinstance(samples, list) or len(samples) != 3:
            raise ValueError("VA launch stage samples are malformed")
        if [sample.get("time_code") for sample in samples if isinstance(sample, Mapping)] != list(STAGE_TIME_CODES):
            raise ValueError("VA launch stage time codes drifted")
        for sample in samples:
            if not isinstance(sample, Mapping) or not isinstance(sample.get("observed_stage_linear_rgb"), list):
                raise ValueError("VA launch observed stage appearance is malformed")
    return selected_launches


def materialize_brev_video_cohort(
    *,
    render_plan_path: str | os.PathLike[str],
    twin_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    runtime_lock: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    implementation_commit: str,
    isaac_python: str,
) -> dict[str, object]:
    """Materialize the exact VA cohort without rendering or running Brev."""

    plan, plan_snapshot = load_strict_json_snapshot(render_plan_path)
    if plan_snapshot["sha256"] != SOURCE_RENDER_PLAN_SHA256:
        raise ValueError("render plan SHA-256 does not match the frozen source plan")
    launches = _validate_frozen_plan(plan)
    if not isinstance(implementation_commit, str) or _COMMIT.fullmatch(implementation_commit) is None:
        raise ValueError("implementation commit must be a lowercase 40-hex Git commit")
    if not isinstance(isaac_python, str) or not isaac_python or not Path(isaac_python).is_absolute():
        raise ValueError("Isaac Python must be a nonempty absolute path")
    repository = Path(repo_root).resolve(strict=True)
    if not repository.is_dir():
        raise ValueError("repository root must be a directory")
    runtime_lock_snapshot = hash_regular_file(runtime_lock)
    runtime_lock_path = str(Path(runtime_lock).resolve(strict=True))
    source_hashes = {
        name: hash_regular_file(repository / name)["sha256"] for name in _SOURCE_FILES
    }
    twin = Path(twin_root).resolve(strict=True)
    if not twin.is_dir():
        raise ValueError("twin root must be a directory")
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.exists() or output_candidate.is_symlink():
        raise FileExistsError(f"cohort output already exists: {output_candidate}")
    parent = output_candidate.parent.resolve(strict=True)
    output = parent / output_candidate.name
    os.mkdir(output, 0o700)
    os.chmod(output, 0o700)
    manifests = output / "experiment-manifests"
    os.mkdir(manifests, 0o700)
    os.chmod(manifests, 0o700)

    runtime = _runtime(isaac_python)
    frame_spec = ",".join(str(code) for code in VIDEO_TIME_CODES)
    render_controls = {
        "camera": "FillCam",
        "view_id": "vial_fill",
        "renderer": "PathTracing",
        "resolution": [640, 480],
        "time_codes": list(VIDEO_TIME_CODES),
        "frame_spec": frame_spec,
        "stage_time_codes": list(STAGE_TIME_CODES),
        "subframes": 64,
        "warmup": 60,
        "max_tries": 20,
        "timeout_seconds": 1800,
        "require_liquid_pixels": True,
    }
    video_encoding = {
        "codec_request": "mp4v",
        "container": "mp4",
        "fps": 6.0,
        "expected_decoded_frames": 133,
        "timing": "constant_fps_with_inserted_time_code_342_not_source_timing_exact",
        "use": "lossy_review_visualization_only",
        "measurement_source": "hash_bound_capture_png_and_liquid_mask_only",
    }
    rows: list[dict[str, object]] = []
    for ordinal, source in enumerate(launches, start=1):
        launch_id = str(source["launch_id"])
        samples = [
            {
                "time_code": int(sample["time_code"]),
                "diffuse_color_linear": list(sample["observed_stage_linear_rgb"]),
                "opacity": 0.6,
            }
            for sample in source["stage_samples"]
        ]
        layer = create_stage_appearance_layer(twin, output, launch_id, samples)
        appearance_plan = stage_appearance_command_plan(layer)
        manifest = {
            "schema_version": 1,
            "run_id": launch_id,
            "hypothesis": (
                "One Brev Isaac Sim 6.0.1 process renders all 133 frozen full-fill "
                "time codes for this synthetic validation vial, including the three "
                "authored post-dispense stage frames, with every mechanical RGB and "
                "liquid-semantic gate passing."
            ),
            "falsifier": (
                "The sole process attempt returns nonzero or times out; any requested "
                "frame is absent, constant, lacks liquid pixels, or fails semantic-instance "
                "agreement; provenance changes; or the review video cannot be encoded and verified."
            ),
            "command_plan": {
                "entry_point": "cli/twin/render/render_twin_dataset.py",
                "implementation_commit": implementation_commit,
                "source_render_plan_sha256": SOURCE_RENDER_PLAN_SHA256,
                "launch_id": launch_id,
                "family_id": "VA",
                "partition": "val",
                "cohort": source["cohort"],
                "synthetic_profile_id": next(
                    batch[2] for batch in _EXPECTED_BATCHES if batch[0] == source["batch_id"]
                ),
                "synthetic_onset_id": next(
                    batch[3] for batch in _EXPECTED_BATCHES if batch[0] == source["batch_id"]
                ),
                "runtime_deviation": (
                    "whole VA cohort moved from the source plan's local Isaac 6.2 profile "
                    "to Brev Isaac 6.0.1; no RGB/Lab pooling across runtime versions"
                ),
                "frames": frame_spec,
                "render_controls": render_controls,
                "appearance_layer": appearance_plan,
                "runtime_lock_sha256": runtime_lock_snapshot["sha256"],
                "source_sha256": source_hashes,
                "video_encoding": video_encoding,
            },
            "render_runtime": runtime,
            "process_attempt_policy": {
                "max_process_attempts": 1,
                "replacement_attempts": 0,
                "mode": "generic_single_attempt_stop_on_failure",
            },
            "seed": source["render_seed"],
            "denominators": {
                "maximum_process_attempts": 1,
                "requested_frames": 133,
                "required_stage_frames": 3,
                "shared_capture_tries_per_frame": 20,
            },
            "stop_conditions": [
                "one process attempt only and no replacement",
                "1800-second total process bound",
                "stop cohort on first nonzero or missing mechanical gate",
                "all 133 requested frames required",
                "stage frames 171, 342, and 686 must come from the same capture",
                "stop on provenance, encoder, or decoded-video verification failure",
            ],
            "claim_boundary": (
                "Synthetic uncalibrated material interpolation and review video only; no "
                "physical mixing, recipe colour, carryover extent or cause, defect, real "
                "baseline, plant action, or sim-to-real claim."
            ),
        }
        manifest_path = atomic_write_json(manifests, f"{launch_id}.json", manifest)
        manifest_snapshot = hash_regular_file(manifest_path)
        rows.append(
            {
                "ordinal": ordinal,
                "launch_id": launch_id,
                "family_id": "VA",
                "partition": "val",
                "batch_id": source["batch_id"],
                "cohort": source["cohort"],
                "vial_id": source["vial_id"],
                "seed": source["render_seed"],
                "max_process_attempts": 1,
                "replacement_attempts": 0,
                "appearance_layer": layer["path"],
                "appearance_layer_sha256": layer["stage_sha256"],
                "experiment_manifest": str(manifest_path),
                "experiment_manifest_sha256": manifest_snapshot["sha256"],
                "capture_root": str(output / launch_id / "capture"),
                "video_output_dir": str(output / "videos" / launch_id),
            }
        )

    graph = {
        "schema_version": 1,
        "artifact_type": "brev_validation_video_cohort",
        "status": "materialized_unrendered",
        "source_render_plan": str(Path(render_plan_path).resolve(strict=True)),
        "source_render_plan_sha256": SOURCE_RENDER_PLAN_SHA256,
        "implementation_commit": implementation_commit,
        "repository_root": str(repository),
        "runtime_lock": {"path": runtime_lock_path, **runtime_lock_snapshot},
        "twin_root": str(twin),
        "source_sha256": source_hashes,
        "family": {
            "family_id": "VA",
            "partition": "val",
            "previous_sim_recipe_id": "S0160",
            "current_sim_recipe_id": "S0380",
            "synthetic_onset_id": "after_t400",
        },
        "render_runtime": runtime,
        "runtime_mixing_policy": (
            "whole cohort on one Brev host and Isaac 6.0.1; never pool RGB/Lab "
            "with Isaac 6.2 cohorts"
        ),
        "render_controls": render_controls,
        "video_encoding": video_encoding,
        "process_policy": {
            "max_process_attempts_per_launch": 1,
            "replacement_attempts": 0,
            "cohort_stop": "first_failure",
        },
        "denominators": {
            "batches": 8,
            "vials_and_render_launches": 24,
            "frames_per_launch": 133,
            "total_frames": 3192,
            "stage_frames_per_launch": 3,
            "stage_frames_total": 72,
            "videos": 24,
            "maximum_process_attempts_per_launch": 1,
        },
        "launches": rows,
        "claim_boundary": (
            "Synthetic uncalibrated direct-material video cohort. Appearance between "
            "the three authored samples is USD interpolation and is not physical mixing evidence."
        ),
        "oracle_separation": (
            "Profile, onset, and blend provenance remain render/label-only and must not "
            "be copied into model-input evidence."
        ),
    }
    atomic_write_json(output, "cohort.json", graph)
    return graph


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_logged(command: list[str], log_path: Path) -> dict[str, object]:
    started = _utc_now()
    before = time.monotonic()
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    completed = None
    spawn_error = None
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as log:
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as error:
                spawn_error = f"{type(error).__name__}: {error}"
            log.flush()
            os.fsync(log.fileno())
    finally:
        os.close(descriptor)
    return {
        "command": command,
        "started_utc": started,
        "ended_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - before,
        "returncode": completed.returncode if completed is not None else None,
        "spawn_error": spawn_error,
        "log": {"path": str(log_path), **hash_regular_file(log_path)},
    }


def _validate_materialized_graph(graph: object, root: Path) -> dict[str, object]:
    if not isinstance(graph, Mapping):
        raise ValueError("cohort graph must be an object")
    if graph.get("schema_version") != 1 or graph.get("artifact_type") != "brev_validation_video_cohort":
        raise ValueError("cohort graph schema or artifact type is invalid")
    if graph.get("status") != "materialized_unrendered":
        raise ValueError("cohort graph is not an unrendered materialization")
    if graph.get("source_render_plan_sha256") != SOURCE_RENDER_PLAN_SHA256:
        raise ValueError("cohort graph source plan binding drifted")
    rows = graph.get("launches")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("cohort graph must contain exactly 24 launches")
    expected_ids = [f"{batch[0]}-V{position}" for batch in _EXPECTED_BATCHES for position in (1, 2, 3)]
    if [row.get("launch_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
        raise ValueError("cohort graph launch order drifted")
    if [row.get("ordinal") for row in rows if isinstance(row, Mapping)] != list(range(1, 25)):
        raise ValueError("cohort graph launch ordinals drifted")
    if graph.get("render_controls", {}).get("time_codes") != list(VIDEO_TIME_CODES):
        raise ValueError("cohort graph video time-code schedule drifted")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cohort launch row is malformed")
        for key in ("appearance_layer", "experiment_manifest", "capture_root", "video_output_dir"):
            value = row.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"cohort launch {key} must be an absolute path")
        if Path(str(row["appearance_layer"])).parent != root / "appearance-layers":
            raise ValueError("cohort appearance layer escaped its output root")
        if Path(str(row["experiment_manifest"])).parent != root / "experiment-manifests":
            raise ValueError("cohort experiment manifest escaped its output root")
        if Path(str(row["capture_root"])) != root / str(row["launch_id"]) / "capture":
            raise ValueError("cohort capture root drifted")
        if Path(str(row["video_output_dir"])) != root / "videos" / str(row["launch_id"]):
            raise ValueError("cohort video output root drifted")
        if row.get("max_process_attempts") != 1 or row.get("replacement_attempts") != 0:
            raise ValueError("cohort process-attempt policy drifted")
        if hash_regular_file(row["appearance_layer"])["sha256"] != row.get("appearance_layer_sha256"):
            raise ValueError("cohort appearance layer hash drifted")
        if hash_regular_file(row["experiment_manifest"])["sha256"] != row.get("experiment_manifest_sha256"):
            raise ValueError("cohort experiment manifest hash drifted")
    return dict(graph)


def run_brev_video_cohort(
    cohort_path: str | os.PathLike[str],
    *,
    controller_python: str | os.PathLike[str],
    launcher_script: str | os.PathLike[str],
    encoder_script: str | os.PathLike[str],
) -> dict[str, object]:
    """Run each frozen row once, encode it, and stop on the first failure."""

    cohort_candidate = Path(cohort_path)
    graph_raw, graph_snapshot = load_strict_json_snapshot(cohort_candidate)
    root = cohort_candidate.resolve(strict=True).parent
    graph = _validate_materialized_graph(graph_raw, root)
    if (root / "cohort-run.json").exists():
        raise FileExistsError("cohort already has a run record")
    python = Path(os.path.abspath(Path(controller_python).expanduser()))
    python_target = python.resolve(strict=True)
    launcher = Path(launcher_script).resolve(strict=True)
    encoder = Path(encoder_script).resolve(strict=True)
    for role, path in (("controller Python", python), ("launcher", launcher), ("encoder", encoder)):
        if not path.is_file() or (role == "controller Python" and not os.access(path, os.X_OK)):
            raise ValueError(f"{role} is not an executable regular file: {path}")
    if not python_target.is_file() or not os.access(python_target, os.X_OK):
        raise ValueError(f"controller Python target is not executable: {python_target}")
    repository = Path(str(graph.get("repository_root"))).resolve(strict=True)
    source_hashes = graph.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(_SOURCE_FILES):
        raise ValueError("cohort source hash binding is malformed")
    for name in _SOURCE_FILES:
        if hash_regular_file(repository / name)["sha256"] != source_hashes[name]:
            raise ValueError(f"cohort source changed before execution: {name}")
    runtime_lock = graph.get("runtime_lock")
    if not isinstance(runtime_lock, Mapping):
        raise ValueError("cohort runtime lock binding is malformed")
    if hash_regular_file(runtime_lock["path"]) != {
        "sha256": runtime_lock.get("sha256"),
        "bytes": runtime_lock.get("bytes"),
    }:
        raise ValueError("cohort runtime lock changed before execution")
    videos = root / "videos"
    logs = root / "orchestration-logs"
    for directory in (videos, logs):
        os.mkdir(directory, 0o700)
        os.chmod(directory, 0o700)

    controls = graph["render_controls"]
    runtime = graph["render_runtime"]
    rows_out: list[dict[str, object]] = []
    completed_count = 0
    failed_launch: str | None = None
    started = _utc_now()
    for row in graph["launches"]:
        launch_id = str(row["launch_id"])
        render_command = [
            str(python), str(launcher),
            "--isaac-python", str(runtime["isaac_python"]),
            "--runtime-lock", str(graph["runtime_lock"]["path"]),
            "--experiment-manifest", str(row["experiment_manifest"]),
            "--twin", str(graph["twin_root"]),
            "--run-id", launch_id,
            "--out-parent", str(root),
            "--stage-appearance-stage", str(row["appearance_layer"]),
            "--frames", str(controls["frame_spec"]),
            "--subframes", str(controls["subframes"]),
            "--warmup", str(controls["warmup"]),
            "--max-tries", str(controls["max_tries"]),
            "--seed", str(row["seed"]),
            "--renderer", str(controls["renderer"]),
            "--timeout", str(controls["timeout_seconds"]),
            "--environment-profile", str(runtime["environment_profile"]),
            "--require-liquid-pixels",
        ]
        render_result = _run_logged(
            render_command, logs / f"{int(row['ordinal']):02d}-{launch_id}-render.log"
        )
        row_result: dict[str, object] = {
            "ordinal": row["ordinal"],
            "launch_id": launch_id,
            "render": render_result,
            "encoder": None,
            "outcome": None,
            "gate_error": None,
        }
        if render_result["returncode"] != 0:
            row_result["outcome"] = "render_failed"
            rows_out.append(row_result)
            failed_launch = launch_id
            break
        try:
            attempt, _ = load_strict_json_snapshot(root / launch_id / "attempt.json")
            if not isinstance(attempt, Mapping):
                raise ValueError("attempt record is not an object")
            if attempt.get("status") != "mechanical_gates_passed":
                raise ValueError("render attempt did not pass mechanical gates")
            capture_record = attempt.get("capture")
            if not isinstance(capture_record, Mapping) or capture_record.get("verified") is not True:
                raise ValueError("render attempt capture is not verified")
            if Path(str(capture_record.get("root"))).resolve(strict=True) != Path(str(row["capture_root"])).resolve(strict=True):
                raise ValueError("render attempt capture root does not match cohort row")
        except (OSError, TypeError, ValueError) as error:
            row_result["outcome"] = "mechanical_gate_failed"
            row_result["gate_error"] = f"{type(error).__name__}: {error}"
            rows_out.append(row_result)
            failed_launch = launch_id
            break

        encode_command = [
            str(python), str(encoder),
            "--capture-root", str(row["capture_root"]),
            "--output-dir", str(row["video_output_dir"]),
        ]
        encode_result = _run_logged(
            encode_command, logs / f"{int(row['ordinal']):02d}-{launch_id}-encode.log"
        )
        row_result["encoder"] = encode_result
        if encode_result["returncode"] != 0:
            row_result["outcome"] = "encoder_failed"
            rows_out.append(row_result)
            failed_launch = launch_id
            break
        try:
            video_record, _ = load_strict_json_snapshot(Path(str(row["video_output_dir"])) / "video.json")
            if not isinstance(video_record, Mapping) or video_record.get("status") != "encoded_verified":
                raise ValueError("video record is not encoded_verified")
            capture_record = video_record.get("capture")
            if not isinstance(capture_record, Mapping) or Path(str(capture_record.get("root"))).resolve(strict=True) != Path(str(row["capture_root"])).resolve(strict=True):
                raise ValueError("video record capture root does not match cohort row")
            video = video_record.get("video")
            if not isinstance(video, Mapping) or video.get("path") != "sequence.mp4":
                raise ValueError("video record artifact is malformed")
            snapshot = hash_regular_file(Path(str(row["video_output_dir"])) / "sequence.mp4")
            if video.get("sha256") != snapshot["sha256"] or video.get("bytes") != snapshot["bytes"]:
                raise ValueError("video artifact hash or byte count does not match record")
        except (OSError, TypeError, ValueError) as error:
            row_result["outcome"] = "video_gate_failed"
            row_result["gate_error"] = f"{type(error).__name__}: {error}"
            rows_out.append(row_result)
            failed_launch = launch_id
            break
        row_result["outcome"] = "passed"
        rows_out.append(row_result)
        completed_count += 1

    graph_after, graph_after_snapshot = load_strict_json_snapshot(cohort_candidate)
    if graph_after != graph or graph_after_snapshot != graph_snapshot:
        raise RuntimeError("cohort graph changed during execution")
    for name in _SOURCE_FILES:
        if hash_regular_file(repository / name)["sha256"] != source_hashes[name]:
            raise RuntimeError(f"cohort source changed during execution: {name}")
    result = {
        "schema_version": 1,
        "status": "completed" if completed_count == 24 else "failed",
        "started_utc": started,
        "ended_utc": _utc_now(),
        "cohort": {"path": str(cohort_candidate.resolve(strict=True)), **graph_snapshot},
        "controller_python": str(python),
        "launcher": {"path": str(launcher), **hash_regular_file(launcher)},
        "encoder": {"path": str(encoder), **hash_regular_file(encoder)},
        "expected_launches": 24,
        "attempted_launches": len(rows_out),
        "completed_launches": completed_count,
        "failed_launch_id": failed_launch,
        "replacement_attempts": 0,
        "stop_policy": "first_failure",
        "rows": rows_out,
    }
    atomic_write_json(root, "cohort-run.json", result)
    return result
