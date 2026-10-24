"""Synthetic 640x480 pilot trees for CPU-only contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from fluidmix.pilot_dataset import (
    CONFIG_PATH,
    build_experiment_graph,
    build_pilot_dataset,
    canonical_pilot_config,
)


COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_graph_inputs(pilot_root: Path) -> None:
    config = canonical_pilot_config()
    source = pilot_root / "source-twin/stations/fill_station/fill_station.usda"
    source.parent.mkdir(parents=True)
    source.write_text("#usda 1.0\n# fixture canonical source\n")
    runtime = pilot_root / "runtime"
    runtime.mkdir()
    controller = runtime / "controller-python"
    isaac = runtime / "isaac-python"
    lock = runtime / "runtime-lock.json"
    controller.write_text("#!/bin/sh\n")
    isaac.write_text("#!/bin/sh\n")
    controller.chmod(0o755)
    isaac.chmod(0o755)
    lock.write_text("{}\n")
    (pilot_root / "appearance-layers").mkdir(parents=True)
    (pilot_root / "attempt-manifests").mkdir()
    for row in config["rows"]:
        sample_id = row["sample_id"]
        layer = pilot_root / "appearance-layers" / f"{sample_id}.usda"
        layer.write_text(f"#usda 1.0\n# fixture {sample_id}\n")
        appearance = config["appearances"][row["appearance_id"]]
        manifest = {
            "schema_version": 1,
            "run_id": sample_id,
            "hypothesis": "The fixed renderer emits one mechanically valid pilot capture.",
            "falsifier": "Any lifecycle, provenance, or capture gate fails.",
            "command_plan": {
                "entry_point": "scripts/render_twin_dataset.py",
                "implementation_commit": COMMIT,
                "command": [
                    str(controller.resolve()),
                    "scripts/render_twin_dataset.py",
                    "--isaac-python",
                    str(isaac.resolve()),
                    "--runtime-lock",
                    str(lock.resolve()),
                    "--experiment-manifest",
                    str((pilot_root / "attempt-manifests" / f"{sample_id}.json").resolve()),
                    "--twin",
                    str((pilot_root / "source-twin").resolve()),
                    "--run-id",
                    sample_id,
                    "--out-parent",
                    str(pilot_root.resolve()),
                    "--appearance-stage",
                    str(layer.resolve()),
                    "--frames",
                    "686:687:1",
                    "--subframes",
                    "64",
                    "--warmup",
                    "60",
                    "--max-tries",
                    "20",
                    "--seed",
                    "310241",
                    "--require-liquid-pixels",
                    "--renderer",
                    "PathTracing",
                    "--timeout",
                    "1800",
                    "--environment-profile",
                    "gb10-aarch64",
                ],
                "appearance_layer": {
                    "schema": "direct_appearance_v1",
                    "variant_id": sample_id,
                    "stage_sha256": sha256(layer),
                    "stage_bytes": layer.stat().st_size,
                    "source_stage": "stations/fill_station/fill_station.usda",
                    "source_stage_sha256": sha256(source),
                    "material_path": "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF",
                    "diffuse_color_linear": appearance["linear_diffuse_rgb"],
                    "opacity": appearance["opacity"],
                },
                "requested_controls": config["controls"],
                "process_attempt_limit": 1,
                "replacement_process_attempts": 0,
            },
            "seed": config["controls"]["seed"],
            "denominators": {
                "pilot_rows": 8,
                "requested_frames": 1,
                "process_attempts": 1,
            },
            "stop_conditions": [
                "no replacement process attempt",
                "bounded timeout",
                "provenance or safety failure",
            ],
        }
        (pilot_root / "attempt-manifests" / f"{sample_id}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


def _artifact_record(path: Path, capture: Path) -> dict[str, str]:
    return {"path": path.relative_to(capture).as_posix(), "sha256": sha256(path)}


def _write_capture(pilot_root: Path, row: dict[str, object], color: tuple[int, int, int]) -> None:
    sample_id = str(row["sample_id"])
    attempt_root = pilot_root / sample_id
    capture = attempt_root / "capture"
    for kind in ("rgb", "semantic", "instance", "liquid"):
        (capture / kind).mkdir(parents=True, exist_ok=True)

    height, width = 480, 640
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint16)[None, :] % 37
    rgb[..., 1] = 18
    rgb[..., 2] = 24
    rgb[200:400, 250:390] = color
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[160:430, 230:410] = 1
    semantic[200:400, 250:390] = 2
    instance = np.zeros((height, width), dtype=np.uint8)
    instance[160:430, 230:410] = 3
    instance[200:400, 250:390] = 4
    mask = (semantic == 2).astype(np.uint8) * 255

    rgb_path = capture / "rgb/0686.png"
    semantic_path = capture / "semantic/0686.npy"
    instance_path = capture / "instance/0686.npy"
    mask_path = capture / "liquid/0686.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(semantic_path, semantic, allow_pickle=False)
    np.save(instance_path, instance, allow_pickle=False)
    Image.fromarray(mask, mode="L").save(mask_path)
    artifacts = {
        "rgb": _artifact_record(rgb_path, capture),
        "semantic": _artifact_record(semantic_path, capture),
        "instance": _artifact_record(instance_path, capture),
        "liquid": _artifact_record(mask_path, capture),
    }
    layer = pilot_root / "appearance-layers" / f"{sample_id}.usda"
    source = pilot_root / "source-twin/stations/fill_station/fill_station.usda"
    settings = {
        "camera": "FillCam",
        "stage": str(layer.resolve()),
        "frames": "686:687:1",
        "subframes": 64,
        "warmup": 60,
        "max_tries": 20,
        "seed": 310241,
        "require_liquid_pixels": True,
        "renderer": "PathTracing",
    }
    liquid_count = int(np.count_nonzero(mask))
    vial_count = int(np.count_nonzero(semantic == 1))
    summary = {
        "schema_version": 1,
        "mechanical_gates_passed": True,
        "visual_check_status": "pending",
        "inputs": {
            "stage": {
                "requested": settings["stage"],
                "resolved_path": settings["stage"],
                "sha256": sha256(layer),
            },
            "camera": {"name": "FillCam"},
            "non_session_usd_layer_stack": [
                {
                    "ordinal": 1,
                    "identifier": settings["stage"],
                    "resolved_path": settings["stage"],
                    "sha256": sha256(layer),
                    "bytes": layer.stat().st_size,
                },
                {
                    "ordinal": 2,
                    "identifier": str(source.resolve()),
                    "resolved_path": str(source.resolve()),
                    "sha256": sha256(source),
                    "bytes": source.stat().st_size,
                },
            ],
        },
        "render_settings": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "frame_spec": "686:687:1",
            "time_codes": [686],
            "time_codes_per_second": 60.0,
            "stage_time_code_bounds": [0.0, 686.0],
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_tries_per_frame": 20,
            "require_liquid_pixels": True,
            "replicator_global_seed": 310241,
        },
        "frame_count": 1,
        "frames": [
            {
                "capture_ordinal": 1,
                "time_code": 686,
                "reason_codes": [],
                "rgb": {"shape": [480, 640, 3], "dtype": "uint8"},
                "semantic": {"shape": [480, 640], "dtype": "uint8"},
                "instance": {"shape": [480, 640], "dtype": "uint8"},
                "class_ids": {"liquid": [2], "vial": [1]},
                "class_pixel_counts": {"liquid": liquid_count, "vial": vial_count},
                "instance_agreement": {
                    "passed": True,
                    "selected_liquid_instance_ids": [4],
                    "semantic_liquid_pixels": liquid_count,
                    "instance_liquid_pixels": liquid_count,
                },
                "artifacts": artifacts,
            }
        ],
    }
    summary_path = capture / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest_path = pilot_root / "attempt-manifests" / f"{sample_id}.json"
    manifest_payload = json.loads(manifest_path.read_text())
    planned_command = manifest_payload["command_plan"]["command"]
    preregistered_layer = manifest_payload["command_plan"]["appearance_layer"]
    inspection = {
        "path": str(layer.resolve()),
        "schema": preregistered_layer["schema"],
        "variant_id": preregistered_layer["variant_id"],
        "stage_sha256": preregistered_layer["stage_sha256"],
        "stage_bytes": preregistered_layer["stage_bytes"],
        "source_stage": preregistered_layer["source_stage"],
        "source_stage_sha256": preregistered_layer["source_stage_sha256"],
        "material_path": preregistered_layer["material_path"],
        "diffuse_color_linear": preregistered_layer["diffuse_color_linear"],
        "opacity": preregistered_layer["opacity"],
        "uncalibrated": True,
        "timing": {
            "start_time_code": 0.0,
            "end_time_code": 686.0,
            "frames_per_second": 60.0,
            "time_codes_per_second": 60.0,
        },
        "non_session_usd_layer_stack": [
            {
                "ordinal": record["ordinal"],
                "resolved_path": record["resolved_path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
            for record in summary["inputs"]["non_session_usd_layer_stack"]
        ],
    }
    attempt = {
        "schema_version": 1,
        "run_id": sample_id,
        "status": "mechanical_gates_passed",
        "visual_check_status": "pending",
        "failure_reasons": [],
        "requested_settings": settings,
        "timeout_seconds": 1800,
        "safe_command": [
            planned_command[3],
            str((REPO_ROOT / "scripts/twin_render_sequence.py").resolve()),
            "--twin",
            planned_command[9],
            "--stage",
            str(layer.resolve()),
            "--camera",
            "FillCam",
            "--frames",
            "686:687:1",
            "--subframes",
            "64",
            "--warmup",
            "60",
            "--max-tries",
            "20",
            "--seed",
            "310241",
            "--renderer",
            "PathTracing",
            "--require-liquid-pixels",
            "--out",
            str(capture.resolve()),
        ],
        "environment_policy": {"name": "fixture", "profile": "gb10-aarch64"},
        "capture": {
            "root": str(capture.resolve()),
            "verified": True,
            "appearance_provenance_verified": True,
            "inner_summary_sha256": sha256(summary_path),
            "artifact_count": 4,
        },
        "provenance": {
            "git": {"commit": COMMIT, "dirty": False},
            "experiment_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "sha256_after": sha256(manifest_path),
                "bytes_after": manifest_path.stat().st_size,
                "unchanged": True,
            },
            "appearance_layer": {
                "required": True,
                "path": str(layer.resolve()),
                "sha256": sha256(layer),
                "bytes": layer.stat().st_size,
                "sha256_after": sha256(layer),
                "bytes_after": layer.stat().st_size,
                "unchanged": True,
                "inspection_before": inspection,
                "inspection_after": inspection,
            },
            "twin": {"root": str((pilot_root / "source-twin").resolve()), "unchanged": True},
            "files": {
                "wrapper": {"unchanged": True},
                "renderer": {
                    "path": str((REPO_ROOT / "scripts/twin_render_sequence.py").resolve()),
                    "unchanged": True,
                },
                "runtime_lock": {"path": planned_command[5], "unchanged": True},
                "isaac_python": {
                    "before": {"lexical_path": planned_command[3]},
                    "unchanged": True,
                },
            },
        },
    }
    (attempt_root / "attempt.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n"
    )


DEFAULT_COLORS = {
    "R000": (200, 60, 20),
    "R001": (201, 60, 20),
    "R002": (74, 88, 111),
    "R003": (20, 100, 150),
    "R004": (21, 100, 150),
    "R005": (38, 96, 137),
    "R006": (20, 101, 150),
    "R007": (20, 100, 151),
}


def build_valid_pilot(
    pilot_root: Path,
    repo_root: Path,
    *,
    colors: dict[str, tuple[int, int, int]] | None = None,
    publish: bool = True,
) -> Path:
    write_graph_inputs(pilot_root)
    graph_path = build_experiment_graph(
        pilot_root,
        config_path=CONFIG_PATH,
        implementation_commit=COMMIT,
        repo_root=repo_root,
    )
    config = canonical_pilot_config()
    palette = DEFAULT_COLORS if colors is None else colors
    for row in config["rows"]:
        _write_capture(pilot_root, row, palette[row["sample_id"]])
    (pilot_root / "reviews").mkdir()
    reviewed_artifacts = []
    for row in config["rows"]:
        sample_id = row["sample_id"]
        attempt_root = pilot_root / sample_id
        summary_path = attempt_root / "capture/summary.json"
        summary = json.loads(summary_path.read_text())
        artifacts = summary["frames"][0]["artifacts"]
        reviewed_artifacts.append(
            {
                "sample_id": sample_id,
                "rgb_sha256": artifacts["rgb"]["sha256"],
                "semantic_sha256": artifacts["semantic"]["sha256"],
                "instance_sha256": artifacts["instance"]["sha256"],
                "mask_sha256": artifacts["liquid"]["sha256"],
                "summary_sha256": sha256(summary_path),
                "attempt_sha256": sha256(attempt_root / "attempt.json"),
            }
        )
    review = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "config_sha256": sha256(CONFIG_PATH),
        "experiment_graph_sha256": sha256(graph_path),
        "status": "accepted",
        "reviewer_id": "independent-reviewer",
        "independent_from_implementation": True,
        "reviewed_utc": "2026-10-24T22:00:00+00:00",
        "reviewed_sample_ids": [row["sample_id"] for row in config["rows"]],
        "reviewed_artifacts": reviewed_artifacts,
        "checks": {
            "recognizable_station_glass_liquid": True,
            "invariant_mask_alignment": True,
            "coherent_ordered_appearances": True,
            "broken_geometry_absent": True,
        },
    }
    review_path = pilot_root / "reviews/visual-review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    if publish:
        build_pilot_dataset(
            pilot_root,
            config_path=CONFIG_PATH,
            graph_path=graph_path,
            visual_review_path=review_path,
            repo_root=repo_root,
        )
    return pilot_root
