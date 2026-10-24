#!/usr/bin/env python3
"""Run exactly one frozen smoke or Phase-B FillCam render batch.

This orchestrator uses argument lists rather than a shell, records every command and
has one timeout with no retry. Outputs and logs belong under the ignored output root.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.render_manifest import load_render_manifest
from fluidmix.render_runner import (
    build_child_environment,
    build_render_command,
    build_variant_command,
    output_path_occupied,
    png_dimensions,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(command: list[str], log_path: Path, timeout_seconds: int, env: dict) -> dict:
    started = utc_now()
    before = time.monotonic()
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    return {
        "command": command,
        "started_utc": started,
        "ended_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - before,
        "returncode": returncode,
        "timed_out": timed_out,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    raise ValueError(f"executable not found: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("smoke", "batch"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
        manifest = load_render_manifest(manifest_path)
        isaac_python = resolve_executable(args.isaac_python)
        twin = Path(args.twin).expanduser().resolve(strict=True)
        renderer = (twin / "tools/render_example.py").resolve(strict=True)
        station = (twin / "stations/fill_station/fill_station.usda").resolve(strict=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    layers = out / "layers"
    images = out / "images"
    logs = out / "logs"
    layers.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    summary_path = out / f"summary-{args.phase}.json"
    if output_path_occupied(summary_path):
        parser.error(f"refusing to overwrite prior phase summary: {summary_path}")

    jobs = manifest.smoke if args.phase == "smoke" else manifest.batch
    for job in jobs:
        targets = [
            images / f"{job.variant_id}.png",
            layers / f"{job.variant_id}.usda",
            logs / f"{job.variant_id}.render.log",
        ]
        if job.material is not None:
            targets.append(logs / f"{job.variant_id}.variant.log")
        for target in targets:
            if output_path_occupied(target):
                parser.error(f"refusing to overwrite prior output: {target}")

    summary = {
        "schema_version": 1,
        "phase": args.phase,
        "started_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "isaac_python": isaac_python,
        "twin": str(twin),
        "station_sha256_before": sha256_file(station),
        "environment_overrides": {"OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA": "1"},
        "jobs": [],
        "status": "running",
    }
    save_summary(summary_path, summary)

    environment = build_child_environment(os.environ)
    variant_script = ROOT / "scripts/twin_liquid_variant.py"
    for job in jobs:
        record = {"id": job.variant_id, "frame": job.frame, "status": "running"}
        summary["jobs"].append(record)
        save_summary(summary_path, summary)

        layer = layers / f"{job.variant_id}.usda"
        variant_command = build_variant_command(
            isaac_python, str(variant_script), str(twin), layer, job
        )
        if variant_command is not None:
            variant_result = run_command(
                variant_command,
                logs / f"{job.variant_id}.variant.log",
                manifest.timeout_seconds,
                environment,
            )
            record["variant"] = variant_result
            if variant_result["returncode"] != 0 or variant_result["timed_out"]:
                record["status"] = "variant_failed"
                summary["status"] = "failed"
                summary["ended_utc"] = utc_now()
                save_summary(summary_path, summary)
                return 1
            record["layer_sha256"] = sha256_file(layer)
            stage = layer
        else:
            stage = station

        image = images / f"{job.variant_id}.png"
        render_command = build_render_command(
            isaac_python, str(renderer), stage, image, manifest, job
        )
        render_result = run_command(
            render_command,
            logs / f"{job.variant_id}.render.log",
            manifest.timeout_seconds,
            environment,
        )
        record["render"] = render_result
        if render_result["returncode"] != 0 or render_result["timed_out"]:
            record["status"] = "render_failed"
            summary["status"] = "failed"
            summary["ended_utc"] = utc_now()
            save_summary(summary_path, summary)
            return 1
        try:
            dimensions = png_dimensions(image)
        except (OSError, ValueError) as exc:
            record["status"] = "invalid_output"
            record["output_error"] = str(exc)
            summary["status"] = "failed"
            summary["ended_utc"] = utc_now()
            save_summary(summary_path, summary)
            return 1
        record["dimensions"] = list(dimensions)
        record["image_bytes"] = image.stat().st_size
        record["image_sha256"] = sha256_file(image)
        if dimensions != (640, 480):
            record["status"] = "wrong_dimensions"
            summary["status"] = "failed"
            summary["ended_utc"] = utc_now()
            save_summary(summary_path, summary)
            return 1
        record["status"] = "mechanical_checks_passed"
        save_summary(summary_path, summary)

    summary["station_sha256_after"] = sha256_file(station)
    summary["source_station_unchanged"] = (
        summary["station_sha256_before"] == summary["station_sha256_after"]
    )
    summary["status"] = "mechanical_checks_passed" if summary["source_station_unchanged"] else "failed"
    summary["ended_utc"] = utc_now()
    save_summary(summary_path, summary)
    return 0 if summary["source_station_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
