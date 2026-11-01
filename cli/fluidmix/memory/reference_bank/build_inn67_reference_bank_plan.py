#!/usr/bin/env python3
"""Publish the frozen INN-67 CPU-only reference-bank planning bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.memory.reference_bank.bank import build_reference_bank_bundle, probe_gb10_resources  # noqa: E402
from fluidmix.storage.artifacts.reference_bank_manifest import (  # noqa: E402
    load_reference_bank_twin_artifacts,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _plan_markdown(plan: dict[str, object], freeze: dict[str, object]) -> str:
    lines = [
        "# INN-67 development reference-bank render lane",
        "",
        "Status: frozen and unrendered. This bundle performs no GPU launch and authorizes no Brev use.",
        "",
        "The lane contains three independent development batches. Each batch has one amber/teal and one crimson/indigo subrun, three fills per subrun, for 18 fills and 48 stage observations. Batches 1–2 are reference donors; batch 3 is held-out development for the delta-E configuration check.",
        "",
        "Known-good means the synthetic case is prescribed with a clean start, zero leading residue, and no injected defect, then must pass the accepted render contract. It is not physical cleanliness or plant truth.",
        "",
        "## Hypothesis and stops",
        "",
        f"Hypothesis: {freeze['hypothesis']}",
        "",
        f"Falsifier: {freeze['falsifier']}",
        "",
        "Stop on foreign GPU/Isaac load, low or unreadable RAM, any binding mismatch, any accepted-validator failure, denominator drift, or access to WP1 one-pass/video5 artifacts.",
        "",
        "## Resource and wave protocol",
        "",
        "Before every wave run both `nvidia-smi` queries recorded in `reference-bank-plan.json` and `free -g`. Refuse foreign load. Route only to the local NVIDIA GB10, exactly two authorized sibling subruns at once, and never to Brev.",
        "",
        "The accepted serial supervisor currently classifies the second sibling as foreign. Therefore actual launch is blocked until a reviewed concurrency-2 coordinator can distinguish the two authorized siblings; do not bypass that safety check.",
        "",
        "## Post-render acceptance",
        "",
        "Run `python cli/twin/physx/validate_physx_f2_artifacts.py $CASE_ROOT/app-output` for every case. This is the accepted occlusion-aware WP1 validator. All 18 cases and all 48 planned stage observations must pass before population or delta-E work.",
        "",
        "## No-access boundary",
        "",
        "Only WP1 model-input recipe IDs, programmed volumes, and dispense orders were copied. No WP1 test case/run/batch ID, hash, locator, render output, one-pass input, or video5 artifact is included or opened by this bundle.",
    ]
    return "\n".join(lines) + "\n"


def _remove_owned_output(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the same directory inode created by this invocation."""
    try:
        current = path.stat(follow_symlinks=False)
    except OSError:
        return
    if (current.st_dev, current.st_ino) != identity or path.is_symlink():
        return
    shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-twin-manifest", required=True, type=Path)
    parser.add_argument("--published-twin-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-live-resource-probe", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("output directory already exists; publication is no-clobber")
    published_twin_artifacts = load_reference_bank_twin_artifacts(
        args.published_twin_manifest,
        expected_sha256=args.published_twin_manifest_sha256,
    )
    bundle = build_reference_bank_bundle(
        published_twin_artifacts=published_twin_artifacts
    )
    preflight = (
        {
            "schema_version": 1,
            "artifact_type": "inn67_gb10_resource_preflight",
            "status": "not_probed_by_explicit_cpu_option",
            "launch_performed": False,
            "brev_used": False,
        }
        if args.skip_live_resource_probe
        else probe_gb10_resources()
    )
    bundle["resource-preflight.json"] = preflight
    bundle["PLAN.md"] = _plan_markdown(
        bundle["reference-bank-plan.json"], bundle["delta-e-freeze-config.json"]
    )
    args.output_dir.mkdir(parents=True, mode=0o750)
    created = args.output_dir.stat(follow_symlinks=False)
    identity = (created.st_dev, created.st_ino)
    try:
        hashes = {}
        for relative, payload in sorted(bundle.items()):
            path = args.output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            body = payload.encode() if isinstance(payload, str) else _json_bytes(payload)
            path.write_bytes(body)
            hashes[relative] = hashlib.sha256(body).hexdigest()
        inventory = {
            "schema_version": 1,
            "artifact_type": "inn67_reference_bank_plan_hash_inventory",
            "files": hashes,
        }
        (args.output_dir / "artifact-hashes.json").write_bytes(_json_bytes(inventory))
        print(json.dumps({"status": "published_no_launch", "output_dir": str(args.output_dir), "resource_status": preflight["status"]}, sort_keys=True))
        return 0
    except BaseException:
        _remove_owned_output(args.output_dir, identity)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
