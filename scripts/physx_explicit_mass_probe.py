#!/usr/bin/env python3
"""Run A37: A36 physics with explicit density-equivalent runtime total mass."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fluidmix.explicit_mass_probe import (  # noqa: E402
    DENSITY_KG_M3,
    EXPERIMENT,
    MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
    NVIDIA_PARTICLE_GUIDE_URL,
    NVIDIA_PHYSX_PBD_URL,
    NVIDIA_RUNTIME_MASS_CHANGELOG_URL,
    PARTICLE_MASS_KG,
    SCHEMA_VERSION,
    SEED,
    TRANSITION_COUNT,
    build_probe_plan,
    evaluate_hypothesis,
    expected_total_mass_kg,
    nearest_neighbor_diagnostic,
    nearest_neighbor_distances,
    validate_mass_transitions,
    validate_probe_receipt,
)
from fluidmix.particle_array_probe import WALL_LIMIT_SECONDS  # noqa: E402

import physx_particle_array_probe as a36  # noqa: E402


INSTALLED_PARTICLE_UTILS_SOURCE = (
    "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/lib/python3.12/"
    "site-packages/isaacsim/extscache/omni.physx-110.4.4+110.4.0.la64.r.cp312.u7f4/"
    "omni/physx/scripts/particleUtils.py"
)
INSTALLED_PARTICLE_UTILS_SHA256 = (
    "85da0b0a39931eff3e54cea5ca8987c07d1c903557d87cb609b205ab85f9b3df"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    parser.add_argument(
        "--wall-limit-seconds",
        type=int,
        choices=(WALL_LIMIT_SECONDS,),
        default=WALL_LIMIT_SECONDS,
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _author_total_mass_transition(
    mass_api: Any,
    *,
    transition_ordinal: int,
    physics_step: int,
    active_particle_count: int,
) -> dict[str, object]:
    """Author and immediately read the NVIDIA-documented total set mass."""

    requested_total = expected_total_mass_kg(active_particle_count)
    mass_attr = mass_api.CreateMassAttr()
    set_result = mass_attr.Set(requested_total)
    readback = float(mass_attr.Get())
    density_attr = mass_api.GetDensityAttr()
    density_readback = float(density_attr.Get())
    record = {
        "transition_ordinal": transition_ordinal,
        "physics_step": physics_step,
        "active_particle_count": active_particle_count,
        "per_particle_mass_kg": PARTICLE_MASS_KG,
        "requested_total_mass_kg": requested_total,
        "readback_total_mass_kg": readback,
        "readback_per_particle_mass_kg": readback / active_particle_count,
        "mass_attr_authored": bool(mass_attr.HasAuthoredValueOpinion()),
        "density_readback_kg_m3": density_readback,
        "density_attr_authored": bool(density_attr.HasAuthoredValueOpinion()),
        "mass_overrides_density": True,
        "set_succeeded": bool(set_result),
        "readback_matches": math.isclose(
            readback,
            requested_total,
            rel_tol=0.0,
            abs_tol=MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
        ),
    }
    if (
        record["mass_attr_authored"] is not True
        or record["density_attr_authored"] is not True
        or record["set_succeeded"] is not True
        or record["readback_matches"] is not True
        or not math.isclose(
            density_readback,
            DENSITY_KG_M3,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or not math.isclose(
            float(record["readback_per_particle_mass_kg"]),
            PARTICLE_MASS_KG,
            rel_tol=0.0,
            abs_tol=(
                MASS_READBACK_ABSOLUTE_TOLERANCE_KG / active_particle_count
            ),
        )
    ):
        raise RuntimeError(
            "runtime total MassAPI readback mismatch: "
            + json.dumps(record, sort_keys=True)
        )
    return record


def _initialize_receipt(args: argparse.Namespace) -> dict[str, object]:
    receipt = a36._empty_receipt(args)
    receipt.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "plan": build_probe_plan(),
            "explicit_mass_treatment": {
                "diagnosis": (
                    "particleUtils authored total mass zero for the initially "
                    "empty PointInstancer; A37 dynamically authors the documented "
                    "density-equivalent total after every append"
                ),
                "formula": "8*fluid_rest_offset_m^3*density_kg_m3",
                "per_particle_mass_kg": PARTICLE_MASS_KG,
                "density_kg_m3": DENSITY_KG_M3,
                "transition_count": None,
                "expected_transition_count": TRANSITION_COUNT,
                "transitions": [],
                "installed_source": {
                    "path": INSTALLED_PARTICLE_UTILS_SOURCE,
                    "sha256": INSTALLED_PARTICLE_UTILS_SHA256,
                    "pointinstancer_lines": "256-324",
                    "mass_precedence_lines": "327-342",
                },
                "documentation": [
                    NVIDIA_PARTICLE_GUIDE_URL,
                    NVIDIA_RUNTIME_MASS_CHANGELOG_URL,
                    NVIDIA_PHYSX_PBD_URL,
                ],
                "claim_limit": (
                    "ablation tests the runtime-empty mass path; documentation "
                    "does not prove it caused A36 compactness"
                ),
            },
            "hypothesis_evaluation": None,
            "experiment_outcome": None,
        }
    )
    receipt["command"]["script_sha256"] = a36._sha256(Path(__file__))
    receipt["command"]["git_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt["command"]["git_status_short"] = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return receipt


def _attach_nn_diagnostics(
    output: Path,
    receipt: dict[str, object],
    manifest_records: list[dict[str, object]],
) -> None:
    import numpy as np

    for stage in receipt["stages"]:
        step = int(stage["physics_step"])
        raw_record = stage["arrays"]["raw_simulation_positions"]
        raw = np.load(output / raw_record["path"], allow_pickle=False)
        nearest = nearest_neighbor_distances(raw)
        relative = Path(f"step-{step:05d}/nearest_neighbor_distances.npy")
        artifact = a36._save_array(output, relative, nearest)
        stage["nearest_neighbor_diagnostic"] = nearest_neighbor_diagnostic(raw)
        stage["nearest_neighbor_artifact"] = artifact
        manifest_records.append(
            {
                "kind": "nearest_neighbor_distances",
                "physics_step": step,
                **artifact,
            }
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.stage.is_absolute() or not args.stage.is_file():
        parser.error("stage must be an existing absolute file")
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    a36._write_json(args.out / "plan.json", build_probe_plan())
    if args.plan_only:
        return 0

    receipt = _initialize_receipt(args)
    a36._write_json(args.out / "command.json", receipt["command"])
    app_owner = None
    runtime_records: list[dict[str, object]] = []
    exit_code = 2
    try:
        app, app_owner = a36.f2._start_or_reuse_kit()
        receipt, runtime_records = a36.run_probe(
            args,
            app,
            args.out,
            receipt,
            particle_mass_argument_kg=PARTICLE_MASS_KG,
            mass_transition_callback=_author_total_mass_transition,
        )
        treatment = receipt["explicit_mass_treatment"]
        treatment["transition_count"] = len(treatment["transitions"])
        transition_failures = validate_mass_transitions(treatment["transitions"])
        if transition_failures:
            raise RuntimeError(
                "explicit mass transition contract failed: "
                + ",".join(transition_failures)
            )
        _attach_nn_diagnostics(args.out, receipt, runtime_records)
        receipt["status"] = "pass"
        receipt["hypothesis_evaluation"] = evaluate_hypothesis(receipt)
        receipt["experiment_outcome"] = receipt["hypothesis_evaluation"][
            "decision"
        ]
    except Exception as exc:
        receipt["errors"].append(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
    try:
        manifest = a36._build_manifest(args.out, receipt, runtime_records)
        manifest["experiment"] = EXPERIMENT
        manifest_path = args.out / "artifact-manifest.json"
        a36._write_json(manifest_path, manifest)
        receipt["artifacts"] = {
            "manifest_path": manifest_path.relative_to(args.out).as_posix(),
            "manifest_sha256": a36._sha256(manifest_path),
            "record_count": len(manifest["records"]),
        }
        if receipt["status"] == "pass":
            failures = validate_probe_receipt(receipt, artifact_root=args.out)
            if failures:
                receipt["status"] = "insufficient_evidence"
                receipt["errors"].extend(failures)
        receipt_path = args.out / "receipt.json"
        a36._write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pass" else 2
    finally:
        if app_owner is not None:
            app_owner.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
