#!/usr/bin/env python3
"""Run A40: a bounded 1 mm-resolution F2 particle-volume probe."""

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


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from twin.probes import particle_array_probe as base_contract  # noqa: E402
from twin.probes.particle_array_probe import WALL_LIMIT_SECONDS  # noqa: E402
from twin.probes.resolution_probe import (  # noqa: E402
    CAPTURE_STEPS,
    CONTACT_OFFSET_M,
    DENSITY_KG_M3,
    EXPECTED_PARTICLE_COUNTS,
    EXPERIMENT,
    FLUID_REST_OFFSET_M,
    MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
    MAXIMUM_PARTICLE_COUNT,
    MAX_VELOCITY_M_S,
    PARTICLE_CONTACT_OFFSET_M,
    PARTICLE_MASS_KG,
    PHYSICS_HZ,
    REST_OFFSET_M,
    SCHEMA_VERSION,
    SEED,
    SOLID_REST_OFFSET_M,
    build_probe_plan,
    build_resolution_runtime_plan,
    evaluate_hypothesis,
    expected_total_mass_kg,
    nearest_neighbor_diagnostic,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cli.twin.probes.physx_particle_array_probe as a36  # noqa: E402


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


def _install_resolution_contract() -> None:
    """Configure the shared A36 runner without changing its default behavior."""

    shared = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "EXPERIMENT": EXPERIMENT,
        "EXPECTED_PARTICLE_COUNTS": EXPECTED_PARTICLE_COUNTS,
        "PARTICLE_CONTACT_OFFSET_M": PARTICLE_CONTACT_OFFSET_M,
        "CONTACT_OFFSET_M": CONTACT_OFFSET_M,
        "PARTICLE_TO_NONPARTICLE_REST_OFFSET_M": REST_OFFSET_M,
        "SOLID_REST_OFFSET_M": SOLID_REST_OFFSET_M,
        "PARTICLE_DENSITY_ARGUMENT_KG_M3": DENSITY_KG_M3,
        "PARTICLE_MASS_ARGUMENT_KG": PARTICLE_MASS_KG,
        "FLUID_REST_OFFSET_M": FLUID_REST_OFFSET_M,
        "MAX_VELOCITY_M_S": MAX_VELOCITY_M_S,
    }
    for module in (a36, base_contract):
        for name, value in shared.items():
            setattr(module, name, value)
        module.build_probe_plan = build_probe_plan
    a36.build_paint_ball_three_pour_plan = build_resolution_runtime_plan

    def chunk_positions(Gf: Any, count: int, rng: Any) -> list[Any]:
        spacing = 2.0 * FLUID_REST_OFFSET_M
        values = []
        for index in range(count):
            ix = index % 3
            iy = (index // 3) % 3
            iz = index // 9
            values.append(
                Gf.Vec3f(
                    a36.NOZZLE_EXIT_CENTER_M[0]
                    + (ix - 1) * spacing
                    + (rng.random() - 0.5) * spacing * 0.1,
                    a36.NOZZLE_EXIT_CENTER_M[1]
                    + (iy - 1) * spacing
                    + (rng.random() - 0.5) * spacing * 0.1,
                    min(
                        a36.NOZZLE_EXIT_CENTER_M[2] - spacing,
                        a36.VIAL_MOUTH_M - 0.001,
                    )
                    - iz * spacing,
                )
            )
        return values

    a36.f2._twin_chunk_positions = chunk_positions


def _author_mass_transition(
    mass_api: Any,
    *,
    transition_ordinal: int,
    physics_step: int,
    active_particle_count: int,
) -> dict[str, object]:
    requested = expected_total_mass_kg(active_particle_count)
    mass_attr = mass_api.CreateMassAttr()
    set_succeeded = mass_attr.Set(requested)
    readback = float(mass_attr.Get())
    density_attr = mass_api.GetDensityAttr()
    density = float(density_attr.Get())
    record = {
        "transition_ordinal": transition_ordinal,
        "physics_step": physics_step,
        "active_particle_count": active_particle_count,
        "per_particle_mass_kg": PARTICLE_MASS_KG,
        "requested_total_mass_kg": requested,
        "readback_total_mass_kg": readback,
        "readback_per_particle_mass_kg": readback / active_particle_count,
        "mass_attr_authored": bool(mass_attr.HasAuthoredValueOpinion()),
        "density_readback_kg_m3": density,
        "density_attr_authored": bool(density_attr.HasAuthoredValueOpinion()),
        "mass_overrides_density": True,
        "set_succeeded": bool(set_succeeded),
        "readback_matches": math.isclose(
            readback,
            requested,
            rel_tol=0.0,
            abs_tol=MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
        ),
    }
    if (
        record["mass_attr_authored"] is not True
        or record["density_attr_authored"] is not True
        or record["set_succeeded"] is not True
        or record["readback_matches"] is not True
        or not math.isclose(density, DENSITY_KG_M3, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise RuntimeError(
            "A40 runtime mass readback mismatch: "
            + json.dumps(record, sort_keys=True)
        )
    return record


def _expected_transitions() -> list[dict[str, int]]:
    cumulative = 0
    expected = []
    for pour in build_resolution_runtime_plan()["pours"][:2]:
        for event in pour["emission_schedule"]:
            cumulative += int(event["particle_count"])
            expected.append(
                {
                    "transition_ordinal": len(expected) + 1,
                    "physics_step": int(event["physics_step"]),
                    "active_particle_count": cumulative,
                }
            )
    return expected


def _validate_mass_transitions(records: object) -> tuple[str, ...]:
    expected = _expected_transitions()
    if not isinstance(records, list) or len(records) != len(expected):
        return ("mass_transition_count_mismatch",)
    failures = []
    for record, frozen in zip(records, expected):
        if not isinstance(record, dict) or any(
            record.get(name) != value for name, value in frozen.items()
        ):
            failures.append("mass_transition_schedule_mismatch")
            continue
        count = frozen["active_particle_count"]
        total = expected_total_mass_kg(count)
        if (
            not math.isclose(
                float(record.get("per_particle_mass_kg", math.nan)),
                PARTICLE_MASS_KG,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(record.get("requested_total_mass_kg", math.nan)),
                total,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(record.get("readback_total_mass_kg", math.nan)),
                total,
                rel_tol=0.0,
                abs_tol=MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
            )
            or record.get("readback_matches") is not True
        ):
            failures.append("mass_transition_readback_mismatch")
    return tuple(dict.fromkeys(failures))


def _initialize_receipt(args: argparse.Namespace) -> dict[str, object]:
    receipt = a36._empty_receipt(args)
    receipt.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "plan": build_probe_plan(),
            "resolution_intervention": {
                "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
                "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
                "contact_offset_m": CONTACT_OFFSET_M,
                "rest_offset_m": REST_OFFSET_M,
                "solid_rest_offset_m": SOLID_REST_OFFSET_M,
                "per_particle_mass_kg": PARTICLE_MASS_KG,
                "maximum_particle_count": MAXIMUM_PARTICLE_COUNT,
                "spawn_spacing_m": 2.0 * FLUID_REST_OFFSET_M,
                "diffusion_enabled": False,
            },
            "explicit_mass_treatment": {
                "transitions": [],
                "transition_count": None,
                "expected_transition_count": len(_expected_transitions()),
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
        diagnostic = nearest_neighbor_diagnostic(raw)
        stage["nearest_neighbor_diagnostic"] = diagnostic
        nearest_path = Path(f"step-{step:05d}/nearest-neighbor-summary.json")
        a36._write_json(output / nearest_path, diagnostic)
        record = {
            "kind": "nearest_neighbor_summary",
            "physics_step": step,
            "path": nearest_path.as_posix(),
            "sha256": a36._sha256(output / nearest_path),
        }
        stage["nearest_neighbor_artifact"] = record
        manifest_records.append(record)


def _validate_receipt(receipt: dict[str, object], output: Path) -> tuple[str, ...]:
    failures = list(
        base_contract.validate_probe_receipt(
            receipt,
            artifact_root=output,
            expected_plan=build_probe_plan(),
            expected_experiment=EXPERIMENT,
            expected_particle_mass_argument_kg=PARTICLE_MASS_KG,
            expected_max_particles=MAXIMUM_PARTICLE_COUNT,
        )
    )

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    treatment = receipt.get("explicit_mass_treatment")
    transitions = treatment.get("transitions") if isinstance(treatment, dict) else None
    for reason in _validate_mass_transitions(transitions):
        fail(reason)
    if (
        not isinstance(treatment, dict)
        or treatment.get("transition_count") != len(_expected_transitions())
    ):
        fail("mass_transition_count_mismatch")
    diffusion = receipt.get("diffusion")
    if (
        not isinstance(diffusion, dict)
        or diffusion.get("enabled") is not False
        or diffusion.get("update_count") != 0
        or diffusion.get("total_neighbor_count") != 0
    ):
        fail("diffusion_not_disabled")
    for stage in receipt.get("stages", []):
        diagnostic = stage.get("nearest_neighbor_diagnostic")
        artifact = stage.get("nearest_neighbor_artifact")
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("acceptance_gate") is not False
            or not isinstance(artifact, dict)
        ):
            fail("nearest_neighbor_diagnostic_missing")
            continue
        path = output / str(artifact.get("path", ""))
        if (
            not path.is_file()
            or a36._sha256(path) != artifact.get("sha256")
            or json.loads(path.read_text(encoding="utf-8")) != diagnostic
        ):
            fail("nearest_neighbor_artifact_invalid")
    expected_evaluation = evaluate_hypothesis(receipt)
    if receipt.get("hypothesis_evaluation") != expected_evaluation:
        fail("hypothesis_evaluation_mismatch")
    if receipt.get("experiment_outcome") != expected_evaluation["decision"]:
        fail("experiment_outcome_mismatch")
    return tuple(failures)


def main(argv: list[str] | None = None) -> int:
    _install_resolution_contract()
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
            mass_transition_callback=_author_mass_transition,
            max_particles=MAXIMUM_PARTICLE_COUNT,
            diffusion_enabled=False,
        )
        receipt["explicit_mass_treatment"]["transition_count"] = len(
            receipt["explicit_mass_treatment"]["transitions"]
        )
        mass_failures = _validate_mass_transitions(
            receipt["explicit_mass_treatment"]["transitions"]
        )
        if mass_failures:
            raise RuntimeError(
                "A40 mass transition contract failed: " + ",".join(mass_failures)
            )
        _attach_nn_diagnostics(args.out, receipt, runtime_records)
        receipt["status"] = "pass"
        receipt["hypothesis_evaluation"] = evaluate_hypothesis(receipt)
        receipt["experiment_outcome"] = receipt["hypothesis_evaluation"]["decision"]
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
            failures = _validate_receipt(receipt, args.out)
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
