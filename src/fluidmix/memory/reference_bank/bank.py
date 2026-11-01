"""INN-67 development reference-bank planning boundary.

This module builds plans only.  It cannot launch Isaac Sim or any GPU work.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

from twin.cohorts.one_run_manifest_v1 import build_one_run_source_artifacts
from twin.physx.run_variation import (
    bind_tank_colour_resolution_to_artifacts,
    build_tank_colour_resolution,
)


ACCEPTED_RENDERER_BASE = "24889cad3cf9d7302622a4f844fb4b2a4977d4c9"
_RECIPES = {
    "amber_teal": (
        ("S0439", {"t100": 26.0, "t200": 28.0, "t400": 0.0}),
        ("S0440", {"t100": 11.0, "t200": 31.0, "t400": 20.0}),
        ("S0441", {"t100": 11.0, "t200": 11.0, "t400": 43.0}),
    ),
    "crimson_indigo": (
        ("S0442", {"t100": 56.0, "t200": 24.0, "t400": 0.0}),
        ("S0443", {"t100": 10.0, "t200": 50.0, "t400": 16.0}),
        ("S0444", {"t100": 11.0, "t200": 10.0, "t400": 39.0}),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order(volumes: dict[str, float]) -> list[str]:
    return [tank for tank in ("t200", "t400", "t100") if volumes[tank] > 0.0]


def _subrun_artifacts(
    repo_root: Path, batch_ordinal: int, palette: str
) -> dict[str, dict[str, object]]:
    palette_code = "AT" if palette == "amber_teal" else "CI"
    run_id = f"INN67-DEV-RB-B{batch_ordinal:02d}-{palette_code}-RUN"
    recipes = _RECIPES[palette]
    seed_base = 6_700_000 + batch_ordinal * 100 + (0 if palette_code == "AT" else 10)
    source = {
        "schema_version": 1,
        "artifact_type": "fluidmix-one-run-source-spec-v1",
        "run_id": run_id,
        "split_role": "development",
        "split_version": "inn67-development-reference-bank-v1",
        "source_sha256": {
            "recipe_manifest": _sha256(repo_root / "data" / "sim_recipes_v3.csv"),
            "split_spec": _sha256(repo_root / "data" / "sim_split_spec_v3.json"),
        },
        "recipes": [
            {"recipe_id": recipe_id, "programmed_ml": deepcopy(volumes)}
            for recipe_id, volumes in recipes
        ],
        "initial_condition": "clean_start",
        "initial_predecessor_recipe_id": None,
        "vial_recipe_ids": [recipe_id for recipe_id, _ in recipes],
        "residual_volume_ml_by_vial": [0.0, 0.0, 0.0],
        "render_seed_by_vial": [seed_base + offset for offset in (1, 2, 3)],
        "nuisance_group_id_by_vial": [
            f"INN67-DEV-RB-B{batch_ordinal:02d}-{palette_code}-V{ordinal:02d}"
            for ordinal in (1, 2, 3)
        ],
    }
    return bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(source),
        build_tank_colour_resolution(palette),
    )


def build_reference_bank_bundle(repo_root: str | Path) -> dict[str, object]:
    """Build the frozen CPU-only reference-bank plan."""
    root = Path(repo_root).resolve()
    bundle: dict[str, object] = {}
    batches = []
    subruns = []
    cases = []
    for batch_ordinal in (1, 2, 3):
        batch_id = f"INN67-DEV-RB-B{batch_ordinal:02d}"
        batch_role = (
            "development_reference_donor"
            if batch_ordinal < 3
            else "development_config_holdout"
        )
        batch_subruns = []
        for palette in ("amber_teal", "crimson_indigo"):
            palette_code = "AT" if palette == "amber_teal" else "CI"
            subrun_id = f"{batch_id}-{palette_code}-SUBRUN"
            run_id = f"{batch_id}-{palette_code}-RUN"
            prefix = f"subruns/{subrun_id}"
            generated = _subrun_artifacts(root, batch_ordinal, palette)
            for filename, payload in generated.items():
                bundle[f"{prefix}/{filename}"] = payload
            render_name = f"{prefix}/render-schedule-plan.json"
            subruns.append(
                {
                    "subrun_id": subrun_id,
                    "run_id": run_id,
                    "development_batch_id": batch_id,
                    "experimental_role": batch_role,
                    "palette": palette,
                    "fill_case_count": 3,
                    "render_plan_artifact": render_name,
                    "renderer_internal_batch_id": generated[
                        "render-schedule-plan.json"
                    ]["batches"][0]["batch_id"],
                }
            )
            batch_subruns.append(subrun_id)
            internal_rows = generated["render-schedule-plan.json"][
                "execution_schedules"
            ]
            for ordinal, ((recipe_id, volumes), internal) in enumerate(
                zip(_RECIPES[palette], internal_rows, strict=True), start=1
            ):
                order = _order(volumes)
                cases.append(
                    {
                        "case_id": f"{batch_id}-{palette_code}-{recipe_id}-CASE",
                        "batch_id": batch_id,
                        "subrun_id": subrun_id,
                        "run_id": run_id,
                        "split_role": "development",
                        "experimental_role": batch_role,
                        "palette": palette,
                        "recipe_id": recipe_id,
                        "programmed_ml": deepcopy(volumes),
                        "dispense_order": order,
                        "stages": [
                            {"ordinal": index, "tank_id": tank}
                            for index, tank in enumerate(order, start=1)
                        ],
                        "renderer_case_id": internal["case_binding"]["case_id"],
                        "render_seed": generated["render-schedule-plan.json"][
                            "batches"
                        ][0]["render_seed_by_vial"][ordinal - 1],
                    }
                )
        batches.append(
            {
                "batch_id": batch_id,
                "experimental_role": batch_role,
                "subrun_ids": batch_subruns,
                "fill_case_count": 6,
                "stage_observation_count": 16,
            }
        )

    wave_commands = []
    for wave, batch in enumerate(batches, start=1):
        commands = []
        for subrun_id in batch["subrun_ids"]:
            plan_path = f"$PLAN_ROOT/subruns/{subrun_id}/render-schedule-plan.json"
            commands.append(
                "python cli/twin/cohorts/run_physx_train_residual_cohort.py "
                f"--plan {plan_path} --plan-sha256 $PLAN_SHA256_{subrun_id.replace('-', '_')} "
                "--accepted-qualification-receipt $QUALIFICATION_RECEIPT "
                "--accepted-qualification-receipt-sha256 $QUALIFICATION_SHA256 "
                "--nuisance-receipt $KNOWN_GOOD_NUISANCE_RECEIPT "
                "--nuisance-receipt-sha256 $KNOWN_GOOD_NUISANCE_SHA256 "
                f"--output-root $RENDER_ROOT/{subrun_id} --stage $STAGE "
                "--renderer-script cli/twin/physx/physx_three_pour_sequence.py "
                "--python-executable $ISAAC_PYTHON "
                f"--host-lock /tmp/inn67-reference-bank-wave{wave}-{subrun_id}.lock"
            )
        wave_commands.append(
            {
                "wave": wave,
                "development_batch_id": batch["batch_id"],
                "concurrency": 2,
                "prelaunch_resource_check": (
                    "nvidia-smi --query-gpu=name --format=csv,noheader && "
                    "nvidia-smi --query-compute-apps=pid,process_name "
                    "--format=csv,noheader && free -g"
                ),
                "parallel_commands": commands,
                "execution_note": (
                    "run exactly these two sibling subruns together only through a "
                    "coordinator that distinguishes sibling Isaac processes from foreign load"
                ),
            }
        )

    plan = {
        "schema_version": 1,
        "artifact_type": "inn67_development_reference_bank_plan",
        "status": "frozen_unrendered",
        "accepted_renderer_base": ACCEPTED_RENDERER_BASE,
        "route": "local_GB10_only_no_Brev",
        "concurrency": 2,
        "denominators": {
            "development_batches": 3,
            "palette_subruns": 6,
            "fill_cases": 18,
            "stage_observations": 48,
            "reference_donor_batches": 2,
            "config_holdout_batches": 1,
        },
        "batches": batches,
        "subruns": subruns,
        "cases": cases,
        "wave_commands": wave_commands,
        "postrender_acceptance": {
            "required_validator": "cli/twin/physx/validate_physx_f2_artifacts.py",
            "command_per_case": (
                "python cli/twin/physx/validate_physx_f2_artifacts.py "
                "$CASE_ROOT/app-output"
            ),
            "occlusion_aware_wp1_validator_required": True,
            "accept_only_if_all_18_cases_pass": True,
        },
        "launch_blocker": (
            "the accepted serial supervisor treats the second authorized sibling as foreign; "
            "a reviewed concurrency-2 coordinator is required before these wave commands run"
        ),
    }
    profile = {
        "schema_version": 1,
        "artifact_type": "inn67_frozen_synthetic_known_good_profile",
        "status": "frozen",
        "injected_defect": False,
        "prescribed_leading_residual_volume_ml": 0.0,
        "initial_condition": "clean_start",
        "camera_lights_geometry_changed": False,
        "accepted_code_base": ACCEPTED_RENDERER_BASE,
        "physical_truth_claimed": False,
        "meaning": (
            "contract-known-good synthetic control only; acceptance does not establish "
            "physical cleanliness, plant truth, or sim-to-real validity"
        ),
    }
    freeze = {
        "schema_version": 1,
        "artifact_type": "inn67_delta_e_freeze_protocol",
        "status": "frozen_protocol_pending_render_measurements",
        "fit_batches": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
        "held_out_development_batch": "INN67-DEV-RB-B03",
        "threshold_value": None,
        "threshold_must_be_frozen_before_wp1_access": True,
        "hypothesis": (
            "two accepted independent known-good development donors per recipe and stage "
            "form a stable enough reference bank to freeze a finite delta-E configuration"
        ),
        "falsifier": (
            "any missing accepted stage, donor disagreement that prevents a finite rule, "
            "or failure to reproduce decisions on batch 3"
        ),
        "confounds": [
            "synthetic palette appearance is not physical absorption",
            "fixed camera and lighting can hide transfer failures",
            "recipe identity intentionally overlaps WP1 while render instances remain disjoint",
            "contract-known-good labels are prescribed rather than measured plant truth",
        ],
        "stops": [
            "foreign GPU or Isaac/Kit process",
            "resource preflight failure",
            "any source, plan, receipt, or validator hash mismatch",
            "any case fails the accepted occlusion-aware validator",
            "denominator differs from 18 fills or 48 stages",
            "any access to WP1 one-pass or video5 artifacts",
        ],
    }
    leakage = {
        "schema_version": 1,
        "artifact_type": "inn67_reference_bank_leakage_audit",
        "status": "pass",
        "development_ids_unique": True,
        "development_namespace": "INN67-DEV-RB-",
        "wp1_and_video5_identifier_namespaces_reused": False,
        "recipe_identity_overlap_with_wp1_test_is_required": True,
        "recipe_overlap_scope": "programmed recipe values only",
        "test_case_run_batch_ids_copied": False,
        "test_hashes_or_locators_copied": False,
        "wp1_render_outputs_opened": False,
        "video5_artifacts_opened": False,
        "batch3_excluded_from_threshold_fitting": True,
    }
    attestation = {
        "schema_version": 1,
        "artifact_type": "inn67_plan_prep_no_access_attestation",
        "status": "attested",
        "wp1_one_pass_opened": False,
        "video5_opened": False,
        "wp1_render_outputs_opened": False,
        "allowed_source_observation": (
            "accepted WP1 recipe/render-plan metadata was consulted; only recipe IDs, "
            "programmed volumes, and dispense orders were copied"
        ),
        "copied_fields": ["recipe_id", "programmed_ml", "dispense_order"],
        "excluded_fields": ["test case IDs", "test run IDs", "test batch IDs", "hashes", "locators"],
    }
    bundle.update(
        {
            "reference-bank-plan.json": plan,
            "known-good-profile.json": profile,
            "delta-e-freeze-config.json": freeze,
            "leakage-audit.json": leakage,
            "no-access-attestation.json": attestation,
        }
    )
    return bundle


def probe_gb10_resources(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    process_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Inspect GB10 resources without launching a renderer."""
    commands = [
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        ["free", "-g"],
    ]
    results = [
        run(command, text=True, capture_output=True, check=False) for command in commands
    ]
    gpu_name = results[0].stdout.strip() if results[0].returncode == 0 else ""
    foreign_gpu = []
    if results[1].returncode == 0:
        for line in results[1].stdout.splitlines():
            if not line.strip() or "No running" in line:
                continue
            fields = [field.strip() for field in line.split(",", 1)]
            if len(fields) == 2 and fields[0].isdigit():
                foreign_gpu.append({"pid": int(fields[0]), "process_name": fields[1]})
    if process_rows is None:
        foreign_processes = []
        markers = ("physx_three_pour_sequence.py", "isaac-sim", "isaacsim.exp", "omni.kit")
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                continue
            if command and any(marker in command for marker in markers):
                foreign_processes.append(
                    {"pid": int(entry.name), "command": command[:2048]}
                )
        foreign_processes.sort(key=lambda row: row["pid"])
    else:
        foreign_processes = list(process_rows)
    available_gib = None
    if results[2].returncode == 0:
        for line in results[2].stdout.splitlines():
            if line.lstrip().startswith("Mem:"):
                fields = line.split()
                if len(fields) >= 7 and fields[-1].isdigit():
                    available_gib = int(fields[-1])
    failures = []
    if gpu_name != "NVIDIA GB10":
        failures.append("host_gpu_is_not_exactly_NVIDIA_GB10")
    if foreign_gpu or foreign_processes:
        failures.append("foreign_gpu_or_isaac_load")
    if available_gib is None or available_gib < 32:
        failures.append("available_ram_below_32_GiB_or_unreadable")
    status = "pass_no_launch" if not failures else "refused"
    if foreign_gpu or foreign_processes:
        status = "refused_foreign_load"
    return {
        "schema_version": 1,
        "artifact_type": "inn67_gb10_resource_preflight",
        "status": status,
        "route": "GB10_only",
        "gpu_name": gpu_name or None,
        "available_ram_gib": available_gib,
        "foreign_gpu_processes": foreign_gpu,
        "foreign_isaac_or_kit_processes": foreign_processes,
        "commands": commands,
        "failures": failures,
        "launch_performed": False,
        "brev_used": False,
    }
