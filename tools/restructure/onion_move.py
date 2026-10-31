#!/usr/bin/env python3
"""Plan and replay the approved onion-layout move without guessing names."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

DEFAULT_MAPPING = Path(__file__).resolve().parents[2].parent / "overnight-ops-20261028/ONION-MAPPING.json"

# Responsibility names are explicit: ticket/version prefixes are never stripped by a rule.
RENAMES = {
    "deploy_input_policy.py": "policy.py",
    "rgb_mask_prediction.py": "masks.py",
    "predicted_mask_materializer.py": "materializer.py",
    "stage_decision.py": "stage.py",
    "wp7_final_decision.py": "vial.py",
    "f2_varied_triage.py": "triage.py",
    "inn67_reference_bank.py": "bank.py",
    "inn67_bank_materializer.py": "materializer.py",
    "inn67_predicted_store_producer.py": "predicted_store_producer.py",
    "inn67_postrender_producer.py": "postrender_producer.py",
    "wp7_history_preparation.py": "history.py",
    "wp7_development_reference.py": "development_reference.py",
    "wp7_rgb_runtime.py": "rgb_runtime.py",
    "f2_varied_demo.py": "varied_demo.py",
    "f2_varied_stage_measurement.py": "varied_stage_measurement.py",
    "f2_varied_explanation.py": "varied_explanation.py",
    "f2_downstream_binding.py": "downstream_binding.py",
    "f2_receipt_manifest_publisher.py": "receipt_manifest_publisher.py",
    "f2_stage_cohort_materializer.py": "stage_cohort_materializer.py",
    "f2_varied_stage_materializer.py": "varied_stage_materializer.py",
    "f2_varied_image_locator.py": "varied_image_locator.py",
    "f2_fill_clip_inputs.py": "fill_clip_inputs.py",
    "v3_cosmos_contract.py": "cosmos_contract.py",
    "v3_cosmos_inputs.py": "cosmos_inputs.py",
    "v3_cosmos_runtime.py": "cosmos_runtime.py",
    "v3_cosmos_scoring.py": "cosmos_scoring.py",
    "v3_cosmos_plan_artifact.py": "cosmos_plan_artifact.py",
    "physx_case_plan.py": "case_plan.py",
    "physx_composition_cohort.py": "composition_cohort.py",
    "physx_dispense_residual.py": "dispense_residual.py",
    "physx_particle_contract.py": "particle_contract.py",
    "physx_residual_case.py": "residual_case.py",
    "physx_residual_pilot.py": "residual_pilot.py",
    "physx_residual_schedule.py": "residual_schedule.py",
    "physx_run_variation.py": "run_variation.py",
    "physx_three_pour.py": "three_pour.py",
    "physx_volume_gate.py": "volume_gate.py",
    "physx_one_run_manifest_v1.py": "one_run_manifest_v1.py",
    "physx_train_residual_manifest_v2.py": "train_residual_manifest_v2.py",
    "physx_varied_cohort_manifest_v3.py": "varied_cohort_manifest_v3.py",
    "physx_varied_cohort_manifest_v4.py": "varied_cohort_manifest_v4.py",
    "carryover_pipeline.py": "carryover_pipeline.py",
}

# These mapping keys name leaf responsibilities, not directories. Their members live in
# the parent directory and RENAMES selects the responsibility filename where necessary.
LEAF_KEYS = {"core/policy", "perception/masks"}

# Scripts do not reliably share a name with their implementation module. Every routed
# script is therefore listed. Unlisted tracked scripts are deliberately reported unmapped.
SCRIPT_ROUTES = {
    "align_still_labels.py": "core/policy",
    "audit_cnn_holdout_recovery.py": "perception/cnn",
    "audit_cosmos_nano_inputs.py": "explanation/vlm",
    "audit_cosmos_nano_usable_inputs.py": "explanation/vlm",
    "audit_sixth_development_reads.py": "memory/history",
    "audit_stage_cosmos_nano_inputs.py": "explanation/vlm",
    "audit_v3_cosmos_nano_inputs.py": "explanation/vlm",
    "brev_isaac_setup.sh": "sim/physx",
    "build_brev_video_cohort.py": "sim/render",
    "build_cnn_training_inventory.py": "perception/cnn",
    "build_f2_fill_clip_inputs.py": "sim/cohorts",
    "build_f2_varied_demo.py": "pipeline",
    "build_f2_varied_image_locators.py": "sim/cohorts",
    "build_f2_varied_stage_measurements.py": "measurement",
    "build_inn67_reference_bank_plan.py": "memory/reference_bank",
    "build_model_input_plan_from_manifests.py": "explanation/vlm",
    "build_physx_case_plan.py": "sim/physx",
    "build_physx_composition_cohort.py": "sim/physx",
    "build_physx_review_contact_sheet.py": "sim/render",
    "build_physx_train_residual_manifest_v2.py": "sim/cohorts",
    "build_physx_varied_cohort_manifest_v3.py": "sim/cohorts",
    "build_physx_varied_cohort_manifest_v4.py": "sim/cohorts",
    "build_rca_bench_inputs.py": "evaluation/rca_bench",
    "build_rca_bench_oracle.py": "evaluation/rca_bench",
    "build_real_photometric_manifest.py": "storage/artifacts",
    "build_stage_cosmos_attempt_plan.py": "evaluation/scoring",
    "build_stage_measured_evidence.py": "measurement",
    "build_tank_station.py": "sim/scene",
    "build_transitions.py": "memory/history",
    "build_transport_stage.py": "sim/transport",
    "build_twin_pilot_dataset.py": "sim/probes",
    "build_twin_vision_plan.py": "memory/history",
    "build_v3_cosmos_attempt_plan.py": "evaluation/scoring",
    "build_validation_measurement_inputs.py": "measurement",
    "check_data.py": "core/policy",
    "check_leakage.py": "core/policy",
    "check_render_dataset.py": "sim/probes",
    "check_rigid_vial_transport.py": "sim/transport",
    "check_tank_capture.py": "sim/render",
    "check_transport_stage.py": "sim/transport",
    "color.py": "core/policy",
    "compose_run_sequence_videos.py": "sim/render",
    "cosmos_batch.py": "explanation/vlm",
    "cosmos_smoke.py": "explanation/vlm",
    "decide_f2_varied_triads.py": "decision",
    "decide_wp1_visibility.py": "sim/postrender",
    "decide_wp7_varied_triads.py": "decision",
    "encode_palette_cut_reel.py": "sim/render",
    "encode_physx_review_video.py": "sim/render",
    "encode_twin_capture_video.py": "sim/render",
    "evaluate_appearance_pilot.py": "sim/probes",
    "fit_absorbance_video5.py": "colour",
    "fixlabels.py": "core/policy",
    "inventory_data.py": "core/policy",
    "make_splits.py": "sim/cohorts",
    "materialize_empty_reference_predictions.py": "perception/masks",
    "materialize_f2_stage_cohort.py": "sim/cohorts",
    "materialize_f2_varied_stage_cohort.py": "sim/cohorts",
    "materialize_inn67_predicted_store.py": "memory/reference_bank",
    "materialize_inn67_reference_bank.py": "memory/reference_bank",
    "materialize_sixth_development_run.py": "memory/history",
    "measure_fill_to_cap_delay.py": "measurement",
    "measure_real_photometric_ranges.py": "measurement",
    "measure_validation_stage_evidence.py": "measurement",
    "measure_video_clock.py": "core/policy",
    "merge_physx_train_residual_shards.py": "sim/cohorts",
    "package_run_sequence_dataset.py": "sim/postrender",
    "package_wp1_test_videos.py": "sim/postrender",
    "physx_explicit_mass_probe.py": "sim/probes",
    "physx_isosurface_emission_probe.py": "sim/probes",
    "physx_manual_step_probe.py": "sim/probes",
    "physx_parse_registration_probe.py": "sim/probes",
    "physx_particle_array_probe.py": "sim/probes",
    "physx_particle_compat_probe.py": "sim/probes",
    "physx_particle_palette_smoke.py": "sim/probes",
    "physx_resolution_probe.py": "sim/probes",
    "physx_three_pour_sequence.py": "sim/physx",
    "physx_timeline_boundary_probe.py": "sim/probes",
    "physx_timeline_emission_probe.py": "sim/probes",
    "physx_timeline_emission_probe_a18.py": "sim/probes",
    "physx_volume_review.py": "sim/physx",
    "populate_predicted_evidence_store.py": "storage/sqlite",
    "prepare_wp7_development_history.py": "memory/history",
    "preview_tank_station.py": "sim/scene",
    "produce_inn67_postrender_inputs.py": "memory/reference_bank",
    "publish_empty_reference_development.py": "perception/baselines",
    "publish_f2_stage_batch.py": "sim/cohorts",
    "publish_f2_varied_stage_batch.py": "sim/cohorts",
    "real_check_windows.py": "core/policy",
    "render_capseal_structural_pilot.py": "sim/scene",
    "render_rigid_vial_transport.py": "sim/transport",
    "render_tank_station.py": "sim/scene",
    "render_transport_stage.py": "sim/transport",
    "render_twin_dataset.py": "sim/render",
    "replay_cnn_holdout_bounds.py": "perception/cnn",
    "run_brev_video_cohort.py": "sim/render",
    "run_carryover_pipeline.py": "pipeline",
    "run_cnn_empty_control_once.py": "perception/cnn",
    "run_cosmos_large_eval.py": "explanation/vlm",
    "run_cosmos_nano_usable_eval.py": "explanation/vlm",
    "run_model_bakeoff.py": "evaluation/rca_bench",
    "run_physx_f2_remote.sh": "sim/physx",
    "run_physx_palette_smoke_remote.sh": "sim/physx",
    "run_physx_train_residual_cohort.py": "sim/cohorts",
    "run_rca_bench.py": "evaluation/rca_bench",
    "run_spatial_rgb_mask_predictor.py": "perception/baselines",
    "run_stage_cosmos_nano_eval.py": "explanation/vlm",
    "run_temporal_rgb_mask_baseline.py": "perception/baselines",
    "run_v3_cosmos_nano_eval.py": "explanation/vlm",
    "score_cnn_holdout_once.py": "perception/cnn",
    "score_cosmos_nano_matched.py": "evaluation/scoring",
    "score_cosmos_nano_usable.py": "evaluation/scoring",
    "score_empty_reference_development.py": "evaluation/scoring",
    "score_stage_cosmos_nano.py": "evaluation/scoring",
    "score_synthetic_stage_decisions.py": "evaluation/scoring",
    "score_v3_cosmos_nano.py": "evaluation/scoring",
    "smoke_absorbance.py": "colour",
    "spatial_omnisurface_capability_probe.py": "sim/probes",
    "stage_cosmos_nano_reasoner.py": "explanation/vlm",
    "stream_sequence.py": "core/policy",
    "tank_capture_provenance.py": "sim/render",
    "tank_preview.html": "sim/scene",
    "train_cnn_liquid_mask.py": "perception/cnn",
    "twin_appearance_variant.py": "sim/scene",
    "twin_contact_sheet.py": "sim/render",
    "twin_liquid_variant.py": "sim/scene",
    "twin_render_sequence.py": "sim/render",
    "twin_transmissive_liquid_variant.py": "sim/scene",
    "validate_physx_f2_artifacts.py": "sim/physx",
    "validate_physx_isosurface_emission_probe.py": "sim/probes",
    "validate_physx_timeline_emission_probe.py": "sim/probes",
    "validate_physx_timeline_emission_probe_a18.py": "sim/probes",
    "validate_wp1_acceptance.py": "sim/postrender",
    "verify_still_alignment.py": "core/policy",
    "wp1_dp3_fallback.py": "sim/postrender",
}

# Tests without a one-to-one mapped module name are routed explicitly. Destination
# names describe the responsibility and remove ticket prefixes where the folder does.
TEST_ROUTES = {
    "fixtures/inn67-materialization-protocol-amendment.json": (
        "memory/reference_bank", "fixtures/inn67-materialization-protocol-amendment.json"
    ),
    "pilot_fixture.py": ("sim/probes", "pilot_fixture.py"),
    "test_c20_physics_substeps.py": ("sim/physx", "test_physics_substeps.py"),
    "test_c21_particle_resolution.py": ("sim/physx", "test_particle_resolution.py"),
    "test_carryover_pipeline_final_wiring.py": (
        "pipeline", "test_carryover_pipeline_final_wiring.py"
    ),
    "test_build_stage_measured_evidence.py": (
        "measurement", "test_build_stage_measured_evidence.py"
    ),
    "test_contract_schemas.py": ("core/contracts", "test_schemas.py"),
    "test_cosmos_nano_inputs_runner.py": ("explanation/vlm", "test_cosmos_nano_inputs_runner.py"),
    "test_cosmos_nano_preflight.py": ("explanation/vlm", "test_cosmos_nano_preflight.py"),
    "test_cosmos_usable_preflight.py": ("explanation/vlm", "test_cosmos_usable_preflight.py"),
    "test_data_paths.py": ("core/policy", "test_paths.py"),
    "test_f2_downstream_contract.py": ("sim/cohorts", "test_downstream_contract.py"),
    "test_f2_varied_triage_cli.py": ("decision", "test_triage_cli.py"),
    "test_paint_ball_emitter_fallback.py": ("sim/physx", "test_paint_ball_emitter_fallback.py"),
    "test_part_boundary.py": ("storage/artifacts", "test_part_boundary.py"),
    "test_physx_f2_artifact_validator.py": ("sim/physx", "test_f2_artifact_validator.py"),
    "test_physx_f2_remote_launcher.py": ("sim/physx", "test_f2_remote_launcher.py"),
    "test_physx_isosurface_emission_probe.py": ("sim/probes", "test_isosurface_emission_probe.py"),
    "test_physx_isosurface_emission_validator.py": (
        "sim/probes", "test_isosurface_emission_validator.py"
    ),
    "test_physx_manual_step_probe.py": ("sim/probes", "test_manual_step_probe.py"),
    "test_physx_nuisance_authoring.py": ("sim/scene", "test_nuisance_authoring.py"),
    "test_physx_palette_smoke_remote.py": ("sim/physx", "test_palette_smoke_remote.py"),
    "test_physx_particle_palette_smoke.py": ("sim/probes", "test_particle_palette_smoke.py"),
    "test_physx_particle_probe.py": ("sim/probes", "test_particle_probe.py"),
    "test_physx_runtime_contract.py": ("sim/physx", "test_runtime_contract.py"),
    "test_physx_three_pour_runner_fallback.py": (
        "sim/physx", "test_three_pour_runner_fallback.py"
    ),
    "test_physx_timeline_boundary_probe.py": ("sim/probes", "test_timeline_boundary_probe.py"),
    "test_physx_train_residual_supervisor.py": (
        "sim/cohorts", "test_train_residual_supervisor.py"
    ),
    "test_physx_train_schedule_runner.py": ("sim/cohorts", "test_train_schedule_runner.py"),
    "test_physx_varied_cohort_supervisor.py": (
        "sim/cohorts", "test_varied_cohort_supervisor.py"
    ),
    "test_physx_volume_review.py": ("sim/physx", "test_volume_review.py"),
    "test_pilot_scripts.py": ("sim/probes", "test_pilot_scripts.py"),
    "test_q2_no_render_mode.py": ("sim/render", "test_no_render_mode.py"),
    "test_recipe8_runner_path.py": ("pipeline", "test_runner_path.py"),
    "test_render_launcher.py": ("sim/render", "test_launcher.py"),
    "test_render_watchdog.py": ("sim/render", "test_watchdog.py"),
    "test_review_regressions.py": ("colour", "test_review_regressions.py"),
    "test_splits.py": ("sim/cohorts", "test_splits.py"),
    "test_stage_cosmos_runner_scoring.py": (
        "explanation/vlm", "test_stage_cosmos_runner_scoring.py"
    ),
    "test_stream_sequence.py": ("core/policy", "test_stream_sequence.py"),
    "test_transitions.py": ("memory/history", "test_transitions.py"),
    "test_twin_contact_sheet.py": ("sim/render", "test_contact_sheet.py"),
    "test_wp7_development_reference_store.py": (
        "memory/history", "test_development_reference_store.py"
    ),
    "test_wp7_route_preflight.py": ("pipeline", "test_route_preflight.py"),
}


def _mapping(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("refusing: mapping must be a JSON object")
    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        _validate_key(key)
        if not isinstance(value, str):
            raise SystemExit(f"refusing: mapping value for {key!r} must be a string")
        names = value.split()
        if not names or any(not name.isidentifier() for name in names):
            raise SystemExit(f"refusing: unsafe module name in mapping key {key!r}")
        result[key] = names
    return result


def _validate_key(key: object) -> None:
    if not isinstance(key, str) or not key:
        raise SystemExit(f"refusing: unsafe mapping key {key!r}")
    path = Path(key)
    parts = path.parts
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise SystemExit(f"refusing: unsafe mapping key {key!r}")
    if key == "core/contracts":
        return
    first = parts[0]
    allowed = {
        "core", "colour", "perception", "measurement", "decision",
        "explanation", "storage", "memory", "pipeline", "sim", "evaluation",
    }
    if first not in allowed:
        raise SystemExit(f"refusing: unsafe mapping key {key!r}")


def _package_and_tail(key: str) -> tuple[str, Path]:
    path = Path(key)
    if path.parts[0] == "sim":
        return "twin", Path(*path.parts[1:])
    if path.parts[0] == "evaluation":
        return "evaluation", Path(*path.parts[1:])
    return "fluidmix", path


def _source_destination(key: str, source_name: str) -> Path:
    package, tail = _package_and_tail(key)
    filename = RENAMES.get(source_name + ".py", source_name + ".py")
    if key in LEAF_KEYS:
        tail = tail.parent
    return Path("src") / package / tail / filename


def _route_directory(key: str, top: str) -> Path:
    package, tail = _package_and_tail(key)
    if key in LEAF_KEYS:
        tail = tail.parent
    return Path(top) / package / tail


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _tracked(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "src", "scripts", "tests"]
    )
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def _add_move(
    root: Path,
    tracked: set[Path],
    moves: list[tuple[Path, Path]],
    covered: set[Path],
    source: Path,
    destination: Path,
) -> None:
    if not _inside(root, root / destination):
        raise SystemExit(f"refusing: destination escapes repository: {destination}")
    if destination in tracked:
        covered.add(destination)
    if source in tracked:
        moves.append((root / source, root / destination))
        covered.add(source)


def plan(root: Path, mapping_path: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    mapping = _mapping(mapping_path)
    tracked_list = _tracked(root)
    tracked = set(tracked_list)
    moves: list[tuple[Path, Path]] = []
    covered: set[Path] = {Path("src/fluidmix/__init__.py")} & tracked

    for key, names in mapping.items():
        if key == "core/contracts":
            if names != ["contracts"]:
                raise SystemExit("refusing: core/contracts must map the contracts package")
            old_prefix = Path("src/fluidmix/contracts")
            new_prefix = Path("src/fluidmix/core/contracts")
            for item in tracked_list:
                if item.is_relative_to(old_prefix):
                    _add_move(root, tracked, moves, covered, item, new_prefix / item.relative_to(old_prefix))
                elif item.is_relative_to(new_prefix):
                    covered.add(item)
            continue
        for name in names:
            source = Path("src/fluidmix") / f"{name}.py"
            destination = _source_destination(key, name)
            _add_move(root, tracked, moves, covered, source, destination)

            test_source = Path("tests") / f"test_{name}.py"
            responsibility = Path(RENAMES.get(f"{name}.py", f"{name}.py")).stem
            test_destination = _route_directory(key, "tests") / f"test_{responsibility}.py"
            _add_move(root, tracked, moves, covered, test_source, test_destination)

    for filename, key in SCRIPT_ROUTES.items():
        if key not in mapping:
            continue
        source = Path("scripts") / filename
        destination = _route_directory(key, "cli") / filename
        _add_move(root, tracked, moves, covered, source, destination)

    for filename, (key, destination_name) in TEST_ROUTES.items():
        if key not in mapping:
            continue
        source = Path("tests") / filename
        destination = _route_directory(key, "tests") / destination_name
        _add_move(root, tracked, moves, covered, source, destination)

    unmapped = sorted(item for item in tracked_list if item not in covered)
    return sorted(moves), unmapped


def _module_name(relative: Path) -> str | None:
    if relative.suffix != ".py" or not relative.parts or relative.parts[0] != "src":
        return None
    parts = list(relative.with_suffix("").parts[1:])
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_package(relative: Path) -> str:
    if relative.suffix == ".py" and relative.parts[0] == "src":
        parts = list(relative.with_suffix("").parts[1:])
        if parts[-1] != "__init__":
            parts.pop()
        else:
            parts.pop()
        return ".".join(parts)
    return ".".join(relative.parent.parts)


def _mapped_module(name: str, module_map: dict[str, str]) -> str:
    for old in sorted(module_map, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return module_map[old] + name[len(old):]
    return name


def _absolute_from_relative(module: str | None, level: int, package_name: str) -> str:
    package = package_name.split(".") if package_name else []
    keep = len(package) - (level - 1)
    prefix = package[:keep]
    return ".".join(prefix + (module.split(".") if module else []))


def _relative_name(target: str, package_name: str) -> str:
    package = package_name.split(".") if package_name else []
    target_parts = target.split(".")
    common = 0
    while common < min(len(package), len(target_parts)) and package[common] == target_parts[common]:
        common += 1
    return "." * (len(package) - common + 1) + ".".join(target_parts[common:])


def _position(lines: list[str], lineno: int, column: int) -> int:
    # AST columns are UTF-8 byte offsets; convert the offset on this line back to
    # a Python character offset before slicing the original source text.
    prefix = lines[lineno - 1].encode("utf-8")[:column].decode("utf-8")
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix)


def _rewrite_text(text: str, old_package: str, new_package: str, module_map: dict[str, str]) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mapped = _mapped_module(alias.name, module_map)
                if mapped != alias.name:
                    start = _position(lines, alias.lineno, alias.col_offset)
                    replacements.append((start, start + len(alias.name), mapped))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                absolute = _absolute_from_relative(node.module, node.level, old_package)
            else:
                absolute = node.module or ""
            mapped = _mapped_module(absolute, module_map)
            alias_targets: list[tuple[ast.alias, str]] = []
            if mapped == absolute:
                for alias in node.names:
                    candidate = f"{absolute}.{alias.name}" if absolute else alias.name
                    target = module_map.get(candidate)
                    if target is not None:
                        alias_targets.append((alias, target))
                if alias_targets:
                    if len(alias_targets) != len(node.names):
                        raise SystemExit(
                            f"refusing: cannot safely split mixed package import at line {node.lineno}"
                        )
                    parents = {target.rpartition(".")[0] for _, target in alias_targets}
                    if len(parents) != 1 or "" in parents:
                        raise SystemExit(
                            f"refusing: package import has multiple destinations at line {node.lineno}"
                        )
                    mapped = parents.pop()
            replacement = _relative_name(mapped, new_package) if node.level else mapped
            original_token = "." * node.level + (node.module or "")
            if replacement == original_token and not alias_targets:
                continue
            start = _position(lines, node.lineno, node.col_offset)
            end = _position(lines, node.end_lineno, node.end_col_offset)
            segment = text[start:end]
            match = re.match(r"(from\s+)(\.*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|\.+)(\s+import\b)", segment)
            if not match:
                raise SystemExit(f"refusing: cannot safely rewrite import at line {node.lineno}")
            module_start = start + match.start(2)
            if replacement != original_token:
                replacements.append((module_start, start + match.end(2), replacement))
            for alias, target in alias_targets:
                new_name = target.rpartition(".")[2]
                if new_name == alias.name:
                    continue
                alias_start = _position(lines, alias.lineno, alias.col_offset)
                alias_replacement = new_name if alias.asname else f"{new_name} as {alias.name}"
                replacements.append(
                    (alias_start, alias_start + len(alias.name), alias_replacement)
                )
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _preflight(root: Path, moves: list[tuple[Path, Path]]) -> None:
    source_list = [source for source, _ in moves]
    sources = set(source_list)
    destinations = [destination for _, destination in moves]
    if len(source_list) != len(sources):
        raise SystemExit("refusing: a source has multiple destinations")
    if len(destinations) != len(set(destinations)):
        raise SystemExit("refusing: destination collision in move plan")
    for destination in destinations:
        if not _inside(root, destination):
            raise SystemExit(f"refusing: destination escapes repository: {destination}")
        if destination.exists() and destination not in sources:
            raise SystemExit(f"refusing: destination collision: {destination.relative_to(root)}")


def _render_pathmap(pairs: list[tuple[Path, Path]]) -> str:
    rows = ["# Path map", "", "| Old path | New path |", "|---|---|"]
    rows += [f"| `{old}` | `{new}` |" for old, new in pairs]
    return "\n".join(rows) + "\n"


def _pathmap_pairs(root: Path) -> list[tuple[Path, Path]] | None:
    path = root / "docs/architecture/PATH-MAP.md"
    if not path.is_file():
        return None
    pairs: list[tuple[Path, Path]] = []
    pattern = re.compile(r"\| `([^`]+)` \| `([^`]+)` \|")
    for line in path.read_text().splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        old, new = Path(match.group(1)), Path(match.group(2))
        if old.is_absolute() or new.is_absolute() or not _inside(root, root / old) or not _inside(root, root / new):
            return None
        pairs.append((old, new))
    if not pairs or path.read_text() != _render_pathmap(pairs):
        return None
    return pairs


def _nul_paths(output: bytes) -> set[Path]:
    return {Path(item.decode()) for item in output.split(b"\0") if item}


def _cached_matches_moves(root: Path, pairs: list[tuple[Path, Path]]) -> bool:
    output = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--cached", "--name-status", "-z", "--find-renames=100%"]
    )
    fields = [item.decode() for item in output.split(b"\0") if item]
    old_expected = {old for old, _ in pairs}
    new_expected = {new for _, new in pairs}
    old_seen: set[Path] = set()
    new_seen: set[Path] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith("R") and index + 1 < len(fields):
            pair = (Path(fields[index]), Path(fields[index + 1]))
            index += 2
            # Git infers rename pairing from content. Multiple identical (especially
            # empty) files may be paired differently from the original git-mv calls,
            # so validate the exact old/new sets recorded by this tool's path map.
            if pair[0] not in old_expected or pair[1] not in new_expected:
                return False
            old_seen.add(pair[0])
            new_seen.add(pair[1])
        elif status == "D" and index < len(fields):
            path = Path(fields[index])
            index += 1
            if path not in old_expected:
                return False
            old_seen.add(path)
        elif status == "A" and index < len(fields):
            path = Path(fields[index])
            index += 1
            if path not in new_expected:
                return False
            new_seen.add(path)
        else:
            return False
    return old_seen == old_expected and new_seen == new_expected


def _expected_rewrite(root: Path, path: Path, pairs: list[tuple[Path, Path]]) -> bool:
    if path.suffix != ".py" or path.parts[0] not in {"src", "cli", "scripts", "tests"}:
        return False
    reverse = {new: old for old, new in pairs}
    old_relative = reverse.get(path, path)
    module_map: dict[str, str] = {}
    for old, new in pairs:
        old_module = _module_name(old)
        new_module = _module_name(new)
        if old_module and new_module:
            module_map[old_module] = new_module
    staged = subprocess.run(
        ["git", "-C", str(root), "show", f":{path.as_posix()}"],
        capture_output=True,
    )
    if staged.returncode:
        return False
    try:
        original = staged.stdout.decode("utf-8")
        expected = _rewrite_text(
            original,
            _import_package(old_relative),
            _import_package(path),
            module_map,
        )
        return (root / path).read_text() == expected
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False


def _is_expected_replay_dirty(root: Path) -> bool:
    pairs = _pathmap_pairs(root)
    if pairs is None or not _cached_matches_moves(root, pairs):
        return False
    unstaged = _nul_paths(
        subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "-z"])
    )
    pathmap = Path("docs/architecture/PATH-MAP.md")
    if any(path != pathmap and not _expected_rewrite(root, path, pairs) for path in unstaged):
        return False
    untracked = _nul_paths(
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]
        )
    )
    return untracked <= {pathmap}


def _rewrites(root: Path, moves: list[tuple[Path, Path]]) -> dict[Path, str]:
    move_by_source = {source: destination for source, destination in moves}
    module_map: dict[str, str] = {}
    for source, destination in moves:
        old = _module_name(source.relative_to(root))
        new = _module_name(destination.relative_to(root))
        if old and new:
            module_map[old] = new
    result: dict[Path, str] = {}
    for relative in _tracked(root):
        source = root / relative
        if source.suffix != ".py" or relative.parts[0] not in {"src", "scripts", "tests"}:
            continue
        destination = move_by_source.get(source, source)
        old_relative = source.relative_to(root)
        new_relative = destination.relative_to(root)
        old_package = _import_package(old_relative)
        new_package = _import_package(new_relative)
        rewritten = _rewrite_text(source.read_text(), old_package, new_package, module_map)
        if rewritten != source.read_text():
            result[destination] = rewritten
    return result

def run(root: Path, mapping_path: Path, dry_run: bool, report_only: bool) -> int:
    root = root.resolve()
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True, capture_output=True,
    )
    if probe.returncode or probe.stdout.strip() != "true":
        raise SystemExit("refusing: root must be an existing Git worktree")
    moves, unmapped = plan(root, mapping_path)
    _preflight(root, moves)
    for old, new in moves:
        print(f"{old.relative_to(root)} -> {new.relative_to(root)}")
    if unmapped:
        print("UNMAPPED:")
        for path in unmapped:
            print(path)
    if report_only:
        return 0
    if unmapped:
        return 2
    if dry_run:
        return 0
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if status.strip() and (moves or not _is_expected_replay_dirty(root)):
        raise SystemExit("refusing: worktree is dirty")
    if not moves:
        return 0
    rewrites = _rewrites(root, moves)
    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(root), "mv", str(old), str(new)], check=True)
    for path, text in rewrites.items():
        path.write_text(text)
    pathmap = root / "docs/architecture/PATH-MAP.md"
    pathmap.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(old.relative_to(root), new.relative_to(root)) for old, new in moves]
    pathmap.write_text(_render_pathmap(pairs))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    return run(args.root, args.mapping.resolve(), args.dry_run, args.report_only)

if __name__ == "__main__":
    raise SystemExit(main())
