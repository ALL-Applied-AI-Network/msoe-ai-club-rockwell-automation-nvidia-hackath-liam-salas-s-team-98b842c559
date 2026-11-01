#!/usr/bin/env python3
"""Plan and replay the approved onion-layout move without guessing names."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import textwrap
from pathlib import Path

DEFAULT_MAPPING = Path(__file__).resolve().parents[2].parent / "overnight-ops-20261028/ONION-MAPPING.json"

# Responsibility names are explicit: ticket/version prefixes are never stripped by a rule.
RENAMES = {
    "wp7_evaluation_only_preparation.py": "evaluation_preparation.py",
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
    "empty_reference_timing_adapter.py": "timing_adapter.py",
    "wp7_cohort_preparation.py": "cohort_preparation.py",
    "wp7_fill_step_calibration.py": "fill_step_calibration.py",
    "wp7_fill_step_production.py": "fill_step_production.py",
    "wp7_fill_step_tracker.py": "fill_step_tracker.py",
    "wp7_phase_capture.py": "phase_capture.py",
    "wp7_stock_runtime.py": "stock_runtime.py",
    "wp7_tracker_receipt_adapter.py": "tracker_receipt_adapter.py",
    "wp7_tracker_runtime.py": "tracker_runtime.py",
    "wp7_vlm_constrained.py": "constrained.py",
    "wp7_vlm_decision.py": "vlm.py",
    "wp7_vlm_development_materializer.py": "development_materializer.py",
    "wp7_vlm_image_support.py": "image_support.py",
    "wp7_vlm_native_roi_rematerializer.py": "native_roi_rematerializer.py",
    "wp7_vlm_production_materializer.py": "production_materializer.py",
    "pipeline_package.py": "__init__.py",
    "pipeline_service_package.py": "__init__.py",
}

# Source responsibilities that must be selected even when the authoritative mapping
# has not yet listed the source module by name.
SOURCE_ROUTES = {
    "wp7_evaluation_only_preparation.py": "pipeline",
    "empty_reference_timing_adapter.py": "perception/baselines",
    "pipeline_package.py": "pipeline",
    "pipeline_service_package.py": "pipeline/service",
    "accepted_adapter.py": "pipeline/service",
    "contract.py": "pipeline/service",
    "device.py": "pipeline/service",
    "http.py": "pipeline/service",
    "manager.py": "pipeline/service",
    "models.py": "pipeline/service",
    "production_factory.py": "pipeline/service",
    "sources.py": "pipeline/service",
    "wp7_cohort_preparation.py": "pipeline",
    "wp7_fill_step_calibration.py": "measurement",
    "wp7_fill_step_production.py": "perception/tracker",
    "wp7_fill_step_tracker.py": "perception/tracker",
    "wp7_phase_capture.py": "pipeline",
    "wp7_stock_runtime.py": "explanation/vlm",
    "wp7_tracker_receipt_adapter.py": "pipeline",
    "wp7_tracker_runtime.py": "perception/tracker",
    "wp7_vlm_constrained.py": "explanation/vlm",
    "wp7_vlm_decision.py": "decision",
    "wp7_vlm_development_materializer.py": "explanation/vlm",
    "wp7_vlm_image_support.py": "explanation/vlm",
    "wp7_vlm_native_roi_rematerializer.py": "explanation/vlm",
    "wp7_vlm_production_materializer.py": "explanation/vlm",
}

# These mapping keys name leaf responsibilities, not directories. Their members live in
# the parent directory and RENAMES selects the responsibility filename where necessary.
LEAF_KEYS = {"core/policy", "perception/masks"}

# Scripts do not reliably share a name with their implementation module. Every routed
# script is therefore listed. Unlisted tracked scripts are deliberately reported unmapped.
SCRIPT_ROUTES = {
    "aggregate_wp7_vlm_attempts.py": "decision",
    "check_agent_docs.py": "core/policy",
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
    "materialize_wp7_vlm_development_bundle.py": "explanation/vlm",
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
    "prepare_wp7_cohort.py": "pipeline",
    "prepare_wp7_development_history.py": "memory/history",
    "prepare_wp7_development_reference_route.py": "pipeline",
    "prepare_wp7_evaluation_only.py": "pipeline",
    "preview_tank_station.py": "sim/scene",
    "produce_inn67_postrender_inputs.py": "memory/reference_bank",
    "publish_empty_reference_development.py": "perception/baselines",
    "publish_f2_stage_batch.py": "sim/cohorts",
    "publish_f2_varied_stage_batch.py": "sim/cohorts",
    "publish_wp7_fill_step_calibration.py": "measurement",
    "real_check_windows.py": "core/policy",
    "render_capseal_structural_pilot.py": "sim/scene",
    "render_rigid_vial_transport.py": "sim/transport",
    "render_tank_station.py": "sim/scene",
    "render_transport_stage.py": "sim/transport",
    "render_twin_dataset.py": "sim/render",
    "rematerialize_wp7_vlm_native_roi.py": "explanation/vlm",
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
    "run_wp7_vlm_model.py": "explanation/vlm",
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
    "wp7_vlm_decision.py": "decision",
}

SCRIPT_RENAMES = {
    "prepare_wp7_evaluation_only.py": "prepare_evaluation_only.py",
}

# Tests without a one-to-one mapped module name are routed explicitly. Destination
# names describe the responsibility and remove ticket prefixes where the folder does.
TEST_ROUTES = {
    "test_production_factory.py": (
        "pipeline", "service/test_production_factory.py"
    ),
    "test_runner_contract.py": (
        "pipeline", "service/test_runner_contract.py"
    ),
    "test_runner_lifecycle.py": (
        "pipeline", "service/test_runner_lifecycle.py"
    ),
    "test_runner_service.py": (
        "pipeline", "service/test_runner_service.py"
    ),
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
    # ``test_materializer.py`` is already used by the reference-bank route;
    # retain a responsibility suffix for the perception materializer so pytest
    # can collect both modules without a top-level-name collision.
    "test_predicted_mask_materializer.py": (
        "perception/masks", "test_mask_materializer.py"
    ),
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
    "test_accepted_adapter_tracker_observer.py": (
        "pipeline", "service/test_accepted_adapter_tracker_observer.py"
    ),
    "test_runner_vlm_lineage_integration.py": (
        "pipeline", "service/test_runner_vlm_lineage_integration.py"
    ),
    "test_wp7_cohort_preparation.py": ("pipeline", "test_cohort_preparation.py"),
    "test_wp7_fill_step_calibration.py": (
        "measurement", "test_fill_step_calibration.py"
    ),
    "test_wp7_fill_step_production.py": (
        "perception/masks", "tracker/test_fill_step_production.py"
    ),
    "test_wp7_fill_step_tracker.py": (
        "perception/masks", "tracker/test_fill_step_tracker.py"
    ),
    "test_wp7_option3_cli.py": ("pipeline", "test_development_reference_route.py"),
    "test_wp7_phase_capture.py": ("pipeline", "test_phase_capture.py"),
    "test_wp7_second_vial_reference.py": (
        "pipeline", "test_second_vial_reference.py"
    ),
    "test_wp7_stock_runtime.py": ("explanation/vlm", "test_stock_runtime.py"),
    "test_wp7_tracker_receipt_adapter.py": (
        "pipeline", "test_tracker_receipt_adapter.py"
    ),
    "test_wp7_tracker_runtime_route.py": (
        "perception/masks", "tracker/test_tracker_runtime_route.py"
    ),
    "test_wp7_vlm_decision.py": ("decision", "test_vlm.py"),
    "test_wp7_vlm_development_materializer.py": (
        "explanation/vlm", "test_development_materializer.py"
    ),
    "test_wp7_vlm_native_roi_rematerializer.py": (
        "explanation/vlm", "test_native_roi_rematerializer.py"
    ),
    "test_wp7_vlm_production_materializer.py": (
        "explanation/vlm", "test_production_materializer.py"
    ),
    "test_wp7_vlm_reasoning_upgrade.py": (
        "explanation/vlm", "test_reasoning_upgrade.py"
    ),
}

# A few boundary-audit files encode package roots and entry-point paths as data or
# discovery code.  These are deliberate, path-keyed migrations rather than another
# heuristic rewrite rule: a changed upstream file must fail closed and be reviewed.
OLD_CNN_SOURCE_SHA256 = "d2d84b73cfeebac3a173df3b18adf8347759ac75aa03ec965a4870a28cbdb387"
CNN_SOURCE_SHA256 = "ec9f4b589f088fe979f084ebff1ad51cb0fe0dd29cf12a64a7c993bab534be19"
OLD_PLAN_SHA256 = "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79"
PLAN_SHA256 = "e015a0d02aef6fce283f85a51ea4ab3ff9bde2fb01293a5a3d2b42ec08ad765b"
OLD_PLAN_MARKDOWN_SHA256 = "7bcaa8d7a59cc23b5fb9ab930e7bbcc8ba1dfa65af5d4762709593215e99e06f"
PLAN_MARKDOWN_SHA256 = "27dba1b66f680c4444f0c6e6e3505291ff9e7af586a91d4747f1b09a55876a0e"
OLD_PLAN_INVENTORY_SHA256 = "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95"
PLAN_INVENTORY_SHA256 = "f0ab668df8bb2f4abe8708a486ee00a913e129772a83a4d1abd2b95d28d467d7"
OLD_AMENDMENT_FILE_SHA256 = "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8"
AMENDMENT_FILE_SHA256 = "5a39a1c8de636d0d0de83440a8483c2ec6a692b1722c1a77ac6ec613bee93128"
OLD_MATERIALIZATION_AMENDMENT_SHA256 = "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c"
MATERIALIZATION_AMENDMENT_SHA256 = "2f39a4507d429902e241fff0b5709d7c646a702e8d8331a54c9c941f62ccb4bd"
OLD_OCCLUSION_VALIDATOR_SHA256 = "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315"
OCCLUSION_VALIDATOR_SHA256 = "16b8c53fc00373af9fe94fa0305b4482588a1d77a9b6fab513c9ab6e0aa72ed7"
OLD_THREE_POUR_SHA256 = "c8be7ec97131d2e9de3cb1656ffd1f23fbae80b03fc27864158db929c29f8c24"
THREE_POUR_SHA256 = "deb0d7bb98ec6c45151ce6b0a6c58342a58c5903c0c09163fd33ddec9f723edb"


def _frozen_identity_rules(
    *, cnn: int = 0, plan: int = 0, markdown: int = 0, inventory: int = 0,
    amendment: int = 0, materialization_amendment: int = 0, validator: int = 0,
    renderer: int = 0,
) -> tuple[tuple[str, str, int], ...]:
    requested = (
        (OLD_CNN_SOURCE_SHA256, CNN_SOURCE_SHA256, cnn),
        (OLD_PLAN_SHA256, PLAN_SHA256, plan),
        (OLD_PLAN_MARKDOWN_SHA256, PLAN_MARKDOWN_SHA256, markdown),
        (OLD_PLAN_INVENTORY_SHA256, PLAN_INVENTORY_SHA256, inventory),
        (OLD_AMENDMENT_FILE_SHA256, AMENDMENT_FILE_SHA256, amendment),
        (
            OLD_MATERIALIZATION_AMENDMENT_SHA256,
            MATERIALIZATION_AMENDMENT_SHA256,
            materialization_amendment,
        ),
        (OLD_OCCLUSION_VALIDATOR_SHA256, OCCLUSION_VALIDATOR_SHA256, validator),
        (OLD_THREE_POUR_SHA256, THREE_POUR_SHA256, renderer),
    )
    return tuple(rule for rule in requested if rule[2])


SEMANTIC_REWRITES = {
    "scripts/validate_physx_f2_artifacts.py": (
        (
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT / "src"))',
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            'sys.path.insert(0, str(ROOT / "src"))',
            1,
        ),
    ),
    "scripts/run_physx_train_residual_cohort.py": _frozen_identity_rules(renderer=1),
    "scripts/render_capseal_structural_pilot.py": (
        (
            'STRUCTURAL_MODULE = ROOT / "src" / "fluidmix" / "capseal_structural.py"',
            'STRUCTURAL_MODULE = ROOT / "src" / "twin" / "scene" / "capseal_structural.py"',
            1,
        ),
    ),
    "src/fluidmix/inn67_predicted_store_producer.py": (
        (
            'predictor_source = Path(__file__).with_name("cnn_predictor.py")',
            'predictor_source = (\n'
            '        Path(__file__).resolve().parents[4]\n'
            '        / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"\n'
            '    )',
            1,
        ),
        (
            'empty_source = Path(__file__).with_name("empty_reference_rgb_predictor.py")',
            'empty_source = (\n'
            '        Path(__file__).resolve().parents[4]\n'
            '        / "src" / "fluidmix" / "perception" / "baselines"\n'
            '        / "empty_reference_rgb_predictor.py"\n'
            '    )',
            1,
        ),
    ) + _frozen_identity_rules(cnn=1),
    "src/fluidmix/predicted_mask_materializer.py": (
        (
            textwrap.dedent(
                '''\
                predictor_source_path = Path(__file__).with_name(
                    "cnn_predictor.py" if use_cnn else "empty_reference_rgb_predictor.py"
                )'''
            ),
            textwrap.dedent(
                '''\
                predictor_source_path = (
                    _REPOSITORY_ROOT / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"
                    if use_cnn
                    else _REPOSITORY_ROOT / "src" / "fluidmix" / "perception" / "baselines"
                    / "empty_reference_rgb_predictor.py"
                )'''
            ),
            1,
        ),
    ) + _frozen_identity_rules(cnn=1),
    "src/fluidmix/sixth_development_run.py": (
        (
            textwrap.dedent(
                '''\
                predictor_path = Path(__file__).resolve(strict=True).with_name(
                    "cnn_predictor.py"
                )'''
            ),
            textwrap.dedent(
                '''\
                predictor_path = (
                    Path(__file__).resolve().parents[4]
                    / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"
                )'''
            ),
            1,
        ),
    ) + _frozen_identity_rules(cnn=1),
    "src/fluidmix/inn67_bank_materializer.py": _frozen_identity_rules(
        plan=1, inventory=1, amendment=1, materialization_amendment=1, validator=1
    ),
    "src/fluidmix/wp7_development_reference.py": _frozen_identity_rules(
        plan=1, inventory=1, amendment=1, materialization_amendment=1
    ),
    "src/fluidmix/wp7_history_preparation.py": _frozen_identity_rules(
        plan=1, inventory=1, amendment=1, materialization_amendment=1
    ),
    "src/fluidmix/carryover_pipeline.py": _frozen_identity_rules(
        plan=2, inventory=2, amendment=2, materialization_amendment=2
    ),
    "src/fluidmix/wp1_postrender.py": _frozen_identity_rules(validator=1, renderer=1),
    "src/fluidmix/transport_stage.py": _frozen_identity_rules(validator=1, renderer=1),
    "tests/test_inn67_bank_materializer.py": _frozen_identity_rules(
        plan=1, markdown=1, inventory=2, amendment=1,
        materialization_amendment=2,
    ),
    "tests/test_inn67_postrender_producer.py": _frozen_identity_rules(
        markdown=1, inventory=1
    ) + (
        (
            "fluidmix.inn67_postrender_producer",
            "fluidmix.memory.reference_bank.postrender_producer",
            6,
        ),
    ),
    "tests/test_inn67_predicted_store_producer.py": (
        (
            "fluidmix.inn67_postrender_producer",
            "fluidmix.memory.reference_bank.postrender_producer",
            2,
        ),
    ),
    "tests/test_wp7_history_preparation.py": _frozen_identity_rules(
        plan=1, inventory=1, amendment=1, materialization_amendment=1
    ),
    "tests/test_wp7_route_preflight.py": _frozen_identity_rules(
        plan=1, inventory=1, amendment=1, materialization_amendment=1
    ),
    "tests/test_wp7_fill_step_production.py": (
        (
            "from fluidmix import empty_reference_timing_adapter, "
            "wp7_fill_step_production",
            "from fluidmix.perception.baselines import timing_adapter as "
            "empty_reference_timing_adapter\n"
            "    from fluidmix.perception.tracker import fill_step_production as "
            "wp7_fill_step_production",
            1,
        ),
    ),
    "src/fluidmix/wp7_vlm_decision.py": (
        (
            'Path(__file__).with_name("wp7_fill_step_tracker.py")',
            '(Path(__file__).resolve().parents[1] / "perception" / "tracker" '
            '/ "fill_step_tracker.py")',
            4,
        ),
    ),
    "src/fluidmix/wp7_tracker_runtime.py": (
        (
            'Path(__file__).with_name("wp7_fill_step_production.py")',
            'Path(__file__).with_name("fill_step_production.py")',
            1,
        ),
        (
            'Path(__file__).with_name("wp7_fill_step_tracker.py")',
            'Path(__file__).with_name("fill_step_tracker.py")',
            1,
        ),
        (
            'Path(__file__).with_name("wp7_vlm_production_materializer.py")',
            '(Path(__file__).resolve().parent.parent.parent / "explanation" / "vlm" '
            '/ "production_materializer.py")',
            1,
        ),
    ),
    "src/fluidmix/wp7_vlm_native_roi_rematerializer.py": (
        (
            '_ROOT / "src" / "fluidmix" / "wp7_fill_step_tracker.py"',
            '_ROOT / "src" / "fluidmix" / "perception" / "tracker" '
            '/ "fill_step_tracker.py"',
            1,
        ),
        (
            '_ROOT / "src" / "fluidmix" / "wp7_vlm_decision.py"',
            '_ROOT / "src" / "fluidmix" / "decision" / "vlm.py"',
            1,
        ),
    ),
    "scripts/materialize_wp7_vlm_development_bundle.py": (
        (
            'ROOT / "src" / "fluidmix" / "wp7_vlm_development_materializer.py"',
            'ROOT / "src" / "fluidmix" / "explanation" / "vlm" '
            '/ "development_materializer.py"',
            1,
        ),
        (
            'ROOT / "src" / "fluidmix" / "wp7_vlm_decision.py"',
            'ROOT / "src" / "fluidmix" / "decision" / "vlm.py"',
            1,
        ),
    ),
    "tests/test_wp7_vlm_decision.py": (
        (
            'Path(wp7.__file__).with_name("wp7_fill_step_tracker.py")',
            '(Path(wp7.__file__).resolve().parents[1] / "perception" / "tracker" '
            '/ "fill_step_tracker.py")',
            1,
        ),
    ),
    "tests/test_wp1_postrender.py": _frozen_identity_rules(validator=1, renderer=2),
    "tests/test_transport_stage.py": _frozen_identity_rules(validator=2, renderer=2) + (
        (
            'monkeypatch.setitem(sys.modules, "physx_three_pour_sequence", f2)',
            'monkeypatch.setitem(sys.modules, "cli.twin.physx.physx_three_pour_sequence", f2)\n'
            '    monkeypatch.setattr(__import__("cli.twin.physx", fromlist=["physx"]), '
            '"physx_three_pour_sequence", f2, raising=False)',
            1,
        ),
    ),
    "tests/test_physx_train_residual_supervisor.py": _frozen_identity_rules(renderer=1),
    "tests/test_splits.py": (
        (
            "import check_leakage",
            "import cli.fluidmix.core.check_leakage as check_leakage",
            1,
        ),
        (
            "import make_splits",
            "import cli.twin.cohorts.make_splits as make_splits",
            1,
        ),
    ),
    "tests/test_physx_one_run_manifest_v1.py": (
        (
            "fluidmix.physx_run_variation.load_tank_colour_preset_catalog",
            "twin.physx.run_variation.load_tank_colour_preset_catalog",
            1,
        ),
    ),
    "tests/test_physx_varied_cohort_supervisor.py": (
        (
            "fluidmix.physx_run_variation.load_tank_colour_preset_catalog",
            "twin.physx.run_variation.load_tank_colour_preset_catalog",
            2,
        ),
        (
            "fluidmix.tank_colour_presets.load_tank_colour_preset_catalog",
            "twin.scene.tank_colour_presets.load_tank_colour_preset_catalog",
            1,
        ),
        (
            "5bf92fea5a7e941081c9b56a4e937fd2cdeaa9ecc6ba77d125dff1b257076de0",
            "511d3a9012935b2d3ed86a56f57bd434989620295f8061945d621ce05b8c3e63",
            1,
        ),
        (
            "0608ab047c2e0cf733649a9b9c0224b6412f7654613dea84af014e3a347d637b",
            "dcf2f1f22ce60b0e7d2b327bd1f3650185deac2b1a7c4dbcd6e9dddbfc6a6cfc",
            1,
        ),
        (
            "a86bce8f4e80adc985f6d5501cf3f941c84a962d114d04012949b905ff9ed572",
            "a7a97c5d43efc647c4f22d787d5fc0c3ada6c1d8e8ac195b7b25004725e5a92b",
            1,
        ),
    ),
    "tests/test_physx_run_variation.py": (
        (
            "fluidmix.physx_run_variation.load_tank_colour_preset_catalog",
            "twin.physx.run_variation.load_tank_colour_preset_catalog",
            1,
        ),
    ),
    "tests/test_test_recipe_generator.py": (
        (
            "import make_splits  # noqa: E402",
            "import cli.twin.cohorts.make_splits as make_splits  # noqa: E402",
            1,
        ),
    ),
    "tests/fixtures/inn67-materialization-protocol-amendment.json": (
        _frozen_identity_rules(plan=1, inventory=1, amendment=1, validator=1)
    ),
    "scripts/physx_three_pour_sequence.py": (
        (
            'contract_path = ROOT / "src" / "fluidmix" / "physx_three_pour.py"',
            'contract_path = ROOT / "src" / "twin" / "physx" / "three_pour.py"',
            1,
        ),
        (
            'contract_path = ROOT / "src" / "fluidmix" / "physx_particle_contract.py"',
            'contract_path = ROOT / "src" / "twin" / "physx" / "particle_contract.py"',
            1,
        ),
    ),
    "tests/test_review_regressions.py": (
        (
            'ROOT / "scripts" / name',
            'ROOT / "cli/fluidmix/core" / name',
            1,
        ),
        (
            '(tmp_path / "scripts").mkdir()',
            '(tmp_path / "cli/fluidmix/core").mkdir(parents=True)',
            1,
        ),
    ),
    "tests/test_sixth_development_run.py": (
        (
            'str(ROOT / "tests" / ".." / "src" / "fluidmix" / "sixth_development_run.py")',
            'str(ROOT / "src" / "fluidmix" / "memory" / "history" / "sixth_development_run.py")',
            1,
        ),
    ),
    # These tests and launchers keep executable paths as data.  Keep the
    # migration keyed to the pre-move file so a changed helper fails closed
    # instead of silently dispatching an arbitrary script.
    "tests/test_contract_schemas.py": (
        (
            'SCHEMA_ROOT = ROOT / "src" / "fluidmix" / "contracts"',
            'SCHEMA_ROOT = ROOT / "src" / "fluidmix" / "core" / "contracts"',
            1,
        ),
    ),
    "tests/test_cosmos_nano_inputs_runner.py": (
        (
            '[sys.executable, str(ROOT / "scripts" / script), "--help"]',
            '[sys.executable, str({\n'
            '            "stage_cosmos_nano_reasoner.py": ROOT / "cli/fluidmix/explanation/vlm/stage_cosmos_nano_reasoner.py",\n'
            '            "audit_cosmos_nano_inputs.py": ROOT / "cli/fluidmix/explanation/vlm/audit_cosmos_nano_inputs.py",\n'
            '            "run_cosmos_large_eval.py": ROOT / "cli/fluidmix/explanation/vlm/run_cosmos_large_eval.py",\n'
            '            "score_cosmos_nano_matched.py": ROOT / "cli/evaluation/scoring/score_cosmos_nano_matched.py",\n'
            '        }[script]), "--help"]',
            1,
        ),
    ),
    "tests/test_cosmos_usable_preflight.py": (
        (
            '[sys.executable, str(ROOT / "scripts" / script), "--help"]',
            '[sys.executable, str({\n'
            '            "audit_cosmos_nano_usable_inputs.py": ROOT / "cli/fluidmix/explanation/vlm/audit_cosmos_nano_usable_inputs.py",\n'
            '            "run_cosmos_nano_usable_eval.py": ROOT / "cli/fluidmix/explanation/vlm/run_cosmos_nano_usable_eval.py",\n'
            '            "score_cosmos_nano_usable.py": ROOT / "cli/evaluation/scoring/score_cosmos_nano_usable.py",\n'
            '        }[script]), "--help"]',
            1,
        ),
    ),
    "tests/test_rigid_vial_transport.py": (
        (
            'path = REPO_ROOT / "scripts" / f"{name}.py"',
            'path = {\n'
            '        "render_rigid_vial_transport": REPO_ROOT / "cli/twin/transport/render_rigid_vial_transport.py",\n'
            '        "check_rigid_vial_transport": REPO_ROOT / "cli/twin/transport/check_rigid_vial_transport.py",\n'
            '    }[name]',
            1,
        ),
    ),
    "tests/test_physx_palette_smoke_remote.py": (
        (
            '(source / "scripts").mkdir(parents=True)',
            '(source / "cli/twin/probes").mkdir(parents=True)',
            1,
        ),
        (
            '(source / "scripts" / "physx_particle_palette_smoke.py").write_text',
            '(source / "cli/twin/probes" / "physx_particle_palette_smoke.py").write_text',
            1,
        ),
    ),
    "tests/test_physx_f2_remote_launcher.py": (
        (
            'scripts = source / "scripts"',
            'scripts = source / "cli/twin/physx"',
            1,
        ),
    ),
    "scripts/run_physx_f2_remote.sh": (
        (
            'runner="$source_root/scripts/physx_three_pour_sequence.py"',
            'runner="$source_root/cli/twin/physx/physx_three_pour_sequence.py"',
            1,
        ),
        (
            'test -f "$source_root/scripts/physx_three_pour_sequence.py"',
            'test -f "$source_root/cli/twin/physx/physx_three_pour_sequence.py"',
            1,
        ),
    ),
    "scripts/run_physx_palette_smoke_remote.sh": (
        (
            'runner="$source_root/scripts/physx_particle_palette_smoke.py"',
            'runner="$source_root/cli/twin/probes/physx_particle_palette_smoke.py"',
            1,
        ),
        (
            'test -f "$source_root/scripts/physx_particle_palette_smoke.py"',
            'test -f "$source_root/cli/twin/probes/physx_particle_palette_smoke.py"',
            1,
        ),
    ),
    "src/fluidmix/part_boundary_manifest.py": (
        (
            'return name if name == "fluidmix" or name.startswith("fluidmix.") else None',
            'return name if name.split(".", 1)[0] in {"fluidmix", "twin", "evaluation"} else None',
            1,
        ),
        (
            'local != "fluidmix"',
            'local not in {"fluidmix", "twin", "evaluation"}',
            3,
        ),
        (
            'target == "fluidmix" or package_member_import',
            'target.split(".", 1)[0] in {"fluidmix", "twin", "evaluation"} or package_member_import',
            1,
        ),
        ('if name.startswith("scripts/"):', 'if name.startswith("cli/"):', 1),
        (
            'WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.part_b_evidence"',
            'WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.storage.artifacts.part_b_evidence"',
            1,
        ),
        (
            textwrap.dedent(
                '''\
                current_modules = {
                    "fluidmix."
                    + path.relative_to(root / "src" / "fluidmix")
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                    for path in (root / "src" / "fluidmix").rglob("*.py")
                    if path.name != "__init__.py"
                }'''
            ),
            textwrap.dedent(
                '''\
                current_modules = {
                    f"{package}."
                    + path.relative_to(root / "src" / package)
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                    for package in ("fluidmix", "twin", "evaluation")
                    for path in (root / "src" / package).rglob("*.py")
                    if path.name != "__init__.py"
                }'''
            ),
            1,
        ),
    ),
    "tests/test_part_boundary.py": (
        (
            textwrap.dedent(
                '''\
                current_modules = {
                    "fluidmix."
                    + path.relative_to(ROOT / "src" / "fluidmix")
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                    for path in (ROOT / "src" / "fluidmix").rglob("*.py")
                    if path.name != "__init__.py"
                }'''
            ),
            textwrap.dedent(
                '''\
                current_modules = {
                    f"{package}."
                    + path.relative_to(ROOT / "src" / package)
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                    for package in ("fluidmix", "twin", "evaluation")
                    for path in (ROOT / "src" / package).rglob("*.py")
                    if path.name != "__init__.py"
                }'''
            ),
            1,
        ),
        ('from fluidmix import vision_measurement', 'import fluidmix.vision_measurement', 1),
        ('module_name="fluidmix.contracts.example"', 'module_name="fluidmix.core.contracts.example"', 1),
    ),
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
        ["git", "-C", str(root), "ls-files", "-z", "src", "cli", "scripts", "tests"]
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
            explicit_test_route = TEST_ROUTES.get(test_source.name)
            if explicit_test_route is None:
                responsibility = Path(RENAMES.get(f"{name}.py", f"{name}.py")).stem
                test_destination = _route_directory(key, "tests") / f"test_{responsibility}.py"
            else:
                explicit_key, explicit_name = explicit_test_route
                test_destination = _route_directory(explicit_key, "tests") / explicit_name
            _add_move(root, tracked, moves, covered, test_source, test_destination)

    for filename, key in SOURCE_ROUTES.items():
        source = Path("src/fluidmix") / filename
        if source in covered:
            continue
        name = Path(filename).stem
        destination = _source_destination(key, name)
        _add_move(root, tracked, moves, covered, source, destination)

    for filename, key in SCRIPT_ROUTES.items():
        if key not in mapping:
            continue
        source = Path("scripts") / filename
        destination = _route_directory(key, "cli") / SCRIPT_RENAMES.get(filename, filename)
        _add_move(root, tracked, moves, covered, source, destination)

    for filename, (key, destination_name) in TEST_ROUTES.items():
        if key not in mapping:
            continue
        source = Path("tests") / filename
        if source in covered:
            continue
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


def _script_module_map(script_map: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for old, new in script_map.items():
        target = Path(new).with_suffix("").as_posix().replace("/", ".")
        result[Path(old).with_suffix("").as_posix().replace("/", ".")] = target
        result[Path(old).stem] = target
    return result


def _test_module_map(pairs: list[tuple[Path, Path]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for old, new in pairs:
        if not old.parts or old.parts[0] != "tests" or old.suffix != ".py":
            continue
        target = new.with_suffix("").as_posix().replace("/", ".")
        result[old.stem] = target
    return result


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


def _semantic_rewrite(relative: Path, text: str) -> str:
    rules = SEMANTIC_REWRITES.get(relative.as_posix())
    if rules is None:
        return text
    for old, new, expected_count in rules:
        candidates = [(old, new)]
        if "\n" in old:
            candidates.extend(
                (textwrap.indent(old, " " * indent), textwrap.indent(new, " " * indent))
                for indent in range(1, 17)
            )
        match = next(
            ((candidate_old, candidate_new) for candidate_old, candidate_new in candidates
             if text.count(candidate_old) == expected_count),
            None,
        )
        if match is None:
            raise SystemExit(
                "refusing: semantic migration drift for "
                f"{relative}: expected {expected_count} occurrence(s) of {old!r}, "
                f"found {text.count(old)}"
            )
        text = text.replace(*match)
    return text


def _module_string_context(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_map.get(node)
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return True
    if not isinstance(parent, ast.Call):
        return False
    function = parent.func
    if isinstance(function, ast.Name):
        name = function.id
    elif isinstance(function, ast.Attribute):
        name = function.attr
    else:
        return False
    return (
        name in {"__import__", "import_module", "find_spec", "pop", "startswith"}
        or "module" in name
        or "import" in name
        or "dependency" in name
        or "ownership" in name
        or "join" in name
    )


# A module identifier is never a filename.  Collection literals are useful
# module registries, but the same context also carries script names and asset
# paths; those must remain ordinary text for the moved command to dispatch them.
_NON_MODULE_SUFFIXES = frozenset({
    ".py", ".json", ".sh", ".html", ".htm", ".yaml", ".yml", ".toml",
    ".csv", ".txt", ".md", ".usda", ".png", ".jpg", ".jpeg", ".mp4",
    ".sqlite", ".sqlite3",
})


def _ensure_cli_repo_root(text: str, relative: Path | None) -> str:
    if relative is None or not relative.parts or relative.parts[0] != "cli":
        return text
    tree = ast.parse(text)
    cli_imports = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and isinstance(node.module, str)
            and (node.module == "cli" or node.module.startswith("cli."))
        ) or (
            isinstance(node, ast.Import)
            and any(alias.name == "cli" or alias.name.startswith("cli.") for alias in node.names)
        )
    ]
    if not cli_imports:
        return text
    first_line = min(node.lineno for node in cli_imports)
    depth = len(relative.parent.parts)
    direct = f"sys.path.insert(0, str(Path(__file__).resolve().parents[{depth}]))"
    root_assignment = f"ROOT = Path(__file__).resolve().parents[{depth}]"
    before_import = "".join(text.splitlines(keepends=True)[: first_line - 1])
    if direct in before_import or (
        root_assignment in before_import and "sys.path.insert(0, str(ROOT))" in before_import
    ):
        return text
    has_sys = any(
        isinstance(node, ast.Import) and any(alias.name == "sys" for alias in node.names)
        for node in tree.body
    )
    has_path = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pathlib"
        and any(alias.name == "Path" for alias in node.names)
        for node in tree.body
    )
    additions = []
    if not has_path:
        additions.append("from pathlib import Path")
    if not has_sys:
        additions.append("import sys")
    additions.append(direct)
    lines = text.splitlines(keepends=True)
    lines.insert(first_line - 1, "\n".join(additions) + "\n")
    return "".join(lines)


def _remove_repo_scripts_sys_path(text: str) -> str:
    """Drop obsolete repo ``scripts`` imports without touching fixture paths."""
    tree = ast.parse(text)

    def is_repo_root(expression: ast.AST) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id == "ROOT"
        # Match only Path(__file__).resolve().parents[N], the other exact
        # spelling used for the repository root in the audited test files.
        return (
            isinstance(expression, ast.Subscript)
            and isinstance(expression.value, ast.Attribute)
            and expression.value.attr == "parents"
            and isinstance(expression.value.value, ast.Call)
            and isinstance(expression.value.value.func, ast.Attribute)
            and expression.value.value.func.attr == "resolve"
            and isinstance(expression.value.value.func.value, ast.Call)
            and isinstance(expression.value.value.func.value.func, ast.Name)
            and expression.value.value.func.value.func.id == "Path"
            and len(expression.value.value.func.value.args) == 1
            and isinstance(expression.value.value.func.value.args[0], ast.Name)
            and expression.value.value.func.value.args[0].id == "__file__"
        )

    def is_scripts_wrapper(expression: ast.AST) -> bool:
        if not isinstance(expression, ast.Call) or len(expression.args) != 1:
            return False
        wrapper = expression.func
        if isinstance(wrapper, ast.Name):
            valid_wrapper = wrapper.id == "str"
        else:
            valid_wrapper = (
                isinstance(wrapper, ast.Attribute)
                and isinstance(wrapper.value, ast.Name)
                and wrapper.value.id == "os"
                and wrapper.attr == "fspath"
            )
        path = expression.args[0]
        return (
            valid_wrapper
            and isinstance(path, ast.BinOp)
            and isinstance(path.op, ast.Div)
            and is_repo_root(path.left)
            and isinstance(path.right, ast.Constant)
            and path.right.value == "scripts"
        )

    removals: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "insert"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "path"
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "sys"
            and len(call.args) == 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == 0
            and is_scripts_wrapper(call.args[1])
        ):
            continue
        removals.append((node.lineno - 1, node.end_lineno or node.lineno))
    for start, end in sorted(removals, reverse=True):
        del lines[start:end]
    return "".join(lines)


def _rewrite_text(
    text: str,
    old_package: str,
    new_package: str,
    module_map: dict[str, str],
    *,
    old_relative: Path | None = None,
    new_relative: Path | None = None,
    script_map: dict[str, str] | None = None,
    path_map: dict[str, str] | None = None,
) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    parent_map = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
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
                    package_parents = {target.rpartition(".")[0] for _, target in alias_targets}
                    if len(package_parents) != 1 or "" in package_parents:
                        raise SystemExit(
                            f"refusing: package import has multiple destinations at line {node.lineno}"
                        )
                    mapped = package_parents.pop()
            # A relative import can only reach a target within the same top-level
            # package.  Moving a simulator/evaluation module out of fluidmix must
            # therefore produce an absolute import, even when the source was
            # relative before the move.
            same_top_level = (
                bool(mapped)
                and bool(new_package)
                and mapped.split(".", 1)[0] == new_package.split(".", 1)[0]
            )
            replacement = (
                _relative_name(mapped, new_package)
                if node.level and same_top_level
                else mapped
            )
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
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "\n" in node.value and re.search(
                r"(?m)^\s*(?:from|import)\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|"
                r"__import__\s*\(",
                node.value,
            ):
                try:
                    embedded = _rewrite_text(
                        node.value,
                        old_package,
                        new_package,
                        module_map,
                        script_map=script_map,
                        path_map=path_map,
                    )
                except SyntaxError:
                    dedented = textwrap.dedent(node.value)
                    try:
                        embedded_dedented = _rewrite_text(
                            dedented,
                            old_package,
                            new_package,
                            module_map,
                            script_map=script_map,
                            path_map=path_map,
                        )
                    except SyntaxError:
                        embedded = node.value
                    else:
                        embedded = textwrap.indent(
                            embedded_dedented,
                            " " * (len(node.value) - len(node.value.lstrip(" "))),
                        )
                if embedded != node.value:
                    start = _position(lines, node.lineno, node.col_offset)
                    end = _position(lines, node.end_lineno, node.end_col_offset)
                    segment = text[start:end]
                    replacements.append((start, end, segment.replace(node.value, embedded, 1)))
                    continue
            mapped = _mapped_module(node.value, module_map)
            if Path(node.value).suffix.lower() in _NON_MODULE_SUFFIXES:
                continue
            parent = parent_map.get(node)
            dynamic_import = False
            if isinstance(parent, ast.Call):
                function = parent.func
                if isinstance(function, ast.Name):
                    dynamic_import = function.id in {"__import__", "import_module"}
                elif isinstance(function, ast.Attribute):
                    dynamic_import = function.attr in {"import_module", "find_spec"}
            if (
                mapped == node.value
                or "." not in node.value
                or (not dynamic_import and not _module_string_context(node, parent_map))
            ):
                continue
            start = _position(lines, node.lineno, node.col_offset)
            end = _position(lines, node.end_lineno, node.end_col_offset)
            segment = text[start:end]
            if len(segment) < 2 or segment[0] not in {"'", '"'}:
                continue
            replacements.append((start, end, segment.replace(node.value, mapped, 1)))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]

    if old_relative is not None and new_relative is not None:
        old_root_index = len(old_relative.parts) - 1
        new_root_index = len(new_relative.parts) - 1
        if old_root_index != new_root_index:
            root_pattern = re.compile(
                rf"(Path\(__file__\)(?:\.resolve\(\))?\.parents\[){old_root_index}(\])"
            )
            text = root_pattern.sub(rf"\g<1>{new_root_index}\g<2>", text)

    script_map = script_map or {}
    all_path_map = {**(path_map or {}), **script_map}
    for old_path, new_path in all_path_map.items():
        # Match an exact mapped script suffix.  An external fixture such as
        # ``/repo/scripts/foo.py`` is intentionally updated too, while an
        # unrelated path such as ``omni/physx/scripts/particleUtils.py`` does
        # not match unless that basename is itself in the move map.
        path_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.]){re.escape(old_path)}(?![A-Za-z0-9_.])"
        )
        text = path_pattern.sub(new_path, text)

        old_name = Path(old_path).name
        new_directory = Path(new_path).parent.as_posix()
        split_pattern = re.compile(
            rf"(?P<q>[\"'])scripts(?P=q)\s*/\s*(?P<q2>[\"'])"
            rf"{re.escape(old_name)}(?P=q2)"
        )
        text = split_pattern.sub(
            rf"\g<q>{new_directory}\g<q> / \g<q2>{old_name}\g<q2>", text
        )

    # A bare ``ROOT / "scripts"`` is normally a sys.path insertion for a
    # sibling command.  Retarget it only when this file names one unambiguous
    # moved script directory; arbitrary temporary ``scripts`` fixtures remain
    # untouched.
    candidate_dirs = {
        Path(new_path).parent.as_posix()
        for old_path, new_path in script_map.items()
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(Path(old_path).name)}(?![A-Za-z0-9_])",
            text,
        )
    }
    if len(candidate_dirs) == 1:
        replacement_directory = next(iter(candidate_dirs))
        standalone_pattern = re.compile(r'(?P<q>["\'])scripts(?P=q)(?!\s*/)')

        def replace_scripts_directory(match: re.Match[str]) -> str:
            context = text[max(0, match.start() - 40) : match.start()]
            if "tmp_path" in context:
                return match.group(0)
            return f'{match.group("q")}{replacement_directory}{match.group("q")}'

        text = standalone_pattern.sub(replace_scripts_directory, text)
    text = _remove_repo_scripts_sys_path(text)
    return _ensure_cli_repo_root(text, new_relative)


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
    if path.suffix not in {".py", ".sh", ".json"} or path.parts[0] not in {
        "src", "cli", "scripts", "tests"
    }:
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
        original = _semantic_rewrite(old_relative, staged.stdout.decode("utf-8"))
        if path.suffix in {".sh", ".json"}:
            return (root / path).read_text() == original
        script_map = {
            old.as_posix(): new.as_posix()
            for old, new in pairs
            if old.parts and old.parts[0] == "scripts"
        }
        path_map = {old.as_posix(): new.as_posix() for old, new in pairs}
        expected = _rewrite_text(
            original,
            _import_package(old_relative),
            _import_package(path),
            {**module_map, **_script_module_map(script_map), **_test_module_map(pairs)},
            old_relative=old_relative,
            new_relative=path,
            script_map=script_map,
            path_map=path_map,
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
    script_map = {
        old.relative_to(root).as_posix(): destination.relative_to(root).as_posix()
        for old, destination in moves
        if old.relative_to(root).parts[0] == "scripts"
    }
    path_map = {
        old.relative_to(root).as_posix(): destination.relative_to(root).as_posix()
        for old, destination in moves
    }
    module_map.update(_script_module_map(script_map))
    module_map.update(
        _test_module_map(
            [(old.relative_to(root), new.relative_to(root)) for old, new in moves]
        )
    )
    result: dict[Path, str] = {}
    for relative in _tracked(root):
        source = root / relative
        if source.suffix not in {".py", ".sh", ".json"} or relative.parts[0] not in {
            "src", "scripts", "tests"
        }:
            continue
        destination = move_by_source.get(source, source)
        old_relative = source.relative_to(root)
        new_relative = destination.relative_to(root)
        old_package = _import_package(old_relative)
        new_package = _import_package(new_relative)
        original = _semantic_rewrite(old_relative, source.read_text())
        if source.suffix in {".sh", ".json"}:
            if original != source.read_text():
                result[destination] = original
            continue
        rewritten = _rewrite_text(
            original,
            old_package,
            new_package,
            module_map,
            old_relative=old_relative,
            new_relative=new_relative,
            script_map=script_map,
            path_map=path_map,
        )
        if rewritten != source.read_text():
            result[destination] = rewritten
    return result


def _staged_rewrites(
    root: Path, pairs: list[tuple[Path, Path]]
) -> dict[Path, str]:
    """Rebuild audited semantic migrations from index blobs and the path map."""
    reverse = {new: old for old, new in pairs}
    module_map = {
        old_module: new_module
        for old, new in pairs
        if (old_module := _module_name(old)) is not None
        and (new_module := _module_name(new)) is not None
    }
    script_map = {
        old.as_posix(): new.as_posix()
        for old, new in pairs
        if old.parts and old.parts[0] == "scripts"
    }
    path_map = {old.as_posix(): new.as_posix() for old, new in pairs}
    module_map.update(_script_module_map(script_map))
    module_map.update(_test_module_map(pairs))
    result: dict[Path, str] = {}
    for relative in _tracked(root):
        if relative.suffix not in {".py", ".sh", ".json"} or relative.parts[0] not in {
            "src", "cli", "tests"
        }:
            continue
        old_relative = reverse.get(relative, relative)
        # Python files all pass through the generic import/path migrations.
        # Non-Python fixtures are replayed only when they have an audited,
        # fail-closed semantic rule.
        if relative.suffix != ".py" and old_relative.as_posix() not in SEMANTIC_REWRITES:
            continue
        staged = subprocess.run(
            ["git", "-C", str(root), "show", f":{relative.as_posix()}"],
            capture_output=True,
        )
        if staged.returncode:
            raise SystemExit(f"refusing: missing staged blob for {relative}")
        try:
            indexed = staged.stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SystemExit(f"refusing: non-UTF-8 staged text file {relative}") from error
        rewritten = _semantic_rewrite(old_relative, indexed)
        if relative.suffix == ".py":
            rewritten = _rewrite_text(
                rewritten,
                _import_package(old_relative),
                _import_package(relative),
                module_map,
                old_relative=old_relative,
                new_relative=relative,
                script_map=script_map,
                path_map=path_map,
            )
        destination = root / relative
        if destination.read_text() != rewritten:
            result[destination] = rewritten
    return result


def run(
    root: Path, mapping_path: Path, dry_run: bool, report_only: bool,
    refresh_staged_rewrites: bool = False,
) -> int:
    root = root.resolve()
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True, capture_output=True,
    )
    if probe.returncode or probe.stdout.strip() != "true":
        raise SystemExit("refusing: root must be an existing Git worktree")
    moves, unmapped = plan(root, mapping_path)
    _preflight(root, moves)
    if refresh_staged_rewrites:
        if dry_run or report_only:
            raise SystemExit("refusing: staged rewrite refresh cannot be combined with report modes")
        pairs = _pathmap_pairs(root)
        if pairs is None or not _cached_matches_moves(root, pairs):
            raise SystemExit("refusing: staged rewrite refresh requires the exact recorded move index")
        for path, text in _staged_rewrites(root, pairs).items():
            path.write_text(text)
        return 0
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
    parser.add_argument("--refresh-staged-rewrites", action="store_true")
    args = parser.parse_args()
    return run(
        args.root, args.mapping.resolve(), args.dry_run, args.report_only,
        args.refresh_staged_rewrites,
    )

if __name__ == "__main__":
    raise SystemExit(main())
