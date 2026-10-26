"""Single deploy-input field policy for synthetic F2 measurement artifacts.

Render and simulation truth may exist in provenance artifacts, but this module is
the only vocabulary gate used before values enter a deployable context or model
plan.  Pixel-derived measurements are produced after this gate and are not
simulation oracles.
"""

from __future__ import annotations

from collections.abc import Mapping


FORBIDDEN_DEPLOY_INPUT_FIELDS = frozenset(
    {
        "alpha",
        "appearance_blend_alpha",
        "appearance_blend_alpha_by_vial",
        "clean_stage_linear_rgb",
        "composition",
        "expected_disposition",
        "expected_height",
        "expected_height_m",
        "expected_volume",
        "expected_volume_ml",
        "exposure",
        "label",
        "labels",
        "label_source",
        "latent_fault",
        "latent_parameters",
        "light",
        "lighting",
        "nuisance_id",
        "nuisance_seed",
        "oracle",
        "oracle_fields",
        "particle_count",
        "particle_counts",
        "profile_id",
        "render_seed",
        "render_parameters",
        "residual_particle_count",
        "residual_particle_count_by_vial",
        "residual_particle_plan",
        "residual_source_component_counts",
        "semantic_aov",
        "source_component_count",
        "source_component_counts",
        "synthetic_onset_id",
        "synthetic_oracle_record",
        "synthetic_profile_id",
    }
)

# The allowlist is a vocabulary, not a serialization schema.  Boundary-specific
# code still enforces exact object shapes and types.
DEPLOY_INPUT_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "split_version",
        "batches",
        "batch_id",
        "cohort",
        "family_id",
        "current_recipe_id",
        "previous_recipe_id",
        "run_id",
        "programmed_recipe",
        "split_role",
        "dispense_order",
        "previous_ml_by_tank",
        "current_ml_by_tank",
        "t100",
        "t200",
        "t400",
        "data_origin",
        "config_id",
        "decision_config",
        "decision_config_sha256",
        "decision_timing",
        "reference_policy",
        "source_batch_context_sha256",
        "source_capture_manifest_sha256",
        "decision_records",
        "real_data_used",
        "claim_boundary",
        "primary_mode",
        "minimum_peer_count",
        "minimum_distinct_batch_ids",
        "minimum_distinct_run_ids",
        "tight_cluster_max_diameter_delta_e76",
        "threshold_floor_delta_e76",
        "threshold_margin_delta_e76",
        "unique_largest_cluster_required",
        "same_batch_case_is_diagnostic_only",
        "decision_id",
        "partition",
        "previous_sim_recipe_id",
        "current_sim_recipe_id",
        "previous_programmed_ml",
        "current_programmed_ml",
        "programmed_dispense_order",
        "vial_id",
        "vial_position",
        "stages",
        "evidence_id",
        "measurement_id",
        "stage_id",
        "stage_ordinal",
        "stage_event",
        "event_id",
        "tank_id",
        "tank_order",
        "physics_step",
        "frame_id",
        "event_relative_timestamp_s",
        "final_observation_offset_s",
        "time_code",  # historical schema-v1 plans only
        "target_image_id",
        "reference_cases",
        "case_id",
        "mode",
        "alternative_id",
        "candidate_peers",
        "disposition_source",
        "peer_image_id",
        "origin",
        "primary_mode",
        "unit",
        "runtime_binding_sha256",
    }
)

_DYNAMIC_KEY_CONTAINERS = frozenset({"source_capture_manifest_sha256"})


def validate_deploy_input_fields(value: object, *, require_allowlist: bool = True) -> None:
    """Reject oracle names and, optionally, every unknown deploy-input key."""

    def visit(item: object, *, dynamic_keys: bool = False) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("deploy input field names must be strings")
                normalized = key.lower()
                if normalized in FORBIDDEN_DEPLOY_INPUT_FIELDS:
                    raise ValueError(f"forbidden_deploy_input_field: {key}")
                if require_allowlist and not dynamic_keys and key not in DEPLOY_INPUT_ALLOWED_FIELDS:
                    raise ValueError(f"forbidden_deploy_input_field: {key}")
                visit(child, dynamic_keys=key in _DYNAMIC_KEY_CONTAINERS)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
