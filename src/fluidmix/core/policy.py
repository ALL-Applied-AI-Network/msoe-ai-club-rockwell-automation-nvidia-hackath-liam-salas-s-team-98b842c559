"""Single deploy-input field policy for synthetic F2 measurement artifacts.

Render and simulation truth may exist in provenance artifacts, but this module is
the only vocabulary gate used before values enter a deployable context or model
plan.  Pixel-derived measurements are produced after this gate and are not
simulation oracles.
"""

from __future__ import annotations

from collections.abc import Mapping


PART_A_SIM_METADATA_FIELDS = frozenset(
    {
        "tank_colour_preset",
        "resolved_tank_transmission_1cm",
        "advanced_toggles",
        "tank_colour_preset_hash",
        "renderer_palette",
        "renderer_seed",
        "residue_ml",
        "residue_source_liquid",
        "particle_state_summary",
        "label",
        "host_profile",
        "excluded_from_model_inputs",
    }
)


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
        "recipe_ml",
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
        "vials",
        "vial_position",
        "stages",
        "evidence_id",
        "measurement_id",
        "quantitative_evidence_ids",
        "stage_id",
        "stage_image_evidence_ids",
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

_PART_B_EVIDENCE_FIELDS = frozenset(
    {
        "run_id", "case_id", "vial_ordinal", "block_recipe_id", "block_position",
        "recipe_id", "previous_recipe_id", "recipe_ml", "dispense_order",
        "split_role", "split_version", "stage_id", "tank_id", "stage_ordinal",
        "fill_clip_path", "fill_frame_range", "transition_frame_ranges",
        "clip_sha256", "image_id", "rgb_sha256", "mask_sha256", "mask_source",
        "measurement_id", "median_lab", "mad_lab", "iqr_lab",
        "pairwise_delta_e76", "colour_regime_id", "evidence_id", "decision_id",
        "disposition", "limit_codes", "history_refs", "explanation_id",
        "model_revision", "prompt_sha256", "image_ids_sent", "guard_result",
    }
)
_PART_B_PATH_FIELDS = {
    "recipe_ml": frozenset({"t100", "t200", "t400"}),
    "transition_frame_ranges": frozenset({"entry", "exit"}),
    "pairwise_delta_e76[]": frozenset({"other_evidence_id", "delta_e76"}),
    "history_refs[]": frozenset({"evidence_id", "query_sha256"}),
}


def _require_exact_path_fields(
    value: object, expected: frozenset[str], path: str
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"deploy input envelope is invalid at {path}")
    for key in value:
        if not isinstance(key, str):
            raise ValueError("deploy input field names must be strings")
    actual = set(value)
    extra = sorted(actual.difference(expected))
    if extra:
        part_a_extra = sorted(set(extra).intersection(PART_A_SIM_METADATA_FIELDS))
        offending = part_a_extra[0] if part_a_extra else extra[0]
        prefix = f"{path}." if path else ""
        raise ValueError(f"forbidden_deploy_input_field: {prefix}{offending}")
    missing = sorted(expected.difference(actual))
    if missing:
        prefix = f"{path}." if path else ""
        raise ValueError(f"deploy input field missing: {prefix}{missing[0]}")


def _validate_part_b_evidence_envelope(value: object) -> None:
    _require_exact_path_fields(value, _PART_B_EVIDENCE_FIELDS, "")
    assert isinstance(value, Mapping)
    for path in ("recipe_ml", "transition_frame_ranges"):
        _require_exact_path_fields(value[path], _PART_B_PATH_FIELDS[path], path)
    for path in ("pairwise_delta_e76", "history_refs"):
        rows = value[path]
        if not isinstance(rows, list):
            raise ValueError(f"deploy input envelope is invalid at {path}")
        expected = _PART_B_PATH_FIELDS[f"{path}[]"]
        for index, row in enumerate(rows):
            _require_exact_path_fields(row, expected, f"{path}[{index}]")


def validate_deploy_input_fields(
    value: object,
    *,
    require_allowlist: bool = True,
    envelope: str | None = None,
) -> None:
    """Validate a named exact production envelope or a historical vocabulary."""

    if envelope is not None:
        if envelope != "part_b_evidence_v1":
            raise ValueError(f"unknown deploy input envelope: {envelope}")
        _validate_part_b_evidence_envelope(value)
        return

    def visit(item: object, *, dynamic_keys: bool = False) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("deploy input field names must be strings")
                normalized = key.lower()
                if not dynamic_keys and normalized in FORBIDDEN_DEPLOY_INPUT_FIELDS:
                    raise ValueError(f"forbidden_deploy_input_field: {key}")
                if require_allowlist and not dynamic_keys and key not in DEPLOY_INPUT_ALLOWED_FIELDS:
                    raise ValueError(f"forbidden_deploy_input_field: {key}")
                visit(child, dynamic_keys=key in _DYNAMIC_KEY_CONTAINERS)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
