"""Strict Part A sim-metadata-v1 validation and external bindings."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path
from collections.abc import Mapping
import unicodedata

from ._common import (
    bounded_control_free_text,
    canonical_json_bytes,
    contained_relative_path,
    exact_lower_sha256,
    load_schema,
    reject_nonfinite,
    sha256_bytes,
    stable_read_relative_file,
    strict_json_bytes,
    validate_schema,
)


_SCHEMA = load_schema("sim-metadata-v1.schema.json")
_INTERPRETATION = (
    "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
)
_ADVANCED_KEYS = {
    "transmission_overrides",
    "dye_strength_scale",
    "t100_clear_lock",
    "randomize_around_preset",
    "randomization_seed",
    "tank_strength_jitter_bounds",
    "channel_jitter_bounds",
    "tank_strength_jitter",
    "channel_jitter",
    "free_text_label",
}
_ONE_HOT = {
    "t200": [1.0, 0.0, 0.0],
    "t400": [0.0, 1.0, 0.0],
    "t100": [0.0, 0.0, 1.0],
}
_TANKS = ("t100", "t200", "t400")
_PRESET_TRANSMISSION_1CM = {
    "base": {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.95, 0.39, 0.26),
        "t400": (0.14, 0.61, 0.95),
    },
    "amber_teal": {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.92, 0.48, 0.16),
        "t400": (0.12, 0.72, 0.68),
    },
    "crimson_indigo": {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.88, 0.24, 0.22),
        "t400": (0.20, 0.36, 0.90),
    },
}
_TANK_STRENGTH_JITTER_BOUNDS = (-0.15, 0.15)
_CHANNEL_JITTER_BOUNDS = (-0.05, 0.05)


def _finite_inclusive_float(
    value: object, name: str, *, lower: float, upper: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def _finite_transmission(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be finite and in (0.0, 1.0]")
    return result


def _resolution_jitter(
    preset_name: str,
    seed: int | None,
    tank: str,
    kind: str,
    channel: int | None = None,
    *,
    bounds: tuple[float, float],
) -> float:
    payload: list[object] = [
        "fluidmix-tank-colour-jitter-v1",
        preset_name,
        seed,
        tank,
        kind,
    ]
    if channel is not None:
        payload.append(channel)
    unit = int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big"
    ) / float(1 << 64)
    lower, upper = bounds
    return lower + (upper - lower) * unit


def _validate_tank_colour_resolution_semantics(
    record: object,
) -> dict[str, object]:
    """Replay the published Task-1 preset semantics without importing twin."""

    expected_fields = {
        "schema_version",
        "catalog_schema_version",
        "preset_name",
        "colour_space",
        "reference_depth_m",
        "transmission_1cm",
        *_ADVANCED_KEYS,
        "interpretation",
        "excluded_from_model_inputs",
        "tank_colour_preset_hash",
        "renderer_palette_sha256",
        "palette_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("tank-colour resolution fields are invalid")
    preset_name = record.get("preset_name")
    if not isinstance(preset_name, str) or preset_name not in _PRESET_TRANSMISSION_1CM:
        raise ValueError("tank-colour resolution preset_name is invalid")
    clear_lock = record.get("t100_clear_lock")
    randomized = record.get("randomize_around_preset")
    if type(clear_lock) is not bool:
        raise TypeError("t100_clear_lock must be Boolean")
    if type(randomized) is not bool:
        raise TypeError("randomize_around_preset must be Boolean")
    scale = _finite_inclusive_float(
        record.get("dye_strength_scale"),
        "dye_strength_scale",
        lower=0.5,
        upper=1.5,
    )
    label = record.get("free_text_label")
    if label is not None:
        if not isinstance(label, str):
            raise TypeError("free_text_label must be text or None")
        if len(label) > 200:
            raise ValueError("free_text_label must contain at most 200 Unicode characters")
        if any(unicodedata.category(character).startswith("C") for character in label):
            raise ValueError("free_text_label must not contain control characters")
    raw_overrides = record.get("transmission_overrides")
    if raw_overrides is None:
        raw_overrides = {}
    elif not isinstance(raw_overrides, Mapping):
        raise TypeError("transmission_overrides must be a mapping or None")
    if not set(raw_overrides).issubset(_TANKS):
        raise ValueError("transmission_overrides contains an unknown tank")
    overrides: dict[str, tuple[float, float, float]] = {}
    for tank, rgb in raw_overrides.items():
        if (
            isinstance(rgb, (str, bytes))
            or not isinstance(rgb, (tuple, list))
            or len(rgb) != 3
        ):
            raise ValueError(f"override {tank} must have three channels")
        overrides[str(tank)] = tuple(
            _finite_transmission(channel, f"override {tank} channel")
            for channel in rgb
        )
    if clear_lock and "t100" in overrides and overrides["t100"] != (1.0, 1.0, 1.0):
        raise ValueError("t100 clear lock rejects a non-clear override")
    seed = record.get("randomization_seed")
    if not randomized:
        if seed is not None:
            raise ValueError("randomization_seed requires randomize_around_preset=True")
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("randomization_seed must be a non-Boolean integer")
    source = {
        tank: overrides.get(tank, _PRESET_TRANSMISSION_1CM[preset_name][tank])
        for tank in _TANKS
    }
    transmission: dict[str, list[float]] = {}
    tank_jitter: dict[str, float] = {}
    channel_jitter: dict[str, list[float]] = {}
    for tank in _TANKS:
        if clear_lock and tank == "t100":
            transmission[tank] = [1.0, 1.0, 1.0]
            tank_jitter[tank] = 0.0
            channel_jitter[tank] = [0.0, 0.0, 0.0]
            continue
        strength_delta = 0.0
        channel_deltas = (0.0, 0.0, 0.0)
        if randomized:
            strength_delta = _resolution_jitter(
                preset_name,
                seed,
                tank,
                "strength",
                bounds=_TANK_STRENGTH_JITTER_BOUNDS,
            )
            channel_deltas = tuple(
                _resolution_jitter(
                    preset_name,
                    seed,
                    tank,
                    "channel",
                    channel,
                    bounds=_CHANNEL_JITTER_BOUNDS,
                )
                for channel in range(3)
            )
        tank_jitter[tank] = round(strength_delta, 9)
        channel_jitter[tank] = [round(delta, 9) for delta in channel_deltas]
        transmission[tank] = [
            _finite_transmission(
                round(
                    math.exp(
                        -(-math.log(channel))
                        * scale
                        * (1.0 + strength_delta)
                        * (1.0 + channel_delta)
                    ),
                    9,
                ),
                f"resolved {tank} channel",
            )
            for channel, channel_delta in zip(
                source[tank], channel_deltas, strict=True
            )
        ]
    semantic: dict[str, object] = {
        "schema_version": "tank-colour-preset-resolution-v1",
        "catalog_schema_version": "tank-colour-presets-v1",
        "preset_name": preset_name,
        "colour_space": "linear_rgb_transmission",
        "reference_depth_m": 0.01,
        "transmission_1cm": transmission,
        "transmission_overrides": {
            tank: list(overrides[tank]) for tank in _TANKS if tank in overrides
        },
        "dye_strength_scale": scale,
        "t100_clear_lock": clear_lock,
        "randomize_around_preset": randomized,
        "randomization_seed": seed,
        "tank_strength_jitter_bounds": list(_TANK_STRENGTH_JITTER_BOUNDS),
        "channel_jitter_bounds": list(_CHANNEL_JITTER_BOUNDS),
        "tank_strength_jitter": tank_jitter,
        "channel_jitter": channel_jitter,
        "free_text_label": label,
        "interpretation": _INTERPRETATION,
        "excluded_from_model_inputs": True,
    }
    renderer_hash = sha256_bytes(canonical_json_bytes(transmission))
    expected = {
        **semantic,
        "tank_colour_preset_hash": sha256_bytes(canonical_json_bytes(semantic)),
        "renderer_palette_sha256": renderer_hash,
        "palette_sha256": renderer_hash,
    }
    if canonical_json_bytes(dict(record)) != canonical_json_bytes(expected):
        raise ValueError("tank-colour resolution does not match Task 1 semantics")
    return deepcopy(expected)


def _same_non_boolean_number(actual: object, expected: object) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and actual == expected
    )


def _same_numeric_vector(actual: object, expected: object) -> bool:
    return (
        isinstance(actual, list)
        and isinstance(expected, list)
        and len(actual) == len(expected)
        and all(
            _same_non_boolean_number(observed, required)
            for observed, required in zip(actual, expected, strict=True)
        )
    )


def _validate_sim_lexical_fields(value: dict[str, object]) -> None:
    """Apply exact hash and Unicode-category rules at the data boundary."""

    renderer = value["renderer_palette"]
    source = value["residue_source_liquid"]
    particle = value["particle_state_summary"]
    host = value["host_profile"]
    assert isinstance(renderer, dict)
    assert isinstance(source, dict)
    assert isinstance(particle, dict)
    assert isinstance(host, dict)

    for role, digest in (
        ("tank_colour_preset_hash", value["tank_colour_preset_hash"]),
        ("renderer_palette.renderer_palette_sha256", renderer["renderer_palette_sha256"]),
        ("renderer_palette.palette_sha256", renderer["palette_sha256"]),
        ("particle_state_summary.manifest_sha256", particle["manifest_sha256"]),
        ("host_profile.profile_sha256", host["profile_sha256"]),
    ):
        exact_lower_sha256(digest, role)
    if source["previous_recipe_id"] is not None:
        bounded_control_free_text(
            source["previous_recipe_id"],
            "residue_source_liquid.previous_recipe_id",
        )
    for role, identifier in (
        ("particle_state_summary.resolution_profile_id", particle["resolution_profile_id"]),
        ("host_profile.profile_id", host["profile_id"]),
        ("host_profile.gpu_model", host["gpu_model"]),
        ("host_profile.isaac_version", host["isaac_version"]),
        ("host_profile.kit_version", host["kit_version"]),
    ):
        bounded_control_free_text(identifier, role)


def validate_sim_metadata_v1(record: object) -> dict[str, object]:
    """Validate one exact Part A case record and return a defensive copy."""

    reject_nonfinite(record, "sim metadata")
    validate_schema(record, _SCHEMA, "sim metadata")
    assert isinstance(record, dict)
    value = deepcopy(record)
    _validate_sim_lexical_fields(value)
    advanced = value["advanced_toggles"]
    renderer = value["renderer_palette"]
    assert isinstance(advanced, dict) and isinstance(renderer, dict)
    if set(advanced) != _ADVANCED_KEYS:
        raise ValueError("advanced_toggles fields are invalid")
    resolved_palette_bytes = canonical_json_bytes(
        value["resolved_tank_transmission_1cm"]
    )
    renderer_palette_bytes = canonical_json_bytes(renderer["transmission_1cm"])
    if resolved_palette_bytes != renderer_palette_bytes:
        raise ValueError("resolved and renderer transmissions disagree")
    expected_palette_hash = sha256_bytes(renderer_palette_bytes)
    if (
        renderer["renderer_palette_sha256"] != expected_palette_hash
        or renderer["palette_sha256"] != expected_palette_hash
    ):
        raise ValueError("renderer palette hashes disagree with exact transmission bytes")
    resolution_record = {
        "schema_version": "tank-colour-preset-resolution-v1",
        "catalog_schema_version": "tank-colour-presets-v1",
        "preset_name": value["tank_colour_preset"],
        "colour_space": "linear_rgb_transmission",
        "reference_depth_m": renderer["reference_depth_m"],
        "transmission_1cm": value["resolved_tank_transmission_1cm"],
        **deepcopy(advanced),
        "interpretation": _INTERPRETATION,
        "excluded_from_model_inputs": True,
        "tank_colour_preset_hash": value["tank_colour_preset_hash"],
        "renderer_palette_sha256": renderer["renderer_palette_sha256"],
        "palette_sha256": renderer["palette_sha256"],
    }
    normalized = _validate_tank_colour_resolution_semantics(resolution_record)
    if (
        canonical_json_bytes(normalized["transmission_1cm"])
        != resolved_palette_bytes
        or normalized["tank_colour_preset_hash"]
        != value["tank_colour_preset_hash"]
        or normalized["renderer_palette_sha256"]
        != renderer["renderer_palette_sha256"]
        or normalized["palette_sha256"] != renderer["palette_sha256"]
    ):
        raise ValueError("sim metadata tank-colour resolution binding mismatch")

    residue = float(value["residue_ml"])
    source = value["residue_source_liquid"]
    label = value["label"]
    assert isinstance(source, dict) and isinstance(label, dict)
    rows = source["per_dispense"]
    assert isinstance(rows, list)
    if residue == 0.0:
        if source["previous_recipe_id"] is not None or rows:
            raise ValueError("zero residue requires no source liquid")
        expected_disposition = "keep"
    else:
        if source["previous_recipe_id"] is None or len(rows) not in {2, 3}:
            raise ValueError("positive residue requires 2 or 3 source-liquid rows")
        if [row["stage_ordinal"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("residue source stage ordinals must be contiguous")
        for row in rows:
            if row["composition_t200_t400_t100"] != _ONE_HOT[row["source_tank_id"]]:
                raise ValueError("residue source composition must match its source tank")
        expected_disposition = "discard"
    if label["disposition"] != expected_disposition:
        raise ValueError("sim metadata label contradicts residue_ml")
    contained_relative_path(
        value["particle_state_summary"]["manifest_path"],
        "particle_state_summary.manifest_path",
    )
    contained_relative_path(
        value["host_profile"]["profile_path"], "host_profile.profile_path"
    )
    return value


def verify_sim_metadata_artifacts(
    record: object, artifact_root: str | Path
) -> dict[str, object]:
    """Verify particle and host summary files through stable descriptors."""

    value = validate_sim_metadata_v1(record)
    particle = value["particle_state_summary"]
    host = value["host_profile"]
    assert isinstance(particle, dict) and isinstance(host, dict)
    particle_raw = stable_read_relative_file(
        artifact_root,
        particle["manifest_path"],
        "particle manifest",
        max_bytes=64 * 1024 * 1024,
    )
    host_raw = stable_read_relative_file(
        artifact_root,
        host["profile_path"],
        "host profile",
        max_bytes=64 * 1024 * 1024,
    )
    if (
        sha256_bytes(particle_raw) != particle["manifest_sha256"]
        or len(particle_raw) != particle["manifest_bytes"]
    ):
        raise ValueError("particle manifest summary mismatch")
    if (
        sha256_bytes(host_raw) != host["profile_sha256"]
        or len(host_raw) != host["profile_bytes"]
    ):
        raise ValueError("host profile summary mismatch")
    particle_payload = strict_json_bytes(particle_raw, "particle manifest")
    host_payload = strict_json_bytes(host_raw, "host profile")
    if (
        not isinstance(particle_payload, Mapping)
        or type(particle_payload.get("particle_count")) is not int
        or particle_payload.get("particle_count") != particle["particle_count"]
        or particle_payload.get("finite") is not True
        or type(particle_payload.get("resolution_profile_id")) is not str
        or particle_payload.get("resolution_profile_id")
        != particle["resolution_profile_id"]
    ):
        raise ValueError("particle manifest content mismatch")
    host_text_fields = (
        "profile_id",
        "compute_kind",
        "gpu_model",
        "isaac_version",
        "kit_version",
    )
    if (
        not isinstance(host_payload, Mapping)
        or any(
            type(host_payload.get(field)) is not str
            or host_payload.get(field) != host[field]
            for field in host_text_fields
        )
        or type(host_payload.get("gpu_count")) is not int
        or host_payload.get("gpu_count") != host["gpu_count"]
    ):
        raise ValueError("host profile content mismatch")
    return value


def validate_sim_metadata_binding_v1(
    binding_receipt: object,
    *,
    expected_case_id: str,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Verify the orchestration-only case/path/hash binding and return metadata."""

    expected_fields = {
        "schema_version",
        "case_id",
        "sim_metadata_path",
        "sim_metadata_sha256",
    }
    if not isinstance(binding_receipt, Mapping) or set(binding_receipt) != expected_fields:
        raise ValueError("sim metadata binding fields are invalid")
    if binding_receipt.get("schema_version") != "sim-metadata-binding-v1":
        raise ValueError("sim metadata binding schema_version is invalid")
    bounded_control_free_text(expected_case_id, "expected_case_id")
    bounded_control_free_text(binding_receipt.get("case_id"), "binding case_id")
    if binding_receipt.get("case_id") != expected_case_id:
        raise ValueError("sim metadata binding case_id mismatch")
    relative = contained_relative_path(
        binding_receipt.get("sim_metadata_path"), "sim_metadata_path"
    )
    expected_hash = binding_receipt.get("sim_metadata_sha256")
    exact_lower_sha256(expected_hash, "sim metadata binding SHA-256")
    raw = stable_read_relative_file(
        artifact_root, relative, "bound sim metadata", max_bytes=16 * 1024 * 1024
    )
    if sha256_bytes(raw) != expected_hash:
        raise ValueError("bound sim metadata SHA-256 mismatch")
    parsed = strict_json_bytes(raw, "bound sim metadata")
    value = validate_sim_metadata_v1(parsed)
    verify_sim_metadata_artifacts(value, artifact_root)
    return value


def validate_sim_metadata_runtime_schedule_v1(
    record: object,
    runtime_schedule: object,
    label_provenance: object,
    *,
    expected_case_id: str,
) -> None:
    """Cross-check one metadata record against finalized runtime and label truth."""

    bounded_control_free_text(expected_case_id, "expected_case_id")
    value = validate_sim_metadata_v1(record)
    if not isinstance(runtime_schedule, Mapping):
        raise TypeError("runtime_schedule must be an object")
    if (
        type(runtime_schedule.get("schema_version")) is not int
        or runtime_schedule.get("schema_version") != 1
        or runtime_schedule.get("artifact_type")
        != "physx_f2_runtime_leading_residual_schedule"
    ):
        raise ValueError("runtime_schedule must be the finalized runtime artifact")
    if not isinstance(label_provenance, Mapping):
        raise TypeError("label_provenance must be an object")
    required_label_fields = {
        "case_id",
        "prescribed_leading_residual_volume_ml",
        "expected_disposition",
        "label_source",
        "render_schedule_sha256",
        "excluded_from_model_inputs",
        "excluded_from_deploy_inputs",
    }
    if not required_label_fields.issubset(label_provenance):
        raise ValueError("label_provenance is missing required binding fields")
    if label_provenance.get("case_id") != expected_case_id:
        raise ValueError("runtime label case_id mismatch")
    residue = value["residue_ml"]
    if not _same_non_boolean_number(
        label_provenance.get("prescribed_leading_residual_volume_ml"), residue
    ):
        raise ValueError("runtime label residue mismatch")
    label = value["label"]
    assert isinstance(label, dict)
    if (
        label_provenance.get("expected_disposition") != label["disposition"]
        or label_provenance.get("label_source") != label["rule_id"]
        or label_provenance.get("excluded_from_model_inputs") is not True
        or label_provenance.get("excluded_from_deploy_inputs") is not True
    ):
        raise ValueError("runtime label truth or exclusion binding mismatch")
    import hashlib

    schedule_hash = hashlib.sha256(canonical_json_bytes(runtime_schedule)).hexdigest()
    if label_provenance.get("render_schedule_sha256") != schedule_hash:
        raise ValueError("runtime label render_schedule_sha256 mismatch")
    schedule_label = runtime_schedule.get("label_provenance")
    model_input = runtime_schedule.get("model_input_record")
    physical_sequence = runtime_schedule.get("physical_sequence")
    case_binding = runtime_schedule.get("case_binding")
    pours = runtime_schedule.get("pours")
    if (
        not isinstance(schedule_label, Mapping)
        or not _same_non_boolean_number(
            schedule_label.get("leading_residual_volume_ml"), residue
        )
        or schedule_label.get("excluded_from_model_inputs") is not True
        or not isinstance(model_input, Mapping)
        or not isinstance(physical_sequence, Mapping)
        or not isinstance(case_binding, Mapping)
        or case_binding.get("case_id") != expected_case_id
        or not _same_non_boolean_number(
            case_binding.get("residual_volume_ml"), residue
        )
        or not isinstance(pours, list)
        or len(pours) not in {2, 3}
    ):
        raise ValueError("runtime schedule residue structure mismatch")
    if (
        "previous_recipe_id" not in case_binding
        or "previous_recipe_id" not in model_input
        or "previous_final_tank_id" not in model_input
    ):
        raise ValueError("runtime schedule predecessor context fields are missing")
    dispense_order = physical_sequence.get("dispense_order")
    if (
        not isinstance(dispense_order, list)
        or len(dispense_order) not in {2, 3}
        or any(
            type(tank_id) is not str or tank_id not in _ONE_HOT
            for tank_id in dispense_order
        )
        or len(set(dispense_order)) != len(dispense_order)
        or len(dispense_order) != len(pours)
        or [pour.get("tank_id") for pour in pours if isinstance(pour, Mapping)]
        != dispense_order
        or model_input.get("dispense_order") != dispense_order
    ):
        raise ValueError("runtime schedule programmed dispense order mismatch")
    source = value["residue_source_liquid"]
    assert isinstance(source, dict)
    contextual_previous_recipe = model_input.get("previous_recipe_id")
    case_previous_recipe = case_binding.get("previous_recipe_id")
    residue_is_positive = float(residue) > 0.0
    if case_previous_recipe != contextual_previous_recipe:
        raise ValueError("runtime schedule previous_recipe_id mismatch")
    clean_start = contextual_previous_recipe is None
    if clean_start:
        if residue_is_positive:
            raise ValueError("positive residue requires predecessor context")
    else:
        bounded_control_free_text(
            contextual_previous_recipe, "runtime model_input previous_recipe_id"
        )
        bounded_control_free_text(
            case_previous_recipe, "runtime case previous_recipe_id"
        )
    if residue_is_positive and contextual_previous_recipe != source["previous_recipe_id"]:
        raise ValueError("runtime schedule previous_recipe_id mismatch")
    rows = source["per_dispense"]
    assert isinstance(rows, list)
    if residue_is_positive and len(rows) != len(pours):
        raise ValueError("runtime schedule source row count mismatch")
    expected_source = physical_sequence.get("previous_final_tank_id")
    model_previous_final_tank = model_input.get("previous_final_tank_id")
    if expected_source not in _ONE_HOT:
        raise ValueError("runtime schedule previous final tank mismatch")
    if clean_start:
        if model_previous_final_tank is not None:
            raise ValueError("clean-start model input must not invent a previous tank")
    elif model_previous_final_tank != expected_source:
        raise ValueError("runtime schedule previous final tank mismatch")
    for index, pour in enumerate(pours):
        if not isinstance(pour, Mapping):
            raise ValueError("runtime schedule pour is invalid")
        pour_ordinal = pour.get("ordinal")
        if type(pour_ordinal) is not int or pour_ordinal != index + 1:
            raise ValueError("runtime schedule pour ordinal mismatch")
        if not _same_non_boolean_number(
            pour.get("leading_residual_volume_ml"), residue
        ):
            raise ValueError("runtime schedule per-dispense residue mismatch")
        segments = pour.get("emission_segments")
        if (
            not isinstance(segments, list)
            or len(segments) != 2
            or not all(isinstance(segment, Mapping) for segment in segments)
        ):
            raise ValueError("runtime schedule emission segments are invalid")
        current_segment = segments[1]
        programmed_tank = pour.get("tank_id")
        if (
            current_segment.get("segment_role") != "current_tank"
            or current_segment.get("source_tank_id") != programmed_tank
            or not _same_numeric_vector(
                current_segment.get("one_hot_composition"),
                _ONE_HOT[programmed_tank],
            )
        ):
            raise ValueError("runtime schedule current-liquid segment mismatch")
        if not residue_is_positive:
            continue
        row = rows[index]
        if (
            not isinstance(row, Mapping)
            or row.get("stage_ordinal") != pour_ordinal
            or row.get("source_tank_id") != expected_source
            or row.get("source_tank_id") != pour.get("residual_source_tank_id")
            or segments[0].get("segment_role") != "leading_residual"
            or segments[0].get("source_tank_id") != row.get("source_tank_id")
            or not _same_numeric_vector(
                segments[0].get("one_hot_composition"),
                row.get("composition_t200_t400_t100"),
            )
        ):
            raise ValueError("runtime schedule source-liquid binding mismatch")
        expected_source = pour.get("tank_id")
