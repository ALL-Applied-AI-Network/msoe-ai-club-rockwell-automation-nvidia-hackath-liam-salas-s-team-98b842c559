"""Synthetic run-level recipe and appearance variation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
import unicodedata

from fluidmix.tank_colour_presets import (
    DEFAULT_CATALOG_PATH,
    load_tank_colour_preset_catalog,
    resolve_tank_colour_preset,
)


BASE_TANK_TRANSMISSION_1CM = {
    "t200": (0.95, 0.39, 0.26),
    "t400": (0.14, 0.61, 0.95),
    "t100": (1.0, 1.0, 1.0),
}
DISPENSE_ORDER = ("t200", "t400", "t100")
_TANK_KEYS = frozenset(DISPENSE_ORDER)
_TANK_COLOUR_RESOLUTION_KEYS = frozenset(
    {
        "schema_version",
        "catalog_schema_version",
        "preset_name",
        "colour_space",
        "reference_depth_m",
        "transmission_1cm",
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
        "interpretation",
        "excluded_from_model_inputs",
        "tank_colour_preset_hash",
        "renderer_palette_sha256",
        "palette_sha256",
    }
)
_TANK_COLOUR_PRESET_NAMES = frozenset(("base", "amber_teal", "crimson_indigo"))
_TANK_COLOUR_BASE_TRANSMISSION_1CM = {
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
_TANK_COLOUR_INTERPRETATION = (
    "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
)
_LEGACY_VARIED_PLAN_ARTIFACT_TYPES = frozenset(
    {
        "physx_f2_varied_run_residual_render_plan_v3",
        "physx_f2_varied_run_residual_render_plan_v4",
    }
)
_MODEL_INPUT_TANK_COLOUR_SENSITIVE_KEYS = frozenset(
    {
        "tank_colour_resolution",
        "preset_name",
        "transmission_1cm",
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
        "excluded_from_model_inputs",
        "tank_colour_preset_hash",
        "renderer_palette_sha256",
        "palette_sha256",
    }
)


def _normalized_volumes(programmed_ml: Mapping[str, object]) -> dict[str, float | int]:
    if not isinstance(programmed_ml, Mapping) or set(programmed_ml) != _TANK_KEYS:
        raise ValueError("programmed_ml must contain exactly t100, t200, and t400")
    normalized: dict[str, float | int] = {}
    for tank in DISPENSE_ORDER:
        value = programmed_ml[tank]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"programmed_ml[{tank}] must be a real number")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"programmed_ml[{tank}] must be finite and non-negative")
        normalized[tank] = value
    if not any(value > 0 for value in normalized.values()):
        raise ValueError("at least one tank must have positive programmed volume")
    return normalized


def active_dispense_order(programmed_ml: Mapping[str, object]) -> tuple[str, ...]:
    """Return the fixed plant order with zero-volume tanks omitted."""

    normalized = _normalized_volumes(programmed_ml)
    return tuple(tank for tank in DISPENSE_ORDER if normalized[tank] > 0)


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _tank_colour_jitter(
    preset_name: str,
    seed: int,
    tank: str,
    kind: str,
    channel: int | None = None,
    *,
    lower: float,
    upper: float,
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
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    unit = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big") / float(
        1 << 64
    )
    return lower + (upper - lower) * unit


def build_run_palette(*, run_id: str, run_seed: int) -> dict[str, object]:
    """Build one deterministic, uncalibrated 1-cm tank palette for a run.

    T200/T400 variation is applied in extinction space.  A shared per-tank
    factor varies dye strength and small per-channel factors vary hue.  T100
    remains exactly clear and therefore only dilutes a mixture.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError("run_seed must be an integer")

    transmission: dict[str, list[float]] = {}
    variation: dict[str, dict[str, object]] = {}
    for tank in ("t200", "t400"):
        strength = 0.85 + 0.30 * _unit_interval(
            "fluidmix-run-palette-v1", run_id, run_seed, tank, "strength"
        )
        channel_scales = tuple(
            0.95
            + 0.10
            * _unit_interval(
                "fluidmix-run-palette-v1",
                run_id,
                run_seed,
                tank,
                "channel",
                channel,
            )
            for channel in range(3)
        )
        varied = []
        for base, channel_scale in zip(
            BASE_TANK_TRANSMISSION_1CM[tank], channel_scales, strict=True
        ):
            extinction = -math.log(base)
            varied.append(round(math.exp(-extinction * strength * channel_scale), 9))
        transmission[tank] = varied
        variation[tank] = {
            "strength_scale": round(strength, 9),
            "channel_extinction_scales": [
                round(scale, 9) for scale in channel_scales
            ],
        }
    transmission["t100"] = [1.0, 1.0, 1.0]
    palette_sha256 = _canonical_sha256(transmission)
    return {
        "schema_version": "fluidmix-run-palette-v1",
        "run_id": run_id,
        "run_seed": run_seed,
        "reference_depth_m": 0.01,
        "transmission_1cm": transmission,
        "variation_by_tank": variation,
        "palette_sha256": palette_sha256,
        "interpretation": (
            "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
        ),
    }


def validate_tank_colour_resolution(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy one stored Task 1 tank-colour resolution."""

    if not isinstance(record, Mapping) or set(record) != _TANK_COLOUR_RESOLUTION_KEYS:
        raise ValueError("tank_colour_resolution has unexpected fields")

    def require_number(
        value: object, name: str, *, lower: float, upper: float, strict_lower: bool = False
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        try:
            normalized = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be finite") from error
        lower_ok = normalized > lower if strict_lower else normalized >= lower
        if not math.isfinite(normalized) or not lower_ok or normalized > upper:
            interval = "(" if strict_lower else "["
            raise ValueError(f"{name} must be finite and in {interval}{lower}, {upper}]")
        return normalized

    def require_rgb_map(
        value: object, name: str, *, exact_tanks: bool, lower: float, upper: float,
        strict_lower: bool = False,
    ) -> dict[str, list[float]]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        keys = set(value)
        if (exact_tanks and keys != _TANK_KEYS) or (
            not exact_tanks and not keys.issubset(_TANK_KEYS)
        ):
            raise ValueError(f"{name} has an invalid tank set")
        result: dict[str, list[float]] = {}
        for tank in DISPENSE_ORDER:
            if tank not in value:
                continue
            channels = value[tank]
            if not isinstance(channels, list) or len(channels) != 3:
                raise ValueError(f"{name}.{tank} must be a three-channel list")
            result[tank] = [
                require_number(
                    channel,
                    f"{name}.{tank} channel",
                    lower=lower,
                    upper=upper,
                    strict_lower=strict_lower,
                )
                for channel in channels
            ]
        return result

    if record["schema_version"] != "tank-colour-preset-resolution-v1":
        raise ValueError("tank_colour_resolution schema_version is invalid")
    if record["catalog_schema_version"] != "tank-colour-presets-v1":
        raise ValueError("tank_colour_resolution catalog_schema_version is invalid")
    if record["preset_name"] not in _TANK_COLOUR_PRESET_NAMES:
        raise ValueError("tank_colour_resolution preset_name is unknown")
    if record["colour_space"] != "linear_rgb_transmission":
        raise ValueError("tank_colour_resolution colour_space is invalid")
    reference_depth_m = require_number(
        record["reference_depth_m"], "reference_depth_m", lower=0.01, upper=0.01
    )
    transmission = require_rgb_map(
        record["transmission_1cm"],
        "transmission_1cm",
        exact_tanks=True,
        lower=0.0,
        upper=1.0,
        strict_lower=True,
    )
    overrides = require_rgb_map(
        record["transmission_overrides"],
        "transmission_overrides",
        exact_tanks=False,
        lower=0.0,
        upper=1.0,
        strict_lower=True,
    )
    scale = require_number(
        record["dye_strength_scale"],
        "dye_strength_scale",
        lower=0.5,
        upper=1.5,
    )
    clear_lock = record["t100_clear_lock"]
    randomize = record["randomize_around_preset"]
    if type(clear_lock) is not bool or type(randomize) is not bool:
        raise TypeError("tank-colour toggles must be Boolean")
    seed = record["randomization_seed"]
    if randomize:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("randomization_seed must be an integer when randomization is enabled")
    elif seed is not None:
        raise ValueError("randomization_seed must be null when randomization is disabled")

    def require_bounds(value: object, name: str, expected: list[float]) -> list[float]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a two-number list")
        normalized = [
            require_number(channel, name, lower=expected[0], upper=expected[1])
            for channel in value
        ]
        if normalized != expected:
            raise ValueError(f"{name} is invalid")
        return normalized

    tank_bounds = require_bounds(
        record["tank_strength_jitter_bounds"],
        "tank_strength_jitter_bounds",
        [-0.15, 0.15],
    )
    channel_bounds = require_bounds(
        record["channel_jitter_bounds"],
        "channel_jitter_bounds",
        [-0.05, 0.05],
    )
    tank_jitter_raw = record["tank_strength_jitter"]
    if not isinstance(tank_jitter_raw, Mapping) or set(tank_jitter_raw) != _TANK_KEYS:
        raise ValueError("tank_strength_jitter has an invalid tank set")
    tank_jitter = {
        tank: require_number(
            tank_jitter_raw[tank],
            f"tank_strength_jitter.{tank}",
            lower=tank_bounds[0],
            upper=tank_bounds[1],
        )
        for tank in DISPENSE_ORDER
    }
    channel_jitter = require_rgb_map(
        record["channel_jitter"],
        "channel_jitter",
        exact_tanks=True,
        lower=channel_bounds[0],
        upper=channel_bounds[1],
    )
    if not randomize and (
        any(value != 0.0 for value in tank_jitter.values())
        or any(value != 0.0 for row in channel_jitter.values() for value in row)
    ):
        raise ValueError("disabled randomization requires zero jitter")
    if clear_lock:
        if transmission["t100"] != [1.0, 1.0, 1.0]:
            raise ValueError("clear-locked t100 transmission must remain clear")
        if "t100" in overrides and overrides["t100"] != [1.0, 1.0, 1.0]:
            raise ValueError("clear-locked t100 override must remain clear")
        if tank_jitter["t100"] != 0.0 or channel_jitter["t100"] != [0.0, 0.0, 0.0]:
            raise ValueError("clear-locked t100 jitter must be zero")

    base = _TANK_COLOUR_BASE_TRANSMISSION_1CM[record["preset_name"]]
    expected_transmission: dict[str, list[float]] = {}
    expected_tank_jitter: dict[str, float] = {}
    expected_channel_jitter: dict[str, list[float]] = {}
    for tank in DISPENSE_ORDER:
        if clear_lock and tank == "t100":
            expected_transmission[tank] = [1.0, 1.0, 1.0]
            expected_tank_jitter[tank] = 0.0
            expected_channel_jitter[tank] = [0.0, 0.0, 0.0]
            continue
        strength_delta = 0.0
        channel_deltas = [0.0, 0.0, 0.0]
        if randomize:
            strength_delta = _tank_colour_jitter(
                record["preset_name"],
                seed,
                tank,
                "strength",
                lower=tank_bounds[0],
                upper=tank_bounds[1],
            )
            channel_deltas = [
                _tank_colour_jitter(
                    record["preset_name"],
                    seed,
                    tank,
                    "channel",
                    channel,
                    lower=channel_bounds[0],
                    upper=channel_bounds[1],
                )
                for channel in range(3)
            ]
        expected_tank_jitter[tank] = round(strength_delta, 9)
        expected_channel_jitter[tank] = [
            round(delta, 9) for delta in channel_deltas
        ]
        source = overrides.get(tank, list(base[tank]))
        expected_transmission[tank] = [
            round(
                math.exp(
                    -(-math.log(channel))
                    * scale
                    * (1.0 + strength_delta)
                    * (1.0 + channel_delta)
                ),
                9,
            )
            for channel, channel_delta in zip(
                source, channel_deltas, strict=True
            )
        ]
    if (
        tank_jitter != expected_tank_jitter
        or channel_jitter != expected_channel_jitter
        or transmission != expected_transmission
    ):
        raise ValueError(
            "tank_colour_resolution derived semantic fields mismatch"
        )
    label = record["free_text_label"]
    if label is not None:
        if not isinstance(label, str):
            raise TypeError("free_text_label must be text or null")
        if len(label) > 200 or any(
            unicodedata.category(char).startswith("C") for char in label
        ):
            raise ValueError("free_text_label is invalid")
    if record["interpretation"] != _TANK_COLOUR_INTERPRETATION:
        raise ValueError("tank_colour_resolution interpretation is invalid")
    if record["excluded_from_model_inputs"] is not True:
        raise ValueError("tank_colour_resolution must be excluded from model inputs")

    normalized = {
        "schema_version": record["schema_version"],
        "catalog_schema_version": record["catalog_schema_version"],
        "preset_name": record["preset_name"],
        "colour_space": record["colour_space"],
        "reference_depth_m": reference_depth_m,
        "transmission_1cm": transmission,
        "transmission_overrides": overrides,
        "dye_strength_scale": scale,
        "t100_clear_lock": clear_lock,
        "randomize_around_preset": randomize,
        "randomization_seed": seed,
        "tank_strength_jitter_bounds": tank_bounds,
        "channel_jitter_bounds": channel_bounds,
        "tank_strength_jitter": tank_jitter,
        "channel_jitter": channel_jitter,
        "free_text_label": label,
        "interpretation": record["interpretation"],
        "excluded_from_model_inputs": True,
    }
    semantic_hash = _canonical_sha256(normalized)
    renderer_hash = _canonical_sha256(transmission)
    for name in (
        "tank_colour_preset_hash",
        "renderer_palette_sha256",
        "palette_sha256",
    ):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if record["tank_colour_preset_hash"] != semantic_hash:
        raise ValueError("tank_colour_preset_hash mismatch")
    if record["renderer_palette_sha256"] != renderer_hash:
        raise ValueError("renderer_palette_sha256 mismatch")
    if record["palette_sha256"] != renderer_hash:
        raise ValueError("palette_sha256 must exactly alias renderer_palette_sha256")
    return {
        **deepcopy(normalized),
        "tank_colour_preset_hash": semantic_hash,
        "renderer_palette_sha256": renderer_hash,
        "palette_sha256": renderer_hash,
    }


def build_tank_colour_resolution(
    preset_name: str,
    *,
    transmission_overrides: Mapping[str, object] | None = None,
    dye_strength_scale: object = 1.0,
    t100_clear_lock: object = True,
    randomize_around_preset: object = False,
    randomization_seed: object = None,
    free_text_label: object = None,
    catalog_path: object = DEFAULT_CATALOG_PATH,
) -> dict[str, object]:
    """Resolve a named preset once for storage in a render plan."""

    catalog = load_tank_colour_preset_catalog(catalog_path)
    resolved = resolve_tank_colour_preset(
        catalog,
        preset_name,
        transmission_overrides=transmission_overrides,
        dye_strength_scale=dye_strength_scale,
        t100_clear_lock=t100_clear_lock,
        randomize_around_preset=randomize_around_preset,
        randomization_seed=randomization_seed,
        free_text_label=free_text_label,
    )
    return validate_tank_colour_resolution(resolved.to_json_record())


def run_palette_from_tank_colour_resolution(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Adapt one stored resolution to the renderer-facing run-palette shape."""

    resolved = validate_tank_colour_resolution(record)
    return {
        "schema_version": "fluidmix-run-palette-v1-tank-colour-resolution",
        "run_id": f"tank-colour-preset:{resolved['preset_name']}",
        "run_seed": resolved["randomization_seed"],
        "reference_depth_m": resolved["reference_depth_m"],
        "transmission_1cm": deepcopy(resolved["transmission_1cm"]),
        "variation_by_tank": {},
        "palette_sha256": resolved["palette_sha256"],
        "renderer_palette_sha256": resolved["renderer_palette_sha256"],
        "tank_colour_preset_hash": resolved["tank_colour_preset_hash"],
        "excluded_from_model_inputs": True,
        "interpretation": resolved["interpretation"],
    }


def bind_tank_colour_resolution_to_artifacts(
    artifacts: Mapping[str, object], record: Mapping[str, object]
) -> dict[str, object]:
    """Bind Part-A colour provenance without adding it to model inputs."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "render-schedule-plan.json",
        "model-input-plan.json",
        "label-provenance.json",
    }:
        raise ValueError("tank-colour artifact set is invalid")
    resolved = validate_tank_colour_resolution(record)
    result = deepcopy(dict(artifacts))
    for name in artifacts:
        if not isinstance(result[name], dict):
            raise TypeError(f"{name} must contain one object")

    plan = result["render-schedule-plan.json"]
    labels = result["label-provenance.json"]
    if plan.get("artifact_type") in _LEGACY_VARIED_PLAN_ARTIFACT_TYPES:
        raise ValueError("tank-colour resolution requires a genuinely new single-run plan")
    palette = run_palette_from_tank_colour_resolution(resolved)

    if "tank_colour_resolution" in plan or "run_palette" in plan:
        observed = validate_tank_colour_plan_overlay(plan)
        if observed != resolved:
            raise ValueError("existing tank-colour resolution disagrees with requested record")
        _validate_tank_colour_label_bindings(plan, labels, resolved)
        _assert_model_input_excludes_tank_colour(
            result["model-input-plan.json"]
        )
        return result

    runs = plan.get("runs")
    batches = plan.get("batches")
    schedules = plan.get("execution_schedules")
    label_rows = labels.get("vial_labels")
    if (
        not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(schedules, list)
        or not schedules
        or not isinstance(label_rows, list)
    ):
        raise ValueError("tank-colour resolution requires a genuinely new single-run plan")
    run = runs[0]
    batch = batches[0]
    if not isinstance(run, dict) or not isinstance(batch, dict):
        raise ValueError("tank-colour single-run bindings are invalid")
    case_ids = run.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or batch.get("case_ids") != case_ids
        or batch.get("run_id") != run.get("run_id")
    ):
        raise ValueError("tank-colour single-run case bindings are invalid")
    schedule_by_case: dict[str, dict[str, object]] = {}
    for schedule in schedules:
        binding = schedule.get("case_binding") if isinstance(schedule, dict) else None
        case_id = binding.get("case_id") if isinstance(binding, dict) else None
        if (
            not isinstance(case_id, str)
            or case_id in schedule_by_case
            or binding.get("run_id") != run.get("run_id")
            or binding.get("batch_id") != batch.get("batch_id")
        ):
            raise ValueError("tank-colour execution-schedule bindings are invalid")
        schedule_by_case[case_id] = schedule
    if set(schedule_by_case) != set(case_ids):
        raise ValueError("tank-colour execution schedules differ from run case IDs")
    label_by_case: dict[str, dict[str, object]] = {}
    for row in label_rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or case_id in label_by_case:
            raise ValueError("tank-colour label bindings are invalid")
        label_by_case[case_id] = row
    if set(label_by_case) != set(case_ids):
        raise ValueError("tank-colour labels differ from run case IDs")

    plan["tank_colour_resolution"] = deepcopy(resolved)
    plan["run_palette"] = deepcopy(palette)
    run["palette"] = deepcopy(palette)
    batch.pop("run_palette_id", None)
    batch.pop("run_palette_seed", None)
    batch["palette_sha256"] = palette["palette_sha256"]
    schedule_hashes: dict[str, str] = {}
    for case_id, schedule in schedule_by_case.items():
        schedule["case_binding"]["palette_sha256"] = palette["palette_sha256"]
        schedule_hashes[case_id] = _canonical_sha256(schedule)
        label_by_case[case_id]["render_schedule_sha256"] = schedule_hashes[case_id]
    plan["execution_schedule_sha256_by_case"] = schedule_hashes
    labels["tank_colour_resolution"] = deepcopy(resolved)
    validate_tank_colour_plan_overlay(plan)
    _validate_tank_colour_label_bindings(plan, labels, resolved)
    _assert_model_input_excludes_tank_colour(result["model-input-plan.json"])
    return result


def build_tank_colour_plan_artifacts(
    artifacts: Mapping[str, object], preset_name: str, **options: object
) -> dict[str, object]:
    """Resolve once and bind one named preset to a new artifact set."""

    record = build_tank_colour_resolution(preset_name, **options)
    return bind_tank_colour_resolution_to_artifacts(artifacts, record)


def validate_tank_colour_plan_overlay(
    plan: Mapping[str, object],
) -> dict[str, object] | None:
    """Validate a plan's optional resolution and renderer-palette overlay."""

    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a mapping")
    has_resolution = "tank_colour_resolution" in plan
    has_palette = "run_palette" in plan
    if has_resolution != has_palette:
        raise ValueError(
            "tank_colour_resolution and run_palette overlay are required together"
        )
    if not has_resolution:
        return None
    resolved = validate_tank_colour_resolution(plan["tank_colour_resolution"])
    expected_palette = run_palette_from_tank_colour_resolution(resolved)
    if plan["run_palette"] != expected_palette:
        raise ValueError("plan run_palette disagrees with tank_colour_resolution")
    runs = plan.get("runs")
    batches = plan.get("batches")
    schedules = plan.get("execution_schedules")
    if (
        not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
        or not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(batches[0], Mapping)
        or not isinstance(schedules, list)
        or not schedules
    ):
        raise ValueError("tank-colour overlay requires one run and one batch")
    run = runs[0]
    batch = batches[0]
    case_ids = run.get("case_ids")
    if (
        run.get("palette") != expected_palette
        or not isinstance(case_ids, list)
        or not case_ids
        or len(case_ids) != len(set(case_ids))
        or batch.get("case_ids") != case_ids
        or batch.get("run_id") != run.get("run_id")
        or batch.get("palette_sha256") != expected_palette["palette_sha256"]
        or "run_palette_id" in batch
        or "run_palette_seed" in batch
    ):
        raise ValueError("tank-colour run or batch palette binding is invalid")
    schedule_hashes: dict[str, str] = {}
    for schedule in schedules:
        binding = schedule.get("case_binding") if isinstance(schedule, Mapping) else None
        case_id = binding.get("case_id") if isinstance(binding, Mapping) else None
        if (
            not isinstance(case_id, str)
            or case_id in schedule_hashes
            or binding.get("run_id") != run.get("run_id")
            or binding.get("batch_id") != batch.get("batch_id")
            or binding.get("palette_sha256") != expected_palette["palette_sha256"]
        ):
            raise ValueError("tank-colour execution-schedule palette binding is invalid")
        schedule_hashes[case_id] = _canonical_sha256(schedule)
    if set(schedule_hashes) != set(case_ids) or plan.get(
        "execution_schedule_sha256_by_case"
    ) != schedule_hashes:
        raise ValueError("tank-colour execution-schedule hashes are invalid")
    return resolved


def _validate_tank_colour_label_bindings(
    plan: Mapping[str, object],
    labels: Mapping[str, object],
    resolved: Mapping[str, object],
) -> None:
    if labels.get("tank_colour_resolution") != resolved:
        raise ValueError("tank-colour label resolution binding is invalid")
    schedules = plan.get("execution_schedules")
    label_rows = labels.get("vial_labels")
    if not isinstance(schedules, list) or not isinstance(label_rows, list):
        raise ValueError("tank-colour schedule or label rows are invalid")
    expected = {
        schedule["case_binding"]["case_id"]: _canonical_sha256(schedule)
        for schedule in schedules
    }
    observed = {
        row.get("case_id"): row.get("render_schedule_sha256")
        for row in label_rows
        if isinstance(row, Mapping)
    }
    if len(observed) != len(label_rows) or observed != expected:
        raise ValueError("tank-colour label schedule hashes are invalid")


def _assert_model_input_excludes_tank_colour(value: object) -> None:
    def recursive_keys(node: object):
        if isinstance(node, Mapping):
            for key, child in node.items():
                yield key
                yield from recursive_keys(child)
        elif isinstance(node, list):
            for child in node:
                yield from recursive_keys(child)

    leaked = set(recursive_keys(value)) & _MODEL_INPUT_TANK_COLOUR_SENSITIVE_KEYS
    if leaked:
        raise ValueError(
            "model-input artifact contains tank-colour sensitive keys: "
            + ",".join(sorted(leaked))
        )


def build_run_design(
    *,
    run_id: str,
    run_seed: int,
    split_role: str,
    previous_recipe: Mapping[str, object],
    recipe_sequence: tuple[Mapping[str, object], ...]
    | list[Mapping[str, object]],
) -> dict[str, object]:
    """Expand a run-level palette and recipe sequence into vial case metadata."""

    if not isinstance(split_role, str) or not split_role.strip():
        raise ValueError("split_role must be a non-empty string")
    if not isinstance(previous_recipe, Mapping):
        raise TypeError("previous_recipe must be a mapping")
    if set(previous_recipe) != {"recipe_id", "programmed_ml"}:
        raise ValueError(
            "previous_recipe must contain exactly recipe_id and programmed_ml"
        )
    previous = _normalize_recipe(previous_recipe, include_vial_count=False)
    if isinstance(recipe_sequence, (str, bytes)) or not isinstance(
        recipe_sequence, (tuple, list)
    ):
        raise TypeError("recipe_sequence must be a tuple or list")
    if not recipe_sequence:
        raise ValueError("recipe_sequence must not be empty")
    recipes = [
        _normalize_recipe(recipe, include_vial_count=True)
        for recipe in recipe_sequence
    ]
    recipe_ids = [str(recipe["recipe_id"]) for recipe in recipes]
    if len(recipe_ids) != len(set(recipe_ids)):
        raise ValueError("recipe_id must be unique within a run")

    palette = build_run_palette(run_id=run_id, run_seed=run_seed)
    cases: list[dict[str, object]] = []
    predecessor = previous
    run_vial_ordinal = 0
    for recipe in recipes:
        for vial_ordinal in range(1, int(recipe["vial_count"]) + 1):
            run_vial_ordinal += 1
            cases.append(
                {
                    "case_id": (
                        f"{run_id}:{recipe['recipe_id']}:V{vial_ordinal:02d}"
                    ),
                    "run_id": run_id,
                    "split_role": split_role,
                    "palette_sha256": palette["palette_sha256"],
                    "recipe_id": recipe["recipe_id"],
                    "previous_recipe_id": predecessor["recipe_id"],
                    "vial_ordinal_in_recipe": vial_ordinal,
                    "vial_ordinal_in_run": run_vial_ordinal,
                    "programmed_ml": dict(recipe["programmed_ml"]),
                    "dispense_order": list(recipe["dispense_order"]),
                }
            )
        predecessor = recipe
    return {
        "schema_version": "fluidmix-run-design-v1",
        "run_id": run_id,
        "run_seed": run_seed,
        "split_role": split_role,
        "palette": palette,
        "previous_recipe": previous,
        "recipes": recipes,
        "cases": cases,
    }


def _normalize_recipe(
    recipe: Mapping[str, object], *, include_vial_count: bool
) -> dict[str, object]:
    if not isinstance(recipe, Mapping):
        raise TypeError("recipe must be a mapping")
    expected = {"recipe_id", "programmed_ml"}
    if include_vial_count:
        expected.add("vial_count")
    if set(recipe) != expected:
        raise ValueError(f"recipe must contain exactly {sorted(expected)}")
    recipe_id = recipe["recipe_id"]
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError("recipe_id must be a non-empty string")
    volumes = _normalized_volumes(recipe["programmed_ml"])
    normalized: dict[str, object] = {
        "recipe_id": recipe_id,
        "programmed_ml": {
            tank: volumes[tank] for tank in ("t100", "t200", "t400")
        },
        "dispense_order": list(active_dispense_order(volumes)),
    }
    if include_vial_count:
        vial_count = recipe["vial_count"]
        if (
            isinstance(vial_count, bool)
            or not isinstance(vial_count, int)
            or vial_count <= 0
        ):
            raise ValueError("vial_count must be a positive integer")
        normalized["vial_count"] = vial_count
    return normalized


def validate_whole_run_partitioning(
    run_designs: tuple[Mapping[str, object], ...]
    | list[Mapping[str, object]],
) -> None:
    """Reject case-level split drift and palette reuse between distinct runs."""

    if isinstance(run_designs, (str, bytes)) or not isinstance(
        run_designs, (tuple, list)
    ):
        raise TypeError("run_designs must be a tuple or list")
    if not run_designs:
        raise ValueError("run_designs must not be empty")
    run_ids: set[str] = set()
    palette_owner: dict[str, str] = {}
    for design in run_designs:
        if not isinstance(design, Mapping):
            raise TypeError("each run design must be a mapping")
        run_id = design.get("run_id")
        split_role = design.get("split_role")
        palette = design.get("palette")
        cases = design.get("cases")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run design run_id is invalid")
        if run_id in run_ids:
            raise ValueError("run_id must be unique in a cohort")
        run_ids.add(run_id)
        if not isinstance(split_role, str) or not split_role:
            raise ValueError("run design split_role is invalid")
        if not isinstance(palette, Mapping):
            raise ValueError("run design palette is invalid")
        palette_hash = palette.get("palette_sha256")
        if not isinstance(palette_hash, str) or len(palette_hash) != 64:
            raise ValueError("run design palette_sha256 is invalid")
        previous_owner = palette_owner.get(palette_hash)
        if previous_owner is not None and previous_owner != run_id:
            raise ValueError("palette reused across runs")
        palette_owner[palette_hash] = run_id
        if not isinstance(cases, list) or not cases:
            raise ValueError("run design cases are invalid")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("run design case is invalid")
            if case.get("run_id") != run_id:
                raise ValueError("every case must remain in a single run")
            if case.get("split_role") != split_role:
                raise ValueError("every run must remain in a single split")
            if case.get("palette_sha256") != palette_hash:
                raise ValueError("every case in a run must use one palette")
