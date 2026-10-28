"""Immutable synthetic tank-colour preset configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
from types import MappingProxyType
from collections.abc import Mapping
import unicodedata


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tank_colour_presets.json"
_TANKS = ("t100", "t200", "t400")
_PRESET_NAMES = frozenset(("base", "amber_teal", "crimson_indigo"))
_CLEAR_RGB = (1.0, 1.0, 1.0)
_RESOLVED_SCHEMA_VERSION = "tank-colour-preset-resolution-v1"
_INTERPRETATION = "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
_TANK_STRENGTH_JITTER_BOUNDS = (-0.15, 0.15)
_CHANNEL_JITTER_BOUNDS = (-0.05, 0.05)
_MAX_CATALOG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _TankColourPreset:
    description: str
    transmission_1cm: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class TankColourPresetCatalog:
    schema_version: str
    colour_space: str
    reference_depth_m: float
    presets: Mapping[str, _TankColourPreset]


@dataclass(frozen=True)
class ResolvedTankColourPreset:
    """One immutable, Part-A-only tank-colour selection."""

    schema_version: str
    catalog_schema_version: str
    preset_name: str
    colour_space: str
    reference_depth_m: float
    transmission_1cm: Mapping[str, tuple[float, float, float]]
    transmission_overrides: Mapping[str, tuple[float, float, float]]
    dye_strength_scale: float
    t100_clear_lock: bool
    randomize_around_preset: bool
    randomization_seed: int | None
    tank_strength_jitter_bounds: tuple[float, float]
    channel_jitter_bounds: tuple[float, float]
    tank_strength_jitter: Mapping[str, float]
    channel_jitter: Mapping[str, tuple[float, float, float]]
    free_text_label: str | None
    interpretation: str
    excluded_from_model_inputs: bool
    tank_colour_preset_hash: str
    renderer_palette_sha256: str
    palette_sha256: str

    def to_json_record(self) -> dict[str, object]:
        """Return a fresh JSON-compatible record for this resolution."""

        return _resolved_record(self)

    def as_json_record(self) -> dict[str, object]:
        """Compatibility spelling for callers needing a JSON-compatible record."""

        return self.to_json_record()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"JSON contains non-finite value {value!r}")


def _finite_float(value: object, name: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or not lower < result <= upper:
        raise ValueError(f"{name} must be finite and in ({lower}, {upper}]")
    return result


def _read_json(path: Path) -> object:
    try:
        lexical = os.lstat(path)
    except OSError as error:
        raise ValueError(f"catalog is unavailable: {path}") from error
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"catalog must be a regular non-symlink file: {path}")
    if lexical.st_size > _MAX_CATALOG_BYTES:
        raise ValueError(f"catalog exceeds {_MAX_CATALOG_BYTES} bytes: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"catalog is unavailable: {path}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_CATALOG_BYTES + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > _MAX_CATALOG_BYTES:
                raise ValueError(f"catalog exceeds {_MAX_CATALOG_BYTES} bytes: {path}")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"catalog is unavailable: {path}") from error
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or len(raw) != after.st_size:
        raise ValueError(f"catalog changed while reading: {path}")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"catalog is invalid JSON: {path}") from error


def load_tank_colour_preset_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> TankColourPresetCatalog:
    """Load and validate the immutable synthetic preset catalog."""

    payload = _read_json(Path(path))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "colour_space", "reference_depth_m", "presets"
    }:
        raise ValueError("catalog root has unexpected fields")
    if payload["schema_version"] != "tank-colour-presets-v1":
        raise ValueError("catalog schema_version is invalid")
    if payload["colour_space"] != "linear_rgb_transmission":
        raise ValueError("catalog colour_space is invalid")
    reference_depth_m = _finite_float(payload["reference_depth_m"], "reference_depth_m")
    if reference_depth_m != 0.01:
        raise ValueError("catalog reference_depth_m must be exactly 0.01")
    raw_presets = payload["presets"]
    if not isinstance(raw_presets, dict) or set(raw_presets) != _PRESET_NAMES:
        raise ValueError("catalog presets must be base, amber_teal, and crimson_indigo")
    presets: dict[str, _TankColourPreset] = {}
    for name, raw_preset in raw_presets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("preset name must be non-empty text")
        if not isinstance(raw_preset, dict) or set(raw_preset) != {"description", "transmission_1cm"}:
            raise ValueError(f"preset {name!r} has unexpected fields")
        if not isinstance(raw_preset["description"], str):
            raise TypeError(f"preset {name!r} description must be text")
        raw_transmission = raw_preset["transmission_1cm"]
        if not isinstance(raw_transmission, dict) or set(raw_transmission) != set(_TANKS):
            raise ValueError(f"preset {name!r} must contain exactly {_TANKS}")
        transmission: dict[str, tuple[float, float, float]] = {}
        for tank in _TANKS:
            rgb = raw_transmission[tank]
            if not isinstance(rgb, list) or len(rgb) != 3:
                raise ValueError(f"preset {name!r} {tank} must have three channels")
            transmission[tank] = tuple(
                _finite_float(channel, f"preset {name!r} {tank} channel")
                for channel in rgb
            )
        presets[name] = _TankColourPreset(
            description=raw_preset["description"],
            transmission_1cm=MappingProxyType(transmission),
        )
    return TankColourPresetCatalog(
        schema_version="tank-colour-presets-v1",
        colour_space="linear_rgb_transmission",
        reference_depth_m=reference_depth_m,
        presets=MappingProxyType(presets),
    )


def resolve_tank_colour_preset(
    catalog: TankColourPresetCatalog,
    preset_name: str,
    *,
    transmission_overrides: Mapping[str, object] | None = None,
    dye_strength_scale: object = 1.0,
    t100_clear_lock: object = True,
    randomize_around_preset: object = False,
    randomization_seed: object = None,
    free_text_label: object = None,
) -> ResolvedTankColourPreset:
    """Resolve a deterministic synthetic tank palette without renderer side effects."""

    if not isinstance(catalog, TankColourPresetCatalog):
        raise TypeError("catalog must be a TankColourPresetCatalog")
    if not isinstance(preset_name, str):
        raise TypeError("preset_name must be text")
    try:
        selected = catalog.presets[preset_name]
    except KeyError as error:
        raise ValueError(f"unknown tank-colour preset {preset_name!r}") from error
    if type(t100_clear_lock) is not bool:
        raise TypeError("t100_clear_lock must be Boolean")
    if type(randomize_around_preset) is not bool:
        raise TypeError("randomize_around_preset must be Boolean")
    scale = _finite_inclusive_float(
        dye_strength_scale, "dye_strength_scale", lower=0.5, upper=1.5
    )
    label = _validate_label(free_text_label)
    overrides = _normalize_overrides(transmission_overrides)
    if t100_clear_lock and "t100" in overrides and overrides["t100"] != _CLEAR_RGB:
        raise ValueError("t100 clear lock rejects a non-clear override")
    seed = _validate_seed(randomize_around_preset, randomization_seed)

    source = {
        tank: overrides.get(tank, selected.transmission_1cm[tank]) for tank in _TANKS
    }
    final: dict[str, tuple[float, float, float]] = {}
    tank_jitter: dict[str, float] = {}
    channel_jitter: dict[str, tuple[float, float, float]] = {}
    for tank in _TANKS:
        if t100_clear_lock and tank == "t100":
            final[tank] = _CLEAR_RGB
            tank_jitter[tank] = 0.0
            channel_jitter[tank] = (0.0, 0.0, 0.0)
            continue
        strength_delta = 0.0
        channel_deltas = (0.0, 0.0, 0.0)
        if randomize_around_preset:
            strength_delta = _jitter(
                preset_name,
                seed,
                tank,
                "strength",
                bounds=_TANK_STRENGTH_JITTER_BOUNDS,
            )
            channel_deltas = tuple(
                _jitter(
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
        channel_jitter[tank] = tuple(round(delta, 9) for delta in channel_deltas)
        final[tank] = tuple(
            round(
                math.exp(
                    -(-math.log(channel))
                    * scale
                    * (1.0 + strength_delta)
                    * (1.0 + channel_delta)
                ),
                9,
            )
            for channel, channel_delta in zip(source[tank], channel_deltas, strict=True)
        )
    return _build_resolved(
        catalog=catalog,
        preset_name=preset_name,
        transmission=final,
        overrides=overrides,
        dye_strength_scale=scale,
        t100_clear_lock=t100_clear_lock,
        randomize_around_preset=randomize_around_preset,
        randomization_seed=seed,
        tank_strength_jitter=tank_jitter,
        channel_jitter=channel_jitter,
        free_text_label=label,
    )


def renderer_palette_from_resolved(resolved: ResolvedTankColourPreset) -> dict[str, object]:
    """Adapt a resolved preset to a fresh renderer palette dictionary."""

    if not isinstance(resolved, ResolvedTankColourPreset):
        raise TypeError("resolved must be a ResolvedTankColourPreset")
    return {
        "reference_depth_m": resolved.reference_depth_m,
        "transmission_1cm": _json_transmission(resolved.transmission_1cm),
        "renderer_palette_sha256": resolved.renderer_palette_sha256,
        "palette_sha256": resolved.palette_sha256,
        "excluded_from_model_inputs": True,
    }


def _finite_inclusive_float(value: object, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def _normalize_overrides(value: Mapping[str, object] | None) -> dict[str, tuple[float, float, float]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("transmission_overrides must be a mapping or None")
    if not set(value).issubset(_TANKS):
        raise ValueError("transmission_overrides contains an unknown tank")
    normalized: dict[str, tuple[float, float, float]] = {}
    for tank, rgb in value.items():
        if isinstance(rgb, (str, bytes)) or not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
            raise ValueError(f"override {tank} must have three channels")
        normalized[tank] = tuple(
            _finite_float(channel, f"override {tank} channel") for channel in rgb
        )
    return normalized


def _validate_seed(enabled: bool, seed: object) -> int | None:
    if not enabled:
        if seed is not None:
            raise ValueError("randomization_seed requires randomize_around_preset=True")
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("randomization_seed must be a non-Boolean integer")
    return seed


def _validate_label(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("free_text_label must be text or None")
    if len(value) > 200:
        raise ValueError("free_text_label must contain at most 200 Unicode characters")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("free_text_label must not contain control characters")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _jitter(
    preset_name: str,
    seed: int | None,
    tank: str,
    kind: str,
    channel: int | None = None,
    *,
    bounds: tuple[float, float],
) -> float:
    payload = ["fluidmix-tank-colour-jitter-v1", preset_name, seed, tank, kind]
    if channel is not None:
        payload.append(channel)
    unit = int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest()[:8], "big") / float(1 << 64)
    lower, upper = bounds
    return lower + (upper - lower) * unit


def _build_resolved(
    *,
    catalog: TankColourPresetCatalog,
    preset_name: str,
    transmission: Mapping[str, tuple[float, float, float]],
    overrides: Mapping[str, tuple[float, float, float]],
    dye_strength_scale: float,
    t100_clear_lock: bool,
    randomize_around_preset: bool,
    randomization_seed: int | None,
    tank_strength_jitter: Mapping[str, float],
    channel_jitter: Mapping[str, tuple[float, float, float]],
    free_text_label: str | None,
) -> ResolvedTankColourPreset:
    immutable_transmission = MappingProxyType(dict(transmission))
    immutable_overrides = MappingProxyType(dict(overrides))
    immutable_tank_jitter = MappingProxyType(dict(tank_strength_jitter))
    immutable_channel_jitter = MappingProxyType(dict(channel_jitter))
    provisional = ResolvedTankColourPreset(
        schema_version=_RESOLVED_SCHEMA_VERSION,
        catalog_schema_version=catalog.schema_version,
        preset_name=preset_name,
        colour_space=catalog.colour_space,
        reference_depth_m=catalog.reference_depth_m,
        transmission_1cm=immutable_transmission,
        transmission_overrides=immutable_overrides,
        dye_strength_scale=dye_strength_scale,
        t100_clear_lock=t100_clear_lock,
        randomize_around_preset=randomize_around_preset,
        randomization_seed=randomization_seed,
        tank_strength_jitter_bounds=_TANK_STRENGTH_JITTER_BOUNDS,
        channel_jitter_bounds=_CHANNEL_JITTER_BOUNDS,
        tank_strength_jitter=immutable_tank_jitter,
        channel_jitter=immutable_channel_jitter,
        free_text_label=free_text_label,
        interpretation=_INTERPRETATION,
        excluded_from_model_inputs=True,
        tank_colour_preset_hash="",
        renderer_palette_sha256="",
        palette_sha256="",
    )
    semantic = _resolved_semantic_record(provisional)
    renderer_hash = _canonical_sha256(semantic["transmission_1cm"])
    return ResolvedTankColourPreset(
        **{
            **provisional.__dict__,
            "tank_colour_preset_hash": _canonical_sha256(semantic),
            "renderer_palette_sha256": renderer_hash,
            "palette_sha256": renderer_hash,
        }
    )


def _json_transmission(value: Mapping[str, tuple[float, float, float]]) -> dict[str, list[float]]:
    return {tank: list(value[tank]) for tank in _TANKS}


def _resolved_semantic_record(resolved: ResolvedTankColourPreset) -> dict[str, object]:
    return {
        "schema_version": resolved.schema_version,
        "catalog_schema_version": resolved.catalog_schema_version,
        "preset_name": resolved.preset_name,
        "colour_space": resolved.colour_space,
        "reference_depth_m": resolved.reference_depth_m,
        "transmission_1cm": _json_transmission(resolved.transmission_1cm),
        "transmission_overrides": _json_transmission_subset(resolved.transmission_overrides),
        "dye_strength_scale": resolved.dye_strength_scale,
        "t100_clear_lock": resolved.t100_clear_lock,
        "randomize_around_preset": resolved.randomize_around_preset,
        "randomization_seed": resolved.randomization_seed,
        "tank_strength_jitter_bounds": list(resolved.tank_strength_jitter_bounds),
        "channel_jitter_bounds": list(resolved.channel_jitter_bounds),
        "tank_strength_jitter": {tank: resolved.tank_strength_jitter[tank] for tank in _TANKS},
        "channel_jitter": _json_transmission(resolved.channel_jitter),
        "free_text_label": resolved.free_text_label,
        "interpretation": resolved.interpretation,
        "excluded_from_model_inputs": resolved.excluded_from_model_inputs,
    }


def _json_transmission_subset(value: Mapping[str, tuple[float, float, float]]) -> dict[str, list[float]]:
    return {tank: list(value[tank]) for tank in _TANKS if tank in value}


def _resolved_record(resolved: ResolvedTankColourPreset) -> dict[str, object]:
    result = _resolved_semantic_record(resolved)
    result.update(
        {
            "tank_colour_preset_hash": resolved.tank_colour_preset_hash,
            "renderer_palette_sha256": resolved.renderer_palette_sha256,
            "palette_sha256": resolved.palette_sha256,
        }
    )
    return result
