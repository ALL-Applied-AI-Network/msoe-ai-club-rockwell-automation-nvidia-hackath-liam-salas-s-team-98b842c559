from __future__ import annotations

from pathlib import Path
import sys

import hashlib
import json
import math
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_catalog_has_strict_schema_and_named_presets():
    from fluidmix.tank_colour_presets import load_tank_colour_preset_catalog

    catalog = load_tank_colour_preset_catalog()

    assert catalog.schema_version == "tank-colour-presets-v1"
    assert set(catalog.presets) == {"base", "amber_teal", "crimson_indigo"}


def test_catalog_base_values_channels_and_tank_keys_are_exact():
    from fluidmix.tank_colour_presets import load_tank_colour_preset_catalog

    catalog = load_tank_colour_preset_catalog()
    base = catalog.presets["base"].transmission_1cm

    assert base == {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.95, 0.39, 0.26),
        "t400": (0.14, 0.61, 0.95),
    }
    assert catalog.presets["amber_teal"].transmission_1cm == {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.92, 0.48, 0.16),
        "t400": (0.12, 0.72, 0.68),
    }
    assert catalog.presets["crimson_indigo"].transmission_1cm == {
        "t100": (1.0, 1.0, 1.0),
        "t200": (0.88, 0.24, 0.22),
        "t400": (0.20, 0.36, 0.90),
    }
    for preset in catalog.presets.values():
        assert set(preset.transmission_1cm) == {"t100", "t200", "t400"}
        assert all(
            isinstance(channel, float) and math.isfinite(channel) and 0.0 < channel <= 1.0
            for rgb in preset.transmission_1cm.values()
            for channel in rgb
        )
    with pytest.raises(TypeError):
        catalog.presets["base"].transmission_1cm["t100"] = (0.5, 0.5, 0.5)


def test_clear_t100_lock_defaults_on_and_rejects_nonclear_override():
    from fluidmix.tank_colour_presets import (
        load_tank_colour_preset_catalog,
        resolve_tank_colour_preset,
    )

    catalog = load_tank_colour_preset_catalog()
    resolved = resolve_tank_colour_preset(catalog, "base")

    assert resolved.transmission_1cm["t100"] == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="clear"):
        resolve_tank_colour_preset(
            catalog,
            "base",
            transmission_overrides={"t100": (0.9, 1.0, 1.0)},
        )


def test_overrides_scale_and_extinction_randomization_are_bounded_and_deterministic():
    from fluidmix.tank_colour_presets import (
        load_tank_colour_preset_catalog,
        resolve_tank_colour_preset,
    )

    catalog = load_tank_colour_preset_catalog()
    overrides = {"t200": [0.8, 0.4, 0.2]}
    resolved = resolve_tank_colour_preset(
        catalog,
        "base",
        transmission_overrides=overrides,
        dye_strength_scale=0.5,
        randomize_around_preset=True,
        randomization_seed=7,
    )
    same_seed = resolve_tank_colour_preset(
        catalog,
        "base",
        transmission_overrides=overrides,
        dye_strength_scale=0.5,
        randomize_around_preset=True,
        randomization_seed=7,
    )
    different_seed = resolve_tank_colour_preset(
        catalog, "base", randomize_around_preset=True, randomization_seed=8
    )

    assert overrides == {"t200": [0.8, 0.4, 0.2]}
    assert resolved == same_seed
    assert resolved.transmission_1cm != different_seed.transmission_1cm
    assert resolved.transmission_1cm["t100"] == (1.0, 1.0, 1.0)
    assert resolved.tank_strength_jitter["t100"] == 0.0
    assert resolved.channel_jitter["t100"] == (0.0, 0.0, 0.0)
    assert all(-0.15 <= value <= 0.15 for value in resolved.tank_strength_jitter.values())
    assert all(
        -0.05 <= value <= 0.05
        for channels in resolved.channel_jitter.values()
        for value in channels
    )
    assert resolve_tank_colour_preset(
        catalog,
        "base",
        transmission_overrides={"t100": (0.8, 0.9, 1.0)},
        t100_clear_lock=False,
        dye_strength_scale=1.0,
    ).transmission_1cm["t100"] == (0.8, 0.9, 1.0)
    for scale in (0.5, 1.5):
        assert resolve_tank_colour_preset(catalog, "base", dye_strength_scale=scale)
    for bad_scale in (False, 0.49, 1.51, float("inf"), 10**1000):
        with pytest.raises((TypeError, ValueError)):
            resolve_tank_colour_preset(catalog, "base", dye_strength_scale=bad_scale)
    with pytest.raises(ValueError, match="requires"):
        resolve_tank_colour_preset(catalog, "base", randomization_seed=1)
    with pytest.raises(TypeError, match="integer"):
        resolve_tank_colour_preset(
            catalog, "base", randomize_around_preset=True, randomization_seed=True
        )


def test_provenance_label_is_short_text_without_unicode_controls():
    from fluidmix.tank_colour_presets import (
        load_tank_colour_preset_catalog,
        resolve_tank_colour_preset,
    )

    catalog = load_tank_colour_preset_catalog()
    assert resolve_tank_colour_preset(
        catalog, "base", free_text_label="bench \N{GREEK SMALL LETTER BETA}"
    ).free_text_label == "bench \N{GREEK SMALL LETTER BETA}"
    for label in ("line\nbreak", "tab\ttext", "nul\x00text", "bad\x7ftext", "x" * 201):
        with pytest.raises(ValueError):
            resolve_tank_colour_preset(catalog, "base", free_text_label=label)
    with pytest.raises(TypeError):
        resolve_tank_colour_preset(catalog, "base", free_text_label=42)


def test_canonical_hashes_alias_and_fresh_json_adapters_are_stable():
    from fluidmix.tank_colour_presets import (
        load_tank_colour_preset_catalog,
        renderer_palette_from_resolved,
        resolve_tank_colour_preset,
    )

    catalog = load_tank_colour_preset_catalog()
    resolved = resolve_tank_colour_preset(catalog, "base")
    labelled = resolve_tank_colour_preset(catalog, "base", free_text_label="alpha")
    record = resolved.to_json_record()
    palette = renderer_palette_from_resolved(resolved)

    canonical = json.dumps(
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "tank_colour_preset_hash",
                "renderer_palette_sha256",
                "palette_sha256",
            }
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert resolved.tank_colour_preset_hash == hashlib.sha256(canonical).hexdigest()
    assert resolved.tank_colour_preset_hash == (
        "0d0c78c53f14f1a2373cd74477403caef6617c2bdf7d6a6f13b2876235c52dac"
    )
    assert resolved.renderer_palette_sha256 == (
        "a8950d303e3184a7e95ee67a7345d75312369fc9fe5f567550c3fecbc5890932"
    )
    assert resolved.palette_sha256 == resolved.renderer_palette_sha256
    assert labelled.tank_colour_preset_hash != resolved.tank_colour_preset_hash
    assert labelled.renderer_palette_sha256 == resolved.renderer_palette_sha256
    assert palette["excluded_from_model_inputs"] is True
    assert record["excluded_from_model_inputs"] is True
    palette["transmission_1cm"]["t200"][0] = 0.01
    record["transmission_1cm"]["t400"][0] = 0.01
    assert renderer_palette_from_resolved(resolved)["transmission_1cm"]["t200"][0] == 0.95
    assert resolved.to_json_record()["transmission_1cm"]["t400"][0] == 0.14


def test_catalog_reader_rejects_malformed_and_nonphysical_json(tmp_path):
    from fluidmix.tank_colour_presets import (
        DEFAULT_CATALOG_PATH,
        load_tank_colour_preset_catalog,
    )

    payload = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    invalid = tmp_path / "invalid.json"
    for mutate in (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("schema_version"),
        lambda value: value.update({"reference_depth_m": 0.02}),
        lambda value: value["presets"].pop("base"),
        lambda value: value["presets"]["base"]["transmission_1cm"].update(
            {"t100": [True, 1.0, 1.0]}
        ),
    ):
        candidate = json.loads(json.dumps(payload))
        mutate(candidate)
        invalid.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            load_tank_colour_preset_catalog(invalid)

    invalid.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_tank_colour_preset_catalog(invalid)
    invalid.write_text('{"schema_version": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_tank_colour_preset_catalog(invalid)
    invalid.write_text('{"schema_version": 1e400}', encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_tank_colour_preset_catalog(invalid)


def test_resolver_rejects_invalid_input_types_and_unknown_tanks():
    from fluidmix.tank_colour_presets import (
        load_tank_colour_preset_catalog,
        resolve_tank_colour_preset,
    )

    catalog = load_tank_colour_preset_catalog()
    with pytest.raises(ValueError, match="unknown"):
        resolve_tank_colour_preset(catalog, "not-a-preset")
    for overrides in (
        {"unknown": (0.5, 0.5, 0.5)},
        {"t200": (0.5, 0.5)},
        {"t200": (0.5, True, 0.5)},
        {"t200": (0.5, 0.5, 0.0)},
    ):
        with pytest.raises((TypeError, ValueError)):
            resolve_tank_colour_preset(catalog, "base", transmission_overrides=overrides)
    for option, value in (("t100_clear_lock", 1), ("randomize_around_preset", "yes")):
        with pytest.raises(TypeError):
            resolve_tank_colour_preset(catalog, "base", **{option: value})
