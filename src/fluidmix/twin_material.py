"""Validation for synthetic USD material inputs.

These values are renderer parameters, not measured tank colours, concentrations or
carryover fractions.  This module deliberately has no USD or NumPy dependency so the
input contract stays testable in the ordinary CPU environment.
"""

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Sequence


_VARIANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _unit_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def validate_linear_rgb(values: Sequence[Real]) -> tuple[float, float, float]:
    """Return exactly three finite unit-interval renderer channels."""
    if isinstance(values, (str, bytes)):
        raise TypeError("material color must be a sequence of three real numbers")
    try:
        channels = tuple(values)
    except TypeError as exc:
        raise TypeError("material color must be a sequence of three real numbers") from exc
    if len(channels) != 3:
        raise ValueError("material color must contain exactly three channels")
    return tuple(_unit_float(value, f"material color channel {index}")
                 for index, value in enumerate(channels))


def parse_linear_rgb(text: str) -> tuple[float, float, float]:
    """Parse an ``R,G,B`` command-line value without clipping."""
    if not isinstance(text, str):
        raise TypeError("material color must be R,G,B text")
    parts = text.split(",")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise ValueError("material color must contain exactly three comma-separated values")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("material color channels must be numbers") from exc
    return validate_linear_rgb(values)


@dataclass(frozen=True)
class DirectMaterial:
    """Validated, unitless UsdPreviewSurface inputs for one neutral variant ID."""

    color: tuple[float, float, float]
    opacity: float
    variant_id: str

    def layer_metadata(self) -> dict[str, object]:
        color_text = ",".join(format(value, ".15g") for value in self.color)
        return {
            "fluidmix_variant_mode": "direct_material",
            "fluidmix_variant_id": self.variant_id,
            "fluidmix_material_color": color_text,
            "fluidmix_opacity": self.opacity,
        }


def make_direct_material(
    color: Sequence[Real], opacity: Real, variant_id: str
) -> DirectMaterial:
    """Validate a direct renderer material without deriving physical meaning."""
    if not isinstance(variant_id, str) or not _VARIANT_ID.fullmatch(variant_id):
        raise ValueError("variant ID must be 1-64 safe letters, digits, dots, dashes or underscores")
    return DirectMaterial(
        color=validate_linear_rgb(color),
        opacity=_unit_float(opacity, "opacity"),
        variant_id=variant_id,
    )


def select_material_mode(
    *,
    recipe: str | None,
    tanks: str | None,
    material_color: Sequence[Real] | None,
    opacity: Real | None,
    variant_id: str | None,
) -> tuple[str, DirectMaterial | None]:
    """Require one complete legacy recipe mode or one complete direct mode."""
    legacy_values = (recipe, tanks)
    direct_values = (material_color, opacity, variant_id)
    any_legacy = any(value is not None for value in legacy_values)
    all_legacy = all(value is not None for value in legacy_values)
    any_direct = any(value is not None for value in direct_values)
    all_direct = all(value is not None for value in direct_values)

    if any_legacy and any_direct:
        raise ValueError("recipe/tanks mode cannot be combined with direct material mode")
    if all_legacy:
        return "legacy_recipe", None
    if all_direct:
        return "direct_material", make_direct_material(material_color, opacity, variant_id)
    if any_legacy:
        raise ValueError("legacy mode requires both --recipe and --tanks")
    if any_direct:
        raise ValueError(
            "direct mode requires --material-color, --opacity and --variant-id"
        )
    raise ValueError("choose legacy --recipe/--tanks or direct material mode")
