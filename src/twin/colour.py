"""Twin-owned colour math for synthetic recipe and visibility planning."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np


TANKS = ("t100", "t200", "t400")
EPS = 1e-4


def srgb_to_linear(rgb255):
    c = np.asarray(rgb255, dtype=float) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def linear_to_srgb(lin):
    c = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    s = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return s * 255.0


def srgb_to_lab(rgb255):
    lin = srgb_to_linear(rgb255)
    transform = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )
    xyz = lin @ transform.T / np.array([0.95047, 1.0, 1.08883])
    converted = np.where(
        xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116
    )
    return np.stack(
        [
            116 * converted[..., 1] - 16,
            500 * (converted[..., 0] - converted[..., 1]),
            200 * (converted[..., 1] - converted[..., 2]),
        ],
        -1,
    )


def delta_e76(left: Sequence[object], right: Sequence[object]) -> float:
    """Apply the twin's established scalar sqrt/sum CIE76 formula."""

    if len(left) != 3 or len(right) != 3:
        raise ValueError("CIE Lab descriptors must contain three channels")
    return math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right, strict=True)
        )
    )


def delta_e(rgb_a, rgb_b):
    return float(np.linalg.norm(srgb_to_lab(rgb_a) - srgb_to_lab(rgb_b)))


def absorbance(i_lin, i0_lin):
    intensity = np.maximum(np.asarray(i_lin, float), EPS)
    baseline = np.maximum(np.asarray(i0_lin, float), EPS)
    return -np.log10(intensity / baseline)


def transmit(value, i0_lin):
    return np.asarray(i0_lin, float) * 10.0 ** (-np.asarray(value, float))


def volume_fractions(ml: dict) -> dict:
    unknown = set(ml) - set(TANKS)
    if unknown:
        raise ValueError(f"unknown recipe tanks: {sorted(unknown)}")
    volumes = {tank: float(ml.get(tank) or 0) for tank in TANKS}
    for tank, volume in volumes.items():
        if not np.isfinite(volume) or volume < 0:
            raise ValueError(f"{tank} volume must be finite and nonnegative")
    total = sum(volumes.values())
    if not np.isfinite(total):
        raise ValueError("total recipe volume must be finite")
    if total <= 0:
        raise ValueError("recipe has no volume")
    return {tank: volumes[tank] / total for tank in TANKS}


def mix_absorbance(ml: dict, tank_a: dict) -> np.ndarray:
    fractions = volume_fractions(ml)
    result = np.zeros(3)
    for tank in TANKS:
        if fractions[tank] == 0:
            continue
        if tank != "t100" and tank not in tank_a:
            raise ValueError(f"missing absorbance calibration for {tank}")
        value = np.asarray(tank_a.get(tank, (0, 0, 0)), float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{tank} absorbance must be a finite R/G/B vector")
        result += fractions[tank] * value
    return result


def expected_color(ml: dict, tank_a: dict, i0_lin) -> np.ndarray:
    return transmit(mix_absorbance(ml, tank_a), i0_lin)


__all__ = [
    "absorbance",
    "delta_e",
    "delta_e76",
    "expected_color",
    "linear_to_srgb",
    "srgb_to_lab",
]
