"""Colour-math ring."""

from __future__ import annotations

from collections.abc import Sequence
import math


def delta_e76(left: Sequence[object], right: Sequence[object]) -> float:
    """Apply the package's established scalar sqrt/sum CIE76 formula."""

    if len(left) != 3 or len(right) != 3:
        raise ValueError("CIE Lab descriptors must contain three channels")
    return math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right, strict=True)
        )
    )


__all__ = ["delta_e76"]
