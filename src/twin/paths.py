"""Repository paths owned by the standalone twin package."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


__all__ = ["REPO"]
