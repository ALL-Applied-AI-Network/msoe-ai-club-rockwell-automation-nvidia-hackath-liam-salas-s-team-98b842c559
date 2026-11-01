"""Synthetic checks for the INN-52 liquid-level evaluation helpers."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vial_level import fit_line, volume_errors


def test_fit_line_recovers_rows_per_ml():
    ml = np.array([10, 20, 30, 40])
    a, b = fit_line(ml, 380 - 3.3 * ml)
    assert np.isclose(a, 380) and np.isclose(b, 3.3)


def test_volume_errors_are_zero_on_a_perfect_line_with_held_out_recipes():
    readings = [{"recipe": str(r), "tank": "T100", "ml": m, "row": 380 - 3.3 * m, "zone": "below_holder"}
                for r, m in enumerate([10, 20, 30, 40, 25])]
    assert all(e < 1e-9 for _, e in volume_errors(readings))
