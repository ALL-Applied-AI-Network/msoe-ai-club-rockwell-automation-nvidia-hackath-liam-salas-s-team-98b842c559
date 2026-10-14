"""CPU-only known-answer smoke using synthetic fixture values."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.absorbance import carryover_absorbance, fit_carryover_fraction, transmit


def main():
    tanks = {"t200": np.array([0.1, 0.6, 1.5]), "t400": np.array([1.4, 0.4, 0.05])}
    reference = np.array([0.8, 0.8, 0.75])
    new = {"t100": 20, "t400": 50}
    old = {"t100": 30, "t200": 10}
    measured = transmit(carryover_absorbance(new, old, tanks, 0.15), reference)
    fraction, residual = fit_carryover_fraction(measured, new, old, tanks, reference)
    print(f"old-fluid fraction {fraction:.6f}; RMS absorbance residual {residual:.3e}")
    if (not np.isfinite([fraction, residual]).all()
            or abs(fraction - 0.15) >= 1e-6 or residual >= 1e-6):
        raise SystemExit("synthetic smoke failed")
    print("synthetic smoke passed")


if __name__ == "__main__":
    main()
