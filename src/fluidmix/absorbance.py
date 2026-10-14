"""Expected vial color from tank colors and mL, using absorbance (Beer-Lambert).

Why absorbance and not RGB:
    Dye in water absorbs light. The light that passes through is
        I = I0 * 10 ** (-A),   A = epsilon * c * d        (Beer-Lambert)
    A is the absorbance, c is the dye concentration, d is the path length.
    When you mix dyes, the absorbances ADD. The RGB values do NOT add.
    So: convert to absorbance, mix there, convert back.

Path length in this line:
    The camera looks at the vial from the side. The light path d is the vial
    diameter, and it does not change with fill level. So only the volume
    FRACTION of each tank matters, not the total mL:
        A_mix[ch] = sum_i (V_i / V_total) * A_tank_i[ch]
    A_tank_i is the absorbance of pure tank-i fluid, seen through one vial.
    T-100 (clear water) has A = 0, so it only dilutes.

Channels:
    We use 3 camera channels (R, G, B) as 3 coarse "wavelengths". This is an
    approximation: real dyes absorb over a spectrum, and a camera channel
    averages over a band. Expect errors for very dark mixes.

All functions take and return LINEAR RGB (not sRGB). Use srgb_to_linear()
on pixel values from images first.
"""
from __future__ import annotations

import numpy as np

TANKS = ("t100", "t200", "t400")
EPS = 1e-4  # floor for linear intensity, so log10 never sees 0


# ---------------------------------------------------------------- color space
def srgb_to_linear(rgb255):
    """sRGB 0-255 pixel values -> linear 0-1 intensity."""
    c = np.asarray(rgb255, dtype=float) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def linear_to_srgb(lin):
    """Linear 0-1 intensity -> sRGB 0-255 (float)."""
    c = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    s = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return s * 255.0


def srgb_to_lab(rgb255):
    """sRGB 0-255 -> CIE Lab (D65). Used for Delta E."""
    lin = srgb_to_linear(rgb255)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(rgb_a, rgb_b):
    """CIE76 Delta E between two sRGB colors. About 2 = just visible, >10 = clear."""
    return float(np.linalg.norm(srgb_to_lab(rgb_a) - srgb_to_lab(rgb_b)))


# ------------------------------------------------------------ Beer-Lambert core
def absorbance(i_lin, i0_lin):
    """A = -log10(I / I0), per channel. I0 = the same view through clear water."""
    i = np.maximum(np.asarray(i_lin, float), EPS)
    i0 = np.maximum(np.asarray(i0_lin, float), EPS)
    return -np.log10(i / i0)


def transmit(a, i0_lin):
    """Inverse of absorbance(): I = I0 * 10**(-A)."""
    return np.asarray(i0_lin, float) * 10.0 ** (-np.asarray(a, float))


def volume_fractions(ml: dict) -> dict:
    """{'t100': 10, 't200': 10, 't400': 10} -> fractions that sum to 1."""
    unknown = set(ml) - set(TANKS)
    if unknown:
        raise ValueError(f"unknown recipe tanks: {sorted(unknown)}")
    volumes = {t: float(ml.get(t) or 0) for t in TANKS}
    for tank, volume in volumes.items():
        if not np.isfinite(volume) or volume < 0:
            raise ValueError(f"{tank} volume must be finite and nonnegative")
    total = sum(volumes.values())
    if not np.isfinite(total):
        raise ValueError("total recipe volume must be finite")
    if total <= 0:
        raise ValueError("recipe has no volume")
    return {t: volumes[t] / total for t in TANKS}


def mix_absorbance(ml: dict, tank_a: dict) -> np.ndarray:
    """Absorbance of a mix. tank_a[t] = 3-vector absorbance of pure tank t.

    T-100 defaults to clear. Used colored tanks require finite calibration vectors.
    """
    f = volume_fractions(ml)
    result = np.zeros(3)
    for tank in TANKS:
        if f[tank] == 0:
            continue
        if tank != "t100" and tank not in tank_a:
            raise ValueError(f"missing absorbance calibration for {tank}")
        value = np.asarray(tank_a.get(tank, (0, 0, 0)), float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{tank} absorbance must be a finite R/G/B vector")
        result += f[tank] * value
    return result


def expected_color(ml: dict, tank_a: dict, i0_lin) -> np.ndarray:
    """Expected linear RGB of a vial filled with recipe `ml`."""
    return transmit(mix_absorbance(ml, tank_a), i0_lin)


# ------------------------------------------------------------------- carryover
def carryover_absorbance(new_ml: dict, old_ml: dict, tank_a: dict, frac_old: float):
    """Absorbance of "new recipe + frac_old of the old recipe's mix".

    frac_old is the fraction of the vial volume that is old fluid (0..1).
    Each recipe's A is already concentration-like, so the mix is linear in it.
    """
    a_new = mix_absorbance(new_ml, tank_a)
    a_old = mix_absorbance(old_ml, tank_a)
    return (1 - frac_old) * a_new + frac_old * a_old


def fit_carryover_fraction(measured_lin, new_ml, old_ml, tank_a, i0_lin):
    """Best frac_old (0..1) that explains a measured vial as new + old.

    Closed-form 1-D least squares in absorbance space:
        A_meas - A_new = frac * (A_old - A_new)
    Returns (frac_old, rms_residual_absorbance). A large residual means
    "not carryover" (something else is wrong).
    """
    a_meas = absorbance(measured_lin, i0_lin)
    a_new = mix_absorbance(new_ml, tank_a)
    a_old = mix_absorbance(old_ml, tank_a)
    d = a_old - a_new
    denom = float(d @ d)
    if denom < 1e-9:
        return 0.0, float(np.sqrt(np.mean((a_meas - a_new) ** 2)))
    frac = float(np.clip((a_meas - a_new) @ d / denom, 0.0, 1.0))
    resid = a_meas - (a_new + frac * d)
    return frac, float(np.sqrt(np.mean(resid ** 2)))


# ------------------------------------------------------------------ calibration
def fit_tank_absorbance(rows):
    """Fit pure-tank absorbance and I0 from clean vials with known recipes.

    rows: list of (ml_dict, measured_linear_rgb).
    Model per channel:  log10(I) = log10(I0) - f200*A200 - f400*A400
    That is a linear regression with an intercept, so plain least squares works.
    Needs at least 3 vials with different f200/f400 ratios.

    Returns (tank_a, i0_lin, r2_per_channel).
    """
    X, Y = [], []
    for ml, lin in rows:
        f = volume_fractions(ml)
        X.append([1.0, -f["t200"], -f["t400"]])
        lin = np.asarray(lin, float)
        if lin.shape != (3,) or not np.isfinite(lin).all() or (lin < 0).any():
            raise ValueError("calibration intensity must be a finite nonnegative R/G/B vector")
        Y.append(np.log10(np.maximum(lin, EPS)))
    X, Y = np.array(X), np.array(Y)
    if len(X) < 3 or np.linalg.matrix_rank(X) < 3:
        raise ValueError("calibration recipe design must have full rank (3)")
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)  # shape (3 params, 3 channels)
    pred = X @ coef
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    i0 = 10 ** coef[0]
    tank_a = {"t100": np.zeros(3), "t200": coef[1], "t400": coef[2]}
    return tank_a, i0, r2


def tank_absorbance_from_image(tank_lin, clear_lin, tank_path_mm, vial_path_mm=26.4):
    """Tank-camera route: absorbance of pure tank fluid, rescaled to vial width.

    A scales with path length d, so A_vial = A_tank * (vial_path / tank_path).
    clear_lin is the same view through T-100 (clear), as I0.
    tank_path_mm is NOT known yet (we do not have tank dimensions) - measure it.
    """
    return absorbance(tank_lin, clear_lin) * (vial_path_mm / tank_path_mm)
