"""Measure each vial's final-mixture colour from the vial_fill video (cv2 only, no ffmpeg).

Measured: for every frame of a vial's final-mixture window (data/real_check_windows.csv), the
median RGB of the most saturated pixels in a fixed crop of the fill station, converted to CIE Lab
(D65, same formula as scripts/color.py). The vial's colour is the per-channel median of those
per-frame Lab values; its measurement noise is the median CIE76 distance between consecutive
frames.

Assumed (chosen by eye earlier in the project, not fitted to any outcome): the crop rows 150:430,
columns 180:420 contain the vial at the fill station; the liquid is the top 0.7 % most saturated
pixels of that crop; sRGB camera values are close enough to sRGB for the Lab formula.
"""
import numpy as np

CROP_ROWS = slice(150, 430)
CROP_COLS = slice(180, 420)
TOP_SATURATED_PERCENT = 0.7

_RGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805],
                        [0.2126, 0.7152, 0.0722],
                        [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def lab(rgb):
    """sRGB (0-255, last axis) -> CIE Lab (D65); identical to scripts/color.py's lab()."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _RGB_TO_XYZ.T / _D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def cie76(lab_a, lab_b):
    """Euclidean distance in Lab (the CIE76 colour difference)."""
    return float(np.linalg.norm(np.asarray(lab_a, np.float64) - np.asarray(lab_b, np.float64)))


def liquid_rgb(frame_rgb):
    """Median RGB of the top 0.7 % most saturated pixels in the fill-station crop of one frame."""
    pixels = np.asarray(frame_rgb, np.float32)[CROP_ROWS, CROP_COLS]
    mx, mn = pixels.max(-1), pixels.min(-1)
    saturation = (mx - mn) / (mx + 1)
    mask = saturation > np.percentile(saturation, 100 - TOP_SATURATED_PERCENT)
    if not mask.any():
        raise ValueError("no liquid pixels selected")
    return np.median(pixels[mask], 0)


def window_labs(cap, start_frame, end_frame):
    """Per-frame Lab for frames start_frame..end_frame inclusive (same frame index as the other
    two cameras of video 5). Frames are read sequentially after one seek so every frame counts."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = []
    for index in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"could not read frame {index}")
        out.append(lab(liquid_rgb(frame[..., ::-1])))  # cv2 gives BGR
    return np.array(out)


def summarise_window(labs):
    """(median Lab over frames, median consecutive-frame CIE76). One frame gives noise 0.0."""
    labs = np.asarray(labs, np.float64)
    measured = np.median(labs, 0)
    if len(labs) < 2:
        return measured, 0.0
    steps = np.linalg.norm(np.diff(labs, axis=0), axis=1)
    return measured, float(np.median(steps))


def window_diagnostics(labs):
    """Extra, not used by the rule: fraction of consecutive-frame steps that are exactly zero (the
    median-pixel colour is integer-quantised, so most steps are 0) and the peak-to-peak Lab range."""
    labs = np.asarray(labs, np.float64)
    steps = np.linalg.norm(np.diff(labs, axis=0), axis=1) if len(labs) > 1 else np.array([])
    return {"zero_step_fraction": round(float((steps == 0).mean()), 3) if len(steps) else None,
            "window_lab_ptp": [round(float(x), 2) for x in np.ptp(labs, 0)]}


def measure_vials(video_path, windows):
    """windows: list of dicts with vial, eligible (bool), start_frame, end_frame.
    Returns {vial: {measured_lab, measurement_noise_dE, window_frames}} for eligible vials."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    result = {}
    try:
        for w in windows:
            if not w["eligible"]:
                continue
            labs = window_labs(cap, w["start_frame"], w["end_frame"])
            measured, noise = summarise_window(labs)
            result[w["vial"]] = {
                "measured_lab": [round(float(x), 2) for x in measured],
                "measurement_noise_dE": round(noise, 3),
                "window_frames": list(range(w["start_frame"], w["end_frame"] + 1)),
                "diagnostics": window_diagnostics(labs),
            }
    finally:
        cap.release()
    return result
