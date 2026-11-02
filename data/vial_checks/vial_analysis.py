"""Resumable SAM 2 vial-fill analysis for the Rockwell vial videos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE_VERSION = "3"
RESULT_REQUIRED_COLUMNS = {
    "video_name",
    "frame_sampled",
    "source_frame",
    "video_time_s",
    "status",
    "reference_basis",
    "fill_height_change_px",
    "fill_height_px",
    "fill_height_change_smoothed_px",
    "fill_height_smoothed_px",
    "vial_cycle_id",
    "config_fingerprint",
}


@dataclass(frozen=True)
class AnalysisConfig:
    video_dir: Path
    work_dir: Path = Path("batch_work")
    results_dir: Path = Path("batch_results")
    stride: int = 3
    model_id: str = "facebook/sam2.1-hiera-small"
    vial_box: tuple[float, float, float, float] = (300.0, 60.0, 360.0, 420.0)
    reference_frame_count: int = 15
    minimum_reference_frames: int = 3
    verified_empty_frames: dict[str, tuple[int, ...]] = field(default_factory=dict)
    require_verified_reference: bool = False
    de_threshold: float = 6.0
    gap_rows: int = 8
    dark_l: float = 35.0
    min_row_pixels: int = 5
    min_vial_pixels: int = 500
    large_shift_px: float = 3.0
    smoothing_window: int = 7
    downward_jump_px: float = 8.0
    cycle_gap_frames: int = 5
    mm_per_pixel: float | None = None
    volume_coefficients: tuple[float, ...] | None = None
    offload_state_to_cpu: bool = True
    resume: bool = True
    diagnostic_frames: int = 5

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.reference_frame_count < 1:
            raise ValueError("reference_frame_count must be at least 1")
        if self.minimum_reference_frames < 2:
            raise ValueError("minimum_reference_frames must be at least 2")
        for video_name, indices in self.verified_empty_frames.items():
            if not video_name or not indices or any(index < 0 for index in indices):
                raise ValueError("verified_empty_frames must map video names to nonnegative frame indices")
        if self.gap_rows < 1:
            raise ValueError("gap_rows must be at least 1")
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.cycle_gap_frames < 1:
            raise ValueError("cycle_gap_frames must be at least 1")
        if len(self.vial_box) != 4:
            raise ValueError("vial_box must contain x0, y0, x1, y1")
        if self.mm_per_pixel is not None and self.mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be positive")
        if self.volume_coefficients is not None and self.mm_per_pixel is None:
            raise ValueError("volume_coefficients require mm_per_pixel")
        if self.volume_coefficients is not None and (
            not self.volume_coefficients
            or not np.isfinite(np.asarray(self.volume_coefficients, dtype=float)).all()
        ):
            raise ValueError("volume_coefficients must be nonempty finite numbers")


@dataclass(frozen=True)
class ReferenceProfile:
    row_lab: np.ndarray
    usable_rows: np.ndarray
    bottom_y: int
    center_x: float
    frame_count: int


@dataclass(frozen=True)
class ReferenceProvenance:
    basis: str
    requested_indices: tuple[int, ...]
    used_indices: tuple[int, ...]


@dataclass(frozen=True)
class VideoFrames:
    video_path: Path
    frame_dir: Path
    source_fps: float
    source_frame_count: int
    sampled_frame_count: int
    width: int
    height: int
    stride: int
    source_frame_indices: list[int]
    sample_times_s: list[float]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["video_path"] = str(self.video_path)
        result["frame_dir"] = str(self.frame_dir)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VideoFrames":
        values = values.copy()
        values["video_path"] = Path(values["video_path"])
        values["frame_dir"] = Path(values["frame_dir"])
        return cls(**values)


def _config_payload(config: AnalysisConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("video_dir", "work_dir", "results_dir", "resume"):
        payload.pop(key, None)
    payload["pipeline_version"] = PIPELINE_VERSION
    return payload


def configuration_fingerprint(config: AnalysisConfig) -> str:
    payload = json.dumps(_config_payload(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()[:16]


def _video_signature(video_path: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_medians(lab_image: np.ndarray, mask: np.ndarray, min_row_pixels: int = 5) -> np.ndarray:
    if lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must have shape (height, width, 3)")
    if mask.shape != lab_image.shape[:2]:
        raise ValueError("mask and image dimensions differ")
    rows = np.full((mask.shape[0], 3), np.nan, dtype=np.float32)
    for y in np.flatnonzero(mask.sum(axis=1) >= min_row_pixels):
        rows[y] = np.median(lab_image[y][mask[y]], axis=0)
    return rows


def build_reference(
    lab_images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray | None],
    *,
    dark_l: float = 35.0,
    min_row_pixels: int = 5,
    min_vial_pixels: int = 500,
) -> ReferenceProfile:
    profiles: list[np.ndarray] = []
    centers: list[float] = []
    for lab_image, mask in zip(lab_images, masks, strict=True):
        if mask is None or int(np.asarray(mask).sum()) < min_vial_pixels:
            continue
        profiles.append(row_medians(lab_image, mask.astype(bool), min_row_pixels))
        centers.append(float(np.where(mask)[1].mean()))
    if not profiles:
        raise ValueError("no valid masks are available for the empty-vial reference")
    stacked = np.stack(profiles)
    row_lab = np.full(stacked.shape[1:], np.nan, dtype=np.float32)
    finite_rows = np.isfinite(stacked[:, :, 0]).any(axis=0)
    row_lab[finite_rows] = np.nanmedian(stacked[:, finite_rows], axis=0).astype(np.float32)
    usable_rows = np.isfinite(row_lab[:, 0]) & (row_lab[:, 0] > dark_l)
    if not usable_rows.any():
        raise ValueError("empty-vial reference has no usable rows above DARK_L")
    return ReferenceProfile(
        row_lab=row_lab,
        usable_rows=usable_rows,
        bottom_y=int(np.flatnonzero(usable_rows).max()),
        center_x=float(np.median(centers)),
        frame_count=len(profiles),
    )


def _append_flag(current: Any, new_flag: str) -> str:
    flags = [part for part in str(current or "").split(";") if part]
    if new_flag not in flags:
        flags.append(new_flag)
    return ";".join(flags)


def measure_frame(
    lab_image: np.ndarray,
    mask: np.ndarray | None,
    reference: ReferenceProfile,
    *,
    de_threshold: float = 6.0,
    gap_rows: int = 8,
    min_row_pixels: int = 5,
    min_vial_pixels: int = 500,
    large_shift_px: float = 3.0,
) -> dict[str, Any]:
    if mask is None or int(np.asarray(mask).sum()) < min_vial_pixels:
        return {"status": "no_vial", "quality_flags": "no_vial"}
    mask = np.asarray(mask, dtype=bool)
    current = row_medians(lab_image, mask, min_row_pixels)
    center_x = float(np.where(mask)[1].mean())
    shift = center_x - reference.center_x
    flags = ""
    if abs(shift) > large_shift_px:
        flags = _append_flag(flags, "vial_shift_large")

    finite_current = np.isfinite(current[:, 0])
    usable = reference.usable_rows & finite_current
    if not usable.any():
        return {
            "status": "no_vial",
            "vial_center_x": center_x,
            "vial_shift_px": shift,
            "quality_flags": _append_flag(flags, "no_usable_rows"),
        }
    bottom = min(reference.bottom_y, int(np.flatnonzero(usable).max()))
    delta_e = np.linalg.norm(current - reference.row_lab, axis=1)

    top: int | None = None
    gap = 0
    for y in range(bottom, -1, -1):
        if not usable[y] or not np.isfinite(delta_e[y]):
            continue
        if delta_e[y] > de_threshold:
            top, gap = y, 0
        else:
            gap += 1
            if gap >= gap_rows:
                break

    base = {
        "vial_center_x": center_x,
        "vial_shift_px": shift,
        "mask_area_px": int(mask.sum()),
        "quality_flags": flags,
    }
    if top is None:
        return {**base, "status": "empty", "fill_height_px": 0.0}

    liquid_rows = [
        y
        for y in range(top, bottom + 1)
        if usable[y] and np.isfinite(delta_e[y]) and delta_e[y] > de_threshold
    ]
    if not liquid_rows:
        return {**base, "status": "empty", "fill_height_px": 0.0}
    color = np.median(current[liquid_rows], axis=0)
    return {
        **base,
        "status": "ok",
        "fill_top_y": top,
        "fill_bottom_y": bottom,
        "fill_height_px": float(bottom - top + 1),
        "L": float(color[0]),
        "a": float(color[1]),
        "b": float(color[2]),
        "dE_median": float(np.median(delta_e[liquid_rows])),
        "liquid_row_count": len(liquid_rows),
    }


def postprocess_measurements(
    measurements: pd.DataFrame,
    *,
    smoothing_window: int = 7,
    downward_jump_px: float = 8.0,
    cycle_gap_frames: int = 5,
) -> pd.DataFrame:
    result = measurements.copy()
    if "quality_flags" not in result:
        result["quality_flags"] = ""
    result["quality_flags"] = result["quality_flags"].fillna("").astype(str)
    heights = pd.to_numeric(result["fill_height_px"], errors="coerce")
    change_heights = pd.to_numeric(
        result["fill_height_change_px"] if "fill_height_change_px" in result else heights,
        errors="coerce",
    )
    statuses = result["status"] if "status" in result else pd.Series("ok", index=result.index)
    cycle_ids: list[Any] = []
    current_cycle = 0
    missing_run = 0
    seen_vial = False
    for status in statuses:
        if status in {"no_vial", "reference_unavailable"}:
            cycle_ids.append(pd.NA)
            missing_run += 1
            continue
        if seen_vial and missing_run >= cycle_gap_frames:
            current_cycle += 1
        cycle_ids.append(current_cycle)
        seen_vial = True
        missing_run = 0
    result["vial_cycle_id"] = pd.array(cycle_ids, dtype="Int64")
    result["fill_height_change_smoothed_px"] = np.nan
    result["fill_height_change_monotonic_px"] = np.nan
    result["fill_height_smoothed_px"] = np.nan
    result["fill_height_monotonic_px"] = np.nan
    drop_rows = pd.Series(False, index=result.index)
    for _, indices in result.groupby("vial_cycle_id", dropna=True).groups.items():
        group_heights = heights.loc[indices]
        group_changes = change_heights.loc[indices]
        change_smoothed = group_changes.rolling(
            smoothing_window, center=True, min_periods=1
        ).median()
        absolute_smoothed = group_heights.rolling(
            smoothing_window, center=True, min_periods=1
        ).median()
        result.loc[indices, "fill_height_change_smoothed_px"] = change_smoothed
        result.loc[indices, "fill_height_change_monotonic_px"] = change_smoothed.cummax()
        result.loc[indices, "fill_height_smoothed_px"] = absolute_smoothed
        result.loc[indices, "fill_height_monotonic_px"] = absolute_smoothed.cummax()
        drop_rows.loc[indices] = group_changes.diff() < -abs(downward_jump_px)
    for index in result.index[drop_rows.fillna(False)]:
        result.at[index, "quality_flags"] = _append_flag(
            result.at[index, "quality_flags"], "downward_jump"
        )
    return result


def apply_calibration(
    measurements: pd.DataFrame,
    *,
    mm_per_pixel: float | None,
    volume_coefficients: Sequence[float] | None,
) -> pd.DataFrame:
    result = measurements.copy()
    if mm_per_pixel is None:
        if volume_coefficients is not None:
            raise ValueError("volume_coefficients require mm_per_pixel")
        result["fill_height_mm"] = np.nan
        result["volume_ml"] = np.nan
        result["calibration_status"] = "uncalibrated"
        return result
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive")
    result["fill_height_mm"] = result["fill_height_smoothed_px"] * mm_per_pixel
    if volume_coefficients is None:
        result["volume_ml"] = np.nan
        result["calibration_status"] = "height_calibrated"
    else:
        coefficients = np.asarray(tuple(volume_coefficients), dtype=float)
        if coefficients.size == 0 or not np.isfinite(coefficients).all():
            raise ValueError("volume_coefficients must be nonempty finite numbers")
        result["volume_ml"] = np.polyval(coefficients, result["fill_height_mm"])
        finite_heights = result["fill_height_mm"].notna()
        if not np.isfinite(result.loc[finite_heights, "volume_ml"]).all():
            raise ValueError("volume calibration produced non-finite values")
        result["calibration_status"] = "calibrated"
    return result


def _frame_cache_key(video_path: Path, stride: int) -> dict[str, Any]:
    return {"video": _video_signature(video_path), "stride": stride}


def extract_frames(video_path: Path | str, frame_dir: Path | str, *, stride: int = 3) -> VideoFrames:
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    frame_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = frame_dir / "frames.json"
    cache_key = _frame_cache_key(video_path, stride)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            cached = metadata.get("cache_key") == cache_key
            frame_info = metadata.get("frames")
            if cached and frame_info:
                result = VideoFrames.from_dict(frame_info)
                expected_names = {
                    f"{index:06d}.jpg" for index in range(result.sampled_frame_count)
                }
                actual_paths = list(frame_dir.glob("*.jpg"))
                exact_names = {path.name for path in actual_paths} == expected_names
                correct_directory = result.frame_dir.resolve() == frame_dir.resolve()
                readable = exact_names and all(
                    (image := cv2.imread(str(path))) is not None
                    and image.shape[:2] == (result.height, result.width)
                    for path in actual_paths
                )
                if exact_names and correct_directory and readable:
                    return result
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    for old_frame in frame_dir.glob("*.jpg"):
        old_frame.unlink()
    metadata_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"video reports invalid FPS: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_indices: list[int] = []
    sample_times: list[float] = []
    source_index = 0
    sample_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                output_path = frame_dir / f"{sample_index:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"failed to write frame: {output_path}")
                source_indices.append(source_index)
                sample_times.append(source_index / fps)
                sample_index += 1
            source_index += 1
    finally:
        capture.release()
    if not source_indices:
        raise RuntimeError(f"video contains no readable frames: {video_path}")
    result = VideoFrames(
        video_path=video_path.resolve(),
        frame_dir=frame_dir.resolve(),
        source_fps=fps,
        source_frame_count=source_index,
        sampled_frame_count=len(source_indices),
        width=width,
        height=height,
        stride=stride,
        source_frame_indices=source_indices,
        sample_times_s=sample_times,
    )
    _atomic_write_text(
        metadata_path,
        json.dumps({"cache_key": cache_key, "frames": result.to_dict()}, indent=2),
    )
    return result


def load_predictor(model_id: str):
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    return SAM2VideoPredictor.from_pretrained(model_id)


def track_vial(predictor, frames: VideoFrames, config: AnalysisConfig) -> dict[int, np.ndarray]:
    import torch

    masks: dict[int, np.ndarray] = {}
    box = np.asarray(config.vial_box, dtype=np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frames.frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=config.offload_state_to_cpu,
        )
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
        for frame_index, object_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=frames.sampled_frame_count - 1,
        ):
            ids = [int(value) for value in object_ids]
            if 1 not in ids:
                continue
            object_index = ids.index(1)
            masks[int(frame_index)] = (logits[object_index, 0] > 0).cpu().numpy()
    return masks


def _read_lab(frame_dir: Path, index: int) -> np.ndarray:
    path = frame_dir / f"{index:06d}.jpg"
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"cannot read extracted frame: {path}")
    srgb = bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(srgb, cv2.COLOR_BGR2LAB)


def _reference_selection(
    video_name: str, frames: VideoFrames, config: AnalysisConfig
) -> tuple[list[int], str]:
    explicit = config.verified_empty_frames.get(video_name)
    if explicit is None:
        explicit = config.verified_empty_frames.get(Path(video_name).stem)
    if explicit is not None:
        indices = [index for index in explicit if index < frames.sampled_frame_count]
        return indices, "verified_empty"
    if config.require_verified_reference:
        return [], "reference_unavailable"
    return (
        list(range(min(config.reference_frame_count, frames.sampled_frame_count))),
        "relative_initial_unverified",
    )


def _base_measurement(video_name: str, frames: VideoFrames, index: int) -> dict[str, Any]:
    return {
        "video_name": video_name,
        "frame_sampled": index,
        "source_frame": frames.source_frame_indices[index],
        "video_time_s": frames.sample_times_s[index],
        "source_fps": frames.source_fps,
        "stride": frames.stride,
    }


def measurements_from_masks(
    video_name: str,
    frames: VideoFrames,
    masks: dict[int, np.ndarray],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, ReferenceProfile | None, ReferenceProvenance]:
    requested_indices, reference_basis = _reference_selection(video_name, frames, config)
    reference_indices = [
        index
        for index in requested_indices
        if index in masks and int(masks[index].sum()) >= config.min_vial_pixels
    ]
    reference: ReferenceProfile | None = None
    if len(reference_indices) >= config.minimum_reference_frames:
        try:
            reference = build_reference(
                [_read_lab(frames.frame_dir, index) for index in reference_indices],
                [masks[index] for index in reference_indices],
                dark_l=config.dark_l,
                min_row_pixels=config.min_row_pixels,
                min_vial_pixels=config.min_vial_pixels,
            )
        except ValueError:
            reference = None
    if reference is None:
        reference_basis = "reference_unavailable"
        used_reference_indices: tuple[int, ...] = ()
    else:
        used_reference_indices = tuple(reference_indices)
    provenance = ReferenceProvenance(
        basis=reference_basis,
        requested_indices=tuple(requested_indices),
        used_indices=used_reference_indices,
    )
    rows: list[dict[str, Any]] = []
    for index in range(frames.sampled_frame_count):
        base = _base_measurement(video_name, frames, index)
        if reference is None:
            rows.append(
                {
                    **base,
                    "status": "reference_unavailable",
                    "reference_basis": reference_basis,
                    "quality_flags": "reference_unavailable",
                    "fill_height_change_px": np.nan,
                    "fill_height_px": np.nan,
                }
            )
            continue
        measurement = measure_frame(
            _read_lab(frames.frame_dir, index),
            masks.get(index),
            reference,
            de_threshold=config.de_threshold,
            gap_rows=config.gap_rows,
            min_row_pixels=config.min_row_pixels,
            min_vial_pixels=config.min_vial_pixels,
            large_shift_px=config.large_shift_px,
        )
        change_height = measurement.get("fill_height_px", np.nan)
        measurement["fill_height_change_px"] = change_height
        measurement["reference_basis"] = reference_basis
        if reference_basis != "verified_empty":
            measurement["fill_height_px"] = np.nan
            if measurement["status"] == "ok":
                measurement["status"] = "relative_change"
            elif measurement["status"] == "empty":
                measurement["status"] = "no_change_from_reference"
        rows.append({**base, **measurement})
    result = postprocess_measurements(
        pd.DataFrame(rows),
        smoothing_window=config.smoothing_window,
        downward_jump_px=config.downward_jump_px,
        cycle_gap_frames=config.cycle_gap_frames,
    )
    result = apply_calibration(
        result,
        mm_per_pixel=config.mm_per_pixel,
        volume_coefficients=config.volume_coefficients,
    )
    result["config_fingerprint"] = configuration_fingerprint(config)
    return result, reference, provenance


def save_diagnostic(
    output_path: Path,
    frames: VideoFrames,
    masks: dict[int, np.ndarray],
    measurements: pd.DataFrame,
    *,
    count: int = 5,
) -> None:
    count = max(1, min(count, frames.sampled_frame_count))
    indices = np.unique(np.linspace(0, frames.sampled_frame_count - 1, count).astype(int))
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4), squeeze=False)
    for axis, index in zip(axes[0], indices, strict=True):
        bgr = cv2.imread(str(frames.frame_dir / f"{index:06d}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = masks.get(int(index))
        if mask is not None:
            rgb[mask] = (0.7 * rgb[mask] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
        row = measurements.iloc[int(index)]
        if pd.notna(row.get("fill_top_y")):
            axis.axhline(float(row["fill_top_y"]), color="magenta", linewidth=1.5)
        axis.imshow(rgb)
        axis.set_title(f"{row['video_time_s']:.1f}s | {row['status']}")
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    fig.savefig(temporary, dpi=140, format="png")
    plt.close(fig)
    temporary.replace(output_path)


def _result_paths(video_path: Path, config: AnalysisConfig) -> tuple[Path, Path, Path]:
    csv_path = config.results_dir / "per_video" / f"{video_path.stem}.csv"
    metadata_path = config.results_dir / "per_video" / f"{video_path.stem}.json"
    diagnostic_path = config.results_dir / "diagnostics" / f"{video_path.stem}.png"
    return csv_path, metadata_path, diagnostic_path


def _result_is_current(video_path: Path, config: AnalysisConfig) -> bool:
    csv_path, metadata_path, diagnostic_path = _result_paths(video_path, config)
    if (
        not config.resume
        or not csv_path.is_file()
        or not metadata_path.is_file()
        or not diagnostic_path.is_file()
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_fingerprint") != configuration_fingerprint(config):
            return False
        if metadata.get("video_signature") != _video_signature(video_path):
            return False
        if metadata.get("csv_sha256") != _file_sha256(csv_path):
            return False
        if metadata.get("diagnostic_sha256") != _file_sha256(diagnostic_path):
            return False
        measurements = pd.read_csv(csv_path)
        if len(measurements) != metadata.get("row_count"):
            return False
        if not RESULT_REQUIRED_COLUMNS.issubset(measurements.columns):
            return False
        expected_fingerprint = configuration_fingerprint(config)
        return measurements["config_fingerprint"].astype(str).eq(expected_fingerprint).all()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def analyze_video(
    video_path: Path,
    config: AnalysisConfig,
    predictor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.monotonic()
    csv_path, metadata_path, diagnostic_path = _result_paths(video_path, config)
    frame_dir = config.work_dir / "frames" / video_path.stem
    frames = extract_frames(video_path, frame_dir, stride=config.stride)
    masks = track_vial(predictor, frames, config)
    measurements, reference, provenance = measurements_from_masks(
        video_path.name, frames, masks, config
    )
    _atomic_write_csv(measurements, csv_path)
    save_diagnostic(
        diagnostic_path,
        frames,
        masks,
        measurements,
        count=config.diagnostic_frames,
    )
    reference_basis = provenance.basis
    metadata = {
        "video_signature": _video_signature(video_path),
        "config_fingerprint": configuration_fingerprint(config),
        "sampled_frames": frames.sampled_frame_count,
        "source_frames": frames.source_frame_count,
        "source_fps": frames.source_fps,
        "reference_frames": reference.frame_count if reference is not None else 0,
        "reference_basis": reference_basis,
        "requested_reference_basis": _reference_selection(video_path.name, frames, config)[1],
        "requested_reference_frame_indices": list(provenance.requested_indices),
        "used_reference_frame_indices": list(provenance.used_indices),
        "row_count": len(measurements),
        "required_columns": sorted(RESULT_REQUIRED_COLUMNS),
        "csv_sha256": _file_sha256(csv_path),
        "diagnostic_sha256": _file_sha256(diagnostic_path),
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
    manifest = {
        "video_name": video_path.name,
        "status": "completed",
        "sampled_frames": frames.sampled_frame_count,
        "measured_rows": len(measurements),
        "ok_rows": int((measurements["status"] == "ok").sum()),
        "relative_rows": int((measurements["status"] == "relative_change").sum()),
        "reference_basis": reference_basis,
        "scientifically_usable": reference_basis == "verified_empty",
        "elapsed_s": time.monotonic() - started,
        "error": "",
    }
    return measurements, manifest


def _write_manifest(rows: list[dict[str, Any]], config: AnalysisConfig) -> pd.DataFrame:
    manifest = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(manifest, config.results_dir / "processing_manifest.csv")
    return manifest


def analyze_batch(
    config: AnalysisConfig,
    video_paths: Iterable[Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_paths = config.video_dir.glob("*.mp4") if video_paths is None else video_paths
    paths = sorted(Path(path) for path in source_paths)
    if not paths:
        raise ValueError(f"no MP4 videos found in {config.video_dir}")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    predictor = None
    all_measurements: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, video_path in enumerate(paths, start=1):
        video_started = time.monotonic()
        try:
            if _result_is_current(video_path, config):
                csv_path, _, _ = _result_paths(video_path, config)
                measurements = pd.read_csv(csv_path)
                manifest = {
                    "video_name": video_path.name,
                    "status": "resumed",
                    "sampled_frames": len(measurements),
                    "measured_rows": len(measurements),
                    "ok_rows": int((measurements["status"] == "ok").sum()),
                    "relative_rows": int((measurements["status"] == "relative_change").sum()),
                    "reference_basis": str(measurements["reference_basis"].iloc[0]),
                    "scientifically_usable": str(measurements["reference_basis"].iloc[0]) == "verified_empty",
                    "elapsed_s": 0.0,
                    "error": "",
                }
            else:
                if predictor is None:
                    predictor = load_predictor(config.model_id)
                print(f"[{index}/{len(paths)}] analyzing {video_path.name}", flush=True)
                measurements, manifest = analyze_video(video_path, config, predictor)
            all_measurements.append(measurements)
        except Exception as error:
            manifest = {
                "video_name": video_path.name,
                "status": "failed",
                "sampled_frames": 0,
                "measured_rows": 0,
                "ok_rows": 0,
                "relative_rows": 0,
                "reference_basis": "failed",
                "scientifically_usable": False,
                "elapsed_s": time.monotonic() - video_started,
                "error": f"{type(error).__name__}: {error}",
            }
            error_dir = config.results_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(error_dir / f"{video_path.stem}.txt", traceback.format_exc())
            print(f"[{index}/{len(paths)}] FAILED {video_path.name}: {error}", flush=True)
        manifest_rows.append(manifest)
        _write_manifest(manifest_rows, config)
    combined = pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
    _atomic_write_csv(combined, config.results_dir / "combined_measurements.csv")
    manifest = _write_manifest(manifest_rows, config)
    summary = {
        "config_fingerprint": configuration_fingerprint(config),
        "video_count": len(paths),
        "completed_or_resumed": int(manifest["status"].isin(["completed", "resumed"]).sum()),
        "failed": int((manifest["status"] == "failed").sum()),
        "measurement_rows": len(combined),
        "scientifically_usable_videos": int(manifest["scientifically_usable"].fillna(False).sum()),
        "relative_only_videos": int((manifest["reference_basis"] == "relative_initial_unverified").sum()),
    }
    _atomic_write_text(config.results_dir / "summary.json", json.dumps(summary, indent=2))
    return combined, manifest


def preview_vial_box(video_path: Path | str, vial_box: Sequence[float]):
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame: {video_path}")
    x0, y0, x1, y1 = (int(value) for value in vial_box)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(rgb)
    axis.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2)
    )
    axis.set_title(Path(video_path).name)
    return fig, axis

