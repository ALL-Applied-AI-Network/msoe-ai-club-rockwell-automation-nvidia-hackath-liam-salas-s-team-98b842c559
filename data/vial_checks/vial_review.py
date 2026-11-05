"""Human review, calibration, and validation helpers for vial analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _frame_path(frames_root: Path, video_name: str, frame_sampled: int) -> Path:
    return frames_root / Path(video_name).stem / f"{int(frame_sampled):06d}.jpg"


def _require_frame_paths(frame: pd.DataFrame, frames_root: Path) -> pd.DataFrame:
    result = frame.copy()
    result["frame_path"] = [
        str(_frame_path(frames_root, video, sampled))
        for video, sampled in zip(result.video_name, result.frame_sampled, strict=True)
    ]
    missing = [path for path in result.frame_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing extracted frame: {missing[0]}")
    return result


def make_empty_reference_candidates(
    measurements: pd.DataFrame,
    frames_root: Path | str,
    *,
    count_per_video: int = 5,
) -> pd.DataFrame:
    if count_per_video < 3:
        raise ValueError("count_per_video must be at least 3")
    required = {"video_name", "frame_sampled", "video_time_s", "status"}
    if not required.issubset(measurements.columns):
        raise ValueError(f"measurements missing columns: {sorted(required - set(measurements.columns))}")
    groups: list[pd.DataFrame] = []
    excluded = {"no_vial", "reference_unavailable"}
    for _, video_rows in measurements.sort_values("frame_sampled").groupby("video_name", sort=True):
        eligible = video_rows[~video_rows.status.isin(excluded)]
        if len(eligible) < min(3, len(video_rows)):
            eligible = video_rows
        groups.append(eligible.head(count_per_video))
    if not groups:
        return pd.DataFrame()
    candidates = pd.concat(groups, ignore_index=True)
    keep = [
        column
        for column in (
            "video_name", "frame_sampled", "source_frame", "video_time_s", "status",
            "fill_height_change_px", "vial_cycle_id",
        )
        if column in candidates
    ]
    candidates = _require_frame_paths(candidates[keep], Path(frames_root))
    candidates["candidate_rank"] = candidates.groupby("video_name").cumcount() + 1
    candidates["verified_empty"] = pd.array([pd.NA] * len(candidates), dtype="boolean")
    candidates["reviewer_notes"] = ""
    return candidates


def _pick_stratified(video_rows: pd.DataFrame, count: int) -> pd.DataFrame:
    sort_columns = ["frame_sampled"]
    if "source_frame" in video_rows.columns:
        sort_columns.append("source_frame")
    video_rows = video_rows.sort_values(sort_columns, kind="stable")
    chosen: list[int] = []
    status_priority = [
        "no_change_from_reference", "relative_change", "no_vial", "reference_unavailable"
    ]
    statuses = status_priority + sorted(set(video_rows.status) - set(status_priority))
    for status in statuses:
        group = video_rows[video_rows.status == status]
        if group.empty or len(chosen) >= count:
            continue
        chosen.append(int(group.index[len(group) // 2]))
    if len(chosen) < count:
        remaining = video_rows.drop(index=chosen)
        if not remaining.empty:
            positions = np.linspace(0, len(remaining) - 1, min(count - len(chosen), len(remaining)))
            chosen.extend(int(remaining.index[position]) for position in np.unique(positions.astype(int)))
    return video_rows.loc[chosen[:count]].sort_values("frame_sampled")


def make_validation_sample(
    measurements: pd.DataFrame,
    frames_root: Path | str,
    *,
    count_per_video: int = 6,
) -> pd.DataFrame:
    if count_per_video < 1:
        raise ValueError("count_per_video must be positive")
    required = {"video_name", "frame_sampled", "video_time_s", "status"}
    if not required.issubset(measurements.columns):
        raise ValueError(f"measurements missing columns: {sorted(required - set(measurements.columns))}")
    sampled = [
        _pick_stratified(rows, count_per_video)
        for _, rows in measurements.groupby("video_name", sort=True)
    ]
    result = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    result = _require_frame_paths(result, Path(frames_root))
    for column in ("manual_fill_top_y", "manual_fill_bottom_y", "known_volume_ml"):
        result[column] = np.nan
    result["label_status"] = "unlabeled"
    result["reviewer_notes"] = ""
    return result


def _normalize_video_name(name: str, video_names: Iterable[str]) -> str:
    names = list(video_names)
    if name in names:
        return name
    matches = [candidate for candidate in names if Path(candidate).stem == Path(name).stem]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown or ambiguous video name: {name}")


def _exact_frame_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"frame index must be an exact integer, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frame index must be an exact integer, got {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"frame index must be an exact integer, got {value!r}")
    return int(numeric)


def parse_verified_empty_flags(candidates: pd.DataFrame) -> dict[str, list[int]]:
    """Strictly parse reviewer confirmations without treating typos as false."""
    required = {"video_name", "frame_sampled", "verified_empty"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidate sheet missing columns: {sorted(required - set(candidates.columns))}")
    truthy = {"true", "1", "yes"}
    falsey = {"false", "0", "no"}
    confirmed: dict[str, list[int]] = {}
    for row in candidates.itertuples(index=False):
        raw = getattr(row, "verified_empty")
        token = "" if pd.isna(raw) else str(raw).strip().lower()
        if not token or token in falsey:
            continue
        if token not in truthy:
            raise ValueError(
                f"unknown verified_empty value {raw!r} for "
                f"{getattr(row, 'video_name')} frame {getattr(row, 'frame_sampled')}"
            )
        video_name = str(getattr(row, "video_name"))
        confirmed.setdefault(video_name, []).append(
            _exact_frame_index(getattr(row, "frame_sampled"))
        )
    return {
        video_name: sorted(set(indices))
        for video_name, indices in sorted(confirmed.items())
    }


def validate_empty_reference_map(
    mapping: Mapping[str, Iterable[int]],
    video_names: Iterable[str],
    frames_root: Path | str,
    *,
    minimum_frames: int = 3,
) -> dict[str, list[int]]:
    if minimum_frames < 3:
        raise ValueError("minimum_frames must be at least 3")
    names = list(video_names)
    root = Path(frames_root)
    normalized: dict[str, list[int]] = {}
    for supplied_name, supplied_indices in mapping.items():
        video_name = _normalize_video_name(supplied_name, names)
        try:
            indices = sorted({_exact_frame_index(index) for index in supplied_indices})
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid frame index for {supplied_name}: {error}") from error
        if any(index < 0 for index in indices):
            raise ValueError(f"frame indices must be nonnegative for {supplied_name}")
        if len(indices) < minimum_frames:
            raise ValueError(f"{supplied_name} requires at least {minimum_frames} distinct frames")
        for index in indices:
            path = _frame_path(root, video_name, index)
            if not path.is_file():
                raise FileNotFoundError(path)
        normalized[video_name] = indices
    return dict(sorted(normalized.items()))


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


def _preserve_review_columns(
    fresh: pd.DataFrame,
    existing_path: Path,
    columns: list[str],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Carry human-entered values forward, including reviewed rows no longer sampled."""
    if not existing_path.exists():
        return fresh
    existing = pd.read_csv(existing_path)
    keys = ["video_name", "frame_sampled"]
    available = [column for column in columns if column in existing.columns]
    if not set(keys).issubset(existing.columns) or not available:
        return fresh
    existing = existing.drop_duplicates(keys, keep="last")
    preserved = existing[keys + available]
    annotated = pd.Series(False, index=existing.index)
    for column in available:
        values = existing[column]
        default = (defaults or {}).get(column, pd.NA)
        if isinstance(default, str):
            normalized = values.astype("string").fillna("").str.strip()
            annotated |= normalized.ne(default)
        else:
            annotated |= values.notna()
    fresh_keys = pd.MultiIndex.from_frame(fresh[keys])
    existing_keys = pd.MultiIndex.from_frame(existing[keys])
    orphaned = existing[annotated & ~existing_keys.isin(fresh_keys)].copy()
    if not orphaned.empty:
        missing_columns = [column for column in fresh.columns if column not in orphaned.columns]
        for column in missing_columns:
            orphaned[column] = pd.NA
        fresh = pd.concat([fresh, orphaned[fresh.columns]], ignore_index=True)
        fresh = fresh.sort_values(keys, kind="stable").reset_index(drop=True)
    result = fresh.drop(columns=available, errors="ignore").merge(
        preserved, on=keys, how="left", validate="one_to_one"
    )
    for column, default in (defaults or {}).items():
        if column in result.columns:
            if isinstance(default, str):
                result[column] = result[column].astype(object)
            result[column] = result[column].fillna(default)
    return result


def save_verified_empty_frames(
    mapping: Mapping[str, Iterable[int]],
    output_path: Path | str,
    video_names: Iterable[str],
    frames_root: Path | str,
) -> dict[str, list[int]]:
    normalized = validate_empty_reference_map(mapping, video_names, frames_root)
    _atomic_write_text(Path(output_path), json.dumps(normalized, indent=2, sort_keys=True))
    return normalized


def estimate_mm_per_pixel(points: pd.DataFrame) -> float:
    required = {"known_height_mm", "measured_height_px"}
    if not required.issubset(points.columns):
        raise ValueError(f"calibration points missing columns: {sorted(required - set(points.columns))}")
    entered = points["known_height_mm"].notna()
    if not entered.any():
        raise ValueError("at least one positive physical-height calibration point is required")
    if points.loc[entered, "measured_height_px"].isna().any():
        raise ValueError("physical-height calibration has missing measured_height_px")
    usable = points.loc[entered, ["known_height_mm", "measured_height_px"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if usable.isna().any().any() or not np.isfinite(usable.to_numpy(dtype=float)).all():
        raise ValueError("physical-height calibration values must be numeric and finite")
    if (usable <= 0).any().any():
        raise ValueError("physical-height calibration values must be positive")
    if usable.measured_height_px.duplicated().any():
        raise ValueError("physical-height calibration has duplicate measured heights")
    ratios = usable.known_height_mm / usable.measured_height_px
    return float(np.median(ratios))


def fit_volume_calibration(
    points: pd.DataFrame,
    mm_per_pixel: float,
    *,
    degree: int = 1,
) -> tuple[tuple[float, ...], dict[str, float]]:
    if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be a positive finite number")
    if degree < 1:
        raise ValueError("degree must be at least 1")
    required = {"measured_height_px", "known_volume_ml"}
    if not required.issubset(points.columns):
        raise ValueError(f"calibration points missing columns: {sorted(required - set(points.columns))}")
    entered = points["known_volume_ml"].notna()
    if points.loc[entered, "measured_height_px"].isna().any():
        raise ValueError("volume calibration has missing measured_height_px")
    usable = points.loc[entered, ["measured_height_px", "known_volume_ml"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if usable.isna().any().any() or not np.isfinite(usable.to_numpy(dtype=float)).all():
        raise ValueError("volume calibration values must be numeric and finite")
    if (usable < 0).any().any():
        raise ValueError("volume calibration values must be nonnegative")
    if usable.measured_height_px.duplicated().any():
        raise ValueError("volume calibration has duplicate measured heights")
    heights_mm = usable.measured_height_px.to_numpy(dtype=float) * mm_per_pixel
    volumes = usable.known_volume_ml.to_numpy(dtype=float)
    if len(usable) < degree + 1 or len(np.unique(heights_mm)) < degree + 1:
        raise ValueError(f"degree {degree} calibration requires at least {degree + 1} distinct heights")
    coefficients = np.polyfit(heights_mm, volumes, degree)
    predicted = np.polyval(coefficients, heights_mm)
    residual = volumes - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((volumes - volumes.mean()) ** 2))
    if total == 0:
        r2 = 1.0 if np.allclose(residual, 0.0) else 0.0
    else:
        r2 = float(1 - np.sum(residual**2) / total)
    values = np.concatenate([coefficients, np.asarray([rmse, r2])])
    if not np.isfinite(values).all():
        raise ValueError("volume calibration produced non-finite coefficients or metrics")
    return tuple(float(value) for value in coefficients), {
        "n": int(len(usable)), "rmse_ml": rmse, "r2": r2
    }


def validation_metrics(
    labels: pd.DataFrame,
    *,
    predicted_column: str = "fill_height_px",
) -> dict[str, float]:
    required = {
        predicted_column, "manual_fill_top_y", "manual_fill_bottom_y", "label_status"
    }
    if not required.issubset(labels.columns):
        raise ValueError(f"validation labels missing columns: {sorted(required - set(labels.columns))}")
    statuses = labels.label_status.fillna("unlabeled").astype(str).str.strip().str.lower()
    allowed = {"unlabeled", "complete", "rejected"}
    unknown = sorted(set(statuses) - allowed)
    if unknown:
        raise ValueError(f"unknown label_status values: {unknown}")
    complete = statuses.eq("complete")
    manual_columns = ["manual_fill_top_y", "manual_fill_bottom_y"]
    manual = labels.loc[complete, manual_columns].apply(pd.to_numeric, errors="coerce")
    if manual.isna().any().any() or not np.isfinite(manual.to_numpy(dtype=float)).all():
        raise ValueError("completed labels require finite manual top and bottom rows")
    if (manual < 0).any().any():
        raise ValueError("completed labels require nonnegative manual rows")
    all_manual_heights = manual.manual_fill_bottom_y - manual.manual_fill_top_y + 1
    if (all_manual_heights <= 0).any():
        raise ValueError("completed labels require bottom row at or below top row")
    predicted = pd.to_numeric(labels.loc[complete, predicted_column], errors="coerce")
    values = manual.copy()
    values[predicted_column] = predicted
    usable = values.dropna()
    if not np.isfinite(usable[predicted_column].to_numpy(dtype=float)).all():
        raise ValueError("completed-label predictions must be finite when present")
    manual_height = usable.manual_fill_bottom_y - usable.manual_fill_top_y + 1
    total_rows = len(labels)
    completed_rows = int(complete.sum())
    if usable.empty:
        return {
            "total_rows": total_rows, "completed_rows": completed_rows,
            "labeled_rows": 0,
            "coverage": float(completed_rows / total_rows) if total_rows else 0.0,
            "prediction_coverage": 0.0,
            "mae_px": np.nan, "rmse_px": np.nan, "bias_px": np.nan,
        }
    errors = usable[predicted_column] - manual_height
    return {
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "labeled_rows": int(len(usable)),
        "coverage": float(completed_rows / total_rows) if total_rows else 0.0,
        "prediction_coverage": float(len(usable) / completed_rows) if completed_rows else 0.0,
        "mae_px": float(errors.abs().mean()),
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
        "bias_px": float(errors.mean()),
    }


def generate_review_artifacts(
    measurements: pd.DataFrame,
    frames_root: Path | str,
    review_dir: Path | str,
    *,
    candidates_per_video: int = 5,
    validation_per_video: int = 6,
) -> dict[str, Path]:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = review_dir / "empty_reference_candidates.csv"
    validation_path = review_dir / "manual_validation_sample.csv"
    calibration_path = review_dir / "calibration_points.csv"
    candidates = make_empty_reference_candidates(
        measurements, frames_root, count_per_video=candidates_per_video
    )
    validation = make_validation_sample(
        measurements, frames_root, count_per_video=validation_per_video
    )
    candidates = _preserve_review_columns(
        candidates,
        candidates_path,
        ["verified_empty", "reviewer_notes"],
        defaults={"reviewer_notes": ""},
    )
    _atomic_write_csv(candidates, candidates_path)
    validation = _preserve_review_columns(
        validation,
        validation_path,
        [
            "manual_fill_top_y", "manual_fill_bottom_y", "known_volume_ml",
            "label_status", "reviewer_notes",
        ],
        defaults={"label_status": "unlabeled", "reviewer_notes": ""},
    )
    _atomic_write_csv(validation, validation_path)
    if not calibration_path.exists():
        _atomic_write_csv(
            pd.DataFrame(
                columns=[
                    "known_height_mm", "measured_height_px", "known_volume_ml",
                    "source", "reviewer_notes",
                ]
            ),
            calibration_path,
        )
    return {
        "candidates": candidates_path,
        "validation": validation_path,
        "calibration": calibration_path,
    }

