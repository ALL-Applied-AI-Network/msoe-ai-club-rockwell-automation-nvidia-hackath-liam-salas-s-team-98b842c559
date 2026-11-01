"""CPU-only Part-B fill clipping and masked stage measurement.

The boundary in this module is intentionally small: a camera video, the programmed
recipe/vial sequence, and optional timing/frame evidence.  It does not import or read
simulation metadata.  A synthetic receipt may be used by a caller to locate a video or
frame file, but no receipt values are consumed as measurements.

The MP4 clips are review artifacts.  Measurement always uses hash-bound PNG stills and
masks, and the mask source is recorded explicitly.  The current accepted path uses
oracle masks for synthetic regression only; no RGB-only predicted-mask result is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from .vision_measurement import measure_image
from ..core.contracts.evidence import (
    validate_evidence_record_v1,
    validate_evidence_sequence_v1,
)
from ..core.io import open_image, open_path


TRIAD_CONFIG = {
    "tight_peer_pair_max_delta_e76": 2.0,
    "target_outlier_margin_delta_e76": 1.0,
}
REAL_FINAL_MIXTURE_END_OFFSET_S = 0.3
REAL_FINAL_MIXTURE_DEPARTURE_MARGIN_S = 0.1
REAL_FINAL_MIXTURE_MIN_FRAMES = 3

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "label", "labels", "expected_disposition", "residue", "residue_ml",
        "prescribed_residue", "prescribed_leading_residual_volume_ml",
        "residual_volume_ml", "residue_source_liquid", "palette", "tank_palette",
        "tank_colours", "tank_colors", "tank_colour_preset", "tank_color_preset",
        "tank_colour_preset_hash", "renderer_palette", "renderer_palette_sha256",
        "palette_sha256", "preset", "preset_name", "advanced_toggles",
        "tank_colour_resolution", "transmission_overrides", "dye_strength_scale",
        "t100_clear_lock", "randomize_around_preset", "randomization_seed",
        "tank_strength_jitter_bounds", "channel_jitter_bounds", "tank_strength_jitter",
        "channel_jitter", "transmission_1cm", "resolved_tank_transmission_1cm", "particle_state",
        "particle_state_summary", "particles", "renderer_seed", "render_seed",
        "host_profile", "oracle", "synthetic_class", "sim_metadata",
        "excluded_from_model_inputs", "free_text_label", "physics_state",
    }
)
_TIMING_FIELDS = frozenset({"fills"})
_TIMING_FILL_FIELDS = frozenset(
    {"vial_id", "fill_start_frame", "fill_end_frame", "transition_frame_ranges"}
)


def _finite_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe non-empty ID")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_for_forbidden(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = key.lower().replace("-", "_") if isinstance(key, str) else ""
            if normalized_key in _FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"simulation metadata field is not a Part-B input: {key}")
            _scan_for_forbidden(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _scan_for_forbidden(child)


def _range(value: Sequence[object], name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a half-open [start, end] range")
    start = _finite_int(value[0], f"{name}.start")
    end = _finite_int(value[1], f"{name}.end")
    if end <= start:
        raise ValueError(f"{name} must have end > start")
    return start, end


def _normalize_ranges(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of frame ranges")
    return tuple(_range(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class FillBoundary:
    """One fill's half-open source-video frame interval."""

    vial_id: str
    start_frame: int
    end_frame: int
    transition_frame_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _safe_id(self.vial_id, "vial_id")
        _range((self.start_frame, self.end_frame), "fill_frame_range")
        for index, interval in enumerate(self.transition_frame_ranges):
            _range(interval, f"transition_frame_ranges[{index}]")
            if interval[1] > self.start_frame and interval[0] < self.end_frame:
                raise ValueError("transition frame range overlaps measured fill")


@dataclass(frozen=True)
class StageBoundary:
    """One state-after-dispense stage and its RGB/mask artifact locators.

    ``frame_index`` is the source frame used when ``rgb_path`` is absent.  A mask is
    required because this module deliberately has no accepted RGB-only segmenter.
    """

    vial_id: str
    stage_id: str
    stage_ordinal: int
    tank_id: str
    stage_count: int
    rgb_path: str | os.PathLike[str] | None
    mask_path: str | os.PathLike[str]
    frame_index: int | None = None
    mask_source: str = "oracle"
    recipe_id: str | None = None
    run_id: str | None = None
    colour_regime_id: str | None = None
    reference_role: str = "triad"
    source_rgb_path: str | os.PathLike[str] | None = None
    source_mask_path: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        _safe_id(self.vial_id, "vial_id")
        _safe_id(self.stage_id, "stage_id")
        _safe_id(self.tank_id, "tank_id")
        _finite_int(self.stage_ordinal, "stage_ordinal", minimum=1)
        _finite_int(self.stage_count, "stage_count", minimum=1)
        if self.stage_ordinal > self.stage_count:
            raise ValueError("stage ordinal exceeds active stage count")
        if self.rgb_path is None and self.frame_index is None:
            raise ValueError("stage needs rgb_path or source-video frame_index")
        if self.frame_index is not None:
            _finite_int(self.frame_index, "frame_index")
        if self.mask_source != "oracle":
            raise ValueError(
                "predicted masks are not accepted by WP4; use oracle masks only for synthetic regression"
            )
        if self.recipe_id is not None:
            _safe_id(self.recipe_id, "recipe_id")
        if self.run_id is not None:
            _safe_id(self.run_id, "run_id")
        if self.colour_regime_id is not None:
            _safe_id(self.colour_regime_id, "colour_regime_id")
        if self.reference_role not in {"triad", "target", "reference_bank"}:
            raise ValueError("reference_role is invalid")


def _source_path(stage: StageBoundary, field: str) -> Path:
    value = getattr(stage, field)
    if value is None:
        value = stage.rgb_path if field == "source_rgb_path" else stage.mask_path
    return Path(value).expanduser().resolve(strict=True)


def _durable_output_path(value: object, partial: Path, output: Path) -> object:
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    try:
        relative = candidate.resolve(strict=False).relative_to(partial.resolve())
    except ValueError:
        return value
    return str(output / relative)


def cross_batch_threshold(max_radius: float) -> float:
    """Frozen cross-batch threshold: ``max(2.0, max_radius + 1.0)``."""

    if isinstance(max_radius, bool) or not isinstance(max_radius, (int, float)):
        raise TypeError("max_radius must be numeric")
    value = float(max_radius)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("max_radius must be finite and non-negative")
    return max(TRIAD_CONFIG["tight_peer_pair_max_delta_e76"], value + TRIAD_CONFIG["target_outlier_margin_delta_e76"])


def real_final_mixture_window(
    end_frame: int,
    departure_frame: int,
    *,
    fps: float = 30.0,
) -> dict[str, object]:
    """Return the frozen real-video final-mixture frame window.

    The window starts ``end + 0.3 s`` and ends ``0.1 s`` before cart departure.
    This is timing evidence, not a colour or defect label.  Endpoints are half-open
    frame indices and a window with fewer than three frames is ineligible.
    """

    _finite_int(end_frame, "end_frame")
    _finite_int(departure_frame, "departure_frame")
    if departure_frame <= end_frame:
        raise ValueError("departure_frame must follow end_frame")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be finite and positive")
    start = int(math.ceil((end_frame / float(fps) + REAL_FINAL_MIXTURE_END_OFFSET_S) * float(fps)))
    stop = int(math.floor((departure_frame / float(fps) - REAL_FINAL_MIXTURE_DEPARTURE_MARGIN_S) * float(fps))) + 1
    count = max(0, stop - start)
    return {
        "start_frame": start,
        "end_frame": stop,
        "n_frames": count,
        "eligible": count >= REAL_FINAL_MIXTURE_MIN_FRAMES,
        "exclusion_reason": None if count >= REAL_FINAL_MIXTURE_MIN_FRAMES else "window_too_short",
        "timing_only": True,
    }


def boundaries_from_timing_evidence(
    recipe_sequence: object,
    timing_evidence: object,
) -> tuple[FillBoundary, ...]:
    """Convert optional INN-52-style burst timing rows into fill boundaries."""

    recipes = _validate_recipe_sequence(recipe_sequence)
    if not isinstance(timing_evidence, Mapping):
        raise ValueError("timing evidence must be a mapping")
    _scan_for_forbidden(timing_evidence)
    if set(timing_evidence) != _TIMING_FIELDS:
        raise ValueError("timing evidence fields are not the exact Part-B contract")
    rows = timing_evidence.get("fills")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(recipes):
        raise ValueError("timing evidence must contain one fill row per programmed vial")
    result: list[FillBoundary] = []
    for recipe, raw in zip(recipes, rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("timing fill rows must be mappings")
        if set(raw) - _TIMING_FILL_FIELDS:
            raise ValueError("timing fill fields are not the exact Part-B contract")
        vial_id = raw.get("vial_id", recipe["vial_id"])
        if vial_id != recipe["vial_id"]:
            raise ValueError("timing vial order differs from programmed sequence")
        result.append(
            FillBoundary(
                str(vial_id),
                _finite_int(raw.get("fill_start_frame"), "fill_start_frame"),
                _finite_int(raw.get("fill_end_frame"), "fill_end_frame"),
                _normalize_ranges(raw.get("transition_frame_ranges"), "transition_frame_ranges"),
            )
        )
    return tuple(result)


def _validate_global_intervals(fill_boundaries: Sequence[FillBoundary]) -> None:
    intervals: list[tuple[int, int, str]] = []
    previous_fill_end = -1
    for fill in fill_boundaries:
        if fill.start_frame < previous_fill_end:
            raise ValueError("global frame intervals overlap or are out of order")
        previous_fill_end = fill.end_frame
        intervals.append((fill.start_frame, fill.end_frame, f"fill:{fill.vial_id}"))
        for index, (start, end) in enumerate(fill.transition_frame_ranges):
            intervals.append((start, end, f"transition:{fill.vial_id}:{index}"))
    for (left_start, left_end, left_name), (right_start, right_end, right_name) in itertools.combinations(
        sorted(intervals), 2
    ):
        if left_start < right_end and right_start < left_end:
            raise ValueError(
                f"global frame intervals overlap: {left_name} and {right_name}"
            )
    for fill in fill_boundaries:
        for start, end in fill.transition_frame_ranges:
            if not (end <= fill.start_frame or start >= fill.end_frame):
                raise ValueError("transition interval overlaps its measured fill")
    # A transition is either an entry before its fill or an exit after it.  A middle
    # interval has no evidence-record-v1 representation and is rejected.
    for fill in fill_boundaries:
        if any(start > fill.start_frame and end < fill.end_frame for start, end in fill.transition_frame_ranges):
            raise ValueError("transition interval is neither fill entry nor fill exit")


def _validate_stage_set(
    stages: Sequence[StageBoundary], recipes: Sequence[Mapping[str, object]]
) -> None:
    if not stages:
        return
    recipe_by_vial = {str(row["vial_id"]): row for row in recipes}
    by_vial: dict[str, list[StageBoundary]] = {}
    for stage in stages:
        if stage.vial_id not in recipe_by_vial:
            raise ValueError("stage references an unknown programmed vial")
        by_vial.setdefault(stage.vial_id, []).append(stage)
    for recipe in recipes:
        vial_id = str(recipe["vial_id"])
        expected_order = tuple(recipe["dispense_order"])
        rows = by_vial.get(vial_id, [])
        if len(rows) != len(expected_order):
            raise ValueError(
                f"measured fill requires exactly one stage for every programmed dispense: {vial_id}"
            )
        ordinals = [row.stage_ordinal for row in rows]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError(f"duplicate stage ordinal for measured fill: {vial_id}")
        if ordinals != list(range(1, len(expected_order) + 1)):
            raise ValueError(f"ordered stage sequence is not contiguous: {vial_id}")
        for row in rows:
            if row.stage_count != len(expected_order) or row.tank_id != expected_order[row.stage_ordinal - 1]:
                raise ValueError("stage tank/order does not match programmed active dispense")


def _evidence_transition_ranges(fill: FillBoundary) -> dict[str, list[int] | None]:
    entry: list[int] | None = None
    exit_range: list[int] | None = None
    for interval in fill.transition_frame_ranges:
        if interval[1] <= fill.start_frame:
            if entry is not None:
                raise ValueError("multiple entry transitions cannot bind evidence-record-v1")
            entry = list(interval)
        elif interval[0] >= fill.end_frame:
            if exit_range is not None:
                raise ValueError("multiple exit transitions cannot bind evidence-record-v1")
            exit_range = list(interval)
        else:
            raise ValueError("transition cannot bind evidence-record-v1")
    return {"entry": entry, "exit": exit_range}


def _build_evidence_records(
    recipes: Sequence[Mapping[str, object]],
    fills: Sequence[FillBoundary],
    stage_sources: Sequence[tuple[StageBoundary, StageBoundary]],
    measurements: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    clip_hashes: Mapping[str, str],
    source_video_path: Path,
    source_video_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    recipe_by_vial = {str(row["vial_id"]): row for row in recipes}
    fill_by_vial = {row.vial_id: row for row in fills}
    source_by_evidence = {
        f"EVID-{_artifact_id(derived.vial_id, derived.stage_id)}": (derived, original)
        for derived, original in stage_sources
    }
    pairs_by_evidence: dict[str, list[dict[str, object]]] = {}
    for pair in pairs:
        left = str(pair["left_evidence_id"])
        right = str(pair["right_evidence_id"])
        value = float(pair["delta_e76"])
        pairs_by_evidence.setdefault(left, []).append({"other_evidence_id": right, "delta_e76": value})
        pairs_by_evidence.setdefault(right, []).append({"other_evidence_id": left, "delta_e76": value})
    records: list[dict[str, object]] = []
    locators: list[dict[str, object]] = []
    for measurement in measurements:
        evidence_id = str(measurement["evidence_id"])
        derived, original = source_by_evidence[evidence_id]
        recipe = recipe_by_vial[derived.vial_id]
        fill = fill_by_vial[derived.vial_id]
        source_rgb = _source_path(original, "source_rgb_path")
        source_mask = _source_path(original, "source_mask_path")
        record = {
            "run_id": recipe["run_id"],
            "case_id": recipe["case_id"],
            "vial_ordinal": recipe["vial_ordinal"],
            "block_recipe_id": recipe["block_recipe_id"],
            "block_position": recipe["block_position"],
            "recipe_id": recipe["recipe_id"],
            "previous_recipe_id": recipe["previous_recipe_id"],
            "recipe_ml": recipe["recipe_ml"],
            "dispense_order": recipe["dispense_order"],
            "split_role": recipe["split_role"],
            "split_version": recipe["split_version"],
            "stage_id": derived.stage_id,
            "tank_id": derived.tank_id,
            "stage_ordinal": derived.stage_ordinal,
            "fill_clip_path": f"fills/{derived.vial_id}.mp4",
            "fill_frame_range": [fill.start_frame, fill.end_frame],
            "transition_frame_ranges": _evidence_transition_ranges(fill),
            "clip_sha256": clip_hashes[derived.vial_id],
            "image_id": evidence_id,
            # These hashes must identify the bytes consumed by measure_image.  The
            # untouched input bindings are carried separately in image_locators.
            "rgb_sha256": measurement["rgb_sha256"],
            "mask_sha256": measurement["mask_sha256"],
            "mask_source": original.mask_source,
            "measurement_id": measurement["measurement_id"],
            "median_lab": measurement["median_lab"],
            "mad_lab": measurement["mad_lab"],
            "iqr_lab": measurement["iqr_lab"],
            "pairwise_delta_e76": sorted(pairs_by_evidence.get(evidence_id, []), key=lambda row: row["other_evidence_id"]),
            "colour_regime_id": None,
            "evidence_id": evidence_id,
            "decision_id": None,
            "disposition": None,
            "limit_codes": [],
            "history_refs": [],
            "explanation_id": None,
            "model_revision": None,
            "prompt_sha256": None,
            "image_ids_sent": [],
            "guard_result": None,
        }
        records.append(validate_evidence_record_v1(record, phase="measured", mode="storage"))
        measured_rgb = Path(derived.rgb_path).resolve(strict=True)
        measured_mask = Path(derived.mask_path).resolve(strict=True)
        locators.append(
            {
                "image_id": evidence_id,
                # These locators are the exact durable pixel files consumed by
                # measure_image.  Original inputs remain bound separately in
                # payload['stages'] below.
                "rgb_path": str(measured_rgb),
                "rgb_sha256": record["rgb_sha256"],
                "mask_path": str(measured_mask),
                "mask_sha256": record["mask_sha256"],
                "derived_still_path": str(derived.rgb_path),
                "derived_crop_path": str(Path(derived.rgb_path).parent.parent / "crops" / f"{evidence_id}.png"),
                "source_video_path": str(source_video_path),
                "source_video_sha256": source_video_sha256,
                "source_frame_index": original.frame_index,
            }
        )
    for recipe in recipes:
        sequence = [
            record for record in records if record["case_id"] == recipe["case_id"]
        ]
        validate_evidence_sequence_v1(sequence, phase="measured", mode="storage")
    return records, locators


def _load_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

    reader = cv2.VideoCapture(os.fspath(video_path))
    if not reader.isOpened():
        raise ValueError(f"camera video could not be opened: {video_path}")
    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = reader.read()
    finally:
        reader.release()
    if not ok or frame is None:
        raise ValueError(f"source video frame is unavailable: {frame_index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _write_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path, format="PNG")


def _read_png(path: Path, mode: str) -> np.ndarray:
    try:
        with open_image(path) as image:
            image.load()
            if mode == "RGB" and image.mode != "RGB":
                raise ValueError(f"RGB stage still is not RGB PNG: {path}")
            if mode == "L" and image.mode not in {"L", "1"}:
                raise ValueError(f"stage mask is not binary PNG: {path}")
            return np.asarray(image.convert(mode), dtype=np.uint8).copy()
    except OSError as error:
        raise ValueError(f"stage artifact is unreadable: {path}") from error


def _artifact_id(vial_id: str, stage_id: str) -> str:
    return f"{vial_id}-{stage_id}"


def _masked_crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = mask == 255
    rows, cols = np.where(selected)
    if rows.size == 0:
        raise ValueError("stage mask is empty")
    crop = np.zeros_like(rgb)
    crop[selected] = rgb[selected]
    return crop[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def measure_stage_boundaries(stages: Sequence[StageBoundary]) -> dict[str, object]:
    """Measure hash-bound stage RGB/mask pairs and same-stage CIE76 pairs.

    This delegates Lab/MAD/IQR calculation to the existing ``vision_measurement``
    implementation.  It never consumes labels, palettes, residues or simulator state.
    """

    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("stages must be a non-empty list")
    if any(not isinstance(item, StageBoundary) for item in stages):
        raise TypeError("stages must contain StageBoundary values")
    measurements: list[dict[str, object]] = []
    by_stage: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for stage in stages:
        rgb = Path(stage.rgb_path).resolve(strict=True) if stage.rgb_path is not None else None
        mask = Path(stage.mask_path).resolve(strict=True)
        locator = {
            "rgb_path": os.fspath(rgb),
            "rgb_sha256": _sha256(rgb),
            "mask_path": os.fspath(mask),
            "mask_sha256": _sha256(mask),
        }
        evidence_id = f"EVID-{_artifact_id(stage.vial_id, stage.stage_id)}"
        measured = measure_image(evidence_id, locator, _include_dispersion=True)
        if measured.get("validity") != "valid":
            raise ValueError(f"stage measurement invalid: {measured.get('limit_codes')}")
        row = {
            "evidence_id": evidence_id,
            "measurement_id": f"MEAS-{_artifact_id(stage.vial_id, stage.stage_id)}",
            "image_id": _artifact_id(stage.vial_id, stage.stage_id),
            "vial_id": stage.vial_id,
            "stage_id": stage.stage_id,
            "stage_ordinal": stage.stage_ordinal,
            "tank_id": stage.tank_id,
            "recipe_id": stage.recipe_id,
            "run_id": stage.run_id,
            "colour_regime_id": stage.colour_regime_id,
            "reference_role": stage.reference_role,
            "mask_source": stage.mask_source,
            "rgb_sha256": locator["rgb_sha256"],
            "mask_sha256": locator["mask_sha256"],
            "liquid_pixel_count": measured["liquid_pixel_count"],
            "mask_coverage_fraction": measured["mask_coverage_fraction"],
            "mask_height_px": measured["mask_height_px"],
            "median_lab": measured["median_lab"],
            "mad_lab": measured["mad_lab"],
            "iqr_lab": measured["iqr_lab"],
            "lab_unit": "cie_lab_d65_2deg",
        }
        measurements.append(row)
        context_key = (
            stage.stage_id,
            stage.stage_ordinal,
            stage.tank_id,
            stage.recipe_id,
            stage.run_id,
            stage.colour_regime_id,
            stage.reference_role,
        )
        by_stage.setdefault(context_key, []).append(row)
    context_by_stage_id: dict[str, set[tuple[object, ...]]] = {}
    for key in by_stage:
        context_by_stage_id.setdefault(str(key[0]), set()).add(key[1:])
    if any(len(contexts) > 1 for contexts in context_by_stage_id.values()):
        raise ValueError("same-stage rows lack a single comparable context")
    pairs: list[dict[str, object]] = []
    for context_key, rows in sorted(by_stage.items(), key=lambda item: repr(item[0])):
        stage_id = str(context_key[0])
        for left, right in itertools.combinations(sorted(rows, key=lambda row: str(row["vial_id"])), 2):
            delta = math.sqrt(
                sum((float(a) - float(b)) ** 2 for a, b in zip(left["median_lab"], right["median_lab"], strict=True))
            )
            pairs.append(
                {
                    "pair_id": f"PAIR-{stage_id}-{left['vial_id']}-{right['vial_id']}",
                    "stage_id": stage_id,
                    "stage_ordinal": context_key[1],
                    "tank_id": context_key[2],
                    "recipe_id": context_key[3],
                    "run_id": context_key[4],
                    "colour_regime_id": context_key[5],
                    "reference_role": context_key[6],
                    "left_vial_id": left["vial_id"],
                    "right_vial_id": right["vial_id"],
                    "left_evidence_id": left["evidence_id"],
                    "right_evidence_id": right["evidence_id"],
                    "delta_e76": delta,
                    "delta_e76_6dp": f"{delta:.6f}",
                    "unit": "delta_e76",
                }
            )
    return {
        "measurements": measurements,
        "pairwise_delta_e76": pairs,
        "measurement_contract": {
            "location": "coordinatewise_median_cie_lab_d65_2deg",
            "dispersion": "coordinatewise_lab_mad_and_iqr",
            "distance": "cie76_between_coordinatewise_median_lab",
            "mask_policy": "oracle masks only; no accepted RGB-only predicted mask result",
        },
    }


def _open_writer(path: Path, *, fps: float, width: int, height: int) -> Any:
    import cv2

    writer = cv2.VideoWriter(
        os.fspath(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV mp4v writer could not be opened")
    return writer


def _validate_recipe_sequence(recipe_sequence: object) -> list[dict[str, object]]:
    if not isinstance(recipe_sequence, (list, tuple)) or not recipe_sequence:
        raise ValueError("recipe_sequence must be a non-empty list")
    _scan_for_forbidden(recipe_sequence)
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(recipe_sequence, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError("recipe_sequence rows must be mappings")
        vial_id = _safe_id(raw.get("vial_id", f"V{index}"), "vial_id")
        recipe_id = _safe_id(raw.get("recipe_id"), "recipe_id")
        order = raw.get("dispense_order")
        if not isinstance(order, (list, tuple)) or not order or any(not isinstance(tank, str) or not tank for tank in order):
            raise ValueError("dispense_order must be a non-empty tank list")
        if len(set(order)) != len(order):
            raise ValueError("dispense_order must not repeat a tank")
        raw_recipe_ml = raw.get("recipe_ml")
        if raw_recipe_ml is None:
            recipe_ml = {tank: (1.0 if tank in order else 0.0) for tank in ("t100", "t200", "t400")}
        elif isinstance(raw_recipe_ml, Mapping) and set(raw_recipe_ml) == {"t100", "t200", "t400"}:
            recipe_ml = {tank: float(raw_recipe_ml[tank]) for tank in ("t100", "t200", "t400")}
        else:
            raise ValueError("recipe_ml must contain exactly t100, t200 and t400")
        if any(value < 0.0 or not math.isfinite(value) for value in recipe_ml.values()):
            raise ValueError("recipe_ml values must be finite and non-negative")
        if set(tank for tank, value in recipe_ml.items() if value > 0.0) != set(order):
            raise ValueError("recipe_ml positive tanks must equal dispense_order")
        split_role = raw.get("split_role", "evaluation")
        if split_role not in {"train", "val", "validation", "evaluation", "test", "real_validation", "calibration"}:
            raise ValueError("split_role is invalid")
        rows.append(
            {
                "vial_id": vial_id,
                "recipe_id": recipe_id,
                "dispense_order": list(order),
                "run_id": _safe_id(raw.get("run_id", "RUN-UNSPECIFIED"), "run_id"),
                "case_id": _safe_id(raw.get("case_id", vial_id), "case_id"),
                "vial_ordinal": _finite_int(raw.get("vial_ordinal", index), "vial_ordinal", minimum=1),
                "block_recipe_id": _safe_id(raw.get("block_recipe_id", recipe_id), "block_recipe_id"),
                "block_position": _finite_int(raw.get("block_position", index), "block_position", minimum=1),
                "previous_recipe_id": raw.get("previous_recipe_id"),
                "recipe_ml": recipe_ml,
                "split_role": split_role,
                "split_version": _safe_id(raw.get("split_version", "wp4-v1"), "split_version"),
            }
        )
    if len({str(row["vial_id"]) for row in rows}) != len(rows):
        raise ValueError("recipe_sequence vial IDs must be unique")
    return rows


def build_fill_clips(
    video_path: str | os.PathLike[str],
    recipe_sequence: object,
    output_dir: str | os.PathLike[str],
    *,
    published_output_dir: str | os.PathLike[str] | None = None,
    fill_boundaries: Sequence[FillBoundary] | None = None,
    stage_boundaries: Sequence[StageBoundary] = (),
    fps: float | None = None,
    timing_evidence: object | None = None,
) -> dict[str, object]:
    """Create per-fill MP4s and optional per-stage still/crop artifacts.

    Fill and transition frame intervals are zero-based, half-open source-video ranges.
    Timing evidence is optional and provenance-only; explicit ``fill_boundaries`` remain
    the actual boundary input.  Existing output directories are never overwritten.  A
    caller staging a larger atomic directory may provide ``published_output_dir`` so
    absolute locators name the post-rename case directory rather than the staging root.
    """

    video = Path(video_path).expanduser().resolve(strict=True)
    if video.is_symlink() or not video.is_file():
        raise ValueError("video must be a regular file")
    recipes = _validate_recipe_sequence(recipe_sequence)
    timing_projection: dict[str, object] | None = None
    if timing_evidence is not None:
        timing_boundaries = boundaries_from_timing_evidence(recipes, timing_evidence)
        timing_projection = {
            "fills": [
                {
                    "vial_id": boundary.vial_id,
                    "fill_start_frame": boundary.start_frame,
                    "fill_end_frame": boundary.end_frame,
                    "transition_frame_ranges": [list(interval) for interval in boundary.transition_frame_ranges],
                }
                for boundary in timing_boundaries
            ]
        }
    else:
        timing_boundaries = None
    if fill_boundaries is None:
        if timing_evidence is None:
            raise ValueError("fill_boundaries or timing_evidence is required")
        fill_boundaries = timing_boundaries
    elif timing_boundaries is not None and tuple(fill_boundaries) != timing_boundaries:
        raise ValueError("explicit fill boundaries disagree with timing evidence")
    if not isinstance(fill_boundaries, (list, tuple)) or len(fill_boundaries) != len(recipes):
        raise ValueError("one fill boundary is required for every programmed vial")
    if any(not isinstance(item, FillBoundary) for item in fill_boundaries):
        raise TypeError("fill_boundaries must contain FillBoundary values")
    _validate_global_intervals(fill_boundaries)
    if [item.vial_id for item in fill_boundaries] != [str(row["vial_id"]) for row in recipes]:
        raise ValueError("fill boundaries must follow programmed vial sequence")
    if stage_boundaries and any(not isinstance(item, StageBoundary) for item in stage_boundaries):
        raise TypeError("stage_boundaries must contain StageBoundary values")
    _validate_stage_set(stage_boundaries, recipes)
    stage_by_vial: dict[str, list[StageBoundary]] = {}
    recipe_by_vial = {str(row["vial_id"]): row for row in recipes}
    for stage in stage_boundaries:
        stage_by_vial.setdefault(stage.vial_id, []).append(stage)
        if stage.vial_id not in recipe_by_vial:
            raise ValueError("stage references an unknown programmed vial")
        recipe_order = recipe_by_vial[stage.vial_id]["dispense_order"]
        if stage.stage_count != len(recipe_order):
            raise ValueError("stage count does not match programmed active stage order")
        if stage.tank_id != recipe_order[stage.stage_ordinal - 1]:
            raise ValueError("stage tank does not match programmed active stage order")
    # Publish absolute artifact locators even when callers provide a relative
    # destination.  ``absolute`` preserves the final-component symlink check
    # and keeps the partial-directory rename atomic.
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    published_output = (
        output
        if published_output_dir is None
        else Path(published_output_dir).expanduser().absolute()
    )
    if any(".partial-" in part for part in published_output.parts):
        raise ValueError("published_output_dir must not contain a partial-directory component")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        (partial / "fills").mkdir()
        (partial / "stages" / "crops").mkdir(parents=True)
        (partial / "stages" / "stills").mkdir(parents=True)
        import cv2

        reader = cv2.VideoCapture(os.fspath(video))
        if not reader.isOpened():
            raise ValueError(f"camera video could not be opened: {video}")
        source_fps = float(reader.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        output_fps = source_fps if fps is None else float(fps)
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("fps must be finite and positive")
        frame_count = int(round(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(reader.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            raise ValueError("camera video has no raster dimensions")
        for fill in fill_boundaries:
            if fill.end_frame > frame_count:
                raise ValueError("fill frame range exceeds camera video")
            if any(end > frame_count for _, end in fill.transition_frame_ranges):
                raise ValueError("transition frame range exceeds camera video")
            clip = partial / "fills" / f"{fill.vial_id}.mp4"
            writer = _open_writer(clip, fps=output_fps, width=width, height=height)
            try:
                reader.set(cv2.CAP_PROP_POS_FRAMES, fill.start_frame)
                for _ in range(fill.start_frame, fill.end_frame):
                    ok, frame = reader.read()
                    if not ok or frame is None:
                        raise ValueError("camera video ended inside fill range")
                    writer.write(frame)
            finally:
                writer.release()
        reader.release()

        fill_rows: list[dict[str, object]] = []
        for fill, recipe in zip(fill_boundaries, recipes, strict=True):
            transitions = [list(interval) for interval in fill.transition_frame_ranges]
            fill_rows.append(
                {
                    "vial_id": fill.vial_id,
                    "recipe_id": recipe["recipe_id"],
                    "dispense_order": recipe["dispense_order"],
                    "fill_frame_range": [fill.start_frame, fill.end_frame],
                    "transition_frame_ranges": transitions,
                    "transport_frame_ranges": transitions,
                    "transition_tag": "transition",
                    "fill_clip_path": f"fills/{fill.vial_id}.mp4",
                    "clip_sha256": _sha256(partial / "fills" / f"{fill.vial_id}.mp4"),
                }
            )

        stage_rows: list[StageBoundary] = []
        stage_sources: list[tuple[StageBoundary, StageBoundary]] = []
        for vial_id, rows in stage_by_vial.items():
            fill = next(item for item in fill_boundaries if item.vial_id == vial_id)
            for stage in rows:
                if stage.frame_index is not None and not (fill.start_frame <= stage.frame_index < fill.end_frame):
                    raise ValueError("stage frame must be within its measured fill interval")
                rgb_path = Path(stage.rgb_path).expanduser().resolve(strict=True) if stage.rgb_path is not None else None
                if rgb_path is None:
                    rgb = _load_video_frame(video, int(stage.frame_index))
                else:
                    rgb = _read_png(rgb_path, "RGB")
                mask_path = Path(stage.mask_path).expanduser().resolve(strict=True)
                mask = _read_png(mask_path, "L")
                if rgb.shape[:2] != mask.shape or not set(int(v) for v in np.unique(mask)).issubset({0, 255}):
                    raise ValueError("stage RGB/mask shape or binary contract is invalid")
                evidence_id = f"EVID-{_artifact_id(stage.vial_id, stage.stage_id)}"
                still = partial / "stages" / "stills" / f"{evidence_id}.png"
                crop = partial / "stages" / "crops" / f"{evidence_id}.png"
                _write_png(still, rgb)
                _write_png(crop, _masked_crop(rgb, mask))
                copied_mask = partial / "stages" / "stills" / f"{evidence_id}-mask.png"
                _write_png(copied_mask, mask)
                recipe = recipe_by_vial[stage.vial_id]
                original = StageBoundary(
                    stage.vial_id, stage.stage_id, stage.stage_ordinal, stage.tank_id,
                    stage.stage_count, rgb_path, mask_path, stage.frame_index, stage.mask_source,
                    recipe_id=str(recipe["recipe_id"]), run_id=str(recipe["run_id"]),
                    colour_regime_id=stage.colour_regime_id,
                    reference_role=stage.reference_role,
                    source_rgb_path=rgb_path if rgb_path is not None else still,
                    source_mask_path=mask_path,
                )
                derived = StageBoundary(
                    stage.vial_id, stage.stage_id, stage.stage_ordinal, stage.tank_id,
                    stage.stage_count, still, copied_mask, stage.frame_index, stage.mask_source,
                    recipe_id=str(recipe["recipe_id"]), run_id=str(recipe["run_id"]),
                    colour_regime_id=stage.colour_regime_id,
                    reference_role=stage.reference_role,
                    source_rgb_path=rgb_path if rgb_path is not None else still,
                    source_mask_path=mask_path,
                )
                stage_rows.append(derived)
                stage_sources.append((derived, original))
        measured = measure_stage_boundaries(stage_rows) if stage_rows else {"measurements": [], "pairwise_delta_e76": []}
        video_sha256 = _sha256(video)
        clip_hashes = {row["vial_id"]: row["clip_sha256"] for row in fill_rows}
        evidence_records, image_locators = (
            _build_evidence_records(
                recipes,
                fill_boundaries,
                stage_sources,
                measured["measurements"],
                measured["pairwise_delta_e76"],
                clip_hashes,
                video,
                video_sha256,
            )
            if stage_rows
            else ([], [])
        )
        payload = {
            "schema_version": 1,
            "artifact_type": "part_b_fill_clip_measurements_v1",
            "status": "measured" if stage_rows else "clips_only",
            "source": {"video_sha256": video_sha256, "video_path": str(video)},
            "recipe_sequence": recipes,
            "fills": fill_rows,
            "stages": [
                {
                    "evidence_id": f"EVID-{_artifact_id(derived.vial_id, derived.stage_id)}",
                    "vial_id": derived.vial_id,
                    "stage_id": derived.stage_id,
                    "stage_ordinal": derived.stage_ordinal,
                    "tank_id": derived.tank_id,
                    "still_path": f"stages/stills/EVID-{_artifact_id(derived.vial_id, derived.stage_id)}.png",
                    "still_sha256": _sha256(Path(derived.rgb_path)),
                    # mask_path/mask_sha256 are the untouched source binding.  The
                    # derived copy is named separately and is not provenance.
                    "mask_path": str(_source_path(original, "source_mask_path")),
                    "mask_sha256": _sha256(_source_path(original, "source_mask_path")),
                    "derived_mask_path": f"stages/stills/EVID-{_artifact_id(derived.vial_id, derived.stage_id)}-mask.png",
                    "derived_mask_sha256": _sha256(Path(derived.mask_path)),
                    "crop_path": f"stages/crops/EVID-{_artifact_id(derived.vial_id, derived.stage_id)}.png",
                    "crop_sha256": _sha256(partial / "stages" / "crops" / f"EVID-{_artifact_id(derived.vial_id, derived.stage_id)}.png"),
                    "source_rgb_path": str(_source_path(original, "source_rgb_path")),
                    "source_rgb_sha256": _sha256(_source_path(original, "source_rgb_path")),
                    "source_video_path": str(video),
                    "source_video_sha256": video_sha256,
                    "source_frame_index": original.frame_index,
                    "mask_source": derived.mask_source,
                }
                for derived, original in stage_sources
            ],
            "measurements": measured["measurements"],
            "pairwise_delta_e76": measured["pairwise_delta_e76"],
            "timing_evidence": timing_projection,
            "triad_config": dict(TRIAD_CONFIG),
            "evidence_records": evidence_records,
            "image_locators": image_locators,
            "limits": {
                "transport": "transition-tagged and never measured",
                "rgb_only_masks": "no accepted predicted-mask artifact; oracle masks only",
                "part_boundary": "camera video + programmed recipe sequence only",
            },
        }
        for stage in payload["stages"]:
            for field in ("source_rgb_path", "mask_path"):
                stage[field] = _durable_output_path(stage[field], partial, published_output)
        for locator in payload["image_locators"]:
            for field in ("rgb_path", "mask_path", "derived_still_path", "derived_crop_path"):
                locator[field] = _durable_output_path(locator[field], partial, published_output)
        encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        (partial / "fill-clips.json").write_bytes(encoded)
        os.replace(partial, output)
        return payload
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def remeasure_preserved_v3(
    measured_evidence_path: str | os.PathLike[str],
    image_locators_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Remeasure an immutable accepted V3 evidence/locator pair.

    This API reads only Part-B measurement artifacts.  It does not open manifests,
    receipts, labels or any simulation metadata.  It compares all 42 accepted Lab rows
    and 42 accepted same-stage ΔE rows and records whether source hashes stayed stable.
    """

    evidence = Path(measured_evidence_path).expanduser().resolve(strict=True)
    locators = Path(image_locators_path).expanduser().resolve(strict=True)
    before = {str(evidence): _sha256(evidence), str(locators): _sha256(locators)}
    expected = json.loads(evidence.read_text(encoding="utf-8"))
    locator_doc = json.loads(locators.read_text(encoding="utf-8"))
    rows = expected.get("stage_measurements")
    pairs = expected.get("same_run_stage_pairs")
    images = locator_doc.get("images")
    if not isinstance(rows, list) or not isinstance(pairs, list) or not isinstance(images, Mapping):
        raise ValueError("accepted V3 measurement artifacts are malformed")
    actual_rows: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("image_id"), str):
            raise ValueError("accepted V3 measurement row is malformed")
        locator = images.get(row["image_id"])
        if not isinstance(locator, Mapping):
            raise ValueError(f"missing image locator: {row['image_id']}")
        # The accepted locator map carries context for WP5; the measurement helper
        # receives only the four hash-bound pixel locator fields.
        pixel_locator = {
            field: locator[field]
            for field in ("rgb_path", "rgb_sha256", "mask_path", "mask_sha256")
        }
        actual = measure_image(row["image_id"], pixel_locator, _include_dispersion=True)
        if actual.get("validity") != "valid":
            raise ValueError(f"V3 image remeasurement failed: {row['image_id']}")
        actual_rows[row["image_id"]] = actual
    mismatches: list[str] = []
    for row in rows:
        actual = actual_rows[row["image_id"]]
        for field in ("median_lab", "mad_lab", "iqr_lab"):
            if any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(actual[field], row[field], strict=True)):
                mismatches.append(f"{row['image_id']}:{field}")
    actual_pairs: dict[tuple[str, str, str], float] = {}
    for left, right in itertools.combinations(rows, 2):
        if left["batch_id"] != right["batch_id"] or left["stage_id"] != right["stage_id"]:
            continue
        key = (str(left["batch_id"]), str(left["stage_id"]), "|".join(sorted((str(left["vial_id"]), str(right["vial_id"])))) )
        actual_pairs[key] = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual_rows[left["image_id"]]["median_lab"], actual_rows[right["image_id"]]["median_lab"], strict=True)))
    expected_pairs = {}
    for row in pairs:
        key = (str(row["batch_id"]), str(row["stage_id"]), "|".join(sorted((str(row["left_vial_id"]), str(row["right_vial_id"])))) )
        expected_pairs[key] = float(row["delta_e76"])
    if set(actual_pairs) != set(expected_pairs):
        mismatches.append("pair_key_set")
    for key, value in expected_pairs.items():
        if key in actual_pairs and abs(actual_pairs[key] - value) > 1e-12:
            mismatches.append(f"pair:{key}")
    after = {str(evidence): _sha256(evidence), str(locators): _sha256(locators)}
    return {
        "executed": True,
        "status": "matched" if not mismatches else "mismatch",
        "lab_rows_expected": len(rows),
        "lab_rows_remeasured": len(actual_rows),
        "delta_e_rows_expected": len(pairs),
        "delta_e_rows_remeasured": len(actual_pairs),
        "mismatches": mismatches,
        "inputs_unchanged": before == after,
    }


remeasure_v3_regression = remeasure_preserved_v3
