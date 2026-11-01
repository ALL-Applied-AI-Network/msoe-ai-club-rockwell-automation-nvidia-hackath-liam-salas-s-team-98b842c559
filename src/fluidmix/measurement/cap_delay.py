"""Explicit-pair, burned-clock-corrected fill-to-cap delay receipts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core.io import sha256_file as _sha256_path


ALLOWED_VIDEO_IDS = (2, 3, 4)
MIN_UNAMBIGUOUS_PAIRS = 8
MIN_VIDEOS = 2


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _video_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("video_id must be an integer")
    if value == 1:
        raise ValueError("video 1 clock gaps cannot authorize cap timing")
    if value == 5:
        raise ValueError("video 5 is held out and cannot be calibration input")
    if value not in ALLOWED_VIDEO_IDS:
        raise ValueError("video_id must be one of videos 2-4")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _round(value: float) -> float:
    return round(value, 6)


def build_cap_delay_receipt(table_path: str | Path) -> dict[str, object]:
    """Return a receipt from explicit candidate pairs; it never infers event labels.

    Each input pair contains visible crop IDs, observed burned-clock seconds, correction
    seconds for both cameras, and an ``unambiguous`` judgement supplied by the table
    author. The correction is applied before differencing.
    """
    path = Path(table_path).resolve(strict=True)
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("cap-delay pair table is not valid JSON") from error
    if not isinstance(table, Mapping) or table.get("schema_version") != 1:
        raise ValueError("cap-delay pair table must have schema_version 1")
    input_pairs = table.get("pairs")
    if not isinstance(input_pairs, list):
        raise ValueError("cap-delay pair table.pairs must be a list")
    pairs: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in input_pairs:
        if not isinstance(row, Mapping):
            raise ValueError("each cap-delay pair must be an object")
        pair_id = _text(row.get("pair_id"), "pair_id")
        if pair_id in seen_ids:
            raise ValueError("each cap-delay pair needs a unique pair_id")
        seen_ids.add(pair_id)
        video_id = _video_id(row.get("video_id"))
        fill_vial_id = _text(row.get("fill_vial_id"), f"pair {pair_id}.fill_vial_id")
        cap_vial_id = _text(row.get("cap_vial_id"), f"pair {pair_id}.cap_vial_id")
        if fill_vial_id != cap_vial_id:
            raise ValueError(f"pair {pair_id} does not identify the same vial at fill and cap")
        fill_crop_id = _text(row.get("fill_crop_id"), f"pair {pair_id}.fill_crop_id")
        cap_crop_id = _text(row.get("cap_crop_id"), f"pair {pair_id}.cap_crop_id")
        clock_correction_id = _text(row.get("clock_correction_id"), f"pair {pair_id}.clock_correction_id")
        fill_clock = _number(row.get("fill_end_clock_s"), f"pair {pair_id}.fill_end_clock_s")
        cap_clock = _number(row.get("cap_event_clock_s"), f"pair {pair_id}.cap_event_clock_s")
        fill_correction = _number(row.get("fill_clock_correction_s"), f"pair {pair_id}.fill_clock_correction_s")
        cap_correction = _number(row.get("cap_clock_correction_s"), f"pair {pair_id}.cap_clock_correction_s")
        if not isinstance(row.get("unambiguous"), bool):
            raise ValueError(f"pair {pair_id}.unambiguous must be a boolean")
        corrected_delay = (cap_clock + cap_correction) - (fill_clock + fill_correction)
        item: dict[str, object] = {"pair_id": pair_id, "video_id": video_id, "vial_id": fill_vial_id,
                                   "fill_crop_id": fill_crop_id, "cap_crop_id": cap_crop_id,
                                   "clock_correction_id": clock_correction_id,
                                   "fill_end_clock_s": _round(fill_clock), "cap_event_clock_s": _round(cap_clock),
                                   "fill_clock_correction_s": _round(fill_correction), "cap_clock_correction_s": _round(cap_correction),
                                   "corrected_delay_s": _round(corrected_delay)}
        reason: str | None = None
        if not row["unambiguous"]:
            reason = _text(row.get("exclusion_reason", "ambiguous_pair"), f"pair {pair_id}.exclusion_reason")
        elif corrected_delay < 0:
            reason = "negative_corrected_delay"
        if reason is None:
            item["status"] = "usable"
        else:
            item["status"] = "excluded"
            item["exclusion_reason"] = reason
            exclusions.append({"pair_id": pair_id, "reason": reason})
        pairs.append(item)
    usable = [pair for pair in pairs if pair["status"] == "usable"]
    video_ids = sorted({int(pair["video_id"]) for pair in usable})
    insufficiency_reasons: list[str] = []
    if len(usable) < MIN_UNAMBIGUOUS_PAIRS:
        insufficiency_reasons.append("fewer_than_8_unambiguous_pairs")
    if len(video_ids) < MIN_VIDEOS:
        insufficiency_reasons.append("fewer_than_2_videos")
    delays = [float(pair["corrected_delay_s"]) for pair in usable]
    percentiles = None if insufficiency_reasons else {"p10": _round(float(np.percentile(delays, 10))),
                                                       "median": _round(float(np.percentile(delays, 50))),
                                                       "p90": _round(float(np.percentile(delays, 90)))}
    return {"schema_version": 1, "receipt_type": "fill_to_cap_delay",
            "status": "measured" if percentiles is not None else "insufficient_evidence",
            "allowed_video_ids": list(ALLOWED_VIDEO_IDS), "input_table_sha256": _sha256_path(path),
            "success_requirements": {"min_unambiguous_pairs": MIN_UNAMBIGUOUS_PAIRS, "min_videos": MIN_VIDEOS},
            "candidate_pair_count": len(pairs), "usable_pair_count": len(usable), "excluded_pair_count": len(pairs) - len(usable),
            "usable_video_ids": video_ids, "pairs": pairs, "exclusions": exclusions,
            "percentiles_s": percentiles, "insufficiency_reasons": insufficiency_reasons,
            "limits": {"event_source": "explicit candidate pair table only",
                       "video_1": "not authorized because cap_seal clock gaps are documented",
                       "video_5": "held out from calibration",
                       "claim": "observed corrected timing only; no hidden-label inference"}}
