"""Small streaming core for quick, RGB-only vial-fill cohort checks.

This module deliberately has no recipe, tank, camera-profile, or model-loading
policy.  A caller supplies the fill-level predictor and consumes the plain
mapping events it returns.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import math

import numpy as np
from PIL import Image


FILL_SECONDS = 40
SAMPLES_PER_FILL = 6
SAMPLE_INTERVAL_SECONDS = FILL_SECONDS / SAMPLES_PER_FILL
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# These are image-space guards, not calibrated physical measurements.
MIN_VISIBLE_HEIGHT_PX = 8
HEIGHT_STABLE_TOLERANCE_PX = 4
RESET_HEIGHT_PX = 4
PIXEL_STABLE_FRACTION = 0.05
MIN_PIXEL_STABLE_TOLERANCE = 64


def _rgb_640x480(frame: object) -> np.ndarray:
    """Coerce an RGB frame to the predictor's fixed uint8 640x480 input."""
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("frame must be an RGB-like HxWx3 array")
    value = value[..., :3]
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("frame values must be numeric")
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and value.size and np.nanmax(value) <= 1:
            value = value * 255.0
        value = np.clip(np.nan_to_num(value), 0, 255).astype(np.uint8)
    if value.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        value = np.asarray(
            Image.fromarray(value, "RGB").resize(
                (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BILINEAR
            )
        )
    return np.ascontiguousarray(value)


def _mask_from_prediction(prediction: object, shape: tuple[int, int]) -> np.ndarray:
    """Accept the local predictor object, a mapping, or an array mask."""
    raw = prediction
    if isinstance(prediction, Mapping):
        raw = prediction.get("mask")
    elif hasattr(prediction, "mask"):
        raw = getattr(prediction, "mask")
    mask = np.asarray(raw)
    if mask.shape != shape:
        raise ValueError("fill-level predictor returned a mask with the wrong shape")
    if mask.dtype != np.bool_:
        mask = mask > 0
    return np.ascontiguousarray(mask, dtype=bool)


def _mask_metrics(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(np.any(mask, axis=1))
    if rows.size == 0:
        return 0, 0
    return int(rows[-1] - rows[0] + 1), int(np.count_nonzero(mask))


def _lab_summary(rgb: np.ndarray, mask: np.ndarray) -> dict[str, list[float]]:
    from skimage.color import rgb2lab

    values = rgb2lab(rgb.astype(np.float64) / 255.0)[mask]
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    low, high = np.percentile(values, (25.0, 75.0), axis=0)
    return {
        "median_lab": [float(x) for x in median],
        "mad_lab": [float(x) for x in mad],
        "iqr_lab": [float(x) for x in high - low],
    }


class QuickFillCore:
    """Sample a stream, close stable fills, and assess each three-vial cohort."""

    def __init__(self, predictor: Callable[[np.ndarray, np.ndarray], object]):
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        self.predictor = predictor
        self._reference: np.ndarray | None = None
        self._next_sample_at: float | None = None
        self._last_height: int | None = None
        self._last_pixels: int | None = None
        self._awaiting_reset = False
        self._vials: list[dict[str, object]] = []
        self._profile: Mapping[str, object] | None = None
        self._last_timestamp: float | None = None

    @property
    def session_complete(self) -> bool:
        return False

    @property
    def eof_abstention_reason(self) -> str | None:
        if not self._vials:
            return "no_completed_vial"
        if len(self._vials) % 3:
            return "incomplete_three_vial_cohort"
        return None

    @property
    def source_profile_evidence(self) -> Mapping[str, object] | None:
        return deepcopy(self._profile)

    def configure_source_profile(self, profile: Mapping[str, object]) -> None:
        if not isinstance(profile, Mapping):
            raise TypeError("source profile must be a mapping")
        self._profile = deepcopy(profile)

    def observe(self, frame: object, *, frame_index: int | None = None,
                elapsed_seconds: float) -> tuple[Mapping[str, object], ...]:
        return self.process_frame(
            frame, frame_index=frame_index, elapsed_seconds=elapsed_seconds
        )

    def process_frame(self, frame: object, *, frame_index: int | None = None,
                      elapsed_seconds: float) -> tuple[Mapping[str, object], ...]:
        timestamp = float(elapsed_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("elapsed_seconds must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("elapsed_seconds must be monotonic")
        self._last_timestamp = timestamp
        if self._next_sample_at is not None and timestamp < self._next_sample_at:
            return ()
        rgb = _rgb_640x480(frame)
        self._next_sample_at = timestamp + SAMPLE_INTERVAL_SECONDS
        if self._reference is None:
            self._reference = rgb.copy()
            return (self._sample_event("reference", frame_index, timestamp, 0, 0),)

        mask = _mask_from_prediction(self.predictor(rgb, self._reference), rgb.shape[:2])
        height, pixels = _mask_metrics(mask)
        sample = self._sample_event("sample", frame_index, timestamp, height, pixels)
        if self._awaiting_reset:
            if pixels == 0 or height <= RESET_HEIGHT_PX:
                self._awaiting_reset = False
                self._reference = rgb.copy()
                self._last_height = None
                self._last_pixels = None
                return (sample, self._sample_event("reset", frame_index, timestamp, height, pixels))
            return (sample,)

        pixel_tolerance = max(
            MIN_PIXEL_STABLE_TOLERANCE,
            int(round((self._last_pixels or 0) * PIXEL_STABLE_FRACTION)),
        )
        stable = (
            self._last_height is not None
            and self._last_pixels is not None
            and abs(height - self._last_height) <= HEIGHT_STABLE_TOLERANCE_PX
            and abs(pixels - self._last_pixels) <= pixel_tolerance
        )
        self._last_height = height
        self._last_pixels = pixels
        if height < MIN_VISIBLE_HEIGHT_PX or not stable:
            return (sample,)
        vial = {
            "vial_index": len(self._vials) + 1,
            "frame_index": frame_index,
            "elapsed_seconds": timestamp,
            "rgb": rgb.copy(),
            "mask": mask.copy(),
            "visible_height_px": height,
            "mask_pixel_count": pixels,
            **_lab_summary(rgb, mask),
        }
        self._vials.append(vial)
        self._awaiting_reset = True
        result: list[Mapping[str, object]] = [
            {"event": "vial_end", "vial": deepcopy(vial)}
        ]
        if len(self._vials) % 3 == 0:
            result.append(self._cohort_result(self._vials[-3:]))
        return tuple(result)

    def end_of_stream(self) -> tuple[Mapping[str, object], ...]:
        reason = self.eof_abstention_reason
        return () if reason is None else ({"event": "abstention", "reason": reason},)

    @staticmethod
    def _sample_event(event: str, frame_index: int | None, elapsed_seconds: float,
                      height: int, pixels: int) -> Mapping[str, object]:
        return {"event": event, "frame_index": frame_index,
                "elapsed_seconds": elapsed_seconds, "visible_height_px": height,
                "mask_pixel_count": pixels}

    @staticmethod
    def _cohort_result(vials: list[dict[str, object]]) -> Mapping[str, object]:
        labs = np.asarray([vial["median_lab"] for vial in vials], dtype=float)
        d2 = float(np.linalg.norm(labs[0] - labs[1]))
        d3 = float(np.linalg.norm(labs[0] - labs[2]))
        if d2 <= 1.0 and d3 <= 1.0:
            decision = "keep"
        elif d2 >= 2.0 and d3 >= 2.0:
            decision = "discard"
        else:
            decision = "insufficient_evidence"
        return {
            "event": "cohort_result",
            "cohort_vial_indices": [vial["vial_index"] for vial in vials],
            "delta_e76_vial2": d2,
            "delta_e76_vial3": d3,
            "decision": decision,
        }


class QuickFillPipelineAdapter:
    """Translate quick-fill endpoints into the existing runner adapter contract."""

    rule_only = True
    model_id = "quick-fill-rule-only"
    requires_complete = False

    def __init__(
        self,
        predictor: Callable[[np.ndarray, np.ndarray], object],
        run_context: Mapping[str, object],
    ):
        self.core = QuickFillCore(predictor)
        self.run_context = dict(run_context)

    @property
    def session_complete(self) -> bool:
        return False

    @property
    def eof_abstention_reason(self) -> str | None:
        return self.core.eof_abstention_reason

    @property
    def source_profile_evidence(self) -> Mapping[str, object] | None:
        return self.core.source_profile_evidence

    @property
    def end_images(self) -> tuple[Mapping[str, object], ...]:
        return tuple(deepcopy(self.core._vials))

    def configure_source_profile(self, profile: Mapping[str, object]) -> None:
        self.core.configure_source_profile(profile)

    def process_frame(
        self, frame: object, *, frame_index: int, elapsed_seconds: float,
    ) -> tuple[Mapping[str, object], ...]:
        events = self.core.process_frame(
            frame, frame_index=frame_index, elapsed_seconds=elapsed_seconds,
        )
        return tuple(
            self._runtime_result(event["vial"])
            for event in events if event.get("event") == "vial_end"
        )

    def end_of_stream(self) -> tuple[Mapping[str, object], ...]:
        return ()

    def decide_cohort(
        self, vials: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, object]:
        ordered = sorted(vials.values(), key=lambda row: int(row["vial_position"]))
        if [int(row["vial_position"]) for row in ordered] != [1, 2, 3]:
            raise ValueError("quick-fill cohort requires vial positions 1, 2 and 3")
        labs = [row["measurement"]["median_lab"] for row in ordered]
        d2, d3 = math.dist(labs[0], labs[1]), math.dist(labs[0], labs[2])
        if d2 <= 1.0 and d3 <= 1.0:
            disposition, score, limits = "keep", 0.0, []
        elif d2 >= 2.0 and d3 >= 2.0:
            disposition, score, limits = "discard", 1.0, []
        else:
            disposition, score = "insufficient_evidence", None
            limits = ["same_cohort_threshold_deadband"]
        evidence = [str(row["measurement"]["evidence_id"]) for row in ordered]
        return {
            "disposition": disposition,
            "discard_score": score,
            "evidence_citations": evidence,
            "limit_codes": limits,
            "stages": [{
                "stage_id": "end",
                "evidence_id": evidence[0],
                "disposition": disposition,
                "limit_codes": limits,
                "delta_e76_vial2": d2,
                "delta_e76_vial3": d3,
            }],
        }

    def build_vlm_bundle(self, vials: Mapping[str, object]) -> Mapping[str, object]:
        return {"mode": "quick_fill", "vials": deepcopy(dict(vials))}

    def run_vlm(self, _bundle: object) -> tuple[Mapping[str, object], ...]:
        models = (
            ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
            ("nvidia/Cosmos-Reason2-8B", "a9fae2cf89dc64db96b12860417f0eb403013bb9"),
            ("aacudad/AnomalyThink-Qwen3-VL-8B-KCR", "8331bc79ecf01e3466f0e6a27cec8c77f182f430"),
        )
        return tuple({
            "source_id": f"quick-fill-vlm-not-run-{index}",
            "repository": repository,
            "revision": revision,
            "guard_status": "unavailable",
            "disposition": "insufficient_evidence",
            "discard_score": None,
            "confidence": None,
            "explanation": None,
            "evidence_citations": [],
            "image_citations": [],
            "unavailable_reason": "quick_fill_vlm_not_run",
        } for index, (repository, revision) in enumerate(models, start=1))

    def close(self) -> None:
        return None

    def _runtime_result(self, vial: Mapping[str, object]) -> Mapping[str, object]:
        index = int(vial["vial_index"])
        cohort = (index - 1) // 3 + 1
        position = (index - 1) % 3 + 1
        run_id = str(self.run_context.get("operator_run_id") or "quick-fill")
        tray_id = str(self.run_context.get("tray_id") or "tray")
        vial_id = f"{run_id}:{tray_id}:cohort-{cohort}:vial-{position}"
        evidence_id = f"quick-fill-c{cohort}-v{position}-end"
        return {
            "vial_id": vial_id,
            "vial_position": position,
            "fill_step": 1,
            "tank_id": "end",
            "phase": "end",
            "tracker": {
                "agreement": True,
                "measured_volume_ml": None,
                "limit_codes": ["quick_fill_height_only"],
            },
            "mask": {
                "status": "visible_measured",
                "pixel_count": int(vial["mask_pixel_count"]),
                "visible_height_px": int(vial["visible_height_px"]),
            },
            "measurement": {
                "validity": "valid",
                "median_lab": list(vial["median_lab"]),
                "mad_lab": list(vial["mad_lab"]),
                "iqr_lab": list(vial["iqr_lab"]),
                "evidence_id": evidence_id,
                "measured_or_inferred": "measured",
            },
            "closed": True,
        }

