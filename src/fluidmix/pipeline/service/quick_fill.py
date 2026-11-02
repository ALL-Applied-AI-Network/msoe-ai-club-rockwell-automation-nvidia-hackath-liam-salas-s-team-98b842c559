"""Small streaming core for quick, RGB-only vial-fill cohort checks.

This module deliberately has no recipe, tank, camera-profile, or model-loading
policy.  A caller supplies the fill-level predictor and consumes the plain
mapping events it returns.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import math
import re

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
PIXEL_STABLE_FRACTION = 0.06
MIN_PIXEL_STABLE_TOLERANCE = 64
RESET_HEIGHT_FRACTION = 0.60
RECIPE_CHANGE_DELTA_E76 = 12.0
PLAN_VOLUME_TOLERANCE_ML = 2.0
_PLAN_STEP = re.compile(
    r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*"
)


def parse_dispense_plan(value: object) -> tuple[dict[str, object], ...]:
    """Parse one ordered tank/mL group per nonempty line."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return ()
    if not isinstance(value, str):
        raise ValueError("dispense_plan must be text")
    groups = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        body, marker, count_text = line.rpartition("x")
        if not marker or not count_text.strip().isdigit():
            raise ValueError(f"dispense_plan line {line_number} needs x<vial-count>")
        vial_count = int(count_text.strip())
        if vial_count <= 0:
            raise ValueError("dispense_plan vial count must be positive")
        steps = []
        for raw_step in body.split(">"):
            match = _PLAN_STEP.fullmatch(raw_step)
            if match is None:
                raise ValueError(f"dispense_plan line {line_number} has invalid step")
            amount = float(match.group(2))
            if not math.isfinite(amount) or amount <= 0.0:
                raise ValueError("dispense_plan amounts must be positive")
            steps.append({"tank": match.group(1).upper(), "ml": amount})
        if not steps:
            raise ValueError("dispense_plan group needs at least one step")
        groups.append({
            "steps": steps,
            "vial_count": vial_count,
            "expected_total_ml": sum(step["ml"] for step in steps),
        })
    return tuple(groups)


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
        self._last_candidate: dict[str, object] | None = None
        self._awaiting_reset = False
        self._accepted_height: int | None = None
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
            reset_height = (
                RESET_HEIGHT_PX if self._accepted_height is None
                else max(RESET_HEIGHT_PX, int(self._accepted_height * RESET_HEIGHT_FRACTION))
            )
            if pixels == 0 or height <= reset_height:
                self._awaiting_reset = False
                self._reference = rgb.copy()
                self._last_height = None
                self._last_pixels = None
                self._last_candidate = None
                self._accepted_height = None
                return (sample, self._sample_event("reset", frame_index, timestamp, height, pixels))
            return (sample,)

        candidate = {
            "frame_index": frame_index, "elapsed_seconds": timestamp,
            "rgb": rgb.copy(), "mask": mask.copy(),
            "visible_height_px": height, "mask_pixel_count": pixels,
        }
        if (
            self._last_candidate is not None
            and self._last_height is not None
            and self._last_height >= MIN_VISIBLE_HEIGHT_PX
            and height <= max(
                RESET_HEIGHT_PX,
                int(self._last_height * RESET_HEIGHT_FRACTION),
            )
        ):
            vial = self._record_vial(self._last_candidate)
            self._reference = rgb.copy()
            self._last_height = None
            self._last_pixels = None
            self._last_candidate = None
            return (
                sample,
                {"event": "vial_end", "vial": deepcopy(vial)},
                self._sample_event("reset", frame_index, timestamp, height, pixels),
            )

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
        self._last_candidate = candidate
        if height < MIN_VISIBLE_HEIGHT_PX or not stable:
            return (sample,)
        vial = self._record_vial(candidate)
        self._awaiting_reset = True
        self._accepted_height = height
        result: list[Mapping[str, object]] = [
            {"event": "vial_end", "vial": deepcopy(vial)}
        ]
        if len(self._vials) % 3 == 0:
            result.append(self._cohort_result(self._vials[-3:]))
        return tuple(result)

    def _record_vial(self, candidate: Mapping[str, object]) -> dict[str, object]:
        rgb = np.asarray(candidate["rgb"])
        mask = np.asarray(candidate["mask"], dtype=bool)
        vial = {
            "vial_index": len(self._vials) + 1,
            **deepcopy(dict(candidate)),
            **_lab_summary(rgb, mask),
        }
        self._vials.append(vial)
        return vial

    def end_of_stream(self) -> tuple[Mapping[str, object], ...]:
        results: list[Mapping[str, object]] = []
        if (
            not self._awaiting_reset
            and self._last_candidate is not None
            and self._last_height is not None
            and self._last_height >= MIN_VISIBLE_HEIGHT_PX
        ):
            vial = self._record_vial(self._last_candidate)
            self._last_candidate = None
            results.append({"event": "vial_end", "vial": deepcopy(vial)})
        reason = self.eof_abstention_reason
        if reason is not None:
            results.append({"event": "abstention", "reason": reason})
        return tuple(results)

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
        calibration: object | None = None,
    ):
        self.core = QuickFillCore(predictor)
        self.run_context = dict(run_context)
        self.dispense_groups = parse_dispense_plan(
            self.run_context.get("dispense_plan")
        )
        self.calibration = calibration
        self._detected_vials = 0
        self._fallback_cohort = 1
        self._fallback_position = 0
        self._previous_lab: list[float] | None = None
        self._plan_finished_emitted = False
        self._previous_cohort_labs: np.ndarray | None = None
        self._previous_cohort_index: int | None = None
        self._latest_carryover: dict[str, object] = {
            "status": "carryover_not_assessable_first_cohort",
            "previous_recipe_end_lab": None,
            "reference_lab": None,
            "carryover_fraction": None,
            "delta_e_to_refs": None,
            "delta_e_to_prev": None,
            "previous_cohort_index": None,
        }
        self._carryover_log_started = False

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
        results = []
        for event in events:
            if event.get("event") != "vial_end":
                continue
            assignment = self._assign_vial(event["vial"])
            if assignment is not None:
                results.append(self._runtime_result(event["vial"], assignment))
        return tuple(results)

    def end_of_stream(self) -> tuple[Mapping[str, object], ...]:
        results = []
        for event in self.core.end_of_stream():
            if event.get("event") != "vial_end":
                continue
            assignment = self._assign_vial(event["vial"])
            if assignment is not None:
                results.append(self._runtime_result(event["vial"], assignment))
        return tuple(results)

    def decide_cohort(
        self, vials: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, object]:
        ordered = sorted(vials.values(), key=lambda row: int(row["vial_position"]))
        positions = [int(row["vial_position"]) for row in ordered]
        if positions != list(range(1, len(ordered) + 1)):
            raise ValueError("quick-fill cohort positions must be consecutive")
        evidence = [str(row["measurement"]["evidence_id"]) for row in ordered]
        if ordered[0].get("plan_finished") is True:
            return self._insufficient_decision(evidence, "dispense_plan_finished")
        if len(ordered) < 2:
            return self._insufficient_decision(evidence, "cohort_too_short")
        labs = [row["measurement"]["median_lab"] for row in ordered]
        lab_array = np.asarray(labs, dtype=float)
        cohort_index = int(ordered[0].get("cohort_ordinal", 1))
        reference_lab = np.median(lab_array[1:], axis=0)
        carryover = self._carryover_metrics(
            cohort_index, lab_array, reference_lab,
        )
        differences = [math.dist(labs[0], reference) for reference in labs[1:]]
        if all(value <= 1.0 for value in differences):
            disposition, score, limits = "keep", 0.0, []
        elif all(value >= 2.0 for value in differences):
            disposition, score, limits = "discard", 1.0, []
        else:
            disposition, score = "insufficient_evidence", None
            limits = ["same_cohort_threshold_deadband"]
        delta_e = {
            f"vial_{index + 2}": value
            for index, value in enumerate(differences)
        }
        self._latest_carryover = carryover
        self._previous_cohort_labs = lab_array.copy()
        self._previous_cohort_index = cohort_index
        self._log_changeover(carryover, disposition)
        return {
            "disposition": disposition,
            "discard_score": score,
            "evidence_citations": evidence,
            "limit_codes": limits,
            "carryover_fraction": carryover["carryover_fraction"],
            "delta_e_to_refs": carryover["delta_e_to_refs"],
            "delta_e_to_prev": carryover["delta_e_to_prev"],
            "previous_cohort_index": carryover["previous_cohort_index"],
            "stages": [{
                "stage_id": "end",
                "evidence_id": evidence[0],
                "disposition": disposition,
                "limit_codes": limits,
                "delta_e76_references": delta_e,
            }],
        }

    def build_vlm_bundle(self, vials: Mapping[str, object]) -> Mapping[str, object]:
        ordered = sorted(
            vials.values(), key=lambda row: int(row["vial_position"])
        )
        context = ordered[0].get("quick_fill_context", {}) if ordered else {}
        return {
            "mode": "quick_fill", "vials": deepcopy(dict(vials)),
            "dispense_context": deepcopy(context),
            "carryover_context": deepcopy(self._latest_carryover),
        }

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

    @staticmethod
    def _insufficient_decision(
        evidence: list[str], reason: str,
    ) -> Mapping[str, object]:
        return {
            "disposition": "insufficient_evidence", "discard_score": None,
            "evidence_citations": evidence, "limit_codes": [reason],
            "stages": [{
                "stage_id": "end", "evidence_id": evidence[0],
                "disposition": "insufficient_evidence", "limit_codes": [reason],
                "delta_e76_references": {},
            }],
        }

    def _assign_vial(
        self, vial: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        self._detected_vials += 1
        if self.dispense_groups:
            remaining = self._detected_vials
            for cohort, group in enumerate(self.dispense_groups, start=1):
                count = int(group["vial_count"])
                if remaining <= count:
                    return {
                        "cohort": cohort, "position": remaining,
                        "cohort_size": count, "group": deepcopy(group),
                        "boundary_reason": None, "plan_finished": False,
                    }
                remaining -= count
            if self._plan_finished_emitted:
                return None
            self._plan_finished_emitted = True
            return {
                "cohort": len(self.dispense_groups) + 1, "position": 1,
                "cohort_size": 1, "group": None,
                "boundary_reason": None, "plan_finished": True,
            }

        lab = [float(value) for value in vial["median_lab"]]
        boundary_reason = None
        if self._fallback_position >= 3:
            self._fallback_cohort += 1
            self._fallback_position = 0
        elif self._fallback_position and self._previous_lab is not None:
            delta = math.dist(lab, self._previous_lab)
            if delta > RECIPE_CHANGE_DELTA_E76:
                self._fallback_cohort += 1
                self._fallback_position = 0
                boundary_reason = "probable_recipe_change_by_colour"
        self._fallback_position += 1
        self._previous_lab = lab
        return {
            "cohort": self._fallback_cohort, "position": self._fallback_position,
            "cohort_size": 3, "group": None,
            "boundary_reason": boundary_reason, "plan_finished": False,
        }

    def _carryover_metrics(
        self, cohort_index: int, labs: np.ndarray, reference_lab: np.ndarray,
    ) -> dict[str, object]:
        if self._previous_cohort_labs is None:
            return {
                "status": "carryover_not_assessable_first_cohort",
                "previous_recipe_end_lab": None,
                "reference_lab": [float(value) for value in reference_lab],
                "carryover_fraction": None,
                "delta_e_to_refs": float(np.linalg.norm(labs[0] - reference_lab)),
                "delta_e_to_prev": None,
                "previous_cohort_index": None,
            }
        previous_lab = np.median(self._previous_cohort_labs, axis=0)
        shift = labs[0] - reference_lab
        axis = previous_lab - reference_lab
        axis_length = float(np.linalg.norm(axis))
        fraction = (
            None if axis_length < 1.0
            else float(np.dot(shift, axis) / np.dot(axis, axis))
        )
        return {
            "status": (
                "previous_and_current_recipes_visually_similar"
                if fraction is None else "measured_from_run_colours"
            ),
            "previous_recipe_end_lab": [float(value) for value in previous_lab],
            "reference_lab": [float(value) for value in reference_lab],
            "carryover_fraction": fraction,
            "delta_e_to_refs": float(np.linalg.norm(shift)),
            "delta_e_to_prev": float(np.linalg.norm(labs[0] - previous_lab)),
            "previous_cohort_index": self._previous_cohort_index,
            "current_cohort_index": cohort_index,
        }

    def _log_changeover(
        self, carryover: Mapping[str, object], disposition: str,
    ) -> None:
        previous = carryover.get("previous_cohort_index")
        if previous is None:
            return
        if not self._carryover_log_started:
            print(
                "| changeover | delta_e_to_refs | delta_e_to_prev | "
                "carryover_fraction | rule_call |",
                flush=True,
            )
            print("|---|---:|---:|---:|---|", flush=True)
            self._carryover_log_started = True
        fraction = carryover.get("carryover_fraction")
        rendered_fraction = "null" if fraction is None else f"{float(fraction):.6f}"
        print(
            f"| cohort {previous} -> {carryover.get('current_cohort_index')} | "
            f"{float(carryover['delta_e_to_refs']):.6f} | "
            f"{float(carryover['delta_e_to_prev']):.6f} | "
            f"{rendered_fraction} | {disposition} |",
            flush=True,
        )

    def _volume_sanity(
        self, visible_height_px: float, group: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        if self.calibration is None or group is None:
            return {
                "status": "unavailable", "expected_total_ml": None,
                "expected_height_px": None, "measured_volume_ml": None,
                "difference_ml": None,
            }
        heights = np.asarray(self.calibration.fill_height_px_samples, dtype=float)
        volumes = np.asarray(
            self.calibration.twin_geometry_equivalent_volume_ml_samples, dtype=float,
        )
        expected = float(group["expected_total_ml"])
        expected_height = (
            float(np.interp(expected, volumes, heights))
            if volumes[0] <= expected <= volumes[-1] else None
        )
        measured = (
            float(np.interp(visible_height_px, heights, volumes))
            if heights[0] <= visible_height_px <= heights[-1] else None
        )
        difference = None if measured is None else measured - expected
        status = (
            "unavailable" if difference is None
            else "end_level_matches_plan"
            if abs(difference) <= PLAN_VOLUME_TOLERANCE_ML
            else "end_level_differs_from_plan"
        )
        return {
            "status": status, "expected_total_ml": expected,
            "expected_height_px": expected_height,
            "measured_volume_ml": measured, "difference_ml": difference,
        }

    def _runtime_result(
        self, vial: Mapping[str, object], assignment: Mapping[str, object],
    ) -> Mapping[str, object]:
        cohort = int(assignment["cohort"])
        position = int(assignment["position"])
        cohort_size = int(assignment["cohort_size"])
        group = assignment.get("group")
        sanity = self._volume_sanity(float(vial["visible_height_px"]), group)
        run_id = str(self.run_context.get("operator_run_id") or "quick-fill")
        tray_id = str(self.run_context.get("tray_id") or "tray")
        vial_id = f"{run_id}:{tray_id}:cohort-{cohort}:vial-{position}"
        evidence_id = f"quick-fill-c{cohort}-v{position}-end"
        return {
            "vial_id": vial_id,
            "vial_position": position,
            "cohort_size": cohort_size,
            "cohort_ordinal": cohort,
            "source_time_s": float(vial["elapsed_seconds"]),
            "boundary_reason": assignment.get("boundary_reason"),
            "plan_finished": assignment.get("plan_finished") is True,
            "fill_step": 1,
            "tank_id": "end",
            "phase": "end",
            "tracker": {
                "agreement": True,
                "measured_volume_ml": sanity["measured_volume_ml"],
                "limit_codes": (
                    ["quick_fill_height_only"]
                    if sanity["status"] == "unavailable" else []
                ),
                "plan_sanity": sanity,
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
            "quick_fill_context": {
                "cohort_index": cohort,
                "dispense_steps": (
                    deepcopy(group["steps"])
                    if isinstance(group, Mapping) else []
                ),
                "expected_total_ml": sanity["expected_total_ml"],
                "plan_sanity": sanity,
            },
            "closed": True,
        }

